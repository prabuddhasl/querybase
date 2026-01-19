"""
Ingest Postman collection JSON into ChromaDB vector store.
"""

import json
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

COLLECTION_PATH = "collection.json"
VECTORSTORE_PATH = "./vectorstore"


def extract_endpoints(item, path=""):
    """Recursively extract endpoints from Postman collection."""
    documents = []
    
    for entry in item:
        name = entry.get("name", "")
        current_path = f"{path}/{name}" if path else name
        
        # If it has nested items, it's a folder - recurse
        if "item" in entry:
            # Check if folder has description
            folder_desc = entry.get("description", "")
            if folder_desc:
                documents.append(Document(
                    page_content=f"# {current_path}\n\n{folder_desc}",
                    metadata={
                        "type": "folder",
                        "path": current_path,
                        "name": name
                    }
                ))
            documents.extend(extract_endpoints(entry["item"], current_path))
        
        # If it has a request, it's an endpoint
        elif "request" in entry:
            doc_content = build_endpoint_doc(entry, current_path)
            documents.append(Document(
                page_content=doc_content,
                metadata={
                    "type": "endpoint",
                    "path": current_path,
                    "name": name,
                    "method": entry["request"].get("method", ""),
                    "url": extract_url(entry["request"])
                }
            ))
    
    return documents


def extract_url(request):
    """Extract URL from request object."""
    url = request.get("url", {})
    if isinstance(url, str):
        return url
    return url.get("raw", "")


def build_endpoint_doc(entry, path):
    """Build a readable document from an endpoint entry."""
    request = entry["request"]
    
    parts = []
    
    # Header
    method = request.get("method", "")
    url = extract_url(request)
    parts.append(f"# {entry['name']}")
    parts.append(f"**Path:** {path}")
    parts.append(f"**Method:** {method}")
    parts.append(f"**URL:** {url}")
    
    # Description
    description = entry.get("description", "")
    if description:
        parts.append(f"\n## Description\n{description}")
    
    # Headers
    headers = request.get("header", [])
    if headers:
        parts.append("\n## Headers")
        for h in headers:
            if isinstance(h, dict):
                key = h.get("key", "")
                value = h.get("value", "")
                desc = h.get("description", "")
                header_line = f"- `{key}`: {value}"
                if desc:
                    header_line += f" - {desc}"
                parts.append(header_line)
    
    # Request Body
    body = request.get("body", {})
    if body and body.get("raw"):
        parts.append("\n## Request Body Example")
        parts.append(f"```json\n{body['raw']}\n```")
    
    # Response examples
    responses = entry.get("response", [])
    if responses:
        parts.append("\n## Response Examples")
        for resp in responses[:2]:  # Limit to 2 examples
            resp_name = resp.get("name", "Response")
            resp_body = resp.get("body", "")
            status = resp.get("code", "")
            parts.append(f"\n### {resp_name} (Status: {status})")
            if resp_body:
                # Truncate very long responses
                if len(resp_body) > 2000:
                    resp_body = resp_body[:2000] + "\n... (truncated)"
                parts.append(f"```json\n{resp_body}\n```")
    
    return "\n".join(parts)


def main():
    print("Loading Postman collection...")
    with open(COLLECTION_PATH, "r") as f:
        collection = json.load(f)
    
    # Extract top-level description
    documents = []
    info = collection.get("info", {})
    if info.get("description"):
        documents.append(Document(
            page_content=f"# {info.get('name', 'API')}\n\n{info['description']}",
            metadata={
                "type": "overview",
                "name": info.get("name", "API Overview")
            }
        ))
    
    # Extract all endpoints
    items = collection.get("item", [])
    documents.extend(extract_endpoints(items))
    
    print(f"Extracted {len(documents)} documents")
    
    # Show sample
    print("\nSample documents:")
    for doc in documents[:3]:
        print(f"  - {doc.metadata.get('name')} ({doc.metadata.get('type')})")
    
    # Create embeddings and store
    print("\nCreating embeddings and storing in ChromaDB...")
    print("(This may take a few minutes for large collections)")
    print("(Using local HuggingFace embeddings - no API key needed for this step)")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"  # Fast, good quality, runs locally
    )
    
    # Clear existing vectorstore
    if os.path.exists(VECTORSTORE_PATH):
        import shutil
        shutil.rmtree(VECTORSTORE_PATH)
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH
    )
    
    print(f"\nDone! Stored {len(documents)} documents in {VECTORSTORE_PATH}")
    print("\nYou can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
