"""
RAG chain for querying API documentation.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from sentence_transformers import CrossEncoder

load_dotenv()

VECTORSTORE_PATH = "./vectorstore"

# System prompt for the assistant
SYSTEM_PROMPT = """You are an API documentation assistant for The Mortgage Office (TMO) API.

Your job is to help developers understand and use the API correctly.

Rules:
1. Only answer based on the provided context from the documentation
2. If the context doesn't contain the answer, say "I don't have information about that in the documentation"
3. Include specific endpoint names, methods, and URLs when relevant
4. Show request/response examples when available
5. Be concise but complete

Context from documentation:
{context}

Question: {question}

Answer:"""

# Load reranker model (do this once, outside the function)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

def rerank_docs(query, docs, top_k=5):
    """Rerank documents using cross-encoder."""
    if not docs:
        return docs
    
    # Score each doc
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    print("\n=== RERANKING ===")
    print(f"Query: {query}")
    for score, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True):
        name = doc.metadata.get('name', 'Unknown')
        print(f"  {score:.4f} - {name}")
        
    # Sort by score, return top_k
    scored_docs = list(zip(scores, docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    return [doc for score, doc in scored_docs[:top_k]]

def get_vectorstore():
    """Load the existing vectorstore."""
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        encode_kwargs={'normalize_embeddings': True},
        query_instruction="Represent this sentence for searching relevant passages: "
    )
    return Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings
    )


def format_docs(docs):
    """Format retrieved documents for the prompt."""
    formatted = []
    for doc in docs:
        metadata = doc.metadata
        source = metadata.get('path', metadata.get('name', 'Unknown'))
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def get_chain():
    """Create the RAG chain."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20}  # Retrieve 20, rerank to top 5
    )
    
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    
    return retriever, prompt, llm


def query(question: str):
    """Query the documentation with reranking."""
    retriever, prompt, llm = get_chain()
    
    # Step 1: Retrieve top 20
    docs = retriever.invoke(question)
    
    # Step 2: Rerank to top 5
    reranked_docs = rerank_docs(question, docs, top_k=5)
    
    # Step 3: Format context
    context = format_docs(reranked_docs)
    
    # Step 4: Generate answer
    formatted_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(formatted_prompt)
    
    # Get sources from reranked docs
    sources = [doc.metadata.get('path', doc.metadata.get('name', 'Unknown')) for doc in reranked_docs]
    
    return {
        "answer": response.content,
        "sources": sources
    }


if __name__ == "__main__":
    # Test query
    result = query("How do I authenticate to the API?")
    print("Answer:", result["answer"])
    print("\nSources:", result["sources"])
