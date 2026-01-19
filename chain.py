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


def get_vectorstore():
    """Load the existing vectorstore."""
    embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    encode_kwargs={'normalize_embeddings': True},
    query_instruction="Represent this sentence for searching relevant passages: ")
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
        search_kwargs={"k": 10}  # Retrieve top 5 relevant chunks
    )
    
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
    
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    return chain, retriever


def query(question: str):
    """Query the documentation."""
    chain, retriever = get_chain()
    
    # Get answer
    response = chain.invoke(question)
    
    # Get sources for display
    docs = retriever.invoke(question)
    sources = [doc.metadata.get('path', doc.metadata.get('name', 'Unknown')) for doc in docs]
    
    return {
        "answer": response.content,
        "sources": sources
    }


if __name__ == "__main__":
    # Test query
    result = query("How do I authenticate to the API?")
    print("Answer:", result["answer"])
    print("\nSources:", result["sources"])
