"""
Streamlit UI for the API Documentation Assistant.
"""

import streamlit as st
from chain import get_chain, get_vectorstore

st.set_page_config(
    page_title="TMO API Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 TMO API Documentation Assistant")
st.caption("Ask questions about The Mortgage Office API")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize chain (cached)
@st.cache_resource
def load_chain():
    return get_chain()

try:
    chain, retriever = load_chain()
except Exception as e:
    st.error(f"Error loading vectorstore. Did you run `python ingest.py` first?\n\nError: {e}")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📄 Sources"):
                for source in message["sources"]:
                    st.write(f"- {source}")

# Chat input
if prompt := st.chat_input("Ask about the TMO API..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documentation..."):
            # Get answer
            response = chain.invoke(prompt)
            answer = response.content
            
            # Get sources
            docs = retriever.invoke(prompt)
            sources = list(set([
                doc.metadata.get('path', doc.metadata.get('name', 'Unknown')) 
                for doc in docs
            ]))
            
            st.markdown(answer)
            
            with st.expander("📄 Sources"):
                for source in sources:
                    st.write(f"- {source}")
    
    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("""
    This assistant helps you navigate The Mortgage Office API documentation.
    
    **Example questions:**
    - How do I authenticate?
    - What endpoints are available for loans?
    - How do I create a new loan?
    - What fields are required for NewLoan?
    - How do I get a list of borrowers?
    """)
    
    st.divider()
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("Built with LangChain + ChromaDB + Claude")
