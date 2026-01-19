# TMO API Documentation Assistant

A RAG-powered chatbot that answers questions about The Mortgage Office API documentation.

## Tech Stack

- **LangChain**: Orchestration framework
- **ChromaDB**: Vector database (local, no cloud needed)
- **HuggingFace Embeddings**: Free, local embeddings (no API key needed)
- **Anthropic Claude**: LLM for generation
- **Streamlit**: Web UI

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Anthropic API key

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

Or create a `.env` file:
```
ANTHROPIC_API_KEY=your-key-here
```

### 3. Add your Postman collection

Place your `collection.json` file in the project root.

### 4. Ingest the documentation

```bash
python ingest.py
```

This parses the Postman collection and stores embeddings in ChromaDB.
(No API key needed for this step - uses local embeddings)

### 5. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Project Structure

```
api-doc-assistant/
├── app.py              # Streamlit UI
├── chain.py            # RAG chain logic
├── ingest.py           # Postman → ChromaDB ingestion
├── collection.json     # Your Postman export
├── vectorstore/        # ChromaDB storage (created after ingest)
├── requirements.txt
└── README.md
```

## How It Works

1. **Ingestion**: Parses Postman JSON → extracts endpoints, descriptions, examples → creates embeddings (local) → stores in ChromaDB

2. **Query**: User question → embed → find similar docs in ChromaDB → pass to Claude with context → generate answer

## Skills Demonstrated

- RAG architecture (retrieval + generation)
- Vector database usage (ChromaDB)
- Prompt engineering (system prompt design)
- LangChain (chains, retrievers)
- Anthropic API integration

## Example Queries

- "How do I authenticate to the API?"
- "What endpoints are available for loans?"
- "How do I create a new loan application?"
- "What fields are required for the NewLoan endpoint?"
- "How do I get borrower information?"
