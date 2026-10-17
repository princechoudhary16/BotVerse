# BotVerse

A multi-domain RAG chatbot that answers questions using your own documents, databases, and WhatsApp chat exports. Built with LangChain, Gemini, and FAISS vector search.

## What It Does

Upload your documents or data, and BotVerse creates an intelligent conversational interface on top of it. It uses Retrieval-Augmented Generation (RAG) to find relevant context from your knowledge base and generate accurate, grounded answers.

**Supported data sources:**
- Text documents and PDFs
- Database records
- WhatsApp chat exports

## Architecture

```
User Query
    │
    ├── Chainlit UI (chat interface)
    │
    ├── LangChain Agent (MRKL framework)
    │     ├── Vector Store Tool (FAISS index over documents)
    │     ├── Database Tool (SQL queries)
    │     └── Chat History Tool (WhatsApp export analysis)
    │
    ├── Embeddings → HuggingFace Hub
    ├── LLM → Google Gemini
    │
    └── Response (with source attribution)
```

## Tech Stack

- **LangChain** — agent framework and chain orchestration
- **Gemini** — language model for generation
- **FAISS** — vector similarity search for document retrieval
- **HuggingFace Hub** — embedding models
- **Chainlit** — chat UI framework
- **Python** — core application

## Setup

```bash
git clone https://github.com/princechoudhary16/BotVerse.git
cd BotVerse

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install langchain chainlit faiss-cpu google-generativeai huggingface_hub

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Ingest your documents
python ingest.py

# Run the chatbot
chainlit run chat.py
```

## Project Structure

```
BotVerse/
├── chat.py        # Main chatbot application
├── model.py       # LLM configuration and chain setup
├── ingest.py      # Document ingestion and FAISS index creation
├── db.py          # Database connection and query tools
├── data/          # Knowledge base documents
├── index.faiss    # Pre-built FAISS index
├── index.pkl      # Serialized index metadata
└── chainlit.md    # Chainlit UI configuration
```

## Team

Built by Prince Choudhary, Anshuman Gupta, and Tisha Patel.

## License

MIT
