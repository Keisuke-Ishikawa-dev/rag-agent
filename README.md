# RAG Agent

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![PostgreSQL](https://img.shields.io/badge/pgvector-PostgreSQL-336791?logo=postgresql)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)

Production-ready AI agent with **ReAct reasoning**, **RAG pipeline**, and **pgvector** semantic search. Upload any document and chat with it through a REST API.

## How It Works

```
User Question
      │
      ▼
┌─────────────────────────────────────────┐
│            ReAct Agent Loop             │
│                                         │
│  Thought → Action → Observation → ...   │
│                                         │
│  Tools available:                       │
│  ┌─────────────────┐                   │
│  │ 🔍 RAG Search   │ ← pgvector        │
│  │ 🌐 Web Search   │ ← DuckDuckGo      │
│  │ 🧮 Calculator   │ ← Python eval     │
│  └─────────────────┘                   │
└─────────────┬───────────────────────────┘
              │
              ▼
        Final Answer
```

## Features

- **ReAct Agent** — step-by-step reasoning before every action (LangChain)
- **RAG Pipeline** — chunk → embed → store → retrieve → answer
- **pgvector** — cosine similarity search with IVFFlat index
- **Async** — FastAPI + asyncpg + asyncio, no blocking I/O
- **Conversation Memory** — sliding window of last 5 exchanges
- **Multi-format** — upload PDF, TXT, or Markdown files
- **BDD Tests** — Gherkin feature specs for agent behaviors
- **Docker** — single command to run full stack

## Stack

| Layer | Technology |
|-------|-----------|
| Agent framework | LangChain (ReAct) |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | text-embedding-3-small (1536 dim) |
| Vector store | PostgreSQL + pgvector |
| DB driver | asyncpg (fully async) |
| API | FastAPI + uvicorn |
| Container | Docker + docker-compose |
| Tests | pytest-bdd (Gherkin) |

## Quick Start

```bash
# 1. Clone and configure
https://github.com/Keisuke-Ishikawa-dev/rag-agent.git
cd rag-agent
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 2. Run everything
docker-compose up -d

# 3. Check it's running
curl http://localhost:8000/health
```

## Usage

### Upload a document

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@your_document.pdf"

# Response:
# {"message": "Indexed 24 chunks from your_document.pdf"}
```

### Ask a question

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the main topics covered in the document?"}'
```

```json
{
  "answer": "The document covers three main topics: ...",
  "sources": [
    {"content": "Chapter 1 introduces...", "score": 0.94},
    {"content": "The key concepts are...", "score": 0.91}
  ]
}
```

### ReAct trace (verbose mode)

```
Question: What is the refund policy?

Thought: I should search the knowledge base first.
Action: search_knowledge_base
Action Input: "refund policy"
Observation: [Score: 0.94] Refunds are processed within 14 days of purchase...

Thought: I found a clear answer in the knowledge base.
Final Answer: According to the document, refunds are processed within 14 days...
```

## Project Structure

```
rag-agent/
├── src/
│   ├── agent/
│   │   ├── react_agent.py    # ReAct agent (LangChain AgentExecutor)
│   │   └── tools.py          # RAG search, web search, calculator tools
│   ├── rag/
│   │   └── embeddings.py     # Chunking, embedding, similarity search
│   ├── database/
│   │   └── connection.py     # Async PostgreSQL init + pgvector setup
│   ├── api/
│   │   └── main.py           # FastAPI endpoints (upload, chat, health)
│   └── config.py             # Pydantic settings from .env
├── tests/
│   └── test_agent.feature    # BDD scenarios (Gherkin)
├── Dockerfile
├── docker-compose.yml        # App + PostgreSQL with pgvector
└── requirements.txt
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `DATABASE_URL` | PostgreSQL connection string |
| `POSTGRES_USER` | DB username (default: raguser) |
| `POSTGRES_PASSWORD` | DB password |
| `POSTGRES_DB` | DB name (default: ragdb) |

## Architecture Decisions

**Why pgvector over dedicated vector DBs?**
PostgreSQL + pgvector keeps the stack simple — one database for both metadata and vectors. IVFFlat index handles millions of vectors efficiently without adding Pinecone/Weaviate as a dependency.

**Why ReAct over simple RAG?**
ReAct lets the agent decide *when* to use the knowledge base vs. web search vs. calculation. A simple RAG pipeline always retrieves — ReAct only retrieves when it makes sense.

**Why asyncio throughout?**
Document embedding and LLM calls are I/O-bound. Async allows handling concurrent requests without blocking the event loop.

## License

MIT
