# FinAgent RAG — Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        User Interface                                 │
│            REST API (FastAPI)  │  CLI (Typer/Rich)                   │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     Agent Graph          │
                    │     (LangGraph)          │
                    │                          │
                    │  ┌──────────────────┐    │
                    │  │  Planner Node    │    │
                    │  │  (intent class.) │    │
                    │  └────────┬─────────┘    │
                    │           │              │
                    │  ┌────────▼─────────┐    │
                    │  │  Retrieval Node  │    │
                    │  │  (hybrid search) │    │
                    │  └────────┬─────────┘    │
                    │           │              │
                    │  ┌────────▼─────────┐    │
                    │  │  Worker Nodes    │    │
                    │  │  QA / Analysis   │    │
                    │  │  Comparison /    │    │
                    │  │  Report          │    │
                    │  └────────┬─────────┘    │
                    │           │              │
                    │  ┌────────▼─────────┐    │
                    │  │  Output Node     │    │
                    │  │  (format+cite)   │    │
                    │  └──────────────────┘    │
                    └────────────┬────────────┘
                                 │
           ┌─────────────────────┴──────────────────────┐
           │                                            │
┌──────────▼──────────┐                    ┌───────────▼───────────┐
│    RAG Pipeline      │                    │    LLM Backend         │
│    (LlamaIndex)      │                    │                        │
│                      │                    │  ┌─────────────────┐  │
│  ┌────────────────┐  │                    │  │ OpenAI GPT-4o   │  │
│  │ Document       │  │                    │  │ (cloud)         │  │
│  │ Ingestion      │  │                    │  └─────────────────┘  │
│  │ PDF/TXT/MD     │  │                    │           OR          │
│  └───────┬────────┘  │                    │  ┌─────────────────┐  │
│          │           │                    │  │ Ollama          │  │
│  ┌───────▼────────┐  │                    │  │ llama3.2:3b     │  │
│  │ Chunking       │  │                    │  │ (local/CPU)     │  │
│  │ SentenceSplit  │  │                    │  └─────────────────┘  │
│  └───────┬────────┘  │                    └───────────────────────┘
│          │           │
│  ┌───────▼────────┐  │
│  │ Embeddings     │  │
│  │ OpenAI ada-002 │  │
│  │ OR nomic-embed │  │
│  └───────┬────────┘  │
│          │           │
│  ┌───────▼────────┐  │
│  │ ChromaDB       │  │
│  │ (local vector  │  │
│  │  store)        │  │
│  └───────┬────────┘  │
│          │           │
│  ┌───────▼────────┐  │
│  │ Hybrid Search  │  │
│  │ BM25 + Vector  │  │
│  │ + RRF fusion   │  │
│  └────────────────┘  │
└──────────────────────┘
```

## Data Flow

### Ingestion Flow
```
Document (PDF/TXT/MD)
    → SentenceSplitter (512 tokens, 64 overlap)
    → Embedding model (OpenAI or nomic-embed-text)
    → ChromaDB persistent collection
```

### Query Flow
```
User Query
    → Planner (classify intent via LLM)
    → Retrieval (hybrid BM25 + vector search → RRF fusion)
    → Worker Node (QA / Analysis / Comparison / Report)
    → LLM (OpenAI or Ollama)
    → Output Node (format + cite sources)
    → Response (answer + sources + agent_trace)
```

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Agent framework | LangGraph | Explicit state machine, easy to extend nodes |
| RAG framework | LlamaIndex | Best retrieval quality, clean abstractions |
| Vector store | ChromaDB | Zero-config local persistence, no external service |
| LLM abstraction | Custom base class | Not locked to LangChain wrappers |
| Search fusion | Reciprocal Rank Fusion | Proven method, no extra model needed |
| Config | Pydantic Settings + YAML | Type-safe, env-override friendly |

## Agent Intent Routing

```
simple_qa      → direct Q&A from retrieved context
deep_analysis  → extract metrics + calculate ratios
comparison     → side-by-side company comparison
report         → full structured markdown report
```

## Extending the Project

**Add a new tool:**
1. Define function in `src/agents/tools.py`
2. Call it from the relevant node in `src/agents/nodes.py`

**Add a new LLM backend:**
1. Subclass `LLMClient` in `src/llm/`
2. Register in `src/llm/factory.py`

**Add a new document type:**
1. Add reader logic in `src/rag/indexer.py` `ingest_file()`

**Add a new agent node:**
1. Add node function in `src/agents/nodes.py`
2. Register and wire in `src/agents/graph.py`
