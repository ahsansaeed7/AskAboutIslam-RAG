# AskAboutIslam-RAG

A Retrieval-Augmented Generation (RAG) system that answers questions about Islam, grounded in Wikipedia content. Built as a learning project to compare three different implementation approaches — raw Python, LangChain, and LangGraph — on the same underlying data and task.

**Live demo:** [https://askaboutislam-rag.onrender.com](https://askaboutislam-rag.onrender.com)
*(Backend is hosted on Render's free tier and may take 30-60 seconds to wake up on first request after inactivity.)*

---

## What this project does

The system scrapes a Wikipedia article, chunks and embeds it into a vector database, and answers natural-language questions using retrieval-augmented generation — with three parallel pipeline implementations so behavior and code complexity can be directly compared.

- **Raw Python pipeline** — every step (retrieval, prompt construction, generation) implemented manually with no framework abstraction, to understand the underlying mechanics.
- **LangChain pipeline** — the same logic rebuilt using LangChain's components and LCEL chaining.
- **LangGraph pipeline** — an agentic version with a self-correction loop: it checks whether its own answer is actually grounded in retrieved context, and if not, reformulates the query and retries (up to a max retry count) before giving up and honestly saying it doesn't know.

All three are exposed through a FastAPI backend with a pipeline selector, and a React frontend lets you pick any of the three and compare answers, sources, response time, and (for LangGraph) retry count and grounding status.

---![REACT UI](ui_ss.png.png)

## Architecture

```
Wikipedia API
     │
     ▼
  Scraper (scraper.py) ──► section-structured JSON
     │
     ▼
  Chunker (chunking.py) ──► ~500-token chunks with overlap, tagged with section metadata
     │
     ▼
  Embeddings (sentence-transformers, all-MiniLM-L6-v2, local/free)
     │
     ▼
  Chroma vector store (persistent, local)
     │
     ▼
  ┌─────────────┬──────────────────┬───────────────────┐
  │ Raw pipeline │ LangChain pipeline │ LangGraph pipeline │
  └─────────────┴──────────────────┴───────────────────┘
     │
     ▼
  FastAPI backend (pipeline selector via /query endpoint)
     │
     ▼
  React frontend (pipeline comparison UI)
```

**Generation model:** Ollama (Llama 3.2) for local development — free, offline, no rate limits. Groq's hosted Llama API for the deployed version, since cloud hosting can't run a local Ollama server. Switching between them is a single environment variable (`LLM_PROVIDER`), handled through a small provider-abstraction layer (`llm_provider.py`).

---

## Tech stack

| Layer | Technology |
|---|---|
| Data source | Wikipedia API |
| Chunking | Custom token-based splitter (`tiktoken`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`), local, free |
| Vector store | Chroma (persistent, local) |
| Orchestration | Raw Python / LangChain (LCEL) / LangGraph |
| Generation (dev) | Ollama, Llama 3.2 (local, free) |
| Generation (prod) | Groq API, Llama 3.1 (hosted, free tier) |
| Backend | FastAPI |
| Frontend | React (Vite) |
| Backend hosting | Render |
| Frontend hosting | Vercel |

---

## Design decisions and tradeoffs

**Why build all three pipelines instead of just one?**
Building the raw version first forced an understanding of what LangChain and LangGraph actually abstract away — the HTTP calls to the embedding model, the vector similarity search, the prompt construction — rather than treating them as black boxes. Keeping all three in the final product also turns the comparison itself into a demonstrable feature, not just a private learning exercise.

**Raw Python vs. LangChain**
The LangChain version's "glue" logic (retrieve → prompt → generate) is meaningfully shorter, thanks to LCEL's `|` chaining. The tradeoff: when something breaks, the raw version's errors are immediate and traceable to a specific function call, while LangChain's errors pass through several internal abstraction layers, which is harder to debug — something felt directly while building this project, since several early bugs (path resolution, embedding function mismatches) were far easier to diagnose in the raw pipeline than they would have been starting directly in LangChain.

**LangGraph's self-correction loop**
The grounding-check step is itself an LLM call, and early testing showed it was inconsistent — the same type of "I don't know" refusal answer was classified as `GROUNDED`, `NOT_GROUNDED`, and `INSUFFICIENT` across different retries, a known limitation of "LLM-as-judge" patterns, especially with smaller models like Llama 3.2 3B. This was fixed by adding a deterministic, rule-based pre-check (matching common refusal phrases) before falling back to the LLM judge — making refusal detection consistent and cutting an unnecessary LLM call in the common case.

**Retrieval limitations**
Testing revealed that semantic search doesn't always surface the most specific or temporally relevant chunk — e.g., "When did Islam start?" consistently failed to retrieve the section containing the actual founding date, even across multiple LangGraph-driven query reformulations. This points to a real limitation of basic cosine-similarity retrieval and is a natural next area for improvement (e.g., hybrid keyword + semantic search, or increasing `k`).

**Local-first development**
Embeddings run locally via `sentence-transformers` rather than a paid API, avoiding both cost and rate limits while iterating. Generation runs via free local Ollama during development and switches to Groq's free-tier hosted API only for deployment, since local Ollama isn't reachable from a cloud host — a realistic dev/prod environment split.

---

## Evaluation

A test set of factual, comparative, and out-of-scope questions was run against each pipeline (`evaluation/eval_runner.py`), with results manually scored for correctness and groundedness.

| Metric | Result |
|---|---|
| Correct answers | 8/8 |
| Properly grounded answers | 8/8 |
| Correct refusals on out-of-scope questions | 2/2 |

*(See `evaluation/results/` for full run logs.)*

---

## What I'd improve with more time

- Hybrid retrieval (keyword + semantic search) to address the retrieval gaps found during testing
- Merge very short trailing chunks (a side effect of token-based chunking) back into their preceding chunk
- Expand the evaluation set with more ambiguous/multi-section questions
- Add a more deterministic grounding-check step to reduce reliance on LLM-as-judge entirely
- Add response streaming in the frontend for a better UX during LangGraph's longer, multi-step runs

---

## Running locally

```bash
# Clone the repo
git clone https://github.com/ahsansaeed7/AskAboutIslam-RAG.git
cd AskAboutIslam-RAG

# Backend setup
pip install -r requirements.txt

# Set up .env with your GROQ_API_KEY (or run Ollama locally and leave LLM_PROVIDER=ollama)

# Run the backend
python app/main.py

# In a separate terminal, run the frontend
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173` for the frontend, or `http://localhost:8000/docs` for the API's interactive documentation.

---

## Project structure

```
├── app/                  # FastAPI backend
├── data/                 # Raw and processed Wikipedia data, eval questions
├── evaluation/           # Eval runner and results
├── frontend/             # React frontend
├── src/                  # Core pipeline code
│   ├── scraper.py
│   ├── chunking.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── prompt_templates.py
│   ├── llm_provider.py
│   ├── rag_pipeline.py       # raw Python version
│   ├── rag_langchain.py      # LangChain version
│   └── rag_langgraph.py      # LangGraph self-correcting version
├── vector_store/         # Persisted Chroma database
└── config.py
```