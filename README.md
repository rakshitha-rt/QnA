# AI Knowledge Assistant

A lightweight RAG assistant that answers questions grounded in uploaded documents and structured data. Supports multi-turn clarification, GraphRAG-style retrieval, and pandas-powered structured analysis.

---

## Architecture Overview

```
Upload                Query
  │                     │
  ▼                     ▼
Ingestion          Clarification Agent
  │                (ambiguity check)
  ├─ Unstructured        │
  │  ├─ Chunk text       ├─ Ambiguous → return question to user
  │  ├─ Extract          │
  │    entities/rels     └─ Clear → Planner Agent
  │  ├─ Build KG                      │
  │  └─ Store in              ┌───────┴────────┐
  │     Qdrant            query_unstructured  query_structured
  │                           │                    │
  └─ Structured           Qdrant (KG)         PythonREPL
     ├─ Read CSV/Excel         │               (pandas)
     └─ Store schema      naive / local /          │
        in Qdrant          global modes             │
                               └────────────────────┘
                                        │
                                   Final Answer
```

**Key components:**

| Component | File | Responsibility |
|---|---|---|
| FastAPI server | `app/main.py` | HTTP endpoints, file upload routing |
| Ingestion | `app/utlis/ingestion.py` | Chunking, entity extraction, graph building |
| Graph store | `app/utlis/graph.py` | Exports `.graphml` for visualisation |
| Vector store | `app/stores/intelligence.py` | Qdrant collections for chunks, entities, relationships, structured metadata |
| Project store | `app/stores/project.py` | Project CRUD backed by Qdrant |
| LangGraph pipeline | `app/handlers/node.py` | State machine: clarification → planner |
| Clarification agent | `app/agents/clarification.py` | Detects genuine ambiguity, asks one focused question |
| Planner agent | `app/agents/planner.py` | Dual-tool agent (unstructured + structured) |
| Coder agent | `app/agents/coder.py` | Writes and executes pandas code via PythonREPL |
| Prompts | `app/agents/extraction.py` | Entity extraction, keyword, file-summary, and description-merge prompt templates |
| Data models | `app/model.py` | Pydantic state and request/response schemas |

---

## Setup

### Prerequisites

- Docker and Docker Compose
- OpenAI API key

### Run with Docker Compose

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

docker compose up --build
```

Open `http://localhost:8000` in your browser.

### Services

| Service | Port | Description |
|---|---|---|
| `app` | 8000 | FastAPI application |
| `qdrant` | 6333 | Vector database |
| `sandbox` | 8001 (internal) | Code execution sandbox — not exposed to host |

The sandbox is on an `internal` Docker network with no outbound internet access. Only the `app` container can reach it.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend UI |
| `GET` | `/projects` | List all projects |
| `POST` | `/projects` | Create a project `{"name": "..."}` |
| `POST` | `/projects/{id}/upload` | Upload a document (multipart) |
| `POST` | `/query` | Ask a question (see body below) |
| `GET` | `/graph/{id}` | View the knowledge graph for a project |

**Query request body:**

```json
{
  "project_id": "uuid",
  "query": "Which branch has the highest sales?",
  "query_type": "naive",
  "clarification_history": []
}
```

`query_type` controls unstructured retrieval mode:
- `naive` — dense vector search over raw text chunks
- `local` — retrieves entities by low-level keywords, enriched with source chunks
- `global` — retrieves relationships by high-level keywords, enriched with entity chunks

---

## Key Design Choices

### GraphRAG for unstructured data
Rather than naive chunked retrieval, the ingestion pipeline extracts entities and relationships from each chunk (via GPT-4o), deduplicates entities across the document, merges their descriptions, and stores them as a knowledge graph in Qdrant. This means queries can find answers via conceptual relationships, not just keyword overlap.

### Three retrieval modes
- `naive`: best for direct lookup questions where the answer is in a specific passage
- `local`: best for questions about specific entities ("what is X?")
- `global`: best for thematic or relationship questions ("how does X relate to Y?")

### Dual-source planner
The planner agent treats structured and unstructured data as independent sources and queries both before synthesising an answer. It is explicitly instructed not to manufacture connections between the two when none exist.

### Clarification as a graph node
Ambiguity handling is a first-class node in the LangGraph state machine, not a prompt hack. The agent checks every query before passing it to the planner, and full multi-turn clarification history is carried in state so resolved terms are never re-asked.

### File summaries at ingestion time
Each uploaded file gets a short LLM-generated summary stored with the project. These summaries are injected into the clarification agent's context so it can ask informed, file-aware questions ("Do you mean the Q3 sales report or the inventory log?").

---

## Guardrails

### 1. Clarification gate (ambiguity)
**What it does:** Intercepts queries before they reach the planner. If a query contains a term whose misinterpretation would invalidate the entire answer, the agent returns a clarifying question with 3–4 concrete options instead of proceeding.

**Risk mitigated:** Answering the wrong question confidently — e.g. "What is the best product?" answered with arbitrary ranking criteria.

**Limitation:** The LLM determines what counts as ambiguous; it may occasionally over-ask (annoying) or under-ask (silently pick the wrong interpretation). The prompt is tuned to err heavily toward pass-through.

### 2. Hallucination boundary in the planner prompt
**What it does:** The planner is explicitly instructed: "Never fabricate or infer document context. Only cite policies, definitions, or business rules that were explicitly returned by `query_unstructured`. If the documents returned were irrelevant or empty, answer solely from the structured data — and vice versa."

**Risk mitigated:** The model inventing plausible-sounding policies or figures that are not in the uploaded documents.

**Limitation:** Prompt-level guardrails can be overcome by a sufficiently confident model response. There is no post-hoc grounding check that verifies the answer against retrieved chunks.

### 3. Structured-data isolation (coder agent)
**What it does:** The coder agent loads data only from the exact file paths provided in the request context (column metadata + file paths extracted at upload time). It has no access to the internet or the filesystem beyond those paths.

**Risk mitigated:** The model querying wrong files, fabricating data values, or accessing files outside the project.

**Limitation:** The PythonREPL tool executes arbitrary code; a malicious query could in principle read other local files if the file paths were injected via the query string.

---

## Evaluation Approach

A lightweight evaluation was designed across three dimensions:

### 1. Retrieval relevance
For each test query, inspect the top-k chunks / entities / relationships returned before the planner synthesises. Check whether the retrieved items contain the information needed to answer correctly.

Manual spot-check rubric (per retrieved item):
- 2 — directly answers the query
- 1 — related but indirect
- 0 — irrelevant

Target: mean score ≥ 1.5 across a 20-query test set.

### 2. Groundedness
After the planner returns an answer, re-run `query_unstructured` with the same sub-queries the planner used. For each factual claim in the answer, check whether supporting text appears in the retrieved chunks.

Groundedness score = (claims traceable to retrieved context) / (total factual claims)

Target: ≥ 0.85 on unstructured queries.

Run the evaluation script (requires the app to be running):

```bash
python eval.py                          # creates a fresh project, uploads sample files, runs all checks
python eval.py --project-id <id>        # reuse an existing ingested project
python eval.py --output results.json    # also write full JSON results
```

### 3. Clarification precision
For a set of 10 deliberately ambiguous and 10 clearly unambiguous queries, measure:
- False positive rate (asked for clarification when not needed)
- False negative rate (passed through when clarification was needed)

Target: false positive < 20%, false negative < 10%.

---

## Assumptions

- Documents are in English.
- Uploaded PDFs are text-based (not scanned images); OCR is not supported.
- Each project is a logically coherent set of documents (mixing unrelated corpora in one project will produce noisy knowledge graph results).
- Structured files (CSV/Excel/Parquet) remain on disk at the path they were uploaded to for the lifetime of the project; the coder agent loads them directly.
- The OpenAI `o4-mini` model is used for the clarification, planner, and keyword agents; `gpt-4o` is used for entity extraction and summarisation.

---

## Limitations

- **PDF support is incomplete.** PDFs are read as plain text (`open(file, "r")`), which fails on binary-encoded files. `pypdf` is already listed in `requirements.txt` but is not yet wired into the ingestion pipeline.
- **No authentication.** Any caller can query any project by ID.
- **No retrieval confidence threshold.** The planner is called even when all retrieved chunks have low cosine similarity to the query.

---

## Future Improvements

- Wire `pypdf` (already in `requirements.txt`) into `process_unstructured` to replace the plain-text fallback, and add OCR support for scanned PDFs
