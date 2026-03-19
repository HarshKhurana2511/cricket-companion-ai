# Cricket Companion AI

Cricket Companion is a single-chat cricket assistant with **4 routed experiences**:
**Basic Q&A**, **Analyst**, **Strategy Simulator**, and **Fantasy Draft Assistant**.

It's built "library-first" (`cricket_companion/`) and exposed via **FastAPI + Streamlit** with **streaming** and a **tool timeline** (requests, responses, elapsed time, and citations).

## What's implemented

### Experiences
- **Basic Q&A**: RAG over a local retrieval store for cricket knowledge.
- **Analyst**: grounded stats queries via a DuckDB-backed `stats` tool.
- **Web freshness**: Tavily `web_search` + controlled `web_fetch` with citations + caching; optional ESPN scorecard ingestion fallback.
- **Strategy Simulator**: Monte Carlo match-state simulation via `sim-mcp` + UI form.
- **Fantasy Draft Assistant**:
  - `fantasy-mcp` optimizer (roles/budget/team limits + C/VC selection).
  - optional web-based injury/availability enrichment (`preferences.use_news=true`).
  - **UI**: upload player pool (CSV) or paste JSON + preferences + outputs + **alternatives**.

### Tooling & architecture
- **LangGraph** orchestrator: route -> plan -> execute tools -> compose answer.
- **MCP servers** for tools (`mcp_servers/...`) using a standard `{ok,data,error,meta}` response envelope.
- **Streaming** to UI via SSE, plus detailed tool traces (requests/responses/citations) in the UI Details panel.

## Quickstart (local)

### 1) Install
Prereqs: Python **3.11+**.

Using `uv` (recommended):
```powershell
uv sync
```

Or with pip:
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e .
```

### 2) Configure env
Create `.env` (see `.env.example`) and set at least:
```ini
CC_OPENAI_API_KEY=...

# Optional (enables freshness and fantasy news enrichment):
CC_TAVILY_API_KEY=...
```

### 3) Run API + UI
Terminal 1 (API):
```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Terminal 2 (UI):
```powershell
.\.venv\Scripts\streamlit.exe run ui\app.py
```

UI opens on `http://127.0.0.1:8501` and talks to API at `http://127.0.0.1:8000` by default.

## Using the UI
- **Chat** tab: free-form prompts; router decides Basic/Analyst/Web/Fantasy/Sim.
- **Simulator** tab: structured scenario form -> forces `sim` route.
- **Fantasy** tab:
  - **Upload CSV** (player pool) OR **Paste JSON** (`FantasyRequest` payload).
  - toggles: `use_news`, must-include/exclude, constraints.
  - output includes an XI table and (when feasible) an **alternatives** table.

## Docs
- Architecture: `docs/ARCHITECTURE.md`
- Implementation spec: `Implementation.md`
- Build progress log: `progress.md`
- Execution checklist: `task.md`

## Repo layout
- `cricket_companion/`: core library (router/planner/executor/graph, schemas, tool clients, response composers)
- `api/`: FastAPI wrapper (`/chat` and `/chat/stream`)
- `ui/`: Streamlit UI (chat + simulator + fantasy)
- `mcp_servers/`: one MCP server per tool (web, stats, sim, fantasy, retrieval)
- `pipelines/`: ingestion/update jobs (web index, Cricsheet -> DuckDB, etc.)
- `data/`, `cache/`, `artifacts/`: local dev outputs
