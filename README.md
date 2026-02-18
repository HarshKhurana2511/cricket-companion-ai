# Cricket Companion

Single-chat cricket assistant routed via LangGraph (Basic Q&A, Analyst, Simulator, Fantasy).

## Docs
- Architecture (committed): `docs/ARCHITECTURE.md`
- Implementation spec (local-only): `Implementation.md`

## Repo layout
- `cricket_companion/`: core library (agents, LangGraph, schemas, caching, MCP clients)
- `api/`: FastAPI wrapper (single `POST /chat` with streaming)
- `ui/`: Streamlit UI (chat)
- `mcp_servers/`: one MCP server per tool
- `pipelines/`: ingestion/update jobs (Cricsheet YAML -> DuckDB)
- `infra/`: docker/compose + minimal AWS deployment notes
- `docs/`: architecture docs
- `tests/`: tests (minimal, as needed)

