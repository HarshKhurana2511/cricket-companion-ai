# Architecture (simple pet project)

This is a single chat UI that can handle 4 kinds of cricket requests:
- Basic Cricket Chatbot
- Analyst Copilot
- Strategy & Scenario Simulator
- Fantasy Draft Assistant

Users do not pick a mode. A LangGraph orchestrator routes each prompt to the right agent.

## High-level flow

```mermaid
flowchart LR
  U[User] --> UI[Streamlit Chat UI]
  UI --> API[FastAPI: POST /chat (streaming)]
  API --> G[LangGraph Orchestrator]

  G -->|route| A1[Agent: Basic Chatbot]
  G -->|route| A2[Agent: Analyst]
  G -->|route| A3[Agent: Simulator]
  G -->|route| A4[Agent: Fantasy]

  A1 -->|tools via MCP| MCP1[MCP servers (one tool per server)]
  A2 -->|tools via MCP| MCP2[MCP servers (one tool per server)]
  A3 -->|tools via MCP| MCP3[MCP servers (one tool per server)]
  A4 -->|tools via MCP| MCP4[MCP servers (one tool per server)]

  MCP2 --> D1[Cricket data: Cricsheet YAML -> DuckDB]
  MCP1 --> R1[FAISS index (local)]
  MCP2 --> R1
  MCP3 --> R1
  MCP4 --> R1
  R1 --> S3I[(S3: FAISS artifacts)]

  MCP1 --> W1[Web tools: Tavily + ESPN match/scorecards]
  W1 --> C1[JSON cache (TTL 7d, 1 retry)]
  C1 --> S3[(S3: cached web data)]

  G --> S[Summarizer]
  S -->|token/chunk stream| API
  API --> UI
```

## Decisions
- Single UI, single API endpoint; server routes internally.
- All tools are accessed via MCP; each tool is its own MCP server.
- Data: Cricsheet YAML-only into a single DuckDB, with incremental updates.
- Session memory: DuckDB on disk (persisted across restarts).
- Retrieval: FAISS on disk; in prod, sync artifacts to S3 on startup/shutdown.
- Web: Tavily used for both general and current queries; ESPN ingestion focuses on match/scorecard pages (no full articles).
- Web cache: JSON persisted to S3 with 7-day TTL; tool calls retry once and then fall back to cached results.
- Citations: web-derived outputs return source URLs and fetched timestamps.
- AWS: minimal for now (ECS/Fargate single container + S3).

## Containerization
- Local: docker-compose can run `api`, `ui`, and one MCP server per tool as separate services (optional; `uv` local runs are fine too).
- Prod: a single ECS/Fargate container runs API/UI + MCP servers as multiple processes (keep it simple).
