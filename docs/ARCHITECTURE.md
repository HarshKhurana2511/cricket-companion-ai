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

  MCP2 --> D1[Cricket data: DuckDB/Cricsheet]
  MCP1 --> R1[FAISS index (local)]
  MCP2 --> R1
  MCP3 --> R1
  MCP4 --> R1

  MCP1 --> W1[Web tools: Tavily/ESPN]
  W1 --> C1[JSON cache]
  C1 --> S3[(S3: cached web data)]

  G --> S[Summarizer]
  S -->|token/chunk stream| API
  API --> UI
```

## Decisions
- Single UI, single API endpoint; server routes internally.
- All tools are accessed via MCP; each tool is its own MCP server.
- Session memory: DuckDB on disk.
- Retrieval: FAISS on disk; in prod, sync artifacts to S3.
- Web cache: JSON persisted to S3 with 7-day TTL.
- AWS: minimal for now (ECS/Fargate single container + S3).

