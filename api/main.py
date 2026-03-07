from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from cricket_companion.chat_models import ChatRequest, ChatResponse
from cricket_companion.chat_service import handle_chat
from cricket_companion.chat_service import stream_chat


app = FastAPI(title="Cricket Companion API", version="0.1.0")

# Local dev CORS (Streamlit typically runs on 8501)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        return handle_chat(req)
    except ValueError as exc:
        # Bad input/config; surface as a client error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """
    Phase 2.7.2: SSE stream for chat responses.

    Events:
    - chunk: {"text": "..."}  (streamed answer deltas)
    - result: <ChatResponse JSON> (final artifacts/traces)
    - error: {"message": "..."}
    - done: {"ok": true/false}
    """

    def gen():
        for evt in stream_chat(req):
            event_name = evt.get("event") or "message"
            data = evt.get("data") or {}
            payload = json.dumps(data, ensure_ascii=False)
            yield f"event: {event_name}\n"
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
