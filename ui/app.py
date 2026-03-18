from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any, Iterator, Tuple
from uuid import uuid4

import httpx
import pandas as pd
import streamlit as st


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _init_state() -> None:
    api_base_default = os.environ.get("CC_API_BASE_URL") or "http://127.0.0.1:8000"
    st.session_state.setdefault("api_base_url", api_base_default)
    st.session_state.setdefault("user_id", "local-user")
    st.session_state.setdefault("session_id", f"ui-session-{uuid4()}")
    # Each message: {role, content, ts, request_id, result?, debug_events?}
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("selected_request_id", None)
    st.session_state.setdefault("show_details", False)
    st.session_state.setdefault("debug", True)


def _sse_events(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_s: float = 300.0,
) -> Iterator[Tuple[str, dict[str, Any]]]:
    """
    Parse SSE stream from the API.

    Yields (event_name, data_dict).
    """
    event_name: str | None = None
    data_lines: list[str] = []

    with httpx.Client(timeout=timeout_s) as client:
        with client.stream("POST", url, json=payload, headers={"Accept": "text/event-stream"}) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                line = (raw or "").strip()

                # Event boundary.
                if line == "":
                    if event_name is not None:
                        data_raw = "\n".join(data_lines).strip()
                        try:
                            data = json.loads(data_raw) if data_raw else {}
                        except Exception:
                            data = {"_raw": data_raw}
                        yield (event_name, data)
                    event_name = None
                    data_lines = []
                    continue

                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                    continue

                if line.startswith("data:"):
                    data_lines.append(line.split(":", 1)[1].lstrip())
                    continue


def _append_message(
    role: str,
    content: str,
    request_id: str,
    *,
    result: dict[str, Any] | None = None,
    debug_events: list[dict[str, Any]] | None = None,
) -> None:
    msg: dict[str, Any] = {"role": role, "content": content, "ts": _utc_now_iso(), "request_id": request_id}
    if isinstance(result, dict):
        msg["result"] = result
    if isinstance(debug_events, list):
        msg["debug_events"] = debug_events
    st.session_state["messages"].append(msg)


def _overs_to_balls(overs_text: str) -> int | None:
    """
    Convert an overs string like "10" or "10.2" into balls (10*6 + 2).
    """
    s = (overs_text or "").strip()
    if not s:
        return None
    try:
        if "." not in s:
            o = int(s)
            return o * 6
        o_str, b_str = s.split(".", 1)
        o = int(o_str)
        b = int(b_str)
        if b < 0 or b > 5:
            return None
        return o * 6 + b
    except Exception:
        return None


def _stream_turn(
    *,
    user_text: str,
    request_id: str,
    msg_id: str,
    placeholder: Any,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any] | None, list[dict[str, Any]]]:
    stream_url = st.session_state["api_base_url"].rstrip("/") + "/chat/stream"
    payload = {
        "session_id": st.session_state["session_id"],
        "request_id": request_id,
        "user_id": st.session_state["user_id"],
        "message": {
            "message_id": msg_id,
            "role": "user",
            "content": user_text,
            "created_at": _utc_now_iso(),
            "metadata": metadata or {},
        },
        "debug": bool(st.session_state["debug"]),
        "max_context_messages": 30,
    }

    assembled = ""
    debug_events: list[dict[str, Any]] = []
    result_payload: dict[str, Any] | None = None
    try:
        for event, data in _sse_events(url=stream_url, payload=payload):
            if event == "chunk":
                delta = str(data.get("text") or "")
                assembled += delta
                if placeholder is not None:
                    placeholder.markdown(assembled)
            elif event == "result":
                if isinstance(data, dict):
                    result_payload = data
            elif event == "error":
                err = str(data.get("message") or "Unknown error")
                if bool(st.session_state.get("debug")):
                    debug_events.append({"event": event, "data": data, "ts": _utc_now_iso(), "request_id": request_id})
                assembled += f"\n\nError: {err}"
                if placeholder is not None:
                    placeholder.markdown(assembled)
            elif event == "done":
                break
            else:
                if bool(st.session_state.get("debug")):
                    debug_events.append({"event": event, "data": data, "ts": _utc_now_iso(), "request_id": request_id})
    except Exception as exc:
        assembled += f"\n\nError: {exc}"
        if placeholder is not None:
            placeholder.markdown(assembled)

    return assembled.strip() or "(no answer)", result_payload, debug_events


def _render_result_payload(result: dict[str, Any]) -> None:
    subtabs = st.tabs(["Tables", "Charts", "Citations"])
    tables_index = _tables_by_id(result)

    with subtabs[0]:
        tables = result.get("tables") or []
        if not tables:
            st.write("No tables.")
        for t in tables:
            if not isinstance(t, dict):
                continue
            st.subheader(str(t.get("name") or "table"))
            rows = t.get("rows") or []
            cols = [c.get("name") for c in (t.get("columns") or []) if isinstance(c, dict)]
            df = pd.DataFrame(rows)
            if cols:
                keep = [c for c in cols if c in df.columns]
                if keep:
                    df = df[keep]
            st.dataframe(df, use_container_width=True, height=320)

    with subtabs[1]:
        charts = result.get("charts") or []
        if not charts:
            st.write("No charts.")
        for ch in charts:
            if not isinstance(ch, dict):
                continue
            st.subheader(str(ch.get("title") or "chart"))
            chart_type = ch.get("chart_type")
            table_id = ch.get("table_id")
            x = ch.get("x")
            y = ch.get("y") or []

            table = tables_index.get(str(table_id)) if table_id else None
            if not table:
                st.warning("Missing source table for chart.")
                continue

            df = pd.DataFrame(table.get("rows") or [])
            if x not in df.columns:
                st.warning("Chart x column not found in table.")
                continue

            y_cols = [c for c in y if c in df.columns]
            if not y_cols:
                st.warning("Chart y columns not found in table.")
                continue

            df2 = df[[x] + y_cols].copy()
            df2 = df2.set_index(x)

            if chart_type == "bar":
                st.bar_chart(df2, use_container_width=True)
            elif chart_type == "line":
                st.line_chart(df2, use_container_width=True)
            elif chart_type == "scatter":
                st.scatter_chart(df[[x, y_cols[0]]], x=x, y=y_cols[0], use_container_width=True)
            elif chart_type == "hist":
                st.write("Histogram rendering is minimal; showing bar counts for first y column.")
                s = pd.to_numeric(df[y_cols[0]], errors="coerce").dropna()
                bins = pd.cut(s, bins=10).value_counts().sort_index()
                st.bar_chart(bins, use_container_width=True)
            else:
                st.write("Unknown chart_type:", chart_type)
                st.dataframe(df2, use_container_width=True, height=240)

    with subtabs[2]:
        citations = result.get("citations") or []
        if not citations:
            st.write("No citations.")
        for c in citations:
            if not isinstance(c, dict):
                continue
            title = c.get("title") or c.get("url")
            st.markdown(f"- **{title}**  \n  `{c.get('url')}`  \n  fetched_at: `{c.get('fetched_at')}`")


def _render_chat() -> None:
    messages = st.session_state.get("messages") or []
    for i, m in enumerate(messages):
        role = str(m.get("role") or "assistant")
        with st.chat_message(role):
            if role == "assistant":
                col_text, col_btn = st.columns([0.86, 0.14], gap="small")
                with col_text:
                    st.markdown(str(m.get("content") or ""))
                with col_btn:
                    if st.button("Details", key=f"details_{i}", use_container_width=True):
                        st.session_state["selected_request_id"] = m.get("request_id")
                        st.session_state["show_details"] = True
            else:
                st.markdown(str(m.get("content") or ""))


def _tables_by_id(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for t in result.get("tables") or []:
        if isinstance(t, dict) and t.get("table_id"):
            out[str(t["table_id"])] = t
    return out


def _selected_assistant_message() -> dict[str, Any] | None:
    req = st.session_state.get("selected_request_id")
    if not req:
        return None
    for m in reversed(st.session_state.get("messages") or []):
        if m.get("role") == "assistant" and m.get("request_id") == req:
            return m
    return None


def _render_details_panel() -> None:
    if not st.session_state.get("show_details"):
        st.info("Click Details on an assistant message to view results and debug info.")
        return

    m = _selected_assistant_message()
    if not m:
        st.warning("No assistant message selected.")
        return

    col_title, col_close = st.columns([0.7, 0.3], gap="small")
    with col_title:
        st.subheader("Details")
        st.caption(f"request_id: `{m.get('request_id')}`")
    with col_close:
        if st.button("Close", use_container_width=True):
            st.session_state["show_details"] = False
            st.session_state["selected_request_id"] = None
            st.rerun()

    result = m.get("result") if isinstance(m.get("result"), dict) else None
    debug_events = m.get("debug_events") if isinstance(m.get("debug_events"), list) else []

    tabs = st.tabs(["Results", "Debug", "Raw JSON"])

    with tabs[0]:
        if not isinstance(result, dict):
            st.info("No result payload captured for this message.")
        else:
            _render_result_payload(result)

    with tabs[1]:
        debug_tabs = st.tabs(["Tool Traces", "Timeline"])

        with debug_tabs[0]:
            if not isinstance(result, dict):
                st.info("No result payload captured for this message.")
            else:
                traces = result.get("tool_traces") or []
                if not traces:
                    st.info("No tool traces captured.")
                for t in traces:
                    if not isinstance(t, dict):
                        continue
                    tool_name = str(t.get("tool_name") or "tool")
                    elapsed_ms = t.get("elapsed_ms")
                    cache_hit = t.get("cache_hit")
                    cache_key = t.get("cache_key")
                    ok = None
                    resp = t.get("response") if isinstance(t.get("response"), dict) else None
                    if isinstance(resp, dict):
                        ok = resp.get("ok")

                    suffix = []
                    if isinstance(ok, bool):
                        suffix.append(f"ok={ok}")
                    if isinstance(elapsed_ms, int):
                        suffix.append(f"{elapsed_ms}ms")
                    if isinstance(cache_hit, bool):
                        suffix.append(f"cache_hit={cache_hit}")
                    if isinstance(cache_key, str) and cache_key:
                        suffix.append(f"cache_key={cache_key}")

                    title = tool_name if not suffix else f"{tool_name} ({', '.join(suffix)})"
                    with st.expander(title, expanded=False):
                        cols = st.columns(2)
                        with cols[0]:
                            st.caption("Request")
                            st.json(t.get("request") or {}, expanded=False)
                        with cols[1]:
                            st.caption("Response")
                            st.json(t.get("response") or {}, expanded=False)

                        citations = t.get("citations") or []
                        if citations:
                            st.caption("Citations")
                            st.json(citations, expanded=False)

                        err = t.get("error")
                        if err:
                            st.caption("Error")
                            st.json(err, expanded=False)

        with debug_tabs[1]:
            if not debug_events:
                st.info("No debug events captured.")
            else:
                for e in debug_events[-200:]:
                    ev = str(e.get("event") or "")
                    ts = str(e.get("ts") or "")
                    with st.expander(f"{ts}  {ev}", expanded=False):
                        st.json(e.get("data") or {}, expanded=False)

    with tabs[2]:
        st.json({"message": m, "result": result, "debug_events": debug_events}, expanded=False)


def main() -> None:
    st.set_page_config(page_title="Cricket Companion", layout="wide")
    _init_state()

    st.title("Cricket Companion")

    st.markdown(
        """
        <style>
        /* Keep chat input near the bottom while scrolling. */
        div[data-testid="stChatInput"] {
          position: sticky;
          bottom: 0;
          background: var(--background-color);
          padding-top: 0.5rem;
          padding-bottom: 0.5rem;
          z-index: 2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Connection")
        st.session_state["api_base_url"] = st.text_input(
            "API base URL",
            value=st.session_state["api_base_url"],
        )
        st.session_state["user_id"] = st.text_input("user_id", value=st.session_state["user_id"])
        st.session_state["session_id"] = st.text_input("session_id", value=st.session_state["session_id"])
        st.session_state["debug"] = st.toggle("debug", value=bool(st.session_state["debug"]))

        if st.button("Reset UI session"):
            st.session_state["session_id"] = f"ui-session-{uuid4()}"
            st.session_state["messages"] = []
            st.session_state["selected_request_id"] = None
            st.session_state["show_details"] = False
            st.rerun()

        st.caption("Tip: you can send `/pref ...` and `/mem ...` commands in chat.")

    tabs = st.tabs(["Chat", "Simulator"])

    with tabs[0]:
        col_chat, col_details = st.columns([3, 1], gap="large")
        with col_chat:
            st.subheader("Chat")
            _render_chat()
            user_text = st.chat_input("Ask a cricket question...")
        with col_details:
            _render_details_panel()

        if user_text:
            request_id = f"ui-req-{uuid4()}"
            msg_id = f"ui-msg-{uuid4()}"
            _append_message("user", user_text, request_id=request_id)
            with col_chat:
                with st.chat_message("assistant"):
                    assistant_placeholder = st.empty()
            assembled, result_payload, debug_events = _stream_turn(
                user_text=user_text,
                request_id=request_id,
                msg_id=msg_id,
                placeholder=assistant_placeholder,
            )
            _append_message(
                "assistant",
                assembled,
                request_id=request_id,
                result=result_payload,
                debug_events=debug_events if bool(st.session_state.get("debug")) else None,
            )
            st.rerun()

    with tabs[1]:
        st.subheader("Strategy Simulator")
        st.caption("Fill a scenario and run the simulator. This submits a structured SimulationRequest to the agent.")

        st.session_state.setdefault("sim_last_result", None)
        st.session_state.setdefault("sim_last_text", None)
        st.session_state.setdefault("sim_last_request_id", None)

        with st.form("sim_form", clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                fmt = st.selectbox("Format", ["IPL", "T20", "ODI"], index=0)
                mode = st.selectbox("Mode", ["chase", "set_target"], index=0)
                model = st.selectbox("Model", ["baseline", "historical_blend"], index=1)
                n_sims = st.slider("Simulations", min_value=100, max_value=20000, value=5000, step=100)
                seed_txt = st.text_input("Seed (optional)", value="")
            with c2:
                max_overs_default = 20 if fmt in {"IPL", "T20"} else 50
                max_overs = st.number_input("Max overs", min_value=1, max_value=200, value=int(max_overs_default))
                runs = st.number_input("Current runs", min_value=0, max_value=1000, value=0)
                wkts = st.number_input("Wickets lost", min_value=0, max_value=10, value=3)
                overs_txt = st.text_input("Overs bowled (e.g., 10.2)", value="10.0")
                target_runs = None
                if mode == "chase":
                    target_runs = st.number_input("Target runs", min_value=1, max_value=1000, value=168)

            with st.expander("Conditions (optional)", expanded=False):
                pitch = st.selectbox("Pitch", ["unknown", "flat", "balanced", "slow", "seam", "spin"], index=0)
                dew = st.selectbox("Dew", ["unknown", "none", "some", "heavy"], index=0)
                boundary = st.selectbox("Boundary size", ["unknown", "small", "medium", "large"], index=0)

            also_chat = st.checkbox("Also add to chat history", value=True)
            submitted = st.form_submit_button("Run simulator")

        if submitted:
            balls = _overs_to_balls(str(overs_txt))
            if balls is None:
                st.error("Invalid overs value. Use format like 10 or 10.2 (balls part 0-5).")
            else:
                sim_payload: dict[str, Any] = {
                    "format": fmt,
                    "mode": mode,
                    "match_state": {
                        "innings": 2 if mode == "chase" else 1,
                        "score": {"runs": int(runs), "wkts": int(wkts), "balls": int(balls)},
                        "limits": {"max_overs": int(max_overs)},
                        "phase": "unknown",
                    },
                    "simulation": {
                        "n_sims": int(n_sims),
                        "seed": int(seed_txt) if str(seed_txt).strip().isdigit() else None,
                        "model": model,
                        "return_distributions": True,
                    },
                }
                if mode == "chase":
                    sim_payload["match_state"]["chase"] = {"target_runs": int(target_runs), "revised": False}  # type: ignore[index]

                if pitch != "unknown" or dew != "unknown" or boundary != "unknown":
                    sim_payload["conditions"] = {"pitch": pitch, "dew": dew, "boundary_size": boundary}

                # Validate locally if possible (helps catch obvious issues before hitting the API).
                try:
                    from cricket_companion.sim_schemas import SimulationRequest

                    SimulationRequest.model_validate(sim_payload)
                except Exception as exc:
                    st.warning(f"Local validation warning (still sending to API): {exc}")

                request_id = f"ui-sim-{uuid4()}"
                msg_id = f"ui-msg-{uuid4()}"
                user_text = json.dumps(sim_payload, ensure_ascii=False)

                if also_chat:
                    _append_message("user", f"sim: {fmt} {mode} (form)\n\n```json\n{user_text}\n```", request_id=request_id)

                with st.chat_message("assistant"):
                    placeholder = st.empty()

                assembled, result_payload, debug_events = _stream_turn(
                    user_text=user_text,
                    request_id=request_id,
                    msg_id=msg_id,
                    placeholder=placeholder,
                    metadata={"force_route": "sim", "ui_mode": "simulator"},
                )

                st.session_state["sim_last_text"] = assembled
                st.session_state["sim_last_result"] = result_payload
                st.session_state["sim_last_request_id"] = request_id

                if also_chat:
                    _append_message(
                        "assistant",
                        assembled,
                        request_id=request_id,
                        result=result_payload,
                        debug_events=debug_events if bool(st.session_state.get("debug")) else None,
                    )
                st.rerun()

        st.divider()
        st.subheader("Last simulator run")
        last_text = st.session_state.get("sim_last_text")
        last_result = st.session_state.get("sim_last_result")
        if not last_text:
            st.info("Run a scenario to see results here.")
        else:
            st.markdown(last_text)
            if isinstance(last_result, dict):
                st.divider()
                _render_result_payload(last_result)


if __name__ == "__main__":
    main()
