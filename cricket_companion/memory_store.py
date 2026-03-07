from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class SessionContext:
    summary: str | None
    messages: list[dict[str, Any]]


class MemoryStore:
    """
    DuckDB-backed short-term session memory.

    Phase 2.6.1 responsibilities:
    - persist per-session messages
    - persist and update a rolling session summary
    """

    def __init__(self, *, db_path: Path):
        self.db_path = db_path

    def connect(self) -> Any:
        import duckdb  # type: ignore

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(self.db_path))
        self._ensure_schema(con)
        return con

    def _ensure_schema(self, con: Any) -> None:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
              session_id TEXT PRIMARY KEY,
              user_id TEXT,
              created_at TIMESTAMP,
              updated_at TIMESTAMP,
              summary_text TEXT,
              summary_updated_at TIMESTAMP,
              summary_model TEXT
            );
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
              message_id TEXT PRIMARY KEY,
              session_id TEXT,
              role TEXT,
              content TEXT,
              created_at TIMESTAMP,
              metadata_json TEXT,
              is_summarized BOOLEAN DEFAULT FALSE
            );
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
              user_id TEXT PRIMARY KEY,
              created_at TIMESTAMP,
              updated_at TIMESTAMP
            );
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
              user_id TEXT,
              key TEXT,
              value_json TEXT,
              updated_at TIMESTAMP,
              PRIMARY KEY (user_id, key)
            );
            """
        )

        # Backfill schema for existing DBs created before `user_id` was added to chat_sessions.
        try:
            con.execute("ALTER TABLE chat_sessions ADD COLUMN user_id TEXT;")
        except Exception:
            pass

    def _has_chat_sessions_user_id(self, con: Any) -> bool:
        try:
            rows = con.execute("PRAGMA table_info('chat_sessions')").fetchall()
            return any(r[1] == "user_id" for r in rows)
        except Exception:
            return False

    def load_context(
        self,
        con: Any,
        *,
        session_id: str,
        max_messages: int,
        user_id: str | None = None,
    ) -> SessionContext:
        row = con.execute(
            "SELECT summary_text FROM chat_sessions WHERE session_id = ?",
            [session_id],
        ).fetchone()
        summary = row[0] if row else None

        rows = con.execute(
            """
            SELECT message_id, role, content, created_at, metadata_json
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [session_id, int(max_messages)],
        ).fetchall()

        messages: list[dict[str, Any]] = []
        for message_id, role, content, created_at, metadata_json in reversed(rows):
            metadata: dict[str, Any] = {}
            if metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                except Exception:
                    metadata = {}
            messages.append(
                {
                    "message_id": message_id,
                    "role": role,
                    "content": content,
                    "created_at": created_at,
                    "metadata": metadata,
                }
            )

        # Ensure the session row exists once a session is used.
        now = _utc_now()
        if self._has_chat_sessions_user_id(con):
            con.execute(
                """
                INSERT INTO chat_sessions (session_id, user_id, created_at, updated_at, summary_text, summary_updated_at, summary_model)
                VALUES (?, ?, ?, ?, NULL, NULL, NULL)
                ON CONFLICT (session_id) DO UPDATE SET updated_at = excluded.updated_at, user_id = COALESCE(excluded.user_id, chat_sessions.user_id)
                """,
                [session_id, user_id, now, now],
            )
        else:
            con.execute(
                """
                INSERT INTO chat_sessions (session_id, created_at, updated_at, summary_text, summary_updated_at, summary_model)
                VALUES (?, ?, ?, NULL, NULL, NULL)
                ON CONFLICT (session_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                [session_id, now, now],
            )

        return SessionContext(summary=summary, messages=messages)

    def append_messages(
        self,
        con: Any,
        *,
        session_id: str,
        messages: list[dict[str, Any]],
        user_id: str | None = None,
    ) -> None:
        now = _utc_now()
        if self._has_chat_sessions_user_id(con):
            con.execute(
                """
                INSERT INTO chat_sessions (session_id, user_id, created_at, updated_at, summary_text, summary_updated_at, summary_model)
                VALUES (?, ?, ?, ?, NULL, NULL, NULL)
                ON CONFLICT (session_id) DO UPDATE SET updated_at = excluded.updated_at, user_id = COALESCE(excluded.user_id, chat_sessions.user_id)
                """,
                [session_id, user_id, now, now],
            )
        else:
            con.execute(
                """
                INSERT INTO chat_sessions (session_id, created_at, updated_at, summary_text, summary_updated_at, summary_model)
                VALUES (?, ?, ?, NULL, NULL, NULL)
                ON CONFLICT (session_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                [session_id, now, now],
            )
        for m in messages:
            con.execute(
                """
                INSERT OR REPLACE INTO chat_messages
                  (message_id, session_id, role, content, created_at, metadata_json, is_summarized)
                VALUES (?, ?, ?, ?, ?, ?, COALESCE(?, FALSE))
                """,
                [
                    str(m.get("message_id") or ""),
                    session_id,
                    str(m.get("role") or ""),
                    str(m.get("content") or ""),
                    m.get("created_at") or now,
                    json.dumps(m.get("metadata") or {}, ensure_ascii=False),
                    m.get("is_summarized"),
                ],
            )

    def list_sessions(self, con: Any, *, user_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        limit_i = max(1, min(int(limit), 200))
        if user_id and self._has_chat_sessions_user_id(con):
            rows = con.execute(
                """
                SELECT session_id, user_id, created_at, updated_at, summary_updated_at,
                       CASE WHEN summary_text IS NULL THEN 0 ELSE LENGTH(summary_text) END AS summary_len
                FROM chat_sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                [user_id, limit_i],
            ).fetchall()
        else:
            rows = con.execute(
                """
                SELECT session_id, user_id, created_at, updated_at, summary_updated_at,
                       CASE WHEN summary_text IS NULL THEN 0 ELSE LENGTH(summary_text) END AS summary_len
                FROM chat_sessions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                [limit_i],
            ).fetchall()

        out: list[dict[str, Any]] = []
        for session_id, uid, created_at, updated_at, summary_updated_at, summary_len in rows:
            out.append(
                {
                    "session_id": session_id,
                    "user_id": uid,
                    "created_at": str(created_at) if created_at is not None else None,
                    "updated_at": str(updated_at) if updated_at is not None else None,
                    "summary_updated_at": str(summary_updated_at) if summary_updated_at is not None else None,
                    "summary_len": int(summary_len or 0),
                }
            )
        return out

    def get_session(self, con: Any, *, session_id: str, message_limit: int = 20) -> dict[str, Any] | None:
        row = con.execute(
            "SELECT session_id, user_id, created_at, updated_at, summary_text, summary_updated_at, summary_model FROM chat_sessions WHERE session_id = ?",
            [session_id],
        ).fetchone()
        if not row:
            return None

        msg_rows = con.execute(
            """
            SELECT message_id, role, content, created_at, is_summarized
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [session_id, max(1, min(int(message_limit), 200))],
        ).fetchall()
        messages = [
            {
                "message_id": r[0],
                "role": r[1],
                "content": r[2],
                "created_at": str(r[3]) if r[3] is not None else None,
                "is_summarized": bool(r[4]),
            }
            for r in reversed(msg_rows)
        ]

        return {
            "session_id": row[0],
            "user_id": row[1],
            "created_at": str(row[2]) if row[2] is not None else None,
            "updated_at": str(row[3]) if row[3] is not None else None,
            "summary_text": row[4],
            "summary_updated_at": str(row[5]) if row[5] is not None else None,
            "summary_model": row[6],
            "recent_messages": messages,
        }

    def clear_session_summary(self, con: Any, *, session_id: str) -> bool:
        before = con.execute(
            "SELECT COUNT(*) FROM chat_sessions WHERE session_id = ? AND summary_text IS NOT NULL",
            [session_id],
        ).fetchone()[0]
        now = _utc_now()
        con.execute(
            """
            UPDATE chat_sessions
            SET summary_text = NULL, summary_updated_at = NULL, summary_model = NULL, updated_at = ?
            WHERE session_id = ?
            """,
            [now, session_id],
        )
        con.execute(
            "UPDATE chat_messages SET is_summarized = FALSE WHERE session_id = ?",
            [session_id],
        )
        return int(before) > 0

    def delete_last_messages(self, con: Any, *, session_id: str, n: int) -> int:
        n_i = max(1, min(int(n), 200))
        rows = con.execute(
            """
            SELECT message_id
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [session_id, n_i],
        ).fetchall()
        ids = [r[0] for r in rows]
        if not ids:
            return 0
        con.execute(
            "DELETE FROM chat_messages WHERE session_id = ? AND message_id IN (SELECT * FROM UNNEST(?))",
            [session_id, ids],
        )
        return len(ids)

    def delete_session(self, con: Any, *, session_id: str) -> bool:
        before = con.execute(
            "SELECT COUNT(*) FROM chat_sessions WHERE session_id = ?",
            [session_id],
        ).fetchone()[0]
        con.execute("DELETE FROM chat_messages WHERE session_id = ?", [session_id])
        con.execute("DELETE FROM chat_sessions WHERE session_id = ?", [session_id])
        return int(before) > 0

    def purge_user(self, con: Any, *, user_id: str) -> dict[str, int]:
        """
        Deletes preferences and any sessions/messages associated with this user_id (if sessions are user-scoped).
        """
        deleted_sessions = 0
        deleted_messages = 0

        if self._has_chat_sessions_user_id(con):
            sess_rows = con.execute("SELECT session_id FROM chat_sessions WHERE user_id = ?", [user_id]).fetchall()
            session_ids = [r[0] for r in sess_rows]
            if session_ids:
                deleted_messages = con.execute(
                    "SELECT COUNT(*) FROM chat_messages WHERE session_id IN (SELECT * FROM UNNEST(?))",
                    [session_ids],
                ).fetchone()[0]
                con.execute(
                    "DELETE FROM chat_messages WHERE session_id IN (SELECT * FROM UNNEST(?))",
                    [session_ids],
                )
                con.execute(
                    "DELETE FROM chat_sessions WHERE session_id IN (SELECT * FROM UNNEST(?))",
                    [session_ids],
                )
                deleted_sessions = len(session_ids)

        deleted_prefs = con.execute("SELECT COUNT(*) FROM user_preferences WHERE user_id = ?", [user_id]).fetchone()[0]
        con.execute("DELETE FROM user_preferences WHERE user_id = ?", [user_id])
        con.execute("DELETE FROM user_profiles WHERE user_id = ?", [user_id])
        return {
            "deleted_sessions": int(deleted_sessions),
            "deleted_messages": int(deleted_messages or 0),
            "deleted_prefs": int(deleted_prefs or 0),
        }

    def export_session_artifacts(self, con: Any, *, session_id: str, out_dir: Path) -> dict[str, str]:
        """
        Export a session (messages + session row) to CSV and JSON for user visibility.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        session_df = con.execute("SELECT * FROM chat_sessions WHERE session_id = ?", [session_id]).df()
        messages_df = con.execute(
            "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at",
            [session_id],
        ).df()

        session_csv = out_dir / f"{session_id}_chat_sessions.csv"
        messages_csv = out_dir / f"{session_id}_chat_messages.csv"
        session_json = out_dir / f"{session_id}.json"

        session_df.to_csv(session_csv, index=False)
        messages_df.to_csv(messages_csv, index=False)

        payload = {
            "session": session_df.to_dict(orient="records"),
            "messages": messages_df.to_dict(orient="records"),
        }
        session_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"session_csv": str(session_csv), "messages_csv": str(messages_csv), "json": str(session_json)}

    def export_user_artifacts(self, con: Any, *, user_id: str, out_dir: Path) -> dict[str, str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        profiles_df = con.execute("SELECT * FROM user_profiles WHERE user_id = ?", [user_id]).df()
        prefs_df = con.execute("SELECT * FROM user_preferences WHERE user_id = ? ORDER BY key", [user_id]).df()
        profiles_csv = out_dir / f"{user_id}_user_profiles.csv"
        prefs_csv = out_dir / f"{user_id}_user_preferences.csv"
        profiles_json = out_dir / f"{user_id}_prefs.json"
        profiles_df.to_csv(profiles_csv, index=False)
        prefs_df.to_csv(prefs_csv, index=False)
        payload = {"user_profiles": profiles_df.to_dict(orient="records"), "user_preferences": prefs_df.to_dict(orient="records")}
        profiles_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"profiles_csv": str(profiles_csv), "prefs_csv": str(prefs_csv), "json": str(profiles_json)}

    def load_preferences(self, con: Any, *, user_id: str) -> dict[str, Any]:
        rows = con.execute(
            """
            SELECT key, value_json
            FROM user_preferences
            WHERE user_id = ?
            """,
            [user_id],
        ).fetchall()

        prefs: dict[str, Any] = {}
        for key, value_json in rows:
            try:
                prefs[str(key)] = json.loads(value_json) if value_json else None
            except Exception:
                prefs[str(key)] = value_json
        return prefs

    def set_preference(self, con: Any, *, user_id: str, key: str, value: Any) -> None:
        now = _utc_now()
        con.execute(
            """
            INSERT INTO user_profiles (user_id, created_at, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT (user_id) DO UPDATE SET updated_at = excluded.updated_at
            """,
            [user_id, now, now],
        )
        con.execute(
            """
            INSERT INTO user_preferences (user_id, key, value_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (user_id, key) DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at
            """,
            [user_id, key, json.dumps(value, ensure_ascii=False), now],
        )

    def delete_preference(self, con: Any, *, user_id: str, key: str) -> bool:
        before = con.execute(
            "SELECT COUNT(*) FROM user_preferences WHERE user_id = ? AND key = ?",
            [user_id, key],
        ).fetchone()[0]
        con.execute(
            "DELETE FROM user_preferences WHERE user_id = ? AND key = ?",
            [user_id, key],
        )
        return int(before) > 0

    def summarize_if_needed(
        self,
        con: Any,
        *,
        session_id: str,
        keep_last_n: int,
        summarize_chunk_n: int,
        summary_max_chars: int,
        summary_model: str,
        openai_api_key: str | None,
    ) -> str | None:
        """
        Summarize older unsummarized messages into `chat_sessions.summary_text` using an LLM.

        Returns the updated summary (or None if no change).
        """
        if not openai_api_key:
            return None

        total_unsummarized = con.execute(
            """
            SELECT COUNT(*) FROM chat_messages
            WHERE session_id = ? AND is_summarized = FALSE
            """,
            [session_id],
        ).fetchone()[0]

        # Not enough material to summarize yet.
        overflow = int(total_unsummarized) - int(keep_last_n)
        if overflow <= 0:
            return None

        take_n = min(int(summarize_chunk_n), overflow)
        rows = con.execute(
            """
            SELECT message_id, role, content, created_at
            FROM chat_messages
            WHERE session_id = ? AND is_summarized = FALSE
            ORDER BY created_at ASC
            LIMIT ?
            """,
            [session_id, take_n],
        ).fetchall()

        if not rows:
            return None

        existing_summary_row = con.execute(
            "SELECT summary_text FROM chat_sessions WHERE session_id = ?",
            [session_id],
        ).fetchone()
        existing_summary = existing_summary_row[0] if existing_summary_row else None

        updated = _llm_update_summary(
            existing_summary=existing_summary,
            new_messages=[
                {"role": r[1], "content": r[2], "created_at": str(r[3])} for r in rows
            ],
            summary_model=summary_model,
            summary_max_chars=summary_max_chars,
            openai_api_key=openai_api_key,
        )

        now = _utc_now()
        con.execute(
            """
            UPDATE chat_sessions
            SET summary_text = ?, summary_updated_at = ?, summary_model = ?, updated_at = ?
            WHERE session_id = ?
            """,
            [updated, now, summary_model, now, session_id],
        )

        # Mark summarized messages.
        ids = [r[0] for r in rows]
        con.execute(
            """
            UPDATE chat_messages
            SET is_summarized = TRUE
            WHERE session_id = ? AND message_id IN (SELECT * FROM UNNEST(?))
            """,
            [session_id, ids],
        )
        return updated


def _llm_update_summary(
    *,
    existing_summary: str | None,
    new_messages: list[dict[str, Any]],
    summary_model: str,
    summary_max_chars: int,
    openai_api_key: str,
) -> str:
    """
    Uses OpenAI to update an existing session summary with newly summarized messages.

    Guardrails:
    - Only include info present in messages.
    - Keep concise, structured bullets.
    """
    try:
        from openai import OpenAI
    except Exception:
        # If the SDK isn't available, keep a minimal deterministic summary.
        existing = (existing_summary or "").strip()
        return existing[:summary_max_chars]

    client = OpenAI(api_key=openai_api_key)

    system = (
        "You are a session summarizer for a cricket assistant.\n"
        "Update the existing summary using the new messages.\n"
        "Rules:\n"
        "- Only summarize what is explicitly stated in the messages.\n"
        "- Do NOT invent facts or numbers.\n"
        "- Keep the summary concise and useful for future follow-ups.\n"
        "- Prefer short bullet points.\n"
        f"- Hard limit: {int(summary_max_chars)} characters.\n"
        "Return ONLY JSON: {\"summary\": \"...\"}.\n"
    )

    payload = {
        "existing_summary": existing_summary,
        "new_messages": new_messages,
        "desired_sections": ["User goals", "Locked scope/constraints", "Open questions", "Recent conclusions"],
    }

    resp = client.chat.completions.create(
        model=summary_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
    )

    content = resp.choices[0].message.content or "{}"
    try:
        obj = json.loads(content)
        summary = str(obj.get("summary") or "").strip()
    except Exception:
        summary = (existing_summary or "").strip()

    if len(summary) > summary_max_chars:
        summary = summary[: max(0, summary_max_chars - 3)].rstrip() + "..."
    return summary
