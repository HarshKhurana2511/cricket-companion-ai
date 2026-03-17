from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from cricket_companion.config import get_settings


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _norm_ws(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s+\n", "\n\n", t)
    return t.strip()


def _slug(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9\s]+", "-", s)
    s = re.sub(r"\s+", "-", s).strip("-")
    return s or "section"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_id: str
    text: str
    metadata: dict[str, Any]


def _iter_cache_files(cache_root: Path) -> Iterable[Path]:
    if not cache_root.exists():
        return []
    # Only our web cache folder.
    for p in sorted(cache_root.glob("*.json")):
        yield p


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _looks_like_web_fetch(obj: dict[str, Any]) -> bool:
    data = obj.get("data")
    if not isinstance(data, dict):
        return False
    return isinstance(data.get("payload"), dict) and isinstance(data.get("url"), str)


def _looks_like_espn_ingest(obj: dict[str, Any]) -> bool:
    data = obj.get("data")
    if not isinstance(data, dict):
        return False
    return data.get("source") == "espncricinfo" and isinstance(data.get("url"), str)


def _extract_web_fetch_doc(obj: dict[str, Any]) -> tuple[str, str, dict[str, Any]] | None:
    data = obj.get("data")
    if not isinstance(data, dict):
        return None
    url = data.get("url")
    fetched_at = data.get("fetched_at")
    payload = data.get("payload")
    if not isinstance(url, str) or not isinstance(payload, dict):
        return None

    title = str(payload.get("title") or "").strip() or None
    published_at = payload.get("published_at")
    mode = payload.get("mode") or payload.get("extraction_mode")
    text = str(payload.get("text") or "").strip()
    if not text:
        return None

    header = []
    if title:
        header.append(title)
    header.append(url)
    if published_at:
        header.append(f"published_at: {published_at}")
    if fetched_at:
        header.append(f"fetched_at: {fetched_at}")
    if mode:
        header.append(f"mode: {mode}")
    doc_text = _norm_ws("\n".join(header) + "\n\n" + text)

    meta: dict[str, Any] = {
        "kind": "web_fetch",
        "url": url,
        "title": title,
        "published_at": str(published_at) if published_at else None,
        "fetched_at": str(fetched_at) if fetched_at else None,
        "mode": str(mode) if mode else None,
    }
    meta = {k: v for k, v in meta.items() if v is not None}
    return (url, doc_text, meta)


def _extract_espn_doc(obj: dict[str, Any]) -> tuple[str, str, dict[str, Any]] | None:
    data = obj.get("data")
    if not isinstance(data, dict):
        return None
    url = data.get("url")
    fetched_at = data.get("fetched_at")
    if not isinstance(url, str):
        return None

    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    title = str(metadata.get("title") or "").strip() or None
    fallback_text = str(metadata.get("fallback_text") or "").strip()
    scorecard = data.get("scorecard") if isinstance(data.get("scorecard"), dict) else None

    # Prefer structured scorecard summary if present; always include fallback_text if present.
    lines: list[str] = []
    if title:
        lines.append(title)
    lines.append(url)
    if fetched_at:
        lines.append(f"fetched_at: {fetched_at}")
    lines.append("source: espncricinfo")

    if scorecard and isinstance(scorecard.get("innings"), list):
        lines.append("")
        lines.append("Scorecard (structured, best-effort):")
        for inn in scorecard.get("innings")[:4]:
            if not isinstance(inn, dict):
                continue
            team = inn.get("team")
            lines.append(f"- innings_team: {team}")
            batting = inn.get("batting")
            if isinstance(batting, list) and batting:
                for r in batting[:8]:
                    if not isinstance(r, dict):
                        continue
                    player = r.get("player") or r.get("bowler") or ""
                    runs = r.get("runs")
                    balls = r.get("balls")
                    if runs is not None and balls is not None:
                        lines.append(f"  - {player}: {runs} ({balls})")
                    elif runs is not None:
                        lines.append(f"  - {player}: {runs}")

    if fallback_text:
        lines.append("")
        lines.append("Extracted page text (fallback):")
        lines.append(fallback_text)

    doc_text = _norm_ws("\n".join(lines))
    if not doc_text:
        return None

    meta: dict[str, Any] = {
        "kind": "espn_ingest",
        "url": url,
        "title": title,
        "fetched_at": str(fetched_at) if fetched_at else None,
        "fallback": metadata.get("fallback"),
    }
    meta = {k: v for k, v in meta.items() if v is not None}
    return (url, doc_text, meta)


def chunk_text(
    *,
    source_id: str,
    title: str,
    text: str,
    target_chars: int,
    max_chars: int,
    overlap_chars: int,
    base_metadata: dict[str, Any],
) -> list[Chunk]:
    """
    Chunk plain text by paragraph boundaries.
    """
    normalized = _norm_ws(text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]

    chunks: list[Chunk] = []
    chunk_index = 0
    buf = ""
    prev_tail = ""

    for para in paragraphs:
        candidate = (buf + "\n\n" + para).strip() if buf else para
        if len(candidate) <= max_chars and (len(candidate) <= target_chars or not buf):
            buf = candidate
            continue

        if buf:
            chunk_text = (prev_tail + buf).strip() if prev_tail else buf.strip()
            cid = f"{source_id}#{_slug(title)}#{chunk_index}"
            meta = dict(base_metadata)
            meta.update({"chunk_index": chunk_index, "char_len": len(chunk_text)})
            chunks.append(Chunk(chunk_id=cid, source_id=source_id, text=chunk_text, metadata=meta))
            chunk_index += 1
            prev_tail = chunk_text[-overlap_chars:] if overlap_chars > 0 else ""
        buf = para

    if buf:
        chunk_text = (prev_tail + buf).strip() if prev_tail else buf.strip()
        cid = f"{source_id}#{_slug(title)}#{chunk_index}"
        meta = dict(base_metadata)
        meta.update({"chunk_index": chunk_index, "char_len": len(chunk_text)})
        chunks.append(Chunk(chunk_id=cid, source_id=source_id, text=chunk_text, metadata=meta))

    return chunks


def build_web_index(
    *,
    cache_dir: Path,
    out_dir: Path,
    target_chars: int = 1200,
    max_chars: int = 2200,
    overlap_chars: int = 150,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_root = cache_dir / "web"
    files = list(_iter_cache_files(cache_root))

    chunks: list[Chunk] = []
    for p in files:
        obj = _load_json(p)
        if not obj:
            continue

        doc: tuple[str, str, dict[str, Any]] | None = None
        if _looks_like_web_fetch(obj):
            doc = _extract_web_fetch_doc(obj)
        elif _looks_like_espn_ingest(obj):
            doc = _extract_espn_doc(obj)
        else:
            continue

        if not doc:
            continue
        source_id, doc_text, base_meta = doc
        title = str(base_meta.get("title") or base_meta.get("kind") or "web").strip() or "web"
        base_meta = dict(base_meta)
        base_meta["cache_file"] = p.name

        chunks.extend(
            chunk_text(
                source_id=source_id,
                title=title,
                text=doc_text,
                target_chars=target_chars,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                base_metadata=base_meta,
            )
        )

    out_path = out_dir / "web_chunks.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            obj = {
                "faiss_id": i,
                "chunk_id": c.chunk_id,
                "source_id": c.source_id,
                "text": c.text,
                "metadata": c.metadata,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    manifest = {
        "created_at": _utc_now_iso(),
        "cache_root": str(cache_root.as_posix()),
        "files_scanned": len(files),
        "chunks": len(chunks),
        "chunking": {
            "target_chars": target_chars,
            "max_chars": max_chars,
            "overlap_chars": overlap_chars,
        },
        "artifacts": {"web_chunks_jsonl": str(out_path.as_posix())},
    }
    (out_dir / "web_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build retrieval chunks from cached web/ESPN outputs.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache dir (defaults to CC_CACHE_DIR).")
    parser.add_argument("--out-dir", type=str, default="data/retrieval", help="Output directory for web_chunks.jsonl.")
    parser.add_argument("--target-chars", type=int, default=1200)
    parser.add_argument("--max-chars", type=int, default=2200)
    parser.add_argument("--overlap-chars", type=int, default=150)
    args = parser.parse_args(argv)

    settings = get_settings(load_env_file=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else settings.cache_dir
    out_dir = Path(args.out_dir)

    manifest = build_web_index(
        cache_dir=cache_dir,
        out_dir=out_dir,
        target_chars=int(args.target_chars),
        max_chars=int(args.max_chars),
        overlap_chars=int(args.overlap_chars),
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

