from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+\n", "\n", re.sub(r"[ \t]+", " ", text)).strip()


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


def iter_markdown_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*.md")):
        # Skip README-like root docs if needed; for now ingest everything in docs/knowledge/.
        if p.name.lower().endswith(".md"):
            yield p


def chunk_markdown(
    *,
    source_path: Path,
    source_id: str,
    target_chars: int,
    max_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    """
    Chunk markdown by `##` sections (then size-split if needed).
    Produces retrieval-friendly chunks with stable ids.
    """
    text = source_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    title = ""
    current_h2 = ""
    current_block: list[str] = []
    blocks: list[tuple[str, list[str]]] = []

    def _flush() -> None:
        nonlocal current_block, current_h2
        if current_h2 and current_block:
            blocks.append((current_h2, current_block))
        current_block = []

    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            continue
        if line.startswith("## "):
            _flush()
            current_h2 = line[3:].strip()
            continue
        current_block.append(line)
    _flush()

    chunks: list[Chunk] = []
    chunk_index = 0

    for h2, b in blocks:
        header = f"# {title}\n\n## {h2}\n" if title else f"## {h2}\n"
        body = _norm_ws("\n".join(b))
        section_text = _norm_ws(header + "\n" + body) if body else _norm_ws(header)

        # Split long sections by paragraph boundaries.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]
        buf = ""
        prev_tail = ""
        for para in paragraphs:
            candidate = (buf + "\n\n" + para).strip() if buf else para
            if len(candidate) <= max_chars and (len(candidate) <= target_chars or not buf):
                buf = candidate
                continue

            # Emit current buffer.
            if buf:
                chunk_text = (prev_tail + buf).strip() if prev_tail else buf.strip()
                cid = f"{source_id}#{_slug(h2)}#{chunk_index}"
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        source_id=source_id,
                        text=chunk_text,
                        metadata={
                            "title": title,
                            "heading": h2,
                            "heading_path": [h2],
                            "chunk_index": chunk_index,
                            "char_len": len(chunk_text),
                        },
                    )
                )
                chunk_index += 1
                prev_tail = chunk_text[-overlap_chars:] if overlap_chars > 0 else ""

            buf = para

        if buf:
            chunk_text = (prev_tail + buf).strip() if prev_tail else buf.strip()
            cid = f"{source_id}#{_slug(h2)}#{chunk_index}"
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    source_id=source_id,
                    text=chunk_text,
                    metadata={
                        "title": title,
                        "heading": h2,
                        "heading_path": [h2],
                        "chunk_index": chunk_index,
                        "char_len": len(chunk_text),
                    },
                )
            )
            chunk_index += 1

    return chunks


def build_knowledge_index(
    *,
    docs_root: Path,
    out_dir: Path,
    model_name: str,
    target_chars: int = 1200,
    max_chars: int = 2200,
    overlap_chars: int = 150,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[Chunk] = []
    files = list(iter_markdown_files(docs_root))
    for p in files:
        source_id = p.as_posix()
        chunks.extend(
            chunk_markdown(
                source_path=p,
                source_id=source_id,
                target_chars=target_chars,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
        )

    if not chunks:
        raise RuntimeError(f"No markdown chunks produced from: {docs_root}")

    model = SentenceTransformer(model_name)
    texts = [c.text for c in chunks]
    emb = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32)

    dim = int(emb.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    index_path = out_dir / "knowledge.faiss"
    faiss.write_index(index, str(index_path))

    chunks_path = out_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
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
        "docs_root": str(docs_root.as_posix()),
        "model_name": model_name,
        "dimension": dim,
        "files": len(files),
        "chunks": len(chunks),
        "chunking": {
            "target_chars": target_chars,
            "max_chars": max_chars,
            "overlap_chars": overlap_chars,
        },
        "artifacts": {
            "faiss_index": str(index_path.as_posix()),
            "chunks_jsonl": str(chunks_path.as_posix()),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build FAISS retrieval index for docs/knowledge.")
    parser.add_argument("--docs-root", type=str, default="docs/knowledge", help="Root folder of knowledge markdown.")
    parser.add_argument("--out-dir", type=str, default="data/retrieval", help="Output directory for FAISS + chunks.")
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument("--target-chars", type=int, default=1200)
    parser.add_argument("--max-chars", type=int, default=2200)
    parser.add_argument("--overlap-chars", type=int, default=150)
    args = parser.parse_args(argv)

    docs_root = Path(args.docs_root)
    out_dir = Path(args.out_dir)

    manifest = build_knowledge_index(
        docs_root=docs_root,
        out_dir=out_dir,
        model_name=args.model,
        target_chars=int(args.target_chars),
        max_chars=int(args.max_chars),
        overlap_chars=int(args.overlap_chars),
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

