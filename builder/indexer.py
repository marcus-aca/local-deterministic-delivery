"""Repository indexer: scan files, chunk content, embed, and persist to Postgres."""

from __future__ import annotations

import argparse
import hashlib
import os
import posixpath
from datetime import datetime
from pathlib import Path

import db
from embeddings import build_embedding_backend

IGNORE_DIRS = {".git", "node_modules", "dist", "build", "__pycache__", ".venv", "venv"}
IGNORE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".class",
    ".jar",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".lock",
}


def is_binary_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            sample = handle.read(8192)
    except OSError:
        return True

    if b"\x00" in sample:
        return True

    return False


def iter_repo_files(repo_path: Path):
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]

        for filename in files:
            if filename.startswith("."):
                continue

            absolute = Path(root) / filename
            relative = absolute.relative_to(repo_path)

            if any(part in IGNORE_DIRS for part in relative.parts):
                continue
            if absolute.suffix.lower() in IGNORE_SUFFIXES:
                continue
            if is_binary_file(absolute):
                continue

            yield absolute, relative.as_posix()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[tuple[int, str]]:
    if not text:
        return []

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[tuple[int, str]] = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append((index, chunk))

        if end >= len(text):
            break

        start = end - overlap
        index += 1

    return chunks


def summarize_text(text: str, max_len: int = 220) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "Empty file"

    imports = [line for line in lines if line.startswith(("import ", "from ", "require(", "export "))][:2]
    signature = [line for line in lines if line.startswith(("def ", "class ", "function ", "const ", "async "))][:2]

    summary_parts = [lines[0]] + imports + signature
    summary = " | ".join(dict.fromkeys(summary_parts))
    return summary[:max_len]


def _default_chunk_size() -> int:
    return int(os.getenv("CHUNK_SIZE", "1200"))


def _default_chunk_overlap() -> int:
    return int(os.getenv("CHUNK_OVERLAP", "200"))


def _normalize_relative_path(file_path: str) -> str | None:
    normalized = posixpath.normpath(file_path.strip())
    if normalized in {"", "."}:
        return None
    if normalized.startswith("/") or normalized == ".." or normalized.startswith("../"):
        return None
    return normalized


def _is_indexable_path(abs_path: Path, rel_path: str) -> bool:
    rel = Path(rel_path)
    if any(part.startswith(".") for part in rel.parts):
        return False
    if any(part in IGNORE_DIRS for part in rel.parts):
        return False
    if abs_path.suffix.lower() in IGNORE_SUFFIXES:
        return False
    if is_binary_file(abs_path):
        return False
    return True


def _upsert_single_file(
    repo_root: Path,
    rel_path: str,
    chunk_size: int,
    overlap: int,
    backend,
) -> str:
    abs_path = repo_root / rel_path
    repo_root_str = str(repo_root)

    if not abs_path.exists() or not abs_path.is_file():
        deleted = db.delete_indexed_files(repo_root_str, [rel_path])
        return "deleted" if deleted else "skipped"

    if not _is_indexable_path(abs_path, rel_path):
        deleted = db.delete_indexed_files(repo_root_str, [rel_path])
        return "deleted" if deleted else "skipped"

    try:
        text = abs_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = abs_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return "skipped"

    file_hash = content_hash(text)
    existing_hash = db.get_existing_file_hash(repo_root_str, rel_path)
    if existing_hash == file_hash:
        return "skipped"

    summary = summarize_text(text)
    last_modified = datetime.fromtimestamp(abs_path.stat().st_mtime)
    db.upsert_file_metadata(
        repo_path=repo_root_str,
        file_path=rel_path,
        summary=summary,
        content_hash=file_hash,
        last_modified=last_modified,
    )

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    chunk_payload = [content for _, content in chunks]
    embeddings = backend.encode(chunk_payload) if chunk_payload else []

    db.replace_code_chunks(
        repo_path=repo_root_str,
        file_path=rel_path,
        chunks_with_embeddings=[
            (chunk_index, content, embeddings[i])
            for i, (chunk_index, content) in enumerate(chunks)
        ],
    )
    return "changed"


def _cleanup_deleted_files(repo_root: Path, existing_rel_paths: set[str]) -> int:
    indexed = set(db.list_indexed_files(str(repo_root)))
    stale = sorted(indexed - existing_rel_paths)
    if not stale:
        return 0
    return db.delete_indexed_files(str(repo_root), stale)


def index_files(
    repo_path: str,
    file_paths: list[str],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> dict[str, int | str]:
    repo_root = Path(repo_path).resolve()
    effective_chunk_size = chunk_size if chunk_size is not None else _default_chunk_size()
    effective_overlap = overlap if overlap is not None else _default_chunk_overlap()
    backend = build_embedding_backend()

    db.init_schema()

    unique_paths: list[str] = []
    seen: set[str] = set()
    for file_path in file_paths:
        normalized = _normalize_relative_path(file_path)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_paths.append(normalized)

    scanned = len(unique_paths)
    changed = 0
    skipped = 0
    deleted = 0

    for rel_path in unique_paths:
        status = _upsert_single_file(
            repo_root=repo_root,
            rel_path=rel_path,
            chunk_size=effective_chunk_size,
            overlap=effective_overlap,
            backend=backend,
        )
        if status == "changed":
            changed += 1
        elif status == "deleted":
            deleted += 1
        else:
            skipped += 1

    return {
        "repo": str(repo_root),
        "scanned": scanned,
        "changed": changed,
        "skipped": skipped,
        "deleted": deleted,
        "embedding_mode": "fallback" if backend.using_fallback else "sentence-transformers",
    }


def index_repo(repo_path: str, chunk_size: int, overlap: int) -> dict[str, int | str]:
    repo_root = Path(repo_path).resolve()
    backend = build_embedding_backend()

    db.init_schema()

    scanned = 0
    changed = 0
    skipped = 0

    existing_rel_paths: set[str] = set()

    for _, rel_path in iter_repo_files(repo_root):
        scanned += 1
        existing_rel_paths.add(rel_path)
        status = _upsert_single_file(
            repo_root=repo_root,
            rel_path=rel_path,
            chunk_size=chunk_size,
            overlap=overlap,
            backend=backend,
        )
        if status == "changed":
            changed += 1
        else:
            skipped += 1

    deleted = _cleanup_deleted_files(repo_root, existing_rel_paths)

    return {
        "repo": str(repo_root),
        "scanned": scanned,
        "changed": changed,
        "skipped": skipped,
        "deleted": deleted,
        "embedding_mode": "fallback" if backend.using_fallback else "sentence-transformers",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index repository files into pgvector")
    parser.add_argument("--repo", required=True, help="Repository path to index")
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "1200")))
    parser.add_argument("--chunk-overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "200")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = index_repo(args.repo, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
    print(result)


if __name__ == "__main__":
    main()
