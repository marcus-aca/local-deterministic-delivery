"""Retrieval layer: rewrite query, vector search, dependency expansion, and structured context output."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import db
from agents.common import generate_text
from embeddings import build_embedding_backend

KEY_FILES = [
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "README.md",
    "main.py",
    "app.py",
    "index.js",
    "src/main.py",
    "src/index.js",
]

IMPORT_PATTERNS = [
    re.compile(r"^\s*from\s+([a-zA-Z0-9_\.]+)\s+import\s+", re.MULTILINE),
    re.compile(r"^\s*import\s+([a-zA-Z0-9_\.]+)", re.MULTILINE),
    re.compile(r"from\s+[\"'](\./[^\"']+|\.\./[^\"']+)[\"']"),
    re.compile(r"require\(\s*[\"'](\./[^\"']+|\.\./[^\"']+)[\"']\s*\)"),
]


def rewrite_query(query: str) -> str:
    system_prompt = (
        "You rewrite developer requests into concise code-search queries. "
        "Return only the rewritten query text."
    )
    user_prompt = f"Original request: {query}\nRewrite for vector code retrieval."
    rewritten = generate_text(system_prompt, user_prompt)
    return rewritten.strip() if rewritten else query


def _read_snippet(path: Path, max_chars: int = 600) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    return text[:max_chars]


def _resolve_dependency(repo_path: Path, source_file: str, dependency: str) -> str | None:
    source_dir = (repo_path / source_file).parent

    if dependency.startswith("./") or dependency.startswith("../"):
        candidates = [
            source_dir / dependency,
            source_dir / f"{dependency}.py",
            source_dir / f"{dependency}.js",
            source_dir / f"{dependency}.ts",
            source_dir / dependency / "__init__.py",
            source_dir / dependency / "index.js",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate.relative_to(repo_path).as_posix()
        return None

    module_path = dependency.replace(".", "/")
    candidates = [
        repo_path / f"{module_path}.py",
        repo_path / module_path / "__init__.py",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.relative_to(repo_path).as_posix()
    return None


def _extract_dependencies(repo_path: Path, file_path: str) -> list[str]:
    abs_path = repo_path / file_path
    snippet = _read_snippet(abs_path, max_chars=4000)
    dependencies: list[str] = []

    for pattern in IMPORT_PATTERNS:
        for match in pattern.findall(snippet):
            dep = _resolve_dependency(repo_path, file_path, match)
            if dep:
                dependencies.append(dep)

    return list(dict.fromkeys(dependencies))


def _top_level_map(repo_path: Path) -> str:
    entries = sorted(item.name for item in repo_path.iterdir() if not item.name.startswith("."))
    return "Top-level entries:\n" + "\n".join(entries[:100])


def _include_key_files(repo_path: Path, selected: list[str]) -> list[str]:
    for file_name in KEY_FILES:
        candidate = repo_path / file_name
        if candidate.exists() and candidate.is_file() and file_name not in selected:
            selected.append(file_name)
    return selected


def retrieve_context(repo_path: str, user_query: str, limit: int = 8, cap: int = 20) -> dict[str, Any]:
    repo_root = Path(repo_path).resolve()
    rewritten = rewrite_query(user_query)

    backend = build_embedding_backend()
    query_embedding = backend.encode([rewritten])[0]

    rows = db.search_code_chunks(str(repo_root), query_embedding, limit=limit)

    selected_files: list[str] = []
    snippets_by_file: dict[str, list[str]] = {}

    for row in rows:
        file_path = row["file_path"]
        if file_path not in selected_files:
            selected_files.append(file_path)
        snippets_by_file.setdefault(file_path, []).append(row["content"][:600])

    selected_files = _include_key_files(repo_root, selected_files)

    expanded = list(selected_files)
    for file_path in list(selected_files):
        for dependency in _extract_dependencies(repo_root, file_path):
            if dependency not in expanded:
                expanded.append(dependency)

    deduped = list(dict.fromkeys(expanded))[:cap]

    files_payload = []
    for file_path in deduped:
        snippets = snippets_by_file.get(file_path)
        if not snippets:
            snippets = [_read_snippet(repo_root / file_path)]
        files_payload.append(
            {
                "file_path": file_path,
                "snippets": [snippet for snippet in snippets if snippet],
            }
        )

    files_payload.insert(
        0,
        {
            "file_path": "__TOP_LEVEL_MAP__",
            "snippets": [_top_level_map(repo_root)],
        },
    )

    return {
        "original_query": user_query,
        "rewritten_query": rewritten,
        "embedding_mode": "fallback" if backend.using_fallback else "sentence-transformers",
        "files": files_payload,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve relevant repository context")
    parser.add_argument("--repo", required=True, help="Repository path")
    parser.add_argument("--query", required=True, help="User request")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--cap", type=int, default=20)
    parser.add_argument("--smoke", action="store_true", help="Run retrieval smoke mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = retrieve_context(args.repo, args.query, limit=args.limit, cap=args.cap)

    if args.smoke:
        print(
            {
                "rewritten_query": result["rewritten_query"],
                "file_count": len(result["files"]),
                "first_files": [entry["file_path"] for entry in result["files"][:5]],
            }
        )
        return

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
