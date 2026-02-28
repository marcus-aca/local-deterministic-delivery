"""Coder agent produces full-file edits using strict FILE: <path> protocol."""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from typing import Any, Callable

from agents.common import CodexInvocationError, generate_text, save_prompt_debug_snapshot

FILE_HEADER_PATTERN = re.compile(r"^FILE:\s*(.+)$", re.MULTILINE)


@dataclass
class FileChange:
    path: str
    content: str


def parse_file_blocks(raw_text: str) -> list[FileChange]:
    matches = list(FILE_HEADER_PATTERN.finditer(raw_text))
    if not matches:
        raise ValueError("Malformed coder output: expected at least one 'FILE: <path>' header.")

    changes: list[FileChange] = []
    for index, match in enumerate(matches):
        file_path = match.group(1).strip()
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(raw_text)
        content = raw_text[match.end():next_start]
        content = content.lstrip("\n")

        if not file_path:
            raise ValueError("Malformed coder output: empty FILE path.")
        if content == "":
            raise ValueError(f"Malformed coder output: file block for {file_path} is empty.")

        changes.append(FileChange(path=file_path, content=content.rstrip("\n") + "\n"))

    return changes


def validate_file_changes(changes: list[FileChange]) -> None:
    if not changes:
        raise ValueError("No file changes produced.")

    for change in changes:
        normalized = posixpath.normpath(change.path)
        if normalized.startswith("../") or normalized == "..":
            raise ValueError(f"Invalid file path in agent output: {change.path}")
        if normalized.startswith("/"):
            raise ValueError(f"Absolute file path is not allowed: {change.path}")
        if not change.content.strip():
            raise ValueError(f"Generated file content is empty for {change.path}")


def _context_blob(retrieved_context: dict[str, Any] | None) -> str:
    if not retrieved_context:
        return "No retrieval context."

    lines: list[str] = []
    for item in retrieved_context.get("files", [])[:12]:
        path = item.get("file_path", "unknown")
        snippets = item.get("snippets", [])
        snippet_text = snippets[0][:500] if snippets else ""
        lines.append(f"- {path}\n{snippet_text}")
    return "\n".join(lines)


def generate_code_changes(
    user_request: str,
    step_description: str,
    retrieved_context: dict[str, Any] | None = None,
    cwd: str | None = None,
    heartbeat: Callable[[int], None] | None = None,
) -> list[FileChange]:
    system_prompt = (
        "You are a coding agent. Return ONLY file updates in this exact protocol:\n"
        "FILE: relative/path.ext\n"
        "<full file content>\n"
        "Repeat for each file. No markdown fences or explanations."
    )

    user_prompt = (
        f"User request:\n{user_request}\n\n"
        f"Current plan step:\n{step_description}\n\n"
        f"Retrieved context:\n{_context_blob(retrieved_context)}\n\n"
        "Produce complete file contents for all files you need to modify."
    )

    try:
        raw = generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=6000,
            strict=True,
            cwd=cwd,
            heartbeat=heartbeat,
        )
    except CodexInvocationError as exc:
        raise ValueError(f"Coder agent invocation failed: {exc}") from exc
    if not raw:
        raise ValueError("Coder agent did not return output.")

    try:
        changes = parse_file_blocks(raw)
        validate_file_changes(changes)
    except Exception as exc:
        prompt = (
            "SYSTEM INSTRUCTIONS:\n"
            f"{system_prompt}\n\n"
            "USER REQUEST:\n"
            f"{user_prompt}\n\n"
            "OUTPUT RULES:\n"
            "- Keep output under approximately 6000 tokens.\n"
            "- Return only the requested output content.\n"
        )
        snapshot = save_prompt_debug_snapshot(
            prompt=prompt,
            cwd=cwd,
            reason=f"coder_parse_or_validate_failure:{type(exc).__name__}",
            raw_output=raw[:4000],
        )
        suffix = f" Prompt snapshot: {snapshot}" if snapshot else ""
        raise ValueError(f"{exc}.{suffix}") from exc

    return changes
