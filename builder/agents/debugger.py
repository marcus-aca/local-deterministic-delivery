"""Debugger agent proposes corrective file updates after sandbox failures."""

from __future__ import annotations

from typing import Any, Callable

from agents.coder import FileChange, parse_file_blocks, validate_file_changes
from agents.common import CodexInvocationError, generate_text, save_prompt_debug_snapshot


def generate_debug_fixes(
    user_request: str,
    step_description: str,
    stderr: str,
    stdout: str,
    retrieved_context: dict[str, Any] | None = None,
    cwd: str | None = None,
    heartbeat: Callable[[int], None] | None = None,
) -> list[FileChange]:
    context_lines = []
    if retrieved_context:
        for item in retrieved_context.get("files", [])[:10]:
            path = item.get("file_path", "unknown")
            snippet = (item.get("snippets") or [""])[0][:400]
            context_lines.append(f"- {path}\n{snippet}")

    system_prompt = (
        "You are debugging a failing build/test run. "
        "Return ONLY updates in strict protocol:\n"
        "FILE: relative/path.ext\n<full file content>\n"
        "No commentary."
    )

    user_prompt = (
        f"User request:\n{user_request}\n\n"
        f"Plan step:\n{step_description}\n\n"
        f"Sandbox stdout:\n{stdout[-3000:]}\n\n"
        f"Sandbox stderr:\n{stderr[-3000:]}\n\n"
        f"Relevant context:\n{chr(10).join(context_lines) if context_lines else 'none'}"
    )

    try:
        raw = generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=7000,
            strict=True,
            cwd=cwd,
            heartbeat=heartbeat,
        )
    except CodexInvocationError:
        return []
    if not raw:
        return []

    try:
        changes = parse_file_blocks(raw)
        validate_file_changes(changes)
    except Exception:
        prompt = (
            "SYSTEM INSTRUCTIONS:\n"
            f"{system_prompt}\n\n"
            "USER REQUEST:\n"
            f"{user_prompt}\n\n"
            "OUTPUT RULES:\n"
            "- Keep output under approximately 7000 tokens.\n"
            "- Return only the requested output content.\n"
        )
        save_prompt_debug_snapshot(
            prompt=prompt,
            cwd=cwd,
            reason="debugger_parse_or_validate_failure",
            raw_output=raw[:4000],
        )
        raise

    return changes
