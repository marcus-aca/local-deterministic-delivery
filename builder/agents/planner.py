"""Planner agent produces an ordered implementation plan."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import db
from agents.common import generate_json

DEFAULT_PLAN_DEBUG_FILE = ".codex_last_plan.json"


def _plan_debug_file() -> str:
    return os.getenv("PLAN_DEBUG_FILE", DEFAULT_PLAN_DEBUG_FILE).strip()


def _resolve_plan_debug_path(repo_path: str | None = None) -> Path | None:
    setting = _plan_debug_file()
    if not setting:
        return None

    candidate = Path(setting)
    if candidate.is_absolute():
        return candidate

    base = Path(repo_path).resolve() if repo_path else Path.cwd()
    return base / candidate


def save_plan_snapshot(
    *,
    user_request: str,
    mode: str,
    context_files: list[str],
    payload: dict[str, Any],
    steps: list[str],
    repo_path: str | None = None,
    run_id: str | None = None,
) -> str | None:
    target = _resolve_plan_debug_path(repo_path=repo_path)
    if target is None:
        return None

    body = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "repo_path": str(Path(repo_path).resolve()) if repo_path else None,
        "user_request": user_request,
        "mode": mode,
        "context_files": context_files,
        "raw_plan_payload": payload,
        "final_steps": steps,
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(body, indent=2), encoding="utf-8")
    except OSError:
        return None
    return str(target)


def _fallback_steps(user_request: str, mode: str) -> list[str]:
    return [
        f"Analyze repository context and constraints for: {user_request}",
        f"Implement code changes required by mode={mode}",
        "Run tests/build in sandbox and resolve failures",
    ]


def create_plan(
    user_request: str,
    mode: str,
    retrieval_context: dict[str, Any] | None = None,
    repo_path: str | None = None,
    run_id: str | None = None,
) -> list[str]:
    context_files = []
    if retrieval_context:
        context_files = [item.get("file_path") for item in retrieval_context.get("files", [])[:12]]

    default_steps = _fallback_steps(user_request, mode)

    payload = generate_json(
        system_prompt=(
            "Generate an ordered implementation plan for a coding agent run. "
            "Return strict JSON: {\"steps\": [\"step1\", \"step2\", ...]}. "
            "Keep between 2 and 7 concise actionable steps."
        ),
        user_prompt=(
            f"User request: {user_request}\n"
            f"Mode: {mode}\n"
            f"Relevant files: {context_files}"
        ),
        default={"steps": default_steps},
    )

    raw_steps = payload.get("steps", default_steps)
    steps = [str(step).strip() for step in raw_steps if str(step).strip()]
    final_steps = steps if steps else default_steps
    save_plan_snapshot(
        user_request=user_request,
        mode=mode,
        context_files=context_files,
        payload=payload,
        steps=final_steps,
        repo_path=repo_path,
        run_id=run_id,
    )
    return final_steps


def persist_plan(run_id: str, steps: list[str]) -> None:
    db.insert_plan_steps(run_id, steps)
