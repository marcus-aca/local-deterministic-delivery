"""Strategy agent decides whether to modify an existing repo or scaffold a new one."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import db
from agents.common import generate_json

IGNORED_TOP_LEVEL_FOR_STRATEGY = {"design.md"}


@dataclass
class StrategyDecision:
    mode: str
    reason: str


def _meaningful_top_level_entries(repo_path: str) -> list[str]:
    root = Path(repo_path)
    if not root.exists():
        return []

    entries: list[str] = []
    for item in root.iterdir():
        name = item.name
        if name.lower() in IGNORED_TOP_LEVEL_FOR_STRATEGY:
            continue
        entries.append(name)
    return entries


def _fallback_mode(user_request: str, repo_path: str) -> StrategyDecision:
    request = user_request.lower()
    repo_entries = _meaningful_top_level_entries(repo_path)

    wants_new = any(token in request for token in ["from scratch", "new project", "scaffold", "create project"])
    if wants_new or not repo_entries:
        return StrategyDecision(mode="scaffold_new", reason="Request or workspace indicates new project scaffolding.")

    return StrategyDecision(mode="modify_existing", reason="Repository exists and request implies incremental changes.")


def choose_strategy(user_request: str, repo_path: str) -> StrategyDecision:
    top_level = sorted(name for name in _meaningful_top_level_entries(repo_path) if not name.startswith("."))

    default = {
        "mode": _fallback_mode(user_request, repo_path).mode,
        "reason": "Fallback strategy selected without LLM output.",
    }

    payload = generate_json(
        system_prompt=(
            "Choose strategy mode for a coding run. "
            "Return JSON with keys mode and reason. "
            "mode must be either modify_existing or scaffold_new."
        ),
        user_prompt=(
            f"User request: {user_request}\n"
            f"Repository: {os.path.abspath(repo_path)}\n"
            f"Top-level files/dirs: {top_level[:80]}"
        ),
        default=default,
    )

    mode = payload.get("mode", default["mode"])
    if mode not in {"modify_existing", "scaffold_new"}:
        mode = default["mode"]

    reason = payload.get("reason", default["reason"])
    return StrategyDecision(mode=mode, reason=reason)


def persist_strategy(run_id: str, decision: StrategyDecision) -> None:
    db.update_run(run_id, mode=decision.mode, status="planning")
