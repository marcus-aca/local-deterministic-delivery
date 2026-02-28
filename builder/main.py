"""CLI entrypoint for deterministic local delivery orchestrator.

Examples:
  python main.py --repo ./myrepo "Add JWT auth"
  python main.py --mode scaffold_new --scaffold-path ./new-service "Create a REST API"
  python main.py --repo ./myrepo --request-file ./design.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import db
from orchestrator import execute_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic local delivery runner")
    parser.add_argument("request", nargs="?", help="Natural-language feature request text")
    parser.add_argument(
        "--request-file",
        help="Path to a text/markdown file containing the feature request (default lookup: <repo>/design.md)",
    )
    parser.add_argument("--repo", default=".", help="Target repository path")
    parser.add_argument(
        "--mode",
        choices=["auto", "modify_existing", "scaffold_new"],
        default="auto",
        help="Run mode hint for repo handling",
    )
    parser.add_argument(
        "--scaffold-path",
        help="Path to create/use when mode is scaffold_new",
    )
    parser.add_argument("--run-id", help="Existing run ID to resume")
    parser.add_argument(
        "--resume",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Resume latest in-progress run for this repo or start fresh",
    )
    parser.add_argument("--max-retries", type=int, default=int(os.getenv("MAX_RETRIES", "2")))
    parser.add_argument(
        "--graph-debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit structured JSON LangGraph transition events (default: enabled).",
    )
    return parser.parse_args()


def _emit_graph_debug(event: dict[str, object]) -> None:
    print(json.dumps({"graph_debug": event}, default=str))


def _resolve_repo_path(args: argparse.Namespace) -> str:
    if args.mode == "scaffold_new":
        target = os.path.abspath(args.scaffold_path or args.repo)
        os.makedirs(target, exist_ok=True)
        return target

    target = os.path.abspath(args.repo)
    if args.mode == "modify_existing" and not os.path.exists(target):
        raise ValueError(f"Repository path does not exist: {target}")

    if args.mode == "auto" and not os.path.exists(target):
        os.makedirs(target, exist_ok=True)

    return target


def _read_request_file(request_file: str) -> str:
    if not os.path.exists(request_file):
        raise ValueError(f"Request file does not exist: {request_file}")
    if not os.path.isfile(request_file):
        raise ValueError(f"Request file path is not a file: {request_file}")

    try:
        with open(request_file, "r", encoding="utf-8") as handle:
            content = handle.read()
    except OSError as exc:
        raise ValueError(f"Failed reading request file {request_file}: {exc}") from exc

    request = content.strip()
    if not request:
        raise ValueError(f"Request file is empty: {request_file}")

    return request


def _resolve_explicit_request_text(args: argparse.Namespace) -> str | None:
    if args.request and args.request_file:
        raise ValueError("Provide either inline request text or --request-file, not both.")

    if args.request:
        return args.request

    if args.request_file:
        return _read_request_file(os.path.abspath(args.request_file))

    return None


def _resolve_default_request_from_repo(repo_path: str) -> str | None:
    candidate = os.path.join(repo_path, "design.md")
    if not os.path.exists(candidate) or not os.path.isfile(candidate):
        return None
    return _read_request_file(candidate)


def _decide_run_id(args: argparse.Namespace, repo_path: str) -> tuple[str | None, str | None]:
    if args.run_id:
        run = db.get_run(args.run_id)
        if not run:
            raise ValueError(f"Run ID {args.run_id} not found")
        request = args.request or run["user_request"]
        return args.run_id, request

    if args.resume == "no":
        return None, args.request

    resumable = db.get_latest_resumable_run(repo_path, include_failed=(args.resume == "yes"))

    if not resumable:
        return None, args.request

    if args.resume == "yes":
        return resumable["id"], args.request or resumable["user_request"]

    if not os.isatty(0):
        return None, args.request

    run_id = resumable["id"]
    step = resumable.get("current_step")
    prompt = f"Found resumable run {run_id} at step {step}. Resume it? [y/N]: "
    answer = input(prompt).strip().lower()
    if answer in {"y", "yes"}:
        return run_id, args.request or resumable["user_request"]

    return None, args.request


def main() -> None:
    args = parse_args()
    repo_path = _resolve_repo_path(args)
    args.request = _resolve_explicit_request_text(args)

    db.init_schema()

    run_id, request = _decide_run_id(args, repo_path)
    if not request and not run_id:
        request = _resolve_default_request_from_repo(repo_path)
    if not request:
        raise ValueError("A request is required when starting a new run.")

    result = execute_run(
        repo_path=repo_path,
        user_request=request,
        run_id=run_id,
        max_retries=args.max_retries,
        progress=lambda line: print(f"[run] {line}"),
        graph_debug=args.graph_debug,
        debug=_emit_graph_debug,
    )
    print(json.dumps(result, indent=2))
    if result.get("status") != "completed":
        sys.exit(1)


if __name__ == "__main__":
    main()
