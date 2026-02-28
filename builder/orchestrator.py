"""Deterministic LangGraph orchestration with resume support."""

from __future__ import annotations

import os
import uuid
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict

warnings.filterwarnings(
    "ignore",
    message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.14 or greater\.",
    category=UserWarning,
    module=r"langchain_core\._api\.deprecation",
)

from langgraph.graph import END, START, StateGraph

import db
import indexer
import retrieval
from agents import coder, debugger, planner, strategy
from sandbox import run_in_sandbox

ProgressFn = Callable[[str], None]
DebugFn = Callable[[dict[str, Any]], None]


class OrchestrationState(TypedDict, total=False):
    repo_path: str
    user_request: str
    run_id: str
    max_retries: int
    progress: ProgressFn | None
    mode: str
    plan_rows: list[dict[str, Any]]
    total_steps: int
    next_step: int
    status: str
    error: str
    result: dict[str, Any]
    debug: DebugFn | None


def _emit(state: OrchestrationState, message: str) -> None:
    callback = state.get("progress")
    if callback:
        callback(message)


def _debug(state: OrchestrationState, node: str, event: str, **fields: Any) -> None:
    callback = state.get("debug")
    if not callback:
        return

    payload: dict[str, Any] = {
        "event": event,
        "node": node,
        "run_id": state.get("run_id"),
        "status": state.get("status"),
        "next_step": state.get("next_step"),
        "total_steps": state.get("total_steps"),
    }
    payload.update(fields)
    callback(payload)


def _apply_file_changes(repo_path: str, changes: list[coder.FileChange]) -> None:
    for change in changes:
        target = Path(repo_path) / change.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(change.content, encoding="utf-8")


def _sync_changed_embeddings(repo_path: str, changes: list[coder.FileChange]) -> dict[str, int | str]:
    return indexer.index_files(
        repo_path=repo_path,
        file_paths=[change.path for change in changes],
        chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
        overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
    )


def _single_line_reason(stdout: str, stderr: str, max_chars: int = 180) -> str:
    for source in (stderr, stdout):
        if not source:
            continue
        lines = [line.strip() for line in source.splitlines() if line.strip()]
        if not lines:
            continue
        reason = lines[-1]
        if len(reason) > max_chars:
            return reason[: max_chars - 3] + "..."
        return reason
    return "No stdout/stderr details available."


def _bootstrap_or_load_run(repo_path: str, user_request: str, run_id: str | None = None) -> dict[str, Any]:
    if run_id:
        existing = db.get_run(run_id)
        if not existing:
            raise ValueError(f"Run {run_id} does not exist.")
        return existing

    new_run_id = str(uuid.uuid4())
    db.create_run(new_run_id, repo_path=repo_path, user_request=user_request, status="planning")
    run = db.get_run(new_run_id)
    if not run:
        raise RuntimeError("Failed to create run.")
    return run


def _ensure_plan(run: dict[str, Any]) -> list[dict[str, Any]]:
    run_id = run["id"]
    existing_steps = db.get_plan_steps(run_id)
    if existing_steps:
        return existing_steps

    decision = strategy.choose_strategy(run["user_request"], run["repo_path"])
    strategy.persist_strategy(run_id, decision)

    context = retrieval.retrieve_context(run["repo_path"], run["user_request"]) if decision.mode == "modify_existing" else None
    steps = planner.create_plan(
        run["user_request"],
        decision.mode,
        context,
        repo_path=run["repo_path"],
        run_id=str(run_id),
    )
    planner.persist_plan(run_id, steps)

    db.update_run(run_id, status="in_progress", current_step=1, retry_count=0)
    return db.get_plan_steps(run_id)


def _failed_result(state: OrchestrationState, reason: str, current_step: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": state.get("run_id"),
        "status": "failed",
        "reason": reason,
    }
    if current_step is not None:
        payload["current_step"] = current_step
    return payload


def _bootstrap_node(state: OrchestrationState) -> OrchestrationState:
    _debug(state, "bootstrap", "enter", repo_path=state.get("repo_path"))
    repo_root = str(Path(str(state["repo_path"])).resolve())
    run = _bootstrap_or_load_run(
        repo_path=repo_root,
        user_request=state["user_request"],
        run_id=state.get("run_id"),
    )
    run_id = str(run["id"])

    _emit(state, f"Run {run_id} started (repo={repo_root}).")

    if run.get("user_request") != state["user_request"] and not run.get("user_request"):
        db.update_run(run_id, user_request=state["user_request"])

    refreshed = db.get_run(run_id) or run
    mode = str(refreshed.get("mode") or "")
    _debug(state, "bootstrap", "exit", run_id=run_id, mode=mode or None)

    return {
        "repo_path": repo_root,
        "run_id": run_id,
        "user_request": state["user_request"],
        "mode": mode,
        "status": "in_progress",
    }


def _plan_node(state: OrchestrationState) -> OrchestrationState:
    _debug(state, "plan", "enter")
    run = db.get_run(state["run_id"])
    if not run:
        _debug(state, "plan", "fail", reason="run_not_found")
        return {
            "status": "failed",
            "result": _failed_result(state, reason=f"Run {state['run_id']} not found."),
        }

    plan_rows = _ensure_plan(run)
    if not plan_rows:
        db.update_run(state["run_id"], status="failed")
        _emit(state, "Run failed: no plan steps generated.")
        _debug(state, "plan", "fail", reason="no_plan_steps")
        return {
            "status": "failed",
            "result": _failed_result(state, reason="No plan steps generated."),
        }

    refreshed = db.get_run(state["run_id"]) or run
    mode = str(refreshed.get("mode") or "modify_existing")

    if mode != "scaffold_new":
        _emit(state, "Syncing repository index before execution.")
        sync = indexer.index_repo(
            state["repo_path"],
            chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
            overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        )
        _emit(
            state,
            "Initial index sync complete "
            f"(changed={sync['changed']}, skipped={sync['skipped']}, deleted={sync['deleted']}).",
        )
        _debug(
            state,
            "plan",
            "indexed",
            changed=sync["changed"],
            skipped=sync["skipped"],
            deleted=sync["deleted"],
        )

    start_step = max(1, int(refreshed.get("current_step") or 1))
    pending_rows = [
        row
        for row in plan_rows
        if int(row["step_number"]) >= start_step and row.get("status") != "completed"
    ]

    total_steps = len(plan_rows)
    if not pending_rows:
        db.update_run(state["run_id"], status="completed", current_step=total_steps + 1, retry_count=0)
        result = {
            "run_id": state["run_id"],
            "status": "completed",
            "steps": total_steps,
            "repo_path": state["repo_path"],
        }
        _emit(state, f"Run {state['run_id']} completed.")
        _debug(state, "plan", "exit", mode=mode, plan_steps=total_steps, pending_steps=0)
        return {
            "mode": mode,
            "plan_rows": plan_rows,
            "total_steps": total_steps,
            "next_step": total_steps + 1,
            "status": "completed",
            "result": result,
        }

    next_step = int(pending_rows[0]["step_number"])
    _debug(
        state,
        "plan",
        "exit",
        mode=mode,
        plan_steps=total_steps,
        pending_steps=len(pending_rows),
        next_step=next_step,
    )
    return {
        "mode": mode,
        "plan_rows": plan_rows,
        "total_steps": total_steps,
        "next_step": next_step,
        "status": "in_progress",
    }


def _step_node(state: OrchestrationState) -> OrchestrationState:
    step_number = int(state["next_step"])
    _debug(state, "step", "enter", step_number=step_number)
    step_row = next(
        (row for row in state["plan_rows"] if int(row["step_number"]) == step_number),
        None,
    )

    if not step_row:
        db.update_run(state["run_id"], status="failed", current_step=step_number)
        _debug(state, "step", "fail", step_number=step_number, reason="missing_step")
        return {
            "status": "failed",
            "result": _failed_result(state, reason=f"Plan step {step_number} missing.", current_step=step_number),
        }

    step_description = str(step_row["description"])
    _emit(state, f"Step {step_number}/{state['total_steps']} in progress: {step_description}")

    db.update_plan_status(state["run_id"], step_number, "in_progress")
    db.update_run(state["run_id"], status="in_progress", current_step=step_number)

    context: dict[str, Any] | None = None
    if state.get("mode") == "scaffold_new":
        _emit(state, f"Step {step_number}: skipping retrieval for scaffold_new mode.")
        _debug(state, "step", "skip_retrieval", step_number=step_number, mode=state.get("mode"))
    else:
        _emit(state, f"Step {step_number}: retrieving repository context.")
        _debug(state, "step", "retrieve_start", step_number=step_number)
        context = retrieval.retrieve_context(
            state["repo_path"],
            user_query=f"{state['user_request']}\nPlan step: {step_description}",
        )
        _debug(state, "step", "retrieve_done", step_number=step_number, files=len(context.get("files", [])))

    try:
        _emit(state, f"Step {step_number}: requesting coder changes.")
        _debug(state, "step", "coder_start", step_number=step_number)
        code_changes = coder.generate_code_changes(
            user_request=state["user_request"],
            step_description=step_description,
            retrieved_context=context,
            cwd=state["repo_path"],
            heartbeat=lambda elapsed: _emit(
                state,
                f"Step {step_number}: waiting for coder output ({elapsed}s elapsed).",
            ),
        )
        coder.validate_file_changes(code_changes)
        _apply_file_changes(state["repo_path"], code_changes)
        sync = _sync_changed_embeddings(state["repo_path"], code_changes)
        _emit(
            state,
            "Indexed applied changes "
            f"(changed={sync['changed']}, skipped={sync['skipped']}, deleted={sync['deleted']}).",
        )
        _debug(state, "step", "coder_done", step_number=step_number, changed_files=len(code_changes))
    except Exception as exc:
        db.update_plan_status(state["run_id"], step_number, "failed")
        db.update_run(state["run_id"], status="failed", current_step=step_number)
        _emit(state, f"Step {step_number} failed in coder: {exc}")
        _debug(state, "step", "fail", step_number=step_number, reason="coder_failure")
        return {
            "status": "failed",
            "result": _failed_result(state, reason=f"Coder failure: {exc}", current_step=step_number),
        }

    _emit(state, f"Step {step_number}: running sandbox validation.")
    _debug(state, "step", "sandbox_start", step_number=step_number)
    sandbox_result = run_in_sandbox(state["repo_path"])
    db.insert_execution_log(
        run_id=state["run_id"],
        step_number=step_number,
        stdout=sandbox_result.stdout,
        stderr=sandbox_result.stderr,
        exit_code=sandbox_result.exit_code,
    )

    if sandbox_result.exit_code != 0:
        fixed = False
        retries = 0
        reason_line = _single_line_reason(sandbox_result.stdout, sandbox_result.stderr)
        _debug(
            state,
            "step",
            "sandbox_fail",
            step_number=step_number,
            exit_code=sandbox_result.exit_code,
            reason_line=reason_line,
        )

        while retries < int(state["max_retries"]):
            retries += 1
            db.update_run(
                state["run_id"],
                status="retrying",
                current_step=step_number,
                retry_count=retries,
            )
            _emit(
                state,
                (
                    f"Step {step_number} sandbox failed (exit={sandbox_result.exit_code}): "
                    f"{reason_line}. Retry {retries}/{state['max_retries']}."
                ),
            )
            _debug(state, "step", "retry", step_number=step_number, attempt=retries)

            try:
                debug_changes = debugger.generate_debug_fixes(
                    user_request=state["user_request"],
                    step_description=step_description,
                    stdout=sandbox_result.stdout,
                    stderr=sandbox_result.stderr,
                    retrieved_context=context,
                    cwd=state["repo_path"],
                    heartbeat=lambda elapsed: _emit(
                        state,
                        f"Step {step_number}: waiting for debugger output ({elapsed}s elapsed).",
                    ),
                )
                if not debug_changes:
                    break

                coder.validate_file_changes(debug_changes)
                _apply_file_changes(state["repo_path"], debug_changes)
                sync = _sync_changed_embeddings(state["repo_path"], debug_changes)
                _emit(
                    state,
                    "Indexed debugger-applied changes "
                    f"(changed={sync['changed']}, skipped={sync['skipped']}, deleted={sync['deleted']}).",
                )
            except Exception:
                _debug(state, "step", "retry_fail", step_number=step_number, attempt=retries, reason="debugger_failure")
                break

            sandbox_result = run_in_sandbox(state["repo_path"])
            db.insert_execution_log(
                run_id=state["run_id"],
                step_number=step_number,
                stdout=sandbox_result.stdout,
                stderr=sandbox_result.stderr,
                exit_code=sandbox_result.exit_code,
            )

            if sandbox_result.exit_code == 0:
                fixed = True
                _debug(state, "step", "retry_success", step_number=step_number, attempt=retries)
                break
            reason_line = _single_line_reason(sandbox_result.stdout, sandbox_result.stderr)

        if sandbox_result.exit_code != 0 and not fixed:
            db.update_plan_status(state["run_id"], step_number, "failed")
            db.update_run(state["run_id"], status="failed", current_step=step_number)
            _emit(
                state,
                (
                    f"Step {step_number} failed after retries (exit={sandbox_result.exit_code}): "
                    f"{reason_line}. Marking run failed."
                ),
            )
            _debug(
                state,
                "step",
                "fail",
                step_number=step_number,
                reason="sandbox_failure",
                reason_line=reason_line,
            )
            return {
                "status": "failed",
                "result": {
                    "run_id": state["run_id"],
                    "status": "failed",
                    "current_step": step_number,
                    "sandbox": asdict(sandbox_result),
                },
            }

    next_step = step_number + 1
    db.update_plan_status(state["run_id"], step_number, "completed")
    db.update_run(state["run_id"], status="in_progress", current_step=next_step, retry_count=0)
    _emit(state, f"Step {step_number} completed.")
    _debug(state, "step", "exit", step_number=step_number, next_step=next_step)

    return {
        "next_step": next_step,
        "status": "in_progress",
    }


def _finalize_node(state: OrchestrationState) -> OrchestrationState:
    _debug(state, "finalize", "enter")
    total_steps = int(state["total_steps"])
    db.update_run(state["run_id"], status="completed", current_step=total_steps + 1, retry_count=0)

    result = {
        "run_id": state["run_id"],
        "status": "completed",
        "steps": total_steps,
        "repo_path": state["repo_path"],
    }
    _emit(state, f"Run {state['run_id']} completed.")
    _debug(state, "finalize", "exit", total_steps=total_steps)
    return {
        "status": "completed",
        "result": result,
    }


def _route_after_plan(state: OrchestrationState) -> Literal["step", "done", "fail"]:
    status = state.get("status")
    if status == "failed":
        _debug(state, "plan", "route", decision="fail")
        return "fail"
    if status == "completed":
        _debug(state, "plan", "route", decision="done")
        return "done"
    _debug(state, "plan", "route", decision="step")
    return "step"


def _route_after_step(state: OrchestrationState) -> Literal["step", "done", "fail"]:
    status = state.get("status")
    if status == "failed":
        _debug(state, "step", "route", decision="fail")
        return "fail"

    if int(state["next_step"]) > int(state["total_steps"]):
        _debug(state, "step", "route", decision="done")
        return "done"

    _debug(state, "step", "route", decision="step")
    return "step"


def _build_graph():
    graph = StateGraph(OrchestrationState)

    graph.add_node("bootstrap", _bootstrap_node)
    graph.add_node("plan", _plan_node)
    graph.add_node("step", _step_node)
    graph.add_node("finalize", _finalize_node)

    graph.add_edge(START, "bootstrap")
    graph.add_edge("bootstrap", "plan")
    graph.add_conditional_edges(
        "plan",
        _route_after_plan,
        {
            "step": "step",
            "done": END,
            "fail": END,
        },
    )
    graph.add_conditional_edges(
        "step",
        _route_after_step,
        {
            "step": "step",
            "done": "finalize",
            "fail": END,
        },
    )
    graph.add_edge("finalize", END)

    return graph.compile()


def execute_run(
    repo_path: str,
    user_request: str,
    run_id: str | None = None,
    max_retries: int = 2,
    progress: ProgressFn | None = None,
    graph_debug: bool = False,
    debug: DebugFn | None = None,
) -> dict[str, Any]:
    db.init_schema()

    debug_callback = debug if graph_debug else None

    graph = _build_graph()
    final_state = graph.invoke(
        {
            "repo_path": repo_path,
            "user_request": user_request,
            "run_id": run_id,
            "max_retries": max_retries,
            "progress": progress,
            "debug": debug_callback,
            "status": "in_progress",
        },
        config={"recursion_limit": 500},
    )

    result = final_state.get("result")
    if result:
        return result

    if final_state.get("status") == "completed":
        return {
            "run_id": final_state.get("run_id"),
            "status": "completed",
            "steps": final_state.get("total_steps", 0),
            "repo_path": final_state.get("repo_path"),
        }

    failure_reason = final_state.get("error") or "LangGraph run ended without a terminal result."
    failed_payload = {
        "run_id": final_state.get("run_id"),
        "status": "failed",
        "reason": failure_reason,
    }
    if final_state.get("run_id"):
        db.update_run(str(final_state["run_id"]), status="failed")
    return failed_payload
