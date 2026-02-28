"""Shared Codex CLI helper functions used by agent modules."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()

DEFAULT_CODEX_FLAGS = ["--skip-git-repo-check"]
DEFAULT_CODEX_PROMPT_DEBUG_FILE = ".codex_last_prompt.txt"


class CodexInvocationError(RuntimeError):
    """Raised when a strict Codex CLI call fails."""


HeartbeatFn = Callable[[int], None]


def _tail(text: str, max_chars: int = 1200) -> str:
    trimmed = text.strip()
    if len(trimmed) <= max_chars:
        return trimmed
    return trimmed[-max_chars:]


def _codex_bin() -> str:
    return os.getenv("CODEX_CLI_BIN", "codex")


def _codex_extra_flags() -> list[str]:
    raw = os.getenv("CODEX_CLI_EXTRA_FLAGS", "").strip()
    flags = shlex.split(raw) if raw else []
    for default_flag in DEFAULT_CODEX_FLAGS:
        if default_flag not in flags:
            flags.append(default_flag)
    return flags


def _codex_timeout_seconds() -> int:
    return int(os.getenv("CODEX_CLI_TIMEOUT_SECONDS", "600"))


def _codex_prompt_debug_file() -> str:
    return os.getenv("CODEX_PROMPT_DEBUG_FILE", DEFAULT_CODEX_PROMPT_DEBUG_FILE).strip()


def _codex_default_cwd() -> str | None:
    raw = os.getenv("CODEX_CLI_CWD", "").strip()
    if not raw:
        return None
    return str(Path(raw).expanduser().resolve())


def _build_prompt(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    return (
        "SYSTEM INSTRUCTIONS:\n"
        f"{system_prompt}\n\n"
        "USER REQUEST:\n"
        f"{user_prompt}\n\n"
        "OUTPUT RULES:\n"
        f"- Keep output under approximately {max_tokens} tokens.\n"
        "- Return only the requested output content.\n"
    )


def _extract_stdout_text(stdout: str) -> str | None:
    text = stdout.strip()
    return text if text else None


def _resolve_prompt_debug_path(cwd: str | None = None) -> Path | None:
    setting = _codex_prompt_debug_file()
    if not setting:
        return None

    candidate = Path(setting)
    if candidate.is_absolute():
        return candidate

    base = Path(cwd).resolve() if cwd else Path.cwd()
    return base / candidate


def save_prompt_debug_snapshot(
    *,
    prompt: str,
    cwd: str | None = None,
    reason: str | None = None,
    cli_cmd: list[str] | None = None,
    raw_output: str | None = None,
) -> str | None:
    target = _resolve_prompt_debug_path(cwd)
    if target is None:
        return None

    lines: list[str] = [
        "# Codex prompt debug snapshot",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"cwd={cwd or os.getcwd()}",
    ]
    if reason:
        lines.append(f"reason={reason}")
    if cli_cmd:
        lines.append("cli_cmd=" + " ".join(cli_cmd))
    lines.append("")
    lines.append(prompt)
    if raw_output:
        lines.append("")
        lines.append("# Raw output")
        lines.append(raw_output)
    payload = "\n".join(lines).rstrip() + "\n"

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
    return str(target)


def _communicate_with_heartbeat(
    process: subprocess.Popen[str],
    timeout_seconds: int,
    heartbeat: HeartbeatFn | None,
    heartbeat_interval_seconds: int,
) -> tuple[str, str]:
    start_time = time.monotonic()
    interval = max(1, heartbeat_interval_seconds)
    next_heartbeat_at = start_time + interval

    while True:
        elapsed = time.monotonic() - start_time
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout_seconds)

        try:
            return process.communicate(timeout=min(1.0, remaining))
        except subprocess.TimeoutExpired:
            if heartbeat and time.monotonic() >= next_heartbeat_at:
                heartbeat(int(time.monotonic() - start_time))
                next_heartbeat_at += interval


def generate_text(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1200,
    cwd: str | None = None,
    strict: bool = False,
    heartbeat: HeartbeatFn | None = None,
    heartbeat_interval_seconds: int = 30,
) -> str | None:
    codex_bin = _codex_bin()
    run_cwd = cwd or _codex_default_cwd()
    if shutil.which(codex_bin) is None:
        if strict:
            prompt = _build_prompt(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=max_tokens)
            snapshot = save_prompt_debug_snapshot(
                prompt=prompt,
                cwd=run_cwd,
                reason=f"codex_bin_not_found:{codex_bin}",
            )
            suffix = f" Prompt snapshot: {snapshot}" if snapshot else ""
            raise CodexInvocationError(f"Codex CLI binary not found in PATH: {codex_bin}.{suffix}")
        return None

    cli_extra_flags = _codex_extra_flags()
    prompt = _build_prompt(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=max_tokens)
    output_file: str | None = None
    with tempfile.NamedTemporaryFile(prefix="codex-last-", suffix=".txt", delete=False) as temp_output:
        output_file = temp_output.name
    cli_cmd = [codex_bin, "e", *cli_extra_flags, "-o", output_file, prompt]

    stderr = ""
    stdout = ""
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            cli_cmd,
            cwd=run_cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = _communicate_with_heartbeat(
            process=process,
            timeout_seconds=_codex_timeout_seconds(),
            heartbeat=heartbeat,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        )
    except subprocess.TimeoutExpired:
        if process and process.poll() is None:
            process.kill()
        if strict:
            snapshot = save_prompt_debug_snapshot(
                prompt=prompt,
                cwd=run_cwd,
                reason=f"timeout:{_codex_timeout_seconds()}s",
                cli_cmd=cli_cmd,
            )
            suffix = f" Prompt snapshot: {snapshot}" if snapshot else ""
            raise CodexInvocationError(
                f"Codex CLI timed out after {_codex_timeout_seconds()}s (cwd={run_cwd or os.getcwd()}).{suffix}"
            )
        return None
    except OSError as exc:
        if strict:
            snapshot = save_prompt_debug_snapshot(
                prompt=prompt,
                cwd=run_cwd,
                reason=f"os_error:{exc}",
                cli_cmd=cli_cmd,
            )
            suffix = f" Prompt snapshot: {snapshot}" if snapshot else ""
            raise CodexInvocationError(f"Failed to launch Codex CLI: {exc}.{suffix}") from exc
        return None

    file_text = ""
    if output_file:
        try:
            file_text = Path(output_file).read_text(encoding="utf-8")
        except OSError:
            file_text = ""
        finally:
            try:
                Path(output_file).unlink(missing_ok=True)
            except OSError:
                pass

    text = _extract_stdout_text(file_text) or _extract_stdout_text(stdout)
    if process and process.returncode != 0:
        if strict:
            detail = _tail(stderr or stdout or "No CLI output captured.")
            snapshot = save_prompt_debug_snapshot(
                prompt=prompt,
                cwd=run_cwd,
                reason=f"nonzero_exit:{process.returncode}",
                cli_cmd=cli_cmd,
                raw_output=detail,
            )
            suffix = f" Prompt snapshot: {snapshot}" if snapshot else ""
            raise CodexInvocationError(f"Codex CLI exited with code {process.returncode}: {detail}.{suffix}")
        return text

    if not text and strict:
        detail = _tail(stderr or "No stdout/stderr output captured.")
        snapshot = save_prompt_debug_snapshot(
            prompt=prompt,
            cwd=run_cwd,
            reason="empty_output",
            cli_cmd=cli_cmd,
            raw_output=detail,
        )
        suffix = f" Prompt snapshot: {snapshot}" if snapshot else ""
        raise CodexInvocationError(f"Codex CLI returned empty output. {detail}.{suffix}")

    return text


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            return "\n".join(lines[1:-1]).strip()
    return stripped


def generate_json(system_prompt: str, user_prompt: str, default: Any, cwd: str | None = None) -> Any:
    raw = generate_text(system_prompt=system_prompt, user_prompt=user_prompt, cwd=cwd)
    if not raw:
        return default

    candidate = _strip_json_fence(raw)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return default
