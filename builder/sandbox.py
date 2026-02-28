"""Docker sandbox execution with resource limits and structured output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    image: str
    command: str
    timed_out: bool


def detect_repo_runtime(repo_path: str) -> tuple[str, str]:
    root = Path(repo_path)

    if (root / "package.json").exists():
        return (
            "node:20",
            "if [ -f package.json ]; then npm test --if-present; else echo 'No package.json'; fi",
        )

    if (root / "pyproject.toml").exists() or (root / "requirements.txt").exists() or (root / "setup.py").exists():
        return (
            "python:3.11-slim",
            "if command -v pytest >/dev/null 2>&1 && ([ -d tests ] || ls test_*.py >/dev/null 2>&1); then pytest -q; else python -m compileall -q .; fi",
        )

    return ("python:3.11-slim", "python -m compileall -q .")


def _node_lock_hash(repo_root: Path) -> str:
    hasher = hashlib.sha256()
    lockfiles = [
        "package-lock.json",
        "npm-shrinkwrap.json",
        "pnpm-lock.yaml",
        "yarn.lock",
    ]
    found = False
    for lockfile in lockfiles:
        path = repo_root / lockfile
        if not path.exists() or not path.is_file():
            continue
        found = True
        hasher.update(lockfile.encode("utf-8"))
        hasher.update(path.read_bytes())
    if not found:
        return "nolock"
    return hasher.hexdigest()[:12]


def _node_volume_name(repo_root: Path) -> str:
    repo_hash = hashlib.sha256(str(repo_root).encode("utf-8")).hexdigest()[:12]
    lock_hash = _node_lock_hash(repo_root)
    return f"detdeps_node_{repo_hash}_{lock_hash}"


def _run_docker(
    docker_cmd: list[str],
    timeout_seconds: int,
) -> tuple[str, str, int, bool]:
    try:
        completed = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=os.environ.copy(),
        )
        return completed.stdout, completed.stderr, completed.returncode, False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="ignore")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="ignore")
        return stdout, stderr + "\nSandbox timed out.", 124, True
    except FileNotFoundError:
        return "", "Docker is not installed or not available in PATH.", 127, False


def _prepare_node_dependencies(repo_root: str, image: str, volume_name: str, timeout_seconds: int) -> SandboxResult:
    prep_command = (
        "if [ -d node_modules ] && [ \"$(ls -A node_modules 2>/dev/null)\" ]; then "
        "echo 'Using cached node_modules from volume.'; "
        "elif [ -f package-lock.json ] || [ -f npm-shrinkwrap.json ]; then "
        "npm ci --no-audit --no-fund; "
        "elif [ -f package.json ]; then "
        "npm install --no-audit --no-fund; "
        "else "
        "echo 'No package.json found; skipping dependency prep.'; "
        "fi"
    )

    prep_cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{repo_root}:/workspace",
        "-v",
        f"{volume_name}:/workspace/node_modules",
        "-w",
        "/workspace",
        image,
        "bash",
        "-lc",
        prep_command,
    ]

    stdout, stderr, exit_code, timed_out = _run_docker(prep_cmd, timeout_seconds=timeout_seconds)
    return SandboxResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        image=image,
        command=prep_command,
        timed_out=timed_out,
    )


def run_in_sandbox(
    repo_path: str,
    command: str | None = None,
    image: str | None = None,
    timeout_seconds: int = 300,
) -> SandboxResult:
    repo_root = str(Path(repo_path).resolve())

    default_image, default_command = detect_repo_runtime(repo_root)
    image = image or default_image
    command = command or default_command
    root_path = Path(repo_root)
    node_volume: str | None = None

    if (root_path / "package.json").exists():
        node_volume = _node_volume_name(root_path)
        prep_result = _prepare_node_dependencies(
            repo_root=repo_root,
            image=image,
            volume_name=node_volume,
            timeout_seconds=timeout_seconds,
        )
        if prep_result.exit_code != 0:
            return SandboxResult(
                stdout=prep_result.stdout,
                stderr=prep_result.stderr,
                exit_code=prep_result.exit_code,
                image=image,
                command=f"[dependency-prep] {prep_result.command}",
                timed_out=prep_result.timed_out,
            )

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--cpus",
        "1",
        "--memory",
        "512m",
        "--pids-limit",
        "256",
        "-v",
        f"{repo_root}:/workspace",
    ]

    if node_volume:
        docker_cmd.extend(["-v", f"{node_volume}:/workspace/node_modules"])

    docker_cmd.extend(
        [
            "-w",
            "/workspace",
            image,
            "bash",
            "-lc",
            command,
        ]
    )

    stdout, stderr, exit_code, timed_out = _run_docker(docker_cmd, timeout_seconds=timeout_seconds)
    return SandboxResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        image=image,
        command=command,
        timed_out=timed_out,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run command in deterministic Docker sandbox")
    parser.add_argument("--repo", required=True, help="Repository path")
    parser.add_argument("--cmd", help="Override command")
    parser.add_argument("--image", help="Override docker image")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--smoke", action="store_true", help="Print compact smoke result")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_in_sandbox(
        repo_path=args.repo,
        command=args.cmd,
        image=args.image,
        timeout_seconds=args.timeout,
    )

    if args.smoke:
        print({"exit_code": result.exit_code, "timed_out": result.timed_out, "image": result.image})
        return

    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
