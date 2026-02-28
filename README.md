# Local Deterministic Delivery

CLI framework that takes a natural-language coding request, plans work, retrieves relevant repo context with embeddings + pgvector, generates file edits, validates in a Docker sandbox, and persists full run state in Postgres for resume/retry.

## What It Does

- Accepts a feature request (inline text, `--request-file`, or `<repo>/design.md`).
- Chooses run mode:
  - `modify_existing`: edit an existing repo.
  - `scaffold_new`: create/initialize code in an empty/new path.
- Builds and uses a vector index of repository code.
- Generates an ordered plan, executes step-by-step, and validates each step in sandboxed Docker.
- On sandbox failure, invokes debugger fixes and retries.
- Persists runs/plans/logs so interrupted runs can resume deterministically.

Core modules live in [`builder/`](builder).

## Architecture

Flow:

`main.py` -> `orchestrator.py` (LangGraph) -> agents (`strategy`, `planner`, `coder`, `debugger`) -> `indexer.py` / `retrieval.py` -> `sandbox.py` -> `db.py` (Postgres + pgvector)

Persistent tables:

- `code_chunks`: chunk content + vector embeddings (`VECTOR(384)`).
- `files`: file metadata (`summary`, `content_hash`, `last_modified`).
- `runs`: run status, mode, current step, retries.
- `plans`: per-step plan rows and status.
- `execution_logs`: sandbox stdout/stderr/exit code per step.

## Prerequisites

- Python 3.11+ recommended.
- Docker (for pgvector DB and sandbox execution).
- Codex CLI binary available on PATH (default command: `codex`).

## Setup

1. Start pgvector Postgres:

```bash
cd builder
docker compose up -d
```

2. Create Python env and install deps:

```bash
cd builder
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Configure env:

```bash
cd builder
cp .env.example .env
```

4. Initialize schema:

```bash
cd builder
python db.py --init
```

Optional check:

```bash
cd builder
python db.py --smoke
```

## Running the Framework

From `builder/`:

- Modify existing repo:

```bash
python main.py --repo /path/to/repo "Add JWT auth middleware and tests"
```

- Scaffold in a new directory:

```bash
python main.py --mode scaffold_new --scaffold-path /path/to/new-service "Create a REST API service"
```

- Use request file:

```bash
python main.py --repo /path/to/repo --request-file /path/to/request.md
```

- Resume behavior:
  - Default `--resume auto`: prompts in TTY if resumable run exists.
  - Force resume: `--resume yes`
  - Force new run: `--resume no`
  - Resume explicit run: `--run-id <uuid>`

Useful flags:

- `--max-retries` (default from `MAX_RETRIES`, usually `2`)
- `--graph-debug` / `--no-graph-debug` for structured transition events

## Embedding + Retrieval (Indexing and Best-Match Candidate Selection)

This is the path used to index repos and fetch edit candidates.

### 1) Indexing (`indexer.py`)

- Walks repo recursively while ignoring hidden dirs/files and common non-source/binary targets.
- Reads file text and computes `sha256` `content_hash`.
- Skips unchanged files by comparing against `files.content_hash`.
- Chunks text with sliding windows:
  - `CHUNK_SIZE` (default `1200`)
  - `CHUNK_OVERLAP` (default `200`)
- Embeds each chunk to 384 dims:
  - Primary: `sentence-transformers/all-MiniLM-L6-v2` with normalized embeddings.
  - Fallback: deterministic hash-based embedding (offline-safe).
- Stores:
  - file metadata in `files`
  - chunk rows in `code_chunks` (replaced per changed file).
- Cleans stale indexed files removed from repo.

### 2) Query-to-Context Retrieval (`retrieval.py`)

- Rewrites the user query via agent call (`rewrite_query`) for better code search.
- Embeds rewritten query with the same embedding backend.
- Runs top-k vector search in Postgres (`embedding <=> query_vector`, lower distance = closer match).
- Starts selected set from top matching chunks/files.
- Always appends key files when present (for project grounding), e.g. `package.json`, `requirements.txt`, entrypoints.
- Expands context by 1-hop dependency discovery using import/require regex patterns.
- Deduplicates and caps context set (`cap`, default `20`).
- Returns structured payload:
  - `original_query`
  - `rewritten_query`
  - `embedding_mode`
  - `files`: snippets by file + synthetic `__TOP_LEVEL_MAP__`.

### 3) How “best match candidate for editing” is chosen in execution

Per plan step, orchestrator calls:

- `retrieve_context(repo_path, user_query=f"{user_request}\nPlan step: {step_description}")`

Then the coder agent receives that retrieved context and produces full-file outputs in strict protocol:

`FILE: relative/path` + full content.

In practice, the best edit candidates are the files surfaced by vector similarity + key-file inclusion + dependency expansion, constrained to the capped context list the coder sees.

### 4) Continuous index freshness

After every coder/debugger file write, orchestrator re-indexes changed files (`index_files`) so subsequent steps retrieve against updated code, not stale embeddings.

## Important Runtime Logic

- Strategy selection (`agents/strategy.py`):
  - LLM JSON decision with fallback heuristics (empty repo/new-project phrasing => scaffold mode).
- Plan generation (`agents/planner.py`):
  - Produces 2-7 actionable steps; persisted in `plans`.
- Deterministic step loop (`orchestrator.py`):
  - Mark step in-progress -> retrieve -> code -> sandbox validate -> retry/fail/complete.
- Sandbox (`sandbox.py`):
  - Runtime detection (Node vs Python).
  - Docker constraints: `--network none`, CPU/memory/pid limits.
  - Node dependency cache volume keyed by repo + lockfile hash.
- Debug retries:
  - On non-zero sandbox exit, debugger agent proposes fixes and reruns sandbox up to `max_retries`.
- Resume:
  - `runs.current_step` + plan status enable continuation without replaying completed steps.
- Prompt diagnostics:
  - Saves prompt snapshots on parse/CLI failures (`.codex_last_prompt.txt`, `.codex_last_plan.json` by default).

## Standalone Utilities

From `builder/`:

- Index repo manually:

```bash
python indexer.py --repo /path/to/repo
```

- Retrieval smoke:

```bash
python retrieval.py --repo /path/to/repo --query "where auth middleware is wired" --smoke
```

- Sandbox smoke:

```bash
python sandbox.py --repo /path/to/repo --smoke
```

## Key Environment Variables

See [`builder/.env.example`](builder/.env.example). Most important:

- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
- `EMBEDDING_MODEL`
- `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `MAX_RETRIES`
- `CODEX_CLI_BIN`, `CODEX_CLI_TIMEOUT_SECONDS`, `CODEX_CLI_EXTRA_FLAGS`
- `CODEX_PROMPT_DEBUG_FILE`, `PLAN_DEBUG_FILE`

## Notes

- “Deterministic” here is mostly about persisted state transitions, resumability, and constrained execution. Agent text generation itself can still vary.
- First embedding call may download model artifacts if not cached locally.
