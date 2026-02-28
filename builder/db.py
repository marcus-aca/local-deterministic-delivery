"""Database helpers and schema initialization for deterministic local delivery."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

load_dotenv()


RESUMABLE_STATUSES = ("planning", "in_progress", "retrying", "running")
FAILED_STATUS = "failed"


@dataclass(frozen=True)
class DBSettings:
    host: str
    port: int
    dbname: str
    user: str
    password: str

    @classmethod
    def from_env(cls) -> "DBSettings":
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "builder"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        )


def get_connection(autocommit: bool = False):
    settings = DBSettings.from_env()
    conn = psycopg2.connect(
        host=settings.host,
        port=settings.port,
        dbname=settings.dbname,
        user=settings.user,
        password=settings.password,
    )
    conn.autocommit = autocommit
    return conn


def init_schema() -> None:
    ddl_statements = [
        "CREATE EXTENSION IF NOT EXISTS vector;",
        """
        CREATE TABLE IF NOT EXISTS code_chunks (
            id SERIAL PRIMARY KEY,
            repo_path TEXT NOT NULL,
            file_path TEXT NOT NULL,
            chunk_index INT NOT NULL,
            content TEXT NOT NULL,
            embedding VECTOR(384) NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS files (
            id SERIAL PRIMARY KEY,
            repo_path TEXT NOT NULL,
            file_path TEXT NOT NULL,
            summary TEXT,
            content_hash TEXT NOT NULL,
            last_modified TIMESTAMP NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS runs (
            id UUID PRIMARY KEY,
            repo_path TEXT NOT NULL,
            user_request TEXT NOT NULL,
            mode TEXT,
            status TEXT NOT NULL,
            current_step INT DEFAULT 0,
            retry_count INT DEFAULT 0,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS plans (
            id SERIAL PRIMARY KEY,
            run_id UUID REFERENCES runs(id) ON DELETE CASCADE,
            step_number INT NOT NULL,
            description TEXT NOT NULL,
            status TEXT DEFAULT 'pending'
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS execution_logs (
            id SERIAL PRIMARY KEY,
            run_id UUID REFERENCES runs(id) ON DELETE CASCADE,
            step_number INT NOT NULL,
            stdout TEXT,
            stderr TEXT,
            exit_code INT,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_files_repo_file
            ON files (repo_path, file_path);
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_code_chunks_repo_file_chunk
            ON code_chunks (repo_path, file_path, chunk_index);
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_plans_run_step
            ON plans (run_id, step_number);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_code_chunks_repo_file
            ON code_chunks (repo_path, file_path);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_files_repo_file
            ON files (repo_path, file_path);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_runs_status_updated
            ON runs (status, updated_at DESC);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_plans_run_id
            ON plans (run_id);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_execution_logs_run_id
            ON execution_logs (run_id);
        """,
    ]

    with get_connection() as conn:
        with conn.cursor() as cur:
            for statement in ddl_statements:
                cur.execute(statement)
        conn.commit()


def smoke_check() -> dict[str, Any]:
    table_names = ["code_chunks", "files", "runs", "plans", "execution_logs"]
    result: dict[str, Any] = {}

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT extname
                FROM pg_extension
                WHERE extname = 'vector';
                """
            )
            result["vector_extension"] = cur.fetchone() is not None

            for table in table_names:
                cur.execute(sql.SQL("SELECT COUNT(*) AS count FROM {};").format(sql.Identifier(table)))
                row = cur.fetchone()
                result[f"{table}_rows"] = int(row["count"]) if row else 0

    return result


def vector_literal(vector: Iterable[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"


def get_existing_file_hash(repo_path: str, file_path: str) -> str | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT content_hash
                FROM files
                WHERE repo_path = %s AND file_path = %s;
                """,
                (repo_path, file_path),
            )
            row = cur.fetchone()
            return row[0] if row else None


def upsert_file_metadata(
    repo_path: str,
    file_path: str,
    summary: str,
    content_hash: str,
    last_modified: datetime,
) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO files (repo_path, file_path, summary, content_hash, last_modified)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (repo_path, file_path)
                DO UPDATE SET
                    summary = EXCLUDED.summary,
                    content_hash = EXCLUDED.content_hash,
                    last_modified = EXCLUDED.last_modified;
                """,
                (repo_path, file_path, summary, content_hash, last_modified),
            )
        conn.commit()


def replace_code_chunks(
    repo_path: str,
    file_path: str,
    chunks_with_embeddings: list[tuple[int, str, list[float]]],
) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM code_chunks WHERE repo_path = %s AND file_path = %s;",
                (repo_path, file_path),
            )

            if chunks_with_embeddings:
                values = [
                    (repo_path, file_path, chunk_index, content, vector_literal(embedding))
                    for chunk_index, content, embedding in chunks_with_embeddings
                ]
                cur.executemany(
                    """
                    INSERT INTO code_chunks (repo_path, file_path, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector);
                    """,
                    values,
                )
        conn.commit()


def search_code_chunks(repo_path: str, query_embedding: list[float], limit: int = 8) -> list[dict[str, Any]]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT file_path, chunk_index, content,
                       embedding <=> %s::vector AS distance
                FROM code_chunks
                WHERE repo_path = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (vector_literal(query_embedding), repo_path, vector_literal(query_embedding), limit),
            )
            return [dict(row) for row in cur.fetchall()]


def list_indexed_files(repo_path: str) -> list[str]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT file_path
                FROM files
                WHERE repo_path = %s
                ORDER BY file_path;
                """,
                (repo_path,),
            )
            return [row[0] for row in cur.fetchall()]


def delete_indexed_files(repo_path: str, file_paths: list[str]) -> int:
    if not file_paths:
        return 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM code_chunks
                WHERE repo_path = %s AND file_path = ANY(%s);
                """,
                (repo_path, file_paths),
            )
            cur.execute(
                """
                DELETE FROM files
                WHERE repo_path = %s AND file_path = ANY(%s);
                """,
                (repo_path, file_paths),
            )
            deleted = cur.rowcount
        conn.commit()
    return deleted


def create_run(run_id: str, repo_path: str, user_request: str, status: str = "planning") -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (id, repo_path, user_request, status)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING;
                """,
                (run_id, repo_path, user_request, status),
            )
        conn.commit()


def update_run(run_id: str, **fields: Any) -> None:
    if not fields:
        return

    fields["updated_at"] = datetime.utcnow()

    with get_connection() as conn:
        with conn.cursor() as cur:
            assignments = [sql.SQL("{} = %s").format(sql.Identifier(key)) for key in fields]
            values = list(fields.values()) + [run_id]
            query = sql.SQL("UPDATE runs SET {} WHERE id = %s;").format(sql.SQL(", ").join(assignments))
            cur.execute(query, values)
        conn.commit()


def get_run(run_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM runs WHERE id = %s;", (run_id,))
            row = cur.fetchone()
            return dict(row) if row else None


def get_latest_resumable_run(repo_path: str | None = None, include_failed: bool = False) -> dict[str, Any] | None:
    statuses = list(RESUMABLE_STATUSES)
    if include_failed:
        statuses.append(FAILED_STATUS)

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if repo_path:
                cur.execute(
                    """
                    SELECT *
                    FROM runs
                    WHERE repo_path = %s
                      AND status = ANY(%s)
                    ORDER BY updated_at DESC
                    LIMIT 1;
                    """,
                    (repo_path, statuses),
                )
            else:
                cur.execute(
                    """
                    SELECT *
                    FROM runs
                    WHERE status = ANY(%s)
                    ORDER BY updated_at DESC
                    LIMIT 1;
                    """,
                    (statuses,),
                )
            row = cur.fetchone()
            return dict(row) if row else None


def insert_plan_steps(run_id: str, steps: list[str]) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            for idx, description in enumerate(steps, start=1):
                cur.execute(
                    """
                    INSERT INTO plans (run_id, step_number, description, status)
                    VALUES (%s, %s, %s, 'pending')
                    ON CONFLICT (run_id, step_number)
                    DO UPDATE SET description = EXCLUDED.description;
                    """,
                    (run_id, idx, description),
                )
        conn.commit()


def get_plan_steps(run_id: str) -> list[dict[str, Any]]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT run_id, step_number, description, status
                FROM plans
                WHERE run_id = %s
                ORDER BY step_number;
                """,
                (run_id,),
            )
            return [dict(row) for row in cur.fetchall()]


def update_plan_status(run_id: str, step_number: int, status: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE plans
                SET status = %s
                WHERE run_id = %s AND step_number = %s;
                """,
                (status, run_id, step_number),
            )
        conn.commit()


def insert_execution_log(
    run_id: str,
    step_number: int,
    stdout: str,
    stderr: str,
    exit_code: int,
) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO execution_logs (run_id, step_number, stdout, stderr, exit_code)
                VALUES (%s, %s, %s, %s, %s);
                """,
                (run_id, step_number, stdout, stderr, exit_code),
            )
        conn.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Database utility for deterministic local delivery")
    parser.add_argument("--init", action="store_true", help="Initialize database schema")
    parser.add_argument("--smoke", action="store_true", help="Run schema smoke check")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.init:
        init_schema()
        print("Schema initialization complete.")

    if args.smoke:
        result = smoke_check()
        print(result)

    if not args.init and not args.smoke:
        print("No action selected. Use --init and/or --smoke.")


if __name__ == "__main__":
    main()
