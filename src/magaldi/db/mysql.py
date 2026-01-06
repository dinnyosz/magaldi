"""MySQL repository implementations for Magaldi.

Implements the repository protocols defined in parser, storage, summarization,
and embedding modules using MySQL as the backing store.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

import mysql.connector
from mysql.connector import MySQLConnection
from mysql.connector.pooling import PooledMySQLConnection

from magaldi.config import MagaldiConfig, get_config
from magaldi.parser.change_detection import DBFileState
from magaldi.parser.code_parser import CodeElement


class MySQLRepository:
    """Base MySQL repository with connection management."""

    def __init__(self, config: MagaldiConfig | None = None):
        self._config = config or get_config()
        self._connection: MySQLConnection | PooledMySQLConnection | None = None

    def _get_connection(self) -> MySQLConnection | PooledMySQLConnection:
        """Get or create database connection."""
        if self._connection is None or not self._connection.is_connected():
            mysql_config = self._config.mysql
            self._connection = mysql.connector.connect(
                host=mysql_config.host,
                port=mysql_config.port,
                user=mysql_config.user,
                password=mysql_config.password,
                database=mysql_config.database,
                autocommit=True,
            )
        return self._connection

    @contextmanager
    def _cursor(self) -> Generator[Any, None, None]:
        """Context manager for database cursor."""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            yield cursor
        finally:
            cursor.close()

    def close(self) -> None:
        """Close database connection."""
        if self._connection and self._connection.is_connected():
            self._connection.close()
            self._connection = None


class MySQLFileStateRepository(MySQLRepository):
    """MySQL implementation of FileStateRepository for change detection."""

    def get_file_states(
        self, scope: str, repository: str, username: str
    ) -> dict[str, DBFileState]:
        """Get all file states for a scope/repo/user."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT relative_path, file_hash, is_deleted
                FROM file_states
                WHERE scope = %s AND repository = %s AND username = %s
                """,
                (scope, repository, username),
            )
            rows = cursor.fetchall()

        return {
            row["relative_path"]: DBFileState(
                relative_path=row["relative_path"],
                file_hash=row["file_hash"],
                is_deleted=bool(row["is_deleted"]),
            )
            for row in rows
        }

    def main_branch_exists(self, scope: str, repository: str) -> bool:
        """Check if main branch has been parsed."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*) as count FROM file_states
                WHERE scope = %s AND repository = %s AND username = 'main'
                """,
                (scope, repository),
            )
            row = cursor.fetchone()
            return row["count"] > 0 if row else False


class MySQLDatabaseRepository(MySQLRepository):
    """MySQL implementation of DatabaseRepository for storage."""

    def store_file_state(
        self,
        scope: str,
        repository: str,
        username: str,
        relative_path: str,
        file_hash: str | None = None,
        language: str = "",
        file_size: int = 0,
        line_count: int = 0,
        is_deleted: bool = False,
        expires_at: datetime | None = None,
    ) -> None:
        """Store or update file state."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO file_states
                    (scope, repository, username, relative_path, file_hash,
                     language, file_size, line_count, is_deleted, expires_at, parsed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON DUPLICATE KEY UPDATE
                    file_hash = VALUES(file_hash),
                    language = VALUES(language),
                    file_size = VALUES(file_size),
                    line_count = VALUES(line_count),
                    is_deleted = VALUES(is_deleted),
                    expires_at = VALUES(expires_at),
                    parsed_at = NOW()
                """,
                (
                    scope, repository, username, relative_path, file_hash,
                    language, file_size, line_count, is_deleted, expires_at,
                ),
            )

    def get_file_state(
        self, scope: str, repository: str, username: str, relative_path: str
    ) -> dict[str, Any] | None:
        """Get file state."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT * FROM file_states
                WHERE scope = %s AND repository = %s AND username = %s AND relative_path = %s
                """,
                (scope, repository, username, relative_path),
            )
            return cursor.fetchone()

    def store_element(self, element: CodeElement) -> None:
        """Store a code element."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO code_elements
                    (element_id, scope, repository, username, relative_path,
                     element_type, name, line_start, line_end, raw_code, signature,
                     docstring, level, parent_id, language, decorators, visibility,
                     is_async, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    raw_code = VALUES(raw_code),
                    signature = VALUES(signature),
                    docstring = VALUES(docstring),
                    decorators = VALUES(decorators),
                    visibility = VALUES(visibility),
                    is_async = VALUES(is_async),
                    line_end = VALUES(line_end),
                    updated_at = NOW()
                """,
                (
                    element.element_id,
                    element.scope,
                    element.repository,
                    element.username,
                    element.relative_path,
                    element.element_type,
                    element.name,
                    element.line_start,
                    element.line_end,
                    element.raw_code,
                    element.signature,
                    element.docstring,
                    element.level,
                    element.parent_id,
                    element.language,
                    json.dumps(element.decorators) if element.decorators else None,
                    element.visibility,
                    element.is_async,
                    None,  # expires_at - set separately for user branches
                ),
            )

    def get_element(self, element_id: str) -> CodeElement | None:
        """Get a code element by ID."""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM code_elements WHERE element_id = %s",
                (element_id,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        return CodeElement(
            element_id=row["element_id"],
            scope=row["scope"],
            repository=row["repository"],
            username=row["username"],
            relative_path=row["relative_path"],
            element_type=row["element_type"],
            name=row["name"],
            language=row["language"] or "",
            line_start=row["line_start"],
            line_end=row["line_end"] or 0,
            raw_code=row["raw_code"] or "",
            signature=row["signature"],
            docstring=row["docstring"],
            decorators=json.loads(row["decorators"]) if row["decorators"] else [],
            is_async=bool(row["is_async"]),
            visibility=row["visibility"] or "public",
            level=row["level"],
            parent_id=row["parent_id"],
        )

    def delete_elements_by_file(
        self, scope: str, repository: str, username: str, relative_path: str
    ) -> int:
        """Delete all elements for a file."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM code_elements
                WHERE scope = %s AND repository = %s AND username = %s AND relative_path = %s
                """,
                (scope, repository, username, relative_path),
            )
            return cursor.rowcount

    def store_summarization_job(
        self,
        element_id: str,
        level: int,
        parent_id: str | None,
        dependencies_met: bool,
        priority: int,
    ) -> None:
        """Create a summarization job."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO summarization_jobs
                    (element_id, level, parent_element_id, dependencies_met, priority, status)
                VALUES (%s, %s, %s, %s, %s, 'pending')
                ON DUPLICATE KEY UPDATE
                    dependencies_met = VALUES(dependencies_met),
                    priority = VALUES(priority),
                    status = 'pending'
                """,
                (element_id, level, parent_id, dependencies_met, priority),
            )

    def get_summarization_job(self, element_id: str) -> dict[str, Any] | None:
        """Get summarization job by element ID."""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM summarization_jobs WHERE element_id = %s",
                (element_id,),
            )
            return cursor.fetchone()

    def store_embedding_job(self, element_id: str) -> None:
        """Create an embedding job."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO embedding_jobs (element_id, status)
                VALUES (%s, 'pending')
                ON DUPLICATE KEY UPDATE status = 'pending'
                """,
                (element_id,),
            )

    def get_embedding_job(self, element_id: str) -> dict[str, Any] | None:
        """Get embedding job by element ID."""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM embedding_jobs WHERE element_id = %s",
                (element_id,),
            )
            return cursor.fetchone()


class MySQLSummarizationJobRepository(MySQLRepository):
    """MySQL implementation of JobRepository for summarization."""

    def add_job(
        self,
        element_id: str,
        level: int,
        parent_id: str | None,
        dependencies_met: bool,
    ) -> None:
        """Add a summarization job."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO summarization_jobs
                    (element_id, level, parent_element_id, dependencies_met, priority, status)
                VALUES (%s, %s, %s, %s, %s, 'pending')
                ON DUPLICATE KEY UPDATE
                    dependencies_met = VALUES(dependencies_met),
                    status = 'pending'
                """,
                (element_id, level, parent_id, dependencies_met, 100 - level),
            )

    def get_job(self, element_id: str) -> dict[str, Any] | None:
        """Get job by element ID."""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM summarization_jobs WHERE element_id = %s",
                (element_id,),
            )
            return cursor.fetchone()

    def claim_pending_jobs(
        self, worker_id: str, batch_size: int
    ) -> list[dict[str, Any]]:
        """Claim pending jobs that have dependencies met."""
        with self._cursor() as cursor:
            # Find and claim jobs atomically
            cursor.execute(
                """
                UPDATE summarization_jobs
                SET status = 'running', worker_id = %s, claimed_at = NOW()
                WHERE status = 'pending' AND dependencies_met = TRUE
                ORDER BY level ASC, priority DESC
                LIMIT %s
                """,
                (worker_id, batch_size),
            )

            # Fetch the claimed jobs
            cursor.execute(
                """
                SELECT * FROM summarization_jobs
                WHERE worker_id = %s AND status = 'running'
                ORDER BY level ASC, priority DESC
                """,
                (worker_id,),
            )
            return cursor.fetchall()

    def mark_completed(self, element_id: str) -> None:
        """Mark job as completed."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                UPDATE summarization_jobs
                SET status = 'completed', completed_at = NOW()
                WHERE element_id = %s
                """,
                (element_id,),
            )
            # Also update the element's summary_status
            cursor.execute(
                """
                UPDATE code_elements
                SET summary_status = 'completed'
                WHERE element_id = %s
                """,
                (element_id,),
            )

    def mark_failed(self, element_id: str, error_message: str) -> None:
        """Mark job as failed."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                UPDATE summarization_jobs
                SET status = 'failed', error_message = %s, completed_at = NOW()
                WHERE element_id = %s
                """,
                (error_message, element_id),
            )
            cursor.execute(
                """
                UPDATE code_elements
                SET summary_status = 'failed'
                WHERE element_id = %s
                """,
                (element_id,),
            )

    def unlock_dependencies(self, parent_element_id: str) -> int:
        """Unlock jobs that depend on this element."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                UPDATE summarization_jobs
                SET dependencies_met = TRUE
                WHERE parent_element_id = %s AND status = 'pending'
                """,
                (parent_element_id,),
            )
            return cursor.rowcount


class MySQLSummaryStore(MySQLRepository):
    """MySQL implementation of SummaryStore."""

    def store_element(self, element: CodeElement) -> None:
        """Store a code element (delegates to MySQLDatabaseRepository)."""
        db_repo = MySQLDatabaseRepository(self._config)
        db_repo.store_element(element)

    def get_element(self, element_id: str) -> CodeElement | None:
        """Get element by ID."""
        db_repo = MySQLDatabaseRepository(self._config)
        return db_repo.get_element(element_id)

    def store_summary(self, element_id: str, summary: str) -> None:
        """Store summary for an element."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                UPDATE code_elements
                SET summary = %s, updated_at = NOW()
                WHERE element_id = %s
                """,
                (summary, element_id),
            )

    def get_summary(self, element_id: str) -> str | None:
        """Get summary for an element."""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT summary FROM code_elements WHERE element_id = %s",
                (element_id,),
            )
            row = cursor.fetchone()
            return row["summary"] if row else None

    def get_parent_summaries(self, element: CodeElement) -> dict[str, str]:
        """Get summaries from parent elements."""
        summaries: dict[str, str] = {}

        with self._cursor() as cursor:
            # Get file summary
            if element.level > 0:
                cursor.execute(
                    """
                    SELECT summary FROM code_elements
                    WHERE scope = %s AND repository = %s AND username = %s
                      AND relative_path = %s AND element_type = 'file'
                    """,
                    (element.scope, element.repository, element.username, element.relative_path),
                )
                row = cursor.fetchone()
                if row and row["summary"]:
                    summaries["file"] = row["summary"]

            # Get class summary for methods
            if element.parent_id and element.element_type in ("method", "function"):
                cursor.execute(
                    """
                    SELECT summary, element_type FROM code_elements
                    WHERE element_id = %s
                    """,
                    (element.parent_id,),
                )
                row = cursor.fetchone()
                if row and row["element_type"] == "class" and row["summary"]:
                    summaries["class"] = row["summary"]

        return summaries


class MySQLEmbeddingJobRepository(MySQLRepository):
    """MySQL implementation of EmbeddingJobRepository."""

    def add_job(self, element_id: str) -> None:
        """Add an embedding job."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO embedding_jobs (element_id, status)
                VALUES (%s, 'pending')
                ON DUPLICATE KEY UPDATE status = 'pending'
                """,
                (element_id,),
            )

    def get_job(self, element_id: str) -> dict[str, Any] | None:
        """Get job by element ID."""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM embedding_jobs WHERE element_id = %s",
                (element_id,),
            )
            return cursor.fetchone()

    def claim_pending_jobs(
        self, worker_id: str, batch_size: int
    ) -> list[dict[str, Any]]:
        """Claim pending jobs."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                UPDATE embedding_jobs
                SET status = 'running', worker_id = %s, claimed_at = NOW()
                WHERE status = 'pending'
                LIMIT %s
                """,
                (worker_id, batch_size),
            )

            cursor.execute(
                """
                SELECT * FROM embedding_jobs
                WHERE worker_id = %s AND status = 'running'
                """,
                (worker_id,),
            )
            return cursor.fetchall()

    def mark_completed(self, element_id: str) -> None:
        """Mark job as completed."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                UPDATE embedding_jobs
                SET status = 'completed', completed_at = NOW()
                WHERE element_id = %s
                """,
                (element_id,),
            )
            cursor.execute(
                """
                UPDATE code_elements
                SET embedding_status = 'completed'
                WHERE element_id = %s
                """,
                (element_id,),
            )

    def mark_failed(self, element_id: str, error_message: str) -> None:
        """Mark job as failed."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                UPDATE embedding_jobs
                SET status = 'failed', error_message = %s, completed_at = NOW()
                WHERE element_id = %s
                """,
                (error_message, element_id),
            )
            cursor.execute(
                """
                UPDATE code_elements
                SET embedding_status = 'failed'
                WHERE element_id = %s
                """,
                (element_id,),
            )


class MySQLEmbeddingStore(MySQLRepository):
    """MySQL implementation of EmbeddingStore (summaries only, vectors go to ES)."""

    def store_element(self, element: CodeElement) -> None:
        """Store a code element."""
        db_repo = MySQLDatabaseRepository(self._config)
        db_repo.store_element(element)

    def get_element(self, element_id: str) -> CodeElement | None:
        """Get element by ID."""
        db_repo = MySQLDatabaseRepository(self._config)
        return db_repo.get_element(element_id)

    def store_summary(self, element_id: str, summary: str) -> None:
        """Store summary for an element."""
        summary_store = MySQLSummaryStore(self._config)
        summary_store.store_summary(element_id, summary)

    def get_summary(self, element_id: str) -> str | None:
        """Get summary for an element."""
        summary_store = MySQLSummaryStore(self._config)
        return summary_store.get_summary(element_id)

    def store_embedding(self, element_id: str, embedding: list[float]) -> None:
        """Store embedding vector - handled by Elasticsearch, not MySQL."""
        # This is a no-op for MySQL; embeddings are stored in Elasticsearch
        pass

    def get_embedding(self, element_id: str) -> list[float] | None:
        """Get embedding vector - handled by Elasticsearch, not MySQL."""
        return None

    def get_file_summary(self, element: CodeElement) -> str | None:
        """Get file summary for context enrichment."""
        summary_store = MySQLSummaryStore(self._config)
        summaries = summary_store.get_parent_summaries(element)
        return summaries.get("file")

    def get_class_summary(self, element: CodeElement) -> str | None:
        """Get class summary for context enrichment."""
        summary_store = MySQLSummaryStore(self._config)
        summaries = summary_store.get_parent_summaries(element)
        return summaries.get("class")
