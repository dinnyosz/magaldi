"""Job tracking utilities for the processor.

Contains:
- RedisJobTracker: Track processing jobs in Redis for dashboard monitoring
- SummaryCache: In-memory cache for element summaries during processing
"""

from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from magaldi_core.code_parser import CodeElement
    from shared.config import MagaldiConfig

from shared.db.redis import (
    JobType,
    KeyType,
    RedisEmbeddingJobRepository,
    RedisSummarizationJobRepository,
    _make_key,
)


class RedisJobTracker:
    """Track processing jobs in Redis for dashboard monitoring.

    This writes job status to Redis so the dashboard can show queue activity
    during synchronous processing.
    """

    def __init__(
        self,
        config: MagaldiConfig,
        scope: str,
        repository: str,
        username: str,
        should_embed_fn: Callable[[CodeElement], bool],
    ) -> None:
        self._scope = scope
        self._repository = repository
        self._username = username
        self._sum_repo = RedisSummarizationJobRepository(config)
        self._emb_repo = RedisEmbeddingJobRepository(config)
        self._lock = threading.Lock()
        self._should_embed = should_embed_fn

    def clear_queues(self) -> None:
        """Clear all Redis queue keys for this scope/repository/username."""
        client = self._sum_repo._get_client()

        # Keys to delete for summarization and embedding
        keys_to_delete = [
            _make_key(JobType.SUMMARIZATION, KeyType.JOBS, self._scope, self._repository, self._username),
            _make_key(JobType.SUMMARIZATION, KeyType.RUNNING, self._scope, self._repository, self._username),
            _make_key(JobType.SUMMARIZATION, KeyType.QUEUE, self._scope, self._repository, self._username),
            _make_key(JobType.EMBEDDING, KeyType.JOBS, self._scope, self._repository, self._username),
            _make_key(JobType.EMBEDDING, KeyType.RUNNING, self._scope, self._repository, self._username),
            _make_key(JobType.EMBEDDING, KeyType.QUEUE, self._scope, self._repository, self._username),
        ]

        for key in keys_to_delete:
            client.delete(key)

    def add_pending_jobs(self, elements: list[CodeElement]) -> None:
        """Add all elements as pending jobs to Redis."""
        for element in elements:
            # Add summarization job (all elements get summarized)
            self._sum_repo.add_job(
                element_id=element.element_id,
                scope=self._scope,
                repository=self._repository,
                username=self._username,
                level=element.level,
                parent_id=element.parent_id,
                dependencies_met=True,  # We handle dependencies in processor
                priority=100 - element.level,
            )
            # Add embedding job (only for embeddable elements)
            if self._should_embed(element):
                self._emb_repo.add_job(
                    element_id=element.element_id,
                    scope=self._scope,
                    repository=self._repository,
                    username=self._username,
                )

    def mark_running(self, element_id: str, was_embedded: bool = True) -> None:
        """Mark element as running in Redis."""
        with self._lock:
            # Update job status to running and add to running set
            client = self._sum_repo._get_client()
            jobs_key = _make_key(JobType.SUMMARIZATION, KeyType.JOBS, self._scope, self._repository, self._username)
            running_key = _make_key(JobType.SUMMARIZATION, KeyType.RUNNING, self._scope, self._repository, self._username)

            # Update status in job hash
            job_data = client.hget(jobs_key, element_id)
            if job_data:
                job = json.loads(job_data)
                job["status"] = "running"
                client.hset(jobs_key, element_id, json.dumps(job))
                client.sadd(running_key, element_id)

            if was_embedded:
                emb_jobs_key = _make_key(JobType.EMBEDDING, KeyType.JOBS, self._scope, self._repository, self._username)
                emb_running_key = _make_key(JobType.EMBEDDING, KeyType.RUNNING, self._scope, self._repository, self._username)
                emb_data = client.hget(emb_jobs_key, element_id)
                if emb_data:
                    emb_job = json.loads(emb_data)
                    emb_job["status"] = "running"
                    client.hset(emb_jobs_key, element_id, json.dumps(emb_job))
                    client.sadd(emb_running_key, element_id)

    def mark_completed(self, element_id: str, was_embedded: bool = True) -> None:
        """Mark element as completed in Redis."""
        with self._lock:
            self._sum_repo.mark_completed(
                element_id, self._scope, self._repository, self._username
            )
            if was_embedded:
                self._emb_repo.mark_completed(
                    element_id, self._scope, self._repository, self._username
                )

    def mark_failed(self, element_id: str, error: str, was_embedded: bool = True) -> None:
        """Mark element as failed in Redis."""
        with self._lock:
            self._sum_repo.mark_failed(
                element_id, self._scope, self._repository, self._username, error
            )
            if was_embedded:
                self._emb_repo.mark_failed(
                    element_id, self._scope, self._repository, self._username, error
                )

    def close(self) -> None:
        """Close Redis connections."""
        self._sum_repo.close()
        self._emb_repo.close()


class SummaryCache:
    """In-memory cache that acts as EmbeddingStore for build_embedding_text.

    This adapter allows us to use build_embedding_text without requiring
    elements to be stored in ES first. Thread-safe for parallel processing.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._elements: dict[str, CodeElement] = {}
        self._summaries: dict[str, str] = {}

    def add_element(self, element: CodeElement) -> None:
        """Add element to cache."""
        self._elements[element.element_id] = element

    def add_summary(self, element_id: str, summary: str) -> None:
        """Add summary to cache."""
        with self._lock:
            self._summaries[element_id] = summary

    def get_element(self, element_id: str) -> CodeElement | None:
        """Get element from cache."""
        return self._elements.get(element_id)

    def get_summary(self, element_id: str) -> str | None:
        """Get summary from cache."""
        with self._lock:
            return self._summaries.get(element_id)

    def get_file_summary(self, element: CodeElement) -> str | None:
        """Get file summary for an element."""
        # Find file element for this path
        for eid, elem in self._elements.items():
            if (
                elem.scope == element.scope
                and elem.repository == element.repository
                and elem.username == element.username
                and elem.relative_path == element.relative_path
                and elem.element_type == "file"
            ):
                return self.get_summary(eid)
        return None

    def get_class_summary(self, element: CodeElement) -> str | None:
        """Get class summary for an element (via parent_id)."""
        if element.parent_id:
            parent = self.get_element(element.parent_id)
            if parent and parent.element_type == "class":
                return self.get_summary(element.parent_id)
        return None

    def get_parent_summaries(self, element: CodeElement) -> dict[str, str]:
        """Get parent summaries for context."""
        summaries: dict[str, str] = {}

        # Get file summary
        file_summary = self.get_file_summary(element)
        if file_summary:
            summaries["file"] = file_summary

        # Get class summary if method
        if element.element_type == "method":
            class_summary = self.get_class_summary(element)
            if class_summary:
                summaries["class"] = class_summary

        return summaries
