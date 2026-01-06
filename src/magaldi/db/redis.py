"""Redis repository implementation for Magaldi.

Handles job queues for summarization and embedding workers.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

import redis

from magaldi.config import MagaldiConfig, get_config


# Redis key prefixes
SUMMARIZATION_QUEUE = "magaldi:summarization:queue"
SUMMARIZATION_RUNNING = "magaldi:summarization:running"
SUMMARIZATION_JOBS = "magaldi:summarization:jobs"

EMBEDDING_QUEUE = "magaldi:embedding:queue"
EMBEDDING_RUNNING = "magaldi:embedding:running"
EMBEDDING_JOBS = "magaldi:embedding:jobs"


class RedisRepository:
    """Base Redis repository with connection management."""

    def __init__(self, config: MagaldiConfig | None = None):
        self._config = config or get_config()
        self._client: redis.Redis | None = None

    def _get_client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            redis_config = self._config.redis
            self._client = redis.Redis(
                host=redis_config.host,
                port=redis_config.port,
                db=redis_config.db,
                password=redis_config.password,
                decode_responses=True,
            )
        return self._client

    def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            self._client.close()
            self._client = None


class RedisSummarizationJobRepository(RedisRepository):
    """Redis-based summarization job queue."""

    def add_job(
        self,
        element_id: str,
        level: int,
        parent_id: str | None,
        dependencies_met: bool = False,
        priority: int = 0,
    ) -> None:
        """Add a summarization job to the queue.

        Args:
            element_id: Element to summarize.
            level: Hierarchy level (0=file, 1=class, 2=function).
            parent_id: Parent element ID (for dependency tracking).
            dependencies_met: Whether dependencies are satisfied.
            priority: Job priority (higher = more urgent).
        """
        client = self._get_client()

        job_data = {
            "element_id": element_id,
            "level": level,
            "parent_id": parent_id,
            "dependencies_met": dependencies_met,
            "status": "pending",
            "priority": priority,
            "created_at": datetime.now().isoformat(),
            "worker_id": None,
            "claimed_at": None,
            "completed_at": None,
            "error_message": None,
            "retry_count": 0,
        }

        # Store job data
        client.hset(SUMMARIZATION_JOBS, element_id, json.dumps(job_data))

        # Add to queue if dependencies are met (sorted by level desc, then priority)
        if dependencies_met:
            # Score: level * 1000 + priority (process higher levels first)
            score = level * 1000 + priority
            client.zadd(SUMMARIZATION_QUEUE, {element_id: score})

    def get_job(self, element_id: str) -> dict[str, Any] | None:
        """Get job data by element ID."""
        client = self._get_client()
        data = client.hget(SUMMARIZATION_JOBS, element_id)
        if data:
            return json.loads(data)
        return None

    def claim_pending_jobs(
        self, worker_id: str, batch_size: int = 10
    ) -> list[dict[str, Any]]:
        """Claim pending jobs for processing.

        Args:
            worker_id: ID of the claiming worker.
            batch_size: Maximum jobs to claim.

        Returns:
            List of claimed job data.
        """
        client = self._get_client()
        claimed = []

        # Get highest priority jobs (highest scores first)
        element_ids = client.zrevrange(SUMMARIZATION_QUEUE, 0, batch_size - 1)

        for element_id in element_ids:
            # Atomically move from queue to running
            removed = client.zrem(SUMMARIZATION_QUEUE, element_id)
            if removed:
                # Update job status
                job_data = self.get_job(element_id)
                if job_data:
                    job_data["status"] = "running"
                    job_data["worker_id"] = worker_id
                    job_data["claimed_at"] = datetime.now().isoformat()
                    client.hset(SUMMARIZATION_JOBS, element_id, json.dumps(job_data))
                    client.sadd(SUMMARIZATION_RUNNING, element_id)
                    claimed.append(job_data)

        return claimed

    def mark_completed(self, element_id: str) -> None:
        """Mark a job as completed."""
        client = self._get_client()

        job_data = self.get_job(element_id)
        if job_data:
            job_data["status"] = "completed"
            job_data["completed_at"] = datetime.now().isoformat()
            client.hset(SUMMARIZATION_JOBS, element_id, json.dumps(job_data))
            client.srem(SUMMARIZATION_RUNNING, element_id)

            # Unlock dependent jobs
            self.unlock_dependencies(element_id)

    def mark_failed(self, element_id: str, error_message: str) -> None:
        """Mark a job as failed."""
        client = self._get_client()

        job_data = self.get_job(element_id)
        if job_data:
            job_data["status"] = "failed"
            job_data["error_message"] = error_message
            job_data["retry_count"] = job_data.get("retry_count", 0) + 1
            client.hset(SUMMARIZATION_JOBS, element_id, json.dumps(job_data))
            client.srem(SUMMARIZATION_RUNNING, element_id)

    def unlock_dependencies(self, parent_id: str) -> int:
        """Unlock jobs that depend on the completed parent.

        Args:
            parent_id: ID of completed parent element.

        Returns:
            Number of jobs unlocked.
        """
        client = self._get_client()
        unlocked = 0

        # Scan all jobs to find children
        all_jobs = client.hgetall(SUMMARIZATION_JOBS)
        for element_id, data in all_jobs.items():
            job_data = json.loads(data)
            if (
                job_data.get("parent_id") == parent_id
                and not job_data.get("dependencies_met")
                and job_data.get("status") == "pending"
            ):
                # Unlock this job
                job_data["dependencies_met"] = True
                client.hset(SUMMARIZATION_JOBS, element_id, json.dumps(job_data))

                # Add to queue
                score = job_data["level"] * 1000 + job_data.get("priority", 0)
                client.zadd(SUMMARIZATION_QUEUE, {element_id: score})
                unlocked += 1

        return unlocked


class RedisEmbeddingJobRepository(RedisRepository):
    """Redis-based embedding job queue."""

    def add_job(self, element_id: str) -> None:
        """Add an embedding job to the queue."""
        client = self._get_client()

        job_data = {
            "element_id": element_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "worker_id": None,
            "claimed_at": None,
            "completed_at": None,
            "error_message": None,
            "retry_count": 0,
        }

        # Store job data
        client.hset(EMBEDDING_JOBS, element_id, json.dumps(job_data))

        # Add to queue (FIFO order using timestamp as score)
        client.zadd(EMBEDDING_QUEUE, {element_id: time.time()})

    def get_job(self, element_id: str) -> dict[str, Any] | None:
        """Get job data by element ID."""
        client = self._get_client()
        data = client.hget(EMBEDDING_JOBS, element_id)
        if data:
            return json.loads(data)
        return None

    def claim_pending_jobs(
        self, worker_id: str, batch_size: int = 10
    ) -> list[dict[str, Any]]:
        """Claim pending jobs for processing."""
        client = self._get_client()
        claimed = []

        # Get oldest jobs first (FIFO)
        element_ids = client.zrange(EMBEDDING_QUEUE, 0, batch_size - 1)

        for element_id in element_ids:
            removed = client.zrem(EMBEDDING_QUEUE, element_id)
            if removed:
                job_data = self.get_job(element_id)
                if job_data:
                    job_data["status"] = "running"
                    job_data["worker_id"] = worker_id
                    job_data["claimed_at"] = datetime.now().isoformat()
                    client.hset(EMBEDDING_JOBS, element_id, json.dumps(job_data))
                    client.sadd(EMBEDDING_RUNNING, element_id)
                    claimed.append(job_data)

        return claimed

    def mark_completed(self, element_id: str) -> None:
        """Mark a job as completed."""
        client = self._get_client()

        job_data = self.get_job(element_id)
        if job_data:
            job_data["status"] = "completed"
            job_data["completed_at"] = datetime.now().isoformat()
            client.hset(EMBEDDING_JOBS, element_id, json.dumps(job_data))
            client.srem(EMBEDDING_RUNNING, element_id)

    def mark_failed(self, element_id: str, error_message: str) -> None:
        """Mark a job as failed."""
        client = self._get_client()

        job_data = self.get_job(element_id)
        if job_data:
            job_data["status"] = "failed"
            job_data["error_message"] = error_message
            job_data["retry_count"] = job_data.get("retry_count", 0) + 1
            client.hset(EMBEDDING_JOBS, element_id, json.dumps(job_data))
            client.srem(EMBEDDING_RUNNING, element_id)


class RedisSummaryStore(RedisRepository):
    """Redis-based summary store (for quick access during summarization)."""

    SUMMARIES_KEY = "magaldi:summaries"
    ELEMENTS_KEY = "magaldi:elements"

    def store_element(self, element: Any) -> None:
        """Store element data for later retrieval."""
        from magaldi.parser.code_parser import CodeElement

        if isinstance(element, CodeElement):
            client = self._get_client()
            client.hset(
                self.ELEMENTS_KEY,
                element.element_id,
                json.dumps({
                    "element_id": element.element_id,
                    "element_type": element.element_type,
                    "name": element.name,
                    "raw_code": element.raw_code,
                    "docstring": element.docstring,
                    "parent_id": element.parent_id,
                    "level": element.level,
                }),
            )

    def get_element(self, element_id: str) -> Any | None:
        """Get element data."""
        from magaldi.parser.code_parser import CodeElement

        client = self._get_client()
        data = client.hget(self.ELEMENTS_KEY, element_id)
        if data:
            elem_data = json.loads(data)
            # Return minimal CodeElement for summarization
            return CodeElement(
                element_id=elem_data["element_id"],
                scope="",
                repository="",
                username="",
                relative_path="",
                element_type=elem_data["element_type"],
                name=elem_data["name"],
                language="",
                line_start=0,
                raw_code=elem_data.get("raw_code"),
                docstring=elem_data.get("docstring"),
                parent_id=elem_data.get("parent_id"),
                level=elem_data.get("level", 0),
            )
        return None

    def store_summary(self, element_id: str, summary: str) -> None:
        """Store a summary."""
        client = self._get_client()
        client.hset(self.SUMMARIES_KEY, element_id, summary)

    def get_summary(self, element_id: str) -> str | None:
        """Get a summary."""
        client = self._get_client()
        return client.hget(self.SUMMARIES_KEY, element_id)

    def get_parent_summaries(self, element: Any) -> dict[str, str]:
        """Get parent summaries for context."""
        summaries: dict[str, str] = {}

        # Get element data
        client = self._get_client()
        elem_data = client.hget(self.ELEMENTS_KEY, element.element_id)
        if not elem_data:
            return summaries

        parsed = json.loads(elem_data)
        parent_id = parsed.get("parent_id")

        if parent_id:
            parent_data = client.hget(self.ELEMENTS_KEY, parent_id)
            if parent_data:
                parent = json.loads(parent_data)
                parent_summary = self.get_summary(parent_id)
                if parent_summary:
                    summaries[parent["element_type"]] = parent_summary

        return summaries
