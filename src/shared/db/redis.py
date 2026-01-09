"""Redis repository implementation for Magaldi.

Handles job queues for summarization and embedding workers.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

import redis

from shared.config import MagaldiConfig, get_config


# Redis key prefixes with scope/repository/username isolation
# Format: magaldi:{job_type}:{key_type}:{scope}:{repository}:{username}
SUMMARIZATION_QUEUE = "magaldi:summarization:queue:{scope}:{repository}:{username}"
SUMMARIZATION_RUNNING = "magaldi:summarization:running:{scope}:{repository}:{username}"
SUMMARIZATION_JOBS = "magaldi:summarization:jobs:{scope}:{repository}:{username}"

EMBEDDING_QUEUE = "magaldi:embedding:queue:{scope}:{repository}:{username}"
EMBEDDING_RUNNING = "magaldi:embedding:running:{scope}:{repository}:{username}"
EMBEDDING_JOBS = "magaldi:embedding:jobs:{scope}:{repository}:{username}"

FEATURE_QUEUE = "magaldi:feature:queue:{scope}:{repository}:{username}"
FEATURE_RUNNING = "magaldi:feature:running:{scope}:{repository}:{username}"
FEATURE_JOBS = "magaldi:feature:jobs:{scope}:{repository}:{username}"

LABELING_QUEUE = "magaldi:labeling:queue:{scope}:{repository}:{username}"
LABELING_RUNNING = "magaldi:labeling:running:{scope}:{repository}:{username}"
LABELING_JOBS = "magaldi:labeling:jobs:{scope}:{repository}:{username}"

SUBFEATURE_QUEUE = "magaldi:subfeature:queue:{scope}:{repository}:{username}"
SUBFEATURE_RUNNING = "magaldi:subfeature:running:{scope}:{repository}:{username}"
SUBFEATURE_JOBS = "magaldi:subfeature:jobs:{scope}:{repository}:{username}"


def _key(template: str, scope: str, repository: str, username: str) -> str:
    """Format a key template with scope, repository, and username."""
    return template.format(scope=scope, repository=repository, username=username)


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
    """Redis-based summarization job queue with scope/repository/user isolation."""

    def add_job(
        self,
        element_id: str,
        scope: str,
        repository: str,
        username: str,
        level: int,
        parent_id: str | None,
        dependencies_met: bool = False,
        priority: int = 0,
    ) -> None:
        """Add a summarization job to the queue.

        Args:
            element_id: Element to summarize.
            scope: Repository scope.
            repository: Repository name.
            username: User who owns this job (for queue isolation).
            level: Hierarchy level (0=file, 1=class, 2=function).
            parent_id: Parent element ID (for dependency tracking).
            dependencies_met: Whether dependencies are satisfied.
            priority: Job priority (higher = more urgent).
        """
        client = self._get_client()

        job_data = {
            "element_id": element_id,
            "scope": scope,
            "repository": repository,
            "username": username,
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

        # Store job data in job hash
        client.hset(
            _key(SUMMARIZATION_JOBS, scope, repository, username),
            element_id,
            json.dumps(job_data),
        )

        # Add to queue if dependencies are met (sorted by level desc, then priority)
        if dependencies_met:
            # Score: level * 1000 + priority (process higher levels first)
            score = level * 1000 + priority
            client.zadd(
                _key(SUMMARIZATION_QUEUE, scope, repository, username),
                {element_id: score},
            )

    def get_job(
        self, element_id: str, scope: str, repository: str, username: str
    ) -> dict[str, Any] | None:
        """Get job data by element ID.

        Args:
            element_id: Element ID.
            scope: Repository scope.
            repository: Repository name.
            username: User who owns this job.
        """
        client = self._get_client()
        data = client.hget(
            _key(SUMMARIZATION_JOBS, scope, repository, username), element_id
        )
        if data:
            return json.loads(data)
        return None

    def claim_pending_jobs(
        self,
        worker_id: str,
        scope: str,
        repository: str,
        username: str,
        batch_size: int = 10,
    ) -> list[dict[str, Any]]:
        """Claim pending jobs from the queue.

        Args:
            worker_id: ID of the claiming worker.
            scope: Repository scope.
            repository: Repository name.
            username: User whose queue to process.
            batch_size: Maximum jobs to claim.

        Returns:
            List of claimed job data.
        """
        client = self._get_client()
        claimed = []

        queue_key = _key(SUMMARIZATION_QUEUE, scope, repository, username)
        jobs_key = _key(SUMMARIZATION_JOBS, scope, repository, username)
        running_key = _key(SUMMARIZATION_RUNNING, scope, repository, username)

        # Get highest priority jobs (highest scores first)
        element_ids = client.zrevrange(queue_key, 0, batch_size - 1)

        for element_id in element_ids:
            # Atomically move from queue to running
            removed = client.zrem(queue_key, element_id)
            if removed:
                # Update job status
                job_data = self.get_job(element_id, scope, repository, username)
                if job_data:
                    job_data["status"] = "running"
                    job_data["worker_id"] = worker_id
                    job_data["claimed_at"] = datetime.now().isoformat()
                    client.hset(jobs_key, element_id, json.dumps(job_data))
                    client.sadd(running_key, element_id)
                    claimed.append(job_data)

        return claimed

    def mark_completed(
        self, element_id: str, scope: str, repository: str, username: str
    ) -> None:
        """Mark a job as completed.

        Args:
            element_id: Element ID.
            scope: Repository scope.
            repository: Repository name.
            username: User who owns this job.
        """
        client = self._get_client()

        job_data = self.get_job(element_id, scope, repository, username)
        if job_data:
            job_data["status"] = "completed"
            job_data["completed_at"] = datetime.now().isoformat()
            client.hset(
                _key(SUMMARIZATION_JOBS, scope, repository, username),
                element_id,
                json.dumps(job_data),
            )
            client.srem(
                _key(SUMMARIZATION_RUNNING, scope, repository, username), element_id
            )

            # Unlock dependent jobs
            self.unlock_dependencies(element_id, scope, repository, username)

    def mark_failed(
        self,
        element_id: str,
        scope: str,
        repository: str,
        username: str,
        error_message: str,
    ) -> None:
        """Mark a job as failed.

        Args:
            element_id: Element ID.
            scope: Repository scope.
            repository: Repository name.
            username: User who owns this job.
            error_message: Error description.
        """
        client = self._get_client()

        job_data = self.get_job(element_id, scope, repository, username)
        if job_data:
            job_data["status"] = "failed"
            job_data["error_message"] = error_message
            job_data["retry_count"] = job_data.get("retry_count", 0) + 1
            client.hset(
                _key(SUMMARIZATION_JOBS, scope, repository, username),
                element_id,
                json.dumps(job_data),
            )
            client.srem(
                _key(SUMMARIZATION_RUNNING, scope, repository, username), element_id
            )

    def unlock_dependencies(
        self, parent_id: str, scope: str, repository: str, username: str
    ) -> int:
        """Unlock jobs that depend on the completed parent.

        Args:
            parent_id: ID of completed parent element.
            scope: Repository scope.
            repository: Repository name.
            username: User who owns these jobs.

        Returns:
            Number of jobs unlocked.
        """
        client = self._get_client()
        unlocked = 0

        jobs_key = _key(SUMMARIZATION_JOBS, scope, repository, username)
        queue_key = _key(SUMMARIZATION_QUEUE, scope, repository, username)

        # Scan jobs to find children
        all_jobs = client.hgetall(jobs_key)
        for element_id, data in all_jobs.items():
            job_data = json.loads(data)
            if (
                job_data.get("parent_id") == parent_id
                and not job_data.get("dependencies_met")
                and job_data.get("status") == "pending"
            ):
                # Unlock this job
                job_data["dependencies_met"] = True
                client.hset(jobs_key, element_id, json.dumps(job_data))

                # Add to queue
                score = job_data["level"] * 1000 + job_data.get("priority", 0)
                client.zadd(queue_key, {element_id: score})
                unlocked += 1

        return unlocked


class RedisEmbeddingJobRepository(RedisRepository):
    """Redis-based embedding job queue with scope/repository/user isolation."""

    def add_job(
        self, element_id: str, scope: str, repository: str, username: str
    ) -> None:
        """Add an embedding job to the queue.

        Args:
            element_id: Element to embed.
            scope: Repository scope.
            repository: Repository name.
            username: User who owns this job.
        """
        client = self._get_client()

        job_data = {
            "element_id": element_id,
            "scope": scope,
            "repository": repository,
            "username": username,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "worker_id": None,
            "claimed_at": None,
            "completed_at": None,
            "error_message": None,
            "retry_count": 0,
        }

        # Store job data in job hash
        client.hset(
            _key(EMBEDDING_JOBS, scope, repository, username),
            element_id,
            json.dumps(job_data),
        )

        # Add to queue (FIFO order using timestamp as score)
        client.zadd(
            _key(EMBEDDING_QUEUE, scope, repository, username),
            {element_id: time.time()},
        )

    def get_job(
        self, element_id: str, scope: str, repository: str, username: str
    ) -> dict[str, Any] | None:
        """Get job data by element ID.

        Args:
            element_id: Element ID.
            scope: Repository scope.
            repository: Repository name.
            username: User who owns this job.
        """
        client = self._get_client()
        data = client.hget(
            _key(EMBEDDING_JOBS, scope, repository, username), element_id
        )
        if data:
            return json.loads(data)
        return None

    def claim_pending_jobs(
        self,
        worker_id: str,
        scope: str,
        repository: str,
        username: str,
        batch_size: int = 10,
    ) -> list[dict[str, Any]]:
        """Claim pending jobs from the queue.

        Args:
            worker_id: ID of the claiming worker.
            scope: Repository scope.
            repository: Repository name.
            username: User whose queue to process.
            batch_size: Maximum jobs to claim.

        Returns:
            List of claimed job data.
        """
        client = self._get_client()
        claimed = []

        queue_key = _key(EMBEDDING_QUEUE, scope, repository, username)
        jobs_key = _key(EMBEDDING_JOBS, scope, repository, username)
        running_key = _key(EMBEDDING_RUNNING, scope, repository, username)

        # Get oldest jobs first (FIFO)
        element_ids = client.zrange(queue_key, 0, batch_size - 1)

        for element_id in element_ids:
            removed = client.zrem(queue_key, element_id)
            if removed:
                job_data = self.get_job(element_id, scope, repository, username)
                if job_data:
                    job_data["status"] = "running"
                    job_data["worker_id"] = worker_id
                    job_data["claimed_at"] = datetime.now().isoformat()
                    client.hset(jobs_key, element_id, json.dumps(job_data))
                    client.sadd(running_key, element_id)
                    claimed.append(job_data)

        return claimed

    def mark_completed(
        self, element_id: str, scope: str, repository: str, username: str
    ) -> None:
        """Mark a job as completed.

        Args:
            element_id: Element ID.
            scope: Repository scope.
            repository: Repository name.
            username: User who owns this job.
        """
        client = self._get_client()

        job_data = self.get_job(element_id, scope, repository, username)
        if job_data:
            job_data["status"] = "completed"
            job_data["completed_at"] = datetime.now().isoformat()
            client.hset(
                _key(EMBEDDING_JOBS, scope, repository, username),
                element_id,
                json.dumps(job_data),
            )
            client.srem(
                _key(EMBEDDING_RUNNING, scope, repository, username), element_id
            )

    def mark_failed(
        self,
        element_id: str,
        scope: str,
        repository: str,
        username: str,
        error_message: str,
    ) -> None:
        """Mark a job as failed.

        Args:
            element_id: Element ID.
            scope: Repository scope.
            repository: Repository name.
            username: User who owns this job.
            error_message: Error description.
        """
        client = self._get_client()

        job_data = self.get_job(element_id, scope, repository, username)
        if job_data:
            job_data["status"] = "failed"
            job_data["error_message"] = error_message
            job_data["retry_count"] = job_data.get("retry_count", 0) + 1
            client.hset(
                _key(EMBEDDING_JOBS, scope, repository, username),
                element_id,
                json.dumps(job_data),
            )
            client.srem(
                _key(EMBEDDING_RUNNING, scope, repository, username), element_id
            )


class RedisSummaryStore(RedisRepository):
    """Redis-based summary store (for quick access during summarization)."""

    SUMMARIES_KEY = "magaldi:summaries"
    ELEMENTS_KEY = "magaldi:elements"

    def store_element(self, element: Any) -> None:
        """Store element data for later retrieval."""
        from magaldi_core.code_parser import CodeElement

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
        from magaldi_core.code_parser import CodeElement

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


class RedisFeatureJobRepository(RedisRepository):
    """Redis-based feature job queue with scope/repository/user isolation."""

    def add_job(
        self, feature_id: str, scope: str, repository: str, username: str, label: str
    ) -> None:
        """Add a feature job to the queue.

        Args:
            feature_id: Feature identifier.
            scope: Repository scope.
            repository: Repository name.
            username: User who owns this job.
            label: Feature label/name.
        """
        client = self._get_client()

        job_data = {
            "feature_id": feature_id,
            "label": label,
            "scope": scope,
            "repository": repository,
            "username": username,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "worker_id": None,
            "claimed_at": None,
            "completed_at": None,
            "error_message": None,
        }

        # Store job data in job hash
        client.hset(
            _key(FEATURE_JOBS, scope, repository, username),
            feature_id,
            json.dumps(job_data),
        )

        # Add to queue (FIFO order using timestamp as score)
        client.zadd(
            _key(FEATURE_QUEUE, scope, repository, username),
            {feature_id: time.time()},
        )

    def get_job(
        self, feature_id: str, scope: str, repository: str, username: str
    ) -> dict[str, Any] | None:
        """Get job data by feature ID."""
        client = self._get_client()
        data = client.hget(
            _key(FEATURE_JOBS, scope, repository, username), feature_id
        )
        if data:
            return json.loads(data)
        return None

    def mark_running(
        self, feature_id: str, scope: str, repository: str, username: str
    ) -> None:
        """Mark a job as running."""
        client = self._get_client()

        jobs_key = _key(FEATURE_JOBS, scope, repository, username)
        running_key = _key(FEATURE_RUNNING, scope, repository, username)

        job_data = self.get_job(feature_id, scope, repository, username)
        if job_data:
            job_data["status"] = "running"
            job_data["claimed_at"] = datetime.now().isoformat()
            client.hset(jobs_key, feature_id, json.dumps(job_data))
            client.sadd(running_key, feature_id)
            # Remove from pending queue
            client.zrem(_key(FEATURE_QUEUE, scope, repository, username), feature_id)

    def mark_completed(
        self, feature_id: str, scope: str, repository: str, username: str
    ) -> None:
        """Mark a job as completed."""
        client = self._get_client()

        job_data = self.get_job(feature_id, scope, repository, username)
        if job_data:
            job_data["status"] = "completed"
            job_data["completed_at"] = datetime.now().isoformat()
            client.hset(
                _key(FEATURE_JOBS, scope, repository, username),
                feature_id,
                json.dumps(job_data),
            )
            client.srem(
                _key(FEATURE_RUNNING, scope, repository, username), feature_id
            )

    def mark_failed(
        self,
        feature_id: str,
        scope: str,
        repository: str,
        username: str,
        error_message: str,
    ) -> None:
        """Mark a job as failed."""
        client = self._get_client()

        job_data = self.get_job(feature_id, scope, repository, username)
        if job_data:
            job_data["status"] = "failed"
            job_data["error_message"] = error_message
            client.hset(
                _key(FEATURE_JOBS, scope, repository, username),
                feature_id,
                json.dumps(job_data),
            )
            client.srem(
                _key(FEATURE_RUNNING, scope, repository, username), feature_id
            )


class RedisLabelingJobRepository(RedisRepository):
    """Redis-based labeling job queue with scope/repository/user isolation."""

    def add_job(
        self, cluster_id: int, scope: str, repository: str, username: str
    ) -> None:
        """Add a labeling job to the queue."""
        client = self._get_client()
        job_key = str(cluster_id)

        job_data = {
            "cluster_id": cluster_id,
            "scope": scope,
            "repository": repository,
            "username": username,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "error_message": None,
        }

        client.hset(
            _key(LABELING_JOBS, scope, repository, username),
            job_key,
            json.dumps(job_data),
        )
        client.zadd(
            _key(LABELING_QUEUE, scope, repository, username),
            {job_key: time.time()},
        )

    def get_job(
        self, cluster_id: int, scope: str, repository: str, username: str
    ) -> dict[str, Any] | None:
        """Get job data by cluster ID."""
        client = self._get_client()
        data = client.hget(
            _key(LABELING_JOBS, scope, repository, username), str(cluster_id)
        )
        if data:
            return json.loads(data)
        return None

    def mark_running(
        self, cluster_id: int, scope: str, repository: str, username: str
    ) -> None:
        """Mark a job as running."""
        client = self._get_client()
        job_key = str(cluster_id)

        jobs_key = _key(LABELING_JOBS, scope, repository, username)
        running_key = _key(LABELING_RUNNING, scope, repository, username)

        job_data = self.get_job(cluster_id, scope, repository, username)
        if job_data:
            job_data["status"] = "running"
            client.hset(jobs_key, job_key, json.dumps(job_data))
            client.sadd(running_key, job_key)
            client.zrem(_key(LABELING_QUEUE, scope, repository, username), job_key)

    def mark_completed(
        self, cluster_id: int, scope: str, repository: str, username: str
    ) -> None:
        """Mark a job as completed."""
        client = self._get_client()
        job_key = str(cluster_id)

        job_data = self.get_job(cluster_id, scope, repository, username)
        if job_data:
            job_data["status"] = "completed"
            job_data["completed_at"] = datetime.now().isoformat()
            client.hset(
                _key(LABELING_JOBS, scope, repository, username),
                job_key,
                json.dumps(job_data),
            )
            client.srem(
                _key(LABELING_RUNNING, scope, repository, username), job_key
            )

    def mark_failed(
        self,
        cluster_id: int,
        scope: str,
        repository: str,
        username: str,
        error_message: str,
    ) -> None:
        """Mark a job as failed."""
        client = self._get_client()
        job_key = str(cluster_id)

        job_data = self.get_job(cluster_id, scope, repository, username)
        if job_data:
            job_data["status"] = "failed"
            job_data["error_message"] = error_message
            client.hset(
                _key(LABELING_JOBS, scope, repository, username),
                job_key,
                json.dumps(job_data),
            )
            client.srem(
                _key(LABELING_RUNNING, scope, repository, username), job_key
            )


class RedisSubfeatureJobRepository(RedisRepository):
    """Redis-based subfeature job queue with scope/repository/user isolation."""

    def add_job(
        self,
        subfeature_id: str,
        scope: str,
        repository: str,
        username: str,
        label: str,
        parent_label: str,
    ) -> None:
        """Add a subfeature job to the queue.

        Args:
            subfeature_id: Subfeature identifier.
            scope: Repository scope.
            repository: Repository name.
            username: User who owns this job.
            label: Subfeature label/name.
            parent_label: Parent feature label.
        """
        client = self._get_client()

        job_data = {
            "subfeature_id": subfeature_id,
            "label": label,
            "parent_label": parent_label,
            "scope": scope,
            "repository": repository,
            "username": username,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "worker_id": None,
            "claimed_at": None,
            "completed_at": None,
            "error_message": None,
        }

        client.hset(
            _key(SUBFEATURE_JOBS, scope, repository, username),
            subfeature_id,
            json.dumps(job_data),
        )

        client.zadd(
            _key(SUBFEATURE_QUEUE, scope, repository, username),
            {subfeature_id: time.time()},
        )

    def get_job(
        self, subfeature_id: str, scope: str, repository: str, username: str
    ) -> dict[str, Any] | None:
        """Get job data by subfeature ID."""
        client = self._get_client()
        data = client.hget(
            _key(SUBFEATURE_JOBS, scope, repository, username), subfeature_id
        )
        if data:
            return json.loads(data)
        return None

    def mark_running(
        self, subfeature_id: str, scope: str, repository: str, username: str
    ) -> None:
        """Mark a job as running."""
        client = self._get_client()

        jobs_key = _key(SUBFEATURE_JOBS, scope, repository, username)
        running_key = _key(SUBFEATURE_RUNNING, scope, repository, username)

        job_data = self.get_job(subfeature_id, scope, repository, username)
        if job_data:
            job_data["status"] = "running"
            job_data["claimed_at"] = datetime.now().isoformat()
            client.hset(jobs_key, subfeature_id, json.dumps(job_data))
            client.sadd(running_key, subfeature_id)
            client.zrem(_key(SUBFEATURE_QUEUE, scope, repository, username), subfeature_id)

    def mark_completed(
        self, subfeature_id: str, scope: str, repository: str, username: str
    ) -> None:
        """Mark a job as completed."""
        client = self._get_client()

        job_data = self.get_job(subfeature_id, scope, repository, username)
        if job_data:
            job_data["status"] = "completed"
            job_data["completed_at"] = datetime.now().isoformat()
            client.hset(
                _key(SUBFEATURE_JOBS, scope, repository, username),
                subfeature_id,
                json.dumps(job_data),
            )
            client.srem(
                _key(SUBFEATURE_RUNNING, scope, repository, username), subfeature_id
            )

    def mark_failed(
        self,
        subfeature_id: str,
        scope: str,
        repository: str,
        username: str,
        error_message: str,
    ) -> None:
        """Mark a job as failed."""
        client = self._get_client()

        job_data = self.get_job(subfeature_id, scope, repository, username)
        if job_data:
            job_data["status"] = "failed"
            job_data["error_message"] = error_message
            client.hset(
                _key(SUBFEATURE_JOBS, scope, repository, username),
                subfeature_id,
                json.dumps(job_data),
            )
            client.srem(
                _key(SUBFEATURE_RUNNING, scope, repository, username), subfeature_id
            )
