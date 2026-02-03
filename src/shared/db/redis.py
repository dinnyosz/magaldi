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

SUBFEATURE_LABELING_QUEUE = "magaldi:subfeature_labeling:queue:{scope}:{repository}:{username}"
SUBFEATURE_LABELING_RUNNING = "magaldi:subfeature_labeling:running:{scope}:{repository}:{username}"
SUBFEATURE_LABELING_JOBS = "magaldi:subfeature_labeling:jobs:{scope}:{repository}:{username}"


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


class RedisSubfeatureLabelingJobRepository(RedisRepository):
    """Redis-based subfeature labeling job queue with scope/repository/user isolation."""

    def add_job(
        self,
        parent_label: str,
        cluster_count: int,
        scope: str,
        repository: str,
        username: str,
    ) -> None:
        """Add a subfeature labeling job to the queue.

        Args:
            parent_label: Parent feature label being sub-clustered.
            cluster_count: Number of subclusters to label.
            scope: Repository scope.
            repository: Repository name.
            username: User who owns this job.
        """
        client = self._get_client()

        job_data = {
            "parent_label": parent_label,
            "cluster_count": cluster_count,
            "scope": scope,
            "repository": repository,
            "username": username,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "error_message": None,
        }

        client.hset(
            _key(SUBFEATURE_LABELING_JOBS, scope, repository, username),
            parent_label,
            json.dumps(job_data),
        )

        client.zadd(
            _key(SUBFEATURE_LABELING_QUEUE, scope, repository, username),
            {parent_label: time.time()},
        )

    def get_job(
        self, parent_label: str, scope: str, repository: str, username: str
    ) -> dict[str, Any] | None:
        """Get job data by parent label."""
        client = self._get_client()
        data = client.hget(
            _key(SUBFEATURE_LABELING_JOBS, scope, repository, username), parent_label
        )
        if data:
            return json.loads(data)
        return None

    def mark_running(
        self, parent_label: str, scope: str, repository: str, username: str
    ) -> None:
        """Mark a job as running."""
        client = self._get_client()

        jobs_key = _key(SUBFEATURE_LABELING_JOBS, scope, repository, username)
        running_key = _key(SUBFEATURE_LABELING_RUNNING, scope, repository, username)

        job_data = self.get_job(parent_label, scope, repository, username)
        if job_data:
            job_data["status"] = "running"
            client.hset(jobs_key, parent_label, json.dumps(job_data))
            client.sadd(running_key, parent_label)
            client.zrem(_key(SUBFEATURE_LABELING_QUEUE, scope, repository, username), parent_label)

    def mark_completed(
        self, parent_label: str, scope: str, repository: str, username: str
    ) -> None:
        """Mark a job as completed."""
        client = self._get_client()

        job_data = self.get_job(parent_label, scope, repository, username)
        if job_data:
            job_data["status"] = "completed"
            job_data["completed_at"] = datetime.now().isoformat()
            client.hset(
                _key(SUBFEATURE_LABELING_JOBS, scope, repository, username),
                parent_label,
                json.dumps(job_data),
            )
            client.srem(
                _key(SUBFEATURE_LABELING_RUNNING, scope, repository, username), parent_label
            )

    def mark_failed(
        self,
        parent_label: str,
        scope: str,
        repository: str,
        username: str,
        error_message: str,
    ) -> None:
        """Mark a job as failed."""
        client = self._get_client()

        job_data = self.get_job(parent_label, scope, repository, username)
        if job_data:
            job_data["status"] = "failed"
            job_data["error_message"] = error_message
            client.hset(
                _key(SUBFEATURE_LABELING_JOBS, scope, repository, username),
                parent_label,
                json.dumps(job_data),
            )
            client.srem(
                _key(SUBFEATURE_LABELING_RUNNING, scope, repository, username), parent_label
            )


# MCP Analytics Redis keys
MCP_TOOL_CALLS = "magaldi:mcp:tool_calls"
MCP_TOOL_TRANSITIONS = "magaldi:mcp:tool_transitions"
MCP_SESSION_PREFIX = "magaldi:mcp:session:"
MCP_DAILY_CALLS_PREFIX = "magaldi:mcp:daily:"


class RedisMCPAnalyticsRepository(RedisRepository):
    """Redis-based MCP tool usage analytics.

    Tracks:
    - Tool call counts (total and daily)
    - Tool transitions (which tool follows which)
    - Session-based tracking for transition computation
    """

    SESSION_TTL = 3600  # 1 hour session timeout
    DAILY_TTL = 30 * 24 * 3600  # 30 days retention for daily data
    TRANSITION_MAX_GAP_SECONDS = 10  # Max gap between calls to count as a transition

    def record_tool_call(
        self,
        tool_name: str,
        session_id: str | None = None,
    ) -> None:
        """Record a tool call and compute transitions.

        Transitions are only recorded if the gap between the previous tool's
        end time and this tool's start time is <= TRANSITION_MAX_GAP_SECONDS.
        This ensures only related tool calls are counted as transitions.

        Args:
            tool_name: Name of the tool being called.
            session_id: Optional session identifier for transition tracking.
        """
        import json

        client = self._get_client()
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")

        # Increment total call count
        client.hincrby(MCP_TOOL_CALLS, tool_name, 1)

        # Increment daily call count
        daily_key = f"{MCP_DAILY_CALLS_PREFIX}{today}:calls"
        client.hincrby(daily_key, tool_name, 1)
        client.expire(daily_key, self.DAILY_TTL)

        # Track transitions if we have a session
        if session_id:
            session_key = f"{MCP_SESSION_PREFIX}{session_id}"
            last_data = client.get(session_key)

            if last_data:
                try:
                    # Parse previous tool data (JSON: {tool, end_time})
                    data = json.loads(last_data)
                    last_tool = data.get("tool")
                    last_end_time = data.get("end_time")

                    # Only record transition if previous call ended recently
                    if last_tool and last_end_time:
                        end_dt = datetime.fromisoformat(last_end_time)
                        gap_seconds = (now - end_dt).total_seconds()

                        if gap_seconds <= self.TRANSITION_MAX_GAP_SECONDS:
                            # Record transition from last_tool to tool_name
                            transition_key = f"{last_tool}:{tool_name}"
                            client.hincrby(MCP_TOOL_TRANSITIONS, transition_key, 1)

                            # Also track daily transitions
                            daily_trans_key = f"{MCP_DAILY_CALLS_PREFIX}{today}:transitions"
                            client.hincrby(daily_trans_key, transition_key, 1)
                            client.expire(daily_trans_key, self.DAILY_TTL)
                except (json.JSONDecodeError, ValueError):
                    # Invalid data format, skip transition tracking
                    pass

            # Update session with current tool (start_time for duration, end_time for transitions)
            session_data = json.dumps({
                "tool": tool_name,
                "start_time": now.isoformat(),
                "end_time": None,
            })
            client.setex(session_key, self.SESSION_TTL, session_data)

    def record_tool_end(self, session_id: str | None = None) -> None:
        """Record that the current tool call has ended.

        This updates the session with the end time, which is used to determine
        if subsequent tool calls should be counted as transitions. Also records
        the tool execution duration for runtime analytics.

        Args:
            session_id: Session identifier for transition tracking.
        """
        import json

        if not session_id:
            return

        client = self._get_client()
        session_key = f"{MCP_SESSION_PREFIX}{session_id}"
        current_data = client.get(session_key)

        if current_data:
            try:
                data = json.loads(current_data)
                now = datetime.now()
                data["end_time"] = now.isoformat()

                # Calculate and record duration if we have start_time
                start_time = data.get("start_time")
                tool_name = data.get("tool")
                if start_time and tool_name:
                    start_dt = datetime.fromisoformat(start_time)
                    duration_ms = int((now - start_dt).total_seconds() * 1000)

                    # Record duration stats: total_ms and call_count for averaging
                    duration_key = f"{MCP_DAILY_CALLS_PREFIX}durations"
                    client.hincrby(duration_key, f"{tool_name}:total_ms", duration_ms)
                    client.hincrby(duration_key, f"{tool_name}:count", 1)

                client.setex(session_key, self.SESSION_TTL, json.dumps(data))
            except json.JSONDecodeError:
                pass

    def get_tool_durations(self) -> dict[str, dict[str, int | float]]:
        """Get tool execution duration statistics.

        Returns:
            Dict mapping tool name to {total_ms, count, avg_ms}.
        """
        client = self._get_client()
        duration_key = f"{MCP_DAILY_CALLS_PREFIX}durations"
        raw_data = client.hgetall(duration_key)

        # Parse and aggregate
        tools: dict[str, dict[str, int]] = {}
        for key, value in raw_data.items():
            # key format: "tool_name:total_ms" or "tool_name:count"
            parts = key.rsplit(":", 1)
            if len(parts) == 2:
                tool_name, metric = parts
                if tool_name not in tools:
                    tools[tool_name] = {"total_ms": 0, "count": 0}
                tools[tool_name][metric] = int(value)

        # Calculate averages
        result: dict[str, dict[str, int | float]] = {}
        for tool_name, stats in tools.items():
            total_ms = stats.get("total_ms", 0)
            count = stats.get("count", 0)
            avg_ms = total_ms / count if count > 0 else 0
            result[tool_name] = {
                "total_ms": total_ms,
                "count": count,
                "avg_ms": round(avg_ms, 1),
            }

        return result

    def get_tool_counts(self) -> dict[str, int]:
        """Get all tool call counts.

        Returns:
            Dict mapping tool name to call count.
        """
        client = self._get_client()
        counts = client.hgetall(MCP_TOOL_CALLS)
        return {k: int(v) for k, v in counts.items()}

    def get_tool_transitions(self) -> dict[str, dict[str, int]]:
        """Get tool transition matrix.

        Returns:
            Nested dict: from_tool -> to_tool -> count
        """
        client = self._get_client()
        raw_transitions = client.hgetall(MCP_TOOL_TRANSITIONS)

        matrix: dict[str, dict[str, int]] = {}
        for transition_key, count in raw_transitions.items():
            parts = transition_key.split(":", 1)
            if len(parts) == 2:
                from_tool, to_tool = parts
                if from_tool not in matrix:
                    matrix[from_tool] = {}
                matrix[from_tool][to_tool] = int(count)

        return matrix

    def get_top_tools(self, limit: int = 10) -> list[tuple[str, int]]:
        """Get most frequently called tools.

        Args:
            limit: Maximum number of tools to return.

        Returns:
            List of (tool_name, count) tuples, sorted by count descending.
        """
        counts = self.get_tool_counts()
        sorted_tools = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_tools[:limit]

    def get_top_transitions(
        self, limit: int = 10
    ) -> list[tuple[str, str, int]]:
        """Get most common tool transitions.

        Args:
            limit: Maximum number of transitions to return.

        Returns:
            List of (from_tool, to_tool, count) tuples, sorted by count descending.
        """
        client = self._get_client()
        raw_transitions = client.hgetall(MCP_TOOL_TRANSITIONS)

        transitions = []
        for transition_key, count in raw_transitions.items():
            parts = transition_key.split(":", 1)
            if len(parts) == 2:
                from_tool, to_tool = parts
                transitions.append((from_tool, to_tool, int(count)))

        transitions.sort(key=lambda x: x[2], reverse=True)
        return transitions[:limit]

    def get_daily_counts(self, date: str | None = None) -> dict[str, int]:
        """Get tool counts for a specific day.

        Args:
            date: Date string (YYYY-MM-DD). Defaults to today.

        Returns:
            Dict mapping tool name to call count for that day.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        client = self._get_client()
        daily_key = f"{MCP_DAILY_CALLS_PREFIX}{date}:calls"
        counts = client.hgetall(daily_key)
        return {k: int(v) for k, v in counts.items()}

    def get_daily_transitions(
        self, date: str | None = None
    ) -> dict[str, dict[str, int]]:
        """Get transitions for a specific day.

        Args:
            date: Date string (YYYY-MM-DD). Defaults to today.

        Returns:
            Nested dict: from_tool -> to_tool -> count
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        client = self._get_client()
        daily_key = f"{MCP_DAILY_CALLS_PREFIX}{date}:transitions"
        raw_transitions = client.hgetall(daily_key)

        matrix: dict[str, dict[str, int]] = {}
        for transition_key, count in raw_transitions.items():
            parts = transition_key.split(":", 1)
            if len(parts) == 2:
                from_tool, to_tool = parts
                if from_tool not in matrix:
                    matrix[from_tool] = {}
                matrix[from_tool][to_tool] = int(count)

        return matrix

    def clear_analytics(self) -> None:
        """Clear all analytics data.

        Warning: This permanently deletes all tracking data.
        """
        client = self._get_client()

        # Delete main keys
        client.delete(MCP_TOOL_CALLS)
        client.delete(MCP_TOOL_TRANSITIONS)

        # Delete all session keys
        for key in client.scan_iter(f"{MCP_SESSION_PREFIX}*"):
            client.delete(key)

        # Delete all daily keys
        for key in client.scan_iter(f"{MCP_DAILY_CALLS_PREFIX}*"):
            client.delete(key)
