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
MCP_CONFIRMED_TRANSITIONS = "magaldi:mcp:confirmed_transitions"  # Transitions with causal links
MCP_SESSION_PREFIX = "magaldi:mcp:session:"
MCP_DAILY_CALLS_PREFIX = "magaldi:mcp:daily:"
MCP_RECENT_CALLS = "magaldi:mcp:recent_calls"  # List of recent tool calls with details

# Known MCP tool names for causal detection (tool mention in output)
MCP_TOOL_NAMES = {
    "search_code", "search_features", "find_similar", "get_element",
    "batch_get_elements", "get_context", "get_children", "find_files",
    "get_file_structure", "list_features", "get_feature_members",
    "pattern_search", "find_usages", "find_implementations", "get_call_graph",
    "find_callers", "find_call_chain", "find_dead_code", "find_entry_points",
    "list_glossary", "get_glossary_term", "search_glossary",
    "find_dependencies", "find_dependents", "dependency_graph",
    "list_patterns", "find_by_pattern", "find_complex_functions",
    "find_security_issues", "find_undocumented", "find_env_usage",
    "find_async_code", "get_command_tree", "get_route_tree",
    "list_repos", "get_repo_stats", "explain_element", "generate_config",
    "generate_skill", "parser_lab_analyze", "parser_lab_create_test",
    "parser_lab_run_tests", "parser_lab_suggest_fix",
    # Context7 tools
    "resolve-library-id", "query-docs",
}


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

    def get_confirmed_transitions(self) -> dict[str, dict[str, int]]:
        """Get confirmed transition matrix (transitions with causal links).

        Returns:
            Nested dict: from_tool -> to_tool -> count
        """
        client = self._get_client()
        raw_transitions = client.hgetall(MCP_CONFIRMED_TRANSITIONS)

        matrix: dict[str, dict[str, int]] = {}
        for transition_key, count in raw_transitions.items():
            parts = transition_key.split(":", 1)
            if len(parts) == 2:
                from_tool, to_tool = parts
                if from_tool not in matrix:
                    matrix[from_tool] = {}
                matrix[from_tool][to_tool] = int(count)

        return matrix

    def get_transition_comparison(self) -> list[dict[str, Any]]:
        """Compare all transitions with confirmed transitions.

        Returns:
            List of transitions with confirmation rates.
        """
        all_transitions = self.get_tool_transitions()
        confirmed = self.get_confirmed_transitions()

        result = []
        for from_tool, targets in all_transitions.items():
            for to_tool, total_count in targets.items():
                confirmed_count = confirmed.get(from_tool, {}).get(to_tool, 0)
                confirmation_rate = (
                    round(confirmed_count / total_count * 100, 1)
                    if total_count > 0
                    else 0.0
                )
                result.append({
                    "from_tool": from_tool,
                    "to_tool": to_tool,
                    "total_count": total_count,
                    "confirmed_count": confirmed_count,
                    "confirmation_rate": confirmation_rate,
                })

        # Sort by total count descending
        result.sort(key=lambda x: x["total_count"], reverse=True)
        return result

    def clear_analytics(self) -> None:
        """Clear all analytics data.

        Warning: This permanently deletes all tracking data.
        """
        client = self._get_client()

        # Delete main keys
        client.delete(MCP_TOOL_CALLS)
        client.delete(MCP_TOOL_TRANSITIONS)
        client.delete(MCP_CONFIRMED_TRANSITIONS)
        client.delete(MCP_RECENT_CALLS)

        # Delete all session keys
        for key in client.scan_iter(f"{MCP_SESSION_PREFIX}*"):
            client.delete(key)

        # Delete all daily keys
        for key in client.scan_iter(f"{MCP_DAILY_CALLS_PREFIX}*"):
            client.delete(key)

    # Maximum size for stored inputs/outputs (in characters)
    MAX_IO_SIZE = 10000
    # Maximum number of recent calls to keep
    MAX_RECENT_CALLS = 1000
    # Time window for causal link detection (seconds)
    # 2 minutes to allow multiple follow-up calls from one search
    CAUSAL_LINK_WINDOW = 120
    # Number of previous calls to check for causal links
    CAUSAL_HISTORY_LOOKBACK = 15

    def _extract_trackable_values(self, output: str) -> set[str]:
        """Extract values from output that could be tracked as causal links.

        Looks for:
        - hash_id values (64 char hex strings)
        - library IDs (format: /org/project)
        - file paths (containing / and common extensions)
        - element IDs with colons (scope:repo:user:path:type:name:line)

        Args:
            output: Tool output string.

        Returns:
            Set of trackable values found.
        """
        import re

        values: set[str] = set()

        # Hash IDs (64 character hex strings)
        hash_pattern = r"\b([a-f0-9]{64})\b"
        values.update(re.findall(hash_pattern, output))

        # Library IDs (Context7 format: /org/project or /org/project/version)
        lib_pattern = r"(/[a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+(?:/[a-zA-Z0-9._-]+)?)"
        values.update(re.findall(lib_pattern, output))

        # Element IDs with colons (at least 3 colon-separated parts)
        element_pattern = r"\b([a-zA-Z0-9_-]+(?::[a-zA-Z0-9_./+-]+){3,})\b"
        values.update(re.findall(element_pattern, output))

        # File paths with extensions
        path_pattern = r"([a-zA-Z0-9_./+-]+\.(?:py|ts|tsx|js|jsx|rs|php|go|java|rb))\b"
        values.update(re.findall(path_pattern, output))

        # Filter out very short values (likely false positives)
        return {v for v in values if len(v) >= 8}

    def _detect_tool_mentions(self, output: str) -> set[str]:
        """Detect MCP tool names mentioned in output.

        Args:
            output: Tool output string.

        Returns:
            Set of tool names mentioned.
        """
        mentioned: set[str] = set()
        output_lower = output.lower()

        for tool_name in MCP_TOOL_NAMES:
            # Check for exact tool name or with underscores/hyphens
            if tool_name.lower() in output_lower:
                mentioned.add(tool_name)
            # Also check snake_case vs kebab-case variants
            alt_name = tool_name.replace("-", "_")
            if alt_name != tool_name and alt_name.lower() in output_lower:
                mentioned.add(tool_name)

        return mentioned

    def _find_matching_value(
        self,
        input_args: dict[str, Any],
        trackable_values: set[str],
    ) -> str | None:
        """Find if any input argument contains a trackable value.

        Args:
            input_args: Current tool's input arguments.
            trackable_values: Values extracted from previous output.

        Returns:
            The matched value, or None if no match.
        """
        import json

        input_str = json.dumps(input_args, default=str)

        for value in trackable_values:
            if value in input_str:
                return value

        return None

    def _detect_causal_link(
        self,
        tool_name: str,
        session_id: str,
        input_args: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Detect if current tool call was triggered by a previous call.

        Uses hybrid approach:
        1. Parameter matching: Check if input contains values from recent output
        2. Tool suggestion: Check if previous output mentioned this tool

        Args:
            tool_name: Name of the current tool being called.
            session_id: Session identifier.
            input_args: Current tool's input arguments.

        Returns:
            Causal link info dict, or None if no link detected.
        """
        client = self._get_client()

        # Get the most recent completed calls from this session
        # Check up to CAUSAL_HISTORY_LOOKBACK calls to find the original source
        # This handles cases where one search leads to multiple follow-up calls
        session_history_key = f"{MCP_SESSION_PREFIX}{session_id}:history"
        recent_calls_raw = client.lrange(session_history_key, 0, self.CAUSAL_HISTORY_LOOKBACK - 1)

        if not recent_calls_raw:
            return None

        now = datetime.now()

        for call_raw in recent_calls_raw:
            try:
                prev_call = json.loads(call_raw)
            except json.JSONDecodeError:
                continue

            prev_output = prev_call.get("output_result", "")
            prev_tool = prev_call.get("tool_name", "")
            prev_end_time = prev_call.get("end_time")

            if not prev_output or not prev_tool:
                continue

            # Check time window
            if prev_end_time:
                try:
                    end_dt = datetime.fromisoformat(prev_end_time)
                    if (now - end_dt).total_seconds() > self.CAUSAL_LINK_WINDOW:
                        continue
                except ValueError:
                    pass

            # Strategy 1: Parameter matching
            trackable_values = self._extract_trackable_values(prev_output)
            matched_value = self._find_matching_value(input_args, trackable_values)

            if matched_value:
                return {
                    "call_id": prev_call.get("call_id"),
                    "tool_name": prev_tool,
                    "matched_value": matched_value,
                    "match_type": "parameter",
                }

            # Strategy 2: Tool suggestion (previous output mentioned this tool)
            mentioned_tools = self._detect_tool_mentions(prev_output)
            if tool_name in mentioned_tools or tool_name.replace("_", "-") in mentioned_tools:
                return {
                    "call_id": prev_call.get("call_id"),
                    "tool_name": prev_tool,
                    "matched_value": None,
                    "match_type": "tool_suggestion",
                }

        return None

    def record_tool_call_details(
        self,
        tool_name: str,
        session_id: str,
        input_args: dict[str, Any],
        timestamp: str | None = None,
    ) -> str:
        """Record a tool call with its input arguments.

        Also detects causal relationships with previous calls using:
        1. Parameter matching (value from previous output used in current input)
        2. Tool suggestion (previous output mentioned this tool)

        Args:
            tool_name: Name of the tool being called.
            session_id: Session identifier.
            input_args: Tool input arguments.
            timestamp: Optional ISO timestamp (defaults to now).

        Returns:
            Call ID for later updating with output.
        """
        import uuid

        client = self._get_client()
        call_id = f"{session_id}:{uuid.uuid4().hex[:8]}"
        now = timestamp or datetime.now().isoformat()

        # Truncate input args to prevent huge storage
        input_str = json.dumps(input_args, default=str)
        if len(input_str) > self.MAX_IO_SIZE:
            input_str = input_str[: self.MAX_IO_SIZE - 3] + "..."

        # Detect causal link with previous calls
        triggered_by = self._detect_causal_link(tool_name, session_id, input_args)

        call_data = {
            "call_id": call_id,
            "tool_name": tool_name,
            "session_id": session_id,
            "input_args": input_str,
            "output_result": None,
            "start_time": now,
            "end_time": None,
            "duration_ms": None,
            "triggered_by": triggered_by,  # Causal link info
            "suggested_tools": None,  # Will be populated in record_tool_output
        }

        # Store in session for later update
        session_call_key = f"{MCP_SESSION_PREFIX}{session_id}:current_call"
        client.setex(session_call_key, self.SESSION_TTL, json.dumps(call_data))

        return call_id

    def record_tool_output(
        self,
        session_id: str,
        output_result: str,
    ) -> None:
        """Record tool output and finalize the call record.

        Also extracts suggested tools from output and stores the call
        in session history for causal link detection in subsequent calls.

        Args:
            session_id: Session identifier.
            output_result: Tool output result (will be truncated if too large).
        """
        client = self._get_client()
        session_call_key = f"{MCP_SESSION_PREFIX}{session_id}:current_call"
        call_data_str = client.get(session_call_key)

        if not call_data_str:
            return

        try:
            call_data = json.loads(call_data_str)
            now = datetime.now()
            call_data["end_time"] = now.isoformat()

            # Calculate duration
            if call_data.get("start_time"):
                start_dt = datetime.fromisoformat(call_data["start_time"])
                call_data["duration_ms"] = int((now - start_dt).total_seconds() * 1000)

            # Truncate output
            if len(output_result) > self.MAX_IO_SIZE:
                output_result = output_result[: self.MAX_IO_SIZE - 3] + "..."
            call_data["output_result"] = output_result

            # Extract suggested tools from output for causal tracking
            suggested_tools = list(self._detect_tool_mentions(output_result))
            call_data["suggested_tools"] = suggested_tools if suggested_tools else None

            # Push to recent calls list (LPUSH for newest first)
            client.lpush(MCP_RECENT_CALLS, json.dumps(call_data))
            # Trim to keep only recent calls
            client.ltrim(MCP_RECENT_CALLS, 0, self.MAX_RECENT_CALLS - 1)

            # Track confirmed transitions (with causal links)
            triggered_by = call_data.get("triggered_by")
            if triggered_by:
                source_tool = triggered_by.get("tool_name")
                target_tool = call_data.get("tool_name")
                if source_tool and target_tool:
                    # Increment confirmed transition counter
                    transition_key = f"{source_tool}:{target_tool}"
                    client.hincrby(MCP_CONFIRMED_TRANSITIONS, transition_key, 1)

            # Add to session history for causal link detection
            # This allows subsequent calls to check this call's output
            session_history_key = f"{MCP_SESSION_PREFIX}{session_id}:history"
            client.lpush(session_history_key, json.dumps(call_data))
            # Keep enough history for causal detection (CAUSAL_HISTORY_LOOKBACK + buffer)
            client.ltrim(session_history_key, 0, 29)
            client.expire(session_history_key, self.SESSION_TTL)

            # Clean up session key
            client.delete(session_call_key)

        except json.JSONDecodeError:
            pass

    def get_recent_calls(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent tool calls with details.

        Args:
            limit: Maximum number of calls to return.

        Returns:
            List of call records with input/output details.
        """
        import json

        client = self._get_client()
        raw_calls = client.lrange(MCP_RECENT_CALLS, 0, limit - 1)

        calls = []
        for raw_call in raw_calls:
            try:
                calls.append(json.loads(raw_call))
            except json.JSONDecodeError:
                continue

        return calls

    def get_transition_details(
        self,
        from_tool: str | None = None,
        to_tool: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get detailed transition information.

        Finds tool call transitions and returns details about the caller's
        output and callee's input. Includes both consecutive (sequential)
        transitions and non-consecutive causal transitions.

        Args:
            from_tool: Filter by source tool (optional).
            to_tool: Filter by destination tool (optional).
            limit: Maximum number of transitions to return.

        Returns:
            List of transition records with caller output and callee input.
        """
        # Get recent calls and find transitions
        calls = self.get_recent_calls(limit=500)

        # Group by session
        sessions: dict[str, list[dict[str, Any]]] = {}
        for call in calls:
            session_id = call.get("session_id", "")
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(call)

        transitions = []
        seen_pairs: set[tuple[str, str]] = set()  # (caller_call_id, callee_call_id)

        for session_id, session_calls in sessions.items():
            # Sort by start_time (oldest first)
            session_calls.sort(key=lambda x: x.get("start_time", ""))

            # Build lookup by call_id for finding causal sources
            calls_by_id: dict[str, dict[str, Any]] = {}
            for call in session_calls:
                call_id = call.get("call_id")
                if call_id:
                    calls_by_id[call_id] = call

            # Find consecutive pairs (sequential transitions)
            for i in range(len(session_calls) - 1):
                caller = session_calls[i]
                callee = session_calls[i + 1]

                caller_name = caller.get("tool_name", "")
                callee_name = callee.get("tool_name", "")

                # Apply filters
                if from_tool and caller_name != from_tool:
                    continue
                if to_tool and callee_name != to_tool:
                    continue

                pair_key = (caller.get("call_id", str(i)), callee.get("call_id", str(i + 1)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # Calculate elapsed time between calls
                elapsed_ms = None
                if caller.get("end_time") and callee.get("start_time"):
                    try:
                        caller_end = datetime.fromisoformat(caller["end_time"])
                        callee_start = datetime.fromisoformat(callee["start_time"])
                        elapsed_ms = int(
                            (callee_start - caller_end).total_seconds() * 1000
                        )
                    except ValueError:
                        pass

                # Check if callee has causal link to caller
                triggered_by = callee.get("triggered_by")
                is_causal = False
                causal_match_type = None
                causal_matched_value = None

                if triggered_by and triggered_by.get("tool_name") == caller_name:
                    is_causal = True
                    causal_match_type = triggered_by.get("match_type")
                    causal_matched_value = triggered_by.get("matched_value")

                transitions.append(
                    {
                        "caller": caller_name,
                        "caller_input": caller.get("input_args"),
                        "caller_output": caller.get("output_result"),
                        "callee": callee_name,
                        "callee_input": callee.get("input_args"),
                        "callee_output": callee.get("output_result"),
                        "elapsed_ms": elapsed_ms,
                        "caller_duration_ms": caller.get("duration_ms"),
                        "callee_duration_ms": callee.get("duration_ms"),
                        "timestamp": callee.get("start_time"),
                        "is_consecutive": True,  # This is a consecutive (temporal) transition
                        "is_causal": is_causal,
                        "causal_match_type": causal_match_type,
                        "causal_matched_value": causal_matched_value,
                    }
                )

            # Find non-consecutive causal transitions (when filtering by tools)
            if from_tool and to_tool:
                for callee in session_calls:
                    callee_name = callee.get("tool_name", "")
                    if callee_name != to_tool:
                        continue

                    triggered_by = callee.get("triggered_by")
                    if not triggered_by:
                        continue

                    source_tool = triggered_by.get("tool_name", "")
                    if source_tool != from_tool:
                        continue

                    # Find the caller in this session
                    source_call_id = triggered_by.get("call_id")
                    caller = calls_by_id.get(source_call_id) if source_call_id else None

                    # Skip if we already have this pair from consecutive logic
                    pair_key = (source_call_id or "", callee.get("call_id", ""))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    # Calculate elapsed time if we found the caller
                    elapsed_ms = None
                    caller_input = None
                    caller_output = None
                    caller_duration_ms = None

                    if caller:
                        caller_input = caller.get("input_args")
                        caller_output = caller.get("output_result")
                        caller_duration_ms = caller.get("duration_ms")

                        if caller.get("end_time") and callee.get("start_time"):
                            try:
                                caller_end = datetime.fromisoformat(caller["end_time"])
                                callee_start = datetime.fromisoformat(callee["start_time"])
                                elapsed_ms = int(
                                    (callee_start - caller_end).total_seconds() * 1000
                                )
                            except ValueError:
                                pass

                    transitions.append(
                        {
                            "caller": from_tool,
                            "caller_input": caller_input,
                            "caller_output": caller_output,
                            "callee": callee_name,
                            "callee_input": callee.get("input_args"),
                            "callee_output": callee.get("output_result"),
                            "elapsed_ms": elapsed_ms,
                            "caller_duration_ms": caller_duration_ms,
                            "callee_duration_ms": callee.get("duration_ms"),
                            "timestamp": callee.get("start_time"),
                            "is_consecutive": False,  # Non-consecutive causal transition
                            "is_causal": True,
                            "causal_match_type": triggered_by.get("match_type"),
                            "causal_matched_value": triggered_by.get("matched_value"),
                        }
                    )

            if len(transitions) >= limit:
                break

        # Sort: causal first, then by timestamp
        transitions.sort(key=lambda t: (not t.get("is_causal", False), t.get("timestamp", "")))

        return transitions[:limit]

    def get_causal_statistics(self) -> dict[str, Any]:
        """Get statistics about causal relationships between tool calls.

        Returns:
            Dict with causal chain statistics:
            - total_causal_links: Number of calls with detected causal links
            - by_match_type: Counts by match type (parameter vs tool_suggestion)
            - top_causal_pairs: Most common source->target causal pairs
            - causal_chains: Example causal chains
        """
        calls = self.get_recent_calls(limit=500)

        total_links = 0
        by_match_type: dict[str, int] = {"parameter": 0, "tool_suggestion": 0}
        causal_pairs: dict[str, int] = {}  # "source:target" -> count

        for call in calls:
            triggered_by = call.get("triggered_by")
            if triggered_by:
                total_links += 1
                match_type = triggered_by.get("match_type", "unknown")
                by_match_type[match_type] = by_match_type.get(match_type, 0) + 1

                # Track causal pairs
                source_tool = triggered_by.get("tool_name", "unknown")
                target_tool = call.get("tool_name", "unknown")
                pair_key = f"{source_tool}:{target_tool}"
                causal_pairs[pair_key] = causal_pairs.get(pair_key, 0) + 1

        # Get top causal pairs
        sorted_pairs = sorted(causal_pairs.items(), key=lambda x: x[1], reverse=True)
        top_pairs = [
            {"source": pair.split(":")[0], "target": pair.split(":")[1], "count": count}
            for pair, count in sorted_pairs[:10]
        ]

        return {
            "total_causal_links": total_links,
            "total_calls": len(calls),
            "causal_rate": round(total_links / len(calls) * 100, 1) if calls else 0,
            "by_match_type": by_match_type,
            "top_causal_pairs": top_pairs,
        }

    def get_calls_with_causal_info(
        self,
        limit: int = 100,
        only_with_links: bool = False,
    ) -> list[dict[str, Any]]:
        """Get recent calls with causal relationship information.

        Args:
            limit: Maximum number of calls to return.
            only_with_links: If True, only return calls that have causal links.

        Returns:
            List of call records with triggered_by and suggested_tools fields.
        """
        calls = self.get_recent_calls(limit=limit * 2 if only_with_links else limit)

        if only_with_links:
            calls = [c for c in calls if c.get("triggered_by")][:limit]

        return calls
