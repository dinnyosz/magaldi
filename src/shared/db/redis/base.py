"""Base Redis repository with connection management.

This module provides the foundational Redis connection handling and key generation
utilities used by all Redis repositories in Magaldi.
"""

from __future__ import annotations

from typing import cast

import redis

from shared.config import MagaldiConfig, get_config


# =============================================================================
# REDIS KEY FACTORY
# =============================================================================
# Centralized key generation for all job types.
# Format: magaldi:{job_type}:{key_type}:{scope}:{repository}:{username}


class JobType:
    """Job type constants for Redis key generation."""

    SUMMARIZATION = "summarization"
    EMBEDDING = "embedding"
    FEATURE = "feature"
    LABELING = "labeling"
    SUBFEATURE = "subfeature"
    SUBFEATURE_LABELING = "subfeature_labeling"


class KeyType:
    """Key type constants for Redis key generation."""

    QUEUE = "queue"
    RUNNING = "running"
    JOBS = "jobs"


def _make_key(job_type: str, key_type: str, scope: str, repository: str, username: str) -> str:
    """Generate a Redis key for a job type and key type.

    Args:
        job_type: One of JobType constants (summarization, embedding, etc.)
        key_type: One of KeyType constants (queue, running, jobs)
        scope: Repository scope.
        repository: Repository name.
        username: User who owns this job.

    Returns:
        Formatted Redis key string.
    """
    return f"magaldi:{job_type}:{key_type}:{scope}:{repository}:{username}"


# Legacy constants for backwards compatibility
# These are used by existing code that imports them directly
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
    """Format a key template with scope, repository, and username.

    DEPRECATED: Use _make_key() for new code.
    """
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

    # Typed wrapper methods for Redis operations.
    # These cast the union types (Awaitable[T] | T) to sync types.
    # Override these in an async subclass to use await instead.

    def _redis_get(self, key: str) -> str | None:
        """Get a string value from Redis."""
        return cast(str | None, self._get_client().get(key))

    def _redis_hget(self, name: str, key: str) -> str | None:
        """Get a field from a Redis hash."""
        return cast(str | None, self._get_client().hget(name, key))

    def _redis_hgetall(self, name: str) -> dict[str, str]:
        """Get all fields from a Redis hash."""
        return cast(dict[str, str], self._get_client().hgetall(name))

    def _redis_lrange(self, name: str, start: int, end: int) -> list[str]:
        """Get a range of elements from a Redis list."""
        return cast(list[str], self._get_client().lrange(name, start, end))

    def _redis_zrevrange(self, name: str, start: int, end: int) -> list[str]:
        """Get a range of elements from a Redis sorted set (highest scores first)."""
        return cast(list[str], self._get_client().zrevrange(name, start, end))

    def _redis_zrange(self, name: str, start: int, end: int) -> list[str]:
        """Get a range of elements from a Redis sorted set (lowest scores first)."""
        return cast(list[str], self._get_client().zrange(name, start, end))
