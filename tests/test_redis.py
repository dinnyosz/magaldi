"""Tests for the Redis repository module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from shared.db.redis import (
    RedisEmbeddingJobRepository,
    RedisFeatureJobRepository,
    RedisLabelingJobRepository,
    RedisRepository,
    RedisSubfeatureJobRepository,
    RedisSubfeatureLabelingJobRepository,
    RedisSummarizationJobRepository,
    RedisSummaryStore,
    _key,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_config():
    """Create a mock MagaldiConfig."""
    config = MagicMock()
    config.redis.host = "localhost"
    config.redis.port = 6379
    config.redis.db = 0
    config.redis.password = None
    return config


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client."""
    return MagicMock()


# =============================================================================
# KEY FUNCTION TESTS
# =============================================================================


class TestKeyFunction:
    """Tests for _key utility function."""

    def test_formats_key_correctly(self):
        """Test that key template is formatted correctly."""
        template = "magaldi:test:{scope}:{repository}:{username}"
        result = _key(template, "my_scope", "my_repo", "my_user")

        assert result == "magaldi:test:my_scope:my_repo:my_user"

    def test_handles_special_characters(self):
        """Test formatting with special characters."""
        result = _key(
            "prefix:{scope}:{repository}:{username}",
            "scope-1",
            "repo.name",
            "user_name",
        )

        assert result == "prefix:scope-1:repo.name:user_name"


# =============================================================================
# BASE REPOSITORY TESTS
# =============================================================================


class TestRedisRepository:
    """Tests for RedisRepository base class."""

    def test_init_uses_provided_config(self, mock_config):
        """Test that provided config is used."""
        repo = RedisRepository(mock_config)

        assert repo._config == mock_config
        assert repo._client is None

    def test_init_gets_config_if_not_provided(self):
        """Test that config is fetched if not provided."""
        with patch("shared.db.redis.base.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config

            repo = RedisRepository()

            mock_get_config.assert_called_once()
            assert repo._config == mock_config

    def test_get_client_creates_connection(self, mock_config):
        """Test that _get_client creates Redis connection."""
        repo = RedisRepository(mock_config)

        with patch("shared.db.redis.base.redis.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_redis.return_value = mock_client

            result = repo._get_client()

            mock_redis.assert_called_once_with(
                host="localhost",
                port=6379,
                db=0,
                password=None,
                decode_responses=True,
            )
            assert result == mock_client

    def test_get_client_reuses_connection(self, mock_config):
        """Test that _get_client reuses existing connection."""
        repo = RedisRepository(mock_config)

        with patch("shared.db.redis.base.redis.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_redis.return_value = mock_client

            # First call creates connection
            repo._get_client()
            # Second call reuses it
            repo._get_client()

            # Redis should only be called once
            assert mock_redis.call_count == 1

    def test_close_closes_connection(self, mock_config):
        """Test that close closes Redis connection."""
        repo = RedisRepository(mock_config)
        mock_client = MagicMock()
        repo._client = mock_client

        repo.close()

        mock_client.close.assert_called_once()
        assert repo._client is None

    def test_close_handles_no_connection(self, mock_config):
        """Test that close handles missing connection gracefully."""
        repo = RedisRepository(mock_config)
        repo._client = None

        # Should not raise
        repo.close()


# =============================================================================
# SUMMARIZATION JOB REPOSITORY TESTS
# =============================================================================


class TestRedisSummarizationJobRepository:
    """Tests for RedisSummarizationJobRepository."""

    def test_add_job_stores_job_data(self, mock_config, mock_redis_client):
        """Test that add_job stores job data in Redis."""
        repo = RedisSummarizationJobRepository(mock_config)
        repo._client = mock_redis_client

        repo.add_job(
            element_id="test_element",
            scope="scope",
            repository="repo",
            username="user",
            level=1,
            parent_id="parent_element",
            dependencies_met=True,
            priority=10,
        )

        # Should store job data
        mock_redis_client.hset.assert_called()
        # Should add to queue since dependencies_met=True
        mock_redis_client.zadd.assert_called()

    def test_add_job_not_queued_when_dependencies_not_met(
        self, mock_config, mock_redis_client
    ):
        """Test that job is not queued when dependencies not met."""
        repo = RedisSummarizationJobRepository(mock_config)
        repo._client = mock_redis_client

        repo.add_job(
            element_id="test_element",
            scope="scope",
            repository="repo",
            username="user",
            level=1,
            parent_id="parent_element",
            dependencies_met=False,
        )

        # Should store job data
        assert mock_redis_client.hset.called
        # Should NOT add to queue
        assert not mock_redis_client.zadd.called

    def test_get_job_returns_job_data(self, mock_config, mock_redis_client):
        """Test that get_job returns job data."""
        repo = RedisSummarizationJobRepository(mock_config)
        repo._client = mock_redis_client

        job_data = {"element_id": "test", "status": "pending"}
        mock_redis_client.hget.return_value = json.dumps(job_data)

        result = repo.get_job("test", "scope", "repo", "user")

        assert result == job_data

    def test_get_job_returns_none_when_not_found(
        self, mock_config, mock_redis_client
    ):
        """Test that get_job returns None when job not found."""
        repo = RedisSummarizationJobRepository(mock_config)
        repo._client = mock_redis_client
        mock_redis_client.hget.return_value = None

        result = repo.get_job("nonexistent", "scope", "repo", "user")

        assert result is None

    def test_claim_pending_jobs_claims_jobs(self, mock_config, mock_redis_client):
        """Test that claim_pending_jobs claims and returns jobs."""
        repo = RedisSummarizationJobRepository(mock_config)
        repo._client = mock_redis_client

        # Setup mock responses
        mock_redis_client.zrevrange.return_value = ["job1", "job2"]
        mock_redis_client.zrem.return_value = 1  # Successfully removed

        job_data = {"element_id": "job1", "status": "pending", "level": 1}
        mock_redis_client.hget.return_value = json.dumps(job_data)

        result = repo.claim_pending_jobs("worker1", "scope", "repo", "user", batch_size=5)

        assert len(result) == 2

    def test_mark_completed_updates_job_status(self, mock_config, mock_redis_client):
        """Test that mark_completed updates job status."""
        repo = RedisSummarizationJobRepository(mock_config)
        repo._client = mock_redis_client

        job_data = {"element_id": "test", "status": "running"}
        mock_redis_client.hget.return_value = json.dumps(job_data)
        mock_redis_client.hgetall.return_value = {}

        repo.mark_completed("test", "scope", "repo", "user")

        mock_redis_client.hset.assert_called()
        mock_redis_client.srem.assert_called()

    def test_mark_failed_updates_job_status(self, mock_config, mock_redis_client):
        """Test that mark_failed updates job status with error."""
        repo = RedisSummarizationJobRepository(mock_config)
        repo._client = mock_redis_client

        job_data = {"element_id": "test", "status": "running", "retry_count": 0}
        mock_redis_client.hget.return_value = json.dumps(job_data)

        repo.mark_failed("test", "scope", "repo", "user", "Test error")

        mock_redis_client.hset.assert_called()

    def test_unlock_dependencies_unlocks_child_jobs(
        self, mock_config, mock_redis_client
    ):
        """Test that unlock_dependencies unlocks child jobs."""
        repo = RedisSummarizationJobRepository(mock_config)
        repo._client = mock_redis_client

        # Setup child job that depends on parent
        child_job = {
            "element_id": "child",
            "parent_id": "parent",
            "dependencies_met": False,
            "status": "pending",
            "level": 2,
            "priority": 0,
        }
        mock_redis_client.hgetall.return_value = {"child": json.dumps(child_job)}

        unlocked = repo.unlock_dependencies("parent", "scope", "repo", "user")

        assert unlocked == 1
        mock_redis_client.zadd.assert_called()


# =============================================================================
# EMBEDDING JOB REPOSITORY TESTS
# =============================================================================


class TestRedisEmbeddingJobRepository:
    """Tests for RedisEmbeddingJobRepository."""

    def test_add_job_stores_and_queues(self, mock_config, mock_redis_client):
        """Test that add_job stores job and adds to queue."""
        repo = RedisEmbeddingJobRepository(mock_config)
        repo._client = mock_redis_client

        repo.add_job("test_element", "scope", "repo", "user")

        mock_redis_client.hset.assert_called()
        mock_redis_client.zadd.assert_called()

    def test_get_job_returns_data(self, mock_config, mock_redis_client):
        """Test get_job returns job data."""
        repo = RedisEmbeddingJobRepository(mock_config)
        repo._client = mock_redis_client

        job_data = {"element_id": "test", "status": "pending"}
        mock_redis_client.hget.return_value = json.dumps(job_data)

        result = repo.get_job("test", "scope", "repo", "user")

        assert result == job_data

    def test_claim_pending_jobs_uses_fifo(self, mock_config, mock_redis_client):
        """Test that claim_pending_jobs uses FIFO order."""
        repo = RedisEmbeddingJobRepository(mock_config)
        repo._client = mock_redis_client

        mock_redis_client.zrange.return_value = ["job1"]
        mock_redis_client.zrem.return_value = 1

        job_data = {"element_id": "job1", "status": "pending"}
        mock_redis_client.hget.return_value = json.dumps(job_data)

        result = repo.claim_pending_jobs("worker1", "scope", "repo", "user")

        # zrange (not zrevrange) for FIFO
        mock_redis_client.zrange.assert_called()
        assert len(result) == 1

    def test_mark_completed_updates_status(self, mock_config, mock_redis_client):
        """Test mark_completed updates job status."""
        repo = RedisEmbeddingJobRepository(mock_config)
        repo._client = mock_redis_client

        job_data = {"element_id": "test", "status": "running"}
        mock_redis_client.hget.return_value = json.dumps(job_data)

        repo.mark_completed("test", "scope", "repo", "user")

        mock_redis_client.hset.assert_called()
        mock_redis_client.srem.assert_called()

    def test_mark_failed_increments_retry_count(
        self, mock_config, mock_redis_client
    ):
        """Test mark_failed increments retry count."""
        repo = RedisEmbeddingJobRepository(mock_config)
        repo._client = mock_redis_client

        job_data = {"element_id": "test", "status": "running", "retry_count": 1}
        mock_redis_client.hget.return_value = json.dumps(job_data)

        repo.mark_failed("test", "scope", "repo", "user", "Error message")

        # Verify retry count was incremented
        call_args = mock_redis_client.hset.call_args
        stored_data = json.loads(call_args[0][2])
        assert stored_data["retry_count"] == 2


# =============================================================================
# SUMMARY STORE TESTS
# =============================================================================


class TestRedisSummaryStore:
    """Tests for RedisSummaryStore."""

    def test_store_element(self, mock_config, mock_redis_client):
        """Test store_element stores element data."""
        from magaldi_core.code_parser import CodeElement

        repo = RedisSummaryStore(mock_config)
        repo._client = mock_redis_client

        element = CodeElement(
            element_id="test_id",
            scope="scope",
            repository="repo",
            username="user",
            relative_path="file.py",
            element_type="function",
            name="test_func",
            language="python",
            line_start=1,
            raw_code="def test(): pass",
            level=0,
        )

        repo.store_element(element)

        mock_redis_client.hset.assert_called()

    def test_get_element_returns_element(self, mock_config, mock_redis_client):
        """Test get_element returns CodeElement."""
        repo = RedisSummaryStore(mock_config)
        repo._client = mock_redis_client

        elem_data = {
            "element_id": "test_id",
            "element_type": "function",
            "name": "test_func",
            "raw_code": "def test(): pass",
            "docstring": None,
            "parent_id": None,
            "level": 0,
        }
        mock_redis_client.hget.return_value = json.dumps(elem_data)

        result = repo.get_element("test_id")

        assert result is not None
        assert result.element_id == "test_id"
        assert result.name == "test_func"

    def test_get_element_returns_none_when_not_found(
        self, mock_config, mock_redis_client
    ):
        """Test get_element returns None when not found."""
        repo = RedisSummaryStore(mock_config)
        repo._client = mock_redis_client
        mock_redis_client.hget.return_value = None

        result = repo.get_element("nonexistent")

        assert result is None

    def test_store_summary(self, mock_config, mock_redis_client):
        """Test store_summary stores summary."""
        repo = RedisSummaryStore(mock_config)
        repo._client = mock_redis_client

        repo.store_summary("test_id", "This is a summary.")

        mock_redis_client.hset.assert_called_with(
            RedisSummaryStore.SUMMARIES_KEY, "test_id", "This is a summary."
        )

    def test_get_summary(self, mock_config, mock_redis_client):
        """Test get_summary returns summary."""
        repo = RedisSummaryStore(mock_config)
        repo._client = mock_redis_client
        mock_redis_client.hget.return_value = "Test summary."

        result = repo.get_summary("test_id")

        assert result == "Test summary."

    def test_get_parent_summaries_returns_parent_summary(
        self, mock_config, mock_redis_client
    ):
        """Test get_parent_summaries returns parent's summary."""
        repo = RedisSummaryStore(mock_config)
        repo._client = mock_redis_client

        # Setup element with parent
        element = MagicMock()
        element.element_id = "child_id"

        child_data = {
            "element_id": "child_id",
            "parent_id": "parent_id",
            "element_type": "method",
        }
        parent_data = {
            "element_id": "parent_id",
            "element_type": "class",
        }

        # Mock hget to return different values based on key AND field
        def hget_side_effect(key, field):
            if key == RedisSummaryStore.ELEMENTS_KEY:
                if field == "child_id":
                    return json.dumps(child_data)
                elif field == "parent_id":
                    return json.dumps(parent_data)
            elif key == RedisSummaryStore.SUMMARIES_KEY and field == "parent_id":
                return "Parent summary."
            return None

        mock_redis_client.hget.side_effect = hget_side_effect

        result = repo.get_parent_summaries(element)

        assert "class" in result
        assert result["class"] == "Parent summary."


# =============================================================================
# FEATURE JOB REPOSITORY TESTS
# =============================================================================


class TestRedisFeatureJobRepository:
    """Tests for RedisFeatureJobRepository."""

    def test_add_job_stores_and_queues(self, mock_config, mock_redis_client):
        """Test add_job stores job and adds to queue."""
        repo = RedisFeatureJobRepository(mock_config)
        repo._client = mock_redis_client

        repo.add_job("feature_1", "scope", "repo", "user", "auth_feature")

        mock_redis_client.hset.assert_called()
        mock_redis_client.zadd.assert_called()

    def test_get_job_returns_data(self, mock_config, mock_redis_client):
        """Test get_job returns job data."""
        repo = RedisFeatureJobRepository(mock_config)
        repo._client = mock_redis_client

        job_data = {"feature_id": "f1", "label": "auth", "status": "pending"}
        mock_redis_client.hget.return_value = json.dumps(job_data)

        result = repo.get_job("f1", "scope", "repo", "user")

        assert result == job_data

    def test_mark_running_updates_status(self, mock_config, mock_redis_client):
        """Test mark_running updates job status."""
        repo = RedisFeatureJobRepository(mock_config)
        repo._client = mock_redis_client

        job_data = {"feature_id": "f1", "status": "pending"}
        mock_redis_client.hget.return_value = json.dumps(job_data)

        repo.mark_running("f1", "scope", "repo", "user")

        mock_redis_client.hset.assert_called()
        mock_redis_client.sadd.assert_called()
        mock_redis_client.zrem.assert_called()

    def test_mark_completed_updates_status(self, mock_config, mock_redis_client):
        """Test mark_completed updates job status."""
        repo = RedisFeatureJobRepository(mock_config)
        repo._client = mock_redis_client

        job_data = {"feature_id": "f1", "status": "running"}
        mock_redis_client.hget.return_value = json.dumps(job_data)

        repo.mark_completed("f1", "scope", "repo", "user")

        mock_redis_client.hset.assert_called()
        mock_redis_client.srem.assert_called()

    def test_mark_failed_updates_status_with_error(
        self, mock_config, mock_redis_client
    ):
        """Test mark_failed updates status with error message."""
        repo = RedisFeatureJobRepository(mock_config)
        repo._client = mock_redis_client

        job_data = {"feature_id": "f1", "status": "running"}
        mock_redis_client.hget.return_value = json.dumps(job_data)

        repo.mark_failed("f1", "scope", "repo", "user", "Error occurred")

        call_args = mock_redis_client.hset.call_args
        stored_data = json.loads(call_args[0][2])
        assert stored_data["status"] == "failed"
        assert stored_data["error_message"] == "Error occurred"


# =============================================================================
# LABELING JOB REPOSITORY TESTS
# =============================================================================


class TestRedisLabelingJobRepository:
    """Tests for RedisLabelingJobRepository."""

    def test_add_job_stores_and_queues(self, mock_config, mock_redis_client):
        """Test add_job stores job and adds to queue."""
        repo = RedisLabelingJobRepository(mock_config)
        repo._client = mock_redis_client

        repo.add_job(1, "scope", "repo", "user")

        mock_redis_client.hset.assert_called()
        mock_redis_client.zadd.assert_called()

    def test_get_job_returns_data(self, mock_config, mock_redis_client):
        """Test get_job returns job data."""
        repo = RedisLabelingJobRepository(mock_config)
        repo._client = mock_redis_client

        job_data = {"cluster_id": 1, "status": "pending"}
        mock_redis_client.hget.return_value = json.dumps(job_data)

        result = repo.get_job(1, "scope", "repo", "user")

        assert result == job_data

    def test_mark_running_updates_status(self, mock_config, mock_redis_client):
        """Test mark_running updates job status."""
        repo = RedisLabelingJobRepository(mock_config)
        repo._client = mock_redis_client

        job_data = {"cluster_id": 1, "status": "pending"}
        mock_redis_client.hget.return_value = json.dumps(job_data)

        repo.mark_running(1, "scope", "repo", "user")

        mock_redis_client.hset.assert_called()

    def test_mark_completed_updates_status(self, mock_config, mock_redis_client):
        """Test mark_completed updates job status."""
        repo = RedisLabelingJobRepository(mock_config)
        repo._client = mock_redis_client

        job_data = {"cluster_id": 1, "status": "running"}
        mock_redis_client.hget.return_value = json.dumps(job_data)

        repo.mark_completed(1, "scope", "repo", "user")

        mock_redis_client.hset.assert_called()

    def test_mark_failed_updates_status(self, mock_config, mock_redis_client):
        """Test mark_failed updates status with error."""
        repo = RedisLabelingJobRepository(mock_config)
        repo._client = mock_redis_client

        job_data = {"cluster_id": 1, "status": "running"}
        mock_redis_client.hget.return_value = json.dumps(job_data)

        repo.mark_failed(1, "scope", "repo", "user", "Error")

        mock_redis_client.hset.assert_called()


# =============================================================================
# SUBFEATURE JOB REPOSITORY TESTS
# =============================================================================


class TestRedisSubfeatureJobRepository:
    """Tests for RedisSubfeatureJobRepository."""

    def test_add_job_stores_and_queues(self, mock_config, mock_redis_client):
        """Test add_job stores job with parent label."""
        repo = RedisSubfeatureJobRepository(mock_config)
        repo._client = mock_redis_client

        repo.add_job(
            "subfeature_1",
            "scope",
            "repo",
            "user",
            "sub_label",
            "parent_label",
        )

        mock_redis_client.hset.assert_called()
        mock_redis_client.zadd.assert_called()

    def test_get_job_returns_data(self, mock_config, mock_redis_client):
        """Test get_job returns subfeature job data."""
        repo = RedisSubfeatureJobRepository(mock_config)
        repo._client = mock_redis_client

        job_data = {
            "subfeature_id": "sf1",
            "label": "sub",
            "parent_label": "parent",
            "status": "pending",
        }
        mock_redis_client.hget.return_value = json.dumps(job_data)

        result = repo.get_job("sf1", "scope", "repo", "user")

        assert result == job_data

    def test_mark_running_updates_status(self, mock_config, mock_redis_client):
        """Test mark_running updates job status."""
        repo = RedisSubfeatureJobRepository(mock_config)
        repo._client = mock_redis_client

        job_data = {"subfeature_id": "sf1", "status": "pending"}
        mock_redis_client.hget.return_value = json.dumps(job_data)

        repo.mark_running("sf1", "scope", "repo", "user")

        mock_redis_client.hset.assert_called()
        mock_redis_client.sadd.assert_called()

    def test_mark_completed_updates_status(self, mock_config, mock_redis_client):
        """Test mark_completed updates job status."""
        repo = RedisSubfeatureJobRepository(mock_config)
        repo._client = mock_redis_client

        job_data = {"subfeature_id": "sf1", "status": "running"}
        mock_redis_client.hget.return_value = json.dumps(job_data)

        repo.mark_completed("sf1", "scope", "repo", "user")

        mock_redis_client.hset.assert_called()
        mock_redis_client.srem.assert_called()

    def test_mark_failed_updates_status(self, mock_config, mock_redis_client):
        """Test mark_failed updates status with error."""
        repo = RedisSubfeatureJobRepository(mock_config)
        repo._client = mock_redis_client

        job_data = {"subfeature_id": "sf1", "status": "running"}
        mock_redis_client.hget.return_value = json.dumps(job_data)

        repo.mark_failed("sf1", "scope", "repo", "user", "Error message")

        mock_redis_client.hset.assert_called()


# =============================================================================
# SUBFEATURE LABELING JOB REPOSITORY TESTS
# =============================================================================


class TestRedisSubfeatureLabelingJobRepository:
    """Tests for RedisSubfeatureLabelingJobRepository."""

    def test_add_job_stores_and_queues(self, mock_config, mock_redis_client):
        """Test add_job stores labeling job."""
        repo = RedisSubfeatureLabelingJobRepository(mock_config)
        repo._client = mock_redis_client

        repo.add_job("parent_feature", 5, "scope", "repo", "user")

        mock_redis_client.hset.assert_called()
        mock_redis_client.zadd.assert_called()

    def test_get_job_returns_data(self, mock_config, mock_redis_client):
        """Test get_job returns labeling job data."""
        repo = RedisSubfeatureLabelingJobRepository(mock_config)
        repo._client = mock_redis_client

        job_data = {
            "parent_label": "feature",
            "cluster_count": 5,
            "status": "pending",
        }
        mock_redis_client.hget.return_value = json.dumps(job_data)

        result = repo.get_job("feature", "scope", "repo", "user")

        assert result == job_data

    def test_mark_running_updates_status(self, mock_config, mock_redis_client):
        """Test mark_running updates job status."""
        repo = RedisSubfeatureLabelingJobRepository(mock_config)
        repo._client = mock_redis_client

        job_data = {"parent_label": "feature", "status": "pending"}
        mock_redis_client.hget.return_value = json.dumps(job_data)

        repo.mark_running("feature", "scope", "repo", "user")

        mock_redis_client.hset.assert_called()
        mock_redis_client.sadd.assert_called()

    def test_mark_completed_updates_status(self, mock_config, mock_redis_client):
        """Test mark_completed updates job status."""
        repo = RedisSubfeatureLabelingJobRepository(mock_config)
        repo._client = mock_redis_client

        job_data = {"parent_label": "feature", "status": "running"}
        mock_redis_client.hget.return_value = json.dumps(job_data)

        repo.mark_completed("feature", "scope", "repo", "user")

        mock_redis_client.hset.assert_called()
        mock_redis_client.srem.assert_called()

    def test_mark_failed_updates_status(self, mock_config, mock_redis_client):
        """Test mark_failed updates status with error."""
        repo = RedisSubfeatureLabelingJobRepository(mock_config)
        repo._client = mock_redis_client

        job_data = {"parent_label": "feature", "status": "running"}
        mock_redis_client.hget.return_value = json.dumps(job_data)

        repo.mark_failed("feature", "scope", "repo", "user", "Error")

        mock_redis_client.hset.assert_called()
