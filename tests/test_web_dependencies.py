"""Tests for web dependencies module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from magaldi_web.dependencies import (
    check_elasticsearch_health,
    check_llm_health,
    check_redis_health,
    get_cached_config,
    get_es_repository,
    get_redis_queue_stats,
)


# =============================================================================
# GET CACHED CONFIG TESTS
# =============================================================================


class TestGetCachedConfig:
    """Tests for get_cached_config function."""

    def test_returns_existing_config(self, test_config):
        """Test returns existing config from get_config."""
        with patch("magaldi_web.dependencies.get_config", return_value=test_config):
            get_cached_config.cache_clear()
            result = get_cached_config()
            assert result == test_config

    def test_loads_config_on_runtime_error(self, test_config):
        """Test loads config when get_config raises RuntimeError."""
        with (
            patch("magaldi_web.dependencies.get_config", side_effect=RuntimeError("No config")),
            patch("magaldi_web.dependencies.load_config", return_value=test_config),
        ):
            get_cached_config.cache_clear()
            result = get_cached_config()
            assert result == test_config

    def test_config_is_cached(self, test_config):
        """Test config is cached between calls."""
        call_count = 0

        def mock_get_config():
            nonlocal call_count
            call_count += 1
            return test_config

        with patch("magaldi_web.dependencies.get_config", side_effect=mock_get_config):
            get_cached_config.cache_clear()
            get_cached_config()
            get_cached_config()
            get_cached_config()
            # Should only be called once due to caching
            assert call_count == 1


# =============================================================================
# GET ES REPOSITORY TESTS
# =============================================================================


class TestGetEsRepository:
    """Tests for get_es_repository generator."""

    def test_yields_repository(self, test_config):
        """Test yields an ElasticsearchRepository."""
        mock_repo = MagicMock()
        with (
            patch("magaldi_web.dependencies.get_cached_config", return_value=test_config),
            patch("magaldi_web.dependencies.ElasticsearchRepository", return_value=mock_repo),
        ):
            gen = get_es_repository()
            repo = next(gen)
            assert repo == mock_repo

    def test_closes_repository_on_exit(self, test_config):
        """Test closes repository when generator exits."""
        mock_repo = MagicMock()
        with (
            patch("magaldi_web.dependencies.get_cached_config", return_value=test_config),
            patch("magaldi_web.dependencies.ElasticsearchRepository", return_value=mock_repo),
        ):
            gen = get_es_repository()
            next(gen)
            # Exhaust the generator
            try:
                next(gen)
            except StopIteration:
                pass
            mock_repo.close.assert_called_once()

    def test_closes_repository_on_exception(self, test_config):
        """Test closes repository even when exception occurs."""
        mock_repo = MagicMock()
        with (
            patch("magaldi_web.dependencies.get_cached_config", return_value=test_config),
            patch("magaldi_web.dependencies.ElasticsearchRepository", return_value=mock_repo),
        ):
            gen = get_es_repository()
            next(gen)
            # Simulate exception and cleanup
            gen.close()
            mock_repo.close.assert_called_once()


# =============================================================================
# CHECK ELASTICSEARCH HEALTH TESTS
# =============================================================================


class TestCheckElasticsearchHealth:
    """Tests for check_elasticsearch_health function."""

    @pytest.mark.asyncio
    async def test_healthy_cluster(self):
        """Test returns healthy status for healthy cluster."""
        mock_repo = MagicMock()
        mock_client = MagicMock()
        mock_client.cluster.health.return_value = {
            "status": "green",
            "number_of_nodes": 3,
        }
        mock_repo._get_client.return_value = mock_client

        result = await check_elasticsearch_health(mock_repo)

        assert result["status"] == "healthy"
        assert result["cluster_status"] == "green"
        assert result["number_of_nodes"] == 3

    @pytest.mark.asyncio
    async def test_unhealthy_on_exception(self):
        """Test returns unhealthy status on exception."""
        mock_repo = MagicMock()
        mock_repo._get_client.side_effect = Exception("Connection refused")

        result = await check_elasticsearch_health(mock_repo)

        assert result["status"] == "unhealthy"
        assert "Connection refused" in result["error"]

    @pytest.mark.asyncio
    async def test_handles_missing_health_fields(self):
        """Test handles missing fields in health response."""
        mock_repo = MagicMock()
        mock_client = MagicMock()
        mock_client.cluster.health.return_value = {}
        mock_repo._get_client.return_value = mock_client

        result = await check_elasticsearch_health(mock_repo)

        assert result["status"] == "healthy"
        assert result["cluster_status"] == "unknown"
        assert result["number_of_nodes"] == 0


# =============================================================================
# CHECK LLM HEALTH TESTS
# =============================================================================


class TestCheckLlmHealth:
    """Tests for check_llm_health function."""

    @pytest.mark.asyncio
    async def test_healthy_llm(self, test_config):
        """Test returns healthy status for healthy LLM."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2"},
                {"name": "codellama"},
            ]
        }

        with patch("requests.get", return_value=mock_response):
            result = await check_llm_health(test_config)

        assert result["status"] == "healthy"
        assert "llama2" in result["models_loaded"]
        assert "codellama" in result["models_loaded"]

    @pytest.mark.asyncio
    async def test_unhealthy_on_http_error(self, test_config):
        """Test returns unhealthy status on HTTP error."""
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 503

        with patch("requests.get", return_value=mock_response):
            result = await check_llm_health(test_config)

        assert result["status"] == "unhealthy"
        assert "HTTP 503" in result["error"]

    @pytest.mark.asyncio
    async def test_unhealthy_on_exception(self, test_config):
        """Test returns unhealthy status on connection error."""
        with patch("requests.get", side_effect=Exception("Connection refused")):
            result = await check_llm_health(test_config)

        assert result["status"] == "unhealthy"
        assert "Connection refused" in result["error"]

    @pytest.mark.asyncio
    async def test_handles_empty_models_list(self, test_config):
        """Test handles empty models list."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"models": []}

        with patch("requests.get", return_value=mock_response):
            result = await check_llm_health(test_config)

        assert result["status"] == "healthy"
        assert result["models_loaded"] == []

    @pytest.mark.asyncio
    async def test_handles_missing_name_field(self, test_config):
        """Test handles models without name field."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "models": [{"size": 123}, {"name": "test"}]
        }

        with patch("requests.get", return_value=mock_response):
            result = await check_llm_health(test_config)

        assert result["status"] == "healthy"
        assert "" in result["models_loaded"]  # Empty string for missing name
        assert "test" in result["models_loaded"]


# =============================================================================
# CHECK REDIS HEALTH TESTS
# =============================================================================


class TestCheckRedisHealth:
    """Tests for check_redis_health function."""

    @pytest.mark.asyncio
    async def test_healthy_redis(self, test_config):
        """Test returns healthy status for healthy Redis."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True

        with patch("redis.Redis", return_value=mock_redis):
            result = await check_redis_health(test_config)

        assert result["status"] == "healthy"
        mock_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_unhealthy_on_exception(self, test_config):
        """Test returns unhealthy status on connection error."""
        with patch("redis.Redis", side_effect=Exception("Connection refused")):
            result = await check_redis_health(test_config)

        assert result["status"] == "unhealthy"
        assert "Connection refused" in result["error"]

    @pytest.mark.asyncio
    async def test_unhealthy_on_ping_failure(self, test_config):
        """Test returns unhealthy status when ping fails."""
        mock_redis = MagicMock()
        mock_redis.ping.side_effect = Exception("NOAUTH Authentication required")

        with patch("redis.Redis", return_value=mock_redis):
            result = await check_redis_health(test_config)

        assert result["status"] == "unhealthy"
        assert "NOAUTH" in result["error"]


# =============================================================================
# GET REDIS QUEUE STATS TESTS
# =============================================================================


class TestGetRedisQueueStats:
    """Tests for get_redis_queue_stats function."""

    @pytest.mark.asyncio
    async def test_returns_empty_stats_on_error(self, test_config):
        """Test returns empty stats on Redis error."""
        with patch("redis.Redis", side_effect=Exception("Connection refused")):
            result = await get_redis_queue_stats(test_config)

        assert result["total_pending"] == 0
        assert result["total_running"] == 0
        assert result["summarization"] == {}
        assert result["embedding"] == {}

    @pytest.mark.asyncio
    async def test_returns_empty_stats_for_empty_redis(self, test_config):
        """Test returns empty stats when no keys exist."""
        mock_redis = MagicMock()
        mock_redis.scan_iter.return_value = iter([])

        with patch("redis.Redis", return_value=mock_redis):
            result = await get_redis_queue_stats(test_config)

        assert result["total_pending"] == 0
        assert result["total_running"] == 0

    @pytest.mark.asyncio
    async def test_counts_summarization_jobs(self, test_config):
        """Test counts summarization jobs correctly."""
        mock_redis = MagicMock()

        # Mock scan_iter to return summarization keys on first call only
        def scan_iter_side_effect(pattern):
            if "summarization" in pattern:
                return iter(["magaldi:summarization:jobs:scope:repo:user"])
            return iter([])

        mock_redis.scan_iter.side_effect = scan_iter_side_effect
        mock_redis.hvals.return_value = [
            json.dumps({"status": "pending"}),
            json.dumps({"status": "pending"}),
            json.dumps({"status": "completed"}),
        ]
        mock_redis.scard.return_value = 1  # 1 running

        with patch("redis.Redis", return_value=mock_redis):
            result = await get_redis_queue_stats(test_config)

        assert "scope/repo/user" in result["summarization"]
        assert result["summarization"]["scope/repo/user"]["pending"] == 2
        assert result["summarization"]["scope/repo/user"]["running"] == 1
        assert result["total_pending"] == 2
        assert result["total_running"] == 1

    @pytest.mark.asyncio
    async def test_counts_embedding_jobs(self, test_config):
        """Test counts embedding jobs correctly."""
        mock_redis = MagicMock()

        def scan_iter_side_effect(pattern):
            if "embedding" in pattern:
                return iter(["magaldi:embedding:jobs:scope:repo:user"])
            return iter([])

        mock_redis.scan_iter.side_effect = scan_iter_side_effect
        mock_redis.hvals.return_value = [
            json.dumps({"status": "pending"}),
        ]
        mock_redis.scard.return_value = 0

        with patch("redis.Redis", return_value=mock_redis):
            result = await get_redis_queue_stats(test_config)

        assert "scope/repo/user" in result["embedding"]
        assert result["embedding"]["scope/repo/user"]["pending"] == 1

    @pytest.mark.asyncio
    async def test_counts_labeling_jobs(self, test_config):
        """Test counts labeling jobs correctly."""
        mock_redis = MagicMock()

        def scan_iter_side_effect(pattern):
            if pattern == "magaldi:labeling:jobs:*":
                return iter(["magaldi:labeling:jobs:scope:repo:user"])
            return iter([])

        mock_redis.scan_iter.side_effect = scan_iter_side_effect
        mock_redis.hvals.return_value = [json.dumps({"status": "pending"})]
        mock_redis.scard.return_value = 2

        with patch("redis.Redis", return_value=mock_redis):
            result = await get_redis_queue_stats(test_config)

        assert "scope/repo/user" in result["labeling"]
        assert result["labeling"]["scope/repo/user"]["running"] == 2

    @pytest.mark.asyncio
    async def test_counts_feature_jobs(self, test_config):
        """Test counts feature jobs correctly."""
        mock_redis = MagicMock()

        def scan_iter_side_effect(pattern):
            if pattern == "magaldi:feature:jobs:*":
                return iter(["magaldi:feature:jobs:scope:repo:user"])
            return iter([])

        mock_redis.scan_iter.side_effect = scan_iter_side_effect
        mock_redis.hvals.return_value = [json.dumps({"status": "pending"})]
        mock_redis.scard.return_value = 0

        with patch("redis.Redis", return_value=mock_redis):
            result = await get_redis_queue_stats(test_config)

        assert "scope/repo/user" in result["feature"]

    @pytest.mark.asyncio
    async def test_counts_subfeature_jobs(self, test_config):
        """Test counts subfeature jobs correctly."""
        mock_redis = MagicMock()

        def scan_iter_side_effect(pattern):
            if pattern == "magaldi:subfeature:jobs:*":
                return iter(["magaldi:subfeature:jobs:scope:repo:user"])
            return iter([])

        mock_redis.scan_iter.side_effect = scan_iter_side_effect
        mock_redis.hvals.return_value = [json.dumps({"status": "pending"})]
        mock_redis.scard.return_value = 0

        with patch("redis.Redis", return_value=mock_redis):
            result = await get_redis_queue_stats(test_config)

        assert "scope/repo/user" in result["subfeature"]

    @pytest.mark.asyncio
    async def test_counts_subfeature_labeling_jobs(self, test_config):
        """Test counts subfeature labeling jobs correctly."""
        mock_redis = MagicMock()

        def scan_iter_side_effect(pattern):
            if pattern == "magaldi:subfeature_labeling:jobs:*":
                return iter(["magaldi:subfeature_labeling:jobs:scope:repo:user"])
            return iter([])

        mock_redis.scan_iter.side_effect = scan_iter_side_effect
        mock_redis.hvals.return_value = [json.dumps({"status": "pending"})]
        mock_redis.scard.return_value = 0

        with patch("redis.Redis", return_value=mock_redis):
            result = await get_redis_queue_stats(test_config)

        assert "scope/repo/user" in result["subfeature_labeling"]

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self, test_config):
        """Test handles invalid JSON in job data."""
        mock_redis = MagicMock()

        def scan_iter_side_effect(pattern):
            if "summarization" in pattern:
                return iter(["magaldi:summarization:jobs:scope:repo:user"])
            return iter([])

        mock_redis.scan_iter.side_effect = scan_iter_side_effect
        mock_redis.hvals.return_value = [
            "invalid json",
            json.dumps({"status": "pending"}),
        ]
        mock_redis.scard.return_value = 0

        with patch("redis.Redis", return_value=mock_redis):
            result = await get_redis_queue_stats(test_config)

        # Should count only the valid pending job
        assert result["summarization"]["scope/repo/user"]["pending"] == 1

    @pytest.mark.asyncio
    async def test_skips_zero_count_queues(self, test_config):
        """Test doesn't include queues with zero pending and running."""
        mock_redis = MagicMock()

        def scan_iter_side_effect(pattern):
            if "summarization" in pattern:
                return iter(["magaldi:summarization:jobs:scope:repo:user"])
            return iter([])

        mock_redis.scan_iter.side_effect = scan_iter_side_effect
        mock_redis.hvals.return_value = [
            json.dumps({"status": "completed"}),
        ]
        mock_redis.scard.return_value = 0  # No running

        with patch("redis.Redis", return_value=mock_redis):
            result = await get_redis_queue_stats(test_config)

        # Queue should not appear because pending=0 and running=0
        assert "scope/repo/user" not in result["summarization"]

    @pytest.mark.asyncio
    async def test_handles_short_key_parts(self, test_config):
        """Test handles keys with fewer than 6 parts."""
        mock_redis = MagicMock()

        def scan_iter_side_effect(pattern):
            if "summarization" in pattern:
                return iter(["magaldi:summarization:jobs:short"])  # Only 4 parts
            return iter([])

        mock_redis.scan_iter.side_effect = scan_iter_side_effect

        with patch("redis.Redis", return_value=mock_redis):
            result = await get_redis_queue_stats(test_config)

        # Should not fail, just skip invalid keys
        assert result["summarization"] == {}

    @pytest.mark.asyncio
    async def test_closes_redis_connection(self, test_config):
        """Test closes Redis connection after querying."""
        mock_redis = MagicMock()
        mock_redis.scan_iter.return_value = iter([])

        with patch("redis.Redis", return_value=mock_redis):
            await get_redis_queue_stats(test_config)

        mock_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_queues_aggregated(self, test_config):
        """Test multiple queues are aggregated in totals."""
        mock_redis = MagicMock()

        call_count = [0]

        def scan_iter_side_effect(pattern):
            call_count[0] += 1
            if "summarization" in pattern:
                return iter([
                    "magaldi:summarization:jobs:scope1:repo1:user1",
                    "magaldi:summarization:jobs:scope2:repo2:user2",
                ])
            return iter([])

        mock_redis.scan_iter.side_effect = scan_iter_side_effect
        mock_redis.hvals.return_value = [json.dumps({"status": "pending"})]
        mock_redis.scard.return_value = 1

        with patch("redis.Redis", return_value=mock_redis):
            result = await get_redis_queue_stats(test_config)

        # Should have 2 pending (1 each) and 2 running (1 each)
        assert result["total_pending"] == 2
        assert result["total_running"] == 2
