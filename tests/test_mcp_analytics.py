"""Tests for MCP tool usage analytics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shared.db.redis import RedisMCPAnalyticsRepository


class TestRedisMCPAnalyticsRepository:
    """Tests for MCP analytics repository."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = MagicMock()
        config.redis.host = "localhost"
        config.redis.port = 6379
        config.redis.db = 0
        config.redis.password = None
        return config

    @pytest.fixture
    def mock_redis_client(self):
        """Create mock Redis client."""
        client = MagicMock()
        # Default return values
        client.hgetall.return_value = {}
        client.get.return_value = None
        return client

    @pytest.fixture
    def repo(self, mock_config, mock_redis_client):
        """Create repository with mocked Redis."""
        with patch("shared.db.redis.redis.Redis", return_value=mock_redis_client):
            repo = RedisMCPAnalyticsRepository(mock_config)
            repo._client = mock_redis_client
            return repo

    def test_record_tool_call_increments_count(self, repo, mock_redis_client):
        """Test that recording a tool call increments the counter."""
        repo.record_tool_call("search_code")

        # Should increment the tool call count
        mock_redis_client.hincrby.assert_any_call(
            "magaldi:mcp:tool_calls", "search_code", 1
        )

    def test_record_tool_call_tracks_daily(self, repo, mock_redis_client):
        """Test that daily counts are tracked."""
        repo.record_tool_call("search_code")

        # Should have called hincrby for daily key
        daily_calls = [
            call for call in mock_redis_client.hincrby.call_args_list
            if "daily" in str(call) and "calls" in str(call)
        ]
        assert len(daily_calls) == 1

    def test_record_tool_call_tracks_transition(self, repo, mock_redis_client):
        """Test that transitions are tracked when there's a previous tool."""
        # Simulate a previous tool in the session
        mock_redis_client.get.return_value = "search_code"

        repo.record_tool_call("get_element", session_id="test-session")

        # Should record the transition
        transition_calls = [
            call for call in mock_redis_client.hincrby.call_args_list
            if "transitions" in str(call) or "search_code:get_element" in str(call)
        ]
        assert len(transition_calls) >= 1

    def test_record_tool_call_updates_session(self, repo, mock_redis_client):
        """Test that session is updated with current tool."""
        repo.record_tool_call("search_code", session_id="test-session")

        # Should set the session key with TTL
        mock_redis_client.setex.assert_called_once()
        args = mock_redis_client.setex.call_args[0]
        assert "test-session" in args[0]
        assert args[2] == "search_code"

    def test_get_tool_counts(self, repo, mock_redis_client):
        """Test getting tool counts."""
        mock_redis_client.hgetall.return_value = {
            "search_code": "100",
            "get_element": "50",
        }

        counts = repo.get_tool_counts()

        assert counts == {"search_code": 100, "get_element": 50}
        mock_redis_client.hgetall.assert_called_with("magaldi:mcp:tool_calls")

    def test_get_tool_transitions(self, repo, mock_redis_client):
        """Test getting transition matrix."""
        mock_redis_client.hgetall.return_value = {
            "search_code:get_element": "45",
            "search_code:find_usages": "20",
            "get_element:get_context": "30",
        }

        matrix = repo.get_tool_transitions()

        assert matrix == {
            "search_code": {"get_element": 45, "find_usages": 20},
            "get_element": {"get_context": 30},
        }

    def test_get_top_tools(self, repo, mock_redis_client):
        """Test getting top tools by usage."""
        mock_redis_client.hgetall.return_value = {
            "search_code": "100",
            "get_element": "50",
            "find_usages": "75",
        }

        top = repo.get_top_tools(limit=2)

        assert len(top) == 2
        assert top[0] == ("search_code", 100)
        assert top[1] == ("find_usages", 75)

    def test_get_top_transitions(self, repo, mock_redis_client):
        """Test getting top transitions."""
        mock_redis_client.hgetall.return_value = {
            "search_code:get_element": "45",
            "search_code:find_usages": "20",
            "get_element:get_context": "30",
        }

        top = repo.get_top_transitions(limit=2)

        assert len(top) == 2
        assert top[0] == ("search_code", "get_element", 45)
        assert top[1] == ("get_element", "get_context", 30)

    def test_clear_analytics(self, repo, mock_redis_client):
        """Test clearing analytics data."""
        mock_redis_client.scan_iter.return_value = []

        repo.clear_analytics()

        mock_redis_client.delete.assert_any_call("magaldi:mcp:tool_calls")
        mock_redis_client.delete.assert_any_call("magaldi:mcp:tool_transitions")

    def test_get_daily_counts(self, repo, mock_redis_client):
        """Test getting daily counts."""
        mock_redis_client.hgetall.return_value = {
            "search_code": "10",
            "get_element": "5",
        }

        counts = repo.get_daily_counts("2024-01-15")

        assert counts == {"search_code": 10, "get_element": 5}
        mock_redis_client.hgetall.assert_called_with("magaldi:mcp:daily:2024-01-15:calls")


class TestMCPServerAnalyticsIntegration:
    """Tests for MCP server analytics integration."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = MagicMock()
        config.redis.host = "localhost"
        config.redis.port = 6379
        config.redis.db = 0
        config.redis.password = None
        config.llm.get_embed_model.return_value = MagicMock(
            url="http://localhost:11434",
            name="test-model",
            provider="ollama",
            api_key=None,
        )
        return config

    def test_server_creates_session_id(self, mock_config):
        """Test that server creates a unique session ID."""
        with patch("shared.config.get_config", return_value=mock_config):
            from magaldi_mcp.server import MagaldiMCPServer

            server = MagaldiMCPServer(enable_analytics=False)

            assert server._session_id is not None
            assert len(server._session_id) > 0

    def test_server_analytics_can_be_disabled(self, mock_config):
        """Test that analytics can be disabled."""
        with patch("shared.config.get_config", return_value=mock_config):
            from magaldi_mcp.server import MagaldiMCPServer

            server = MagaldiMCPServer(enable_analytics=False)

            assert server._enable_analytics is False
            assert server._get_analytics_repo() is None

    def test_record_tool_call_handles_errors_gracefully(self, mock_config):
        """Test that analytics errors don't break tool execution."""
        with patch("shared.config.get_config", return_value=mock_config):
            from magaldi_mcp.server import MagaldiMCPServer

            server = MagaldiMCPServer(enable_analytics=True)

            # Mock analytics repo to raise an exception
            mock_repo = MagicMock()
            mock_repo.record_tool_call.side_effect = Exception("Redis error")
            server._analytics_repo = mock_repo
            server._enable_analytics = True

            # Should not raise an exception
            server._record_tool_call("test_tool")


class TestMCPAnalyticsAPIEndpoint:
    """Tests for MCP analytics admin API endpoint."""

    @pytest.fixture
    def mock_analytics_repo(self):
        """Create mock analytics repository."""
        repo = MagicMock()
        repo.get_tool_counts.return_value = {
            "search_code": 100,
            "get_element": 50,
            "find_usages": 30,
        }
        repo.get_daily_counts.return_value = {
            "search_code": 10,
            "get_element": 5,
        }
        repo.get_top_transitions.return_value = [
            ("search_code", "get_element", 45),
            ("get_element", "get_context", 20),
        ]
        repo.get_tool_transitions.return_value = {
            "search_code": {"get_element": 45, "find_usages": 10},
            "get_element": {"get_context": 20},
        }
        return repo

    @pytest.mark.asyncio
    async def test_get_mcp_analytics(self, mock_analytics_repo):
        """Test the MCP analytics endpoint."""
        from magaldi_web.routes.admin import get_mcp_analytics

        with patch(
            "magaldi_web.routes.admin.RedisMCPAnalyticsRepository",
            return_value=mock_analytics_repo,
        ):
            response = await get_mcp_analytics()

            assert response.total_calls == 180  # 100 + 50 + 30
            assert response.unique_tools == 3
            assert response.today_calls == 15  # 10 + 5
            assert len(response.tool_usage) == 3
            assert response.tool_usage[0].tool_name == "search_code"
            assert response.tool_usage[0].call_count == 100
            assert len(response.top_transitions) == 2

    @pytest.mark.asyncio
    async def test_get_mcp_analytics_empty(self):
        """Test endpoint with no data."""
        from magaldi_web.routes.admin import get_mcp_analytics

        mock_repo = MagicMock()
        mock_repo.get_tool_counts.return_value = {}
        mock_repo.get_daily_counts.return_value = {}
        mock_repo.get_top_transitions.return_value = []
        mock_repo.get_tool_transitions.return_value = {}

        with patch(
            "magaldi_web.routes.admin.RedisMCPAnalyticsRepository",
            return_value=mock_repo,
        ):
            response = await get_mcp_analytics()

            assert response.total_calls == 0
            assert response.unique_tools == 0
            assert response.today_calls == 0
            assert len(response.tool_usage) == 0
            assert len(response.top_transitions) == 0

    @pytest.mark.asyncio
    async def test_get_mcp_analytics_redis_error(self):
        """Test endpoint handles Redis errors gracefully."""
        from magaldi_web.routes.admin import get_mcp_analytics

        with patch(
            "magaldi_web.routes.admin.RedisMCPAnalyticsRepository",
            side_effect=Exception("Redis connection failed"),
        ):
            response = await get_mcp_analytics()

            # Should return empty response, not raise
            assert response.total_calls == 0
            assert response.unique_tools == 0

    @pytest.mark.asyncio
    async def test_clear_mcp_analytics(self):
        """Test clearing analytics data."""
        from magaldi_web.routes.admin import clear_mcp_analytics

        mock_repo = MagicMock()

        with patch(
            "magaldi_web.routes.admin.RedisMCPAnalyticsRepository",
            return_value=mock_repo,
        ):
            response = await clear_mcp_analytics()

            assert response["status"] == "cleared"
            mock_repo.clear_analytics.assert_called_once()
