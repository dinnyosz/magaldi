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
        with patch("shared.db.redis.base.redis.Redis", return_value=mock_redis_client):
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
        import json
        from datetime import datetime

        # Simulate a previous tool in the session (JSON format with end_time)
        prev_session_data = json.dumps({
            "tool": "search_code",
            "end_time": datetime.now().isoformat(),
        })
        mock_redis_client.get.return_value = prev_session_data

        repo.record_tool_call("get_element", session_id="test-session")

        # Should record the transition
        transition_calls = [
            call for call in mock_redis_client.hincrby.call_args_list
            if "transitions" in str(call) or "search_code:get_element" in str(call)
        ]
        assert len(transition_calls) >= 1

    def test_record_tool_call_updates_session(self, repo, mock_redis_client):
        """Test that session is updated with current tool."""
        import json

        repo.record_tool_call("search_code", session_id="test-session")

        # Should set the session key with TTL
        mock_redis_client.setex.assert_called_once()
        args = mock_redis_client.setex.call_args[0]
        assert "test-session" in args[0]
        # Check the JSON data contains the tool
        session_data = json.loads(args[2])
        assert session_data["tool"] == "search_code"

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
        from magaldi_mcp.server import MagaldiMCPServer

        with patch("magaldi_mcp.server.get_config", return_value=mock_config):
            server = MagaldiMCPServer(enable_analytics=False)

            assert server._session_id is not None
            assert len(server._session_id) > 0

    def test_server_analytics_can_be_disabled(self, mock_config):
        """Test that analytics can be disabled."""
        from magaldi_mcp.server import MagaldiMCPServer

        with patch("magaldi_mcp.server.get_config", return_value=mock_config):
            server = MagaldiMCPServer(enable_analytics=False)

            assert server._enable_analytics is False
            assert server._get_analytics_repo() is None

    def test_record_tool_call_handles_errors_gracefully(self, mock_config):
        """Test that analytics errors don't break tool execution."""
        from magaldi_mcp.server import MagaldiMCPServer

        with patch("magaldi_mcp.server.get_config", return_value=mock_config):
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


class TestCausalRelationshipTracking:
    """Tests for causal relationship detection between MCP tool calls."""

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
        client.hgetall.return_value = {}
        client.get.return_value = None
        client.lrange.return_value = []
        return client

    @pytest.fixture
    def repo(self, mock_config, mock_redis_client):
        """Create repository with mocked Redis."""
        with patch("shared.db.redis.base.redis.Redis", return_value=mock_redis_client):
            repo = RedisMCPAnalyticsRepository(mock_config)
            repo._client = mock_redis_client
            return repo

    def test_extract_trackable_values_hash_ids(self, repo):
        """Test extraction of hash_id values from output."""
        output = "Found element with hash_id: a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
        values = repo._extract_trackable_values(output)

        assert "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2" in values

    def test_extract_trackable_values_library_ids(self, repo):
        """Test extraction of library IDs (Context7 format)."""
        output = "Use library /mongodb/docs or /vercel/next.js/v14.3.0"
        values = repo._extract_trackable_values(output)

        assert "/mongodb/docs" in values
        assert "/vercel/next.js/v14.3.0" in values

    def test_extract_trackable_values_file_paths(self, repo):
        """Test extraction of file paths."""
        output = "File found at src/utils/helper.py and tests/test_main.ts"
        values = repo._extract_trackable_values(output)

        assert "src/utils/helper.py" in values
        assert "tests/test_main.ts" in values

    def test_extract_trackable_values_element_ids(self, repo):
        """Test extraction of element IDs with colons."""
        output = "Element: magaldi:repo:main:src/test.py:function:foo:10"
        values = repo._extract_trackable_values(output)

        assert "magaldi:repo:main:src/test.py:function:foo:10" in values

    def test_extract_trackable_values_filters_short(self, repo):
        """Test that short values are filtered out."""
        output = "Some short values: abc, def, 12345"
        values = repo._extract_trackable_values(output)

        # Short values should be excluded (< 8 chars)
        assert "abc" not in values
        assert "def" not in values

    def test_detect_tool_mentions(self, repo):
        """Test detection of tool names in output."""
        output = "You should use get_element to see the code, or try find_usages for references."
        mentioned = repo._detect_tool_mentions(output)

        assert "get_element" in mentioned
        assert "find_usages" in mentioned

    def test_detect_tool_mentions_kebab_case(self, repo):
        """Test detection of kebab-case tool names."""
        output = "Call resolve-library-id first, then query-docs."
        mentioned = repo._detect_tool_mentions(output)

        assert "resolve-library-id" in mentioned
        assert "query-docs" in mentioned

    def test_detect_tool_mentions_case_insensitive(self, repo):
        """Test tool name detection is case-insensitive."""
        output = "Use SEARCH_CODE or Search_Features"
        mentioned = repo._detect_tool_mentions(output)

        assert "search_code" in mentioned
        assert "search_features" in mentioned

    def test_find_matching_value_found(self, repo):
        """Test finding a matching value in input args."""
        trackable_values = {
            "/mongodb/docs",
            "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        }
        input_args = {"libraryId": "/mongodb/docs", "query": "authentication"}

        matched = repo._find_matching_value(input_args, trackable_values)
        assert matched == "/mongodb/docs"

    def test_find_matching_value_not_found(self, repo):
        """Test when no matching value is found."""
        trackable_values = {"/mongodb/docs"}
        input_args = {"libraryId": "/other/library", "query": "something"}

        matched = repo._find_matching_value(input_args, trackable_values)
        assert matched is None

    def test_detect_causal_link_no_history(self, repo, mock_redis_client):
        """Test causal detection with no previous calls."""
        mock_redis_client.lrange.return_value = []

        result = repo._detect_causal_link("get_element", "session-123", {"hash_id": "abc"})
        assert result is None

    def test_detect_causal_link_parameter_match(self, repo, mock_redis_client):
        """Test causal detection via parameter matching."""
        import json
        from datetime import datetime

        prev_call = {
            "call_id": "session-123:prev",
            "tool_name": "search_code",
            "output_result": "Found: a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
            "end_time": datetime.now().isoformat(),
        }
        mock_redis_client.lrange.return_value = [json.dumps(prev_call)]

        result = repo._detect_causal_link(
            "get_element",
            "session-123",
            {"hash_id": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"},
        )

        assert result is not None
        assert result["tool_name"] == "search_code"
        assert result["match_type"] == "parameter"
        assert result["matched_value"] == "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"

    def test_detect_causal_link_tool_suggestion(self, repo, mock_redis_client):
        """Test causal detection via tool suggestion."""
        import json
        from datetime import datetime

        prev_call = {
            "call_id": "session-123:prev",
            "tool_name": "search_code",
            "output_result": "Use get_element to see the full code.",
            "end_time": datetime.now().isoformat(),
        }
        mock_redis_client.lrange.return_value = [json.dumps(prev_call)]

        result = repo._detect_causal_link(
            "get_element",
            "session-123",
            {"hash_id": "some-other-id"},  # No parameter match
        )

        assert result is not None
        assert result["tool_name"] == "search_code"
        assert result["match_type"] == "tool_suggestion"
        assert result["matched_value"] is None

    def test_detect_causal_link_expired_window(self, repo, mock_redis_client):
        """Test that old calls outside time window don't create causal links."""
        import json
        from datetime import datetime, timedelta

        # Call from 2 minutes ago (outside 60 second window)
        old_time = (datetime.now() - timedelta(minutes=2)).isoformat()
        prev_call = {
            "call_id": "session-123:prev",
            "tool_name": "search_code",
            "output_result": "Found: a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
            "end_time": old_time,
        }
        mock_redis_client.lrange.return_value = [json.dumps(prev_call)]

        result = repo._detect_causal_link(
            "get_element",
            "session-123",
            {"hash_id": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"},
        )

        assert result is None

    def test_get_causal_statistics(self, repo, mock_redis_client):
        """Test getting causal statistics."""
        import json

        calls = [
            {
                "tool_name": "get_element",
                "triggered_by": {
                    "tool_name": "search_code",
                    "match_type": "parameter",
                },
            },
            {
                "tool_name": "find_usages",
                "triggered_by": {
                    "tool_name": "search_code",
                    "match_type": "tool_suggestion",
                },
            },
            {
                "tool_name": "search_code",
                "triggered_by": None,
            },
        ]
        mock_redis_client.lrange.return_value = [json.dumps(c) for c in calls]

        stats = repo.get_causal_statistics()

        assert stats["total_causal_links"] == 2
        assert stats["total_calls"] == 3
        assert stats["causal_rate"] == pytest.approx(66.7, abs=0.1)
        assert stats["by_match_type"]["parameter"] == 1
        assert stats["by_match_type"]["tool_suggestion"] == 1
        assert len(stats["top_causal_pairs"]) >= 1

    def test_get_calls_with_causal_info(self, repo, mock_redis_client):
        """Test getting calls with causal info."""
        import json

        calls = [
            {
                "call_id": "sess:1",
                "tool_name": "get_element",
                "triggered_by": {"tool_name": "search_code", "match_type": "parameter"},
            },
            {
                "call_id": "sess:2",
                "tool_name": "search_code",
                "triggered_by": None,
            },
        ]
        mock_redis_client.lrange.return_value = [json.dumps(c) for c in calls]

        # Get all calls
        result = repo.get_calls_with_causal_info(limit=10)
        assert len(result) == 2

        # Get only causal calls
        result = repo.get_calls_with_causal_info(limit=10, only_with_links=True)
        assert len(result) == 1
        assert result[0]["tool_name"] == "get_element"
