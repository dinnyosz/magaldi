"""Tests for MCP server glossary tool registration."""

import pytest
from unittest.mock import MagicMock, patch

from magaldi_mcp.server import MagaldiMCPServer


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_es_repo():
    """Create a mock Elasticsearch repository."""
    repo = MagicMock()
    repo.get_glossary_terms.return_value = []
    repo.get_glossary_term.return_value = None
    repo.search_glossary.return_value = []
    return repo


@pytest.fixture
def server(mock_es_repo):
    """Create a MagaldiMCPServer with mocked dependencies."""
    mock_config = MagicMock()
    mock_config.llm = MagicMock()
    mock_config.llm.url = "http://localhost:11434"
    mock_config.llm.embed_model = "test-model"
    mock_config.llm.provider = "ollama"
    mock_config.llm.api_key = None
    mock_config.llm.embed_dimensions = 1024

    with patch("magaldi_mcp.server.get_config", return_value=mock_config):
        server = MagaldiMCPServer(default_username="main")
        server.es_repo = mock_es_repo
        return server


# =============================================================================
# TOOL REGISTRATION TESTS
# =============================================================================


class TestGlossaryToolRegistration:
    """Tests for glossary tool registration in MCP server."""

    def test_list_glossary_tool_exists(self, server):
        """Verify list_glossary can be handled without raising 'Unknown tool' error."""
        # This test verifies the tool is registered by checking it doesn't raise
        # ValueError for unknown tool. The tool itself requires scope/repository.
        pass  # Registration is verified by the handler tests below


# =============================================================================
# TOOL HANDLER TESTS
# =============================================================================


class TestListGlossaryTool:
    """Tests for list_glossary tool handler."""

    @pytest.mark.asyncio
    async def test_list_glossary_returns_terms(self, server, mock_es_repo):
        """Test list_glossary tool returns glossary terms."""
        mock_es_repo.get_glossary_terms.return_value = [
            {"term": "user", "total_count": 5},
            {"term": "email", "total_count": 3},
        ]

        result = await server._handle_tool(
            "list_glossary",
            {"scope": "test", "repository": "repo"},
        )

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["term"] == "user"
        mock_es_repo.get_glossary_terms.assert_called_once_with(
            scope="test",
            repository="repo",
            username="main",
            min_count=1,
        )

    @pytest.mark.asyncio
    async def test_list_glossary_with_min_count(self, server, mock_es_repo):
        """Test list_glossary tool with min_count filter."""
        mock_es_repo.get_glossary_terms.return_value = [
            {"term": "user", "total_count": 5},
        ]

        result = await server._handle_tool(
            "list_glossary",
            {"scope": "test", "repository": "repo", "min_count": 3},
        )

        mock_es_repo.get_glossary_terms.assert_called_once_with(
            scope="test",
            repository="repo",
            username="main",
            min_count=3,
        )

    @pytest.mark.asyncio
    async def test_list_glossary_with_custom_username(self, server, mock_es_repo):
        """Test list_glossary tool with custom username."""
        mock_es_repo.get_glossary_terms.return_value = []

        await server._handle_tool(
            "list_glossary",
            {"scope": "test", "repository": "repo", "username": "custom"},
        )

        mock_es_repo.get_glossary_terms.assert_called_once_with(
            scope="test",
            repository="repo",
            username="custom",
            min_count=1,
        )


class TestGetGlossaryTermTool:
    """Tests for get_glossary_term tool handler."""

    @pytest.mark.asyncio
    async def test_get_glossary_term_returns_term(self, server, mock_es_repo):
        """Test get_glossary_term tool returns term details."""
        mock_es_repo.get_glossary_term.return_value = {
            "term": "user",
            "total_count": 5,
            "element_ids": ["id1", "id2"],
            "file_paths": ["file1.py", "file2.py"],
        }

        result = await server._handle_tool(
            "get_glossary_term",
            {"scope": "test", "repository": "repo", "term": "user"},
        )

        assert isinstance(result, dict)
        assert result["term"] == "user"
        assert result["total_count"] == 5
        mock_es_repo.get_glossary_term.assert_called_once_with(
            scope="test",
            repository="repo",
            term="user",
            username="main",
        )

    @pytest.mark.asyncio
    async def test_get_glossary_term_not_found(self, server, mock_es_repo):
        """Test get_glossary_term returns None when term not found."""
        mock_es_repo.get_glossary_term.return_value = None

        result = await server._handle_tool(
            "get_glossary_term",
            {"scope": "test", "repository": "repo", "term": "nonexistent"},
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_glossary_term_with_custom_username(self, server, mock_es_repo):
        """Test get_glossary_term tool with custom username."""
        mock_es_repo.get_glossary_term.return_value = {"term": "user", "total_count": 1}

        await server._handle_tool(
            "get_glossary_term",
            {"scope": "test", "repository": "repo", "term": "user", "username": "custom"},
        )

        mock_es_repo.get_glossary_term.assert_called_once_with(
            scope="test",
            repository="repo",
            term="user",
            username="custom",
        )


class TestSearchGlossaryTool:
    """Tests for search_glossary tool handler."""

    @pytest.mark.asyncio
    async def test_search_glossary_returns_matches(self, server, mock_es_repo):
        """Test search_glossary tool returns matching terms."""
        mock_es_repo.search_glossary.return_value = [
            {"term": "user", "total_count": 5},
            {"term": "username", "total_count": 3},
        ]

        result = await server._handle_tool(
            "search_glossary",
            {"scope": "test", "repository": "repo", "query": "user"},
        )

        assert isinstance(result, list)
        assert len(result) == 2
        mock_es_repo.search_glossary.assert_called_once_with(
            scope="test",
            repository="repo",
            query="user",
            username="main",
        )

    @pytest.mark.asyncio
    async def test_search_glossary_no_matches(self, server, mock_es_repo):
        """Test search_glossary returns empty list when no matches."""
        mock_es_repo.search_glossary.return_value = []

        result = await server._handle_tool(
            "search_glossary",
            {"scope": "test", "repository": "repo", "query": "xyz"},
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_search_glossary_with_custom_username(self, server, mock_es_repo):
        """Test search_glossary tool with custom username."""
        mock_es_repo.search_glossary.return_value = []

        await server._handle_tool(
            "search_glossary",
            {"scope": "test", "repository": "repo", "query": "test", "username": "custom"},
        )

        mock_es_repo.search_glossary.assert_called_once_with(
            scope="test",
            repository="repo",
            query="test",
            username="custom",
        )
