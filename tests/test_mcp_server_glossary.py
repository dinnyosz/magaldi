"""Tests for MCP server glossary tool registration."""

from unittest.mock import MagicMock, patch

import pytest

from magaldi_mcp.server import MagaldiMCPServer

# Patch auto-detect so _resolve_scope_repo returns test values instead of magaldi.yaml values
_PATCH_AUTO_DETECT = patch(
    "magaldi_mcp.tools._utils._auto_detect_repo_config",
    return_value=("test_scope", "test_repo"),
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_repo():
    """Create a mock Elasticsearch repository."""
    repo = MagicMock()
    repo.get_glossary_terms.return_value = []
    repo.get_glossary_term.return_value = None
    repo.search_glossary.return_value = []
    return repo


@pytest.fixture
def server(mock_repo):
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
        server.repo = mock_repo
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
    @_PATCH_AUTO_DETECT
    async def test_list_glossary_returns_terms(self, _mock_detect, server, mock_repo):
        """Test list_glossary tool returns glossary terms."""
        mock_repo.get_glossary_terms.return_value = [
            {"term": "user", "total_count": 5, "description": "User-related code"},
            {"term": "email", "total_count": 3, "description": "Email handling"},
        ]

        result = await server._handle_tool(
            "list_glossary",
            {},
        )

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["term"] == "user"
        assert result[0]["description"] == "User-related code"
        mock_repo.get_glossary_terms.assert_called_once_with(
            scope="test_scope",
            repository="test_repo",
            username="main",
            min_count=1,
        )

    @pytest.mark.asyncio
    @_PATCH_AUTO_DETECT
    async def test_list_glossary_with_min_count(self, _mock_detect, server, mock_repo):
        """Test list_glossary tool with min_count filter."""
        mock_repo.get_glossary_terms.return_value = [
            {"term": "user", "total_count": 5},
        ]

        result = await server._handle_tool(
            "list_glossary",
            {"min_count": 3},
        )

        mock_repo.get_glossary_terms.assert_called_once_with(
            scope="test_scope",
            repository="test_repo",
            username="main",
            min_count=3,
        )

    @pytest.mark.asyncio
    @_PATCH_AUTO_DETECT
    async def test_list_glossary_with_custom_username(self, _mock_detect, server, mock_repo):
        """Test list_glossary tool with custom username."""
        mock_repo.get_glossary_terms.return_value = []

        await server._handle_tool(
            "list_glossary",
            {"username": "custom"},
        )

        mock_repo.get_glossary_terms.assert_called_once_with(
            scope="test_scope",
            repository="test_repo",
            username="custom",
            min_count=1,
        )


class TestGetGlossaryTermTool:
    """Tests for get_glossary_term tool handler."""

    @pytest.mark.asyncio
    @_PATCH_AUTO_DETECT
    async def test_get_glossary_term_returns_term(self, _mock_detect, server, mock_repo):
        """Test get_glossary_term tool returns term details."""
        mock_repo.get_glossary_term.return_value = {
            "term": "user",
            "total_count": 5,
            "element_ids": ["id1", "id2"],
            "file_paths": ["file1.py", "file2.py"],
            "description": "Code related to user management and authentication",
        }

        result = await server._handle_tool(
            "get_glossary_term",
            {"term": "user"},
        )

        assert isinstance(result, dict)
        assert result["term"] == "user"
        assert result["total_count"] == 5
        assert result["description"] == "Code related to user management and authentication"
        mock_repo.get_glossary_term.assert_called_once_with(
            scope="test_scope",
            repository="test_repo",
            term="user",
            username="main",
        )

    @pytest.mark.asyncio
    @_PATCH_AUTO_DETECT
    async def test_get_glossary_term_not_found(self, _mock_detect, server, mock_repo):
        """Test get_glossary_term returns None when term not found."""
        mock_repo.get_glossary_term.return_value = None

        result = await server._handle_tool(
            "get_glossary_term",
            {"term": "nonexistent"},
        )

        assert result is None

    @pytest.mark.asyncio
    @_PATCH_AUTO_DETECT
    async def test_get_glossary_term_with_custom_username(self, _mock_detect, server, mock_repo):
        """Test get_glossary_term tool with custom username."""
        mock_repo.get_glossary_term.return_value = {"term": "user", "total_count": 1}

        await server._handle_tool(
            "get_glossary_term",
            {"term": "user", "username": "custom"},
        )

        mock_repo.get_glossary_term.assert_called_once_with(
            scope="test_scope",
            repository="test_repo",
            term="user",
            username="custom",
        )


class TestSearchGlossaryTool:
    """Tests for search_glossary tool handler."""

    @pytest.mark.asyncio
    @_PATCH_AUTO_DETECT
    async def test_search_glossary_returns_matches(self, _mock_detect, server, mock_repo):
        """Test search_glossary tool returns matching terms."""
        mock_repo.search_glossary.return_value = [
            {"term": "user", "total_count": 5, "description": "User management"},
            {"term": "username", "total_count": 3, "description": "Username handling"},
        ]

        result = await server._handle_tool(
            "search_glossary",
            {"query": "user"},
        )

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["description"] == "User management"
        assert result[1]["description"] == "Username handling"
        mock_repo.search_glossary.assert_called_once_with(
            scope="test_scope",
            repository="test_repo",
            query="user",
            username="main",
        )

    @pytest.mark.asyncio
    @_PATCH_AUTO_DETECT
    async def test_search_glossary_no_matches(self, _mock_detect, server, mock_repo):
        """Test search_glossary returns empty list when no matches."""
        mock_repo.search_glossary.return_value = []

        result = await server._handle_tool(
            "search_glossary",
            {"query": "xyz"},
        )

        assert result == []

    @pytest.mark.asyncio
    @_PATCH_AUTO_DETECT
    async def test_search_glossary_with_custom_username(self, _mock_detect, server, mock_repo):
        """Test search_glossary tool with custom username."""
        mock_repo.search_glossary.return_value = []

        await server._handle_tool(
            "search_glossary",
            {"query": "test", "username": "custom"},
        )

        mock_repo.search_glossary.assert_called_once_with(
            scope="test_scope",
            repository="test_repo",
            query="test",
            username="custom",
        )
