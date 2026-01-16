"""Tests for the MCP server module."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magaldi_mcp.server import MagaldiMCPServer, _format_result


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_es_repo():
    """Create a mock Elasticsearch repository."""
    repo = MagicMock()
    repo.search_by_vector.return_value = []
    repo.get_document.return_value = None
    repo.search_by_text.return_value = []
    return repo


@pytest.fixture
def mock_embed_client():
    """Create a mock embedding client."""
    client = MagicMock()
    client.embed_single.return_value = [0.1] * 1024
    return client


@pytest.fixture
def server(mock_es_repo, mock_embed_client):
    """Create a MagaldiMCPServer with mocked dependencies."""
    mock_config = MagicMock()
    mock_config.llm = MagicMock()
    mock_config.llm.url = "http://localhost:11434"
    mock_config.llm.embed_model = "test-model"
    mock_config.llm.provider = "ollama"
    mock_config.llm.api_key = None
    mock_config.llm.embed_dimensions = 1024

    with patch("magaldi_mcp.server.get_config", return_value=mock_config):
        server = MagaldiMCPServer(default_username="test-user")
        server.es_repo = mock_es_repo
        server.embed_client = mock_embed_client
        return server


# =============================================================================
# SERVER INITIALIZATION TESTS
# =============================================================================


class TestMagaldiMCPServerInit:
    """Tests for MagaldiMCPServer initialization."""

    def test_init_sets_default_username(self, server):
        """Test that init sets the default username."""
        assert server.default_username == "test-user"

    def test_init_with_custom_username(self):
        """Test initialization with custom username."""
        mock_config = MagicMock()
        mock_config.llm = MagicMock()

        with patch("magaldi_mcp.server.get_config", return_value=mock_config):
            server = MagaldiMCPServer(default_username="custom-user")
            assert server.default_username == "custom-user"

    def test_get_es_creates_repo_lazily(self, server):
        """Test that _get_es creates the ES repo lazily."""
        server.es_repo = None
        with patch("magaldi_mcp.server.ElasticsearchRepository") as mock_es:
            mock_es.return_value = MagicMock()
            result = server._get_es()
            assert result is not None
            mock_es.assert_called_once()

    def test_get_embed_client_creates_client_lazily(self, server):
        """Test that _get_embed_client creates the client lazily."""
        server.embed_client = None
        with patch("magaldi_mcp.server.CodeEmbeddingClient") as mock_client:
            mock_client.return_value = MagicMock()
            result = server._get_embed_client()
            assert result is not None


# =============================================================================
# TOOL HANDLER TESTS
# =============================================================================


class TestSearchCodeTool:
    """Tests for search_code tool."""

    @pytest.mark.asyncio
    async def test_search_code_returns_results(self, server, mock_es_repo):
        """Test search_code tool returns results."""
        mock_es_repo.search_by_vector.return_value = [
            {
                "element_id": "scope:repo:user:file.py:function:test:1",
                "name": "test",
                "element_type": "function",
                "relative_path": "file.py",
                "line_start": 1,
                "summary": "A test function.",
                "_score": 0.9,
            }
        ]

        result = await server._handle_tool("search_code", {"query": "test function"})

        # Result is a dict with code_results and test_results
        assert isinstance(result, dict)
        assert "code_results" in result
        assert "test_results" in result
        assert len(result["code_results"]) >= 1

    @pytest.mark.asyncio
    async def test_search_code_with_filters(self, server, mock_es_repo):
        """Test search_code tool with element type filter."""
        mock_es_repo.search_by_vector.return_value = []

        result = await server._handle_tool(
            "search_code",
            {"query": "test", "element_types": ["function"], "limit": 5},
        )

        # Empty result is a dict with code_results and test_results
        assert isinstance(result, dict)
        assert "code_results" in result
        assert "test_results" in result


class TestGetElementTool:
    """Tests for get_element tool."""

    @pytest.mark.asyncio
    async def test_get_element_returns_element(self, server, mock_es_repo):
        """Test get_element tool returns element details."""
        mock_es_repo.get_document.return_value = {
            "element_id": "scope:repo:user:file.py:function:test:1",
            "name": "test",
            "element_type": "function",
            "relative_path": "file.py",
            "line_start": 1,
            "line_end": 10,
            "raw_code": "def test(): pass",
            "summary": "A test function.",
        }

        result = await server._handle_tool(
            "get_element",
            {"element_id": "scope:repo:user:file.py:function:test:1"},
        )

        assert result is not None
        assert result["name"] == "test"

    @pytest.mark.asyncio
    async def test_get_element_not_found(self, server, mock_es_repo):
        """Test get_element tool when element not found."""
        mock_es_repo.get_document.return_value = None

        with pytest.raises(ValueError, match="Element not found"):
            await server._handle_tool(
                "get_element",
                {"element_id": "nonexistent"},
            )


class TestListReposTool:
    """Tests for list_repos tool."""

    @pytest.mark.asyncio
    async def test_list_repos_returns_repos(self, server, mock_es_repo):
        """Test list_repos tool returns repository list."""
        mock_es_repo.get_indexed_repositories.return_value = [
            {"scope": "magaldi", "repository": "magaldi", "element_count": 100}
        ]

        result = await server._handle_tool("list_repos", {})

        # Result is a list of repositories
        assert isinstance(result, list)
        assert len(result) == 1


class TestGrepCodeTool:
    """Tests for grep_code tool."""

    @pytest.mark.asyncio
    async def test_grep_code_returns_matches(self, server, mock_es_repo):
        """Test grep_code tool returns pattern matches."""
        mock_es_repo.grep_code.return_value = {
            "matches": [
                {
                    "file": "file.py",
                    "line": 10,
                    "content": "def test_function():",
                }
            ],
            "total_matches": 1,
        }

        result = await server._handle_tool("grep_code", {"pattern": "test_function"})

        assert "matches" in result or result is not None


# =============================================================================
# FORMAT RESULT TESTS
# =============================================================================


class TestFormatResult:
    """Tests for _format_result function."""

    def test_format_list_of_elements(self):
        """Test formatting list of code elements."""
        result = [
            {
                "element_id": "scope:repo:user:file.py:function:test:1",
                "name": "test",
                "element_type": "function",
                "relative_path": "file.py",
                "line_start": 1,
                "summary": "A test function.",
            }
        ]

        formatted = _format_result(result)

        assert "test" in formatted
        assert "function" in formatted
        assert "file.py" in formatted

    def test_format_dict_with_results(self):
        """Test formatting dict with results key."""
        result = {
            "results": [
                {
                    "name": "test",
                    "element_type": "function",
                }
            ],
            "total": 1,
        }

        formatted = _format_result(result)

        assert isinstance(formatted, str)

    def test_format_empty_list(self):
        """Test formatting empty list."""
        result = []

        formatted = _format_result(result)

        # Empty list returns empty representation
        assert formatted == "[]" or "no results" in formatted.lower()

    def test_format_none(self):
        """Test formatting None."""
        result = None

        formatted = _format_result(result)

        assert "not found" in formatted.lower() or "no" in formatted.lower()

    def test_format_element_with_code(self):
        """Test formatting element with code snippet."""
        result = {
            "element_id": "scope:repo:user:file.py:function:test:1",
            "name": "test",
            "element_type": "function",
            "relative_path": "file.py",
            "line_start": 1,
            "line_end": 5,
            "raw_code": "def test():\n    pass",
            "summary": "A test function.",
        }

        formatted = _format_result(result)

        assert "test" in formatted

    def test_format_repositories(self):
        """Test formatting repository list."""
        result = {
            "repositories": [
                {"scope": "magaldi", "repository": "magaldi", "element_count": 100}
            ]
        }

        formatted = _format_result(result)

        assert "magaldi" in formatted

    def test_format_features(self):
        """Test formatting features list."""
        result = {
            "features": [
                {"feature_id": "f1", "label": "Authentication", "size": 10}
            ]
        }

        formatted = _format_result(result)

        assert "Authentication" in formatted or "f1" in formatted

    def test_format_stats(self):
        """Test formatting repository stats."""
        result = {
            "stats": {
                "total_elements": 100,
                "files": 10,
                "classes": 20,
                "functions": 70,
            }
        }

        formatted = _format_result(result)

        assert "100" in formatted or "elements" in formatted.lower()


# =============================================================================
# RUN SERVER TESTS
# =============================================================================


class TestSearchFeaturesTool:
    """Tests for search_features tool."""

    @pytest.mark.asyncio
    async def test_search_features_returns_results(self, server, mock_es_repo):
        """Test search_features tool returns results."""
        mock_es_repo.search_features_by_vector.return_value = [
            {
                "feature_id": "feature:auth:1",
                "label": "Authentication",
                "summary": "Handles user authentication",
                "member_count": 5,
            }
        ]

        result = await server._handle_tool("search_features", {"query": "authentication"})

        assert isinstance(result, list)


class TestFindSimilarTool:
    """Tests for find_similar tool."""

    @pytest.mark.asyncio
    async def test_find_similar_returns_results(self, server, mock_es_repo):
        """Test find_similar tool returns similar elements."""
        mock_es_repo.get_document.return_value = {
            "element_id": "scope:repo:user:file.py:function:test:1",
            "embedding": [0.1] * 1024,
        }
        mock_es_repo.search_by_vector.return_value = [
            {
                "element_id": "scope:repo:user:file2.py:function:similar:1",
                "name": "similar",
                "_score": 0.8,
            }
        ]

        result = await server._handle_tool(
            "find_similar",
            {"element_id": "scope:repo:user:file.py:function:test:1"},
        )

        assert isinstance(result, list)


class TestGetContextTool:
    """Tests for get_context tool."""

    @pytest.mark.asyncio
    async def test_get_context_returns_context(self, server, mock_es_repo):
        """Test get_context tool returns element context."""
        mock_es_repo.get_document.return_value = {
            "element_id": "scope:repo:user:file.py:function:test:1",
            "name": "test",
            "element_type": "function",
            "parent_id": "scope:repo:user:file.py:class:MyClass:1",
        }
        mock_es_repo.get_elements_by_parent.return_value = []

        result = await server._handle_tool(
            "get_context",
            {"element_id": "scope:repo:user:file.py:function:test:1"},
        )

        assert isinstance(result, dict)


class TestGetFileStructureTool:
    """Tests for get_file_structure tool."""

    @pytest.mark.asyncio
    async def test_get_file_structure_returns_structure(self, server, mock_es_repo):
        """Test get_file_structure tool returns file structure."""
        mock_es_repo.search.return_value = {
            "hits": {
                "hits": [
                    {"_source": {"element_id": "scope:repo:user:file.py:file:file.py:0", "element_type": "file"}},
                    {"_source": {"element_id": "scope:repo:user:file.py:function:test:1", "element_type": "function"}},
                ]
            }
        }

        result = await server._handle_tool(
            "get_file_structure",
            {"scope": "scope", "repository": "repo", "file_path": "file.py"},
        )

        assert result is not None


class TestListFeaturesTool:
    """Tests for list_features tool."""

    @pytest.mark.asyncio
    async def test_list_features_returns_features(self, server, mock_es_repo):
        """Test list_features tool returns feature list."""
        mock_es_repo.get_features.return_value = [
            {"feature_id": "f1", "label": "Auth", "member_count": 5}
        ]
        mock_es_repo.get_subfeatures.return_value = []

        result = await server._handle_tool(
            "list_features",
            {"scope": "scope", "repository": "repo"},
        )

        assert isinstance(result, list)


class TestGetRepoStatsTool:
    """Tests for get_repo_stats tool."""

    @pytest.mark.asyncio
    async def test_get_repo_stats_returns_stats(self, server, mock_es_repo):
        """Test get_repo_stats tool returns statistics."""
        mock_es_repo.get_repository_stats.return_value = {
            "total_elements": 100,
            "total_lines": 5000,
            "elements_by_type": {"function": 50, "class": 20},
        }

        result = await server._handle_tool(
            "get_repo_stats",
            {"scope": "scope", "repository": "repo"},
        )

        assert result is not None


class TestGetChildrenTool:
    """Tests for get_children tool."""

    @pytest.mark.asyncio
    async def test_get_children_returns_children(self, server, mock_es_repo):
        """Test get_children tool returns child elements."""
        mock_es_repo.get_elements_by_parent.return_value = [
            {"element_id": "scope:repo:user:file.py:method:child:5", "name": "child"}
        ]

        result = await server._handle_tool(
            "get_children",
            {"element_id": "scope:repo:user:file.py:class:Parent:1"},
        )

        assert isinstance(result, list)


class TestGetFeatureMembersTool:
    """Tests for get_feature_members tool."""

    @pytest.mark.asyncio
    async def test_get_feature_members_returns_members(self, server, mock_es_repo):
        """Test get_feature_members tool returns feature members."""
        # The tool first gets the feature document, then fetches members
        mock_es_repo.get_document.return_value = {
            "feature_id": "feature:auth:1",
            "member_ids": ["elem1", "elem2"],
        }
        mock_es_repo.get_documents.return_value = [
            {"element_id": "elem1", "name": "auth_login"},
            {"element_id": "elem2", "name": "auth_logout"},
        ]

        result = await server._handle_tool(
            "get_feature_members",
            {"feature_id": "feature:auth:1"},
        )

        assert isinstance(result, list)


class TestBatchGetElementsTool:
    """Tests for batch_get_elements tool."""

    @pytest.mark.asyncio
    async def test_batch_get_elements_returns_elements(self, server, mock_es_repo):
        """Test batch_get_elements tool returns multiple elements."""
        mock_es_repo.get_documents.return_value = [
            {"element_id": "id1", "name": "elem1"},
            {"element_id": "id2", "name": "elem2"},
        ]

        result = await server._handle_tool(
            "batch_get_elements",
            {"element_ids": ["id1", "id2"]},
        )

        assert isinstance(result, list)


class TestFindFilesTool:
    """Tests for find_files tool."""

    @pytest.mark.asyncio
    async def test_find_files_returns_files(self, server, mock_es_repo):
        """Test find_files tool returns matching files."""
        mock_es_repo.find_files_by_glob.return_value = [
            {"path": "src/main.py", "size": 1000}
        ]

        result = await server._handle_tool(
            "find_files",
            {"pattern": "**/*.py"},
        )

        assert isinstance(result, list)


class TestFindUsagesTool:
    """Tests for find_usages tool."""

    @pytest.mark.asyncio
    async def test_find_usages_returns_usages(self, server, mock_es_repo):
        """Test find_usages tool returns usage locations."""
        # The tool first gets the element document to find its name
        mock_es_repo.get_document.return_value = {
            "element_id": "scope:repo:user:file.py:function:test:1",
            "name": "test",
            "element_type": "function",
            "relative_path": "file.py",
            "line_start": 1,
        }
        mock_es_repo.grep_code.return_value = {
            "matches": [
                {"file": "caller.py", "line": 10, "content": "result = test()"}
            ],
            "total_matches": 1,
        }

        result = await server._handle_tool(
            "find_usages",
            {"element_id": "scope:repo:user:file.py:function:test:1"},
        )

        assert result is not None


class TestFindImplementationsTool:
    """Tests for find_implementations tool."""

    @pytest.mark.asyncio
    async def test_find_implementations_returns_implementations(self, server, mock_es_repo):
        """Test find_implementations tool returns implementing classes."""
        mock_es_repo.find_implementations.return_value = [
            {"class_name": "MyImpl", "file": "impl.py", "line": 1}
        ]

        result = await server._handle_tool(
            "find_implementations",
            {"class_name": "BaseClass"},
        )

        assert isinstance(result, list)


class TestGetCallGraphTool:
    """Tests for get_call_graph tool."""

    @pytest.mark.asyncio
    async def test_get_call_graph_returns_graph(self, server, mock_es_repo):
        """Test get_call_graph tool returns call graph."""
        mock_es_repo.get_document.return_value = {
            "element_id": "scope:repo:user:file.py:function:test:1",
            "name": "test",
            "element_type": "function",
            "calls": ["helper"],
        }
        mock_es_repo.grep_code.return_value = {"matches": [], "total_matches": 0}

        result = await server._handle_tool(
            "get_call_graph",
            {"element_id": "scope:repo:user:file.py:function:test:1"},
        )

        assert isinstance(result, dict)


class TestGenerateSkillTool:
    """Tests for generate_skill tool."""

    @pytest.mark.asyncio
    async def test_generate_skill_returns_result(self, server, tmp_path):
        """Test generate_skill tool creates skill file."""
        result = await server._handle_tool(
            "generate_skill",
            {"project_root": str(tmp_path), "skill_name": "test-skill", "scope": "project"},
        )

        assert result is not None


class TestUnknownTool:
    """Tests for unknown tool handling."""

    @pytest.mark.asyncio
    async def test_unknown_tool_raises_error(self, server):
        """Test that unknown tool raises ValueError."""
        with pytest.raises(ValueError, match="Unknown tool"):
            await server._handle_tool("nonexistent_tool", {})


# =============================================================================
# ADDITIONAL FORMAT RESULT TESTS
# =============================================================================


class TestFormatResultCodeSearch:
    """Tests for _format_result with code search results."""

    def test_format_code_search_with_signature(self):
        """Test formatting code search results with signature."""
        result = [
            {
                "element_id": "scope:repo:user:file.py:function:test:1",
                "name": "test",
                "type": "function",
                "file": "file.py",
                "line": 1,
                "signature": "def test(arg1, arg2):",
                "summary": "A test function that does something.",
            }
        ]

        formatted = _format_result(result)

        assert "test" in formatted
        assert "def test(arg1, arg2):" in formatted
        assert "A test function" in formatted

    def test_format_code_search_with_code(self):
        """Test formatting code search results with code."""
        result = [
            {
                "element_id": "id1",
                "name": "func",
                "type": "function",
                "file": "test.py",
                "line": 5,
                "code": "def func():\n    pass",
            }
        ]

        formatted = _format_result(result)

        assert "```" in formatted
        assert "def func():" in formatted

    def test_format_code_search_truncates_long_summary(self):
        """Test that long summaries are truncated."""
        long_summary = "A" * 300
        result = [
            {
                "element_id": "id1",
                "name": "func",
                "type": "function",
                "summary": long_summary,
            }
        ]

        formatted = _format_result(result)

        assert "..." in formatted
        assert len(formatted) < len(long_summary) + 100


class TestFormatResultFeatures:
    """Tests for _format_result with feature results."""

    def test_format_feature_search_results(self):
        """Test formatting feature search results."""
        result = [
            {
                "feature_id": "feature:auth:1",
                "label": "Authentication",
                "summary": "Handles user authentication and session management.",
                "member_count": 10,
            }
        ]

        formatted = _format_result(result)

        assert "Authentication" in formatted
        assert "10 members" in formatted

    def test_format_feature_truncates_long_summary(self):
        """Test that feature summaries are truncated."""
        long_summary = "B" * 300
        result = [
            {
                "feature_id": "f1",
                "label": "Test",
                "summary": long_summary,
                "member_count": 5,
            }
        ]

        formatted = _format_result(result)

        assert "..." in formatted


class TestFormatResultRepos:
    """Tests for _format_result with repository results."""

    def test_format_repository_list(self):
        """Test formatting repository list."""
        result = [
            {
                "scope": "company",
                "repository": "project",
                "element_count": 500,
                "file_count": 50,
            }
        ]

        formatted = _format_result(result)

        assert "company/project" in formatted
        assert "500 elements" in formatted
        assert "50 files" in formatted


class TestFormatResultFiles:
    """Tests for _format_result with file results."""

    def test_format_file_list(self):
        """Test formatting file list."""
        result = [
            {"path": "src/main.py", "size": 1500},
            {"path": "src/utils.py", "size": 800},
        ]

        formatted = _format_result(result)

        assert "src/main.py" in formatted
        assert "1500 bytes" in formatted
        assert "Found 2 files" in formatted


class TestFormatResultGrep:
    """Tests for _format_result with grep results."""

    def test_format_grep_results_with_context(self):
        """Test formatting grep results with context lines."""
        result = [
            {
                "file": "test.py",
                "line": 10,
                "content": "def test_function():",
                "context_before": ["# Comment above"],
                "context_after": ["    pass"],
            }
        ]

        formatted = _format_result(result)

        assert "test.py:10" in formatted
        assert "def test_function():" in formatted
        assert "# Comment above" in formatted
        assert "pass" in formatted


class TestFormatResultUsages:
    """Tests for _format_result with usage results."""

    def test_format_usage_results(self):
        """Test formatting usage results (which use grep format with context)."""
        # Usage results are formatted as grep matches with context
        result = [
            {
                "file": "caller.py",
                "line": 25,
                "content": "result = my_function()",
                "context_before": ["    # Call the function"],
                "context_after": ["    print(result)"],
            }
        ]

        formatted = _format_result(result)

        assert "caller.py:25" in formatted
        assert "result = my_function()" in formatted
        # Note: The format function uses "matches" not "usages" since grep format comes first
        assert "Found 1 matches" in formatted


class TestFormatResultImplementations:
    """Tests for _format_result with implementation results."""

    def test_format_implementation_results(self):
        """Test formatting implementation results."""
        result = [
            {
                "class_name": "MyImplementation",
                "file": "impl.py",
                "line": 5,
                "definition": "class MyImplementation(BaseClass):",
            }
        ]

        formatted = _format_result(result)

        assert "MyImplementation" in formatted
        assert "impl.py:5" in formatted
        assert "class MyImplementation(BaseClass):" in formatted


class TestFormatResultElement:
    """Tests for _format_result with single element dict."""

    def test_format_element_details(self):
        """Test formatting element details."""
        result = {
            "element_id": "scope:repo:user:file.py:function:test:1",
            "name": "test",
            "type": "function",
            "file": "file.py",
            "line_start": 1,
            "line_end": 10,
            "signature": "def test(a, b):",
            "summary": "A test function",
            "code": "def test(a, b):\n    return a + b",
        }

        formatted = _format_result(result)

        assert "[function] test" in formatted
        assert "file.py:1-10" in formatted
        assert "def test(a, b):" in formatted


class TestFormatResultFileRead:
    """Tests for _format_result with file read result."""

    def test_format_file_read_result(self):
        """Test formatting file read result."""
        result = {
            "path": "src/main.py",
            "content": "print('hello')",
            "lines_returned": 1,
            "total_lines": 100,
        }

        formatted = _format_result(result)

        assert "src/main.py" in formatted
        assert "1/100 lines" in formatted
        assert "```" in formatted
        assert "print('hello')" in formatted


class TestFormatResultStats:
    """Tests for _format_result with stats result."""

    def test_format_stats_result(self):
        """Test formatting stats result."""
        result = {
            "total_elements": 500,
            "total_lines": 10000,
            "feature_count": 15,
            "elements_by_type": {"function": 300, "class": 50},
        }

        formatted = _format_result(result)

        assert "500" in formatted
        assert "10000" in formatted
        assert "15" in formatted


class TestFormatResultCallGraph:
    """Tests for _format_result with call graph result."""

    def test_format_call_graph_result(self):
        """Test formatting call graph result."""
        result = {
            "element": {"name": "process", "type": "function", "file": "proc.py"},
            "callers": [
                {"file": "main.py", "line": 10, "content": "process(data)"}
            ],
            "callees": [{"name": "validate"}],
        }

        formatted = _format_result(result)

        assert "process" in formatted
        assert "Callers (1)" in formatted
        assert "main.py:10" in formatted
        assert "Callees (1)" in formatted
        assert "validate" in formatted


class TestFormatResultFallback:
    """Tests for _format_result fallback cases."""

    def test_format_arbitrary_dict(self):
        """Test formatting arbitrary dict falls back to JSON."""
        result = {"custom_key": "custom_value", "number": 42}

        formatted = _format_result(result)

        assert "custom_key" in formatted
        assert "custom_value" in formatted

    def test_format_arbitrary_list(self):
        """Test formatting arbitrary list falls back to JSON."""
        result = [{"unknown": "data"}]

        formatted = _format_result(result)

        assert "unknown" in formatted

    def test_format_string_result(self):
        """Test formatting string result."""
        result = "Simple string result"

        formatted = _format_result(result)

        assert formatted == "Simple string result"

    def test_format_none_result(self):
        """Test formatting None result."""
        result = None

        formatted = _format_result(result)

        assert "none" in formatted.lower() or "None" in formatted


# =============================================================================
# LAZY INITIALIZATION TESTS
# =============================================================================


class TestLazyInitialization:
    """Tests for lazy initialization of clients."""

    def test_get_es_returns_same_instance(self, server, mock_es_repo):
        """Test that _get_es returns the same instance on subsequent calls."""
        result1 = server._get_es()
        result2 = server._get_es()
        assert result1 is result2

    def test_get_embed_client_returns_same_instance(self, server, mock_embed_client):
        """Test that _get_embed_client returns the same instance on subsequent calls."""
        result1 = server._get_embed_client()
        result2 = server._get_embed_client()
        assert result1 is result2

    def test_get_embed_client_uses_embed_provider(self):
        """Test that _get_embed_client uses embed_provider if set."""
        mock_config = MagicMock()
        mock_config.llm = MagicMock()
        mock_config.llm.url = "http://localhost:11434"
        mock_config.llm.embed_model = "text-embedding"
        mock_config.llm.provider = "ollama"
        mock_config.llm.embed_provider = "openai"
        mock_config.llm.api_key = "default-key"
        mock_config.llm.embed_api_key = "embed-key"

        with patch("magaldi_mcp.server.get_config", return_value=mock_config), \
             patch("magaldi_mcp.server.CodeEmbeddingClient") as mock_client_class:
            mock_client_class.return_value = MagicMock()
            server = MagaldiMCPServer()

            server._get_embed_client()

            mock_client_class.assert_called_once_with(
                url="http://localhost:11434",
                model="text-embedding",
                provider="openai",
                api_key="embed-key",
            )


# =============================================================================
# CALL TOOL ERROR HANDLING TESTS
# =============================================================================


class TestCallToolErrorHandling:
    """Tests for call_tool error handling."""

    @pytest.mark.asyncio
    async def test_handle_tool_raises_on_unknown_tool(self, server):
        """Test that _handle_tool raises ValueError for unknown tools."""
        with pytest.raises(ValueError, match="Unknown tool"):
            await server._handle_tool("nonexistent_tool", {})

    @pytest.mark.asyncio
    async def test_handle_tool_propagates_errors(self, server, mock_es_repo):
        """Test that _handle_tool propagates errors from tools."""
        mock_es_repo.get_document.return_value = None

        with pytest.raises(ValueError, match="Element not found"):
            await server._handle_tool(
                "get_element",
                {"element_id": "nonexistent"},
            )


# =============================================================================
# RUN SERVER TESTS
# =============================================================================


class TestRunServer:
    """Tests for run_server function."""

    def test_run_server_parses_args(self):
        """Test that run_server parses command line arguments."""
        import sys
        from unittest.mock import patch

        # Mock sys.argv to provide command line arguments
        test_args = ["magaldi-mcp", "--user", "test-user"]

        mock_config = MagicMock()
        mock_config.llm = MagicMock()

        with patch.object(sys, "argv", test_args), \
             patch("magaldi_mcp.server.get_config", return_value=mock_config), \
             patch("magaldi_mcp.server.load_config") as mock_load, \
             patch("magaldi_mcp.server.asyncio.run") as mock_run, \
             patch("magaldi_mcp.server.logging"):

            from magaldi_mcp.server import run_server

            run_server()

            # load_config should have been called
            mock_load.assert_called_once()
            # asyncio.run should have been called
            mock_run.assert_called_once()

    def test_run_server_uses_env_var(self):
        """Test that run_server uses MAGALDI_USER env var."""
        import os
        import sys

        test_args = ["magaldi-mcp"]

        mock_config = MagicMock()
        mock_config.llm = MagicMock()

        with patch.object(sys, "argv", test_args), \
             patch.dict(os.environ, {"MAGALDI_USER": "env-user"}), \
             patch("magaldi_mcp.server.get_config", return_value=mock_config), \
             patch("magaldi_mcp.server.load_config"), \
             patch("magaldi_mcp.server.asyncio.run") as mock_run, \
             patch("magaldi_mcp.server.logging"), \
             patch("magaldi_mcp.server.MagaldiMCPServer") as mock_server_class:

            from magaldi_mcp.server import run_server

            run_server()

            # Server should have been created with the env var user
            mock_server_class.assert_called_once_with(default_username="env-user")

    def test_run_server_default_user(self):
        """Test that run_server uses 'main' as default user."""
        import os
        import sys

        test_args = ["magaldi-mcp"]

        mock_config = MagicMock()
        mock_config.llm = MagicMock()

        # Ensure MAGALDI_USER is not set
        env_without_magaldi = {k: v for k, v in os.environ.items() if k != "MAGALDI_USER"}

        with patch.object(sys, "argv", test_args), \
             patch.dict(os.environ, env_without_magaldi, clear=True), \
             patch("magaldi_mcp.server.get_config", return_value=mock_config), \
             patch("magaldi_mcp.server.load_config"), \
             patch("magaldi_mcp.server.asyncio.run"), \
             patch("magaldi_mcp.server.logging"), \
             patch("magaldi_mcp.server.MagaldiMCPServer") as mock_server_class:

            from magaldi_mcp.server import run_server

            run_server()

            # Server should have been created with 'main' as default
            mock_server_class.assert_called_once_with(default_username="main")
