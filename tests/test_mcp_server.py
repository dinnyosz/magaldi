"""Tests for the MCP server module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from magaldi_mcp.server import MagaldiMCPServer, _format_result

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_repo():
    """Create a mock Elasticsearch repository."""
    repo = MagicMock()
    repo.search_by_vector.return_value = []
    repo.get_document.return_value = None
    repo.search_by_text.return_value = []
    # Delegate get_document_by_id_or_hash to get_document
    repo.get_document_by_id_or_hash.side_effect = lambda x: repo.get_document(x)
    return repo


@pytest.fixture
def mock_embed_client():
    """Create a mock embedding client."""
    client = MagicMock()
    client.embed_single.return_value = [0.1] * 1024
    return client


@pytest.fixture
def server(mock_repo, mock_embed_client):
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
        server.repo = mock_repo
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

    def test_get_repo_creatrepo_lazily(self, server):
        """Test that _get_repo creates the ES repo lazily."""
        server.repo = None
        with patch("magaldi_mcp.server.Repository") as mock_es:
            mock_es.return_value = MagicMock()
            result = server._get_repo()
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
    async def test_search_code_returns_results(self, server, mock_repo):
        """Test search_code tool returns results."""
        mock_repo.search_by_vector.return_value = [
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
    async def test_search_code_with_filters(self, server, mock_repo):
        """Test search_code tool with element type filter."""
        mock_repo.search_by_vector.return_value = []

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
    async def test_get_element_returns_element(self, server, mock_repo):
        """Test get_element tool returns element details."""
        mock_repo.get_document.return_value = {
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
            {"hash_id": "scope:repo:user:file.py:function:test:1"},
        )

        assert result is not None
        assert result["name"] == "test"

    @pytest.mark.asyncio
    async def test_get_element_not_found(self, server, mock_repo):
        """Test get_element tool when element not found."""
        mock_repo.get_document.return_value = None

        with pytest.raises(ValueError, match="Element not found"):
            await server._handle_tool(
                "get_element",
                {"hash_id": "nonexistent"},
            )


class TestListReposTool:
    """Tests for list_repos tool."""

    @pytest.mark.asyncio
    async def test_list_repos_returns_repos(self, server, mock_repo):
        """Test list_repos tool returns repository list."""
        mock_repo.get_indexed_repositories.return_value = [
            {"scope": "magaldi", "repository": "magaldi", "element_count": 100}
        ]

        result = await server._handle_tool("list_repos", {})

        # Result is a list of repositories
        assert isinstance(result, list)
        assert len(result) == 1


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
    async def test_search_features_returns_results(self, server, mock_repo):
        """Test search_features tool returns results."""
        mock_repo.search_features_by_vector.return_value = [
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
    async def test_find_similar_returns_results(self, server, mock_repo):
        """Test find_similar tool returns similar elements grouped by is_test."""
        mock_repo.get_document.return_value = {
            "element_id": "scope:repo:user:file.py:function:test:1",
            "summary_embedding": [0.1] * 1024,
        }
        mock_repo.search_by_vector.return_value = [
            {
                "element_id": "scope:repo:user:file2.py:function:similar:1",
                "name": "similar",
                "_score": 0.8,
            }
        ]

        result = await server._handle_tool(
            "find_similar",
            {"hash_id": "scope:repo:user:file.py:function:test:1"},
        )

        # Result is a dict with code_results and test_results
        assert isinstance(result, dict)
        assert "code_results" in result
        assert "test_results" in result
        assert len(result["code_results"]) == 1


class TestGetContextTool:
    """Tests for get_context tool."""

    @pytest.mark.asyncio
    async def test_get_context_returns_context(self, server, mock_repo):
        """Test get_context tool returns element context."""
        mock_repo.get_document.return_value = {
            "element_id": "scope:repo:user:file.py:function:test:1",
            "name": "test",
            "element_type": "function",
            "parent_id": "scope:repo:user:file.py:class:MyClass:1",
        }
        mock_repo.get_elements_by_parent.return_value = []

        result = await server._handle_tool(
            "get_context",
            {"hash_id": "scope:repo:user:file.py:function:test:1"},
        )

        assert isinstance(result, dict)


class TestGetFileStructureTool:
    """Tests for get_file_structure tool."""

    @pytest.mark.asyncio
    async def test_get_file_structure_returns_structure(self, server, mock_repo):
        """Test get_file_structure tool returns file structure."""
        mock_repo.search.return_value = {
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
    async def test_list_features_returns_features(self, server, mock_repo):
        """Test list_features tool returns feature list."""
        mock_repo.get_features.return_value = [
            {"feature_id": "f1", "label": "Auth", "member_count": 5}
        ]
        mock_repo.get_subfeatures.return_value = []

        result = await server._handle_tool(
            "list_features",
            {"scope": "scope", "repository": "repo"},
        )

        assert isinstance(result, list)


class TestGetRepoStatsTool:
    """Tests for get_repo_stats tool."""

    @pytest.mark.asyncio
    async def test_get_repo_stats_returns_stats(self, server, mock_repo):
        """Test get_repo_stats tool returns statistics."""
        mock_repo.get_repository_stats.return_value = {
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
    async def test_get_children_returns_children(self, server, mock_repo):
        """Test get_children tool returns child elements."""
        mock_repo.get_document_by_id_or_hash.side_effect = None
        mock_repo.get_document_by_id_or_hash.return_value = {
            "element_id": "scope:repo:user:file.py:class:Parent:1",
        }
        mock_repo.get_elements_by_parent.return_value = [
            {"element_id": "scope:repo:user:file.py:method:child:5", "name": "child"}
        ]

        result = await server._handle_tool(
            "get_children",
            {"hash_id": "scope:repo:user:file.py:class:Parent:1"},
        )

        assert isinstance(result, list)


class TestGetFeatureMembersTool:
    """Tests for get_feature_members tool."""

    @pytest.mark.asyncio
    async def test_get_feature_members_returns_members(self, server, mock_repo):
        """Test get_feature_members tool returns feature members with glossary terms."""
        # The tool first gets the feature document, then fetches members
        mock_repo.get_document.side_effect = [
            # First call: get feature document
            {
                "feature_id": "feature:auth:1",
                "member_ids": ["elem1", "elem2"],
            },
            # Second call: get first member
            {"element_id": "elem1", "name": "auth_login", "element_type": "function"},
            # Third call: get second member
            {"element_id": "elem2", "name": "auth_logout", "element_type": "function"},
        ]
        mock_repo.get_glossary_terms.return_value = []

        result = await server._handle_tool(
            "get_feature_members",
            {"feature_id": "feature:auth:1"},
        )

        assert isinstance(result, dict)
        assert "members" in result
        assert "glossary_terms" in result
        assert len(result["members"]) == 2


class TestBatchGetElementsTool:
    """Tests for batch_get_elements tool."""

    @pytest.mark.asyncio
    async def test_batch_get_elements_returns_elements(self, server, mock_repo):
        """Test batch_get_elements tool returns multiple elements."""
        mock_repo.get_documents.return_value = [
            {"element_id": "id1", "name": "elem1"},
            {"element_id": "id2", "name": "elem2"},
        ]

        result = await server._handle_tool(
            "batch_get_elements",
            {"hash_ids": ["id1", "id2"]},
        )

        assert isinstance(result, list)


class TestFindFilesTool:
    """Tests for find_files tool."""

    @pytest.mark.asyncio
    async def test_find_files_returns_files(self, server, mock_repo):
        """Test find_files tool returns matching files."""
        mock_repo.find_files_by_glob.return_value = [
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
    async def test_find_usages_returns_usages(self, server, mock_repo):
        """Test find_usages tool returns usage locations."""
        # The tool first gets the element document to find its name
        mock_repo.get_document.return_value = {
            "element_id": "scope:repo:user:file.py:function:test:1",
            "name": "test",
            "element_type": "function",
            "relative_path": "file.py",
            "line_start": 1,
        }
        mock_repo.grep_code.return_value = {
            "matches": [
                {"file": "caller.py", "line": 10, "content": "result = test()"}
            ],
            "total_matches": 1,
        }

        result = await server._handle_tool(
            "find_usages",
            {"hash_id": "scope:repo:user:file.py:function:test:1"},
        )

        assert result is not None


class TestFindImplementationsTool:
    """Tests for find_implementations tool."""

    @pytest.mark.asyncio
    async def test_find_implementations_returns_implementations(self, server, mock_repo):
        """Test find_implementations tool returns implementing classes."""
        mock_repo.find_implementations.return_value = [
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
    async def test_get_call_graph_returns_graph(self, server, mock_repo):
        """Test get_call_graph tool returns call graph."""
        mock_repo.get_document.return_value = {
            "element_id": "scope:repo:user:file.py:function:test:1",
            "name": "test",
            "element_type": "function",
            "calls": ["helper"],
        }
        mock_repo.grep_code.return_value = {"matches": [], "total_matches": 0}

        result = await server._handle_tool(
            "get_call_graph",
            {"hash_id": "scope:repo:user:file.py:function:test:1"},
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


class TestGenerateConfigTool:
    """Tests for generate_config tool."""

    @pytest.mark.asyncio
    async def test_generate_config_creates_file(self, server, tmp_path):
        """Test generate_config tool creates magaldi.yaml."""
        # Create a fake repo directory structure: /tmp/xxx/code/my-repo
        code_dir = tmp_path / "code"
        repo_dir = code_dir / "my-repo"
        repo_dir.mkdir(parents=True)

        result = await server._handle_tool(
            "generate_config",
            {"repo_path": str(repo_dir)},
        )

        assert result is not None
        assert "path" in result
        assert result["repository"] == "my-repo"
        assert result["scope"] == "code"  # Parent directory name
        assert (repo_dir / "magaldi.yaml").exists()

    @pytest.mark.asyncio
    async def test_generate_config_with_overrides(self, server, tmp_path):
        """Test generate_config respects scope and repository overrides."""
        repo_dir = tmp_path / "some-dir"
        repo_dir.mkdir()

        result = await server._handle_tool(
            "generate_config",
            {"repo_path": str(repo_dir), "scope": "myorg", "repository": "myrepo"},
        )

        assert result["scope"] == "myorg"
        assert result["repository"] == "myrepo"

    @pytest.mark.asyncio
    async def test_generate_config_skips_existing(self, server, tmp_path):
        """Test generate_config skips if config already exists."""
        repo_dir = tmp_path / "existing-repo"
        repo_dir.mkdir()
        (repo_dir / "magaldi.yaml").write_text("scope: test\n")

        result = await server._handle_tool(
            "generate_config",
            {"repo_path": str(repo_dir)},
        )

        assert result.get("skipped") is True
        assert "already exists" in result.get("reason", "")

    @pytest.mark.asyncio
    async def test_generate_config_invalid_path(self, server, tmp_path):
        """Test generate_config handles invalid paths."""
        result = await server._handle_tool(
            "generate_config",
            {"repo_path": str(tmp_path / "nonexistent")},
        )

        assert "error" in result


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
        assert "file.py" in formatted
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

        assert "func" in formatted
        assert "test.py" in formatted

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
        assert "500" in formatted
        assert "50" in formatted


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
        assert "src/utils.py" in formatted
        assert "2" in formatted  # file count shown somewhere


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

        assert "test.py" in formatted
        assert "10" in formatted
        assert "def test_function():" in formatted
        assert "# Comment above" in formatted
        assert "pass" in formatted


class TestFormatResultUsages:
    """Tests for _format_result with usage results."""

    def test_format_usage_results(self):
        """Test formatting usage results (with context_before for distinction from grep)."""
        # Usage results have context_before/after, distinguishing them from grep results
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

        assert "caller.py" in formatted
        assert "25" in formatted
        assert "result = my_function()" in formatted


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
        assert "impl.py" in formatted
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

        assert "fn:test" in formatted
        assert "file.py" in formatted
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
        assert "1/100" in formatted
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
        assert "Callers (1)" in formatted or "callers" in formatted.lower()
        assert "main.py" in formatted
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

    @pytest.mark.usefixtures("mock_repo")
    def test_get_repo_returns_same_instance(self, server):
        """Test that _get_repo returns the same instance on subsequent calls."""
        result1 = server._get_repo()
        result2 = server._get_repo()
        assert result1 is result2

    @pytest.mark.usefixtures("mock_embed_client")
    def test_get_embed_client_returns_same_instance(self, server):
        """Test that _get_embed_client returns the same instance on subsequent calls."""
        result1 = server._get_embed_client()
        result2 = server._get_embed_client()
        assert result1 is result2

    def test_get_embed_client_uses_embed_provider(self):
        """Test that _get_embed_client uses embed_provider if set."""
        mock_config = MagicMock()
        # Mock the get_embed_model() method to return a model config
        mock_embed_model = MagicMock()
        mock_embed_model.url = "http://localhost:11434"
        mock_embed_model.name = "text-embedding"
        mock_embed_model.provider = "openai"
        mock_embed_model.api_key = "embed-key"
        mock_config.llm.get_embed_model.return_value = mock_embed_model

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
    async def test_handle_tool_propagates_errors(self, server, mock_repo):
        """Test that _handle_tool propagates errors from tools."""
        mock_repo.get_document.return_value = None

        with pytest.raises(ValueError, match="Element not found"):
            await server._handle_tool(
                "get_element",
                {"hash_id": "nonexistent"},
            )


# =============================================================================
# RUN SERVER TESTS
# =============================================================================


class TestPatternSearchTool:
    """Tests for pattern_search tool."""

    @pytest.mark.asyncio
    async def test_pattern_search_tool_registered(self, server):
        """Test that pattern_search tool is registered."""
        from mcp.types import ListToolsRequest

        # Get the list_tools handler from the MCP server
        handler = server.server.request_handlers.get(ListToolsRequest)
        assert handler is not None, "ListToolsRequest handler not registered"

        # Call the handler to get tools list
        # The handler returns ServerResult with root.tools
        result = await handler(None)
        tools = result.root.tools
        tool_names = [t.name for t in tools]
        assert "pattern_search" in tool_names

    @pytest.mark.asyncio
    async def test_pattern_search_tool_schema(self, server):
        """Test pattern_search tool has correct schema."""
        from mcp.types import ListToolsRequest

        # Get the list_tools handler from the MCP server
        handler = server.server.request_handlers.get(ListToolsRequest)
        result = await handler(None)
        tools = result.root.tools
        pattern_tool = next(t for t in tools if t.name == "pattern_search")

        schema = pattern_tool.inputSchema
        assert "pattern" in schema["properties"]
        assert "mode" in schema["properties"]
        assert schema["properties"]["mode"]["enum"] == ["regexp", "wildcard", "proximity"]
        # scope and repository have been removed from schema (auto-detected from magaldi.yaml)
        assert "scope" not in schema["properties"]
        assert "repository" not in schema["properties"]
        assert "pattern" in schema["required"]
        assert "mode" in schema["required"]

    @pytest.mark.asyncio
    @patch(
        "magaldi_mcp.tools._utils._auto_detect_repo_config",
        return_value=("test_scope", "test_repo"),
    )
    async def test_pattern_search_returns_results(self, _mock_detect, server, mock_repo):
        """Test pattern_search tool returns results."""
        mock_repo.search_by_regexp.return_value = [
            {
                "element_id": "scope:repo:user:file.py:function:test:1",
                "name": "test",
                "element_type": "function",
                "relative_path": "file.py",
                "line_start": 1,
                "raw_code": "def test(): pass",
                "is_test": False,
            }
        ]

        result = await server._handle_tool(
            "pattern_search",
            {
                "pattern": "test.*",
                "mode": "regexp",
            },
        )

        assert "code_results" in result
        assert "test_results" in result
        assert len(result["code_results"]) == 1


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
             patch("magaldi_mcp.server.asyncio.run"), \
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


# =============================================================================
# PATTERN TOOLS INTEGRATION TESTS
# =============================================================================


@pytest.fixture(scope="function")
def integration_config():
    """Load config for ES connection in integration tests."""
    import os

    from shared.config import MagaldiConfig, SearchBackendConfig, reset_config

    # Reset config singleton
    reset_config()

    # Create config directly with known good values for integration tests
    return MagaldiConfig(
        search_backend=SearchBackendConfig(
            host=os.environ.get("MAGALDI_ELASTICSEARCH_HOST", "localhost"),
            port=int(os.environ.get("MAGALDI_ELASTICSEARCH_PORT", "9200")),
            scheme=os.environ.get("MAGALDI_ELASTICSEARCH_SCHEME", "http"),
        ),
    )


@pytest.fixture
def repo(integration_config):
    """Create ES repository and clean up test data."""
    from shared.db.store import INDEX_NAME, Repository

    repo = Repository(integration_config)

    # Delete test documents before test
    try:
        client = repo._get_client()
        client.delete_by_query(
            index=INDEX_NAME,
            body={"query": {"term": {"scope": "test-patterns"}}},
            refresh=True,
            ignore=[404],
        )
    except Exception:
        pass

    yield repo

    # Delete test documents after test
    try:
        client = repo._get_client()
        client.delete_by_query(
            index=INDEX_NAME,
            body={"query": {"term": {"scope": "test-patterns"}}},
            refresh=True,
            ignore=[404],
        )
    except Exception:
        pass

    repo.close()


@pytest.mark.integration
class TestPatternTools:
    """Integration tests for pattern detection MCP tools."""

    @pytest.fixture
    def es_with_patterns(self, repo):
        """Set up ES with classes having patterns."""
        from magaldi_core.code_parser import CodeElement
        from shared.db.store import INDEX_NAME

        # Index a repository class
        elem = CodeElement(
            scope="test-patterns",
            repository="test-repo",
            username="main",
            relative_path="src/repos.py",
            element_type="class",
            name="UserRepository",
            language="python",
            line_start=1,
            line_end=50,
            raw_code="class UserRepository: ...",
        )
        elem.detected_patterns = ["repository"]
        elem.pattern_confidence = {"repository": 0.8}
        repo.index_element(elem)

        # Index a singleton class
        singleton_elem = CodeElement(
            scope="test-patterns",
            repository="test-repo",
            username="main",
            relative_path="src/singleton.py",
            element_type="class",
            name="DatabaseConnection",
            language="python",
            line_start=1,
            line_end=30,
            raw_code="class DatabaseConnection: ...",
        )
        singleton_elem.detected_patterns = ["singleton"]
        singleton_elem.pattern_confidence = {"singleton": 0.9}
        repo.index_element(singleton_elem)

        # Refresh index
        repo._get_client().indices_refresh(index=INDEX_NAME)
        return repo

    @patch(
        "magaldi_mcp.tools._utils._auto_detect_repo_config",
        return_value=("test-patterns", "test-repo"),
    )
    def test_list_patterns_returns_all_patterns(self, _mock_detect, es_with_patterns):
        """Test list_patterns returns all pattern types found."""
        from magaldi_mcp.tools import list_patterns

        result = list_patterns(es_with_patterns, "test-patterns", "test-repo")

        assert result["patterns"]
        pattern_names = [p["pattern"] for p in result["patterns"]]
        assert "repository" in pattern_names
        assert "singleton" in pattern_names
        assert result["total_classes_with_patterns"] == 2

    @patch(
        "magaldi_mcp.tools._utils._auto_detect_repo_config",
        return_value=("test-patterns", "test-repo"),
    )
    def test_list_patterns_includes_examples(self, _mock_detect, es_with_patterns):
        """Test list_patterns includes example classes for each pattern."""
        from magaldi_mcp.tools import list_patterns

        result = list_patterns(es_with_patterns, "test-patterns", "test-repo")

        repo_pattern = next(p for p in result["patterns"] if p["pattern"] == "repository")
        assert repo_pattern["count"] == 1
        assert len(repo_pattern["examples"]) == 1
        assert repo_pattern["examples"][0]["name"] == "UserRepository"

    @patch(
        "magaldi_mcp.tools._utils._auto_detect_repo_config",
        return_value=("test-patterns", "test-repo"),
    )
    def test_find_by_pattern_returns_matching_classes(self, _mock_detect, es_with_patterns):
        """Test find_by_pattern returns classes with specified pattern."""
        from magaldi_mcp.tools import find_by_pattern

        result = find_by_pattern(es_with_patterns, "repository", "test-patterns", "test-repo")

        assert result["count"] == 1
        assert result["classes"][0]["name"] == "UserRepository"
        assert result["classes"][0]["confidence"] == 0.8

    @patch(
        "magaldi_mcp.tools._utils._auto_detect_repo_config",
        return_value=("test-patterns", "test-repo"),
    )
    def test_find_by_pattern_filters_by_confidence(self, _mock_detect, es_with_patterns):
        """Test find_by_pattern respects min_confidence threshold."""
        from magaldi_mcp.tools import find_by_pattern

        # Should find singleton with confidence 0.9
        result = find_by_pattern(
            es_with_patterns, "singleton", "test-patterns", "test-repo", min_confidence=0.85
        )
        assert result["count"] == 1

        # Should not find with higher threshold
        result = find_by_pattern(
            es_with_patterns, "singleton", "test-patterns", "test-repo", min_confidence=0.95
        )
        assert result["count"] == 0

    @patch(
        "magaldi_mcp.tools._utils._auto_detect_repo_config",
        return_value=("test-patterns", "test-repo"),
    )
    def test_find_by_pattern_no_matches(self, _mock_detect, es_with_patterns):
        """Test find_by_pattern returns empty when no matches."""
        from magaldi_mcp.tools import find_by_pattern

        result = find_by_pattern(es_with_patterns, "builder", "test-patterns", "test-repo")

        assert result["count"] == 0
        assert result["classes"] == []
