# tests/test_mcp_glossary_tools.py
"""Tests for glossary MCP tools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from magaldi_mcp.tools import (
    get_glossary_term,
    list_glossary,
    search_glossary,
)

# Patch auto-detect so _resolve_scope_repo returns the test's expected values
_PATCH_AUTO_DETECT = patch(
    "magaldi_mcp.tools._utils._auto_detect_repo_config",
    return_value=("test_scope", "test_repo"),
)


class TestListGlossary:
    """Tests for list_glossary tool."""

    @_PATCH_AUTO_DETECT
    def test_returns_glossary_terms(self, _mock_detect):
        """Test listing glossary terms."""
        mock_es = MagicMock()
        mock_es.get_glossary_terms.return_value = [
            {"term": "user", "total_count": 10, "description": "User-related functionality"},
            {"term": "email", "total_count": 5, "description": "Email handling code"},
        ]

        result = list_glossary(
            repo=mock_es,
            scope="test_scope",
            repository="test_repo",
        )

        assert len(result) == 2
        assert result[0]["term"] == "user"
        assert result[0]["description"] == "User-related functionality"
        assert result[1]["description"] == "Email handling code"
        mock_es.get_glossary_terms.assert_called_once()

    @_PATCH_AUTO_DETECT
    def test_passes_min_count_parameter(self, _mock_detect):
        """Test that min_count is passed to ES."""
        mock_es = MagicMock()
        mock_es.get_glossary_terms.return_value = []

        list_glossary(
            repo=mock_es,
            scope="test_scope",
            repository="test_repo",
            min_count=5,
        )

        mock_es.get_glossary_terms.assert_called_once_with(
            scope="test_scope",
            repository="test_repo",
            username="main",
            min_count=5,
        )

    @_PATCH_AUTO_DETECT
    def test_passes_custom_username(self, _mock_detect):
        """Test that custom username is passed to ES."""
        mock_es = MagicMock()
        mock_es.get_glossary_terms.return_value = []

        list_glossary(
            repo=mock_es,
            scope="test_scope",
            repository="test_repo",
            username="custom_user",
        )

        mock_es.get_glossary_terms.assert_called_once_with(
            scope="test_scope",
            repository="test_repo",
            username="custom_user",
            min_count=1,
        )


class TestGetGlossaryTerm:
    """Tests for get_glossary_term tool."""

    @_PATCH_AUTO_DETECT
    def test_returns_term_details(self, _mock_detect):
        """Test getting glossary term details."""
        mock_es = MagicMock()
        mock_es.get_glossary_term.return_value = {
            "term": "user",
            "total_count": 10,
            "element_ids": ["id1", "id2"],
            "file_paths": ["user.py"],
            "description": "Code elements related to user management and authentication",
            "feature_associations": [
                {"feature_label": "auth", "frequency": 3, "percentage": 60.0}
            ],
        }

        result = get_glossary_term(
            repo=mock_es,
            scope="test_scope",
            repository="test_repo",
            term="user",
        )

        assert result["term"] == "user"
        assert result["total_count"] == 10
        assert result["description"] == "Code elements related to user management and authentication"
        assert len(result["feature_associations"]) == 1

    @_PATCH_AUTO_DETECT
    def test_returns_none_for_missing_term(self, _mock_detect):
        """Test that missing term returns None."""
        mock_es = MagicMock()
        mock_es.get_glossary_term.return_value = None

        result = get_glossary_term(
            repo=mock_es,
            scope="test_scope",
            repository="test_repo",
            term="nonexistent",
        )

        assert result is None


class TestSearchGlossary:
    """Tests for search_glossary tool."""

    @_PATCH_AUTO_DETECT
    def test_searches_by_partial_match(self, _mock_detect):
        """Test searching glossary by partial match."""
        mock_es = MagicMock()
        mock_es.search_glossary.return_value = [
            {"term": "user", "total_count": 10, "description": "User management code"},
            {"term": "username", "total_count": 3, "description": "Username handling"},
        ]

        result = search_glossary(
            repo=mock_es,
            scope="test_scope",
            repository="test_repo",
            query="user",
        )

        assert len(result) == 2
        assert result[0]["description"] == "User management code"
        assert result[1]["description"] == "Username handling"
        mock_es.search_glossary.assert_called_once_with(
            scope="test_scope",
            repository="test_repo",
            query="user",
            username="main",
        )

    @_PATCH_AUTO_DETECT
    def test_returns_empty_list_for_no_matches(self, _mock_detect):
        """Test that no matches returns empty list."""
        mock_es = MagicMock()
        mock_es.search_glossary.return_value = []

        result = search_glossary(
            repo=mock_es,
            scope="test_scope",
            repository="test_repo",
            query="nonexistent",
        )

        assert result == []
