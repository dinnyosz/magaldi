# tests/test_mcp_glossary_tools.py
"""Tests for glossary MCP tools."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from magaldi_mcp.tools import (
    list_glossary,
    get_glossary_term,
    search_glossary,
)


class TestListGlossary:
    """Tests for list_glossary tool."""

    def test_returns_glossary_terms(self):
        """Test listing glossary terms."""
        mock_es = MagicMock()
        mock_es.get_glossary_terms.return_value = [
            {"term": "user", "total_count": 10},
            {"term": "email", "total_count": 5},
        ]

        result = list_glossary(
            es=mock_es,
            scope="scope",
            repository="repo",
        )

        assert len(result) == 2
        assert result[0]["term"] == "user"
        mock_es.get_glossary_terms.assert_called_once()

    def test_passes_min_count_parameter(self):
        """Test that min_count is passed to ES."""
        mock_es = MagicMock()
        mock_es.get_glossary_terms.return_value = []

        list_glossary(
            es=mock_es,
            scope="scope",
            repository="repo",
            min_count=5,
        )

        mock_es.get_glossary_terms.assert_called_once_with(
            scope="scope",
            repository="repo",
            username="main",
            min_count=5,
        )

    def test_passes_custom_username(self):
        """Test that custom username is passed to ES."""
        mock_es = MagicMock()
        mock_es.get_glossary_terms.return_value = []

        list_glossary(
            es=mock_es,
            scope="scope",
            repository="repo",
            username="custom_user",
        )

        mock_es.get_glossary_terms.assert_called_once_with(
            scope="scope",
            repository="repo",
            username="custom_user",
            min_count=1,
        )


class TestGetGlossaryTerm:
    """Tests for get_glossary_term tool."""

    def test_returns_term_details(self):
        """Test getting glossary term details."""
        mock_es = MagicMock()
        mock_es.get_glossary_term.return_value = {
            "term": "user",
            "total_count": 10,
            "element_ids": ["id1", "id2"],
            "file_paths": ["user.py"],
            "feature_associations": [
                {"feature_label": "auth", "frequency": 3, "percentage": 60.0}
            ],
        }

        result = get_glossary_term(
            es=mock_es,
            scope="scope",
            repository="repo",
            term="user",
        )

        assert result["term"] == "user"
        assert result["total_count"] == 10
        assert len(result["feature_associations"]) == 1

    def test_returns_none_for_missing_term(self):
        """Test that missing term returns None."""
        mock_es = MagicMock()
        mock_es.get_glossary_term.return_value = None

        result = get_glossary_term(
            es=mock_es,
            scope="scope",
            repository="repo",
            term="nonexistent",
        )

        assert result is None


class TestSearchGlossary:
    """Tests for search_glossary tool."""

    def test_searches_by_partial_match(self):
        """Test searching glossary by partial match."""
        mock_es = MagicMock()
        mock_es.search_glossary.return_value = [
            {"term": "user", "total_count": 10},
            {"term": "username", "total_count": 3},
        ]

        result = search_glossary(
            es=mock_es,
            scope="scope",
            repository="repo",
            query="user",
        )

        assert len(result) == 2
        mock_es.search_glossary.assert_called_once_with(
            scope="scope",
            repository="repo",
            query="user",
            username="main",
        )

    def test_returns_empty_list_for_no_matches(self):
        """Test that no matches returns empty list."""
        mock_es = MagicMock()
        mock_es.search_glossary.return_value = []

        result = search_glossary(
            es=mock_es,
            scope="scope",
            repository="repo",
            query="nonexistent",
        )

        assert result == []
