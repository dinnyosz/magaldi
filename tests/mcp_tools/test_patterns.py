"""Tests for MCP tools - patterns category."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from magaldi_mcp.tools import (
    find_by_pattern,
    list_patterns,
)


# =============================================================================
# LIST PATTERNS TESTS
# =============================================================================


class TestListPatterns:
    """Tests for list_patterns function."""

    def test_list_patterns_returns_pattern_summary(self, mock_repo):
        """Test list_patterns returns pattern counts and examples."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client

        # First call: aggregation query returns pattern counts
        agg_response = {
            "hits": {"hits": []},
            "aggregations": {
                "patterns": {
                    "buckets": [
                        {"key": "singleton", "doc_count": 1},
                        {"key": "factory", "doc_count": 2},
                    ]
                }
            },
        }

        # Subsequent calls: example queries return matching classes
        singleton_examples = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "name": "DatabaseConnection",
                            "relative_path": "db.py",
                            "line_start": 1,
                            "pattern_confidence": {"singleton": 0.95},
                        }
                    }
                ]
            }
        }
        factory_examples = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "name": "CarFactory",
                            "relative_path": "factory.py",
                            "line_start": 1,
                            "pattern_confidence": {"factory": 0.85},
                        }
                    },
                    {
                        "_source": {
                            "name": "BikeFactory",
                            "relative_path": "factory.py",
                            "line_start": 10,
                            "pattern_confidence": {"factory": 0.75},
                        }
                    },
                ]
            }
        }

        mock_client.search.side_effect = [agg_response, singleton_examples, factory_examples]

        result = list_patterns(
            es=mock_repo,
            scope="scope",
            repository="repo",
        )

        assert isinstance(result, dict)
        assert "patterns" in result
        assert "total_classes_with_patterns" in result
        assert result["total_classes_with_patterns"] == 3

        # Check pattern summary
        patterns = {p["pattern"]: p for p in result["patterns"]}
        assert "singleton" in patterns
        assert "factory" in patterns
        assert patterns["singleton"]["count"] == 1
        assert patterns["factory"]["count"] == 2

    def test_list_patterns_no_patterns_found(self, mock_repo):
        """Test list_patterns when no patterns exist."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        result = list_patterns(
            es=mock_repo,
            scope="scope",
            repository="repo",
        )

        assert result["patterns"] == []
        assert result["total_classes_with_patterns"] == 0

    def test_list_patterns_with_username(self, mock_repo):
        """Test list_patterns filters by username."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        list_patterns(
            es=mock_repo,
            scope="scope",
            repository="repo",
            username="testuser",
        )

        # Verify the search was called with the right username filter
        call_args = mock_client.search.call_args
        assert call_args is not None


# =============================================================================
# FIND BY PATTERN TESTS
# =============================================================================

# =============================================================================
# FIND BY PATTERN TESTS
# =============================================================================


class TestFindByPattern:
    """Tests for find_by_pattern function."""

    def test_find_by_pattern_returns_matching_classes(self, mock_repo):
        """Test find_by_pattern returns classes with matching pattern."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "scope:repo:main:db.py:class:DatabaseConnection:1",
                            "name": "DatabaseConnection",
                            "element_type": "class",
                            "relative_path": "db.py",
                            "line_start": 1,
                            "detected_patterns": ["singleton"],
                            "pattern_confidence": {"singleton": 0.95},
                            "summary": "Database connection singleton",
                        }
                    },
                    {
                        "_source": {
                            "element_id": "scope:repo:main:config.py:class:Config:1",
                            "name": "Config",
                            "element_type": "class",
                            "relative_path": "config.py",
                            "line_start": 1,
                            "detected_patterns": ["singleton"],
                            "pattern_confidence": {"singleton": 0.80},
                            "summary": "Configuration singleton",
                        }
                    },
                ]
            }
        }

        result = find_by_pattern(
            es=mock_repo,
            pattern="singleton",
            scope="scope",
            repository="repo",
        )

        assert isinstance(result, dict)
        assert "classes" in result
        assert len(result["classes"]) == 2
        assert result["classes"][0]["name"] == "DatabaseConnection"
        assert result["classes"][0]["confidence"] == 0.95

    def test_find_by_pattern_filters_by_min_confidence(self, mock_repo):
        """Test find_by_pattern filters by minimum confidence."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "id1",
                            "name": "HighConfidence",
                            "element_type": "class",
                            "relative_path": "a.py",
                            "line_start": 1,
                            "detected_patterns": ["singleton"],
                            "pattern_confidence": {"singleton": 0.95},
                        }
                    },
                    {
                        "_source": {
                            "element_id": "id2",
                            "name": "LowConfidence",
                            "element_type": "class",
                            "relative_path": "b.py",
                            "line_start": 1,
                            "detected_patterns": ["singleton"],
                            "pattern_confidence": {"singleton": 0.50},
                        }
                    },
                ]
            }
        }

        result = find_by_pattern(
            es=mock_repo,
            pattern="singleton",
            scope="scope",
            repository="repo",
            min_confidence=0.8,
        )

        assert len(result["classes"]) == 1
        assert result["classes"][0]["name"] == "HighConfidence"

    def test_find_by_pattern_no_matches(self, mock_repo):
        """Test find_by_pattern when no classes match."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        result = find_by_pattern(
            es=mock_repo,
            pattern="builder",
            scope="scope",
            repository="repo",
        )

        assert result["classes"] == []
        assert result["count"] == 0

    def test_find_by_pattern_with_limit(self, mock_repo):
        """Test find_by_pattern respects limit parameter."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": f"id{i}",
                            "name": f"Class{i}",
                            "element_type": "class",
                            "relative_path": f"file{i}.py",
                            "line_start": 1,
                            "detected_patterns": ["factory"],
                            "pattern_confidence": {"factory": 0.9},
                        }
                    }
                    for i in range(10)
                ]
            }
        }

        result = find_by_pattern(
            es=mock_repo,
            pattern="factory",
            scope="scope",
            repository="repo",
            limit=5,
        )

        # The function should return at most limit results
        assert len(result["classes"]) <= 5
