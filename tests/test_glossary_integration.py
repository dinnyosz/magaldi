"""Integration tests for the glossary feature workflow with MCP tools."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from magaldi_mcp.tools import (
    get_feature_members,
    get_glossary_term,
    list_glossary,
    search_features,
    search_glossary,
)


class TestMCPToolsIntegration:
    """Integration tests for MCP tools with glossary data."""

    @pytest.fixture
    def mock_repo_with_glossary(self):
        """Mock ES repository with glossary data."""
        es = MagicMock()

        # Glossary terms
        glossary_data = [
            {
                "term": "user",
                "total_count": 5,
                "element_ids": ["e1", "e2", "e3", "e4", "e5"],
                "file_paths": ["src/user.py", "src/auth.py"],
                "feature_associations": [
                    {"feature_id": "f1", "feature_label": "Auth", "frequency": 3, "total_members": 4, "percentage": 75.0},
                    {"feature_id": "f2", "feature_label": "User Management", "frequency": 2, "total_members": 3, "percentage": 66.7},
                ],
            },
            {
                "term": "email",
                "total_count": 2,
                "element_ids": ["e6", "e7"],
                "file_paths": ["src/email.py"],
                "feature_associations": [
                    {"feature_id": "f3", "feature_label": "Notifications", "frequency": 2, "total_members": 5, "percentage": 40.0},
                ],
            },
        ]

        es.get_glossary_terms.return_value = glossary_data

        def get_glossary_term_mock(scope, repository, term, username):  # noqa: ARG001
            return next((g for g in glossary_data if g["term"] == term), None)

        def search_glossary_mock(scope, repository, query, username):  # noqa: ARG001
            return [g for g in glossary_data if query.lower() in g["term"]]

        es.get_glossary_term.side_effect = get_glossary_term_mock
        es.search_glossary.side_effect = search_glossary_mock

        return es

    def test_list_glossary_returns_all_terms(self, mock_repo_with_glossary):
        """list_glossary returns all glossary terms for repository."""
        result = list_glossary(
            mock_repo_with_glossary,
            scope="test",
            repository="repo",
            username="main",
        )

        assert len(result) == 2
        terms = [r["term"] for r in result]
        assert "user" in terms
        assert "email" in terms

    def test_get_glossary_term_returns_details(self, mock_repo_with_glossary):
        """get_glossary_term returns full term details with associations."""
        result = get_glossary_term(
            mock_repo_with_glossary,
            scope="test",
            repository="repo",
            term="user",
            username="main",
        )

        assert result is not None
        assert result["term"] == "user"
        assert result["total_count"] == 5
        assert len(result["feature_associations"]) == 2

    def test_search_glossary_finds_matching_terms(self, mock_repo_with_glossary):
        """search_glossary finds terms by partial match."""
        result = search_glossary(
            mock_repo_with_glossary,
            scope="test",
            repository="repo",
            query="us",
            username="main",
        )

        assert len(result) == 1
        assert result[0]["term"] == "user"

    def test_get_feature_members_includes_glossary_terms(self, mock_repo_with_glossary):
        """get_feature_members includes associated glossary terms."""
        # Setup feature document (get_document_by_id_or_hash for feature lookup)
        mock_repo_with_glossary.get_document_by_id_or_hash.return_value = {
            "member_ids": ["e1", "e2"],
        }
        # Setup member documents (get_document for each member)
        mock_repo_with_glossary.get_document.side_effect = [
            # First member
            {
                "element_id": "e1",
                "name": "UserService",
                "element_type": "class",
                "relative_path": "src/user.py",
                "line_start": 10,
            },
            # Second member
            {
                "element_id": "e2",
                "name": "createUser",
                "element_type": "function",
                "relative_path": "src/user.py",
                "line_start": 50,
            },
        ]

        # Setup glossary terms - only terms associated with this feature
        mock_repo_with_glossary.get_glossary_terms.return_value = [
            {
                "term": "user",
                "total_count": 5,
                "feature_associations": [
                    {"feature_id": "test:repo:main:feature:1", "frequency": 2, "percentage": 100.0},
                ],
            },
        ]

        result = get_feature_members(
            mock_repo_with_glossary,
            feature_id="test:repo:main:feature:1",
        )

        assert "members" in result
        assert "glossary_terms" in result
        assert len(result["members"]) == 2

    def test_search_features_with_glossary_term_filter(self, mock_repo_with_glossary):
        """search_features can filter by glossary term."""
        # Setup search results
        mock_repo_with_glossary.search_by_vector.return_value = []
        mock_repo_with_glossary.search_by_keyword.return_value = [
            {
                "element_id": "f1",
                "cluster_label": "Auth",
                "summary": "Authentication features",
                "member_count": 4,
                "element_type": "feature",
            },
        ]

        # Setup glossary term with feature associations
        mock_repo_with_glossary.get_glossary_term.return_value = {
            "term": "user",
            "feature_associations": [
                {"feature_id": "f1", "percentage": 75.0},
            ],
        }

        result = search_features(
            mock_repo_with_glossary,
            embed_client=None,
            query="authentication",
            scope="test",
            repository="repo",
            glossary_term="user",
            min_percentage=50.0,
        )

        # Feature f1 should be included (user term has 75% presence)
        assert len(result) == 1
        assert result[0]["label"] == "Auth"
