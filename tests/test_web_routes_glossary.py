"""Tests for glossary API routes."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from magaldi_web.routes.glossary import router

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_repo():
    """Create mock ES repository."""
    return MagicMock()


@pytest.fixture
def client(mock_repo):
    """Create test client with mocked dependencies."""
    app = FastAPI()
    app.include_router(router)

    from magaldi_web.dependencies import get_repository

    def get_mock_repo():
        yield mock_repo

    app.dependency_overrides[get_repository] = get_mock_repo
    return TestClient(app)


# =============================================================================
# LIST GLOSSARY TERMS TESTS
# =============================================================================


class TestListGlossaryTerms:
    """Tests for GET /repos/{scope}/{repo}/glossary endpoint."""

    def test_returns_glossary_terms(self, client, mock_repo):
        """Returns list of glossary terms."""
        mock_repo.get_glossary_terms.return_value = [
            {
                "term": "user",
                "total_count": 5,
                "file_paths": ["a.py", "b.py"],
                "feature_associations": [],
            },
            {
                "term": "email",
                "total_count": 3,
                "file_paths": ["c.py"],
                "feature_associations": [],
            },
        ]

        response = client.get("/repos/scope/repo/glossary")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["terms"]) == 2
        assert data["terms"][0]["term"] == "user"  # Sorted by count desc

    def test_filters_by_min_count(self, client, mock_repo):
        """Passes min_count to repository for filtering."""
        # ES filters by min_count, so mock returns only matching terms
        mock_repo.get_glossary_terms.return_value = [
            {"term": "user", "total_count": 5, "file_paths": [], "feature_associations": []},
        ]

        response = client.get("/repos/scope/repo/glossary?min_count=3")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["terms"][0]["term"] == "user"
        # Verify min_count was passed to the repository
        mock_repo.get_glossary_terms.assert_called_once_with("scope", "repo", "main", 3)

    def test_passes_username_to_repository(self, client, mock_repo):
        """Passes username parameter to repository."""
        mock_repo.get_glossary_terms.return_value = []

        response = client.get("/repos/scope/repo/glossary?username=developer")

        assert response.status_code == 200
        mock_repo.get_glossary_terms.assert_called_once_with("scope", "repo", "developer", 1)

    def test_uses_default_username_main(self, client, mock_repo):
        """Uses 'main' as default username."""
        mock_repo.get_glossary_terms.return_value = []

        response = client.get("/repos/scope/repo/glossary")

        assert response.status_code == 200
        mock_repo.get_glossary_terms.assert_called_once_with("scope", "repo", "main", 1)

    def test_returns_empty_list_when_no_terms(self, client, mock_repo):
        """Returns empty list when no terms exist."""
        mock_repo.get_glossary_terms.return_value = []

        response = client.get("/repos/scope/repo/glossary")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["terms"] == []

    def test_sorts_terms_by_count_descending(self, client, mock_repo):
        """Returns terms sorted by total_count in descending order (sorted by ES)."""
        # ES already returns sorted results, mock reflects this
        mock_repo.get_glossary_terms.return_value = [
            {"term": "large", "total_count": 10, "file_paths": [], "feature_associations": []},
            {"term": "medium", "total_count": 5, "file_paths": [], "feature_associations": []},
            {"term": "small", "total_count": 2, "file_paths": [], "feature_associations": []},
        ]

        response = client.get("/repos/scope/repo/glossary")

        assert response.status_code == 200
        data = response.json()
        assert data["terms"][0]["term"] == "large"
        assert data["terms"][1]["term"] == "medium"
        assert data["terms"][2]["term"] == "small"

    def test_calculates_file_count_from_file_paths(self, client, mock_repo):
        """Calculates file_count from file_paths list length."""
        mock_repo.get_glossary_terms.return_value = [
            {
                "term": "user",
                "total_count": 5,
                "file_paths": ["a.py", "b.py", "c.py"],
                "feature_associations": [],
            },
        ]

        response = client.get("/repos/scope/repo/glossary")

        assert response.status_code == 200
        data = response.json()
        assert data["terms"][0]["file_count"] == 3

    def test_calculates_feature_count_from_associations(self, client, mock_repo):
        """Calculates feature_count from feature_associations list length."""
        mock_repo.get_glossary_terms.return_value = [
            {
                "term": "user",
                "total_count": 5,
                "file_paths": [],
                "feature_associations": [
                    {"feature_id": "f1", "feature_label": "Auth"},
                    {"feature_id": "f2", "feature_label": "Users"},
                ],
            },
        ]

        response = client.get("/repos/scope/repo/glossary")

        assert response.status_code == 200
        data = response.json()
        assert data["terms"][0]["feature_count"] == 2


# =============================================================================
# GET GLOSSARY TERM TESTS
# =============================================================================


class TestGetGlossaryTerm:
    """Tests for GET /repos/{scope}/{repo}/glossary/{term} endpoint."""

    def test_returns_term_details(self, client, mock_repo):
        """Returns detailed term information."""
        mock_repo.get_glossary_term.return_value = {
            "term": "user",
            "total_count": 5,
            "element_ids": ["e1", "e2"],
            "file_paths": ["a.py"],
            "feature_associations": [
                {
                    "feature_id": "f1",
                    "feature_label": "Auth",
                    "frequency": 3,
                    "total_members": 4,
                    "percentage": 75.0,
                }
            ],
        }

        response = client.get("/repos/scope/repo/glossary/user")

        assert response.status_code == 200
        data = response.json()
        assert data["term"] == "user"
        assert data["total_count"] == 5
        assert len(data["feature_associations"]) == 1

    def test_returns_404_for_unknown_term(self, client, mock_repo):
        """Returns 404 for unknown term."""
        mock_repo.get_glossary_term.return_value = None

        response = client.get("/repos/scope/repo/glossary/unknown")

        assert response.status_code == 404
        data = response.json()
        assert "unknown" in data["detail"]

    def test_passes_parameters_to_repository(self, client, mock_repo):
        """Passes all parameters correctly to repository."""
        mock_repo.get_glossary_term.return_value = None

        response = client.get("/repos/myscope/myrepo/glossary/myterm?username=developer")

        assert response.status_code == 404
        mock_repo.get_glossary_term.assert_called_once_with(
            "myscope", "myrepo", "myterm", "developer"
        )

    def test_returns_element_ids(self, client, mock_repo):
        """Returns list of element IDs where term appears."""
        mock_repo.get_glossary_term.return_value = {
            "term": "user",
            "total_count": 3,
            "element_ids": ["elem1", "elem2", "elem3"],
            "file_paths": [],
            "feature_associations": [],
        }

        response = client.get("/repos/scope/repo/glossary/user")

        assert response.status_code == 200
        data = response.json()
        assert data["element_ids"] == ["elem1", "elem2", "elem3"]

    def test_returns_file_paths(self, client, mock_repo):
        """Returns list of file paths where term appears."""
        mock_repo.get_glossary_term.return_value = {
            "term": "user",
            "total_count": 2,
            "element_ids": [],
            "file_paths": ["src/auth.py", "src/models.py"],
            "feature_associations": [],
        }

        response = client.get("/repos/scope/repo/glossary/user")

        assert response.status_code == 200
        data = response.json()
        assert data["file_paths"] == ["src/auth.py", "src/models.py"]

    def test_returns_feature_association_details(self, client, mock_repo):
        """Returns full feature association details."""
        mock_repo.get_glossary_term.return_value = {
            "term": "user",
            "total_count": 5,
            "element_ids": [],
            "file_paths": [],
            "feature_associations": [
                {
                    "feature_id": "feature:auth",
                    "feature_label": "Authentication",
                    "frequency": 3,
                    "total_members": 10,
                    "percentage": 30.0,
                },
                {
                    "feature_id": "feature:users",
                    "feature_label": "User Management",
                    "frequency": 2,
                    "total_members": 8,
                    "percentage": 25.0,
                },
            ],
        }

        response = client.get("/repos/scope/repo/glossary/user")

        assert response.status_code == 200
        data = response.json()
        assert len(data["feature_associations"]) == 2
        assert data["feature_associations"][0]["feature_id"] == "feature:auth"
        assert data["feature_associations"][0]["feature_label"] == "Authentication"
        assert data["feature_associations"][0]["frequency"] == 3
        assert data["feature_associations"][0]["total_members"] == 10
        assert data["feature_associations"][0]["percentage"] == 30.0


# =============================================================================
# SEARCH GLOSSARY TERMS TESTS
# =============================================================================


class TestSearchGlossaryTerms:
    """Tests for GET /repos/{scope}/{repo}/glossary/search/{query} endpoint."""

    def test_searches_terms_by_partial_match(self, client, mock_repo):
        """Searches terms by partial match."""
        mock_repo.search_glossary.return_value = [
            {"term": "user", "total_count": 5, "file_paths": [], "feature_associations": []},
            {"term": "username", "total_count": 2, "file_paths": [], "feature_associations": []},
        ]

        response = client.get("/repos/scope/repo/glossary/search/us")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2

    def test_passes_parameters_to_repository(self, client, mock_repo):
        """Passes all parameters correctly to repository."""
        mock_repo.search_glossary.return_value = []

        response = client.get("/repos/myscope/myrepo/glossary/search/query?username=developer")

        assert response.status_code == 200
        mock_repo.search_glossary.assert_called_once_with(
            "myscope", "myrepo", "query", "developer"
        )

    def test_returns_empty_list_when_no_matches(self, client, mock_repo):
        """Returns empty list when no matches found."""
        mock_repo.search_glossary.return_value = []

        response = client.get("/repos/scope/repo/glossary/search/xyz")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["terms"] == []

    def test_returns_term_summary_structure(self, client, mock_repo):
        """Returns correctly structured term summaries."""
        mock_repo.search_glossary.return_value = [
            {
                "term": "config",
                "total_count": 10,
                "file_paths": ["a.py", "b.py"],
                "feature_associations": [{"feature_id": "f1"}],
            },
        ]

        response = client.get("/repos/scope/repo/glossary/search/conf")

        assert response.status_code == 200
        data = response.json()
        assert len(data["terms"]) == 1
        term = data["terms"][0]
        assert term["term"] == "config"
        assert term["total_count"] == 10
        assert term["file_count"] == 2
        assert term["feature_count"] == 1

    def test_uses_default_username_main(self, client, mock_repo):
        """Uses 'main' as default username."""
        mock_repo.search_glossary.return_value = []

        response = client.get("/repos/scope/repo/glossary/search/test")

        assert response.status_code == 200
        mock_repo.search_glossary.assert_called_once_with("scope", "repo", "test", "main")
