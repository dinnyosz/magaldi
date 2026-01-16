"""Tests for web routes/elements.py module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from magaldi_web.routes.elements import router


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_es_repo():
    """Create a mock Elasticsearch repository."""
    return MagicMock()


@pytest.fixture
def app(mock_es_repo):
    """Create FastAPI app with mocked dependencies."""
    app = FastAPI()
    app.include_router(router)

    def get_mock_es_repo():
        yield mock_es_repo

    app.dependency_overrides["magaldi_web.dependencies.get_es_repository"] = get_mock_es_repo
    return app


@pytest.fixture
def client(app, mock_es_repo):
    """Create test client."""
    from magaldi_web.dependencies import get_es_repository

    def get_mock_es_repo():
        yield mock_es_repo

    app.dependency_overrides[get_es_repository] = get_mock_es_repo
    return TestClient(app)


@pytest.fixture
def sample_element():
    """Sample element document."""
    return {
        "element_id": "scope:repo:user:file.py:function:test_func:10",
        "hash_id": "a" * 64,
        "scope": "scope",
        "repository": "repo",
        "username": "user",
        "name": "test_func",
        "element_type": "function",
        "relative_path": "file.py",
        "line_start": 10,
        "line_end": 20,
        "language": "python",
        "summary": "A test function",
        "signature": "def test_func():",
        "docstring": "Test docstring",
        "raw_code": "def test_func():\n    pass",
        "embedding": [0.1] * 1024,
        "decorators": '["@pytest.fixture"]',
        "visibility": "public",
        "is_async": False,
        "parent_id": "scope:repo:user:file.py:class:TestClass:1",
    }


# =============================================================================
# GET SIMILAR ELEMENTS TESTS
# =============================================================================


class TestGetSimilarElements:
    """Tests for GET /elements/similar/{hash_id}."""

    def test_returns_similar_elements(self, client, mock_es_repo, sample_element):
        """Test returns similar elements successfully."""
        mock_es_repo.get_document_by_hash_id.return_value = sample_element

        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "scope:repo:user:file.py:function:similar1:20",
                            "hash_id": "b" * 64,
                            "name": "similar1",
                            "element_type": "function",
                            "relative_path": "file.py",
                            "line_start": 20,
                            "summary": "Similar function 1",
                        },
                        "_score": 1.95,
                    },
                    {
                        "_source": {
                            "element_id": "scope:repo:user:file.py:function:similar2:30",
                            "hash_id": "c" * 64,
                            "name": "similar2",
                            "element_type": "function",
                            "relative_path": "file.py",
                            "line_start": 30,
                            "summary": "Similar function 2",
                        },
                        "_score": 1.85,
                    },
                ]
            }
        }

        response = client.get(f"/elements/similar/{'a' * 64}?limit=10")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["name"] == "similar1"
        assert data[0]["similarity"] == 0.95
        assert data[1]["name"] == "similar2"
        assert data[1]["similarity"] == 0.85

    def test_returns_404_when_element_not_found(self, client, mock_es_repo):
        """Test returns 404 when element not found."""
        mock_es_repo.get_document_by_hash_id.return_value = None

        response = client.get(f"/elements/similar/{'a' * 64}")

        assert response.status_code == 404
        assert "Element not found" in response.json()["detail"]

    def test_returns_400_when_no_embedding(self, client, mock_es_repo, sample_element):
        """Test returns 400 when element has no embedding."""
        element_without_embedding = sample_element.copy()
        del element_without_embedding["embedding"]
        mock_es_repo.get_document_by_hash_id.return_value = element_without_embedding

        response = client.get(f"/elements/similar/{'a' * 64}")

        assert response.status_code == 400
        assert "no embedding" in response.json()["detail"]

    def test_excludes_self_from_results(self, client, mock_es_repo, sample_element):
        """Test excludes the queried element from results."""
        mock_es_repo.get_document_by_hash_id.return_value = sample_element

        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        # Include self in results
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": sample_element["element_id"],  # Self
                            "hash_id": "a" * 64,
                            "name": "test_func",
                            "element_type": "function",
                            "relative_path": "file.py",
                            "line_start": 10,
                        },
                        "_score": 2.0,
                    },
                    {
                        "_source": {
                            "element_id": "scope:repo:user:file.py:function:other:20",
                            "hash_id": "b" * 64,
                            "name": "other",
                            "element_type": "function",
                            "relative_path": "file.py",
                            "line_start": 20,
                        },
                        "_score": 1.9,
                    },
                ]
            }
        }

        response = client.get(f"/elements/similar/{'a' * 64}")

        assert response.status_code == 200
        data = response.json()
        # Self should be excluded
        assert len(data) == 1
        assert data[0]["name"] == "other"

    def test_respects_limit_parameter(self, client, mock_es_repo, sample_element):
        """Test respects the limit parameter."""
        mock_es_repo.get_document_by_hash_id.return_value = sample_element

        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        # Return more results than limit
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": f"scope:repo:user:file.py:function:func{i}:{i * 10}",
                            "hash_id": chr(ord("a") + i) * 64,
                            "name": f"func{i}",
                            "element_type": "function",
                            "relative_path": "file.py",
                            "line_start": i * 10,
                        },
                        "_score": 1.9 - i * 0.1,
                    }
                    for i in range(5)
                ]
            }
        }

        response = client.get(f"/elements/similar/{'a' * 64}?limit=2")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_handles_empty_results(self, client, mock_es_repo, sample_element):
        """Test handles no similar elements."""
        mock_es_repo.get_document_by_hash_id.return_value = sample_element

        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        response = client.get(f"/elements/similar/{'a' * 64}")

        assert response.status_code == 200
        assert response.json() == []


# =============================================================================
# GET ELEMENT DETAIL TESTS
# =============================================================================


class TestGetElementDetail:
    """Tests for GET /elements/{hash_id}."""

    def test_returns_element_detail(self, client, mock_es_repo, sample_element):
        """Test returns element detail successfully."""
        mock_es_repo.get_document_by_hash_id.return_value = sample_element

        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client

        # Mock all search calls
        mock_client.search.return_value = {"hits": {"hits": []}}

        response = client.get(f"/elements/{'a' * 64}")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_func"
        assert data["element_type"] == "function"
        assert data["file_path"] == "file.py"
        assert data["line_start"] == 10
        assert data["summary"] == "A test function"
        assert data["decorators"] == ["@pytest.fixture"]

    def test_returns_404_when_not_found(self, client, mock_es_repo):
        """Test returns 404 when element not found."""
        mock_es_repo.get_document_by_hash_id.return_value = None

        response = client.get(f"/elements/{'a' * 64}")

        assert response.status_code == 404
        assert "Element not found" in response.json()["detail"]

    def test_includes_file_context(self, client, mock_es_repo, sample_element):
        """Test includes file context for non-file elements."""
        mock_es_repo.get_document_by_hash_id.return_value = sample_element

        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client

        def search_side_effect(**kwargs):
            body = kwargs.get("body", {})
            query = body.get("query", {})

            # Check if this is file context query (element_type: file)
            if "bool" in query:
                filters = query.get("bool", {}).get("filter", [])
                for f in filters:
                    if f.get("term", {}).get("element_type") == "file":
                        return {
                            "hits": {
                                "hits": [
                                    {
                                        "_source": {
                                            "element_id": "scope:repo:user:file.py:file:file.py:0",
                                            "hash_id": "f" * 64,
                                            "name": "file.py",
                                            "summary": "File summary",
                                        }
                                    }
                                ]
                            }
                        }
            return {"hits": {"hits": []}}

        mock_client.search.side_effect = search_side_effect

        response = client.get(f"/elements/{'a' * 64}")

        assert response.status_code == 200
        data = response.json()
        assert data["context"]["file"] is not None
        assert data["context"]["file"]["name"] == "file.py"
        assert data["context"]["file"]["summary"] == "File summary"

    def test_includes_parent_context(self, client, mock_es_repo, sample_element):
        """Test includes parent context when element has parent."""
        mock_es_repo.get_document_by_hash_id.return_value = sample_element

        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client

        def search_side_effect(**kwargs):
            body = kwargs.get("body", {})
            query = body.get("query", {})

            # Check if this is parent query (term on element_id)
            if "term" in query and "element_id" in query.get("term", {}):
                return {
                    "hits": {
                        "hits": [
                            {
                                "_source": {
                                    "element_id": sample_element["parent_id"],
                                    "hash_id": "p" * 64,
                                    "name": "TestClass",
                                    "element_type": "class",
                                    "summary": "Parent class",
                                }
                            }
                        ]
                    }
                }
            return {"hits": {"hits": []}}

        mock_client.search.side_effect = search_side_effect

        response = client.get(f"/elements/{'a' * 64}")

        assert response.status_code == 200
        data = response.json()
        assert data["context"]["parent"] is not None
        assert data["context"]["parent"]["name"] == "TestClass"
        assert data["context"]["parent"]["element_type"] == "class"

    def test_includes_children(self, client, mock_es_repo, sample_element):
        """Test includes children elements."""
        mock_es_repo.get_document_by_hash_id.return_value = sample_element

        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client

        def search_side_effect(**kwargs):
            body = kwargs.get("body", {})
            query = body.get("query", {})

            # Check if this is children query (term on parent_id matching element_id)
            if "term" in query and "parent_id" in query.get("term", {}):
                parent_id = query["term"]["parent_id"]
                if parent_id == sample_element["element_id"]:
                    return {
                        "hits": {
                            "hits": [
                                {
                                    "_source": {
                                        "element_id": "scope:repo:user:file.py:function:child1:15",
                                        "hash_id": "c1" + "a" * 62,
                                        "name": "child1",
                                        "element_type": "function",
                                        "line_start": 15,
                                        "summary": "Child 1",
                                        "signature": "def child1():",
                                    }
                                },
                                {
                                    "_source": {
                                        "element_id": "scope:repo:user:file.py:function:child2:18",
                                        "hash_id": "c2" + "a" * 62,
                                        "name": "child2",
                                        "element_type": "method",
                                        "line_start": 18,
                                        "summary": "Child 2",
                                    }
                                },
                            ]
                        }
                    }
            return {"hits": {"hits": []}}

        mock_client.search.side_effect = search_side_effect

        response = client.get(f"/elements/{'a' * 64}")

        assert response.status_code == 200
        data = response.json()
        assert len(data["context"]["children"]) == 2
        assert data["context"]["children"][0]["name"] == "child1"
        assert data["context"]["children"][1]["name"] == "child2"

    def test_includes_siblings(self, client, mock_es_repo, sample_element):
        """Test includes sibling elements when element has parent."""
        mock_es_repo.get_document_by_hash_id.return_value = sample_element

        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client

        def search_side_effect(**kwargs):
            body = kwargs.get("body", {})
            query = body.get("query", {})

            # Check if this is siblings query (bool with filter parent_id and must_not element_id)
            if "bool" in query:
                bool_query = query["bool"]
                if "filter" in bool_query and "must_not" in bool_query:
                    return {
                        "hits": {
                            "hits": [
                                {
                                    "_source": {
                                        "element_id": "scope:repo:user:file.py:function:sibling1:5",
                                        "hash_id": "s1" + "a" * 62,
                                        "name": "sibling1",
                                        "element_type": "function",
                                        "line_start": 5,
                                        "summary": "Sibling 1",
                                    }
                                }
                            ]
                        }
                    }
            return {"hits": {"hits": []}}

        mock_client.search.side_effect = search_side_effect

        response = client.get(f"/elements/{'a' * 64}")

        assert response.status_code == 200
        data = response.json()
        assert len(data["context"]["siblings"]) == 1
        assert data["context"]["siblings"][0]["name"] == "sibling1"

    def test_handles_invalid_decorators_json(self, client, mock_es_repo, sample_element):
        """Test handles invalid JSON in decorators field."""
        element_with_bad_decorators = sample_element.copy()
        element_with_bad_decorators["decorators"] = "not valid json"
        mock_es_repo.get_document_by_hash_id.return_value = element_with_bad_decorators

        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        response = client.get(f"/elements/{'a' * 64}")

        assert response.status_code == 200
        data = response.json()
        assert data["decorators"] == []

    def test_handles_null_decorators(self, client, mock_es_repo, sample_element):
        """Test handles null decorators field."""
        element_without_decorators = sample_element.copy()
        element_without_decorators["decorators"] = None
        mock_es_repo.get_document_by_hash_id.return_value = element_without_decorators

        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        response = client.get(f"/elements/{'a' * 64}")

        assert response.status_code == 200
        data = response.json()
        assert data["decorators"] == []

    def test_skips_file_context_for_file_elements(self, client, mock_es_repo, sample_element):
        """Test skips file context lookup for file type elements."""
        file_element = sample_element.copy()
        file_element["element_type"] = "file"
        file_element["parent_id"] = None
        mock_es_repo.get_document_by_hash_id.return_value = file_element

        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        response = client.get(f"/elements/{'a' * 64}")

        assert response.status_code == 200
        data = response.json()
        # File context should be None for file elements
        assert data["context"]["file"] is None

    def test_handles_element_without_parent(self, client, mock_es_repo, sample_element):
        """Test handles element without parent."""
        element_no_parent = sample_element.copy()
        element_no_parent["parent_id"] = None
        mock_es_repo.get_document_by_hash_id.return_value = element_no_parent

        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        response = client.get(f"/elements/{'a' * 64}")

        assert response.status_code == 200
        data = response.json()
        assert data["context"]["parent"] is None
        assert data["context"]["siblings"] == []

    def test_includes_repository_info(self, client, mock_es_repo, sample_element):
        """Test includes repository info in response."""
        mock_es_repo.get_document_by_hash_id.return_value = sample_element

        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        response = client.get(f"/elements/{'a' * 64}")

        assert response.status_code == 200
        data = response.json()
        assert data["repository"]["scope"] == "scope"
        assert data["repository"]["name"] == "repo"

    def test_handles_missing_optional_fields(self, client, mock_es_repo):
        """Test handles missing optional fields in element."""
        minimal_element = {
            "element_id": "scope:repo:user:file.py:function:test:10",
            "scope": "scope",
            "repository": "repo",
            "username": "user",
            "name": "test",
            "element_type": "function",
            "relative_path": "file.py",
            "line_start": 10,
        }
        mock_es_repo.get_document_by_hash_id.return_value = minimal_element

        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        response = client.get(f"/elements/{'a' * 64}")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test"
        assert data["hash_id"] is None
        assert data["language"] == "unknown"
        assert data["summary"] is None
        assert data["is_async"] is False
