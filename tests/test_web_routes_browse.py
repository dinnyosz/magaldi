"""Tests for web routes/browse.py module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from magaldi_web.routes.browse import router


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_repo():
    """Create a mock Elasticsearch repository."""
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
# BROWSE ELEMENTS TESTS
# =============================================================================


class TestBrowseElements:
    """Tests for GET /browse/elements."""

    def test_returns_empty_list_when_no_elements(self, client, mock_repo):
        """Test returns empty list when no elements exist."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"total": {"value": 0}, "hits": []}}

        response = client.get("/browse/elements")

        assert response.status_code == 200
        data = response.json()
        assert data["elements"] == []
        assert data["total"] == 0
        assert data["page"] == 1

    def test_returns_elements_with_pagination(self, client, mock_repo):
        """Test returns elements with pagination info."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "total": {"value": 100},
                "hits": [
                    {
                        "_source": {
                            "element_id": "id1",
                            "name": "func1",
                            "element_type": "function",
                            "relative_path": "file.py",
                            "line_start": 10,
                            "repository": "repo",
                            "scope": "scope",
                        }
                    },
                    {
                        "_source": {
                            "element_id": "id2",
                            "name": "func2",
                            "element_type": "function",
                            "relative_path": "file.py",
                            "line_start": 20,
                            "repository": "repo",
                            "scope": "scope",
                        }
                    },
                ],
            }
        }
        mock_client.mget.return_value = {"docs": []}

        response = client.get("/browse/elements?page=1&limit=10")

        assert response.status_code == 200
        data = response.json()
        assert len(data["elements"]) == 2
        assert data["total"] == 100
        assert data["page"] == 1
        assert data["limit"] == 10
        assert data["total_pages"] == 10

    def test_filters_by_scope(self, client, mock_repo):
        """Test filters by scope."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"total": {"value": 0}, "hits": []}}

        response = client.get("/browse/elements?scope=github")

        assert response.status_code == 200
        # Verify scope filter was applied
        call_args = mock_client.search.call_args
        body = call_args[1]["body"]
        filters = body["query"]["bool"]["filter"]
        scope_filters = [f for f in filters if "scope" in f.get("term", {})]
        assert len(scope_filters) == 1
        assert scope_filters[0]["term"]["scope"] == "github"

    def test_filters_by_repository(self, client, mock_repo):
        """Test filters by repository."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"total": {"value": 0}, "hits": []}}

        response = client.get("/browse/elements?repository=myproject")

        assert response.status_code == 200
        call_args = mock_client.search.call_args
        body = call_args[1]["body"]
        filters = body["query"]["bool"]["filter"]
        repo_filters = [f for f in filters if "repository" in f.get("term", {})]
        assert len(repo_filters) == 1

    def test_filters_by_element_type(self, client, mock_repo):
        """Test filters by element type."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"total": {"value": 0}, "hits": []}}

        response = client.get("/browse/elements?element_type=function")

        assert response.status_code == 200
        call_args = mock_client.search.call_args
        body = call_args[1]["body"]
        filters = body["query"]["bool"]["filter"]
        type_filters = [f for f in filters if "element_type" in f.get("term", {})]
        assert len(type_filters) == 1

    def test_filters_by_language(self, client, mock_repo):
        """Test filters by language."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"total": {"value": 0}, "hits": []}}

        response = client.get("/browse/elements?language=python")

        assert response.status_code == 200
        call_args = mock_client.search.call_args
        body = call_args[1]["body"]
        filters = body["query"]["bool"]["filter"]
        lang_filters = [f for f in filters if "language" in f.get("term", {})]
        assert len(lang_filters) == 1

    def test_filters_by_parent_id(self, client, mock_repo):
        """Test filters by parent_id."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"total": {"value": 0}, "hits": []}}

        response = client.get("/browse/elements?parent_id=parent123")

        assert response.status_code == 200
        call_args = mock_client.search.call_args
        body = call_args[1]["body"]
        filters = body["query"]["bool"]["filter"]
        parent_filters = [f for f in filters if "parent_id" in f.get("term", {})]
        assert len(parent_filters) == 1

    def test_includes_user_branch_in_filter(self, client, mock_repo):
        """Test includes user branch along with main in filter."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"total": {"value": 0}, "hits": []}}

        response = client.get("/browse/elements?username=developer")

        assert response.status_code == 200
        call_args = mock_client.search.call_args
        body = call_args[1]["body"]
        filters = body["query"]["bool"]["filter"]
        username_filters = [f for f in filters if "username" in f.get("terms", {})]
        assert len(username_filters) == 1
        assert "main" in username_filters[0]["terms"]["username"]
        assert "developer" in username_filters[0]["terms"]["username"]

    def test_resolves_parent_info(self, client, mock_repo):
        """Test resolves parent info for container."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "element_id": "id1",
                            "name": "method1",
                            "element_type": "method",
                            "relative_path": "file.py",
                            "line_start": 10,
                            "repository": "repo",
                            "scope": "scope",
                            "parent_id": "parent_class_id",
                        }
                    }
                ],
            }
        }
        mock_client.mget.return_value = {
            "docs": [
                {
                    "_id": "parent_class_id",
                    "found": True,
                    "_source": {
                        "name": "MyClass",
                        "element_type": "class",
                        "relative_path": "file.py",
                    },
                }
            ]
        }

        response = client.get("/browse/elements")

        assert response.status_code == 200
        data = response.json()
        assert len(data["elements"]) == 1
        assert data["elements"][0]["container"]["name"] == "MyClass"
        assert data["elements"][0]["container"]["element_type"] == "class"

    def test_handles_unfound_parent(self, client, mock_repo):
        """Test handles unfound parent gracefully."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "element_id": "id1",
                            "name": "func1",
                            "element_type": "function",
                            "relative_path": "file.py",
                            "line_start": 10,
                            "repository": "repo",
                            "scope": "scope",
                            "parent_id": "missing_parent",
                        }
                    }
                ],
            }
        }
        mock_client.mget.return_value = {
            "docs": [{"_id": "missing_parent", "found": False}]
        }

        response = client.get("/browse/elements")

        assert response.status_code == 200
        data = response.json()
        assert data["elements"][0]["container"] is None


# =============================================================================
# GET ELEMENT CHILDREN TESTS
# =============================================================================


class TestGetElementChildren:
    """Tests for GET /browse/element/{hash_id}/children."""

    def test_returns_children_grouped_by_type(self, client, mock_repo):
        """Test returns children grouped by type."""
        mock_repo.get_document_by_hash_id.return_value = {
            "element_id": "parent_id",
        }
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "c1",
                            "name": "method1",
                            "element_type": "method",
                            "line_start": 10,
                        }
                    },
                    {
                        "_source": {
                            "element_id": "c2",
                            "name": "method2",
                            "element_type": "method",
                            "line_start": 20,
                        }
                    },
                    {
                        "_source": {
                            "element_id": "c3",
                            "name": "var1",
                            "element_type": "variable",
                            "line_start": 5,
                        }
                    },
                ]
            }
        }

        response = client.get("/browse/element/abcd1234/children")

        assert response.status_code == 200
        data = response.json()
        assert "method" in data["children"]
        assert "variable" in data["children"]
        assert len(data["children"]["method"]) == 2
        assert len(data["children"]["variable"]) == 1
        assert data["total_children"] == 3

    def test_returns_error_when_element_not_found(self, client, mock_repo):
        """Test returns error when element not found."""
        mock_repo.get_document_by_hash_id.return_value = None

        response = client.get("/browse/element/nonexistent/children")

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"] == "Element not found"
        assert data["children"] == {}
        assert data["total_children"] == 0

    def test_orders_types_logically(self, client, mock_repo):
        """Test orders types in logical order."""
        mock_repo.get_document_by_hash_id.return_value = {"element_id": "parent_id"}
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {"_source": {"element_id": "1", "name": "v1", "element_type": "variable", "line_start": 1}},
                    {"_source": {"element_id": "2", "name": "c1", "element_type": "class", "line_start": 2}},
                    {"_source": {"element_id": "3", "name": "f1", "element_type": "function", "line_start": 3}},
                ]
            }
        }

        response = client.get("/browse/element/hash123/children")

        assert response.status_code == 200
        data = response.json()
        # class should come before function, function before variable
        keys = list(data["children"].keys())
        assert keys.index("class") < keys.index("function")
        assert keys.index("function") < keys.index("variable")


# =============================================================================
# GET BROWSE FILTERS TESTS
# =============================================================================


class TestGetBrowseFilters:
    """Tests for GET /browse/filters."""

    def test_returns_all_filter_options(self, client, mock_repo):
        """Test returns all filter options."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "aggregations": {
                "scopes": {"buckets": [{"key": "github"}, {"key": "gitlab"}]},
                "repositories": {
                    "buckets": [
                        {"key": {"scope": "github", "repository": "project1"}},
                        {"key": {"scope": "gitlab", "repository": "project2"}},
                    ]
                },
                "element_types": {"buckets": [{"key": "function"}, {"key": "class"}]},
                "languages": {"buckets": [{"key": "python"}, {"key": "javascript"}]},
                "usernames": {"buckets": [{"key": "main"}, {"key": "developer"}]},
            }
        }

        response = client.get("/browse/filters")

        assert response.status_code == 200
        data = response.json()
        assert data["scopes"] == ["github", "gitlab"]
        assert len(data["repositories"]) == 2
        assert data["repositories"][0]["scope"] == "github"
        assert data["element_types"] == ["function", "class"]
        assert data["languages"] == ["python", "javascript"]
        assert data["usernames"] == ["main", "developer"]

    def test_handles_empty_aggregations(self, client, mock_repo):
        """Test handles empty aggregations."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"aggregations": {}}

        response = client.get("/browse/filters")

        assert response.status_code == 200
        data = response.json()
        assert data["scopes"] == []
        assert data["repositories"] == []
        assert data["element_types"] == []
        assert data["languages"] == []
        assert data["usernames"] == []


# =============================================================================
# GET BROWSE STATS TESTS
# =============================================================================


class TestGetBrowseStats:
    """Tests for GET /browse/stats."""

    def test_returns_stats(self, client, mock_repo):
        """Test returns element statistics."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "aggregations": {
                "by_type": {
                    "buckets": [
                        {"key": "function", "doc_count": 100},
                        {"key": "class", "doc_count": 20},
                        {"key": "method", "doc_count": 80},
                    ]
                },
                "by_language": {
                    "buckets": [
                        {"key": "python", "doc_count": 150},
                        {"key": "javascript", "doc_count": 50},
                    ]
                },
            }
        }

        response = client.get("/browse/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["type_counts"]["function"] == 100
        assert data["type_counts"]["class"] == 20
        assert data["type_counts"]["method"] == 80
        assert data["language_counts"]["python"] == 150
        assert data["total"] == 200

    def test_filters_by_scope_and_repo(self, client, mock_repo):
        """Test filters by scope and repository."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "aggregations": {
                "by_type": {"buckets": []},
                "by_language": {"buckets": []},
            }
        }

        response = client.get("/browse/stats?scope=github&repository=myproject")

        assert response.status_code == 200
        call_args = mock_client.search.call_args
        body = call_args[1]["body"]
        filters = body["query"]["bool"]["filter"]
        scope_filters = [f for f in filters if "scope" in f.get("term", {})]
        repo_filters = [f for f in filters if "repository" in f.get("term", {})]
        assert len(scope_filters) == 1
        assert len(repo_filters) == 1


# =============================================================================
# GET ELEMENT DETAILS TESTS
# =============================================================================


class TestGetElementDetails:
    """Tests for GET /browse/element/{hash_id}/details."""

    def test_returns_element_details(self, client, mock_repo):
        """Test returns element details."""
        mock_repo.get_document_by_hash_id.return_value = {
            "element_id": "id1",
            "hash_id": "h1",
            "name": "my_function",
            "element_type": "function",
            "relative_path": "file.py",
            "line_start": 10,
            "line_end": 20,
            "language": "python",
            "summary": "A function",
            "signature": "def my_function():",
            "repository": "repo",
            "scope": "scope",
        }
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.count.return_value = {"count": 0}

        response = client.get("/browse/element/hash123/details")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "my_function"
        assert data["element_type"] == "function"
        assert data["file_path"] == "file.py"
        assert data["containers"] == []
        assert data["child_count"] == 0

    def test_returns_error_when_not_found(self, client, mock_repo):
        """Test returns error when element not found."""
        mock_repo.get_document_by_hash_id.return_value = None

        response = client.get("/browse/element/nonexistent/details")

        assert response.status_code == 200
        data = response.json()
        assert data["error"] == "Element not found"

    def test_builds_container_chain(self, client, mock_repo):
        """Test builds container chain from parent to file."""
        mock_repo.get_document_by_hash_id.return_value = {
            "element_id": "method_id",
            "name": "method1",
            "element_type": "method",
            "relative_path": "file.py",
            "parent_id": "class_id",
            "repository": "repo",
            "scope": "scope",
        }
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client

        # First call: get parent (class)
        # Second call: get grandparent (file)
        get_count = [0]

        def get_side_effect(**kwargs):
            get_count[0] += 1
            if get_count[0] == 1:
                return {
                    "_source": {
                        "element_id": "class_id",
                        "hash_id": "class_hash",
                        "name": "MyClass",
                        "element_type": "class",
                        "relative_path": "file.py",
                        "line_start": 5,
                        "parent_id": "file_id",
                    }
                }
            else:
                return {
                    "_source": {
                        "element_id": "file_id",
                        "hash_id": "file_hash",
                        "name": "file.py",
                        "element_type": "file",
                        "relative_path": "file.py",
                        "line_start": 0,
                    }
                }

        mock_client.get_document.side_effect = get_side_effect
        mock_client.count.return_value = {"count": 0}

        response = client.get("/browse/element/hash123/details")

        assert response.status_code == 200
        data = response.json()
        assert len(data["containers"]) == 2
        assert data["containers"][0]["name"] == "MyClass"
        assert data["containers"][0]["element_type"] == "class"
        assert data["containers"][1]["name"] == "file.py"

    def test_includes_call_graph_when_requested(self, client, mock_repo):
        """Test includes call graph when requested."""
        mock_repo.get_document_by_hash_id.return_value = {
            "element_id": "func_id",
            "name": "my_function",
            "element_type": "function",
            "relative_path": "file.py",
            "raw_code": "def my_function():\n    other_func()\n    helper()",
            "repository": "repo",
            "scope": "scope",
        }
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.count.return_value = {"count": 0}

        # Search for callers
        search_count = [0]

        def search_side_effect(**kwargs):
            search_count[0] += 1
            if search_count[0] == 1:
                # Callers search
                return {
                    "hits": {
                        "hits": [
                            {
                                "_source": {
                                    "element_id": "caller1",
                                    "name": "caller_func",
                                    "element_type": "function",
                                    "relative_path": "other.py",
                                    "line_start": 50,
                                }
                            }
                        ]
                    }
                }
            else:
                # Callees search
                return {
                    "hits": {
                        "hits": [
                            {
                                "_source": {
                                    "element_id": "callee1",
                                    "name": "other_func",
                                    "element_type": "function",
                                    "relative_path": "utils.py",
                                    "line_start": 10,
                                }
                            }
                        ]
                    }
                }

        mock_client.search.side_effect = search_side_effect

        response = client.get("/browse/element/hash123/details?include_call_graph=true")

        assert response.status_code == 200
        data = response.json()
        assert "callers" in data
        assert "callees" in data
        assert len(data["callers"]) == 1
        assert data["callers"][0]["name"] == "caller_func"
        assert len(data["callees"]) == 1
        assert data["callees"][0]["name"] == "other_func"

    def test_skips_call_graph_for_non_functions(self, client, mock_repo):
        """Test skips call graph for non-function elements."""
        mock_repo.get_document_by_hash_id.return_value = {
            "element_id": "class_id",
            "name": "MyClass",
            "element_type": "class",
            "relative_path": "file.py",
            "repository": "repo",
            "scope": "scope",
        }
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.count.return_value = {"count": 0}

        response = client.get("/browse/element/hash123/details?include_call_graph=true")

        assert response.status_code == 200
        data = response.json()
        assert "callers" not in data
        assert "callees" not in data

    def test_handles_parent_lookup_error(self, client, mock_repo):
        """Test handles error when looking up parent."""
        mock_repo.get_document_by_hash_id.return_value = {
            "element_id": "id1",
            "name": "func1",
            "element_type": "function",
            "relative_path": "file.py",
            "parent_id": "missing_parent",
            "repository": "repo",
            "scope": "scope",
        }
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.get_document.side_effect = Exception("Not found")
        mock_client.count.return_value = {"count": 0}

        response = client.get("/browse/element/hash123/details")

        assert response.status_code == 200
        data = response.json()
        # Should not crash, containers should be empty
        assert data["containers"] == []

    def test_counts_children(self, client, mock_repo):
        """Test counts children correctly."""
        mock_repo.get_document_by_hash_id.return_value = {
            "element_id": "class_id",
            "name": "MyClass",
            "element_type": "class",
            "relative_path": "file.py",
            "repository": "repo",
            "scope": "scope",
        }
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.count.return_value = {"count": 5}

        response = client.get("/browse/element/hash123/details")

        assert response.status_code == 200
        data = response.json()
        assert data["child_count"] == 5
