"""Tests for web routes/repos.py module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from magaldi_web.routes.repos import router


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
# LIST REPOS TESTS
# =============================================================================


class TestListRepos:
    """Tests for GET /repos."""

    def test_returns_empty_list_when_no_repos(self, client, mock_repo):
        """Test returns empty list when no repositories exist."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "aggregations": {"repos": {"buckets": []}}
        }

        response = client.get("/repos")

        assert response.status_code == 200
        data = response.json()
        assert data["repos"] == []
        assert data["total"] == 0

    def test_returns_repos_list(self, client, mock_repo):
        """Test returns list of repositories."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "aggregations": {
                "repos": {
                    "buckets": [
                        {
                            "key": {"scope": "github", "repository": "project1"},
                            "doc_count": 100,
                            "file_count": {"doc_count": 10},
                            "languages": {
                                "buckets": [
                                    {"key": "python"},
                                    {"key": "javascript"},
                                ]
                            },
                        },
                        {
                            "key": {"scope": "gitlab", "repository": "project2"},
                            "doc_count": 50,
                            "file_count": {"doc_count": 5},
                            "languages": {"buckets": [{"key": "rust"}]},
                        },
                    ]
                }
            }
        }

        response = client.get("/repos")

        assert response.status_code == 200
        data = response.json()
        assert len(data["repos"]) == 2
        assert data["total"] == 2

        assert data["repos"][0]["scope"] == "github"
        assert data["repos"][0]["name"] == "project1"
        assert data["repos"][0]["file_count"] == 10
        assert data["repos"][0]["element_count"] == 100
        assert data["repos"][0]["languages"] == ["python", "javascript"]

        assert data["repos"][1]["scope"] == "gitlab"
        assert data["repos"][1]["name"] == "project2"
        assert data["repos"][1]["languages"] == ["rust"]

    def test_handles_missing_aggregations(self, client, mock_repo):
        """Test handles missing aggregations gracefully."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {}

        response = client.get("/repos")

        assert response.status_code == 200
        data = response.json()
        assert data["repos"] == []
        assert data["total"] == 0

    def test_handles_missing_file_count(self, client, mock_repo):
        """Test handles missing file_count in bucket."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "aggregations": {
                "repos": {
                    "buckets": [
                        {
                            "key": {"scope": "github", "repository": "project"},
                            "doc_count": 50,
                            "languages": {"buckets": []},
                        }
                    ]
                }
            }
        }

        response = client.get("/repos")

        assert response.status_code == 200
        data = response.json()
        assert data["repos"][0]["file_count"] == 0


# =============================================================================
# GET REPO DETAIL TESTS
# =============================================================================


class TestGetRepoDetail:
    """Tests for GET /repos/{scope}/{repository}."""

    def test_returns_repo_detail(self, client, mock_repo):
        """Test returns repository detail successfully."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client

        # First call: repo stats
        # Second call: active users
        call_count = [0]

        def search_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "hits": {"total": {"value": 100}},
                    "aggregations": {
                        "file_count": {"doc_count": 10},
                        "languages": {
                            "buckets": [{"key": "python"}, {"key": "typescript"}]
                        },
                    },
                }
            else:
                return {
                    "aggregations": {
                        "users": {
                            "buckets": [
                                {"key": "developer1", "file_count": {"doc_count": 3}},
                                {"key": "developer2", "file_count": {"doc_count": 1}},
                            ]
                        }
                    }
                }

        mock_client.search.side_effect = search_side_effect

        response = client.get("/repos/github/myproject")

        assert response.status_code == 200
        data = response.json()
        assert data["scope"] == "github"
        assert data["name"] == "myproject"
        assert data["file_count"] == 10
        assert data["element_count"] == 100
        assert data["languages"] == ["python", "typescript"]
        assert len(data["active_users"]) == 2
        assert data["active_users"][0]["username"] == "developer1"
        assert data["active_users"][0]["files_modified"] == 3

    def test_returns_404_when_not_found(self, client, mock_repo):
        """Test returns 404 when repository not found."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {"total": {"value": 0}},
            "aggregations": {},
        }

        response = client.get("/repos/nonexistent/repo")

        assert response.status_code == 404
        assert "Repository not found" in response.json()["detail"]

    def test_handlrepo_with_no_active_users(self, client, mock_repo):
        """Test handles repo with no active users."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client

        call_count = [0]

        def search_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "hits": {"total": {"value": 50}},
                    "aggregations": {
                        "file_count": {"doc_count": 5},
                        "languages": {"buckets": []},
                    },
                }
            else:
                return {"aggregations": {"users": {"buckets": []}}}

        mock_client.search.side_effect = search_side_effect

        response = client.get("/repos/github/project")

        assert response.status_code == 200
        data = response.json()
        assert data["active_users"] == []


# =============================================================================
# GET FILE TREE TESTS
# =============================================================================


class TestGetFileTree:
    """Tests for GET /repos/{scope}/{repository}/tree."""

    def test_returns_empty_tree(self, client, mock_repo):
        """Test returns empty tree when no files."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        response = client.get("/repos/github/project/tree")

        assert response.status_code == 200
        data = response.json()
        assert data["tree"] == []
        assert data["scope"] == "github"
        assert data["repository"] == "project"

    def test_returns_flat_file_list(self, client, mock_repo):
        """Test returns flat file list (no subdirectories)."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "relative_path": "main.py",
                            "language": "python",
                            "summary": "Main file",
                        }
                    },
                    {
                        "_source": {
                            "relative_path": "utils.py",
                            "language": "python",
                        }
                    },
                ]
            }
        }

        response = client.get("/repos/github/project/tree")

        assert response.status_code == 200
        data = response.json()
        assert len(data["tree"]) == 2
        assert data["tree"][0]["name"] == "main.py"
        assert data["tree"][0]["type"] == "file"
        assert data["tree"][0]["path"] == "main.py"
        assert data["tree"][0]["language"] == "python"

    def test_returns_nested_tree(self, client, mock_repo):
        """Test returns nested directory tree."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "relative_path": "src/main.py",
                            "language": "python",
                        }
                    },
                    {
                        "_source": {
                            "relative_path": "src/utils/helpers.py",
                            "language": "python",
                        }
                    },
                    {
                        "_source": {
                            "relative_path": "tests/test_main.py",
                            "language": "python",
                        }
                    },
                ]
            }
        }

        response = client.get("/repos/github/project/tree")

        assert response.status_code == 200
        data = response.json()
        # Should have 2 top-level directories: src and tests
        assert len(data["tree"]) == 2
        # Directories should come first (sorted)
        dir_names = [node["name"] for node in data["tree"]]
        assert "src" in dir_names
        assert "tests" in dir_names

    def test_sorts_directories_before_files(self, client, mock_repo):
        """Test directories are sorted before files."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {"_source": {"relative_path": "aaa.py", "language": "python"}},
                    {"_source": {"relative_path": "src/main.py", "language": "python"}},
                    {"_source": {"relative_path": "zzz.py", "language": "python"}},
                ]
            }
        }

        response = client.get("/repos/github/project/tree")

        assert response.status_code == 200
        data = response.json()
        # First should be directory (src), then files (alphabetically)
        assert data["tree"][0]["type"] == "directory"
        assert data["tree"][0]["name"] == "src"
        assert data["tree"][1]["type"] == "file"
        assert data["tree"][1]["name"] == "aaa.py"
        assert data["tree"][2]["type"] == "file"
        assert data["tree"][2]["name"] == "zzz.py"

    def test_accepts_username_parameter(self, client, mock_repo):
        """Test accepts username query parameter."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        response = client.get("/repos/github/project/tree?username=developer")

        assert response.status_code == 200
        # Verify query was made with username filter
        call_args = mock_client.search.call_args
        body = call_args[1]["body"]
        filters = body["query"]["bool"]["filter"]
        usernames = [f["term"]["username"] for f in filters if "username" in f.get("term", {})]
        assert "developer" in usernames


# =============================================================================
# GET FILE DETAIL TESTS
# =============================================================================


class TestGetFileDetail:
    """Tests for GET /repos/{scope}/{repository}/files/{file_path}."""

    def test_returns_file_detail(self, client, mock_repo):
        """Test returns file detail successfully."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client

        call_count = [0]

        def search_side_effect(**kwargs):
            call_count[0] += 1
            body = kwargs.get("body", {})

            if call_count[0] == 1:
                # File element query
                return {
                    "hits": {
                        "hits": [
                            {
                                "_source": {
                                    "relative_path": "src/main.py",
                                    "language": "python",
                                    "summary": "Main module",
                                    "line_end": 150,
                                }
                            }
                        ]
                    }
                }
            elif call_count[0] == 2:
                # Elements in file query
                return {
                    "hits": {
                        "hits": [
                            {
                                "_source": {
                                    "element_id": "id1",
                                    "name": "MyClass",
                                    "element_type": "class",
                                    "line_start": 10,
                                    "line_end": 50,
                                    "summary": "A class",
                                }
                            },
                            {
                                "_source": {
                                    "element_id": "id2",
                                    "name": "my_function",
                                    "element_type": "function",
                                    "line_start": 55,
                                    "line_end": 70,
                                    "signature": "def my_function():",
                                }
                            },
                        ]
                    }
                }
            else:
                # Contributors query
                return {
                    "aggregations": {
                        "users": {
                            "buckets": [{"key": "dev1"}, {"key": "dev2"}]
                        }
                    }
                }

        mock_client.search.side_effect = search_side_effect

        response = client.get("/repos/github/project/files/src/main.py")

        assert response.status_code == 200
        data = response.json()
        assert data["file"]["path"] == "src/main.py"
        assert data["file"]["language"] == "python"
        assert data["file"]["summary"] == "Main module"
        assert data["file"]["line_count"] == 150

        assert len(data["structure"]) == 2
        assert data["structure"][0]["name"] == "MyClass"
        assert data["structure"][1]["name"] == "my_function"

        assert data["stats"]["classes"] == 1
        assert data["stats"]["functions"] == 1

        assert len(data["contributors"]) == 2
        assert data["contributors"][0]["username"] == "dev1"

    def test_returns_404_when_file_not_found(self, client, mock_repo):
        """Test returns 404 when file not found."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        response = client.get("/repos/github/project/files/nonexistent.py")

        assert response.status_code == 404
        assert "File not found" in response.json()["detail"]

    def test_counts_element_types(self, client, mock_repo):
        """Test correctly counts different element types."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client

        call_count = [0]

        def search_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "hits": {
                        "hits": [
                            {"_source": {"relative_path": "file.py", "language": "python"}}
                        ]
                    }
                }
            elif call_count[0] == 2:
                return {
                    "hits": {
                        "hits": [
                            {"_source": {"element_id": "1", "name": "C1", "element_type": "class", "line_start": 1}},
                            {"_source": {"element_id": "2", "name": "C2", "element_type": "class", "line_start": 10}},
                            {"_source": {"element_id": "3", "name": "f1", "element_type": "function", "line_start": 20}},
                            {"_source": {"element_id": "4", "name": "m1", "element_type": "method", "line_start": 30}},
                            {"_source": {"element_id": "5", "name": "m2", "element_type": "method", "line_start": 40}},
                            {"_source": {"element_id": "6", "name": "m3", "element_type": "method", "line_start": 50}},
                            {"_source": {"element_id": "7", "name": "v1", "element_type": "variable", "line_start": 60}},
                        ]
                    }
                }
            else:
                return {"aggregations": {"users": {"buckets": []}}}

        mock_client.search.side_effect = search_side_effect

        response = client.get("/repos/github/project/files/file.py")

        assert response.status_code == 200
        data = response.json()
        assert data["stats"]["classes"] == 2
        assert data["stats"]["functions"] == 1
        assert data["stats"]["methods"] == 3
        assert data["stats"]["variables"] == 1

    def test_handles_file_with_no_elements(self, client, mock_repo):
        """Test handles file with no code elements."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client

        call_count = [0]

        def search_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "hits": {
                        "hits": [
                            {"_source": {"relative_path": "empty.py", "language": "python"}}
                        ]
                    }
                }
            elif call_count[0] == 2:
                return {"hits": {"hits": []}}
            else:
                return {"aggregations": {"users": {"buckets": []}}}

        mock_client.search.side_effect = search_side_effect

        response = client.get("/repos/github/project/files/empty.py")

        assert response.status_code == 200
        data = response.json()
        assert data["structure"] == []
        assert data["stats"]["classes"] == 0
        assert data["stats"]["functions"] == 0

    def test_accepts_username_parameter(self, client, mock_repo):
        """Test accepts username query parameter."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        response = client.get("/repos/github/project/files/main.py?username=dev")

        assert response.status_code == 404  # Not found is expected
        # Verify the query included the username filter
        call_args = mock_client.search.call_args
        body = call_args[1]["body"]
        filters = body["query"]["bool"]["filter"]
        usernames = [f["term"]["username"] for f in filters if "username" in f.get("term", {})]
        assert "dev" in usernames

    def test_handles_missing_optional_fields(self, client, mock_repo):
        """Test handles missing optional fields."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client

        call_count = [0]

        def search_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "hits": {
                        "hits": [
                            {"_source": {"relative_path": "file.py"}}  # Missing language, summary, line_end
                        ]
                    }
                }
            elif call_count[0] == 2:
                return {"hits": {"hits": []}}
            else:
                return {"aggregations": {"users": {"buckets": []}}}

        mock_client.search.side_effect = search_side_effect

        response = client.get("/repos/github/project/files/file.py")

        assert response.status_code == 200
        data = response.json()
        assert data["file"]["language"] == "unknown"
        assert data["file"]["summary"] is None
        assert data["file"]["line_count"] == 0

    def test_handles_nested_file_path(self, client, mock_repo):
        """Test handles deeply nested file paths."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client

        call_count = [0]

        def search_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "hits": {
                        "hits": [
                            {"_source": {"relative_path": "src/core/utils/helpers.py", "language": "python"}}
                        ]
                    }
                }
            else:
                return {"hits": {"hits": []}, "aggregations": {"users": {"buckets": []}}}

        mock_client.search.side_effect = search_side_effect

        response = client.get("/repos/github/project/files/src/core/utils/helpers.py")

        assert response.status_code == 200
        data = response.json()
        assert data["file"]["path"] == "src/core/utils/helpers.py"
