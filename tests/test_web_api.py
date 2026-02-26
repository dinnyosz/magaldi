"""Tests for Magaldi Web API endpoints.

These tests use FastAPI's TestClient to test the API endpoints in isolation
with mocked dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from magaldi_web.app import create_app
from magaldi_web.dependencies import get_repository
from shared.config import MagaldiConfig

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_client():
    """Create a mock Elasticsearch client."""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_repo(mock_client):
    """Create a mock Elasticsearch repository."""
    mock = MagicMock()
    mock._get_client.return_value = mock_client
    return mock


@pytest.fixture
def app(test_config: MagaldiConfig, mock_repo: MagicMock):
    """Create a test FastAPI application with mocked dependencies."""
    app = create_app(test_config)

    # Override the ES repository dependency
    app.dependency_overrides[get_repository] = lambda: mock_repo

    return app


@pytest.fixture
def client(app) -> TestClient:
    """Create a test client."""
    return TestClient(app)


# =============================================================================
# DASHBOARD TESTS
# =============================================================================


class TestDashboardEndpoint:
    """Tests for the /api/v1/dashboard endpoint."""

    @pytest.mark.usefixtures("mock_repo")
    def test_dashboard_returns_stats(
        self, client: TestClient, mock_client: MagicMock
    ):
        """Test that dashboard returns repository statistics."""
        # Mock two search calls: repos aggregation and type aggregation
        mock_client.search.side_effect = [
            # First call: repos aggregation
            {
                "aggregations": {
                    "repos": {
                        "buckets": [
                            {
                                "key": {"scope": "test-scope", "repository": "test-repo"},
                                "doc_count": 50,
                                "file_count": {"doc_count": 10},
                                "class_count": {"doc_count": 20},
                                "function_count": {"doc_count": 30},
                                "method_count": {"doc_count": 40},
                                "variable_count": {"doc_count": 0},
                                "constant_count": {"doc_count": 0},
                                "feature_count": {"doc_count": 0},
                                "languages": {"buckets": [{"key": "python"}]},
                            }
                        ]
                    },
                    "total_repos": {"value": 1},
                },
            },
            # Second call: type aggregation
            {
                "aggregations": {
                    "by_type": {
                        "buckets": [
                            {"key": "file", "doc_count": 10},
                            {"key": "class", "doc_count": 20},
                            {"key": "function", "doc_count": 30},
                            {"key": "method", "doc_count": 40},
                        ]
                    },
                },
            },
        ]

        with patch("magaldi_web.routes.dashboard.check_search_health") as mock_search_health, \
             patch("magaldi_web.routes.dashboard.check_llm_health") as mock_llm_health, \
             patch("magaldi_web.routes.dashboard.check_redis_health") as mock_redis_health, \
             patch("magaldi_web.routes.dashboard.get_redis_queue_stats") as mock_queue_stats:

            mock_search_health.return_value = {"status": "healthy"}
            mock_llm_health.return_value = {"status": "healthy"}
            mock_redis_health.return_value = {"status": "healthy"}
            mock_queue_stats.return_value = {
                "summarization": {},
                "embedding": {},
                "total_pending": 0,
                "total_running": 0,
            }

            response = client.get("/api/v1/dashboard")

        assert response.status_code == 200
        data = response.json()
        assert "stats" in data
        assert "health" in data
        assert "recent_repos" in data
        assert "queue_status" in data
        assert data["stats"]["repository_count"] == 1
        assert data["stats"]["element_count"] == 100  # 10 + 20 + 30 + 40
        assert data["stats"]["file_count"] == 10
        assert data["stats"]["class_count"] == 20
        assert data["stats"]["function_count"] == 30
        assert data["stats"]["method_count"] == 40
        assert data["queue_status"]["total_pending"] == 0

    @pytest.mark.usefixtures("mock_repo")
    def test_dashboard_empty_index(
        self, client: TestClient, mock_client: MagicMock
    ):
        """Test dashboard with empty Elasticsearch index."""
        mock_client.search.side_effect = [
            # First call: repos aggregation
            {
                "aggregations": {
                    "repos": {"buckets": []},
                    "total_repos": {"value": 0},
                },
            },
            # Second call: type aggregation
            {
                "aggregations": {
                    "by_type": {"buckets": []},
                },
            },
        ]

        with patch("magaldi_web.routes.dashboard.check_search_health") as mock_search_health, \
             patch("magaldi_web.routes.dashboard.check_llm_health") as mock_llm_health, \
             patch("magaldi_web.routes.dashboard.check_redis_health") as mock_redis_health, \
             patch("magaldi_web.routes.dashboard.get_redis_queue_stats") as mock_queue_stats:

            mock_search_health.return_value = {"status": "healthy"}
            mock_llm_health.return_value = {"status": "healthy"}
            mock_redis_health.return_value = {"status": "healthy"}
            mock_queue_stats.return_value = {
                "summarization": {},
                "embedding": {},
                "total_pending": 0,
                "total_running": 0,
            }

            response = client.get("/api/v1/dashboard")

        assert response.status_code == 200
        data = response.json()
        assert data["stats"]["element_count"] == 0
        assert data["recent_repos"] == []


# =============================================================================
# SEARCH TESTS
# =============================================================================


class TestSearchEndpoint:
    """Tests for the /api/v1/search endpoint."""

    @pytest.mark.usefixtures("mock_repo")
    def test_search_returns_results(
        self, client: TestClient, mock_client: MagicMock
    ):
        """Test that search returns matching results."""
        mock_client.search.return_value = {
            "took": 10,
            "hits": {
                "total": {"value": 2},
                "max_score": 1.0,
                "hits": [
                    {
                        "_score": 0.95,
                        "_source": {
                            "element_id": "scope:repo:main:src/app.py:function:process:10",
                            "name": "process",
                            "element_type": "function",
                            "relative_path": "src/app.py",
                            "line_start": 10,
                            "language": "python",
                            "summary": "Processes input data",
                            "signature": "def process(data)",
                            "repository": "repo",
                            "scope": "scope",
                        },
                        "highlight": {},
                    },
                    {
                        "_score": 0.85,
                        "_source": {
                            "element_id": "scope:repo:main:src/utils.py:function:validate:20",
                            "name": "validate",
                            "element_type": "function",
                            "relative_path": "src/utils.py",
                            "line_start": 20,
                            "language": "python",
                            "summary": "Validates input",
                            "signature": "def validate(x)",
                            "repository": "repo",
                            "scope": "scope",
                        },
                        "highlight": {},
                    },
                ],
            },
        }

        # Mock the embedding client to avoid external calls
        with patch("shared.ai.embedding.CodeEmbeddingClient") as mock_embed:
            mock_embed.return_value.embed.side_effect = Exception("Skip embedding")

            response = client.post(
                "/api/v1/search",
                json={"query": "process data"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["name"] == "process"
        assert data["took_ms"] == 10

    @pytest.mark.usefixtures("mock_repo")
    def test_search_with_filters(
        self, client: TestClient, mock_client: MagicMock
    ):
        """Test search with scope, repository, and type filters."""
        mock_client.search.return_value = {
            "took": 5,
            "hits": {"total": {"value": 0}, "max_score": None, "hits": []},
        }

        with patch("shared.ai.embedding.CodeEmbeddingClient") as mock_embed:
            mock_embed.return_value.embed.side_effect = Exception("Skip embedding")

            response = client.post(
                "/api/v1/search",
                json={
                    "query": "authentication",
                    "scope": "my-scope",
                    "repository": "my-repo",
                    "element_types": ["class", "function"],
                    "limit": 10,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["results"] == []


# =============================================================================
# VECTOR MAP TESTS
# =============================================================================


class TestVectorMapEndpoints:
    """Tests for the /api/v1/repos/{scope}/{repo}/vector-map endpoint."""

    @pytest.mark.usefixtures("mock_repo")
    def test_get_vector_map(
        self, client: TestClient, mock_client: MagicMock
    ):
        """Test getting vector map coordinates."""
        # Create mock embeddings for 10 elements
        mock_hits = []
        for i in range(10):
            mock_hits.append({
                "_source": {
                    "element_id": f"scope:repo:main:src/file{i}.py:function:func{i}:{i*10}",
                    "name": f"func{i}",
                    "element_type": "function",
                    "relative_path": f"src/file{i}.py",
                    "line_start": i * 10,
                    "summary": f"Function {i}",
                    "embedding": [0.1 + (i * 0.01)] * 1024,
                }
            })

        mock_client.search.return_value = {
            "hits": {"hits": mock_hits}
        }

        response = client.get(
            "/api/v1/repos/scope/repo/vector-map",
            params={"element_types": ["function"], "limit": 100},
        )

        assert response.status_code == 200
        data = response.json()
        assert "points" in data
        assert "bounds" in data
        assert "algorithm" in data
        assert data["element_count"] == 10

        # Check points have coordinates
        if data["points"]:
            point = data["points"][0]
            assert "x" in point
            assert "y" in point
            assert "element_id" in point
            assert "name" in point

    @pytest.mark.usefixtures("mock_repo")
    def test_get_vector_map_empty(
        self, client: TestClient, mock_client: MagicMock
    ):
        """Test vector map with no elements."""
        mock_client.search.return_value = {
            "hits": {"hits": []}
        }

        response = client.get("/api/v1/repos/scope/repo/vector-map")

        assert response.status_code == 200
        data = response.json()
        assert data["points"] == []
        assert data["element_count"] == 0


# =============================================================================
# CLUSTERS TESTS
# =============================================================================


class TestClustersEndpoint:
    """Tests for the /api/v1/repos/{scope}/{repo}/clusters endpoint."""

    @pytest.mark.usefixtures("mock_repo")
    def test_get_clusters(
        self, client: TestClient, mock_client: MagicMock
    ):
        """Test getting HDBSCAN feature clusters."""
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "scope:repo:main:feature:auth:0",
                            "name": "Authentication",
                            "element_type": "feature",
                            "relative_path": "",
                            "summary": "Handles user authentication and authorization",
                            "embedding": [0.1] * 1024,
                        }
                    },
                    {
                        "_source": {
                            "element_id": "scope:repo:main:feature:data:0",
                            "name": "Data Processing",
                            "element_type": "feature",
                            "relative_path": "",
                            "summary": "Processes and transforms data",
                            "embedding": [0.2] * 1024,
                        }
                    },
                ]
            }
        }

        response = client.get("/api/v1/repos/scope/repo/clusters")

        assert response.status_code == 200
        data = response.json()
        assert "clusters" in data
        assert "total_elements" in data
        assert len(data["clusters"]) == 2

        cluster = data["clusters"][0]
        assert "cluster_id" in cluster
        assert "size" in cluster
        assert "representative" in cluster
        assert cluster["representative"]["name"] == "Authentication"

    @pytest.mark.usefixtures("mock_repo")
    def test_get_clusters_empty(
        self, client: TestClient, mock_client: MagicMock
    ):
        """Test clusters endpoint with no features."""
        mock_client.search.return_value = {
            "hits": {"hits": []}
        }

        response = client.get("/api/v1/repos/scope/repo/clusters")

        assert response.status_code == 200
        data = response.json()
        assert data["clusters"] == []
        assert data["total_elements"] == 0


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for API error handling."""

    def test_invalid_json_body(self, client: TestClient):
        """Test that invalid JSON returns appropriate error."""
        response = client.post(
            "/api/v1/search",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    def test_missing_required_field(self, client: TestClient):
        """Test that missing required fields return validation error."""
        response = client.post("/api/v1/search", json={})

        assert response.status_code == 422


# =============================================================================
# CORS TESTS
# =============================================================================


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self, client: TestClient):
        """Test that CORS headers are present in response."""
        response = client.options(
            "/api/v1/dashboard",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        # CORS preflight should succeed
        assert response.status_code in [200, 204, 405]


# =============================================================================
# BROWSE ENDPOINT TESTS
# =============================================================================


class TestBrowseEndpoints:
    """Tests for /api/v1/browse/* endpoints."""

    @pytest.mark.usefixtures("mock_repo")
    def test_browse_elements(
        self, client: TestClient, mock_client: MagicMock
    ):
        """Test browsing code elements."""
        mock_client.search.return_value = {
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {
                        "_source": {
                            "element_id": "scope:repo:main:file.py:function:test:1",
                            "name": "test",
                            "element_type": "function",
                            "relative_path": "file.py",
                            "line_start": 1,
                            "line_end": 10,
                            "language": "python",
                            "summary": "A test function.",
                            "repository": "repo",
                            "scope": "scope",
                        }
                    },
                    {
                        "_source": {
                            "element_id": "scope:repo:main:file.py:function:other:20",
                            "name": "other",
                            "element_type": "function",
                            "relative_path": "file.py",
                            "line_start": 20,
                            "line_end": 30,
                            "repository": "repo",
                            "scope": "scope",
                        }
                    },
                ],
            }
        }

        response = client.get("/api/v1/browse/elements?scope=scope&repository=repo")

        assert response.status_code == 200
        data = response.json()
        assert "elements" in data
        assert "total" in data
        assert data["total"] == 2

    @pytest.mark.usefixtures("mock_repo")
    def test_browse_elements_with_filters(
        self, client: TestClient, mock_client: MagicMock
    ):
        """Test browsing elements with type filter."""
        mock_client.search.return_value = {
            "hits": {"total": {"value": 0}, "hits": []}
        }

        response = client.get(
            "/api/v1/browse/elements",
            params={
                "element_type": "class",
                "language": "python",
                "page": 1,
                "limit": 10,
            },
        )

        assert response.status_code == 200


# =============================================================================
# REPOS ENDPOINT TESTS
# =============================================================================


class TestReposEndpoints:
    """Tests for /api/v1/repos/* endpoints."""

    @pytest.mark.usefixtures("mock_repo")
    def test_list_repositories(
        self, client: TestClient, mock_client: MagicMock
    ):
        """Test listing all repositories."""
        mock_client.search.return_value = {
            "aggregations": {
                "repos": {
                    "buckets": [
                        {
                            "key": {"scope": "magaldi", "repository": "magaldi"},
                            "doc_count": 100,
                            "languages": {"buckets": [{"key": "python"}]},
                        }
                    ]
                }
            }
        }

        response = client.get("/api/v1/repos")

        assert response.status_code == 200
        data = response.json()
        assert "repos" in data

    @pytest.mark.usefixtures("mock_repo")
    def test_get_repository(
        self, client: TestClient, mock_client: MagicMock
    ):
        """Test getting a specific repository."""
        mock_client.search.return_value = {
            "aggregations": {
                "by_type": {
                    "buckets": [
                        {"key": "file", "doc_count": 10},
                        {"key": "class", "doc_count": 5},
                        {"key": "function", "doc_count": 20},
                    ]
                },
                "by_language": {
                    "buckets": [{"key": "python", "doc_count": 35}]
                },
            },
            "hits": {"total": {"value": 35}},
        }

        response = client.get("/api/v1/repos/magaldi/magaldi")

        assert response.status_code == 200
        data = response.json()
        # Response includes scope, name, element_count, etc.
        assert "scope" in data
        assert "name" in data


# =============================================================================
# ADMIN ENDPOINT TESTS
# =============================================================================


class TestAdminEndpoints:
    """Tests for /api/v1/admin/* endpoints."""

    @pytest.mark.usefixtures("mock_repo")
    def test_health_check(
        self, client: TestClient
    ):
        """Test health check endpoint."""
        with patch("magaldi_web.routes.admin.check_search_health") as mock_search_health, \
             patch("magaldi_web.routes.admin.check_llm_health") as mock_llm_health, \
             patch("magaldi_web.routes.admin.check_redis_health") as mock_redis_health:

            mock_search_health.return_value = {"status": "healthy", "cluster": "test"}
            mock_llm_health.return_value = {"status": "healthy", "model": "test"}
            mock_redis_health.return_value = {"status": "healthy"}

            response = client.get("/api/v1/admin/health")

        assert response.status_code == 200
        data = response.json()
        assert "search" in data
        assert "llm" in data
        assert "redis" in data
