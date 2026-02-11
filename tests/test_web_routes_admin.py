"""Tests for the admin API routes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from magaldi_web.routes.admin import router, format_bytes


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def app():
    """Create a test FastAPI app."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def mock_repo():
    """Create a mock Elasticsearch repository."""
    repo = MagicMock()
    mock_client = MagicMock()
    repo._get_client.return_value = mock_client
    return repo


@pytest.fixture
def client(app, mock_repo):
    """Create a test client with mocked dependencies."""
    from magaldi_web.dependencies import get_repository

    def get_mock_repo():
        yield mock_repo

    app.dependency_overrides[get_repository] = get_mock_repo
    return TestClient(app)


# =============================================================================
# FORMAT BYTES TESTS
# =============================================================================


class TestFormatBytes:
    """Tests for format_bytes utility function."""

    def test_format_bytes_b(self):
        """Test formatting bytes."""
        assert format_bytes(500) == "500.0 B"

    def test_format_bytes_kb(self):
        """Test formatting kilobytes."""
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(1536) == "1.5 KB"

    def test_format_bytes_mb(self):
        """Test formatting megabytes."""
        assert format_bytes(1024 * 1024) == "1.0 MB"
        assert format_bytes(1024 * 1024 * 5) == "5.0 MB"

    def test_format_bytes_gb(self):
        """Test formatting gigabytes."""
        assert format_bytes(1024 * 1024 * 1024) == "1.0 GB"

    def test_format_bytes_tb(self):
        """Test formatting terabytes."""
        assert format_bytes(1024 * 1024 * 1024 * 1024) == "1.0 TB"

    def test_format_bytes_pb(self):
        """Test formatting petabytes."""
        assert format_bytes(1024 * 1024 * 1024 * 1024 * 1024) == "1.0 PB"


# =============================================================================
# HEALTH ENDPOINT TESTS
# =============================================================================


class TestGetHealth:
    """Tests for GET /admin/health endpoint."""

    def test_get_health_all_healthy(self, client, mock_repo):
        """Test health check when all services are healthy."""
        # Mock ES health
        mock_client = mock_repo._get_client.return_value
        mock_client.cluster.health.return_value = {
            "status": "green",
            "number_of_nodes": 3,
        }

        # Mock config and health checks
        mock_config = MagicMock()
        mock_config.llm = MagicMock()
        mock_config.llm.url = "http://localhost:11434"
        mock_config.redis = MagicMock()
        mock_config.redis.url = "redis://localhost:6379"

        with patch("magaldi_web.routes.admin.get_cached_config", return_value=mock_config), \
             patch("magaldi_web.routes.admin.check_search_health") as mock_search_health, \
             patch("magaldi_web.routes.admin.check_llm_health") as mock_llm_health, \
             patch("magaldi_web.routes.admin.check_redis_health") as mock_redis_health:

            mock_search_health.return_value = {
                "status": "healthy",
                "cluster_status": "green",
                "number_of_nodes": 3,
            }
            mock_llm_health.return_value = {
                "status": "healthy",
                "models_loaded": ["llama3", "nomic-embed"],
            }
            mock_redis_health.return_value = {
                "status": "healthy",
            }

            response = client.get("/admin/health")

        assert response.status_code == 200
        data = response.json()
        assert data["search"]["status"] == "healthy"
        assert data["llm"]["status"] == "healthy"
        assert data["redis"]["status"] == "healthy"
        assert "latency_ms" in data["search"]
        assert "latency_ms" in data["llm"]
        assert "latency_ms" in data["redis"]

    def test_get_health_includes_details(self, client, mock_repo):
        """Test that health check includes service details."""
        mock_config = MagicMock()

        with patch("magaldi_web.routes.admin.get_cached_config", return_value=mock_config), \
             patch("magaldi_web.routes.admin.check_search_health") as mock_search_health, \
             patch("magaldi_web.routes.admin.check_llm_health") as mock_llm_health, \
             patch("magaldi_web.routes.admin.check_redis_health") as mock_redis_health:

            mock_search_health.return_value = {
                "status": "healthy",
                "cluster_status": "yellow",
                "number_of_nodes": 1,
            }
            mock_llm_health.return_value = {
                "status": "healthy",
                "models_loaded": ["llama3"],
            }
            mock_redis_health.return_value = {"status": "healthy"}

            response = client.get("/admin/health")

        assert response.status_code == 200
        data = response.json()
        assert data["search"]["details"]["cluster_status"] == "yellow"
        assert data["search"]["details"]["nodes"] == 1
        assert "llama3" in data["llm"]["details"]["models"]


# =============================================================================
# JOB STATS ENDPOINT TESTS
# =============================================================================


class TestGetJobStats:
    """Tests for GET /admin/jobs endpoint."""

    def test_get_job_stats_returns_zeros(self, client, mock_repo):
        """Test that job stats returns zeros when Redis is empty."""
        mock_stats = {
            "summarization": {"pending": 0, "running": 0, "completed": 0, "failed": 0},
            "embedding": {"pending": 0, "running": 0, "completed": 0, "failed": 0},
        }
        with patch("magaldi_web.routes.admin.get_job_queue_totals", return_value=mock_stats):
            response = client.get("/admin/jobs")

        assert response.status_code == 200
        data = response.json()
        assert data["summarization"]["pending"] == 0
        assert data["summarization"]["running"] == 0
        assert data["summarization"]["completed"] == 0
        assert data["summarization"]["failed"] == 0
        assert data["embedding"]["pending"] == 0

    def test_get_job_stats_returns_counts(self, client, mock_repo):
        """Test that job stats returns actual counts from Redis."""
        mock_stats = {
            "summarization": {"pending": 5, "running": 2, "completed": 10, "failed": 1},
            "embedding": {"pending": 3, "running": 1, "completed": 8, "failed": 0},
        }
        with patch("magaldi_web.routes.admin.get_job_queue_totals", return_value=mock_stats):
            response = client.get("/admin/jobs")

        assert response.status_code == 200
        data = response.json()
        assert data["summarization"]["pending"] == 5
        assert data["summarization"]["running"] == 2
        assert data["summarization"]["completed"] == 10
        assert data["summarization"]["failed"] == 1
        assert data["embedding"]["pending"] == 3
        assert data["embedding"]["failed"] == 0


# =============================================================================
# INDEX STATS ENDPOINT TESTS
# =============================================================================


class TestGetIndexStats:
    """Tests for GET /admin/index-stats endpoint."""

    def test_get_index_stats_success(self, client, mock_repo):
        """Test getting index statistics."""
        from shared.db.store import INDEX_NAME

        mock_client = mock_repo._get_client.return_value
        mock_client.indices_stats.return_value = {
            "indices": {
                INDEX_NAME: {
                    "primaries": {
                        "docs": {"count": 1000},
                        "store": {"size_in_bytes": 1024 * 1024 * 50},  # 50MB
                    }
                }
            }
        }
        mock_client.count.return_value = {"count": 800}  # 80% with vectors

        response = client.get("/admin/index-stats")

        assert response.status_code == 200
        data = response.json()
        assert data["document_count"] == 1000
        assert data["with_vectors"] == 800
        assert data["vector_coverage_pct"] == 80.0
        assert "50.0 MB" in data["size_human"]

    def test_get_index_stats_handles_missing_index(self, client, mock_repo):
        """Test graceful handling when index doesn't exist."""
        mock_client = mock_repo._get_client.return_value
        mock_client.indices_stats.side_effect = Exception("Index not found")
        mock_client.count.side_effect = Exception("Index not found")

        response = client.get("/admin/index-stats")

        assert response.status_code == 200
        data = response.json()
        assert data["document_count"] == 0
        assert data["with_vectors"] == 0
        assert data["vector_coverage_pct"] == 0.0

    def test_get_index_stats_zero_docs(self, client, mock_repo):
        """Test index stats when there are no documents."""
        from shared.db.store import INDEX_NAME

        mock_client = mock_repo._get_client.return_value
        mock_client.indices_stats.return_value = {
            "indices": {
                INDEX_NAME: {
                    "primaries": {
                        "docs": {"count": 0},
                        "store": {"size_in_bytes": 0},
                    }
                }
            }
        }
        mock_client.count.return_value = {"count": 0}

        response = client.get("/admin/index-stats")

        assert response.status_code == 200
        data = response.json()
        assert data["document_count"] == 0
        assert data["vector_coverage_pct"] == 0.0  # No division by zero


# =============================================================================
# ADMIN OVERVIEW ENDPOINT TESTS
# =============================================================================


class TestGetAdminOverview:
    """Tests for GET /admin/overview endpoint."""

    def test_get_admin_overview_returns_all_data(self, client, mock_repo):
        """Test that admin overview returns all combined data."""
        from shared.db.store import INDEX_NAME

        mock_client = mock_repo._get_client.return_value
        mock_client.cluster.health.return_value = {"status": "green", "number_of_nodes": 3}
        mock_client.indices_stats.return_value = {
            "indices": {
                INDEX_NAME: {
                    "primaries": {
                        "docs": {"count": 500},
                        "store": {"size_in_bytes": 1024 * 1024},
                    }
                }
            }
        }
        mock_client.count.return_value = {"count": 400}

        mock_config = MagicMock()
        mock_job_stats = {
            "summarization": {"pending": 0, "running": 0, "completed": 0, "failed": 0},
            "embedding": {"pending": 0, "running": 0, "completed": 0, "failed": 0},
        }

        with patch("magaldi_web.routes.admin.get_cached_config", return_value=mock_config), \
             patch("magaldi_web.routes.admin.check_search_health") as mock_search_health, \
             patch("magaldi_web.routes.admin.check_llm_health") as mock_llm_health, \
             patch("magaldi_web.routes.admin.check_redis_health") as mock_redis_health, \
             patch("magaldi_web.routes.admin.get_job_queue_totals", return_value=mock_job_stats):

            mock_search_health.return_value = {"status": "healthy"}
            mock_llm_health.return_value = {"status": "healthy"}
            mock_redis_health.return_value = {"status": "healthy"}

            response = client.get("/admin/overview")

        assert response.status_code == 200
        data = response.json()
        assert "health" in data
        assert "jobs" in data
        assert "index_stats" in data
        assert "recent_activity" in data


# =============================================================================
# RETRY FAILED JOBS ENDPOINT TESTS
# =============================================================================


class TestRetryFailedJobs:
    """Tests for POST /admin/jobs/retry endpoint."""

    def test_retry_summarization_jobs(self, client):
        """Test retrying summarization jobs."""
        with patch("magaldi_web.routes.admin.retry_failed_jobs_in_redis", return_value=0):
            response = client.post("/admin/jobs/retry?job_type=summarization")

        assert response.status_code == 200
        data = response.json()
        assert data["jobs_reset"] == 0

    def test_retry_summarization_jobs_resets_failed(self, client):
        """Test that retry actually resets failed jobs."""
        with patch("magaldi_web.routes.admin.retry_failed_jobs_in_redis", return_value=5):
            response = client.post("/admin/jobs/retry?job_type=summarization")

        assert response.status_code == 200
        data = response.json()
        assert data["jobs_reset"] == 5

    def test_retry_embedding_jobs(self, client):
        """Test retrying embedding jobs."""
        with patch("magaldi_web.routes.admin.retry_failed_jobs_in_redis", return_value=0):
            response = client.post("/admin/jobs/retry?job_type=embedding")

        assert response.status_code == 200
        data = response.json()
        assert data["jobs_reset"] == 0

    def test_retry_invalid_job_type(self, client):
        """Test retrying with invalid job type."""
        response = client.post("/admin/jobs/retry?job_type=invalid")

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["jobs_reset"] == 0


# =============================================================================
# REFRESH INDEX ENDPOINT TESTS
# =============================================================================


class TestRefreshIndex:
    """Tests for POST /admin/index/refresh endpoint."""

    def test_refresh_index_success(self, client, mock_repo):
        """Test refreshing the index."""
        mock_client = mock_repo._get_client.return_value
        mock_client.indices_refresh.return_value = None

        response = client.post("/admin/index/refresh")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "refreshed"
        mock_client.indices_refresh.assert_called_once()
