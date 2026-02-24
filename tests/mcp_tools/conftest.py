"""Shared fixtures for MCP tools tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _patch_auto_detect():
    """Prevent auto-detection from reading real magaldi.yaml during tests.

    Returns (None, None) so _resolve_scope_repo falls through to whatever
    scope/repository the test passes explicitly.
    """
    with patch(
        "magaldi_mcp.tools._utils._auto_detect_repo_config",
        return_value=(None, None),
    ):
        yield


@pytest.fixture
def mock_repo():
    """Create a mock repository."""
    repo = MagicMock()
    repo.search_by_vector.return_value = []
    repo.get_document.return_value = None
    repo.search_by_text.return_value = []
    repo.get_indexed_repositories.return_value = []

    # Make get_document_by_id_or_hash delegate to get_document
    # This ensures tests that set get_document.return_value or side_effect
    # also work with get_document_by_id_or_hash
    repo.get_document_by_id_or_hash.side_effect = lambda x: repo.get_document(x)

    return repo


@pytest.fixture
def mock_embed_client():
    """Create a mock embedding client."""
    client = MagicMock()
    client.embed_single.return_value = [0.1] * 1024
    return client
