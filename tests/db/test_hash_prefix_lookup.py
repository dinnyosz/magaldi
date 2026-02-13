"""Unit tests for hash prefix lookup in ElementRepository."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from shared.db.repositories.elements import ElementRepository


@pytest.fixture
def element_repo():
    """Create an ElementRepository with a mocked base."""
    base = MagicMock()
    base.backend_type = "opensearch"
    return ElementRepository(base)


@pytest.fixture
def mock_client(element_repo):
    """Get the mock client from the element repo."""
    client = MagicMock()
    element_repo._base._get_client.return_value = client
    return client


class TestGetDocumentByHashId:
    """Tests for get_document_by_hash_id with prefix support."""

    def test_exact_hash_64_chars(self, element_repo, mock_client):
        """Full 64-char hash uses exact term query."""
        full_hash = "a" * 64
        doc = {"element_id": "test:id", "hash_id": full_hash}
        mock_client.search.return_value = {"hits": {"hits": [{"_source": doc}]}}

        result = element_repo.get_document_by_hash_id(full_hash)

        assert result == doc
        call_body = mock_client.search.call_args[1]["body"]
        assert call_body["query"] == {"term": {"hash_id": full_hash}}
        assert call_body["size"] == 1

    def test_prefix_8_chars_unique_match(self, element_repo, mock_client):
        """8-char prefix with unique match returns the document."""
        prefix = "b1d0e591"
        doc = {"element_id": "test:id", "hash_id": "b1d0e591" + "f" * 56}
        mock_client.search.return_value = {"hits": {"hits": [{"_source": doc}]}}

        result = element_repo.get_document_by_hash_id(prefix)

        assert result == doc
        call_body = mock_client.search.call_args[1]["body"]
        assert call_body["query"] == {"prefix": {"hash_id": prefix}}
        assert call_body["size"] == 2  # Fetches 2 to detect ambiguity

    def test_prefix_ambiguous_raises_error(self, element_repo, mock_client):
        """Short prefix matching multiple documents raises ValueError."""
        prefix = "b1d0"
        doc1 = {"element_id": "test:id1", "hash_id": "b1d0" + "a" * 60}
        doc2 = {"element_id": "test:id2", "hash_id": "b1d0" + "b" * 60}
        mock_client.search.return_value = {
            "hits": {"hits": [{"_source": doc1}, {"_source": doc2}]}
        }

        with pytest.raises(ValueError, match="Ambiguous hash prefix"):
            element_repo.get_document_by_hash_id(prefix)

    def test_prefix_not_found_returns_none(self, element_repo, mock_client):
        """Prefix with no matches returns None."""
        mock_client.search.return_value = {"hits": {"hits": []}}

        result = element_repo.get_document_by_hash_id("deadbeef")
        assert result is None

    def test_exception_returns_none(self, element_repo, mock_client):
        """Non-ValueError exceptions return None."""
        mock_client.search.side_effect = ConnectionError("connection refused")

        result = element_repo.get_document_by_hash_id("deadbeef")
        assert result is None

    def test_ambiguous_error_propagates(self, element_repo, mock_client):
        """ValueError from ambiguity is not swallowed."""
        prefix = "ab"
        doc1 = {"element_id": "id1"}
        doc2 = {"element_id": "id2"}
        mock_client.search.return_value = {
            "hits": {"hits": [{"_source": doc1}, {"_source": doc2}]}
        }

        with pytest.raises(ValueError):
            element_repo.get_document_by_hash_id(prefix)


class TestGetDocumentByIdOrHash:
    """Tests for get_document_by_id_or_hash detection logic."""

    def test_64_char_hex_detected_as_hash(self, element_repo, mock_client):
        """64-char all-hex string is detected as hash_id."""
        full_hash = "abcdef0123456789" * 4  # 64 chars
        doc = {"element_id": "test:id", "hash_id": full_hash}
        mock_client.search.return_value = {"hits": {"hits": [{"_source": doc}]}}

        result = element_repo.get_document_by_id_or_hash(full_hash)

        assert result == doc
        # Should have called search (hash path), not get (element_id path)
        mock_client.search.assert_called_once()

    def test_8_char_hex_detected_as_hash_prefix(self, element_repo, mock_client):
        """8-char all-hex string is detected as hash prefix."""
        prefix = "b1d0e591"
        doc = {"element_id": "test:id", "hash_id": prefix + "0" * 56}
        mock_client.search.return_value = {"hits": {"hits": [{"_source": doc}]}}

        result = element_repo.get_document_by_id_or_hash(prefix)

        assert result == doc

    def test_colon_detected_as_element_id(self, element_repo, mock_client):
        """String with colons is detected as element_id."""
        element_id = "scope:repo:main:file.py:function:test:42"
        doc = {"element_id": element_id}
        mock_client.get_document.return_value = {"_source": doc}

        result = element_repo.get_document_by_id_or_hash(element_id)

        assert result == doc

    def test_non_hex_chars_detected_as_element_id(self, element_repo, mock_client):
        """Non-hex characters fall through to element_id lookup."""
        non_hex = "ghijklmn"  # g-n are not hex chars
        doc = {"element_id": non_hex}
        mock_client.get_document.return_value = {"_source": doc}

        result = element_repo.get_document_by_id_or_hash(non_hex)

        assert result == doc

    def test_uppercase_hex_detected_as_hash(self, element_repo, mock_client):
        """Uppercase hex is still detected as hash."""
        prefix = "B1D0E591"
        doc = {"element_id": "test:id"}
        mock_client.search.return_value = {"hits": {"hits": [{"_source": doc}]}}

        result = element_repo.get_document_by_id_or_hash(prefix)

        assert result == doc
