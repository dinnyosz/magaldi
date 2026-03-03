"""Tests for index mapping definitions and migration logic."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from shared.db.repositories.base import INDEX_MAPPING, RepositoryBase


class TestRawCodeMapping:
    """Tests for raw_code field mapping."""

    def test_raw_code_is_text_type(self):
        """raw_code should be a text field."""
        raw_code = INDEX_MAPPING["mappings"]["properties"]["raw_code"]
        assert raw_code["type"] == "text"

    def test_raw_code_has_keyword_subfield(self):
        """raw_code should have a keyword sub-field for regexp/wildcard queries."""
        raw_code = INDEX_MAPPING["mappings"]["properties"]["raw_code"]
        assert "keyword" in raw_code["fields"]
        assert raw_code["fields"]["keyword"]["type"] == "keyword"

    def test_raw_code_keyword_ignore_above_under_lucene_limit(self):
        """ignore_above must be well under Lucene's 32766-byte term limit.

        The old value of 32766 caused indexing failures with multi-byte UTF-8
        content (e.g. files with non-ASCII chars where char count <= 32766 but
        UTF-8 byte length > 32766).
        """
        raw_code = INDEX_MAPPING["mappings"]["properties"]["raw_code"]
        ignore_above = raw_code["fields"]["keyword"]["ignore_above"]

        # Must be well under the Lucene limit to account for multi-byte UTF-8
        assert ignore_above <= 16000, (
            f"ignore_above={ignore_above} is too close to Lucene's 32766-byte limit"
        )
        # Must be large enough to cover most functions/methods
        assert ignore_above >= 4000, (
            f"ignore_above={ignore_above} is too low, would exclude many elements"
        )


class TestEnsureIndexMigration:
    """Tests for the _ensure_index mapping migration."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = MagicMock()
        config.search_backend.type = "opensearch"
        config.search_backend.host = "localhost"
        config.search_backend.port = 9200
        config.search_backend.scheme = "http"
        return config

    def test_updates_ignore_above_on_existing_index(self, mock_config):
        """When the index already exists, _ensure_index should update ignore_above."""
        mock_client = MagicMock()
        mock_client.index_exists.return_value = True  # Index exists

        repo = RepositoryBase(mock_config)
        repo._client = mock_client

        repo._ensure_index()

        # Should call put_mapping to update raw_code.keyword ignore_above
        put_mapping_calls = [
            c for c in mock_client.indices_put_mapping.call_args_list
        ]
        assert len(put_mapping_calls) >= 1

        # Verify the mapping body contains the updated ignore_above
        mapping_body = put_mapping_calls[0][0][1]  # Second positional arg
        keyword_mapping = mapping_body["properties"]["raw_code"]["fields"]["keyword"]
        assert keyword_mapping["ignore_above"] == 8191

    def test_put_mapping_failure_does_not_raise(self, mock_config):
        """If put_mapping fails, it should log and continue, not crash."""
        mock_client = MagicMock()
        mock_client.index_exists.return_value = True
        mock_client.indices_put_mapping.side_effect = Exception("mapping update failed")

        repo = RepositoryBase(mock_config)
        repo._client = mock_client

        # Should not raise
        repo._ensure_index()

    def test_new_index_uses_correct_ignore_above(self, mock_config):
        """New indices should use the updated ignore_above value."""
        mock_client = MagicMock()
        mock_client.index_exists.return_value = False  # Index doesn't exist

        repo = RepositoryBase(mock_config)
        repo._client = mock_client

        repo._ensure_index()

        # Should call index_create
        create_calls = mock_client.index_create.call_args_list
        assert len(create_calls) >= 1

        # Verify the mapping has the correct ignore_above
        mapping = create_calls[0][0][1]  # Second positional arg (body)
        raw_code = mapping["mappings"]["properties"]["raw_code"]
        assert raw_code["fields"]["keyword"]["ignore_above"] == 8191
