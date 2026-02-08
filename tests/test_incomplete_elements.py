"""Tests for incomplete element detection and re-processing.

Verifies that files with elements missing required embeddings (e.g. caller_embedding)
are detected during change detection and queued for re-processing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from magaldi_core.change_detection import DBFileState


class TestFindFilesWithIncompleteElements:
    """Tests for MetadataRepository.find_files_with_incomplete_elements."""

    def test_finds_files_with_missing_caller_embedding(self) -> None:
        from shared.db.repositories.metadata import MetadataRepository

        repo = MetadataRepository.__new__(MetadataRepository)
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "aggregations": {
                "files": {
                    "buckets": [
                        {"key": "src/foo.py", "doc_count": 3},
                        {"key": "src/bar.py", "doc_count": 1},
                    ]
                }
            }
        }
        repo._base = MagicMock()
        repo._base._get_client.return_value = mock_client

        result = repo.find_files_with_incomplete_elements("s", "r", "main")

        assert result == ["src/foo.py", "src/bar.py"]
        # Verify the query filters correctly
        call_args = mock_client.search.call_args
        body = call_args.kwargs.get("body") or call_args[1].get("body")
        query = body["query"]["bool"]
        # Must have summary_embedding but not caller_embedding
        assert {"exists": {"field": "summary_embedding"}} in query["filter"]
        assert {"exists": {"field": "caller_embedding"}} in query["must_not"]

    def test_returns_empty_when_all_complete(self) -> None:
        from shared.db.repositories.metadata import MetadataRepository

        repo = MetadataRepository.__new__(MetadataRepository)
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "aggregations": {"files": {"buckets": []}}
        }
        repo._base = MagicMock()
        repo._base._get_client.return_value = mock_client

        result = repo.find_files_with_incomplete_elements("s", "r", "main")
        assert result == []


class TestGetFileStatesWithIncompleteElements:
    """Tests for ElasticsearchFileStateRepository.get_file_states incomplete element detection."""

    def test_incomplete_file_gets_null_hash(self) -> None:
        """Files with incomplete elements should have file_hash=None."""
        from shared.db.elasticsearch import ElasticsearchFileStateRepository

        repo = ElasticsearchFileStateRepository.__new__(ElasticsearchFileStateRepository)
        repo._es = MagicMock()
        repo._config = MagicMock()

        # Simulate ES returning file states
        repo._es.get_file_states.return_value = {
            "src/foo.py": {
                "file_hash": "abc123",
                "element_count": 5,
            },
            "src/bar.py": {
                "file_hash": "def456",
                "element_count": 3,
            },
        }
        # src/foo.py has incomplete elements
        repo._es.find_files_with_incomplete_elements.return_value = ["src/foo.py"]
        # element counts match (no count mismatch)
        repo._es.count_elements_by_path.return_value = 5

        states = repo.get_file_states("s", "r", "main")

        # src/foo.py should have file_hash=None (incomplete elements)
        assert states["src/foo.py"].file_hash is None
        # src/bar.py should keep its hash (complete, but count check may apply)
        # We need to mock count for bar too
        repo._es.count_elements_by_path.side_effect = lambda *a: {
            ("s", "r", "main", "src/foo.py"): 5,
            ("s", "r", "main", "src/bar.py"): 3,
        }.get(a, 0)

        states = repo.get_file_states("s", "r", "main")
        assert states["src/foo.py"].file_hash is None
        assert states["src/bar.py"].file_hash == "def456"

    def test_complete_files_keep_hash(self) -> None:
        """Files with all embeddings complete should keep their file_hash."""
        from shared.db.elasticsearch import ElasticsearchFileStateRepository

        repo = ElasticsearchFileStateRepository.__new__(ElasticsearchFileStateRepository)
        repo._es = MagicMock()
        repo._config = MagicMock()

        repo._es.get_file_states.return_value = {
            "src/foo.py": {"file_hash": "abc123", "element_count": 3},
        }
        # No incomplete elements
        repo._es.find_files_with_incomplete_elements.return_value = []
        repo._es.count_elements_by_path.return_value = 3

        states = repo.get_file_states("s", "r", "main")
        assert states["src/foo.py"].file_hash == "abc123"

    def test_no_file_states_returns_empty(self) -> None:
        """Empty repo should return empty states."""
        from shared.db.elasticsearch import ElasticsearchFileStateRepository

        repo = ElasticsearchFileStateRepository.__new__(ElasticsearchFileStateRepository)
        repo._es = MagicMock()
        repo._config = MagicMock()

        repo._es.get_file_states.return_value = {}
        repo._es.find_files_with_incomplete_elements.return_value = []

        states = repo.get_file_states("s", "r", "main")
        assert states == {}


class TestIncompleteElementsIntegration:
    """Integration test: incomplete elements cause change detection to mark files as modified."""

    def test_incomplete_file_detected_as_modified(self) -> None:
        """An unchanged file with incomplete elements should be detected as modified."""
        from magaldi_core.change_detection import FileInfo, compare_main_branch

        # File on disk: src/foo.py with hash "abc123"
        files = [
            FileInfo(
                relative_path="src/foo.py",
                absolute_path="",
                language="python",
                hash="abc123",
            ),
        ]

        # DB state: same hash but None'd out due to incomplete elements
        db_states = {
            "src/foo.py": DBFileState(
                relative_path="src/foo.py",
                file_hash=None,  # Nullified because of incomplete elements
                element_count=5,
            ),
        }

        new_files, modified_files, deleted_files, unchanged_count = compare_main_branch(
            files, db_states
        )

        # File should be "modified" since db hash (None) != disk hash ("abc123")
        assert len(modified_files) == 1
        assert modified_files[0].relative_path == "src/foo.py"
        assert unchanged_count == 0

    def test_complete_file_stays_unchanged(self) -> None:
        """A file with all embeddings complete should remain unchanged."""
        from magaldi_core.change_detection import FileInfo, compare_main_branch

        files = [
            FileInfo(
                relative_path="src/foo.py",
                absolute_path="",
                language="python",
                hash="abc123",
            ),
        ]

        db_states = {
            "src/foo.py": DBFileState(
                relative_path="src/foo.py",
                file_hash="abc123",  # Matches disk hash
                element_count=5,
            ),
        }

        new_files, modified_files, deleted_files, unchanged_count = compare_main_branch(
            files, db_states
        )

        assert len(modified_files) == 0
        assert unchanged_count == 1
