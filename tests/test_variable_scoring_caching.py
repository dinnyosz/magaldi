"""Tests for variable score caching in the repository layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from magaldi_core.variable_scoring.models import ScoringResult, VariableScore


class TestGetVariableScoringState:
    """Tests for ElementRepository.get_variable_scoring_state()."""

    def _make_repo(self, mget_response: dict) -> MagicMock:
        """Create a mock repo with mget returning the given response."""
        repo = MagicMock()
        repo._elements = MagicMock()

        # Import the real method and bind it
        from shared.db.repositories.elements import ElementRepository

        # We'll just test the method directly by calling the real implementation
        # with a mock client
        return repo

    def test_empty_ids_returns_empty(self):
        from shared.db.repositories.elements import ElementRepository

        elem_repo = MagicMock(spec=ElementRepository)
        elem_repo.get_variable_scoring_state = ElementRepository.get_variable_scoring_state.__get__(
            elem_repo, ElementRepository
        )

        result = elem_repo.get_variable_scoring_state([])
        assert result == {}

    def test_returns_cached_scores(self):
        """Elements with variable_score should be returned."""
        from shared.db.repositories.elements import ElementRepository

        elem_repo = MagicMock(spec=ElementRepository)
        client = MagicMock()
        elem_repo._get_client.return_value = client

        client.mget.return_value = {
            "docs": [
                {
                    "_id": "scope:repo:main:file.py:variable:MY_CONST:10",
                    "found": True,
                    "_source": {
                        "content_hash": "abc123",
                        "variable_score": {
                            "config_value": 9,
                            "architectural_role": 1,
                            "data_definition": 1,
                            "general_usefulness": 8,
                        },
                        "score_model": "qwen3:4b",
                    },
                },
            ],
        }

        # Call the real method
        elem_repo.get_variable_scoring_state = ElementRepository.get_variable_scoring_state.__get__(
            elem_repo, ElementRepository
        )
        result = elem_repo.get_variable_scoring_state(
            ["scope:repo:main:file.py:variable:MY_CONST:10"],
        )

        assert "scope:repo:main:file.py:variable:MY_CONST:10" in result
        state = result["scope:repo:main:file.py:variable:MY_CONST:10"]
        assert state["content_hash"] == "abc123"
        assert state["variable_score"]["config_value"] == 9
        assert state["score_model"] == "qwen3:4b"

    def test_skips_elements_without_score(self):
        """Elements without variable_score should not be returned."""
        from shared.db.repositories.elements import ElementRepository

        elem_repo = MagicMock(spec=ElementRepository)
        client = MagicMock()
        elem_repo._get_client.return_value = client

        client.mget.return_value = {
            "docs": [
                {
                    "_id": "id1",
                    "found": True,
                    "_source": {
                        "content_hash": "abc",
                        # No variable_score
                    },
                },
                {
                    "_id": "id2",
                    "found": False,
                },
            ],
        }

        elem_repo.get_variable_scoring_state = ElementRepository.get_variable_scoring_state.__get__(
            elem_repo, ElementRepository
        )
        result = elem_repo.get_variable_scoring_state(["id1", "id2"])

        assert result == {}  # Neither has a score


class TestIndexElementCompleteWithScore:
    """Test that variable_score and score_model are stored during indexing."""

    def test_score_stored_in_document(self):
        """variable_score and score_model should appear in the indexed document."""
        from magaldi_core.parsers.base import CodeElement

        element = CodeElement(
            element_id="scope:repo:main:file.py:variable:MY_CONST:10",
            scope="scope",
            repository="repo",
            username="main",
            relative_path="file.py",
            element_type="variable",
            name="MY_CONST",
            language="python",
            line_start=10,
            line_end=10,
            raw_code="MY_CONST = 42",
            variable_score={
                "config_value": 9,
                "architectural_role": 1,
                "data_definition": 1,
                "general_usefulness": 8,
            },
        )

        from shared.db.repositories.elements import ElementRepository

        elem_repo = MagicMock(spec=ElementRepository)
        elem_repo._bulk_buffer = None
        client = MagicMock()
        elem_repo._get_client.return_value = client

        # Call the real method
        elem_repo.index_element_complete = ElementRepository.index_element_complete.__get__(
            elem_repo, ElementRepository
        )
        elem_repo.index_element_complete(
            element, "Summary of MY_CONST",
            score_model="qwen3:4b",
        )

        # The document passed to index_document should contain variable_score
        call_args = client.index_document.call_args
        doc = call_args[0][2]  # (index_name, doc_id, doc)
        assert doc["variable_score"] == {
            "config_value": 9,
            "architectural_role": 1,
            "data_definition": 1,
            "general_usefulness": 8,
        }
        assert doc["score_model"] == "qwen3:4b"

    def test_no_score_no_field(self):
        """Elements without variable_score should not have the field in doc."""
        from magaldi_core.parsers.base import CodeElement

        element = CodeElement(
            element_id="scope:repo:main:file.py:function:main:5",
            scope="scope",
            repository="repo",
            username="main",
            relative_path="file.py",
            element_type="function",
            name="main",
            language="python",
            line_start=5,
            line_end=20,
            raw_code="def main(): pass",
        )

        from shared.db.repositories.elements import ElementRepository

        elem_repo = MagicMock(spec=ElementRepository)
        elem_repo._bulk_buffer = None
        client = MagicMock()
        elem_repo._get_client.return_value = client

        elem_repo.index_element_complete = ElementRepository.index_element_complete.__get__(
            elem_repo, ElementRepository
        )
        elem_repo.index_element_complete(element, "Main function")

        doc = client.index_document.call_args[0][2]
        assert "variable_score" not in doc
        assert "score_model" not in doc


class TestScoringResultCacheFields:
    """Test that ScoringResult properly tracks cache statistics."""

    def test_default_cache_fields(self):
        result = ScoringResult()
        assert result.heuristic_dropped == 0
        assert result.cached_scores == 0
        assert result.llm_scored == 0

    def test_cache_fields_set(self):
        result = ScoringResult(
            total_variables=100,
            heuristic_dropped=20,
            cached_scores=60,
            llm_scored=20,
            kept=50,
            dropped=50,
        )
        assert result.heuristic_dropped == 20
        assert result.cached_scores == 60
        assert result.llm_scored == 20


class TestApplyScoresToElements:
    """Test the _apply_scores_to_elements helper."""

    def test_removes_below_threshold(self):
        from shared.cli._runners import _apply_scores_to_elements
        from magaldi_core.parsers.base import CodeElement

        # Create mock parsing result
        class MockPF:
            def __init__(self, elements):
                self.elements = elements

        elem_keep = CodeElement(
            element_id="id1", scope="s", repository="r", username="u",
            relative_path="f.py", element_type="variable", name="KEEP",
            language="python", line_start=1, line_end=1, raw_code="KEEP = 1",
        )
        elem_drop = CodeElement(
            element_id="id2", scope="s", repository="r", username="u",
            relative_path="f.py", element_type="variable", name="drop",
            language="python", line_start=2, line_end=2, raw_code="drop = x",
        )

        class MockResult:
            parsed_files = [MockPF([elem_keep, elem_drop])]

        scores = {
            "id1": VariableScore(config_value=9, general_usefulness=8),
            "id2": VariableScore(config_value=1, general_usefulness=1),
        }

        _apply_scores_to_elements(MockResult(), scores, threshold=5)

        # elem_drop should be removed
        assert len(MockResult.parsed_files[0].elements) == 1

    def test_attaches_score_dict_to_kept_elements(self):
        from shared.cli._runners import _apply_scores_to_elements
        from magaldi_core.parsers.base import CodeElement

        class MockPF:
            def __init__(self, elements):
                self.elements = elements

        elem = CodeElement(
            element_id="id1", scope="s", repository="r", username="u",
            relative_path="f.py", element_type="variable", name="MY_VAR",
            language="python", line_start=1, line_end=1, raw_code="MY_VAR = 42",
        )

        class MockResult:
            parsed_files = [MockPF([elem])]

        scores = {
            "id1": VariableScore(config_value=9, architectural_role=1,
                                 data_definition=1, general_usefulness=8),
        }

        _apply_scores_to_elements(MockResult(), scores, threshold=5)

        assert elem.variable_score is not None
        assert elem.variable_score["config_value"] == 9
        assert elem.variable_score["general_usefulness"] == 8
