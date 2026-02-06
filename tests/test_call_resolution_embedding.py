"""Tests for embedding-based call resolution and semantic relationships.

Tests Strategy 6 (embedding resolution), semantic relationship computation,
and MCP tool integration.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from magaldi_core.call_resolution import (
    _cosine_similarity,
    _merge_candidates,
    compute_semantic_relationships,
    resolve_calls_by_embedding,
)
from magaldi_mcp.formatters.analysis import CallGraphFormatter
from magaldi_mcp.tools.call_graph import get_call_graph


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_es():
    """Create a mock Elasticsearch repository for call resolution tests."""
    es = MagicMock()
    es.find_all_elements_with_calls.return_value = []
    es.find_candidates_by_name.return_value = []
    es.get_embedding.return_value = None
    es.store_calls.return_value = True
    es.store_semantic_related.return_value = True
    es.search_by_vector.return_value = []
    es.get_document.return_value = None
    es.get_document_by_id_or_hash.side_effect = lambda x: es.get_document(x)
    es.get_calls.return_value = []
    es.find_elements_calling.return_value = []
    es._get_client.return_value = MagicMock()
    return es


# =============================================================================
# COSINE SIMILARITY TESTS
# =============================================================================


class TestCosineSimilarity:
    """Tests for _cosine_similarity helper."""

    def test_identical_vectors(self):
        """Identical normalized vectors → similarity 1.0."""
        v = [0.5, 0.5, 0.5, 0.5]
        assert abs(_cosine_similarity(v, v) - 1.0) < 0.001

    def test_orthogonal_vectors(self):
        """Orthogonal vectors → similarity 0.0."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(_cosine_similarity(a, b)) < 0.001

    def test_opposite_vectors(self):
        """Opposite vectors → similarity -1.0."""
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 0.001

    def test_partial_similarity(self):
        """Partially similar vectors → value between 0 and 1."""
        a = [0.8, 0.6]
        b = [0.6, 0.8]
        result = _cosine_similarity(a, b)
        assert 0.9 < result < 1.0  # High but not perfect similarity

    def test_empty_vectors(self):
        """Empty vectors → 0.0."""
        assert _cosine_similarity([], []) == 0.0


# =============================================================================
# MERGE CANDIDATES TESTS
# =============================================================================


class TestMergeCandidates:
    """Tests for _merge_candidates helper."""

    def test_no_overlap(self):
        """Disjoint candidates are all preserved."""
        user = [{"relative_path": "a.py", "name": "foo", "element_type": "function", "element_id": "u1"}]
        main = [{"relative_path": "b.py", "name": "bar", "element_type": "function", "element_id": "m1"}]
        merged = _merge_candidates(user, main)
        assert len(merged) == 2

    def test_user_wins_on_duplicate(self):
        """User version takes priority when same element exists in both."""
        user = [{"relative_path": "a.py", "name": "foo", "element_type": "function", "element_id": "u1"}]
        main = [{"relative_path": "a.py", "name": "foo", "element_type": "function", "element_id": "m1"}]
        merged = _merge_candidates(user, main)
        assert len(merged) == 1
        assert merged[0]["element_id"] == "u1"

    def test_empty_inputs(self):
        """Empty lists return empty."""
        assert _merge_candidates([], []) == []

    def test_only_main(self):
        """When user has no candidates, main candidates are returned."""
        main = [{"relative_path": "a.py", "name": "foo", "element_type": "function", "element_id": "m1"}]
        merged = _merge_candidates([], main)
        assert len(merged) == 1
        assert merged[0]["element_id"] == "m1"


# =============================================================================
# EMBEDDING-BASED CALL RESOLUTION TESTS
# =============================================================================


class TestResolveCallsByEmbedding:
    """Tests for resolve_calls_by_embedding (Strategy 6)."""

    def test_single_candidate_resolves_directly(self, mock_es):
        """When only one candidate exists for a name, resolve without embedding."""
        mock_es.find_all_elements_with_calls.return_value = [
            {
                "element_id": "caller1",
                "calls": [
                    {"name": "save", "receiver": "repo", "category": "untyped", "resolved_id": None},
                ],
            }
        ]
        mock_es.find_candidates_by_name.return_value = [
            {"element_id": "target1", "name": "save", "element_type": "method"},
        ]

        total, single, embedding = resolve_calls_by_embedding(mock_es, "s", "r", "main")

        assert total == 1
        assert single == 1
        assert embedding == 0
        mock_es.store_calls.assert_called_once()
        stored_calls = mock_es.store_calls.call_args[0][1]
        assert stored_calls[0]["resolved_id"] == "target1"
        assert stored_calls[0]["category"] == "embedding_resolved"

    def test_multiple_candidates_best_embedding_wins(self, mock_es):
        """With multiple candidates, highest similarity wins."""
        caller_embedding = [0.8, 0.6, 0.0]

        mock_es.find_all_elements_with_calls.return_value = [
            {
                "element_id": "caller1",
                "calls": [
                    {"name": "process", "receiver": "svc", "category": "untyped", "resolved_id": None},
                ],
            }
        ]
        mock_es.find_candidates_by_name.return_value = [
            {"element_id": "bad_match", "name": "process", "element_type": "method", "summary_embedding": [0.0, 0.0, 1.0]},
            {"element_id": "good_match", "name": "process", "element_type": "method", "summary_embedding": [0.7, 0.7, 0.0]},
        ]
        mock_es.get_embedding.return_value = caller_embedding

        total, single, embedding = resolve_calls_by_embedding(mock_es, "s", "r", "main")

        assert total == 1
        assert single == 0
        assert embedding == 1
        stored_calls = mock_es.store_calls.call_args[0][1]
        assert stored_calls[0]["resolved_id"] == "good_match"

    def test_below_threshold_stays_unresolved(self, mock_es):
        """Candidates below similarity threshold are not resolved."""
        mock_es.find_all_elements_with_calls.return_value = [
            {
                "element_id": "caller1",
                "calls": [
                    {"name": "run", "receiver": "obj", "category": "untyped", "resolved_id": None},
                ],
            }
        ]
        mock_es.find_candidates_by_name.return_value = [
            {"element_id": "c1", "name": "run", "element_type": "method", "summary_embedding": [0.0, 1.0, 0.0]},
            {"element_id": "c2", "name": "run", "element_type": "method", "summary_embedding": [0.0, 0.0, 1.0]},
        ]
        # Caller embedding is very different from both candidates
        mock_es.get_embedding.return_value = [1.0, 0.0, 0.0]

        total, single, embedding = resolve_calls_by_embedding(mock_es, "s", "r", "main", similarity_threshold=0.9)

        assert total == 1
        assert single == 0
        assert embedding == 0
        mock_es.store_calls.assert_not_called()

    def test_bare_call_skipped(self, mock_es):
        """Calls without receiver are skipped (too ambiguous)."""
        mock_es.find_all_elements_with_calls.return_value = [
            {
                "element_id": "caller1",
                "calls": [
                    {"name": "run", "receiver": None, "category": "untyped", "resolved_id": None},
                ],
            }
        ]

        total, single, embedding = resolve_calls_by_embedding(mock_es, "s", "r", "main")

        assert total == 0
        assert single == 0
        assert embedding == 0

    def test_already_resolved_untouched(self, mock_es):
        """Calls that already have resolved_id are skipped."""
        mock_es.find_all_elements_with_calls.return_value = [
            {
                "element_id": "caller1",
                "calls": [
                    {"name": "save", "receiver": "repo", "category": "import", "resolved_id": "existing_target"},
                ],
            }
        ]

        total, single, embedding = resolve_calls_by_embedding(mock_es, "s", "r", "main")

        assert total == 0
        mock_es.store_calls.assert_not_called()

    def test_caller_without_embedding_skips_comparison(self, mock_es):
        """When caller has no embedding, multi-candidate matching is skipped."""
        mock_es.find_all_elements_with_calls.return_value = [
            {
                "element_id": "caller1",
                "calls": [
                    {"name": "process", "receiver": "svc", "category": "untyped", "resolved_id": None},
                ],
            }
        ]
        mock_es.find_candidates_by_name.return_value = [
            {"element_id": "c1", "name": "process", "element_type": "method", "summary_embedding": [1.0]},
            {"element_id": "c2", "name": "process", "element_type": "method", "summary_embedding": [0.0]},
        ]
        mock_es.get_embedding.return_value = None  # No embedding for caller

        total, single, embedding = resolve_calls_by_embedding(mock_es, "s", "r", "main")

        assert total == 1
        assert embedding == 0
        mock_es.store_calls.assert_not_called()

    def test_no_candidates_found(self, mock_es):
        """When no candidates exist for a name, call stays unresolved."""
        mock_es.find_all_elements_with_calls.return_value = [
            {
                "element_id": "caller1",
                "calls": [
                    {"name": "nonexistent", "receiver": "obj", "category": "untyped", "resolved_id": None},
                ],
            }
        ]
        mock_es.find_candidates_by_name.return_value = []

        total, single, embedding = resolve_calls_by_embedding(mock_es, "s", "r", "main")

        assert total == 1
        assert single == 0
        assert embedding == 0

    def test_category_set_to_embedding_resolved(self, mock_es):
        """Resolved calls get category set to 'embedding_resolved'."""
        mock_es.find_all_elements_with_calls.return_value = [
            {
                "element_id": "caller1",
                "calls": [
                    {"name": "save", "receiver": "repo", "category": "untyped", "resolved_id": None},
                ],
            }
        ]
        mock_es.find_candidates_by_name.return_value = [
            {"element_id": "t1", "name": "save", "element_type": "method"},
        ]

        resolve_calls_by_embedding(mock_es, "s", "r", "main")

        stored_calls = mock_es.store_calls.call_args[0][1]
        assert stored_calls[0]["category"] == "embedding_resolved"

    def test_multi_user_merges_candidates(self, mock_es):
        """When username != 'main', candidates from both user and main are merged."""
        mock_es.find_all_elements_with_calls.return_value = [
            {
                "element_id": "caller1",
                "calls": [
                    {"name": "save", "receiver": "repo", "category": "untyped", "resolved_id": None},
                ],
            }
        ]
        # First call returns user candidates, second returns main candidates
        mock_es.find_candidates_by_name.side_effect = [
            [{"element_id": "user_t", "name": "save", "element_type": "method",
              "relative_path": "a.py"}],
            [{"element_id": "main_t", "name": "save", "element_type": "method",
              "relative_path": "b.py"}],
        ]

        resolve_calls_by_embedding(mock_es, "s", "r", "dev_user")

        # Should call find_candidates_by_name twice (user + main)
        assert mock_es.find_candidates_by_name.call_count == 2


# =============================================================================
# SEMANTIC RELATIONSHIP TESTS
# =============================================================================


class TestComputeSemanticRelationships:
    """Tests for compute_semantic_relationships."""

    def test_processes_elements_and_stores_relationships(self, mock_es):
        """Functions get semantic_related field populated."""
        client = MagicMock()
        mock_es._get_client.return_value = client

        # Return one function with embedding
        client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "elem1",
                            "hash_id": "h1",
                            "username": "main",
                            "name": "process_data",
                            "element_type": "function",
                            "relative_path": "utils.py",
                            "summary_embedding": [0.5, 0.5],
                        }
                    }
                ]
            }
        }

        # search_by_vector returns similar elements
        mock_es.search_by_vector.return_value = [
            {"element_id": "elem2", "hash_id": "h2", "_score": 0.85},
            {"element_id": "elem1", "hash_id": "h1", "_score": 1.0},  # Self - should be excluded
        ]

        processed, relationships = compute_semantic_relationships(mock_es, "s", "r", "main")

        assert processed == 1
        assert relationships == 1
        mock_es.store_semantic_related.assert_called_once()
        stored = mock_es.store_semantic_related.call_args[0][1]
        assert len(stored) == 1
        assert stored[0]["element_id"] == "elem2"
        assert stored[0]["score"] == 0.85

    def test_self_excluded_from_results(self, mock_es):
        """Element's own ID is filtered out from semantic results."""
        client = MagicMock()
        mock_es._get_client.return_value = client

        client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "elem1",
                            "hash_id": "h1",
                            "username": "main",
                            "name": "foo",
                            "element_type": "function",
                            "relative_path": "a.py",
                            "summary_embedding": [1.0],
                        }
                    }
                ]
            }
        }

        # Only returns self
        mock_es.search_by_vector.return_value = [
            {"element_id": "elem1", "hash_id": "h1", "_score": 1.0},
        ]

        processed, relationships = compute_semantic_relationships(mock_es, "s", "r", "main")

        assert processed == 1
        assert relationships == 0
        mock_es.store_semantic_related.assert_not_called()

    def test_elements_without_embedding_skipped(self, mock_es):
        """Elements without summary_embedding are not processed."""
        client = MagicMock()
        mock_es._get_client.return_value = client

        client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "elem1",
                            "hash_id": "h1",
                            "username": "main",
                            "name": "foo",
                            "element_type": "function",
                            "relative_path": "a.py",
                            "summary_embedding": None,
                        }
                    }
                ]
            }
        }

        processed, relationships = compute_semantic_relationships(mock_es, "s", "r", "main")

        assert processed == 0
        assert relationships == 0

    def test_results_respect_top_k(self, mock_es):
        """At most top_k relationships are stored per element."""
        client = MagicMock()
        mock_es._get_client.return_value = client

        client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "elem1",
                            "hash_id": "h1",
                            "username": "main",
                            "name": "foo",
                            "element_type": "function",
                            "relative_path": "a.py",
                            "summary_embedding": [1.0],
                        }
                    }
                ]
            }
        }

        # Return more results than top_k=2
        mock_es.search_by_vector.return_value = [
            {"element_id": "e2", "hash_id": "h2", "_score": 0.9},
            {"element_id": "e3", "hash_id": "h3", "_score": 0.8},
            {"element_id": "e4", "hash_id": "h4", "_score": 0.7},
            {"element_id": "e5", "hash_id": "h5", "_score": 0.6},
        ]

        processed, relationships = compute_semantic_relationships(
            mock_es, "s", "r", "main", top_k=2,
        )

        assert relationships == 2
        stored = mock_es.store_semantic_related.call_args[0][1]
        assert len(stored) == 2

    def test_multi_user_deduplication(self, mock_es):
        """When username != 'main', user version wins over main."""
        client = MagicMock()
        mock_es._get_client.return_value = client

        # Return same element from both user and main
        client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "user_e1",
                            "hash_id": "h1",
                            "username": "dev",
                            "name": "foo",
                            "element_type": "function",
                            "relative_path": "a.py",
                            "summary_embedding": [1.0],
                        }
                    },
                    {
                        "_source": {
                            "element_id": "main_e1",
                            "hash_id": "h1m",
                            "username": "main",
                            "name": "foo",
                            "element_type": "function",
                            "relative_path": "a.py",
                            "summary_embedding": [0.9],
                        }
                    },
                ]
            }
        }

        mock_es.search_by_vector.return_value = []

        processed, relationships = compute_semantic_relationships(
            mock_es, "s", "r", "dev",
        )

        # Only one element processed (deduplicated)
        assert processed == 1


# =============================================================================
# MCP TOOL INTEGRATION TESTS
# =============================================================================


class TestGetCallGraphSemanticRelated:
    """Tests for semantic_related in get_call_graph MCP tool."""

    @pytest.fixture
    def mock_es_repo(self):
        """Create a mock ES repo for MCP tool tests."""
        es = MagicMock()
        es.get_document.return_value = None
        es.get_document_by_id_or_hash.side_effect = lambda x: es.get_document(x)
        es.find_elements_calling.return_value = []
        es.get_calls.return_value = []
        return es

    def test_includes_semantic_related_when_data_exists(self, mock_es_repo):
        """get_call_graph includes semantic_related when data exists on element."""
        mock_es_repo.get_document.return_value = {
            "element_id": "func1",
            "hash_id": "h1",
            "name": "my_func",
            "element_type": "function",
            "relative_path": "utils.py",
            "line_start": 10,
            "scope": "s",
            "repository": "r",
            "username": "main",
            "semantic_related": [
                {"element_id": "rel1", "hash_id": "rh1", "score": 0.85},
                {"element_id": "rel2", "hash_id": "rh2", "score": 0.72},
            ],
        }

        # Mock the resolution of related elements
        def get_document_side_effect(eid):
            docs = {
                "func1": mock_es_repo.get_document.return_value,
                "rel1": {
                    "element_id": "rel1",
                    "hash_id": "rh1",
                    "name": "similar_func",
                    "element_type": "function",
                    "relative_path": "helpers.py",
                    "line_start": 20,
                    "summary": "Does something similar",
                },
                "rel2": {
                    "element_id": "rel2",
                    "hash_id": "rh2",
                    "name": "another_func",
                    "element_type": "function",
                    "relative_path": "other.py",
                    "line_start": 5,
                    "summary": "Also related",
                },
            }
            return docs.get(eid)

        mock_es_repo.get_document.side_effect = get_document_side_effect
        mock_es_repo.get_document_by_id_or_hash.side_effect = get_document_side_effect

        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        result = get_call_graph(es=mock_es_repo, element_id="func1")

        assert "semantic_related" in result
        assert len(result["semantic_related"]) == 2
        assert result["semantic_related"][0]["name"] == "similar_func"
        assert result["semantic_related"][0]["score"] == 0.85
        assert result["semantic_related"][1]["name"] == "another_func"

    def test_no_semantic_related_when_empty(self, mock_es_repo):
        """get_call_graph works fine when no semantic data exists."""
        mock_es_repo.get_document.return_value = {
            "element_id": "func1",
            "hash_id": "h1",
            "name": "my_func",
            "element_type": "function",
            "relative_path": "utils.py",
            "line_start": 10,
            "scope": "s",
            "repository": "r",
            "username": "main",
        }

        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        result = get_call_graph(es=mock_es_repo, element_id="func1")

        assert "semantic_related" not in result


# =============================================================================
# FORMATTER TESTS
# =============================================================================


class TestCallGraphFormatterSemanticRelated:
    """Tests for semantic_related section in CallGraphFormatter."""

    def test_renders_semantic_related_section(self):
        """Formatter renders 'Semantically Related' section."""
        result = {
            "element": {"name": "my_func", "type": "function", "file": "utils.py", "line": 10},
            "callers": [],
            "callees": [],
            "semantic_related": [
                {
                    "name": "similar_func",
                    "type": "function",
                    "file": "helpers.py",
                    "line": 20,
                    "score": 0.85,
                    "summary": "Does something similar",
                },
                {
                    "name": "another_func",
                    "type": "method",
                    "file": "other.py",
                    "line": 5,
                    "score": 0.72,
                    "summary": "Also related",
                },
            ],
        }

        formatter = CallGraphFormatter()
        output = formatter.format(result)

        assert "Semantically Related (2):" in output
        assert "similar_func" in output
        assert "85% similar" in output
        assert "another_func" in output
        assert "72% similar" in output
        assert "Does something similar" in output

    def test_no_semantic_section_when_absent(self):
        """Formatter doesn't render section when no semantic data."""
        result = {
            "element": {"name": "my_func", "type": "function", "file": "utils.py", "line": 10},
            "callers": [],
            "callees": [],
        }

        formatter = CallGraphFormatter()
        output = formatter.format(result)

        assert "Semantically Related" not in output
