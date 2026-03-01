"""Tests for embedding-based call resolution and semantic relationships.

Tests Strategy 6 (embedding resolution), semantic relationship computation,
and MCP tool integration.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from magaldi_core.call_resolution import (
    _build_constructor_type_map,
    _build_resolved_maps,
    _cosine_similarity,
    _is_likely_class_name,
    _merge_candidates,
    _receiver_class_affinity,
    _resolve_via_constructors,
    _resolve_via_scope_bindings,
    _score_candidates_rrf,
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

    def test_multiple_candidates_rrf_picks_best(self, mock_es):
        """With multiple candidates, RRF-scored best match wins."""
        caller_emb = [0.8, 0.6, 0.0]

        mock_es.find_all_elements_with_calls.return_value = [
            {
                "element_id": "caller1",
                "relative_path": "src/main.py",
                "calls": [
                    {"name": "process", "receiver": "svc", "category": "untyped", "resolved_id": None},
                ],
            }
        ]
        mock_es.find_candidates_by_name.return_value = [
            {"element_id": "bad_match", "name": "process", "element_type": "method",
             "summary_embedding": [0.0, 0.0, 1.0], "parent_id": "", "relative_path": "a.py"},
            {"element_id": "good_match", "name": "process", "element_type": "method",
             "summary_embedding": [0.7, 0.7, 0.0], "parent_id": "x:r:main:svc.py:class:Service:1",
             "relative_path": "svc.py"},
        ]
        mock_es.get_embedding.return_value = caller_emb

        total, single, embedding = resolve_calls_by_embedding(mock_es, "s", "r", "main")

        assert total == 1
        assert single == 0
        assert embedding == 1
        stored_calls = mock_es.store_calls.call_args[0][1]
        assert stored_calls[0]["resolved_id"] == "good_match"
        # Verify caller embedding is fetched with "caller" type (asymmetric)
        mock_es.get_embedding.assert_called_with("caller1", "caller")

    def test_below_threshold_stays_unresolved(self, mock_es):
        """Candidates below RRF threshold are not resolved."""
        mock_es.find_all_elements_with_calls.return_value = [
            {
                "element_id": "caller1",
                "calls": [
                    {"name": "run", "receiver": "obj", "category": "untyped", "resolved_id": None},
                ],
            }
        ]
        mock_es.find_candidates_by_name.return_value = [
            {"element_id": "c1", "name": "run", "element_type": "method", "summary_embedding": [0.0, 1.0, 0.0], "parent_id": "", "relative_path": "a.py"},
            {"element_id": "c2", "name": "run", "element_type": "method", "summary_embedding": [0.0, 0.0, 1.0], "parent_id": "", "relative_path": "b.py"},
        ]
        # Caller embedding is very different from both candidates
        mock_es.get_embedding.return_value = [1.0, 0.0, 0.0]

        # Use an impossibly high threshold to force rejection
        total, single, embedding = resolve_calls_by_embedding(mock_es, "s", "r", "main", min_rrf_score=1.0)

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

    def test_caller_without_embedding_uses_name_signals(self, mock_es):
        """When caller has no embedding, RRF still works via name/path signals."""
        mock_es.find_all_elements_with_calls.return_value = [
            {
                "element_id": "caller1",
                "relative_path": "src/main.py",
                "calls": [
                    {"name": "process", "receiver": "svc", "category": "untyped", "resolved_id": None},
                ],
            }
        ]
        mock_es.find_candidates_by_name.return_value = [
            {"element_id": "c1", "name": "process", "element_type": "method",
             "summary_embedding": [1.0], "parent_id": "", "relative_path": "a.py"},
            {"element_id": "c2", "name": "process", "element_type": "method",
             "summary_embedding": [0.0], "parent_id": "x:r:main:svc.py:class:Service:1",
             "relative_path": "svc.py"},
        ]
        mock_es.get_embedding.return_value = None  # No embedding for caller

        total, single, embedding = resolve_calls_by_embedding(mock_es, "s", "r", "main")

        assert total == 1
        # RRF uses name/path signals even without embedding — c2 matches "svc"
        assert embedding == 1
        stored_calls = mock_es.store_calls.call_args[0][1]
        assert stored_calls[0]["resolved_id"] == "c2"

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
    def mock_repo(self):
        """Create a mock ES repo for MCP tool tests."""
        es = MagicMock()
        es.get_document.return_value = None
        es.get_document_by_id_or_hash.side_effect = lambda x: es.get_document(x)
        es.find_elements_calling.return_value = []
        es.get_calls.return_value = []
        return es

    def test_includes_semantic_related_when_data_exists(self, mock_repo):
        """get_call_graph includes semantic_related when data exists on element."""
        mock_repo.get_document.return_value = {
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
                "func1": mock_repo.get_document.return_value,
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

        mock_repo.get_document.side_effect = get_document_side_effect
        mock_repo.get_document_by_id_or_hash.side_effect = get_document_side_effect

        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        result = get_call_graph(repo=mock_repo, element_id="func1")

        assert "semantic_related" in result
        assert len(result["semantic_related"]) == 2
        assert result["semantic_related"][0]["name"] == "similar_func"
        assert result["semantic_related"][0]["score"] == 0.85
        assert result["semantic_related"][1]["name"] == "another_func"

    def test_no_semantic_related_when_empty(self, mock_repo):
        """get_call_graph works fine when no semantic data exists."""
        mock_repo.get_document.return_value = {
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
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        result = get_call_graph(repo=mock_repo, element_id="func1")

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


# =============================================================================
# CONSTRUCTOR-BASED TYPE INFERENCE TESTS (Strategy 5.6)
# =============================================================================


class TestIsLikelyClassName:
    """Tests for _is_likely_class_name helper."""

    def test_pascal_case(self):
        assert _is_likely_class_name("Repository") is True

    def test_multi_word_pascal(self):
        assert _is_likely_class_name("HttpClient") is True

    def test_lowercase(self):
        assert _is_likely_class_name("repository") is False

    def test_all_caps(self):
        """All-caps names are constants, not classes."""
        assert _is_likely_class_name("HTTP") is False

    def test_single_char(self):
        assert _is_likely_class_name("R") is False

    def test_empty(self):
        assert _is_likely_class_name("") is False


class TestBuildConstructorTypeMap:
    """Tests for _build_constructor_type_map."""

    def test_simple_constructor(self):
        code = "repo = Repository(config)"
        result = _build_constructor_type_map(code)
        assert result == {"repo": "Repository"}

    def test_module_prefixed_constructor(self):
        """var = module.ClassName() should capture ClassName."""
        code = "parser = db.Parser()"
        result = _build_constructor_type_map(code)
        assert result == {"parser": "Parser"}

    def test_await_constructor(self):
        code = "client = await AsyncClient()"
        result = _build_constructor_type_map(code)
        assert result == {"client": "AsyncClient"}

    def test_lowercase_not_captured(self):
        """Lowercase function calls are not constructors."""
        code = "result = get_data()"
        result = _build_constructor_type_map(code)
        assert result == {}

    def test_builtin_constructor_excluded(self):
        """Built-in exception types are excluded."""
        code = "err = ValueError('bad')"
        result = _build_constructor_type_map(code)
        assert result == {}

    def test_all_caps_excluded(self):
        """ALL_CAPS names are constants, not classes."""
        code = "x = HTTP()"
        result = _build_constructor_type_map(code)
        assert result == {}

    def test_multiple_constructors(self):
        code = """
repo = Repository()
client = HttpClient(base_url)
state = AppState(config)
"""
        result = _build_constructor_type_map(code)
        assert result == {
            "repo": "Repository",
            "client": "HttpClient",
            "state": "AppState",
        }

    def test_reassignment_last_wins(self):
        code = """
x = Foo()
x = Bar()
"""
        result = _build_constructor_type_map(code)
        assert result == {"x": "Bar"}


class TestResolveViaConstructors:
    """Tests for _resolve_via_constructors (Strategy 5.6)."""

    @pytest.fixture
    def mock_repo(self):
        repo = MagicMock()
        repo.get_document.return_value = None
        repo.get_document_by_name_only.return_value = None
        repo.get_method_by_class.return_value = None
        repo.store_calls.return_value = True

        # Delegate batch fetch to individual get_document calls
        def _batch_fetch(ids):
            result = {}
            for eid in ids:
                doc = repo.get_document(eid)
                if doc:
                    result[eid] = doc
            return result

        repo.get_documents_batch.side_effect = _batch_fetch
        return repo

    def test_simple_constructor_resolution(self, mock_repo):
        """repo = Repository(); repo.get() resolves to Repository.get."""
        elements = [
            {
                "element_id": "func1",
                "calls": [
                    {"name": "get", "receiver": "repo", "category": "untyped", "resolved_id": None},
                ],
            }
        ]

        mock_repo.get_document.return_value = {
            "element_id": "func1",
            "raw_code": "repo = Repository()\nrepo.get()",
            "calls": [
                {"name": "get", "receiver": "repo", "category": "untyped", "resolved_id": None},
            ],
        }
        mock_repo.get_document_by_name_only.return_value = {
            "element_id": "class_repo",
            "name": "Repository",
        }
        mock_repo.get_method_by_class.return_value = {
            "element_id": "repo_get_method",
        }

        count = _resolve_via_constructors(
            mock_repo, elements, "s", "r", "main"
        )

        assert count == 1
        mock_repo.store_calls.assert_called_once()
        stored = mock_repo.store_calls.call_args[0][1]
        assert stored[0]["resolved_id"] == "repo_get_method"
        assert stored[0]["category"] == "constructor_resolved"

    def test_module_prefixed_constructor(self, mock_repo):
        """repo = db.Repository(); repo.get() resolves."""
        elements = [
            {
                "element_id": "func1",
                "calls": [
                    {"name": "get", "receiver": "repo", "category": "untyped", "resolved_id": None},
                ],
            }
        ]

        mock_repo.get_document.return_value = {
            "element_id": "func1",
            "raw_code": "repo = db.Repository()\nrepo.get()",
            "calls": [
                {"name": "get", "receiver": "repo", "category": "untyped", "resolved_id": None},
            ],
        }
        mock_repo.get_document_by_name_only.return_value = {
            "element_id": "class_repo",
            "name": "Repository",
        }
        mock_repo.get_method_by_class.return_value = {
            "element_id": "repo_get_method",
        }

        count = _resolve_via_constructors(
            mock_repo, elements, "s", "r", "main"
        )

        assert count == 1

    def test_lowercase_func_not_treated_as_constructor(self, mock_repo):
        """result = get_data(); result.process() should NOT be resolved."""
        elements = [
            {
                "element_id": "func1",
                "calls": [
                    {"name": "process", "receiver": "result", "category": "untyped", "resolved_id": None},
                ],
            }
        ]

        mock_repo.get_document.return_value = {
            "element_id": "func1",
            "raw_code": "result = get_data()\nresult.process()",
            "calls": [
                {"name": "process", "receiver": "result", "category": "untyped", "resolved_id": None},
            ],
        }

        count = _resolve_via_constructors(
            mock_repo, elements, "s", "r", "main"
        )

        assert count == 0
        mock_repo.store_calls.assert_not_called()

    def test_class_not_found_stays_unresolved(self, mock_repo):
        """Constructor for unknown class stays unresolved."""
        elements = [
            {
                "element_id": "func1",
                "calls": [
                    {"name": "foo", "receiver": "obj", "category": "untyped", "resolved_id": None},
                ],
            }
        ]

        mock_repo.get_document.return_value = {
            "element_id": "func1",
            "raw_code": "obj = NonExistent()\nobj.foo()",
            "calls": [
                {"name": "foo", "receiver": "obj", "category": "untyped", "resolved_id": None},
            ],
        }
        mock_repo.get_document_by_name_only.return_value = None  # Class not found

        count = _resolve_via_constructors(
            mock_repo, elements, "s", "r", "main"
        )

        assert count == 0
        mock_repo.store_calls.assert_not_called()

    def test_already_resolved_calls_skipped(self, mock_repo):
        """Calls that are already resolved should not be re-resolved."""
        elements = [
            {
                "element_id": "func1",
                "calls": [
                    {"name": "get", "receiver": "repo", "category": "resolved", "resolved_id": "existing"},
                ],
            }
        ]

        mock_repo.get_document.return_value = {
            "element_id": "func1",
            "raw_code": "repo = Repository()\nrepo.get()",
            "calls": [
                {"name": "get", "receiver": "repo", "category": "resolved", "resolved_id": "existing"},
            ],
        }

        count = _resolve_via_constructors(
            mock_repo, elements, "s", "r", "main"
        )

        assert count == 0

    def test_no_raw_code_skipped(self, mock_repo):
        """Elements without raw_code are skipped."""
        elements = [
            {
                "element_id": "func1",
                "calls": [
                    {"name": "get", "receiver": "repo", "category": "untyped", "resolved_id": None},
                ],
            }
        ]

        mock_repo.get_document.return_value = {
            "element_id": "func1",
            "raw_code": "",
            "calls": [
                {"name": "get", "receiver": "repo", "category": "untyped", "resolved_id": None},
            ],
        }

        count = _resolve_via_constructors(
            mock_repo, elements, "s", "r", "main"
        )

        assert count == 0


# =============================================================================
# RRF SCORING TESTS
# =============================================================================


class TestReceiverClassAffinity:
    """Tests for _receiver_class_affinity helper."""

    def test_prefix_match(self):
        """'repo' is prefix of parent class 'Repository' → 1.0."""
        candidate = {
            "parent_id": "s:r:main:file.py:class:Repository:1",
            "relative_path": "file.py",
        }
        assert _receiver_class_affinity("repo", candidate) == 1.0

    def test_substring_match(self):
        """'search' is prefix of 'SearchRepository' (lowercased) → 1.0."""
        candidate = {
            "parent_id": "s:r:main:file.py:class:SearchRepository:1",
            "relative_path": "file.py",
        }
        # "search" is a prefix of "searchrepository" (lowercased), so 1.0
        assert _receiver_class_affinity("search", candidate) == 1.0

    def test_true_substring_match(self):
        """'meta' in 'ElementMetadata' → 0.8 (substring but not prefix)."""
        candidate = {
            "parent_id": "s:r:main:file.py:class:ElementMetadata:1",
            "relative_path": "file.py",
        }
        assert _receiver_class_affinity("meta", candidate) == 0.8

    def test_underscore_part_match(self):
        """'es_repo' splits to ['repo'] which is prefix of 'Repository' → 0.7."""
        candidate = {
            "parent_id": "s:r:main:file.py:class:Repository:1",
            "relative_path": "file.py",
        }
        score = _receiver_class_affinity("es_repo", candidate)
        assert score >= 0.7

    def test_path_match(self):
        """'repo' in path 'repository.py' → at least 0.5."""
        candidate = {
            "parent_id": "",
            "relative_path": "src/repository.py",
        }
        score = _receiver_class_affinity("repo", candidate)
        assert score >= 0.5

    def test_no_match(self):
        """No relationship → 0.0."""
        candidate = {
            "parent_id": "s:r:main:file.py:class:HttpClient:1",
            "relative_path": "client.py",
        }
        assert _receiver_class_affinity("repo", candidate) == 0.0

    def test_short_receiver_ignored(self):
        """Very short receivers (< 3 chars) don't match to avoid noise."""
        candidate = {
            "parent_id": "s:r:main:file.py:class:Repository:1",
            "relative_path": "repository.py",
        }
        assert _receiver_class_affinity("re", candidate) == 0.0

    def test_no_parent_id(self):
        """Candidate without parent_id can still match on path."""
        candidate = {
            "parent_id": None,
            "relative_path": "src/repository.py",
        }
        score = _receiver_class_affinity("repo", candidate)
        assert score >= 0.5


class TestScoreCandidatesRRF:
    """Tests for _score_candidates_rrf."""

    def test_single_candidate_gets_max_score(self):
        """Single candidate gets rank 1 in all signals."""
        call = {"receiver": "repo", "name": "get"}
        candidates = [
            {"element_id": "c1", "parent_id": "s:r:main:f.py:class:Repository:1",
             "relative_path": "repository.py", "summary_embedding": [1.0, 0.0]},
        ]
        caller_emb = [0.9, 0.1]

        results = _score_candidates_rrf(call, candidates, caller_emb)

        assert len(results) == 1
        assert results[0][0]["element_id"] == "c1"
        # Score should be 3 * 1/(60+1) ≈ 0.049
        assert abs(results[0][1] - 3 / 61) < 0.001

    def test_receiver_affinity_breaks_tie(self):
        """When embeddings are similar, receiver-class affinity disambiguates."""
        call = {"receiver": "repo", "name": "save"}
        candidates = [
            {"element_id": "wrong", "parent_id": "s:r:main:f.py:class:HttpClient:1",
             "relative_path": "client.py", "summary_embedding": [0.7, 0.7]},
            {"element_id": "right", "parent_id": "s:r:main:f.py:class:Repository:1",
             "relative_path": "repository.py", "summary_embedding": [0.7, 0.7]},
        ]
        # Same embedding for both candidates — tie on embedding signal
        caller_emb = [0.7, 0.7]

        results = _score_candidates_rrf(call, candidates, caller_emb)

        # "right" should score higher due to receiver-class affinity
        assert results[0][0]["element_id"] == "right"
        assert results[0][1] > results[1][1]

    def test_no_embedding_still_scores(self):
        """Without caller embedding, name/path signals still produce scores."""
        call = {"receiver": "repo", "name": "get"}
        candidates = [
            {"element_id": "c1", "parent_id": "", "relative_path": "a.py",
             "summary_embedding": [1.0]},
            {"element_id": "c2", "parent_id": "s:r:main:repo.py:class:Repository:1",
             "relative_path": "repo.py", "summary_embedding": [0.5]},
        ]

        results = _score_candidates_rrf(call, candidates, caller_embedding=None)

        assert len(results) == 2
        # c2 should score higher (receiver "repo" matches parent "Repository")
        assert results[0][0]["element_id"] == "c2"

    def test_empty_candidates(self):
        """Empty candidates list returns empty."""
        results = _score_candidates_rrf(
            {"receiver": "x", "name": "y"}, [], [1.0]
        )
        assert results == []

    def test_embedding_signal_contributes(self):
        """Strong embedding match boosts score."""
        call = {"receiver": "thing", "name": "do"}
        candidates = [
            {"element_id": "c1", "parent_id": "", "relative_path": "a.py",
             "summary_embedding": [1.0, 0.0]},
            {"element_id": "c2", "parent_id": "", "relative_path": "b.py",
             "summary_embedding": [0.0, 1.0]},
        ]
        # Embedding strongly prefers c1
        caller_emb = [0.99, 0.01]

        results = _score_candidates_rrf(call, candidates, caller_emb)

        # c1 should win on embedding signal
        assert results[0][0]["element_id"] == "c1"


# =============================================================================
# SCOPE BINDING RESOLUTION TESTS (Strategy 5.7)
# =============================================================================


class TestBuildResolvedMaps:
    """Tests for _build_resolved_maps helper."""

    def test_bare_calls_mapped(self):
        calls = [
            {"name": "get_user", "receiver": None, "resolved_id": "target1"},
        ]
        result = _build_resolved_maps(calls)
        assert result == {"get_user": "target1"}

    def test_receiver_calls_mapped(self):
        calls = [
            {"name": "connect", "receiver": "db", "resolved_id": "target2"},
        ]
        result = _build_resolved_maps(calls)
        assert result == {"db.connect": "target2"}

    def test_unresolved_calls_excluded(self):
        calls = [
            {"name": "foo", "receiver": "bar", "resolved_id": None},
        ]
        result = _build_resolved_maps(calls)
        assert result == {}

    def test_mixed_calls(self):
        calls = [
            {"name": "get_user", "receiver": None, "resolved_id": "t1"},
            {"name": "connect", "receiver": "db", "resolved_id": "t2"},
            {"name": "foo", "receiver": "bar", "resolved_id": None},
        ]
        result = _build_resolved_maps(calls)
        assert result == {"get_user": "t1", "db.connect": "t2"}


class TestResolveViaScopeBindings:
    """Tests for _resolve_via_scope_bindings (Strategy 5.7)."""

    @pytest.fixture
    def mock_repo(self):
        repo = MagicMock()
        repo.get_document.return_value = None
        repo.get_document_by_name_only.return_value = None
        repo.get_method_by_class.return_value = None
        repo.store_calls.return_value = True

        # Delegate batch fetch to individual get_document calls
        def _batch_fetch(ids):
            result = {}
            for eid in ids:
                doc = repo.get_document(eid)
                if doc:
                    result[eid] = doc
            return result

        repo.get_documents_batch.side_effect = _batch_fetch
        return repo

    def test_method_return_chain(self, mock_repo):
        """conn = db.connect(); conn.cursor() resolves when connect() is
        already resolved with return_type."""
        raw_code = (
            "def process():\n"
            "    conn = db.connect()\n"
            "    conn.cursor()\n"
        )
        elements = [
            {
                "element_id": "func1",
                "calls": [
                    {"name": "connect", "receiver": "db", "resolved_id": "db_connect_id",
                     "category": "resolved"},
                    {"name": "cursor", "receiver": "conn", "resolved_id": None,
                     "category": "untyped"},
                ],
            }
        ]

        # get_document for the element itself
        mock_repo.get_document.side_effect = lambda eid: {
            "func1": {
                "element_id": "func1",
                "raw_code": raw_code,
                "language": "python",
                "parameters": [],
                "calls": elements[0]["calls"].copy(),
            },
            # get_document for db.connect() target — has return_type
            "db_connect_id": {
                "element_id": "db_connect_id",
                "return_type": "Connection",
            },
        }.get(eid)

        mock_repo.get_document_by_name_only.return_value = {
            "element_id": "class_connection",
            "name": "Connection",
        }
        mock_repo.get_method_by_class.return_value = {
            "element_id": "connection_cursor_method",
            "return_type": "Cursor",
        }

        count = _resolve_via_scope_bindings(
            mock_repo, elements, "s", "r", "main"
        )

        assert count == 1
        mock_repo.store_calls.assert_called_once()
        stored = mock_repo.store_calls.call_args[0][1]
        resolved_call = next(c for c in stored if c["name"] == "cursor")
        assert resolved_call["resolved_id"] == "connection_cursor_method"
        assert resolved_call["category"] == "scope_resolved"

    def test_except_as_resolution(self, mock_repo):
        """except ValueError as e: e.args resolves to ValueError.args."""
        raw_code = (
            "def process():\n"
            "    try:\n"
            "        pass\n"
            "    except ValueError as e:\n"
            "        e.args\n"
        )
        calls = [
            {"name": "args", "receiver": "e", "resolved_id": None,
             "category": "untyped"},
        ]
        elements = [{"element_id": "func1", "calls": calls}]

        mock_repo.get_document.return_value = {
            "element_id": "func1",
            "raw_code": raw_code,
            "language": "python",
            "parameters": [],
            "calls": calls.copy(),
        }
        mock_repo.get_document_by_name_only.return_value = {
            "element_id": "class_valueerror",
            "name": "ValueError",
        }
        mock_repo.get_method_by_class.return_value = {
            "element_id": "valueerror_args",
        }

        count = _resolve_via_scope_bindings(
            mock_repo, elements, "s", "r", "main"
        )

        assert count == 1

    def test_with_as_resolution(self, mock_repo):
        """with open(f) as handle: handle.read() resolves."""
        raw_code = (
            "def process():\n"
            "    with open('f') as handle:\n"
            "        handle.read()\n"
        )
        calls = [
            {"name": "open", "receiver": None, "resolved_id": "builtin_open",
             "category": "resolved"},
            {"name": "read", "receiver": "handle", "resolved_id": None,
             "category": "untyped"},
        ]
        elements = [{"element_id": "func1", "calls": calls}]

        mock_repo.get_document.side_effect = lambda eid: {
            "func1": {
                "element_id": "func1",
                "raw_code": raw_code,
                "language": "python",
                "parameters": [],
                "calls": calls.copy(),
            },
            "builtin_open": {
                "element_id": "builtin_open",
                "return_type": "TextIOWrapper",
            },
        }.get(eid)

        mock_repo.get_document_by_name_only.return_value = {
            "element_id": "class_textio",
            "name": "TextIOWrapper",
        }
        mock_repo.get_method_by_class.return_value = {
            "element_id": "textio_read",
        }

        count = _resolve_via_scope_bindings(
            mock_repo, elements, "s", "r", "main"
        )

        assert count == 1

    def test_non_python_skipped(self, mock_repo):
        """Non-Python elements are skipped."""
        elements = [
            {
                "element_id": "func1",
                "calls": [
                    {"name": "foo", "receiver": "obj", "category": "untyped",
                     "resolved_id": None},
                ],
            }
        ]

        mock_repo.get_document.return_value = {
            "element_id": "func1",
            "raw_code": "function process() { obj.foo() }",
            "language": "javascript",
            "parameters": [],
            "calls": elements[0]["calls"],
        }

        count = _resolve_via_scope_bindings(
            mock_repo, elements, "s", "r", "main"
        )

        assert count == 0

    def test_no_unresolved_receiver_calls(self, mock_repo):
        """Elements with all resolved calls are skipped."""
        elements = [
            {
                "element_id": "func1",
                "calls": [
                    {"name": "get", "receiver": "repo", "category": "resolved",
                     "resolved_id": "existing"},
                ],
            }
        ]

        count = _resolve_via_scope_bindings(
            mock_repo, elements, "s", "r", "main"
        )

        assert count == 0
        mock_repo.get_document.assert_not_called()

    def test_param_type_used_for_receiver(self, mock_repo):
        """Receiver type from param annotations enables method return type lookup."""
        raw_code = (
            "def process(db: Database):\n"
            "    conn = db.connect()\n"
            "    conn.cursor()\n"
        )
        calls = [
            {"name": "connect", "receiver": "db", "resolved_id": None,
             "category": "untyped"},
            {"name": "cursor", "receiver": "conn", "resolved_id": None,
             "category": "untyped"},
        ]
        elements = [{"element_id": "func1", "calls": calls}]

        mock_repo.get_document.return_value = {
            "element_id": "func1",
            "raw_code": raw_code,
            "language": "python",
            "parameters": [{"name": "db", "type": "Database"}],
            "calls": calls.copy(),
        }

        # db.connect() lookup: Database class -> connect method with return_type
        mock_repo.get_document_by_name_only.return_value = {
            "element_id": "class_database",
            "name": "Database",
        }

        def method_by_class(class_id, method_name, **_kw):
            if class_id == "class_database" and method_name == "connect":
                return {"element_id": "db_connect", "return_type": "Connection"}
            if class_id == "class_connection" and method_name == "cursor":
                return {"element_id": "conn_cursor"}
            return None

        mock_repo.get_method_by_class.side_effect = method_by_class

        # For the second pass, also need to look up Connection class
        def doc_by_name(name, element_type, **_kw):  # noqa: ARG001
            if name == "Database":
                return {"element_id": "class_database", "name": "Database"}
            if name == "Connection":
                return {"element_id": "class_connection", "name": "Connection"}
            return None

        mock_repo.get_document_by_name_only.side_effect = doc_by_name

        count = _resolve_via_scope_bindings(
            mock_repo, elements, "s", "r", "main"
        )

        # Should resolve at least conn.cursor() via chained resolution
        assert count >= 1


# =============================================================================
# TEST FIXTURE FILTERING TESTS
# =============================================================================


class TestFilterCandidatesForCaller:
    """Tests for _filter_candidates_for_caller."""

    def test_production_caller_excludes_fixture_candidates(self):
        from magaldi_core.call_resolution import _filter_candidates_for_caller

        candidates = [
            {"element_id": "prod1", "relative_path": "src/utils.py"},
            {"element_id": "fixture1", "relative_path": "tests/fixtures/repos/utils.py"},
        ]
        result = _filter_candidates_for_caller(candidates, "src/main.py")
        assert len(result) == 1
        assert result[0]["element_id"] == "prod1"

    def test_test_caller_keeps_all_candidates(self):
        from magaldi_core.call_resolution import _filter_candidates_for_caller

        candidates = [
            {"element_id": "prod1", "relative_path": "src/utils.py"},
            {"element_id": "fixture1", "relative_path": "tests/fixtures/repos/utils.py"},
        ]
        result = _filter_candidates_for_caller(candidates, "tests/test_main.py")
        assert len(result) == 2

    def test_production_caller_keeps_non_fixture_test(self):
        """Regular test files (not fixtures) are NOT excluded."""
        from magaldi_core.call_resolution import _filter_candidates_for_caller

        candidates = [
            {"element_id": "test1", "relative_path": "tests/test_utils.py"},
            {"element_id": "prod1", "relative_path": "src/utils.py"},
        ]
        result = _filter_candidates_for_caller(candidates, "src/main.py")
        assert len(result) == 2

    def test_all_candidates_filtered_returns_empty(self):
        from magaldi_core.call_resolution import _filter_candidates_for_caller

        candidates = [
            {"element_id": "f1", "relative_path": "tests/fixtures/a.py"},
            {"element_id": "f2", "relative_path": "test/fixtures/b.py"},
        ]
        result = _filter_candidates_for_caller(candidates, "src/main.py")
        assert len(result) == 0


class TestTestFixturePaths:
    """Tests for _is_test_fixture_path and _is_test_path helpers."""

    def test_fixture_paths_detected(self):
        from magaldi_core.call_resolution import _is_test_fixture_path

        assert _is_test_fixture_path("tests/fixtures/repos/valid_repo/src/utils.py")
        assert _is_test_fixture_path("test/fixtures/data.json")
        assert _is_test_fixture_path("src/__fixtures__/mock.ts")
        assert _is_test_fixture_path("tests/fixture/sample.py")

    def test_non_fixture_paths_not_detected(self):
        from magaldi_core.call_resolution import _is_test_fixture_path

        assert not _is_test_fixture_path("src/utils.py")
        assert not _is_test_fixture_path("tests/test_utils.py")
        assert not _is_test_fixture_path("src/shared/db/repositories/search.py")

    def test_test_paths_detected(self):
        from magaldi_core.call_resolution import _is_test_path

        assert _is_test_path("tests/test_main.py")
        assert _is_test_path("test/test_utils.py")
        assert _is_test_path("src/tests/test_config.py")
        assert _is_test_path("tests/unit/test_parser.py")

    def test_production_paths_not_test(self):
        from magaldi_core.call_resolution import _is_test_path

        assert not _is_test_path("src/main.py")
        assert not _is_test_path("src/shared/config.py")
        assert not _is_test_path("lib/utils.py")


# =============================================================================
# FIXTURE FILTERING IN EMBEDDING RESOLUTION E2E
# =============================================================================


class TestEmbeddingResolutionFixtureFiltering:
    """E2E tests: fixture candidates are excluded for production callers."""

    def test_fixture_candidate_excluded_for_production_caller(self, mock_es):
        """Production code calling .add() should not resolve to test fixture."""
        mock_es.find_all_elements_with_calls.return_value = [
            {
                "element_id": "caller1",
                "relative_path": "src/processor.py",
                "calls": [
                    {"name": "add", "receiver": "tracker", "category": "untyped", "resolved_id": None},
                ],
            }
        ]
        # One real candidate, one fixture
        mock_es.find_candidates_by_name.return_value = [
            {"element_id": "fixture_add", "name": "add", "element_type": "function",
             "summary_embedding": [0.9, 0.1], "parent_id": "",
             "relative_path": "tests/fixtures/repos/valid_repo/src/utils.py"},
            {"element_id": "real_add", "name": "add", "element_type": "method",
             "summary_embedding": [0.5, 0.5], "parent_id": "x:r:main:tracker.py:class:Tracker:1",
             "relative_path": "src/tracker.py"},
        ]
        mock_es.get_embedding.return_value = [0.7, 0.3]

        total, single, embedding = resolve_calls_by_embedding(mock_es, "s", "r", "main")

        # Fixture filtered out, only real_add remains as single candidate
        assert single == 1 or embedding == 1
        stored_calls = mock_es.store_calls.call_args[0][1]
        assert stored_calls[0]["resolved_id"] == "real_add"

    def test_test_caller_can_resolve_to_fixture(self, mock_es):
        """Test code calling .add() CAN resolve to test fixture."""
        mock_es.find_all_elements_with_calls.return_value = [
            {
                "element_id": "test_caller",
                "relative_path": "tests/test_processor.py",
                "calls": [
                    {"name": "add", "receiver": "tracker", "category": "untyped", "resolved_id": None},
                ],
            }
        ]
        mock_es.find_candidates_by_name.return_value = [
            {"element_id": "fixture_add", "name": "add", "element_type": "function",
             "summary_embedding": [0.9, 0.1], "parent_id": "",
             "relative_path": "tests/fixtures/repos/valid_repo/src/utils.py"},
        ]

        total, single, embedding = resolve_calls_by_embedding(mock_es, "s", "r", "main")

        # Test caller: fixture NOT filtered out, single candidate resolves
        assert single == 1
        stored_calls = mock_es.store_calls.call_args[0][1]
        assert stored_calls[0]["resolved_id"] == "fixture_add"
