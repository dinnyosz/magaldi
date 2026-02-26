"""Integration tests for Elasticsearch search functionality.

Tests for text search, vector search, and pattern search.
"""

from __future__ import annotations

import pytest

from magaldi_core.code_parser import CodeElement
from shared.db.store import INDEX_NAME

pytestmark = pytest.mark.integration


# =============================================================================
# TEXT SEARCH TESTS
# =============================================================================


class TestElasticsearchTextSearch:
    """Tests for text-based search."""

    def test_search_by_name(self, repo, multiple_elements):
        """Test searching by element name."""
        for elem in multiple_elements:
            repo.index_element(elem)
        repo._get_client().indices_refresh(index=INDEX_NAME)

        results = repo.search_by_text("calculate")

        assert len(results) >= 1
        assert any(r["name"] == "calculate" for r in results)

    def test_search_by_docstring(self, repo, multiple_elements):
        """Test searching by docstring content."""
        for elem in multiple_elements:
            repo.index_element(elem)
        repo._get_client().indices_refresh(index=INDEX_NAME)

        # Search for "User model" which appears in the User class docstring
        results = repo.search_by_text("User model")

        assert len(results) >= 1
        assert any(r["name"] == "User" for r in results)

    def test_search_with_scope_filter(self, repo, multiple_elements):
        """Test searching with scope filter."""
        for elem in multiple_elements:
            repo.index_element(elem)
        repo._get_client().indices_refresh(index=INDEX_NAME)

        # Search with matching scope
        results = repo.search_by_text("validate", scope="test-es")
        assert len(results) >= 1

        # Search with non-matching scope
        results = repo.search_by_text("validate", scope="other-scope")
        assert len(results) == 0

    def test_search_with_type_filter(self, repo, multiple_elements):
        """Test searching with element type filter."""
        for elem in multiple_elements:
            repo.index_element(elem)
        repo._get_client().indices_refresh(index=INDEX_NAME)

        # Search only for classes
        results = repo.search_by_text("User", element_types=["class"])
        assert len(results) >= 1
        assert all(r["element_type"] == "class" for r in results)

        # Search only for functions - should not find User class
        results = repo.search_by_text("User", element_types=["function"])
        assert not any(r["name"] == "User" for r in results)


# =============================================================================
# VECTOR SEARCH TESTS
# =============================================================================


class TestElasticsearchVectorSearch:
    """Tests for vector-based semantic search."""

    def test_vector_search(self, repo, multiple_elements):
        """Test searching by vector similarity."""
        # Index elements with embeddings
        for i, elem in enumerate(multiple_elements):
            repo.index_element(elem)
            # Give each element a slightly different embedding
            embedding = [0.1 + (i * 0.1)] * 1024
            repo.store_embedding(elem.element_id, embedding)

        repo._get_client().indices_refresh(index=INDEX_NAME)

        # Search with a query vector
        query_vector = [0.1] * 1024
        results = repo.search_by_vector(query_vector, min_score=0.5)

        # Should return results
        assert len(results) >= 1
        # Results should have scores
        assert "_score" in results[0]
        # All indexed elements should be findable
        result_ids = {r["element_id"] for r in results}
        assert any(elem.element_id in result_ids for elem in multiple_elements)

    def test_vector_search_with_filters(self, repo, multiple_elements):
        """Test vector search with additional filters."""
        for _i, elem in enumerate(multiple_elements):
            repo.index_element(elem)
            embedding = [0.5] * 1024
            repo.store_embedding(elem.element_id, embedding)

        repo._get_client().indices_refresh(index=INDEX_NAME)

        query_vector = [0.5] * 1024

        # Filter by element type
        results = repo.search_by_vector(
            query_vector,
            element_types=["function"],
            min_score=0.5
        )

        assert len(results) >= 1
        assert all(r["element_type"] == "function" for r in results)

    def test_vector_search_by_summary_embedding(self, repo, multiple_elements):
        """Test vector search using summary embeddings."""
        for i, elem in enumerate(multiple_elements):
            repo.index_element(elem)
            # Store only summary embeddings
            summary_embedding = [0.1 + (i * 0.1)] * 1024
            repo.store_embedding(elem.element_id, summary_embedding, embedding_type="summary")

        repo._get_client().indices_refresh(index=INDEX_NAME)

        query_vector = [0.1] * 1024
        results = repo.search_by_vector(query_vector, min_score=0.5, embedding_type="summary")

        assert len(results) >= 1
        assert "_score" in results[0]

    def test_vector_search_by_code_embedding(self, repo, multiple_elements):
        """Test vector search using code embeddings."""
        for i, elem in enumerate(multiple_elements):
            repo.index_element(elem)
            # Store only code embeddings
            code_embedding = [0.2 + (i * 0.1)] * 1024
            repo.store_embedding(elem.element_id, code_embedding, embedding_type="code")

        repo._get_client().indices_refresh(index=INDEX_NAME)

        query_vector = [0.2] * 1024
        results = repo.search_by_vector(query_vector, min_score=0.5, embedding_type="code")

        assert len(results) >= 1
        assert "_score" in results[0]

    def test_vector_search_default_uses_summary_embedding(self, repo, multiple_elements):
        """Test that default vector search uses summary embeddings for backwards compatibility."""
        for _i, elem in enumerate(multiple_elements):
            repo.index_element(elem)
            # Store both types of embeddings with different values
            summary_embedding = [0.3] * 1024
            code_embedding = [0.9] * 1024
            repo.store_embedding(elem.element_id, summary_embedding, embedding_type="summary")
            repo.store_embedding(elem.element_id, code_embedding, embedding_type="code")

        repo._get_client().indices_refresh(index=INDEX_NAME)

        # Search without specifying embedding_type - should use summary
        query_vector = [0.3] * 1024
        results = repo.search_by_vector(query_vector, min_score=0.5)

        assert len(results) >= 1
        # Results should come from summary_embedding since query is closer to [0.3]

    def test_get_all_embeddings_summary_type(self, repo, multiple_elements):
        """Test get_all_embeddings returns summary embeddings by default."""
        for i, elem in enumerate(multiple_elements):
            repo.index_element(elem)
            summary_embedding = [0.1 + (i * 0.1)] * 1024
            repo.store_embedding(elem.element_id, summary_embedding, embedding_type="summary")

        repo._get_client().indices_refresh(index=INDEX_NAME)

        results = repo.get_all_embeddings(
            scope="test-es",
            repository="test-repo",
            username="main",
        )

        assert len(results) >= 1
        # Results should have summary_embedding field
        assert any("summary_embedding" in r for r in results)

    def test_get_all_embeddings_code_type(self, repo, multiple_elements):
        """Test get_all_embeddings returns code embeddings when specified."""
        for i, elem in enumerate(multiple_elements):
            repo.index_element(elem)
            code_embedding = [0.2 + (i * 0.1)] * 1024
            repo.store_embedding(elem.element_id, code_embedding, embedding_type="code")

        repo._get_client().indices_refresh(index=INDEX_NAME)

        results = repo.get_all_embeddings(
            scope="test-es",
            repository="test-repo",
            username="main",
            embedding_type="code",
        )

        assert len(results) >= 1
        # Results should have code_embedding field
        assert any("code_embedding" in r for r in results)

    def test_get_all_embeddings_excludes_tests_by_default(self, repo):
        """Test get_all_embeddings excludes test elements by default."""
        # Create a regular element
        regular_elem = CodeElement(
            element_id="test-es:test-repo:main:src/utils.py:function:process:1",
            scope="test-es",
            repository="test-repo",
            username="main",
            relative_path="src/utils.py",
            element_type="function",
            name="process",
            language="python",
            line_start=1,
            line_end=10,
            raw_code="def process(): pass",
            is_test=False,
        )

        # Create a test element
        test_elem = CodeElement(
            element_id="test-es:test-repo:main:tests/test_utils.py:function:test_process:1",
            scope="test-es",
            repository="test-repo",
            username="main",
            relative_path="tests/test_utils.py",
            element_type="function",
            name="test_process",
            language="python",
            line_start=1,
            line_end=10,
            raw_code="def test_process(): pass",
            is_test=True,
        )

        repo.index_element(regular_elem)
        repo.index_element(test_elem)
        repo.store_embedding(regular_elem.element_id, [0.1] * 1024, embedding_type="summary")
        repo.store_embedding(test_elem.element_id, [0.2] * 1024, embedding_type="summary")

        repo._get_client().indices_refresh(index=INDEX_NAME)

        # Default: exclude tests
        results = repo.get_all_embeddings(
            scope="test-es",
            repository="test-repo",
            username="main",
        )

        element_ids = [r["element_id"] for r in results]
        assert regular_elem.element_id in element_ids
        assert test_elem.element_id not in element_ids

    def test_get_all_embeddings_includes_tests_when_requested(self, repo):
        """Test get_all_embeddings includes test elements when exclude_tests=False."""
        # Create a regular element
        regular_elem = CodeElement(
            element_id="test-es:test-repo:main:src/utils2.py:function:process2:1",
            scope="test-es",
            repository="test-repo",
            username="main",
            relative_path="src/utils2.py",
            element_type="function",
            name="process2",
            language="python",
            line_start=1,
            line_end=10,
            raw_code="def process2(): pass",
            is_test=False,
        )

        # Create a test element
        test_elem = CodeElement(
            element_id="test-es:test-repo:main:tests/test_utils2.py:function:test_process2:1",
            scope="test-es",
            repository="test-repo",
            username="main",
            relative_path="tests/test_utils2.py",
            element_type="function",
            name="test_process2",
            language="python",
            line_start=1,
            line_end=10,
            raw_code="def test_process2(): pass",
            is_test=True,
        )

        repo.index_element(regular_elem)
        repo.index_element(test_elem)
        repo.store_embedding(regular_elem.element_id, [0.3] * 1024, embedding_type="summary")
        repo.store_embedding(test_elem.element_id, [0.4] * 1024, embedding_type="summary")

        repo._get_client().indices_refresh(index=INDEX_NAME)

        # Include tests
        results = repo.get_all_embeddings(
            scope="test-es",
            repository="test-repo",
            username="main",
            exclude_tests=False,
        )

        element_ids = [r["element_id"] for r in results]
        assert regular_elem.element_id in element_ids
        assert test_elem.element_id in element_ids


# =============================================================================
# PATTERN SEARCH TESTS
# =============================================================================


class TestPatternSearch:
    """Tests for pattern search methods."""

    @pytest.fixture
    def sample_elements(self, repo):
        """Create sample elements with raw_code for pattern testing."""
        elements = [
            CodeElement(
                element_id="test-pattern:repo:main:file.py:function:add_column:10",
                scope="test-pattern",
                repository="repo",
                username="main",
                relative_path="file.py",
                element_type="function",
                name="add_column",
                language="python",
                line_start=10,
                line_end=12,
                raw_code="def add_column(table, Model):\n    table.add_column('name', String)",
                is_test=False,
                level=2,
            ),
            CodeElement(
                element_id="test-pattern:repo:main:utils.py:function:process:20",
                scope="test-pattern",
                repository="repo",
                username="main",
                relative_path="utils.py",
                element_type="function",
                name="process",
                language="python",
                line_start=20,
                line_end=22,
                raw_code="def process(data):\n    return data.strip()",
                is_test=False,
                level=2,
            ),
            CodeElement(
                element_id="test-pattern:repo:main:test_file.py:function:test_add:30",
                scope="test-pattern",
                repository="repo",
                username="main",
                relative_path="test_file.py",
                element_type="function",
                name="test_add",
                language="python",
                line_start=30,
                line_end=32,
                raw_code="def test_add():\n    add_column(t, Model)",
                is_test=True,
                level=2,
            ),
        ]
        for elem in elements:
            repo.index_element(elem)
        repo._get_client().indices_refresh(index="magaldi-code-elements")
        return elements

    @pytest.mark.usefixtures("sample_elements")
    def test_search_by_regexp(self, repo):
        """Test regexp pattern search."""
        results = repo.search_by_regexp(
            pattern="add_column.*Model",
            scope="test-pattern",
            repository="repo",
        )
        assert len(results) >= 1
        assert any("add_column" in r.get("raw_code", "") for r in results)

    @pytest.mark.usefixtures("sample_elements")
    def test_search_by_regexp_no_match(self, repo):
        """Test regexp with no matches."""
        results = repo.search_by_regexp(
            pattern="nonexistent_function",
            scope="test-pattern",
            repository="repo",
        )
        assert len(results) == 0

    @pytest.mark.usefixtures("sample_elements")
    def test_search_by_wildcard(self, repo):
        """Test wildcard pattern search."""
        results = repo.search_by_wildcard(
            pattern="*column*Model*",
            scope="test-pattern",
            repository="repo",
        )
        assert len(results) >= 1
        assert any("add_column" in r.get("name", "") for r in results)

    @pytest.mark.usefixtures("sample_elements")
    def test_search_by_wildcard_question_mark(self, repo):
        """Test wildcard with ? for single character."""
        results = repo.search_by_wildcard(
            pattern="*proce??*",
            scope="test-pattern",
            repository="repo",
        )
        assert len(results) >= 1

    @pytest.mark.usefixtures("sample_elements")
    def test_search_by_proximity(self, repo):
        """Test proximity search with slop."""
        # Search for terms that appear near each other in the raw_code
        # "def add_column(table, Model)" - "table" and "Model" are close
        results = repo.search_by_proximity(
            terms="table Model",
            slop=3,
            scope="test-pattern",
            repository="repo",
        )
        assert len(results) >= 1

    @pytest.mark.usefixtures("sample_elements")
    def test_search_by_proximity_exact_phrase(self, repo):
        """Test proximity search with slop=0 for exact phrase."""
        results = repo.search_by_proximity(
            terms="def process",
            slop=0,
            scope="test-pattern",
            repository="repo",
        )
        assert len(results) >= 1
