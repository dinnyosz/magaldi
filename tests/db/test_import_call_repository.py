"""Integration tests for Elasticsearch imports and calls repository.

Tests for storing and querying import and call relationships.
"""

from __future__ import annotations

import pytest

from magaldi_core.code_parser import CodeElement
from shared.db.store import INDEX_NAME

pytestmark = pytest.mark.integration


# =============================================================================
# IMPORTS AND CALLS STORAGE TESTS
# =============================================================================


class TestImportsAndCalls:
    """Tests for storing and retrieving imports and calls."""

    def test_store_and_get_imports(self, repo, sample_element):
        """Test storing and retrieving imports for a file element."""
        # Create a file element for imports
        file_elem = CodeElement(
            element_id="test-es:test-repo:main:src/app.py:file:app.py:1",
            scope="test-es",
            repository="test-repo",
            username="main",
            relative_path="src/app.py",
            element_type="file",
            name="app.py",
            language="python",
            line_start=1,
            line_end=100,
            level=0,
        )
        repo.index_element(file_elem)
        repo._get_client().indices_refresh(index=INDEX_NAME)

        imports = [
            {"name": "os", "module": "os", "alias": None, "line": 1},
            {"name": "json", "module": "json", "alias": None, "line": 2},
            {"name": "Path", "module": "pathlib", "alias": None, "line": 3},
            {"name": "numpy", "module": "numpy", "alias": "np", "line": 4},
        ]

        result = repo.store_imports(file_elem.element_id, imports)
        assert result is True

        repo._get_client().indices_refresh(index=INDEX_NAME)

        retrieved = repo.get_imports(file_elem.element_id)
        assert len(retrieved) == 4
        assert retrieved[0]["name"] == "os"
        assert retrieved[0]["module"] == "os"
        assert retrieved[3]["alias"] == "np"

    def test_store_imports_nonexistent_element(self, repo):
        """Test storing imports for nonexistent element returns False."""
        imports = [{"name": "os", "module": "os", "alias": None, "line": 1}]
        result = repo.store_imports("nonexistent-id", imports)
        assert result is False

    def test_get_imports_nonexistent_element(self, repo):
        """Test getting imports for nonexistent element returns empty list."""
        result = repo.get_imports("nonexistent-id")
        assert result == []

    def test_get_imports_element_without_imports(self, repo, sample_element):
        """Test getting imports for element without imports returns empty list."""
        repo.index_element(sample_element)
        repo._get_client().indices_refresh(index=INDEX_NAME)

        result = repo.get_imports(sample_element.element_id)
        assert result == []

    def test_store_and_get_calls(self, repo, sample_element):
        """Test storing and retrieving calls for a function element."""
        repo.index_element(sample_element)
        repo._get_client().indices_refresh(index=INDEX_NAME)

        calls = [
            {"name": "print", "receiver": None, "line": 15, "resolved_id": None},
            {
                "name": "upper",
                "receiver": "data",
                "line": 16,
                "resolved_id": "test-es:test-repo:main:str:method:upper:1",
            },
            {
                "name": "helper",
                "receiver": "self",
                "line": 17,
                "resolved_id": "test-es:test-repo:main:src/app.py:method:helper:50",
            },
        ]

        result = repo.store_calls(sample_element.element_id, calls)
        assert result is True

        repo._get_client().indices_refresh(index=INDEX_NAME)

        retrieved = repo.get_calls(sample_element.element_id)
        assert len(retrieved) == 3
        assert retrieved[0]["name"] == "print"
        assert retrieved[0]["receiver"] is None
        assert retrieved[1]["receiver"] == "data"
        assert retrieved[2]["resolved_id"] == "test-es:test-repo:main:src/app.py:method:helper:50"

    def test_store_calls_nonexistent_element(self, repo):
        """Test storing calls for nonexistent element returns False."""
        calls = [{"name": "print", "receiver": None, "line": 1, "resolved_id": None}]
        result = repo.store_calls("nonexistent-id", calls)
        assert result is False

    def test_get_calls_nonexistent_element(self, repo):
        """Test getting calls for nonexistent element returns empty list."""
        result = repo.get_calls("nonexistent-id")
        assert result == []

    def test_get_calls_element_without_calls(self, repo, sample_element):
        """Test getting calls for element without calls returns empty list."""
        repo.index_element(sample_element)
        repo._get_client().indices_refresh(index=INDEX_NAME)

        result = repo.get_calls(sample_element.element_id)
        assert result == []

    def test_store_empty_imports(self, repo, sample_element):
        """Test storing empty imports list."""
        repo.index_element(sample_element)
        repo._get_client().indices_refresh(index=INDEX_NAME)

        result = repo.store_imports(sample_element.element_id, [])
        assert result is True

        repo._get_client().indices_refresh(index=INDEX_NAME)

        retrieved = repo.get_imports(sample_element.element_id)
        assert retrieved == []

    def test_store_empty_calls(self, repo, sample_element):
        """Test storing empty calls list."""
        repo.index_element(sample_element)
        repo._get_client().indices_refresh(index=INDEX_NAME)

        result = repo.store_calls(sample_element.element_id, [])
        assert result is True

        repo._get_client().indices_refresh(index=INDEX_NAME)

        retrieved = repo.get_calls(sample_element.element_id)
        assert retrieved == []


# =============================================================================
# FINDING ELEMENTS BY CALLS
# =============================================================================


class TestFindElementsCalling:
    """Tests for finding elements that call a target."""

    @pytest.fixture
    def elements_with_calls(self, repo):
        """Create elements with call relationships."""
        target_id = "test-calls:repo:main:src/utils.py:function:helper:10"

        # Target function
        target = CodeElement(
            element_id=target_id,
            scope="test-calls",
            repository="repo",
            username="main",
            relative_path="src/utils.py",
            element_type="function",
            name="helper",
            language="python",
            line_start=10,
            line_end=20,
            level=2,
        )
        repo.index_element(target)

        # Caller 1
        caller1 = CodeElement(
            element_id="test-calls:repo:main:src/app.py:function:main:1",
            scope="test-calls",
            repository="repo",
            username="main",
            relative_path="src/app.py",
            element_type="function",
            name="main",
            language="python",
            line_start=1,
            line_end=30,
            level=2,
        )
        repo.index_element(caller1)
        repo.store_calls(
            caller1.element_id,
            [{"name": "helper", "receiver": None, "line": 15, "resolved_id": target_id}],
        )

        # Caller 2
        caller2 = CodeElement(
            element_id="test-calls:repo:main:src/service.py:function:process:50",
            scope="test-calls",
            repository="repo",
            username="main",
            relative_path="src/service.py",
            element_type="function",
            name="process",
            language="python",
            line_start=50,
            line_end=80,
            level=2,
        )
        repo.index_element(caller2)
        repo.store_calls(
            caller2.element_id,
            [
                {"name": "helper", "receiver": None, "line": 60, "resolved_id": target_id},
                {"name": "other", "receiver": None, "line": 65, "resolved_id": None},
            ],
        )

        # Non-caller (calls different function)
        non_caller = CodeElement(
            element_id="test-calls:repo:main:src/other.py:function:foo:1",
            scope="test-calls",
            repository="repo",
            username="main",
            relative_path="src/other.py",
            element_type="function",
            name="foo",
            language="python",
            line_start=1,
            line_end=10,
            level=2,
        )
        repo.index_element(non_caller)
        repo.store_calls(
            non_caller.element_id,
            [{"name": "bar", "receiver": None, "line": 5, "resolved_id": "other-target"}],
        )

        repo._get_client().indices_refresh(index=INDEX_NAME)
        return target_id

    def test_find_elements_calling(self, repo, elements_with_calls):
        """Test finding elements that call a target."""
        target_id = elements_with_calls

        results = repo.find_elements_calling(target_id)

        assert len(results) == 2
        names = {r["name"] for r in results}
        assert names == {"main", "process"}

    def test_find_elements_calling_with_scope_filter(self, repo, elements_with_calls):
        """Test finding callers with scope filter."""
        target_id = elements_with_calls

        # Matching scope
        results = repo.find_elements_calling(target_id, scope="test-calls")
        assert len(results) == 2

        # Non-matching scope
        results = repo.find_elements_calling(target_id, scope="other-scope")
        assert len(results) == 0

    def test_find_elements_calling_with_repository_filter(self, repo, elements_with_calls):
        """Test finding callers with repository filter."""
        target_id = elements_with_calls

        # Matching repository
        results = repo.find_elements_calling(target_id, repository="repo")
        assert len(results) == 2

        # Non-matching repository
        results = repo.find_elements_calling(target_id, repository="other-repo")
        assert len(results) == 0

    def test_find_elements_calling_no_callers(self, repo, elements_with_calls):
        """Test finding callers for target with no callers."""
        results = repo.find_elements_calling("nonexistent-target")
        assert len(results) == 0

    def test_find_elements_calling_with_limit(self, repo, elements_with_calls):
        """Test finding callers with limit."""
        target_id = elements_with_calls

        results = repo.find_elements_calling(target_id, limit=1)
        assert len(results) == 1


# =============================================================================
# FINDING ELEMENTS BY IMPORTS
# =============================================================================


class TestFindElementsImporting:
    """Tests for finding elements that import a module."""

    @pytest.fixture
    def elements_with_imports(self, repo):
        """Create file elements with imports."""
        # File 1 imports os and json
        file1 = CodeElement(
            element_id="test-imports:repo:main:src/app.py:file:app.py:1",
            scope="test-imports",
            repository="repo",
            username="main",
            relative_path="src/app.py",
            element_type="file",
            name="app.py",
            language="python",
            line_start=1,
            line_end=100,
            level=0,
        )
        repo.index_element(file1)
        repo.store_imports(
            file1.element_id,
            [
                {"name": "os", "module": "os", "alias": None, "line": 1},
                {"name": "json", "module": "json", "alias": None, "line": 2},
            ],
        )

        # File 2 imports os and pathlib
        file2 = CodeElement(
            element_id="test-imports:repo:main:src/utils.py:file:utils.py:1",
            scope="test-imports",
            repository="repo",
            username="main",
            relative_path="src/utils.py",
            element_type="file",
            name="utils.py",
            language="python",
            line_start=1,
            line_end=50,
            level=0,
        )
        repo.index_element(file2)
        repo.store_imports(
            file2.element_id,
            [
                {"name": "os", "module": "os", "alias": None, "line": 1},
                {"name": "Path", "module": "pathlib", "alias": None, "line": 2},
            ],
        )

        # File 3 imports only re
        file3 = CodeElement(
            element_id="test-imports:repo:main:src/parser.py:file:parser.py:1",
            scope="test-imports",
            repository="repo",
            username="main",
            relative_path="src/parser.py",
            element_type="file",
            name="parser.py",
            language="python",
            line_start=1,
            line_end=80,
            level=0,
        )
        repo.index_element(file3)
        repo.store_imports(
            file3.element_id,
            [{"name": "re", "module": "re", "alias": None, "line": 1}],
        )

        repo._get_client().indices_refresh(index=INDEX_NAME)

    def test_find_elements_importing(self, repo, elements_with_imports):
        """Test finding elements that import a module."""
        # Both app.py and utils.py import os (filter by scope to avoid other test data)
        results = repo.find_elements_importing("os", scope="test-imports")
        assert len(results) == 2
        names = {r["name"] for r in results}
        assert names == {"app.py", "utils.py"}

    def test_find_elements_importing_single_importer(self, repo, elements_with_imports):
        """Test finding elements when only one imports the module."""
        results = repo.find_elements_importing("json", scope="test-imports")
        assert len(results) == 1
        assert results[0]["name"] == "app.py"

    def test_find_elements_importing_with_scope_filter(self, repo, elements_with_imports):
        """Test finding importers with scope filter."""
        # Matching scope
        results = repo.find_elements_importing("os", scope="test-imports")
        assert len(results) == 2

        # Non-matching scope
        results = repo.find_elements_importing("os", scope="other-scope")
        assert len(results) == 0

    def test_find_elements_importing_with_repository_filter(self, repo, elements_with_imports):
        """Test finding importers with repository filter."""
        # Matching repository
        results = repo.find_elements_importing("os", repository="repo")
        assert len(results) == 2

        # Non-matching repository
        results = repo.find_elements_importing("os", repository="other-repo")
        assert len(results) == 0

    def test_find_elements_importing_no_importers(self, repo, elements_with_imports):
        """Test finding importers for module no one imports."""
        results = repo.find_elements_importing("nonexistent_module")
        assert len(results) == 0

    def test_find_elements_importing_with_limit(self, repo, elements_with_imports):
        """Test finding importers with limit."""
        results = repo.find_elements_importing("os", limit=1)
        assert len(results) == 1
