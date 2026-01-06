"""Tests for the storage module (Phase 4)."""

from datetime import datetime
from pathlib import Path

import pytest

from magaldi.parser.change_detection import ChangeManifest, FileInfo
from magaldi.parser.code_parser import CodeElement, ParsedFile, ParsingResult
from magaldi.storage import (
    InMemoryDatabaseRepository,
    InMemorySearchRepository,
    StorageError,
    StorageResult,
    create_embedding_jobs,
    create_summarization_jobs,
    handle_deletions,
    index_elements,
    should_embed,
    store_parsed_files,
    store_storage_result,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_file_info() -> FileInfo:
    return FileInfo(
        relative_path="src/app.py",
        absolute_path=Path("/repo/src/app.py"),
        language="python",
        hash="abc123",
    )


@pytest.fixture
def sample_element() -> CodeElement:
    return CodeElement(
        element_id="scope:repo:main:src/app.py:function:my_func:10",
        scope="scope",
        repository="repo",
        username="main",
        relative_path="src/app.py",
        element_type="function",
        name="my_func",
        language="python",
        line_start=10,
        line_end=20,
        raw_code="def my_func(): pass",
        signature="def my_func()",
        docstring="A function.",
        level=2,
    )


@pytest.fixture
def sample_parsed_file(sample_file_info: FileInfo, sample_element: CodeElement) -> ParsedFile:
    file_element = CodeElement(
        element_id="scope:repo:main:src/app.py:file:app.py:1",
        scope="scope",
        repository="repo",
        username="main",
        relative_path="src/app.py",
        element_type="file",
        name="app.py",
        language="python",
        line_start=1,
        line_end=30,
        level=0,
    )
    sample_element.parent_id = file_element.element_id
    return ParsedFile(
        file_info=sample_file_info,
        elements=[file_element, sample_element],
        line_count=30,
    )


@pytest.fixture
def sample_parsing_result(sample_parsed_file: ParsedFile) -> ParsingResult:
    result = ParsingResult(scope="scope", repository="repo", username="main")
    result.parsed_files = [sample_parsed_file]
    return result


@pytest.fixture
def sample_manifest(sample_file_info: FileInfo) -> ChangeManifest:
    return ChangeManifest(
        scope="scope",
        repository="repo",
        username="main",
        timestamp=datetime.now(),
        total_files_scanned=1,
        new_files=[sample_file_info],
    )


@pytest.fixture
def db_repo() -> InMemoryDatabaseRepository:
    return InMemoryDatabaseRepository()


@pytest.fixture
def search_repo() -> InMemorySearchRepository:
    return InMemorySearchRepository()


# =============================================================================
# IN-MEMORY REPOSITORY TESTS
# =============================================================================


class TestInMemoryDatabaseRepository:
    """Tests for in-memory database repository."""

    def test_store_file_state(self, db_repo: InMemoryDatabaseRepository):
        db_repo.store_file_state(
            scope="scope",
            repository="repo",
            username="main",
            relative_path="app.py",
            file_hash="hash123",
            language="python",
            file_size=1024,
            line_count=50,
        )

        state = db_repo.get_file_state("scope", "repo", "main", "app.py")
        assert state is not None
        assert state["file_hash"] == "hash123"
        assert state["language"] == "python"

    def test_store_element(self, db_repo: InMemoryDatabaseRepository, sample_element: CodeElement):
        db_repo.store_element(sample_element)

        elem = db_repo.get_element(sample_element.element_id)
        assert elem is not None
        assert elem.name == "my_func"

    def test_delete_elements_by_file(
        self, db_repo: InMemoryDatabaseRepository, sample_element: CodeElement
    ):
        db_repo.store_element(sample_element)
        assert db_repo.get_element(sample_element.element_id) is not None

        db_repo.delete_elements_by_file("scope", "repo", "main", "src/app.py")
        assert db_repo.get_element(sample_element.element_id) is None

    def test_store_summarization_job(self, db_repo: InMemoryDatabaseRepository):
        db_repo.store_summarization_job(
            element_id="test:id",
            level=1,
            parent_id="parent:id",
            dependencies_met=False,
            priority=50,
        )

        job = db_repo.get_summarization_job("test:id")
        assert job is not None
        assert job["level"] == 1
        assert job["dependencies_met"] is False

    def test_store_embedding_job(self, db_repo: InMemoryDatabaseRepository):
        db_repo.store_embedding_job(element_id="test:id")

        job = db_repo.get_embedding_job("test:id")
        assert job is not None
        assert job["status"] == "pending"


class TestInMemorySearchRepository:
    """Tests for in-memory search repository."""

    def test_index_element(
        self, search_repo: InMemorySearchRepository, sample_element: CodeElement
    ):
        search_repo.index_element(sample_element)

        doc = search_repo.get_document(sample_element.element_id)
        assert doc is not None
        assert doc["name"] == "my_func"

    def test_delete_by_file(
        self, search_repo: InMemorySearchRepository, sample_element: CodeElement
    ):
        search_repo.index_element(sample_element)
        assert search_repo.get_document(sample_element.element_id) is not None

        search_repo.delete_by_file("scope", "repo", "main", "src/app.py")
        assert search_repo.get_document(sample_element.element_id) is None


# =============================================================================
# HANDLE DELETIONS
# =============================================================================


class TestHandleDeletions:
    """Tests for deletion handling."""

    def test_deletes_removed_files(
        self, db_repo: InMemoryDatabaseRepository, search_repo: InMemorySearchRepository
    ):
        # Store element for file to be deleted
        elem = CodeElement(
            element_id="scope:repo:main:deleted.py:function:foo:1",
            scope="scope",
            repository="repo",
            username="main",
            relative_path="deleted.py",
            element_type="function",
            name="foo",
            line_start=1,
            level=2,
        )
        db_repo.store_element(elem)
        search_repo.index_element(elem)

        # Create manifest with deleted file
        manifest = ChangeManifest(
            scope="scope",
            repository="repo",
            username="main",
            timestamp=datetime.now(),
            total_files_scanned=0,
            deleted_files=[
                FileInfo("deleted.py", Path("/repo/deleted.py"), "python", previous_hash="old")
            ],
        )

        handle_deletions(manifest, db_repo, search_repo)

        assert db_repo.get_element(elem.element_id) is None
        assert search_repo.get_document(elem.element_id) is None

    def test_deletes_modified_file_elements(
        self, db_repo: InMemoryDatabaseRepository, search_repo: InMemorySearchRepository
    ):
        # Store old element
        elem = CodeElement(
            element_id="scope:repo:main:modified.py:function:old:1",
            scope="scope",
            repository="repo",
            username="main",
            relative_path="modified.py",
            element_type="function",
            name="old",
            line_start=1,
            level=2,
        )
        db_repo.store_element(elem)
        search_repo.index_element(elem)

        # Create manifest with modified file
        manifest = ChangeManifest(
            scope="scope",
            repository="repo",
            username="main",
            timestamp=datetime.now(),
            total_files_scanned=1,
            modified_files=[
                FileInfo(
                    "modified.py", Path("/repo/modified.py"), "python", hash="new", previous_hash="old"
                )
            ],
        )

        handle_deletions(manifest, db_repo, search_repo)

        assert db_repo.get_element(elem.element_id) is None

    def test_marks_file_deleted_for_user_branch(
        self, db_repo: InMemoryDatabaseRepository, search_repo: InMemorySearchRepository
    ):
        manifest = ChangeManifest(
            scope="scope",
            repository="repo",
            username="alice",
            timestamp=datetime.now(),
            total_files_scanned=0,
            deleted_files=[FileInfo("removed.py", Path(), "python")],
        )

        handle_deletions(manifest, db_repo, search_repo)

        state = db_repo.get_file_state("scope", "repo", "alice", "removed.py")
        assert state is not None
        assert state["is_deleted"] is True


# =============================================================================
# STORE PARSED FILES
# =============================================================================


class TestStoreParsedFiles:
    """Tests for storing parsed files."""

    def test_stores_file_state(
        self, sample_parsed_file: ParsedFile, db_repo: InMemoryDatabaseRepository
    ):
        store_parsed_files([sample_parsed_file], "main", db_repo)

        state = db_repo.get_file_state("scope", "repo", "main", "src/app.py")
        assert state is not None
        assert state["line_count"] == 30

    def test_stores_elements(
        self, sample_parsed_file: ParsedFile, db_repo: InMemoryDatabaseRepository
    ):
        store_parsed_files([sample_parsed_file], "main", db_repo)

        # Check file element
        file_elem = db_repo.get_element("scope:repo:main:src/app.py:file:app.py:1")
        assert file_elem is not None
        assert file_elem.element_type == "file"

        # Check function element
        func_elem = db_repo.get_element("scope:repo:main:src/app.py:function:my_func:10")
        assert func_elem is not None
        assert func_elem.name == "my_func"

    def test_sets_expiry_for_user_branch(
        self, sample_parsed_file: ParsedFile, db_repo: InMemoryDatabaseRepository
    ):
        # Change to user branch
        sample_parsed_file.file_info.relative_path = "src/app.py"
        for elem in sample_parsed_file.elements:
            elem.username = "alice"

        store_parsed_files([sample_parsed_file], "alice", db_repo)

        state = db_repo.get_file_state("scope", "repo", "alice", "src/app.py")
        assert state is not None
        assert state.get("expires_at") is not None


# =============================================================================
# INDEX ELEMENTS
# =============================================================================


class TestIndexElements:
    """Tests for Elasticsearch indexing."""

    def test_indexes_all_elements(
        self, sample_parsed_file: ParsedFile, search_repo: InMemorySearchRepository
    ):
        success = index_elements([sample_parsed_file], search_repo)

        assert success == 2  # file + function

        for elem in sample_parsed_file.elements:
            doc = search_repo.get_document(elem.element_id)
            assert doc is not None

    def test_sets_indexed_at(
        self, sample_parsed_file: ParsedFile, search_repo: InMemorySearchRepository
    ):
        index_elements([sample_parsed_file], search_repo)

        doc = search_repo.get_document(sample_parsed_file.elements[0].element_id)
        assert "indexed_at" in doc

    def test_stores_file_hash_on_file_elements(
        self, sample_parsed_file: ParsedFile, search_repo: InMemorySearchRepository
    ):
        """Test that file_hash is stored on file-level elements."""
        index_elements([sample_parsed_file], search_repo)

        # Find the file element
        file_elem = next(e for e in sample_parsed_file.elements if e.element_type == "file")
        doc = search_repo.get_document(file_elem.element_id)

        assert doc is not None
        assert "file_hash" in doc
        assert doc["file_hash"] == sample_parsed_file.file_info.hash


# =============================================================================
# CREATE JOBS
# =============================================================================


class TestCreateSummarizationJobs:
    """Tests for summarization job creation."""

    def test_creates_jobs_for_elements(
        self, sample_parsed_file: ParsedFile, db_repo: InMemoryDatabaseRepository
    ):
        create_summarization_jobs(sample_parsed_file.elements, db_repo)

        # Check file job (level 0)
        file_job = db_repo.get_summarization_job("scope:repo:main:src/app.py:file:app.py:1")
        assert file_job is not None
        assert file_job["level"] == 0
        assert file_job["dependencies_met"] is True  # No parent dependency

        # Check function job (level 2)
        func_job = db_repo.get_summarization_job("scope:repo:main:src/app.py:function:my_func:10")
        assert func_job is not None
        assert func_job["level"] == 2
        assert func_job["dependencies_met"] is False  # Depends on file

    def test_sets_priority_by_level(
        self, sample_parsed_file: ParsedFile, db_repo: InMemoryDatabaseRepository
    ):
        create_summarization_jobs(sample_parsed_file.elements, db_repo)

        file_job = db_repo.get_summarization_job("scope:repo:main:src/app.py:file:app.py:1")
        func_job = db_repo.get_summarization_job("scope:repo:main:src/app.py:function:my_func:10")

        # File (level 0) should have higher priority than function (level 2)
        assert file_job["priority"] > func_job["priority"]


class TestCreateEmbeddingJobs:
    """Tests for embedding job creation."""

    def test_creates_jobs_for_embeddable_elements(self, db_repo: InMemoryDatabaseRepository):
        elements = [
            CodeElement(element_id="e1", element_type="file", name="app.py", level=0),
            CodeElement(element_id="e2", element_type="class", name="MyClass", level=1),
            CodeElement(element_id="e3", element_type="function", name="my_func", level=2),
            CodeElement(element_id="e4", element_type="variable", name="CONFIG", level=3),
        ]

        create_embedding_jobs(elements, db_repo)

        assert db_repo.get_embedding_job("e1") is not None  # file
        assert db_repo.get_embedding_job("e2") is not None  # class
        assert db_repo.get_embedding_job("e3") is not None  # function
        assert db_repo.get_embedding_job("e4") is not None  # CONFIG (uppercase)


class TestShouldEmbed:
    """Tests for embedding eligibility."""

    def test_always_embed_files(self):
        elem = CodeElement(element_type="file", name="app.py")
        assert should_embed(elem) is True

    def test_always_embed_classes(self):
        elem = CodeElement(element_type="class", name="MyClass")
        assert should_embed(elem) is True

    def test_always_embed_functions(self):
        elem = CodeElement(element_type="function", name="my_func")
        assert should_embed(elem) is True

    def test_always_embed_methods(self):
        elem = CodeElement(element_type="method", name="do_thing")
        assert should_embed(elem) is True

    def test_embed_uppercase_variables(self):
        elem = CodeElement(element_type="variable", name="CONFIG_VALUE")
        assert should_embed(elem) is True

    def test_embed_variables_with_docstring(self):
        elem = CodeElement(element_type="variable", name="setting", docstring="Important setting")
        assert should_embed(elem) is True

    def test_skip_private_variables(self):
        elem = CodeElement(element_type="variable", name="_private")
        assert should_embed(elem) is False

    def test_skip_regular_lowercase_variables(self):
        elem = CodeElement(element_type="variable", name="temp_value")
        assert should_embed(elem) is False


# =============================================================================
# FULL STORAGE FLOW
# =============================================================================


class TestStoreStorageResult:
    """Tests for the full storage flow."""

    def test_full_storage_flow(
        self,
        sample_parsing_result: ParsingResult,
        sample_manifest: ChangeManifest,
        db_repo: InMemoryDatabaseRepository,
        search_repo: InMemorySearchRepository,
    ):
        result = store_storage_result(
            sample_parsing_result, sample_manifest, db_repo, search_repo
        )

        assert isinstance(result, StorageResult)
        assert result.files_stored == 1
        assert result.elements_stored == 2  # file + function
        assert result.elements_indexed == 2
        assert result.summarization_jobs_created == 2

    def test_reports_errors(
        self,
        sample_parsing_result: ParsingResult,
        sample_manifest: ChangeManifest,
        db_repo: InMemoryDatabaseRepository,
        search_repo: InMemorySearchRepository,
    ):
        # Simulate a failed file
        sample_parsing_result.failed_files = [
            (FileInfo("broken.py", Path(), "python"), "Parse error")
        ]

        result = store_storage_result(
            sample_parsing_result, sample_manifest, db_repo, search_repo
        )

        assert len(result.storage_errors) >= 1

    def test_handles_deletion_first(
        self,
        sample_parsing_result: ParsingResult,
        db_repo: InMemoryDatabaseRepository,
        search_repo: InMemorySearchRepository,
    ):
        # Store old element that should be deleted
        old_elem = CodeElement(
            element_id="scope:repo:main:src/old.py:function:old_func:1",
            scope="scope",
            repository="repo",
            username="main",
            relative_path="src/old.py",
            element_type="function",
            name="old_func",
            line_start=1,
            level=2,
        )
        db_repo.store_element(old_elem)
        search_repo.index_element(old_elem)

        # Create manifest with deleted file
        manifest = ChangeManifest(
            scope="scope",
            repository="repo",
            username="main",
            timestamp=datetime.now(),
            total_files_scanned=1,
            deleted_files=[FileInfo("src/old.py", Path(), "python")],
            new_files=[sample_parsing_result.parsed_files[0].file_info],
        )

        result = store_storage_result(sample_parsing_result, manifest, db_repo, search_repo)

        # Old element should be deleted
        assert db_repo.get_element(old_elem.element_id) is None
        # New elements should be stored
        assert result.elements_stored == 2
        assert result.elements_deleted == 1


# =============================================================================
# DATA CLASSES
# =============================================================================


class TestStorageResult:
    """Tests for StorageResult dataclass."""

    def test_default_values(self):
        result = StorageResult(scope="s", repository="r", username="u")

        assert result.files_stored == 0
        assert result.elements_stored == 0
        assert result.storage_errors == []
        assert result.indexing_errors == []
