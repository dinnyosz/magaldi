"""Phase 4: Storage - Persist elements to MySQL and Elasticsearch.

This module handles:
1. Handle deletions - Remove old elements before inserting new
2. Store to MySQL - File states and code elements
3. Index to Elasticsearch - For semantic search
4. Create jobs - Summarization and embedding jobs
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

from magaldi.parser.change_detection import ChangeManifest, FileInfo
from magaldi.parser.code_parser import CodeElement, ParsedFile, ParsingResult


class StorageError(Exception):
    """Raised when storage operations fail."""

    pass


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class StorageResult:
    """Result of the storage phase."""

    scope: str
    repository: str
    username: str

    # Counts
    files_stored: int = 0
    elements_stored: int = 0
    elements_indexed: int = 0
    elements_deleted: int = 0

    # Jobs created
    summarization_jobs_created: int = 0
    embedding_jobs_created: int = 0

    # Errors
    storage_errors: list[str] = field(default_factory=list)
    indexing_errors: list[str] = field(default_factory=list)


# =============================================================================
# DATABASE REPOSITORY PROTOCOL
# =============================================================================


class DatabaseRepository(Protocol):
    """Interface for database operations."""

    def store_file_state(
        self,
        scope: str,
        repository: str,
        username: str,
        relative_path: str,
        file_hash: str | None,
        language: str,
        file_size: int,
        line_count: int,
        is_deleted: bool = False,
        expires_at: datetime | None = None,
    ) -> None:
        """Store or update file state."""
        ...

    def get_file_state(
        self, scope: str, repository: str, username: str, relative_path: str
    ) -> dict[str, Any] | None:
        """Get file state."""
        ...

    def store_element(self, element: CodeElement) -> None:
        """Store a code element."""
        ...

    def get_element(self, element_id: str) -> CodeElement | None:
        """Get a code element by ID."""
        ...

    def delete_elements_by_file(
        self, scope: str, repository: str, username: str, relative_path: str
    ) -> int:
        """Delete all elements for a file. Returns count deleted."""
        ...

    def store_summarization_job(
        self,
        element_id: str,
        level: int,
        parent_id: str | None,
        dependencies_met: bool,
        priority: int,
    ) -> None:
        """Create a summarization job."""
        ...

    def get_summarization_job(self, element_id: str) -> dict[str, Any] | None:
        """Get summarization job by element ID."""
        ...

    def store_embedding_job(self, element_id: str) -> None:
        """Create an embedding job."""
        ...

    def get_embedding_job(self, element_id: str) -> dict[str, Any] | None:
        """Get embedding job by element ID."""
        ...


class SearchRepository(Protocol):
    """Interface for search/indexing operations."""

    def index_element(self, element: CodeElement, indexed_at: datetime | None = None) -> bool:
        """Index an element. Returns True on success."""
        ...

    def get_document(self, element_id: str) -> dict[str, Any] | None:
        """Get indexed document by ID."""
        ...

    def delete_by_file(
        self, scope: str, repository: str, username: str, relative_path: str
    ) -> int:
        """Delete all documents for a file. Returns count deleted."""
        ...


# =============================================================================
# IN-MEMORY IMPLEMENTATIONS (for testing)
# =============================================================================


class InMemoryDatabaseRepository:
    """In-memory implementation of database repository for testing."""

    def __init__(self) -> None:
        self._file_states: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        self._elements: dict[str, CodeElement] = {}
        self._summarization_jobs: dict[str, dict[str, Any]] = {}
        self._embedding_jobs: dict[str, dict[str, Any]] = {}

    def store_file_state(
        self,
        scope: str,
        repository: str,
        username: str,
        relative_path: str,
        file_hash: str | None = None,
        language: str = "",
        file_size: int = 0,
        line_count: int = 0,
        is_deleted: bool = False,
        expires_at: datetime | None = None,
    ) -> None:
        key = (scope, repository, username, relative_path)
        self._file_states[key] = {
            "scope": scope,
            "repository": repository,
            "username": username,
            "relative_path": relative_path,
            "file_hash": file_hash,
            "language": language,
            "file_size": file_size,
            "line_count": line_count,
            "is_deleted": is_deleted,
            "expires_at": expires_at,
            "parsed_at": datetime.now(),
        }

    def get_file_state(
        self, scope: str, repository: str, username: str, relative_path: str
    ) -> dict[str, Any] | None:
        key = (scope, repository, username, relative_path)
        return self._file_states.get(key)

    def store_element(self, element: CodeElement) -> None:
        self._elements[element.element_id] = element

    def get_element(self, element_id: str) -> CodeElement | None:
        return self._elements.get(element_id)

    def delete_elements_by_file(
        self, scope: str, repository: str, username: str, relative_path: str
    ) -> int:
        to_delete = [
            eid
            for eid, elem in self._elements.items()
            if (
                elem.scope == scope
                and elem.repository == repository
                and elem.username == username
                and elem.relative_path == relative_path
            )
        ]
        for eid in to_delete:
            del self._elements[eid]

        # Also delete related jobs
        job_prefix = f"{scope}:{repository}:{username}:{relative_path}:"
        sum_jobs_to_delete = [j for j in self._summarization_jobs if j.startswith(job_prefix)]
        for jid in sum_jobs_to_delete:
            del self._summarization_jobs[jid]

        emb_jobs_to_delete = [j for j in self._embedding_jobs if j.startswith(job_prefix)]
        for jid in emb_jobs_to_delete:
            del self._embedding_jobs[jid]

        return len(to_delete)

    def store_summarization_job(
        self,
        element_id: str,
        level: int,
        parent_id: str | None,
        dependencies_met: bool,
        priority: int,
    ) -> None:
        self._summarization_jobs[element_id] = {
            "element_id": element_id,
            "level": level,
            "parent_id": parent_id,
            "dependencies_met": dependencies_met,
            "priority": priority,
            "status": "pending",
            "created_at": datetime.now(),
        }

    def get_summarization_job(self, element_id: str) -> dict[str, Any] | None:
        return self._summarization_jobs.get(element_id)

    def store_embedding_job(self, element_id: str) -> None:
        self._embedding_jobs[element_id] = {
            "element_id": element_id,
            "status": "pending",
            "created_at": datetime.now(),
        }

    def get_embedding_job(self, element_id: str) -> dict[str, Any] | None:
        return self._embedding_jobs.get(element_id)


class InMemorySearchRepository:
    """In-memory implementation of search repository for testing."""

    def __init__(self) -> None:
        self._documents: dict[str, dict[str, Any]] = {}

    def index_element(
        self, element: CodeElement, indexed_at: datetime | None = None
    ) -> bool:
        if indexed_at is None:
            indexed_at = datetime.now()

        self._documents[element.element_id] = {
            "element_id": element.element_id,
            "scope": element.scope,
            "repository": element.repository,
            "username": element.username,
            "relative_path": element.relative_path,
            "element_type": element.element_type,
            "name": element.name,
            "language": element.language,
            "line_start": element.line_start,
            "line_end": element.line_end,
            "raw_code": element.raw_code,
            "signature": element.signature,
            "docstring": element.docstring,
            "level": element.level,
            "parent_id": element.parent_id,
            "decorators": element.decorators,
            "visibility": element.visibility,
            "is_async": element.is_async,
            "indexed_at": indexed_at.isoformat(),
        }
        return True

    def get_document(self, element_id: str) -> dict[str, Any] | None:
        return self._documents.get(element_id)

    def delete_by_file(
        self, scope: str, repository: str, username: str, relative_path: str
    ) -> int:
        to_delete = [
            eid
            for eid, doc in self._documents.items()
            if (
                doc.get("scope") == scope
                and doc.get("repository") == repository
                and doc.get("username") == username
                and doc.get("relative_path") == relative_path
            )
        ]
        for eid in to_delete:
            del self._documents[eid]
        return len(to_delete)


# =============================================================================
# 4.1 HANDLE DELETIONS
# =============================================================================


def handle_deletions(
    manifest: ChangeManifest,
    db_repo: DatabaseRepository,
    search_repo: SearchRepository,
) -> int:
    """Handle all deletion scenarios.

    Args:
        manifest: Change manifest with deleted/modified files.
        db_repo: Database repository.
        search_repo: Search repository.

    Returns:
        Total count of elements deleted.
    """
    total_deleted = 0

    # 1. Handle deleted files
    for file_info in manifest.deleted_files:
        # Delete elements from both DB and search
        count = db_repo.delete_elements_by_file(
            manifest.scope, manifest.repository, manifest.username, file_info.relative_path
        )
        search_repo.delete_by_file(
            manifest.scope, manifest.repository, manifest.username, file_info.relative_path
        )
        total_deleted += count

        # Mark file as deleted (for user branches) or remove state (for main)
        if manifest.username != "main":
            # User branch: mark as deleted
            db_repo.store_file_state(
                scope=manifest.scope,
                repository=manifest.repository,
                username=manifest.username,
                relative_path=file_info.relative_path,
                file_hash=None,
                language="",
                file_size=0,
                line_count=0,
                is_deleted=True,
                expires_at=datetime.now() + timedelta(days=30),
            )

    # 2. Handle modified files (delete old elements before new ones are inserted)
    for file_info in manifest.modified_files:
        count = db_repo.delete_elements_by_file(
            manifest.scope, manifest.repository, manifest.username, file_info.relative_path
        )
        search_repo.delete_by_file(
            manifest.scope, manifest.repository, manifest.username, file_info.relative_path
        )
        total_deleted += count

    return total_deleted


# =============================================================================
# 4.2 STORE TO MYSQL
# =============================================================================


def store_parsed_files(
    parsed_files: list[ParsedFile],
    username: str,
    db_repo: DatabaseRepository,
) -> tuple[int, int]:
    """Store parsed files to database.

    Args:
        parsed_files: List of parsed files.
        username: Username (branch).
        db_repo: Database repository.

    Returns:
        Tuple of (files_stored, elements_stored).
    """
    files_stored = 0
    elements_stored = 0

    # Calculate expiry for user branches
    expires_at = None if username == "main" else datetime.now() + timedelta(days=30)

    for parsed_file in parsed_files:
        file_info = parsed_file.file_info

        # Store file state
        db_repo.store_file_state(
            scope=parsed_file.elements[0].scope if parsed_file.elements else "",
            repository=parsed_file.elements[0].repository if parsed_file.elements else "",
            username=username,
            relative_path=file_info.relative_path,
            file_hash=file_info.hash,
            language=file_info.language,
            file_size=file_info.file_size,
            line_count=parsed_file.line_count,
            is_deleted=False,
            expires_at=expires_at,
        )
        files_stored += 1

        # Store elements (parents first to satisfy foreign key constraints)
        # Sort by level to ensure parents are inserted before children
        sorted_elements = sorted(parsed_file.elements, key=lambda e: e.level)

        for element in sorted_elements:
            db_repo.store_element(element)
            elements_stored += 1

    return files_stored, elements_stored


# =============================================================================
# 4.3 INDEX TO ELASTICSEARCH
# =============================================================================


def index_elements(
    elements: list[CodeElement],
    search_repo: SearchRepository,
) -> int:
    """Index elements to search repository.

    Args:
        elements: List of code elements.
        search_repo: Search repository.

    Returns:
        Count of successfully indexed elements.
    """
    indexed = 0
    indexed_at = datetime.now()

    for element in elements:
        if search_repo.index_element(element, indexed_at):
            indexed += 1

    return indexed


# =============================================================================
# 4.4 CREATE JOBS
# =============================================================================


def create_summarization_jobs(
    elements: list[CodeElement],
    db_repo: DatabaseRepository,
) -> int:
    """Create summarization jobs for elements.

    Jobs are created with dependencies based on hierarchy:
    - Level 0 (files): No dependencies, dependencies_met=True
    - Level 1+ : Depend on parent, dependencies_met=False

    Args:
        elements: List of code elements.
        db_repo: Database repository.

    Returns:
        Count of jobs created.
    """
    count = 0

    for element in elements:
        # Level 0 (files) have no dependencies
        dependencies_met = element.level == 0

        # Priority: higher level = lower priority (100 - level)
        priority = 100 - element.level

        db_repo.store_summarization_job(
            element_id=element.element_id,
            level=element.level,
            parent_id=element.parent_id,
            dependencies_met=dependencies_met,
            priority=priority,
        )
        count += 1

    return count


def should_embed(element: CodeElement) -> bool:
    """Determine if an element should be embedded.

    Args:
        element: Code element to check.

    Returns:
        True if element should be embedded.
    """
    # Always embed files, classes, functions, methods
    if element.element_type in ("file", "class", "function", "method"):
        return True

    # For variables: only embed significant ones
    if element.element_type == "variable":
        name = element.name

        # Embed uppercase constants (CONFIG, MAX_SIZE, etc.)
        if name.isupper():
            return True

        # Embed if has docstring
        if element.docstring:
            return True

        # Skip private variables
        if name.startswith("_"):
            return False

    return False


def create_embedding_jobs(
    elements: list[CodeElement],
    db_repo: DatabaseRepository,
) -> int:
    """Create embedding jobs for embeddable elements.

    Args:
        elements: List of code elements.
        db_repo: Database repository.

    Returns:
        Count of jobs created.
    """
    count = 0

    for element in elements:
        if should_embed(element):
            db_repo.store_embedding_job(element.element_id)
            count += 1

    return count


# =============================================================================
# MAIN STORAGE FUNCTION
# =============================================================================


def store_storage_result(
    parsing_result: ParsingResult,
    manifest: ChangeManifest,
    db_repo: DatabaseRepository,
    search_repo: SearchRepository,
) -> StorageResult:
    """Execute the full storage pipeline.

    Args:
        parsing_result: Result from Phase 3 parsing.
        manifest: Change manifest from Phase 2.
        db_repo: Database repository.
        search_repo: Search repository.

    Returns:
        StorageResult with counts and any errors.
    """
    result = StorageResult(
        scope=parsing_result.scope,
        repository=parsing_result.repository,
        username=parsing_result.username,
    )

    # Track failed files as errors
    for file_info, error in parsing_result.failed_files:
        result.storage_errors.append(f"{file_info.relative_path}: {error}")

    try:
        # 4.1 Handle deletions first
        result.elements_deleted = handle_deletions(manifest, db_repo, search_repo)

        # 4.2 Store to MySQL
        files_stored, elements_stored = store_parsed_files(
            parsing_result.parsed_files, parsing_result.username, db_repo
        )
        result.files_stored = files_stored
        result.elements_stored = elements_stored

        # 4.3 Index to Elasticsearch
        all_elements = []
        for pf in parsing_result.parsed_files:
            all_elements.extend(pf.elements)

        result.elements_indexed = index_elements(all_elements, search_repo)

        # 4.4 Create jobs
        result.summarization_jobs_created = create_summarization_jobs(all_elements, db_repo)
        result.embedding_jobs_created = create_embedding_jobs(all_elements, db_repo)

    except Exception as e:
        result.storage_errors.append(f"Storage error: {e}")

    return result
