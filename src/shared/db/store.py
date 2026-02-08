"""Repository facade and specialized stores for Magaldi.

Handles indexing of code elements and storage of embedding vectors.

This module re-exports the Repository from the repositories subpackage
and provides FileStateRepository and EmbeddingStore.
"""

from __future__ import annotations

from typing import Any

from magaldi_core.code_parser import CodeElement
from shared.config import MagaldiConfig

# Re-export from the repositories subpackage
from shared.db.repositories import (
    INDEX_MAPPING,
    INDEX_NAME,
    Repository,
    RepositoryBase,
    generate_hash_id,
)


class FileStateRepository:
    """File state repository for change detection."""

    def __init__(self, config: MagaldiConfig | None = None):
        self._repo = Repository(config)
        self._config = config

    def get_file_states(
        self, scope: str, repository: str, username: str
    ) -> dict[str, Any]:
        """Get all file states for a scope/repo/user.

        Returns dict mapping relative_path to DBFileState-like dict.

        IMPORTANT: Verifies completeness - returns None for file_hash when:
        - Expected element_count doesn't match actual count (parser drift)
        - Any element in the file is missing required embeddings (migration)
        This causes change detection to treat the file as needing reprocessing.
        """
        from magaldi_core.change_detection import DBFileState

        states = self._repo.get_file_states(scope, repository, username)

        # Find files with elements missing required embeddings (e.g. after
        # adding caller_embedding). These need full re-processing.
        incomplete_paths = set(
            self._repo.find_files_with_incomplete_elements(scope, repository, username)
        )

        result = {}

        for path, state in states.items():
            file_hash = state.get("file_hash")
            element_count = state.get("element_count")

            # If element count mismatches, update it to match actual count.
            # This handles cases where parser changes altered element extraction
            # without changing file content. File hash is the source of truth.
            if file_hash and element_count is not None:
                actual_count = self._repo.count_elements_by_path(
                    scope, repository, username, path
                )
                if actual_count != element_count:
                    # Update the stored element_count to match reality
                    self._repo.update_file_element_count(
                        scope, repository, username, path, actual_count
                    )
                    element_count = actual_count

            # Force re-processing for files with incomplete elements
            if path in incomplete_paths:
                file_hash = None

            result[path] = DBFileState(
                relative_path=path,
                file_hash=file_hash,
                is_deleted=state.get("is_deleted", False),
                element_count=element_count,
            )

        return result

    def main_branch_exists(self, scope: str, repository: str) -> bool:
        """Check if main branch has been parsed."""
        return self._repo.main_branch_exists(scope, repository)

    def close(self) -> None:
        """Close connection."""
        self._repo.close()


class EmbeddingStore(Repository):
    """Embedding store backed by the search backend."""

    def __init__(self, config: MagaldiConfig | None = None):
        super().__init__(config)
        # Cache for elements stored in this session
        # Named _elements_cache to avoid conflict with _elements repository
        self._elements_cache: dict[str, CodeElement] = {}

    def store_element(self, element: CodeElement, file_hash: str | None = None) -> None:
        """Store a code element."""
        self._elements_cache[element.element_id] = element
        self.index_element(element, file_hash=file_hash)

    def get_element(self, element_id: str) -> CodeElement | None:
        """Get element by ID (from cache or reconstruct from store)."""
        if element_id in self._elements_cache:
            return self._elements_cache[element_id]

        doc = self.get_document(element_id)
        if doc is None:
            return None

        return CodeElement(
            element_id=doc["element_id"],
            scope=doc["scope"],
            repository=doc["repository"],
            username=doc["username"],
            relative_path=doc["relative_path"],
            element_type=doc["element_type"],
            name=doc["name"],
            language=doc.get("language", ""),
            line_start=doc["line_start"],
            line_end=doc.get("line_end"),
            raw_code=doc.get("raw_code"),
            signature=doc.get("signature"),
            docstring=doc.get("docstring"),
            level=doc.get("level", 0),
            parent_id=doc.get("parent_id"),
            decorators=doc.get("decorators"),
            visibility=doc.get("visibility"),
            is_async=doc.get("is_async", False),
            is_test=doc.get("is_test", False),
        )

    def get_summary(self, element_id: str) -> str | None:
        """Get summary for an element."""
        doc = self.get_document(element_id)
        if doc:
            return doc.get("summary")
        return None

    def get_file_summary(self, element: CodeElement) -> str | None:
        """Get file summary."""
        # Build file element ID
        file_id = f"{element.scope}:{element.repository}:{element.username}:{element.relative_path}:file:{element.relative_path.split('/')[-1]}:1"
        return self.get_summary(file_id)

    def get_class_summary(self, element: CodeElement) -> str | None:
        """Get class summary (via parent_id chain)."""
        if element.parent_id:
            parent_doc = self.get_document(element.parent_id)
            if parent_doc and parent_doc.get("element_type") == "class":
                return parent_doc.get("summary")
        return None

    def get_parent_summaries(self, element: CodeElement) -> dict[str, str]:
        """Get parent summaries for context."""
        summaries: dict[str, str] = {}

        # Get file summary
        file_summary = self.get_file_summary(element)
        if file_summary:
            summaries["file"] = file_summary

        # Get class summary if method
        if element.element_type == "method":
            class_summary = self.get_class_summary(element)
            if class_summary:
                summaries["class"] = class_summary

        return summaries


# Backward-compatible aliases
ElasticsearchRepository = Repository
ElasticsearchBase = RepositoryBase
ElasticsearchFileStateRepository = FileStateRepository
ElasticsearchEmbeddingStore = EmbeddingStore

__all__ = [
    "Repository",
    "RepositoryBase",
    "FileStateRepository",
    "EmbeddingStore",
    "ElasticsearchRepository",
    "ElasticsearchBase",
    "ElasticsearchFileStateRepository",
    "ElasticsearchEmbeddingStore",
    "INDEX_NAME",
    "INDEX_MAPPING",
    "generate_hash_id",
]
