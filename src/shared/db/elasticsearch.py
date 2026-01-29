"""Elasticsearch repository implementation for Magaldi.

Handles indexing of code elements and storage of embedding vectors.

This module re-exports the refactored ElasticsearchRepository from
the repositories subpackage for backward compatibility.
"""

from __future__ import annotations

from typing import Any

from magaldi_core.code_parser import CodeElement
from shared.config import MagaldiConfig

# Re-export from the repositories subpackage
from shared.db.repositories import (
    INDEX_MAPPING,
    INDEX_NAME,
    ElasticsearchBase,
    ElasticsearchRepository,
    generate_hash_id,
)


class ElasticsearchFileStateRepository:
    """Elasticsearch-based file state repository for change detection."""

    def __init__(self, config: MagaldiConfig | None = None):
        self._es = ElasticsearchRepository(config)
        self._config = config

    def get_file_states(
        self, scope: str, repository: str, username: str
    ) -> dict[str, Any]:
        """Get all file states for a scope/repo/user.

        Returns dict mapping relative_path to DBFileState-like dict.

        IMPORTANT: Verifies completeness - if expected element_count doesn't match
        actual count of elements with that file_hash, returns None for file_hash
        so the file will be treated as needing reprocessing.
        """
        from magaldi_core.change_detection import DBFileState

        es_states = self._es.get_file_states(scope, repository, username)
        result = {}

        for path, state in es_states.items():
            file_hash = state.get("file_hash")
            element_count = state.get("element_count")

            # Verify completeness when we have a file_hash
            if file_hash:
                if element_count is None:
                    # Old data without element_count - treat as incomplete
                    # This forces reindexing of data from before completeness tracking
                    file_hash = None
                else:
                    # Check that actual element count matches expected
                    # Use count_elements_by_path (not by_hash) because multiple files
                    # can have identical content (same hash)
                    actual_count = self._es.count_elements_by_path(
                        scope, repository, username, path
                    )
                    if actual_count != element_count:
                        # Incomplete - set file_hash to None so it's treated as modified
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
        return self._es.main_branch_exists(scope, repository)

    def close(self) -> None:
        """Close ES connection."""
        self._es.close()


class ElasticsearchEmbeddingStore(ElasticsearchRepository):
    """Elasticsearch-backed embedding store (ES only, no MySQL)."""

    def __init__(self, config: MagaldiConfig | None = None):
        super().__init__(config)
        # Cache for elements stored in this session
        # Named _elements_cache to avoid conflict with _elements repository
        self._elements_cache: dict[str, CodeElement] = {}

    def store_element(self, element: CodeElement, file_hash: str | None = None) -> None:
        """Store a code element (index to ES)."""
        self._elements_cache[element.element_id] = element
        self.index_element(element, file_hash=file_hash)

    def get_element(self, element_id: str) -> CodeElement | None:
        """Get element by ID (from cache or reconstruct from ES)."""
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
        """Get summary from ES."""
        doc = self.get_document(element_id)
        if doc:
            return doc.get("summary")
        return None

    def get_file_summary(self, element: CodeElement) -> str | None:
        """Get file summary from ES."""
        # Build file element ID
        file_id = f"{element.scope}:{element.repository}:{element.username}:{element.relative_path}:file:{element.relative_path.split('/')[-1]}:1"
        return self.get_summary(file_id)

    def get_class_summary(self, element: CodeElement) -> str | None:
        """Get class summary from ES (via parent_id chain)."""
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


# Export all public names for backward compatibility
__all__ = [
    "ElasticsearchRepository",
    "ElasticsearchBase",
    "ElasticsearchFileStateRepository",
    "ElasticsearchEmbeddingStore",
    "INDEX_NAME",
    "INDEX_MAPPING",
    "generate_hash_id",
]
