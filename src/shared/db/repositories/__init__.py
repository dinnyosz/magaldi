"""Repository modules for search backend interaction.

This module provides a composed Repository that maintains
backward compatibility while delegating to focused sub-repositories.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .base import INDEX_MAPPING, INDEX_NAME, RepositoryBase, generate_hash_id
from .elements import ElementRepository
from .features import FeatureRepository
from .glossary import GlossaryRepository
from .metadata import MetadataRepository
from .search import SearchRepository
from .stats import StatsRepository

if TYPE_CHECKING:
    from collections.abc import Generator

    from magaldi_core.code_parser import CodeElement
    from shared.config import MagaldiConfig
    from shared.db.bulk_buffer import BulkIndexBuffer


class Repository:
    """Repository for code element indexing and search.

    This class composes specialized sub-repositories and delegates
    all operations to them, maintaining backward compatibility with
    the original monolithic implementation.
    """

    def __init__(self, config: MagaldiConfig | None = None):
        self._base = RepositoryBase(config)
        self._elements = ElementRepository(self._base)
        self._metadata = MetadataRepository(self._base)
        self._search = SearchRepository(self._base)
        self._features = FeatureRepository(self._base)
        self._glossary = GlossaryRepository(self._base)
        self._stats = StatsRepository(self._base)

    # =========================================================================
    # Base operations
    # =========================================================================

    @property
    def config(self) -> MagaldiConfig:
        """Get configuration."""
        return self._base._config

    @property
    def _config(self) -> MagaldiConfig:
        """Get configuration (for backward compatibility)."""
        return self._base._config

    @property
    def _client(self) -> Any:
        """Get search client (for backward compatibility)."""
        return self._base._client

    @_client.setter
    def _client(self, value: Any) -> None:
        """Set search client (for backward compatibility in tests)."""
        self._base._client = value

    def _get_client(self) -> Any:
        """Get or create search backend client."""
        return self._base._get_client()

    def _ensure_index(self) -> None:
        """Create index if it doesn't exist."""
        return self._base._ensure_index()

    def close(self) -> None:
        """Close search backend client."""
        return self._base.close()

    @contextmanager
    def bulk_buffer(
        self,
        max_count: int = 50,
        max_interval: float = 5.0,
    ) -> Generator[BulkIndexBuffer, None, None]:
        """Context manager that batches OpenSearch writes for performance.

        While active, index_element / store_summary / store_embedding /
        store_imports / store_calls will be buffered and flushed in bulk
        when either *max_count* operations accumulate or *max_interval*
        seconds elapse since the last flush.  A final flush happens on exit.

        Usage::

            with repo.bulk_buffer(max_count=100, max_interval=3.0) as buf:
                for elem in elements:
                    repo.index_element(elem, indexed_at=now)
            # buf.total_flushed  — number of ops committed

        Args:
            max_count: Flush after this many buffered operations.
            max_interval: Flush after this many seconds since last flush.

        Yields:
            The active BulkIndexBuffer (useful for inspecting stats).
        """
        from shared.db.bulk_buffer import BulkIndexBuffer

        client = self._base._get_client()
        buf = BulkIndexBuffer(client, INDEX_NAME, max_count=max_count, max_interval=max_interval)
        self._elements._bulk_buffer = buf
        self._metadata._bulk_buffer = buf
        self._features._bulk_buffer = buf
        self._glossary._bulk_buffer = buf
        try:
            with buf:
                yield buf
        finally:
            self._elements._bulk_buffer = None
            self._metadata._bulk_buffer = None
            self._features._bulk_buffer = None
            self._glossary._bulk_buffer = None

    # =========================================================================
    # Element operations (delegated to ElementRepository)
    # =========================================================================

    def index_element(
        self,
        element: CodeElement,
        indexed_at: datetime | None = None,
        file_hash: str | None = None,
        element_count: int | None = None,
    ) -> bool:
        """Index a code element."""
        return self._elements.index_element(element, indexed_at, file_hash, element_count)

    def get_document(self, element_id: str) -> dict[str, Any] | None:
        """Get indexed document by ID."""
        return self._elements.get_document(element_id)

    def get_document_by_hash_id(self, hash_id: str) -> dict[str, Any] | None:
        """Get indexed document by hash_id."""
        return self._elements.get_document_by_hash_id(hash_id)

    def get_document_by_id_or_hash(self, id_or_hash: str) -> dict[str, Any] | None:
        """Get indexed document by either element_id or hash_id."""
        return self._elements.get_document_by_id_or_hash(id_or_hash)

    def element_exists(self, element_id: str) -> bool:
        """Check if element exists."""
        return self._elements.element_exists(element_id)

    def get_existing_element_ids(self, element_ids: list[str]) -> set[str]:
        """Check which elements already exist."""
        return self._elements.get_existing_element_ids(element_ids)

    def get_element_content_hashes(self, element_ids: list[str]) -> dict[str, str | None]:
        """Get content hashes for existing elements."""
        return self._elements.get_element_content_hashes(element_ids)

    def get_element_processing_state(
        self, element_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Get content hash and summary state for smart skip logic."""
        return self._elements.get_element_processing_state(element_ids)

    def find_elements_by_content_hash(
        self,
        scope: str,
        repository: str,
        username: str,
        content_hashes: list[str],
        relative_path: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Find elements by content_hash for relocated element matching."""
        return self._elements.find_elements_by_content_hash(
            scope, repository, username, content_hashes, relative_path
        )

    def get_element_ids_by_file(
        self, scope: str, repository: str, username: str, relative_path: str
    ) -> set[str]:
        """Get all element IDs for a specific file."""
        return self._elements.get_element_ids_by_file(scope, repository, username, relative_path)

    def delete_elements(self, element_ids: list[str]) -> int:
        """Delete specific elements by their IDs."""
        return self._elements.delete_elements(element_ids)

    def delete_by_file(
        self, scope: str, repository: str, username: str, relative_path: str
    ) -> int:
        """Delete all documents for a file."""
        return self._elements.delete_by_file(scope, repository, username, relative_path)

    def count_elements_by_file_hash(
        self, scope: str, repository: str, username: str, file_hash: str
    ) -> int:
        """Count elements with a specific file hash."""
        return self._elements.count_elements_by_file_hash(scope, repository, username, file_hash)

    def count_elements_by_path(
        self, scope: str, repository: str, username: str, relative_path: str
    ) -> int:
        """Count elements for a specific file path."""
        return self._elements.count_elements_by_path(scope, repository, username, relative_path)

    def delete_by_file_hash(
        self, scope: str, repository: str, username: str, file_hash: str
    ) -> int:
        """Delete all elements with a specific file hash."""
        return self._elements.delete_by_file_hash(scope, repository, username, file_hash)

    def delete_by_repository(self, scope: str, repository: str, username: str) -> int:
        """Delete all elements for a repository/user combination."""
        return self._elements.delete_by_repository(scope, repository, username)

    def update_file_hashes(
        self,
        scope: str,
        repository: str,
        username: str,
        file_updates: dict[str, str],
        element_counts: dict[str, int] | None = None,
    ) -> int:
        """Bulk update file_hash and element_count for elements by relative_path."""
        return self._elements.update_file_hashes(
            scope, repository, username, file_updates, element_counts
        )

    def get_documents_batch(self, element_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Get full documents for multiple elements in batch."""
        return self._elements.get_documents_batch(element_ids)

    # =========================================================================
    # Metadata operations (delegated to MetadataRepository)
    # =========================================================================

    def store_embedding(
        self,
        element_id: str,
        embedding: list[float],
        embedding_type: str = "summary",
    ) -> bool:
        """Store embedding vector for an element."""
        return self._metadata.store_embedding(element_id, embedding, embedding_type)

    def store_summary_embedding(self, element_id: str, embedding: list[float]) -> bool:
        """Store summary embedding."""
        return self._metadata.store_summary_embedding(element_id, embedding)

    def store_code_embedding(self, element_id: str, embedding: list[float]) -> bool:
        """Store code embedding."""
        return self._metadata.store_code_embedding(element_id, embedding)

    def store_caller_embedding(self, element_id: str, embedding: list[float]) -> bool:
        """Store caller embedding."""
        return self._metadata.store_caller_embedding(element_id, embedding)

    def find_files_with_incomplete_elements(
        self, scope: str, repository: str, username: str
    ) -> list[str]:
        """Find files with elements missing required embeddings."""
        return self._metadata.find_files_with_incomplete_elements(scope, repository, username)

    def get_embedding(
        self, element_id: str, embedding_type: str = "summary"
    ) -> list[float] | None:
        """Get embedding vector for an element."""
        return self._metadata.get_embedding(element_id, embedding_type)

    def store_summary(
        self, element_id: str, summary: str, craft_reason: str | None = None
    ) -> bool:
        """Store summary for an element."""
        return self._metadata.store_summary(element_id, summary, craft_reason=craft_reason)

    def store_imports(self, element_id: str, imports: list[dict]) -> bool:
        """Store imports for a file element."""
        return self._metadata.store_imports(element_id, imports)

    def store_calls(self, element_id: str, calls: list[dict]) -> bool:
        """Store calls for a function/method element."""
        return self._metadata.store_calls(element_id, calls)

    def store_semantic_related(self, element_id: str, related: list[dict]) -> bool:
        """Store pre-computed semantic relationships for an element."""
        return self._metadata.store_semantic_related(element_id, related)

    def get_imports(self, element_id: str) -> list[dict]:
        """Get imports for a file element."""
        return self._metadata.get_imports(element_id)

    def get_calls(self, element_id: str) -> list[dict]:
        """Get calls for a function/method element."""
        return self._metadata.get_calls(element_id)

    def get_file_imports(
        self,
        relative_path: str,
        scope: str,
        repository: str,
        username: str = "main",
    ) -> list[dict]:
        """Get imports for a file element by path."""
        return self._metadata.get_file_imports(relative_path, scope, repository, username)

    def get_document_by_name(
        self,
        name: str,
        element_type: str,
        relative_path: str,
        scope: str,
        repository: str,
        username: str = "main",
    ) -> dict[str, Any] | None:
        """Get indexed document by name and path."""
        return self._metadata.get_document_by_name(
            name, element_type, relative_path, scope, repository, username
        )

    def get_document_by_name_only(
        self,
        name: str,
        element_type: str,
        scope: str,
        repository: str,
        username: str = "main",
    ) -> dict[str, Any] | None:
        """Get indexed document by name only (without path)."""
        return self._metadata.get_document_by_name_only(
            name, element_type, scope, repository, username
        )

    def get_method_by_class(
        self,
        class_id: str,
        method_name: str,
        scope: str,
        repository: str,
        username: str = "main",
    ) -> dict[str, Any] | None:
        """Get method document by class parent ID and method name."""
        return self._metadata.get_method_by_class(
            class_id, method_name, scope, repository, username
        )

    def get_elements_by_file(
        self,
        relative_path: str,
        scope: str,
        repository: str,
        username: str = "main",
    ) -> list[dict[str, Any]]:
        """Get all elements defined in a file."""
        return self._metadata.get_elements_by_file(
            relative_path, scope, repository, username
        )

    def get_summaries_batch(self, element_ids: list[str]) -> dict[str, str]:
        """Get summaries for multiple elements in batch."""
        return self._metadata.get_summaries_batch(element_ids)

    def get_file_states(
        self, scope: str, repository: str, username: str
    ) -> dict[str, dict[str, Any]]:
        """Get file states for change detection."""
        return self._metadata.get_file_states(scope, repository, username)

    def update_file_element_count(
        self,
        scope: str,
        repository: str,
        username: str,
        relative_path: str,
        element_count: int,
    ) -> bool:
        """Update element_count for a FILE element."""
        return self._metadata.update_file_element_count(
            scope, repository, username, relative_path, element_count
        )

    def main_branch_exists(self, scope: str, repository: str) -> bool:
        """Check if main branch has been indexed."""
        return self._metadata.main_branch_exists(scope, repository)

    def get_all_embeddings(
        self,
        scope: str,
        repository: str,
        username: str,
        element_types: list[str] | None = None,
        embedding_type: str = "summary",
        exclude_tests: bool = True,
    ) -> list[dict[str, Any]]:
        """Fetch all elements with embeddings for clustering."""
        return self._metadata.get_all_embeddings(
            scope, repository, username, element_types, embedding_type, exclude_tests
        )

    # =========================================================================
    # Search operations (delegated to SearchRepository)
    # =========================================================================

    def search_by_text(
        self,
        query: str,
        scope: str | None = None,
        repository: str | None = None,
        username: str | None = None,
        element_types: list[str] | None = None,
        size: int = 10,
    ) -> list[dict[str, Any]]:
        """Search elements by text query."""
        return self._search.search_by_text(
            query, scope, repository, username, element_types, size
        )

    def search_by_vector(
        self,
        embedding: list[float],
        scope: str | None = None,
        repository: str | None = None,
        username: str | None = None,
        element_types: list[str] | None = None,
        size: int = 10,
        min_score: float = 0.3,
        embedding_type: str = "summary",
    ) -> list[dict[str, Any]]:
        """Search elements by vector similarity."""
        return self._search.search_by_vector(
            embedding, scope, repository, username, element_types, size, min_score, embedding_type
        )

    def search_by_keyword(
        self,
        query: str,
        scope: str | None = None,
        repository: str | None = None,
        username: str | None = None,
        element_types: list[str] | None = None,
        size: int = 10,
    ) -> list[dict[str, Any]]:
        """Search elements by keyword matching (BM25)."""
        return self._search.search_by_keyword(
            query, scope, repository, username, element_types, size
        )

    def search_by_regexp(
        self,
        pattern: str,
        scope: str | None = None,
        repository: str | None = None,
        username: str | None = None,
        glob: str | None = None,
        size: int = 50,
        include_tests: bool = True,
    ) -> list[dict[str, Any]]:
        """Search raw_code field using regexp query."""
        return self._search.search_by_regexp(
            pattern, scope, repository, username, glob, size, include_tests
        )

    def search_by_wildcard(
        self,
        pattern: str,
        scope: str | None = None,
        repository: str | None = None,
        username: str | None = None,
        glob: str | None = None,
        size: int = 50,
        include_tests: bool = True,
    ) -> list[dict[str, Any]]:
        """Search raw_code field using wildcard query."""
        return self._search.search_by_wildcard(
            pattern, scope, repository, username, glob, size, include_tests
        )

    def search_by_proximity(
        self,
        terms: str,
        slop: int = 5,
        scope: str | None = None,
        repository: str | None = None,
        username: str | None = None,
        glob: str | None = None,
        size: int = 50,
        include_tests: bool = True,
    ) -> list[dict[str, Any]]:
        """Search raw_code for terms within proximity of each other."""
        return self._search.search_by_proximity(
            terms, slop, scope, repository, username, glob, size, include_tests
        )

    def find_elements_calling(
        self,
        target_id: str,
        scope: str | None = None,
        repository: str | None = None,
        username: str | None = None,
        limit: int = 30,
    ) -> list[dict]:
        """Find elements that call the target."""
        return self._search.find_elements_calling(
            target_id, scope, repository, username, limit
        )

    def find_elements_importing(
        self,
        module: str,
        scope: str | None = None,
        repository: str | None = None,
        username: str | None = None,
        limit: int = 30,
    ) -> list[dict]:
        """Find elements that import the module."""
        return self._search.find_elements_importing(
            module, scope, repository, username, limit
        )

    def find_all_elements_with_calls(
        self,
        scope: str,
        repository: str,
        username: str = "main",
        limit: int = 10000,
    ) -> list[dict]:
        """Find all elements that have any calls (resolved or unresolved)."""
        return self._search.find_all_elements_with_calls(
            scope, repository, username, limit
        )

    def find_elements_with_unresolved_calls(
        self,
        scope: str,
        repository: str,
        username: str = "main",
        limit: int = 1000,
    ) -> list[dict]:
        """Find elements that have calls without resolved_id."""
        return self._search.find_elements_with_unresolved_calls(
            scope, repository, username, limit
        )

    def find_candidates_by_name(
        self,
        name: str,
        scope: str,
        repository: str,
        username: str = "main",
        element_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Find all elements with a given name across the repo."""
        return self._search.find_candidates_by_name(
            name, scope, repository, username, element_types, limit
        )

    # =========================================================================
    # Feature operations (delegated to FeatureRepository)
    # =========================================================================

    def index_feature(
        self,
        feature_id: str,
        scope: str,
        repository: str,
        username: str,
        cluster_id: str,
        label: str,
        summary: str,
        embedding: list[float] | None,
        member_ids: list[str],
        connected_features: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Index a feature document."""
        return self._features.index_feature(
            feature_id, scope, repository, username, cluster_id, label, summary, embedding, member_ids,
            connected_features=connected_features,
        )

    def index_subfeature(
        self,
        subfeature_id: str,
        scope: str,
        repository: str,
        username: str,
        cluster_id: str,
        label: str,
        summary: str,
        embedding: list[float] | None,
        member_ids: list[str],
        parent_feature_label: str,
        parent_feature_summary: str,
    ) -> bool:
        """Index a subfeature document."""
        return self._features.index_subfeature(
            subfeature_id, scope, repository, username, cluster_id, label, summary,
            embedding, member_ids, parent_feature_label, parent_feature_summary
        )

    def get_features(
        self,
        scope: str,
        repository: str,
        username: str = "main",
    ) -> list[dict[str, Any]]:
        """Get all features for a repository."""
        return self._features.get_features(scope, repository, username)

    def get_subfeatures(
        self,
        scope: str,
        repository: str,
        username: str = "main",
        parent_feature_label: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get all subfeatures for a repository."""
        return self._features.get_subfeatures(scope, repository, username, parent_feature_label)

    def delete_features(self, scope: str, repository: str, username: str) -> int:
        """Delete all feature documents for a repository."""
        return self._features.delete_features(scope, repository, username)

    def delete_subfeatures(self, scope: str, repository: str, username: str) -> int:
        """Delete all subfeature documents for a repository."""
        return self._features.delete_subfeatures(scope, repository, username)

    def update_cluster_assignments(self, assignments: list[dict[str, Any]]) -> int:
        """Bulk update cluster assignments for elements."""
        return self._features.update_cluster_assignments(assignments)

    def get_clusters(
        self,
        scope: str,
        repository: str,
        username: str,
    ) -> list[dict[str, Any]]:
        """Get all clusters with their elements."""
        return self._features.get_clusters(scope, repository, username)

    def clear_cluster_assignments(
        self,
        scope: str,
        repository: str,
        username: str,
    ) -> int:
        """Clear all cluster assignments for a repository."""
        return self._features.clear_cluster_assignments(scope, repository, username)

    def get_features_for_element(
        self,
        element_id: str,
        scope: str | None = None,
        repository: str | None = None,
        username: str = "main",
    ) -> list[dict[str, Any]]:
        """Get all features and subfeatures that contain a specific element."""
        return self._features.get_features_for_element(element_id, scope, repository, username)

    def update_element_feature_memberships(
        self,
        scope: str,
        repository: str,
        username: str,
        cluster_id_to_feature: dict[str, tuple[str, str]],
    ) -> int:
        """Update element feature_memberships with proper feature_ids and labels."""
        return self._features.update_element_feature_memberships(
            scope, repository, username, cluster_id_to_feature
        )

    # =========================================================================
    # Glossary operations (delegated to GlossaryRepository)
    # =========================================================================

    def index_glossary(
        self,
        glossary_id: str,
        scope: str,
        repository: str,
        username: str,
        term: str,
        total_count: int,
        element_ids: list[str],
        file_paths: list[str],
        description: str = "",
        feature_associations: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Index a glossary entry document."""
        return self._glossary.index_glossary(
            glossary_id, scope, repository, username, term, total_count,
            element_ids, file_paths, description, feature_associations
        )

    def get_glossary_terms(
        self,
        scope: str,
        repository: str,
        username: str = "main",
        min_count: int = 1,
    ) -> list[dict[str, Any]]:
        """Get all glossary terms for a repository."""
        return self._glossary.get_glossary_terms(scope, repository, username, min_count)

    def get_glossary_term(
        self,
        scope: str,
        repository: str,
        term: str,
        username: str = "main",
    ) -> dict[str, Any] | None:
        """Get a specific glossary term."""
        return self._glossary.get_glossary_term(scope, repository, term, username)

    def search_glossary(
        self,
        scope: str,
        repository: str,
        query: str,
        username: str = "main",
    ) -> list[dict[str, Any]]:
        """Search glossary terms by partial match."""
        return self._glossary.search_glossary(scope, repository, query, username)

    def get_glossary_terms_for_feature(self, feature_id: str) -> list[dict[str, Any]]:
        """Get glossary terms extracted from a specific feature/subfeature."""
        return self._glossary.get_glossary_terms_for_feature(feature_id)

    def update_glossary_feature_associations(
        self,
        glossary_id: str,
        feature_associations: list[dict[str, Any]],
    ) -> bool:
        """Update feature associations for a glossary entry."""
        return self._glossary.update_glossary_feature_associations(
            glossary_id, feature_associations
        )

    def delete_glossary(self, scope: str, repository: str, username: str) -> int:
        """Delete all glossary entries for a repository."""
        return self._glossary.delete_glossary(scope, repository, username)

    # =========================================================================
    # Stats operations (delegated to StatsRepository)
    # =========================================================================

    def get_indexed_repositories(
        self,
        scope: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get all indexed repositories with statistics."""
        return self._stats.get_indexed_repositories(scope)

    def get_all_elements(
        self,
        scope: str,
        repository: str,
        username: str = "main",
        element_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get all elements for a repository."""
        return self._stats.get_all_elements(scope, repository, username, element_types)

    def get_repository_stats(
        self,
        scope: str,
        repository: str,
        username: str = "main",
    ) -> dict[str, Any]:
        """Get statistics for a repository."""
        return self._stats.get_repository_stats(scope, repository, username)


# Backward-compatible alias
ElasticsearchRepository = Repository

__all__ = [
    "Repository",
    "ElasticsearchRepository",  # backward compat alias
    "RepositoryBase",
    "ElementRepository",
    "MetadataRepository",
    "SearchRepository",
    "FeatureRepository",
    "GlossaryRepository",
    "StatsRepository",
    "INDEX_NAME",
    "INDEX_MAPPING",
    "generate_hash_id",
]
