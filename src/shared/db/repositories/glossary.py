"""Glossary repository for managing glossary terms.

Handles indexing, retrieval, and searching of glossary entries.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .base import INDEX_NAME, RepositoryBase, generate_hash_id


class GlossaryRepository:
    """Repository for glossary term operations."""

    def __init__(self, base: RepositoryBase):
        self._base = base

    def _get_client(self) -> Any:
        """Get search client from base."""
        return self._base._get_client()

    def _get_bulk_timeout(self) -> int:
        """Get bulk operation timeout from config."""
        return self._base._config.search_backend.bulk_timeout  # type: ignore[no-any-return]

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
        """Index a glossary entry document.

        Args:
            glossary_id: Unique glossary ID (format: scope:repo:username:glossary:term).
            scope: Repository scope.
            repository: Repository name.
            username: Username/branch.
            term: The glossary term (e.g., "user", "email").
            total_count: Total occurrences of this term.
            element_ids: List of element IDs containing this term.
            file_paths: List of file paths containing this term.
            description: AI-generated description of the term.
            feature_associations: List of feature association dicts with feature_id,
                feature_label, frequency, total_members, percentage.

        Returns:
            True on success.
        """
        doc: dict[str, Any] = {
            "element_id": glossary_id,
            "hash_id": generate_hash_id(glossary_id),
            "scope": scope,
            "repository": repository,
            "username": username,
            "element_type": "glossary",
            "name": term,
            "term": term,
            "total_count": total_count,
            "element_ids": element_ids,
            "file_paths": file_paths,
            "description": description,
            "feature_associations": feature_associations or [],
            "indexed_at": datetime.now().isoformat(),
            "level": -3,  # Glossary terms are at the highest conceptual level
        }

        client = self._get_client()
        client.index_document(INDEX_NAME, glossary_id, doc)
        return True

    def get_glossary_terms(
        self,
        scope: str,
        repository: str,
        username: str = "main",
        min_count: int = 1,
    ) -> list[dict[str, Any]]:
        """Get all glossary terms for a repository.

        Args:
            scope: Repository scope.
            repository: Repository name.
            username: User branch.
            min_count: Minimum occurrence count to filter by.

        Returns:
            List of glossary terms with their counts and associations.
        """
        client = self._get_client()

        query: dict[str, Any] = {
            "bool": {
                "must": [
                    {"term": {"scope": scope}},
                    {"term": {"repository": repository}},
                    {"term": {"username": username}},
                    {"term": {"element_type": "glossary"}},
                    {"range": {"total_count": {"gte": min_count}}},
                ]
            }
        }

        response = client.search(
            index=INDEX_NAME,
            body={"query": query, "size": 1000, "sort": [{"total_count": "desc"}]},
        )

        results: list[dict[str, Any]] = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            results.append({
                "glossary_id": hit.get("_id"),
                "term": source.get("term"),
                "total_count": source.get("total_count"),
                "element_ids": source.get("element_ids", []),
                "file_paths": source.get("file_paths", []),
                "description": source.get("description", ""),
                "feature_associations": source.get("feature_associations", []),
            })

        return results

    def get_glossary_term(
        self,
        scope: str,
        repository: str,
        term: str,
        username: str = "main",
    ) -> dict[str, Any] | None:
        """Get a specific glossary term.

        Args:
            scope: Repository scope.
            repository: Repository name.
            term: The glossary term to retrieve.
            username: User branch.

        Returns:
            Glossary term data or None if not found.
        """
        glossary_id = f"{scope}:{repository}:{username}:glossary:{term}"
        client = self._get_client()

        try:
            response = client.get_document(INDEX_NAME, glossary_id)
            if response.get("found"):
                source = response.get("_source", {})
                return {
                    "glossary_id": glossary_id,
                    "term": source.get("term"),
                    "total_count": source.get("total_count"),
                    "element_ids": source.get("element_ids", []),
                    "file_paths": source.get("file_paths", []),
                    "description": source.get("description", ""),
                    "feature_associations": source.get("feature_associations", []),
                }
        except Exception:
            pass

        return None

    def search_glossary(
        self,
        scope: str,
        repository: str,
        query: str,
        username: str = "main",
    ) -> list[dict[str, Any]]:
        """Search glossary terms by partial match.

        Args:
            scope: Repository scope.
            repository: Repository name.
            query: Search query (partial match).
            username: User branch.

        Returns:
            List of matching glossary terms.
        """
        client = self._get_client()

        es_query: dict[str, Any] = {
            "bool": {
                "must": [
                    {"term": {"scope": scope}},
                    {"term": {"repository": repository}},
                    {"term": {"username": username}},
                    {"term": {"element_type": "glossary"}},
                    {"wildcard": {"term": f"*{query.lower()}*"}},
                ]
            }
        }

        response = client.search(
            index=INDEX_NAME,
            body={"query": es_query, "size": 100, "sort": [{"total_count": "desc"}]},
        )

        results: list[dict[str, Any]] = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            results.append({
                "glossary_id": hit.get("_id"),
                "term": source.get("term"),
                "total_count": source.get("total_count"),
                "description": source.get("description", ""),
            })

        return results

    def get_glossary_terms_for_feature(
        self,
        feature_id: str,
    ) -> list[dict[str, Any]]:
        """Get glossary terms extracted from a specific feature/subfeature.

        Args:
            feature_id: The feature or subfeature element_id.

        Returns:
            List of glossary terms that were extracted from this feature.
        """
        client = self._get_client()

        # feature_associations is not nested, so use simple term query
        es_query: dict[str, Any] = {
            "bool": {
                "must": [
                    {"term": {"element_type": "glossary"}},
                    {"term": {"feature_associations.feature_id": feature_id}},
                ]
            }
        }

        response = client.search(
            index=INDEX_NAME,
            body={"query": es_query, "size": 100, "sort": [{"term": "asc"}]},
        )

        results: list[dict[str, Any]] = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            results.append({
                "glossary_id": hit.get("_id"),
                "term": source.get("term"),
                "description": source.get("description", ""),
                "total_count": source.get("total_count", 0),
            })

        return results

    def update_glossary_feature_associations(
        self,
        glossary_id: str,
        feature_associations: list[dict[str, Any]],
    ) -> bool:
        """Update feature associations for a glossary entry.

        Args:
            glossary_id: Glossary entry ID.
            feature_associations: List of feature association dicts with
                feature_id, feature_label, frequency, total_members, percentage.

        Returns:
            True on success.
        """
        client = self._get_client()

        client.update_document(
            INDEX_NAME,
            glossary_id,
            body={"doc": {
                "feature_associations": feature_associations,
                "updated_at": datetime.now().isoformat(),
            }},
        )

        return True

    def delete_glossary(
        self,
        scope: str,
        repository: str,
        username: str,
    ) -> int:
        """Delete all glossary entries for a repository.

        Args:
            scope: Repository scope.
            repository: Repository name.
            username: User branch.

        Returns:
            Number of glossary entries deleted.
        """
        client = self._get_client()
        bulk_timeout = self._get_bulk_timeout()

        response = client.delete_by_query(
            index=INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"scope": scope}},
                            {"term": {"repository": repository}},
                            {"term": {"username": username}},
                            {"term": {"element_type": "glossary"}},
                        ]
                    }
                }
            },
            refresh=True,
            timeout=f"{bulk_timeout}s",
            request_timeout=bulk_timeout,
        )

        return response.get("deleted", 0)  # type: ignore[no-any-return]
