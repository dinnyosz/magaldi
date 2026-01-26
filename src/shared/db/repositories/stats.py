"""Stats repository for retrieving repository statistics.

Handles retrieval of indexed repository information and element statistics.
"""

from __future__ import annotations

from typing import Any

from .base import INDEX_NAME, ElasticsearchBase


class StatsRepository:
    """Repository for statistics and repository listing operations."""

    def __init__(self, base: ElasticsearchBase):
        self._base = base

    def _get_client(self) -> Any:
        """Get Elasticsearch client from base."""
        return self._base._get_client()

    def get_indexed_repositories(
        self,
        scope: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get all indexed repositories with statistics.

        Args:
            scope: Filter by scope (optional).

        Returns:
            List of repositories with file/element counts.
        """
        client = self._get_client()

        # Build filter
        filters = [{"term": {"username": "main"}}]
        if scope:
            filters.append({"term": {"scope": scope}})

        # Aggregate by scope and repository
        result = client.search(
            index=INDEX_NAME,
            body={
                "size": 0,
                "query": {
                    "bool": {
                        "filter": filters,
                        "must_not": [{"term": {"element_type": "feature"}}],
                    }
                },
                "aggs": {
                    "repos": {
                        "composite": {
                            "size": 1000,
                            "sources": [
                                {"scope": {"terms": {"field": "scope"}}},
                                {"repository": {"terms": {"field": "repository"}}},
                            ],
                        },
                        "aggs": {
                            "file_count": {
                                "filter": {"term": {"element_type": "file"}},
                            },
                            "languages": {
                                "terms": {"field": "language", "size": 20},
                            },
                        },
                    }
                },
            },
        )

        repos = []
        for bucket in result.get("aggregations", {}).get("repos", {}).get("buckets", []):
            repos.append({
                "scope": bucket["key"]["scope"],
                "repository": bucket["key"]["repository"],
                "element_count": bucket["doc_count"],
                "file_count": bucket["file_count"]["doc_count"],
                "languages": [
                    lang["key"]
                    for lang in bucket.get("languages", {}).get("buckets", [])
                    if lang["key"]
                ],
            })

        return repos

    def get_all_elements(
        self,
        scope: str,
        repository: str,
        username: str = "main",
        element_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get all elements for a repository (excluding features, subfeatures, glossary).

        Args:
            scope: Scope to filter by.
            repository: Repository to filter by.
            username: Username/branch (default: main).
            element_types: Optional filter by element types.

        Returns:
            List of element dicts.
        """
        client = self._get_client()

        must_clauses: list[dict[str, Any]] = [
            {"term": {"scope": scope}},
            {"term": {"repository": repository}},
            {"term": {"username": username}},
        ]

        # Exclude non-element types
        must_not_clauses: list[dict[str, Any]] = [
            {"term": {"element_type": "feature"}},
            {"term": {"element_type": "subfeature"}},
            {"term": {"element_type": "glossary"}},
        ]

        if element_types:
            must_clauses.append({"terms": {"element_type": element_types}})

        query: dict[str, Any] = {
            "bool": {
                "must": must_clauses,
                "must_not": must_not_clauses,
            }
        }

        # Use scroll for large result sets
        results: list[dict[str, Any]] = []

        response = client.search(
            index=INDEX_NAME,
            query=query,
            size=1000,
            scroll="2m",
        )

        scroll_id = response.get("_scroll_id")
        hits = response.get("hits", {}).get("hits", [])

        while hits:
            for hit in hits:
                source = hit.get("_source", {})
                source["element_id"] = hit.get("_id")
                results.append(source)

            response = client.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = response.get("_scroll_id")
            hits = response.get("hits", {}).get("hits", [])

        # Clear scroll
        if scroll_id:
            import contextlib
            with contextlib.suppress(Exception):
                client.clear_scroll(scroll_id=scroll_id)

        return results

    def get_repository_stats(
        self,
        scope: str,
        repository: str,
        username: str = "main",
    ) -> dict[str, Any]:
        """Get statistics for a repository.

        Args:
            scope: Repository scope.
            repository: Repository name.
            username: User branch.

        Returns:
            Dictionary with element counts, language breakdown, feature count.
        """
        client = self._get_client()

        # Get element stats
        result = client.search(
            index=INDEX_NAME,
            body={
                "size": 0,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"scope": scope}},
                            {"term": {"repository": repository}},
                            {"term": {"username": username}},
                        ]
                    }
                },
                "aggs": {
                    "by_type": {
                        "terms": {"field": "element_type", "size": 20},
                    },
                    "by_language": {
                        "terms": {"field": "language", "size": 20},
                    },
                    "total_lines": {
                        "filter": {"term": {"element_type": "file"}},
                        "aggs": {
                            "lines": {"sum": {"field": "line_end"}},
                        },
                    },
                },
            },
        )

        aggs = result.get("aggregations", {})

        # Build type counts
        type_counts = {}
        for bucket in aggs.get("by_type", {}).get("buckets", []):
            type_counts[bucket["key"]] = bucket["doc_count"]

        # Build language counts
        language_counts = {}
        for bucket in aggs.get("by_language", {}).get("buckets", []):
            if bucket["key"]:  # Skip empty language
                language_counts[bucket["key"]] = bucket["doc_count"]

        feature_count = type_counts.pop("feature", 0)

        return {
            "scope": scope,
            "repository": repository,
            "elements_by_type": type_counts,
            "elements_by_language": language_counts,
            "total_elements": sum(type_counts.values()),
            "total_lines": int(aggs.get("total_lines", {}).get("lines", {}).get("value", 0)),
            "feature_count": feature_count,
        }
