"""Search repository for various search operations.

Handles text search, vector search, pattern matching, and call/import queries.
"""

from __future__ import annotations

from typing import Any

from shared.db.backends.base import build_vector_query

from .base import INDEX_NAME, RepositoryBase


class SearchRepository:
    """Repository for search operations."""

    def __init__(self, base: RepositoryBase):
        self._base = base

    def _get_client(self) -> Any:
        """Get search client from base."""
        return self._base._get_client()

    @staticmethod
    def _build_username_filter(username: str | None) -> dict[str, Any] | None:
        """Build ES filter for username, merging user + main branches.

        When username is not "main", queries both user and main so that
        user-specific data overlays the full "main" index.  When username
        is None, returns None (no filter applied).
        """
        if not username:
            return None
        if username == "main":
            return {"term": {"username": "main"}}
        return {
            "bool": {
                "should": [
                    {"term": {"username": username}},
                    {"term": {"username": "main"}},
                ],
                "minimum_should_match": 1,
            }
        }

    def search_by_text(
        self,
        query: str,
        scope: str | None = None,
        repository: str | None = None,
        username: str | None = None,
        element_types: list[str] | None = None,
        size: int = 10,
    ) -> list[dict[str, Any]]:
        """Search elements by text query.

        Args:
            query: Search query string.
            scope: Filter by scope.
            repository: Filter by repository.
            username: Filter by username.
            element_types: Filter by element types.
            size: Maximum results to return.

        Returns:
            List of matching documents.
        """
        must_clauses: list[dict[str, Any]] = [
            {
                "multi_match": {
                    "query": query,
                    "fields": ["name^3", "summary^2", "docstring", "raw_code"],
                }
            }
        ]

        filter_clauses: list[dict[str, Any]] = []

        if scope:
            filter_clauses.append({"term": {"scope": scope}})
        if repository:
            filter_clauses.append({"term": {"repository": repository}})
        username_filter = self._build_username_filter(username)
        if username_filter:
            filter_clauses.append(username_filter)
        if element_types:
            filter_clauses.append({"terms": {"element_type": element_types}})

        query: dict[str, Any] = {"bool": {"must": must_clauses}}
        if filter_clauses:
            query["bool"]["filter"] = filter_clauses

        client = self._get_client()
        result = client.search(
            index=INDEX_NAME,
            body={"query": query, "size": size},
        )

        return [hit["_source"] for hit in result["hits"]["hits"]]

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
        """Search elements by vector similarity.

        Args:
            embedding: Query embedding vector.
            scope: Filter by scope.
            repository: Filter by repository.
            username: Filter by username.
            element_types: Filter by element types.
            size: Maximum results to return.
            min_score: Minimum similarity score.
            embedding_type: Type of embedding to search - "summary" or "code".

        Returns:
            List of matching documents with scores.
        """
        field_name = f"{embedding_type}_embedding"
        filter_clauses: list[dict[str, Any]] = [
            # Only search documents that have embeddings of the specified type
            {"exists": {"field": field_name}}
        ]

        if scope:
            filter_clauses.append({"term": {"scope": scope}})
        if repository:
            filter_clauses.append({"term": {"repository": repository}})
        username_filter = self._build_username_filter(username)
        if username_filter:
            filter_clauses.append(username_filter)
        if element_types:
            filter_clauses.append({"terms": {"element_type": element_types}})

        backend_type = self._base.backend_type
        query, adjusted_min_score = build_vector_query(
            backend_type, field_name, embedding, filter_clauses, size, min_score,
        )

        client = self._get_client()
        result = client.search(
            index=INDEX_NAME,
            body={"query": query, "size": size, "min_score": adjusted_min_score},
        )

        # Normalize scores: script_score adds +1.0 (ES), knn returns raw cosine (OpenSearch)
        score_offset = 1.0 if backend_type == "elasticsearch" else 0.0
        return [
            {**hit["_source"], "_score": hit["_score"] - score_offset}
            for hit in result["hits"]["hits"]
        ]

    def search_by_keyword(
        self,
        query: str,
        scope: str | None = None,
        repository: str | None = None,
        username: str | None = None,
        element_types: list[str] | None = None,
        size: int = 10,
    ) -> list[dict[str, Any]]:
        """Search elements by keyword matching (BM25).

        Fallback search when vector embeddings are unavailable.

        Args:
            query: Search query string.
            scope: Filter by scope.
            repository: Filter by repository.
            username: Filter by username.
            element_types: Filter by element types.
            size: Maximum results to return.

        Returns:
            List of matching documents with scores.
        """
        filter_clauses: list[dict[str, Any]] = []

        if scope:
            filter_clauses.append({"term": {"scope": scope}})
        if repository:
            filter_clauses.append({"term": {"repository": repository}})
        username_filter = self._build_username_filter(username)
        if username_filter:
            filter_clauses.append(username_filter)
        if element_types:
            filter_clauses.append({"terms": {"element_type": element_types}})

        # Multi-match across name, summary, and signature fields
        must_clause: dict[str, Any] = {
            "multi_match": {
                "query": query,
                "fields": ["name^3", "summary^2", "signature", "docstring"],
                "type": "best_fields",
                "fuzziness": "AUTO",
            }
        }

        search_query: dict[str, Any] = {
            "bool": {
                "must": [must_clause],
            }
        }

        if filter_clauses:
            search_query["bool"]["filter"] = filter_clauses

        client = self._get_client()
        result = client.search(
            index=INDEX_NAME,
            body={"query": search_query, "size": size},
        )

        return [
            {**hit["_source"], "_score": hit["_score"]}
            for hit in result["hits"]["hits"]
        ]

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
        """Search raw_code field using regexp query.

        Uses Lucene regexp syntax (not Python re). Key differences:
        - . matches any character (no need to escape)
        - * is zero or more of previous
        - .* matches any string
        - No lookahead/lookbehind support

        Note: Patterns are automatically wrapped with .* for substring matching
        (Lucene regexp requires matching the entire field by default).
        Elements with raw_code exceeding 8191 chars are excluded from
        regexp/wildcard search (too large for keyword indexing).

        Args:
            pattern: Lucene regexp pattern (e.g., 'add_column.*Model').
            scope: Filter by scope.
            repository: Filter by repository.
            username: Filter by username.
            glob: File path glob filter (e.g., '*.py').
            size: Maximum results.
            include_tests: Include test elements.

        Returns:
            List of matching documents.
        """
        # Wrap pattern with .* for substring matching if not already wrapped
        # Lucene regexp matches the ENTIRE field, so we need anchors for substring search
        search_pattern = pattern
        if not pattern.startswith(".*"):
            search_pattern = ".*" + search_pattern
        if not pattern.endswith(".*"):
            search_pattern = search_pattern + ".*"

        must_clauses: list[dict[str, Any]] = [
            {"regexp": {"raw_code.keyword": {"value": search_pattern, "flags": "ALL"}}},
        ]

        filter_clauses: list[dict[str, Any]] = []

        username_filter = self._build_username_filter(username)
        if username_filter:
            filter_clauses.append(username_filter)
        if scope:
            filter_clauses.append({"term": {"scope": scope}})
        if repository:
            filter_clauses.append({"term": {"repository": repository}})
        if glob:
            filter_clauses.append({"wildcard": {"relative_path": glob}})
        if not include_tests:
            filter_clauses.append({"term": {"is_test": False}})

        query = {
            "bool": {
                "must": must_clauses,
                "filter": filter_clauses,
            }
        }

        client = self._get_client()
        result = client.search(
            index=INDEX_NAME,
            body={"query": query, "size": size},
        )

        return [hit["_source"] for hit in result["hits"]["hits"]]

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
        """Search raw_code field using wildcard query.

        Wildcard syntax:
        - * matches zero or more characters
        - ? matches exactly one character

        Args:
            pattern: Wildcard pattern.
            scope: Filter by scope.
            repository: Filter by repository.
            username: Filter by username (optional).
            glob: File path glob filter.
            size: Maximum results.
            include_tests: Include test elements.

        Returns:
            List of matching documents.
        """
        must_clauses: list[dict[str, Any]] = [
            {"wildcard": {"raw_code.keyword": {"value": pattern, "case_insensitive": True}}},
        ]

        filter_clauses: list[dict[str, Any]] = []

        username_filter = self._build_username_filter(username)
        if username_filter:
            filter_clauses.append(username_filter)
        if scope:
            filter_clauses.append({"term": {"scope": scope}})
        if repository:
            filter_clauses.append({"term": {"repository": repository}})
        if glob:
            filter_clauses.append({"wildcard": {"relative_path": glob}})
        if not include_tests:
            filter_clauses.append({"term": {"is_test": False}})

        query: dict[str, Any] = {"bool": {"must": must_clauses}}
        if filter_clauses:
            query["bool"]["filter"] = filter_clauses

        client = self._get_client()
        result = client.search(
            index=INDEX_NAME,
            body={"query": query, "size": size},
        )

        return [hit["_source"] for hit in result["hits"]["hits"]]

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
        """Search raw_code for terms within proximity of each other.

        Uses match_phrase with slop to find terms that appear near each other.

        Args:
            terms: Space-separated terms to find near each other.
            slop: Maximum number of positions between terms (0 = exact phrase).
            scope: Filter by scope.
            repository: Filter by repository.
            username: Filter by username (optional).
            glob: File path glob filter.
            size: Maximum results.
            include_tests: Include test elements.

        Returns:
            List of matching documents.
        """
        must_clauses: list[dict[str, Any]] = [
            {"match_phrase": {"raw_code": {"query": terms, "slop": slop}}},
        ]

        filter_clauses: list[dict[str, Any]] = []

        username_filter = self._build_username_filter(username)
        if username_filter:
            filter_clauses.append(username_filter)
        if scope:
            filter_clauses.append({"term": {"scope": scope}})
        if repository:
            filter_clauses.append({"term": {"repository": repository}})
        if glob:
            filter_clauses.append({"wildcard": {"relative_path": glob}})
        if not include_tests:
            filter_clauses.append({"term": {"is_test": False}})

        query: dict[str, Any] = {"bool": {"must": must_clauses}}
        if filter_clauses:
            query["bool"]["filter"] = filter_clauses

        client = self._get_client()
        result = client.search(
            index=INDEX_NAME,
            body={"query": query, "size": size},
        )

        return [hit["_source"] for hit in result["hits"]["hits"]]

    def find_elements_calling(
        self,
        target_id: str,
        scope: str | None = None,
        repository: str | None = None,
        username: str | None = None,
        limit: int = 30,
    ) -> list[dict]:
        """Find elements that call the target (query calls.resolved_id).

        Args:
            target_id: Element ID of the target being called.
            scope: Filter by scope.
            repository: Filter by repository.
            username: Filter by username.
            limit: Maximum results to return.

        Returns:
            List of element documents that call the target.
        """
        filter_clauses: list[dict[str, Any]] = [
            # Only search elements that can have calls (functions, methods)
            {"terms": {"element_type": ["function", "method"]}},
            # Note: No need for {"exists": {"field": "calls"}} - the nested query
            # in must already ensures calls exist with a matching resolved_id
        ]

        if scope:
            filter_clauses.append({"term": {"scope": scope}})
        if repository:
            filter_clauses.append({"term": {"repository": repository}})
        username_filter = self._build_username_filter(username)
        if username_filter:
            filter_clauses.append(username_filter)

        # Nested query to find elements with calls.resolved_id matching target
        nested_query: dict[str, Any] = {
            "nested": {
                "path": "calls",
                "query": {
                    "term": {"calls.resolved_id": target_id}
                },
            }
        }

        query: dict[str, Any] = {
            "bool": {
                "must": [nested_query],
                "filter": filter_clauses,
            }
        }

        client = self._get_client()
        try:
            result = client.search(
                index=INDEX_NAME,
                body={"query": query, "size": limit},
            )
            return [hit["_source"] for hit in result.get("hits", {}).get("hits", [])]
        except Exception:
            # Handle case where some documents don't have the nested calls field
            # This can happen with older indexed data or abstract methods
            return []

    def find_elements_importing(
        self,
        module: str,
        scope: str | None = None,
        repository: str | None = None,
        username: str | None = None,
        limit: int = 30,
    ) -> list[dict]:
        """Find elements that import the module (query imports.module).

        Args:
            module: Module name to search for.
            scope: Filter by scope.
            repository: Filter by repository.
            username: Filter by username.
            limit: Maximum results to return.

        Returns:
            List of element documents that import the module.
        """
        filter_clauses: list[dict[str, Any]] = [
            # Only search file elements (imports are on files)
            {"term": {"element_type": "file"}},
            # Only search elements that have the imports field
            {"exists": {"field": "imports"}},
        ]

        if scope:
            filter_clauses.append({"term": {"scope": scope}})
        if repository:
            filter_clauses.append({"term": {"repository": repository}})
        username_filter = self._build_username_filter(username)
        if username_filter:
            filter_clauses.append(username_filter)

        # Nested query to find elements with imports.module matching
        nested_query: dict[str, Any] = {
            "nested": {
                "path": "imports",
                "query": {
                    "term": {"imports.module": module}
                },
            }
        }

        query: dict[str, Any] = {
            "bool": {
                "must": [nested_query],
                "filter": filter_clauses,
            }
        }

        client = self._get_client()
        result = client.search(
            index=INDEX_NAME,
            body={"query": query, "size": limit},
        )

        return [hit["_source"] for hit in result.get("hits", {}).get("hits", [])]

    def find_all_elements_with_calls(
        self,
        scope: str,
        repository: str,
        username: str = "main",
        limit: int = 10000,
    ) -> list[dict]:
        """Find all elements that have any calls (resolved or unresolved).

        Used for full call resolution pass during partial parsing.

        Args:
            scope: Repository scope.
            repository: Repository name.
            username: Username branch.
            limit: Maximum results to return.

        Returns:
            List of element documents with calls.
        """
        query: dict[str, Any] = {
            "bool": {
                "must": [
                    {"term": {"scope": scope}},
                    {"term": {"repository": repository}},
                    {"term": {"username": username}},
                    # For nested fields, must use nested query with exists
                    {"nested": {"path": "calls", "query": {"exists": {"field": "calls.name"}}}},
                ],
            }
        }

        client = self._get_client()
        result = client.search(
            index=INDEX_NAME,
            body={
                "query": query,
                "size": limit,
                "_source": [
                    "element_id",
                    "relative_path",
                    "language",
                    "parameters",
                    "calls",
                    "parent_id",
                ],
            },
        )

        return [hit["_source"] for hit in result.get("hits", {}).get("hits", [])]

    def find_elements_with_unresolved_calls(
        self,
        scope: str,
        repository: str,
        username: str = "main",
        limit: int = 1000,
    ) -> list[dict]:
        """Find elements that have calls without resolved_id.

        Used for Phase 2 cross-file call resolution.

        Args:
            scope: Repository scope.
            repository: Repository name.
            username: Username branch.
            limit: Maximum results to return.

        Returns:
            List of element documents with unresolved calls.
        """
        # Find elements that:
        # 1. Match scope/repository/username
        # 2. Have at least one call with name but no resolved_id
        query: dict[str, Any] = {
            "bool": {
                "must": [
                    {"term": {"scope": scope}},
                    {"term": {"repository": repository}},
                    {"term": {"username": username}},
                    {
                        "nested": {
                            "path": "calls",
                            "query": {
                                "bool": {
                                    "must": [{"exists": {"field": "calls.name"}}],
                                    "must_not": [{"exists": {"field": "calls.resolved_id"}}],
                                }
                            },
                        }
                    },
                ],
            }
        }

        client = self._get_client()
        result = client.search(
            index=INDEX_NAME,
            body={"query": query, "size": limit},
        )

        return [hit["_source"] for hit in result.get("hits", {}).get("hits", [])]

    def find_candidates_by_name(
        self,
        name: str,
        scope: str,
        repository: str,
        username: str = "main",
        element_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Find all elements with a given name across the repo.

        Used for embedding-based call resolution to find candidate targets.

        Args:
            name: Exact element name to search for.
            scope: Repository scope.
            repository: Repository name.
            username: Username branch.
            element_types: Filter by element types (default: function, method).
            limit: Maximum results to return.

        Returns:
            List of element documents with element_id and summary_embedding.
        """
        if element_types is None:
            element_types = ["function", "method"]

        filter_clauses: list[dict[str, Any]] = [
            {"term": {"scope": scope}},
            {"term": {"repository": repository}},
        ]
        username_filter = self._build_username_filter(username)
        if username_filter:
            filter_clauses.append(username_filter)
        else:
            # Default to "main" when no username provided
            filter_clauses.append({"term": {"username": "main"}})

        query: dict[str, Any] = {
            "bool": {
                "must": [
                    {"term": {"name": name}},
                    {"terms": {"element_type": element_types}},
                ],
                "filter": filter_clauses,
            }
        }

        client = self._get_client()
        result = client.search(
            index=INDEX_NAME,
            body={
                "query": query,
                "size": limit,
                "_source": [
                    "element_id",
                    "hash_id",
                    "summary_embedding",
                    "name",
                    "element_type",
                    "relative_path",
                    "line_start",
                    "parent_id",
                ],
            },
        )

        return [hit["_source"] for hit in result.get("hits", {}).get("hits", [])]
