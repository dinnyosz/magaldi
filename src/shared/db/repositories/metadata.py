"""Metadata repository for storing embeddings, summaries, imports, and calls.

Handles storage and retrieval of AI-generated metadata like summaries,
embeddings, imports, and function calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shared.db.backends.base import NotFoundError

from .base import INDEX_NAME, RepositoryBase

if TYPE_CHECKING:
    from shared.db.bulk_buffer import BulkIndexBuffer


class MetadataRepository:
    """Repository for metadata operations (embeddings, summaries, imports, calls)."""

    def __init__(self, base: RepositoryBase):
        self._base = base
        self._bulk_buffer: BulkIndexBuffer | None = None

    def _get_client(self) -> Any:
        """Get search client from base."""
        return self._base._get_client()

    def get_document(self, element_id: str) -> dict[str, Any] | None:
        """Get indexed document by ID."""
        try:
            client = self._get_client()
            result = client.get_document(INDEX_NAME, element_id)
            return result["_source"]  # type: ignore[no-any-return]
        except NotFoundError:
            return None

    def get_document_by_name(
        self,
        name: str,
        element_type: str,
        relative_path: str,
        scope: str,
        repository: str,
        username: str = "main",
    ) -> dict[str, Any] | None:
        """Get indexed document by name and path.

        Used for cross-file call resolution when we don't know the line number.

        Args:
            name: Element name.
            element_type: Element type (function, class, method).
            relative_path: Relative file path.
            scope: Repository scope.
            repository: Repository name.
            username: Username branch.

        Returns:
            Document if found, None otherwise.
        """
        client = self._get_client()
        query = {
            "bool": {
                "must": [
                    {"term": {"name.keyword": name}},
                    {"term": {"element_type": element_type}},
                    {"term": {"relative_path": relative_path}},
                    {"term": {"scope": scope}},
                    {"term": {"repository": repository}},
                    {"term": {"username": username}},
                ]
            }
        }

        result = client.search(
            index=INDEX_NAME,
            body={"query": query, "size": 1},
        )

        hits = result.get("hits", {}).get("hits", [])
        if hits:
            return hits[0]["_source"]  # type: ignore[no-any-return]
        return None

    def get_document_by_name_only(
        self,
        name: str,
        element_type: str,
        scope: str,
        repository: str,
        username: str = "main",
    ) -> dict[str, Any] | None:
        """Get indexed document by name only (without path).

        Used for type-based resolution when we need to find a class by name.

        Args:
            name: Element name.
            element_type: Element type (function, class, method).
            scope: Repository scope.
            repository: Repository name.
            username: Username branch.

        Returns:
            Document if found, None otherwise.
        """
        client = self._get_client()
        query = {
            "bool": {
                "must": [
                    {"term": {"name.keyword": name}},
                    {"term": {"element_type": element_type}},
                    {"term": {"scope": scope}},
                    {"term": {"repository": repository}},
                    {"term": {"username": username}},
                ]
            }
        }

        result = client.search(
            index=INDEX_NAME,
            body={"query": query, "size": 1},
        )

        hits = result.get("hits", {}).get("hits", [])
        if hits:
            return hits[0]["_source"]  # type: ignore[no-any-return]
        return None

    def get_elements_by_file(
        self,
        relative_path: str,
        scope: str,
        repository: str,
        username: str = "main",
    ) -> list[dict[str, Any]]:
        """Get all elements defined in a file.

        Returns functions, classes, methods, etc. for wildcard import expansion.

        Args:
            relative_path: Relative file path.
            scope: Repository scope.
            repository: Repository name.
            username: Username branch.

        Returns:
            List of element dicts with name, element_type, element_id.
        """
        client = self._get_client()
        query = {
            "bool": {
                "must": [
                    {"term": {"relative_path": relative_path}},
                    {"term": {"scope": scope}},
                    {"term": {"repository": repository}},
                    {"term": {"username": username}},
                ],
                "must_not": [
                    {"term": {"element_type": "file"}},
                ],
            }
        }

        result = client.search(
            index=INDEX_NAME,
            body={
                "query": query,
                "size": 500,
                "_source": ["name", "element_type", "element_id"],
            },
        )

        return [hit["_source"] for hit in result.get("hits", {}).get("hits", [])]

    def get_method_by_class(
        self,
        class_id: str,
        method_name: str,
        scope: str,
        repository: str,
        username: str = "main",
    ) -> dict[str, Any] | None:
        """Get method document by class parent ID and method name.

        Args:
            class_id: Element ID of the parent class.
            method_name: Name of the method.
            scope: Repository scope.
            repository: Repository name.
            username: Username branch.

        Returns:
            Method document if found, None otherwise.
        """
        client = self._get_client()
        query = {
            "bool": {
                "must": [
                    {"term": {"name.keyword": method_name}},
                    {"term": {"element_type": "method"}},
                    {"term": {"parent_id": class_id}},
                    {"term": {"scope": scope}},
                    {"term": {"repository": repository}},
                    {"term": {"username": username}},
                ]
            }
        }

        result = client.search(
            index=INDEX_NAME,
            body={"query": query, "size": 1},
        )

        hits = result.get("hits", {}).get("hits", [])
        if hits:
            return hits[0]["_source"]  # type: ignore[no-any-return]
        return None

    def store_embedding(
        self,
        element_id: str,
        embedding: list[float],
        embedding_type: str = "summary",
    ) -> bool:
        """Store embedding vector for an element.

        Args:
            element_id: Element ID to update.
            embedding: Vector embedding (1024 dimensions).
            embedding_type: Type of embedding - "summary" or "code".

        Returns:
            True on success.
        """
        field_name = f"{embedding_type}_embedding"
        if self._bulk_buffer is not None:
            self._bulk_buffer.add_update(element_id, {field_name: embedding})
            return True
        try:
            client = self._get_client()
            client.update_document(
                INDEX_NAME,
                element_id,
                body={"doc": {field_name: embedding}},
            )
            return True
        except NotFoundError:
            return False

    def store_summary_embedding(self, element_id: str, embedding: list[float]) -> bool:
        """Store summary embedding (convenience wrapper).

        Args:
            element_id: Element ID to update.
            embedding: Vector embedding (1024 dimensions).

        Returns:
            True on success.
        """
        return self.store_embedding(element_id, embedding, embedding_type="summary")

    def store_code_embedding(self, element_id: str, embedding: list[float]) -> bool:
        """Store code embedding (convenience wrapper).

        Args:
            element_id: Element ID to update.
            embedding: Vector embedding (1024 dimensions).

        Returns:
            True on success.
        """
        return self.store_embedding(element_id, embedding, embedding_type="code")

    def store_caller_embedding(self, element_id: str, embedding: list[float]) -> bool:
        """Store caller embedding (convenience wrapper).

        Caller embedding = summary embedding enriched with outbound calls.
        Used for asymmetric call resolution (Strategy 6).

        Args:
            element_id: Element ID to update.
            embedding: Vector embedding (1024 dimensions).

        Returns:
            True on success.
        """
        return self.store_embedding(element_id, embedding, embedding_type="caller")

    def get_embedding(
        self, element_id: str, embedding_type: str = "summary"
    ) -> list[float] | None:
        """Get embedding vector for an element.

        Args:
            element_id: Element ID to retrieve.
            embedding_type: Type of embedding - "summary" or "code".

        Returns:
            Embedding vector or None if not found.
        """
        field_name = f"{embedding_type}_embedding"
        doc = self.get_document(element_id)
        if doc:
            return doc.get(field_name)
        return None

    def store_summary(
        self, element_id: str, summary: str, craft_reason: str | None = None
    ) -> bool:
        """Store summary for an element in the index.

        Args:
            element_id: Element ID to update.
            summary: Summary text.
            craft_reason: Why this element was handcrafted ("test", "import",
                "small", "docstring"), or None for LLM-generated summaries.

        Returns:
            True on success.
        """
        doc: dict = {"summary": summary}
        if craft_reason is not None:
            doc["craft_reason"] = craft_reason
        if self._bulk_buffer is not None:
            self._bulk_buffer.add_update(element_id, doc)
            return True
        try:
            client = self._get_client()
            client.update_document(
                INDEX_NAME,
                element_id,
                body={"doc": doc},
            )
            return True
        except NotFoundError:
            return False

    def store_imports(self, element_id: str, imports: list[dict]) -> bool:
        """Store imports for a file element.

        Args:
            element_id: Element ID to update (should be a file element).
            imports: List of import dicts with keys: name, module, alias, line.

        Returns:
            True on success, False if element not found.
        """
        if self._bulk_buffer is not None:
            self._bulk_buffer.add_update(element_id, {"imports": imports})
            return True
        try:
            client = self._get_client()
            client.update_document(
                INDEX_NAME,
                element_id,
                body={"doc": {"imports": imports}},
            )
            return True
        except NotFoundError:
            return False

    def store_calls(self, element_id: str, calls: list[dict]) -> bool:
        """Store calls for a function/method element.

        Args:
            element_id: Element ID to update (should be a function/method).
            calls: List of call dicts with keys: name, receiver, line, resolved_id.

        Returns:
            True on success, False if element not found.
        """
        if self._bulk_buffer is not None:
            self._bulk_buffer.add_update(element_id, {"calls": calls})
            return True
        try:
            client = self._get_client()
            client.update_document(
                INDEX_NAME,
                element_id,
                body={"doc": {"calls": calls}},
            )
            return True
        except NotFoundError:
            return False

    def store_semantic_related(self, element_id: str, related: list[dict]) -> bool:
        """Store pre-computed semantic relationships for an element.

        Args:
            element_id: Element ID to update.
            related: List of dicts with keys: element_id, hash_id, score.

        Returns:
            True on success, False if element not found.
        """
        if self._bulk_buffer is not None:
            self._bulk_buffer.add_update(element_id, {"semantic_related": related})
            return True
        try:
            client = self._get_client()
            client.update_document(
                INDEX_NAME,
                element_id,
                body={"doc": {"semantic_related": related}},
            )
            return True
        except NotFoundError:
            return False

    def get_imports(self, element_id: str) -> list[dict]:
        """Get imports for a file element.

        Args:
            element_id: Element ID to retrieve imports for.

        Returns:
            List of import dicts, or empty list if not found.
        """
        doc = self.get_document(element_id)
        if doc:
            return doc.get("imports", []) or []
        return []

    def get_calls(self, element_id: str) -> list[dict]:
        """Get calls for a function/method element.

        Args:
            element_id: Element ID to retrieve calls for.

        Returns:
            List of call dicts, or empty list if not found.
        """
        doc = self.get_document(element_id)
        if doc:
            return doc.get("calls", []) or []
        return []

    def get_file_imports(
        self,
        relative_path: str,
        scope: str,
        repository: str,
        username: str = "main",
    ) -> list[dict]:
        """Get imports for a file element.

        Args:
            relative_path: Relative path of the file.
            scope: Repository scope.
            repository: Repository name.
            username: Username branch.

        Returns:
            List of import dicts with keys: name, module, alias, line.
        """
        # Build the file element ID
        from pathlib import Path

        file_name = Path(relative_path).name
        file_element_id = f"{scope}:{repository}:{username}:{relative_path}:file:{file_name}:1"

        doc = self.get_document(file_element_id)
        if doc:
            return doc.get("imports", []) or []
        return []

    def get_summaries_batch(
        self,
        element_ids: list[str],
    ) -> dict[str, str]:
        """Get summaries for multiple elements in batch.

        Args:
            element_ids: List of element IDs to fetch summaries for.

        Returns:
            Dict mapping element_id to summary (only for elements with summaries).
        """
        if not element_ids:
            return {}

        client = self._get_client()

        # Use mget for efficient batch lookup
        response = client.mget(
            INDEX_NAME,
            element_ids,
            source=["summary"],
        )

        result: dict[str, str] = {}
        for doc in response.get("docs", []):
            if doc.get("found") and doc.get("_source", {}).get("summary"):
                result[doc["_id"]] = doc["_source"]["summary"]

        return result

    def get_file_states(
        self, scope: str, repository: str, username: str
    ) -> dict[str, dict[str, Any]]:
        """Get file states for change detection.

        Retrieves all file-level elements and their hashes.
        Also detects orphan files (files with elements but no FILE element)
        from interrupted runs, returning them with file_hash=None so they
        get re-parsed.

        Args:
            scope: Scope to filter by.
            repository: Repository to filter by.
            username: Username to filter by.

        Returns:
            Dict mapping relative_path to {file_hash, is_deleted, element_count}.
        """
        client = self._get_client()

        # Search for all file-level elements
        result = client.search(
            index=INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"scope": scope}},
                            {"term": {"repository": repository}},
                            {"term": {"username": username}},
                            {"term": {"element_type": "file"}},
                        ]
                    }
                },
                "size": 10000,  # Adjust if needed
                "_source": ["relative_path", "file_hash", "element_count"],
            },
        )

        # Also find orphan files: paths with elements but no FILE element
        # This handles interrupted runs where FILE element wasn't created
        orphan_agg = client.search(
            index=INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"scope": scope}},
                            {"term": {"repository": repository}},
                            {"term": {"username": username}},
                        ],
                        "must_not": [
                            {"term": {"element_type": "file"}},
                        ],
                    }
                },
                "size": 0,
                "aggs": {
                    "paths": {
                        "terms": {
                            "field": "relative_path",  # keyword field
                            "size": 10000,
                        }
                    }
                },
            },
        )
        # Get all paths that have non-FILE elements
        all_element_paths = {
            bucket["key"]
            for bucket in orphan_agg.get("aggregations", {}).get("paths", {}).get("buckets", [])
        }

        states = {}
        for hit in result["hits"]["hits"]:
            source = hit["_source"]
            states[source["relative_path"]] = {
                "file_hash": source.get("file_hash"),
                "is_deleted": False,
                "element_count": source.get("element_count"),
            }

        # Detect orphan paths: have elements but no FILE element (from interrupted runs)
        # Add them to states with file_hash=None so they get re-parsed
        orphan_paths = all_element_paths - set(states.keys())
        for orphan_path in orphan_paths:
            states[orphan_path] = {
                "file_hash": None,  # Will trigger re-parse
                "is_deleted": False,
                "element_count": None,
                "is_orphan": True,  # Mark as orphan for logging
            }

        return states

    def main_branch_exists(self, scope: str, repository: str) -> bool:
        """Check if main branch has been indexed.

        Args:
            scope: Scope to check.
            repository: Repository to check.

        Returns:
            True if any elements exist for main branch.
        """
        client = self._get_client()

        result = client.count(
            index=INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"scope": scope}},
                            {"term": {"repository": repository}},
                            {"term": {"username": "main"}},
                        ]
                    }
                }
            },
        )

        return result["count"] > 0  # type: ignore[no-any-return]

    def update_file_element_count(
        self,
        scope: str,
        repository: str,
        username: str,
        relative_path: str,
        element_count: int,
    ) -> bool:
        """Update element_count for a FILE element.

        Called when element count drifts (e.g., due to parser changes) to keep
        the stored metadata accurate.

        Args:
            scope: Scope to filter by.
            repository: Repository to filter by.
            username: Username to filter by.
            relative_path: File path to update.
            element_count: New element count.

        Returns:
            True if update succeeded.
        """
        client = self._get_client()

        result = client.update_by_query(
            index=INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"scope": scope}},
                            {"term": {"repository": repository}},
                            {"term": {"username": username}},
                            {"term": {"relative_path": relative_path}},
                            {"term": {"element_type": "file"}},
                        ]
                    }
                },
                "script": {
                    "source": "ctx._source.element_count = params.count",
                    "params": {"count": element_count},
                },
            },
            refresh=True,
        )

        return result.get("updated", 0) > 0  # type: ignore[no-any-return]

    def find_files_with_incomplete_elements(
        self,
        scope: str,
        repository: str,
        username: str,
    ) -> list[str]:
        """Find file paths that have elements missing required embeddings.

        Returns file paths where at least one element has ``calls`` and
        ``summary_embedding`` but is missing ``caller_embedding``.  Only
        elements with calls are expected to have ``caller_embedding``; elements
        without calls (variables, constants, classes, etc.) intentionally never
        get one, so they should *not* trigger re-processing.

        Args:
            scope: Repository scope.
            repository: Repository name.
            username: Username/branch.

        Returns:
            List of relative file paths with incomplete elements.
        """
        client = self._get_client()

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
                            {"exists": {"field": "summary_embedding"}},
                            {"exists": {"field": "calls"}},
                        ],
                        "must_not": [
                            {"exists": {"field": "caller_embedding"}},
                        ],
                    }
                },
                "aggs": {
                    "files": {
                        "terms": {
                            "field": "relative_path",
                            "size": 10000,
                        }
                    }
                },
            },
        )

        buckets = result.get("aggregations", {}).get("files", {}).get("buckets", [])
        return [bucket["key"] for bucket in buckets]

    def get_all_embeddings(
        self,
        scope: str,
        repository: str,
        username: str,
        element_types: list[str] | None = None,
        embedding_type: str = "summary",
        exclude_tests: bool = True,
    ) -> list[dict[str, Any]]:
        """Fetch all elements with embeddings for clustering.

        Args:
            scope: Scope to filter by.
            repository: Repository to filter by.
            username: Username to filter by.
            element_types: Filter by element types (e.g., ["function", "method"]).
            embedding_type: Type of embedding to fetch - "summary" or "code".
            exclude_tests: If True, exclude elements marked as tests (default: True).

        Returns:
            List of dicts with element_id, {embedding_type}_embedding, element_type, name, relative_path.
        """
        field_name = f"{embedding_type}_embedding"
        must_clauses: list[dict[str, Any]] = [
            {"term": {"scope": scope}},
            {"term": {"repository": repository}},
            {"term": {"username": username}},
            {"exists": {"field": field_name}},
        ]

        if element_types:
            must_clauses.append({"terms": {"element_type": element_types}})

        # Build query with optional test exclusion
        query: dict[str, Any] = {"bool": {"must": must_clauses}}
        if exclude_tests:
            query["bool"]["must_not"] = [{"term": {"is_test": True}}]

        client = self._get_client()

        # Use scroll for large result sets
        results: list[dict[str, Any]] = []
        response = client.search(
            index=INDEX_NAME,
            body={
                "query": query,
                "size": 1000,
                "_source": ["element_id", field_name, "element_type", "name", "relative_path", "summary"],
            },
            scroll="2m",
        )

        scroll_id = response.get("_scroll_id")
        hits = response["hits"]["hits"]

        while hits:
            for hit in hits:
                results.append(hit["_source"])

            response = client.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = response.get("_scroll_id")
            hits = response["hits"]["hits"]

        # Clear scroll
        if scroll_id:
            client.clear_scroll(scroll_id=scroll_id)

        return results
