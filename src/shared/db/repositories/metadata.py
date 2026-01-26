"""Metadata repository for storing embeddings, summaries, imports, and calls.

Handles storage and retrieval of AI-generated metadata like summaries,
embeddings, imports, and function calls.
"""

from __future__ import annotations

from typing import Any

from elasticsearch import NotFoundError

from .base import INDEX_NAME, ElasticsearchBase


class MetadataRepository:
    """Repository for metadata operations (embeddings, summaries, imports, calls)."""

    def __init__(self, base: ElasticsearchBase):
        self._base = base

    def _get_client(self) -> Any:
        """Get Elasticsearch client from base."""
        return self._base._get_client()

    def get_document(self, element_id: str) -> dict[str, Any] | None:
        """Get indexed document by ID."""
        try:
            client = self._get_client()
            result = client.get(index=INDEX_NAME, id=element_id)
            return result["_source"]
        except NotFoundError:
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
        try:
            client = self._get_client()
            client.update(
                index=INDEX_NAME,
                id=element_id,
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

    def store_summary(self, element_id: str, summary: str) -> bool:
        """Store summary for an element in the index.

        Args:
            element_id: Element ID to update.
            summary: Summary text.

        Returns:
            True on success.
        """
        try:
            client = self._get_client()
            client.update(
                index=INDEX_NAME,
                id=element_id,
                body={"doc": {"summary": summary}},
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
        try:
            client = self._get_client()
            client.update(
                index=INDEX_NAME,
                id=element_id,
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
        try:
            client = self._get_client()
            client.update(
                index=INDEX_NAME,
                id=element_id,
                body={"doc": {"calls": calls}},
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
            index=INDEX_NAME,
            ids=element_ids,
            _source=["summary"],
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

        states = {}
        for hit in result["hits"]["hits"]:
            source = hit["_source"]
            states[source["relative_path"]] = {
                "file_hash": source.get("file_hash"),
                "is_deleted": False,
                "element_count": source.get("element_count"),
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

        return result["count"] > 0

    def get_all_embeddings(
        self,
        scope: str,
        repository: str,
        username: str,
        element_types: list[str] | None = None,
        embedding_type: str = "summary",
    ) -> list[dict[str, Any]]:
        """Fetch all elements with embeddings for clustering.

        Args:
            scope: Scope to filter by.
            repository: Repository to filter by.
            username: Username to filter by.
            element_types: Filter by element types (e.g., ["function", "method"]).
            embedding_type: Type of embedding to fetch - "summary" or "code".

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

        client = self._get_client()

        # Use scroll for large result sets
        results: list[dict[str, Any]] = []
        response = client.search(
            index=INDEX_NAME,
            body={
                "query": {"bool": {"must": must_clauses}},
                "size": 1000,
                "_source": ["element_id", field_name, "element_type", "name", "relative_path"],
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
