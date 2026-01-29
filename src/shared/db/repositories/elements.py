"""Element repository for indexing and managing code elements.

Handles CRUD operations for code elements in Elasticsearch.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from elasticsearch import NotFoundError

from magaldi_core.code_parser import CodeElement

from .base import INDEX_NAME, ElasticsearchBase, generate_hash_id


class ElementRepository:
    """Repository for code element indexing and management."""

    def __init__(self, base: ElasticsearchBase):
        self._base = base

    def _get_client(self) -> Any:
        """Get Elasticsearch client from base."""
        return self._base._get_client()

    def index_element(
        self,
        element: CodeElement,
        indexed_at: datetime | None = None,
        file_hash: str | None = None,
        element_count: int | None = None,
    ) -> bool:
        """Index a code element.

        Args:
            element: Code element to index.
            indexed_at: Timestamp for indexing.
            file_hash: File hash for change detection (stored on all elements).
            element_count: Total element count in file (only for file-level elements).

        Returns:
            True on success.
        """
        if indexed_at is None:
            indexed_at = datetime.now()

        doc = {
            "element_id": element.element_id,
            "hash_id": generate_hash_id(element.element_id),
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
            "is_test": element.is_test,
            "content_hash": element.content_hash,  # For element-level change detection
            "indexed_at": indexed_at.isoformat(),
        }

        # Add file_hash for all elements (used for change detection and cleanup)
        if file_hash is not None:
            doc["file_hash"] = file_hash

        # Add element_count for file-level elements (used for completeness verification)
        if element_count is not None:
            doc["element_count"] = element_count

        # Add enhanced context fields if present
        if element.class_attributes:
            doc["class_attributes"] = element.class_attributes
        if element.base_classes:
            doc["base_classes"] = element.base_classes
        if element.exceptions_raised:
            doc["exceptions_raised"] = element.exceptions_raised
        if element.attributes_modified:
            doc["attributes_modified"] = element.attributes_modified
        if element.decorator_details:
            doc["decorator_details"] = element.decorator_details
        if element.return_type:
            doc["return_type"] = element.return_type
        if element.parameters:
            doc["parameters"] = element.parameters

        # Extended code intelligence fields
        if element.type_annotations:
            doc["type_annotations"] = element.type_annotations
        if element.detected_patterns:
            doc["detected_patterns"] = element.detected_patterns
        if element.pattern_confidence:
            doc["pattern_confidence"] = element.pattern_confidence
        if element.todos:
            doc["todos"] = element.todos
        if element.section_markers:
            doc["section_markers"] = element.section_markers
        if element.associated_comments:
            doc["associated_comments"] = element.associated_comments
        if element.is_public_api:
            doc["is_public_api"] = element.is_public_api
        if element.http_routes:
            doc["http_routes"] = element.http_routes
        if element.cli_commands:
            doc["cli_commands"] = element.cli_commands
        if element.purity:
            doc["purity"] = element.purity
        if element.side_effects:
            doc["side_effects"] = element.side_effects
        if element.mutated_state:
            doc["mutated_state"] = element.mutated_state

        # Code metrics fields
        if element.complexity:
            doc["complexity"] = element.complexity
        if element.code_metrics:
            doc["code_metrics"] = element.code_metrics
        if element.docstring_quality:
            doc["docstring_quality"] = element.docstring_quality
        if element.security_issues:
            doc["security_issues"] = element.security_issues
        if element.env_vars:
            doc["env_vars"] = element.env_vars
        if element.concurrency:
            doc["concurrency"] = element.concurrency
        if element.metrics_summary:
            doc["metrics_summary"] = element.metrics_summary

        client = self._get_client()
        client.index(index=INDEX_NAME, id=element.element_id, document=doc)
        return True

    def get_document(self, element_id: str) -> dict[str, Any] | None:
        """Get indexed document by ID."""
        try:
            client = self._get_client()
            result = client.get(index=INDEX_NAME, id=element_id)
            return result["_source"]
        except NotFoundError:
            return None

    def get_document_by_hash_id(self, hash_id: str) -> dict[str, Any] | None:
        """Get indexed document by hash_id.

        Args:
            hash_id: The 64-character SHA256 hash of the element_id.

        Returns:
            Document source or None if not found.
        """
        try:
            client = self._get_client()
            result = client.search(
                index=INDEX_NAME,
                body={
                    "size": 1,
                    "query": {"term": {"hash_id": hash_id}},
                },
            )
            hits = result.get("hits", {}).get("hits", [])
            if hits:
                return hits[0]["_source"]
            return None
        except Exception:
            return None

    def get_document_by_id_or_hash(self, id_or_hash: str) -> dict[str, Any] | None:
        """Get indexed document by either element_id or hash_id.

        Automatically detects which type of ID is provided:
        - 64-char hex string without colons = hash_id
        - Contains colons = element_id

        Args:
            id_or_hash: Either an element_id or hash_id.

        Returns:
            Document source or None if not found.
        """
        # Detect if it's a hash_id (64 hex chars, no colons) or element_id
        is_hash = len(id_or_hash) == 64 and ":" not in id_or_hash and all(c in "0123456789abcdef" for c in id_or_hash.lower())

        if is_hash:
            return self.get_document_by_hash_id(id_or_hash)
        else:
            return self.get_document(id_or_hash)

    def element_exists(self, element_id: str) -> bool:
        """Check if element exists in ES (meaning it's fully processed).

        Args:
            element_id: Element ID to check.

        Returns:
            True if element exists (fully processed).
        """
        try:
            client = self._get_client()
            return client.exists(index=INDEX_NAME, id=element_id)
        except Exception:
            return False

    def get_existing_element_ids(self, element_ids: list[str]) -> set[str]:
        """Check which elements already exist in ES.

        Args:
            element_ids: List of element IDs to check.

        Returns:
            Set of element IDs that exist.
        """
        if not element_ids:
            return set()

        client = self._get_client()

        # Use mget for efficient batch lookup
        response = client.mget(index=INDEX_NAME, ids=element_ids, _source=False)

        return {
            doc["_id"] for doc in response["docs"] if doc.get("found", False)
        }

    def get_element_content_hashes(self, element_ids: list[str]) -> dict[str, str | None]:
        """Get content hashes for existing elements.

        Args:
            element_ids: List of element IDs to check.

        Returns:
            Dict mapping element_id to content_hash (None if element doesn't exist or has no hash).
        """
        if not element_ids:
            return {}

        client = self._get_client()

        # Use mget for efficient batch lookup, only fetch content_hash
        response = client.mget(index=INDEX_NAME, ids=element_ids, _source=["content_hash"])

        result = {}
        for doc in response["docs"]:
            if doc.get("found", False):
                result[doc["_id"]] = doc.get("_source", {}).get("content_hash")

        return result

    def get_element_ids_by_file(
        self, scope: str, repository: str, username: str, relative_path: str
    ) -> set[str]:
        """Get all element IDs for a specific file.

        Args:
            scope: Repository scope.
            repository: Repository name.
            username: Username.
            relative_path: File path relative to repo root.

        Returns:
            Set of element IDs belonging to this file.
        """
        client = self._get_client()
        result = client.search(
            index=INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"scope": scope}},
                            {"term": {"repository": repository}},
                            {"term": {"username": username}},
                            {"term": {"relative_path": relative_path}},
                        ]
                    }
                },
                "_source": False,
                "size": 10000,  # Should be enough for any file
            },
        )
        return {hit["_id"] for hit in result.get("hits", {}).get("hits", [])}

    def delete_elements(self, element_ids: list[str]) -> int:
        """Delete specific elements by their IDs.

        Args:
            element_ids: List of element IDs to delete.

        Returns:
            Number of elements deleted.
        """
        if not element_ids:
            return 0

        client = self._get_client()
        # Use bulk delete for efficiency
        from elasticsearch.helpers import bulk

        actions = [
            {"_op_type": "delete", "_index": INDEX_NAME, "_id": eid}
            for eid in element_ids
        ]
        success, _ = bulk(client, actions, raise_on_error=False, refresh=True)
        return success

    def delete_by_file(
        self, scope: str, repository: str, username: str, relative_path: str
    ) -> int:
        """Delete all documents for a file."""
        client = self._get_client()
        result = client.delete_by_query(
            index=INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"scope": scope}},
                            {"term": {"repository": repository}},
                            {"term": {"username": username}},
                            {"term": {"relative_path": relative_path}},
                        ]
                    }
                }
            },
            refresh=True,
        )
        return result.get("deleted", 0)

    def count_elements_by_file_hash(
        self, scope: str, repository: str, username: str, file_hash: str
    ) -> int:
        """Count elements with a specific file hash.

        Used for completeness verification - compare to expected element_count.
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
                            {"term": {"username": username}},
                            {"term": {"file_hash": file_hash}},
                        ]
                    }
                }
            },
        )
        return result.get("count", 0)

    def delete_by_file_hash(
        self, scope: str, repository: str, username: str, file_hash: str
    ) -> int:
        """Delete all elements with a specific file hash.

        Used for cleanup when file needs to be reprocessed.
        """
        client = self._get_client()
        result = client.delete_by_query(
            index=INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"scope": scope}},
                            {"term": {"repository": repository}},
                            {"term": {"username": username}},
                            {"term": {"file_hash": file_hash}},
                        ]
                    }
                }
            },
            refresh=True,
        )
        return result.get("deleted", 0)

    def delete_by_repository(self, scope: str, repository: str, username: str) -> int:
        """Delete all elements for a repository/user combination.

        Returns:
            Number of documents deleted.
        """
        client = self._get_client()
        response = client.delete_by_query(
            index=INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"scope": scope}},
                            {"term": {"repository": repository}},
                            {"term": {"username": username}},
                        ]
                    }
                }
            },
            refresh=True,
        )
        return response.get("deleted", 0)

    def update_file_hashes(
        self,
        scope: str,
        repository: str,
        username: str,
        file_updates: dict[str, str],
    ) -> int:
        """Bulk update file_hash for elements by relative_path.

        Used when elements are unchanged (content_hash match) but file_hash
        changed. Without this update, files would appear as "modified" on
        every subsequent run.

        Args:
            scope: Scope to filter by.
            repository: Repository to filter by.
            username: Username to filter by.
            file_updates: Dict mapping relative_path -> new file_hash.

        Returns:
            Number of elements updated.
        """
        if not file_updates:
            return 0

        client = self._get_client()
        total_updated = 0

        # Update by query for each file path
        _logged = 0
        for relative_path, new_file_hash in file_updates.items():
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
                            ]
                        }
                    },
                    "script": {
                        "source": "ctx._source.file_hash = params.file_hash",
                        "params": {"file_hash": new_file_hash},
                    },
                },
                refresh=True,
            )
            updated = result.get("updated", 0)
            total_updated += updated

            # Log first few files with details
            if _logged < 3:
                with open("/tmp/magaldi_file_hash.log", "a") as f:
                    f.write(f"[ES UPDATE] {relative_path}: updated={updated}, total={result.get('total', 0)}, failures={result.get('failures', [])}\n")

                # Verify FILE element was updated
                verify = client.search(
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
                        "_source": ["file_hash", "element_type"],
                        "size": 1,
                    },
                )
                hits = verify.get("hits", {}).get("hits", [])
                if hits:
                    src = hits[0].get("_source", {})
                    with open("/tmp/magaldi_file_hash.log", "a") as f:
                        f.write(f"[VERIFY] FILE element file_hash={src.get('file_hash', 'MISSING')[:16] if src.get('file_hash') else 'None'}...\n")
                else:
                    with open("/tmp/magaldi_file_hash.log", "a") as f:
                        f.write(f"[VERIFY] FILE element NOT FOUND for {relative_path}\n")

                _logged += 1

        return total_updated

    def get_documents_batch(
        self,
        element_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Get full documents for multiple elements in batch.

        Args:
            element_ids: List of element IDs to fetch.

        Returns:
            Dict mapping element_id to document (only for found elements).
        """
        if not element_ids:
            return {}

        client = self._get_client()

        # Use mget for efficient batch lookup
        response = client.mget(
            index=INDEX_NAME,
            ids=element_ids,
        )

        result: dict[str, dict[str, Any]] = {}
        for doc in response.get("docs", []):
            if doc.get("found") and doc.get("_source"):
                result[doc["_id"]] = doc["_source"]

        return result
