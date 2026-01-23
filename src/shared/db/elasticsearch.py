"""Elasticsearch repository implementation for Magaldi.

Handles indexing of code elements and storage of embedding vectors.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

from elasticsearch import Elasticsearch, NotFoundError

from shared.config import MagaldiConfig, get_config
from magaldi_core.code_parser import CodeElement


def generate_hash_id(element_id: str) -> str:
    """Generate a URL-safe hash ID from an element ID.

    Uses full SHA256 hex digest (64 characters, 256 bits).
    Guaranteed unique - no collision risk.

    Args:
        element_id: Full element ID string.

    Returns:
        64-character hex string suitable for URLs.
    """
    return hashlib.sha256(element_id.encode()).hexdigest()


# Index name for code elements
INDEX_NAME = "magaldi-code-elements"

# Index mapping with dense_vector for embeddings
INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "element_id": {"type": "keyword"},
            "hash_id": {"type": "keyword"},  # Short URL-safe ID for routing
            "scope": {"type": "keyword"},
            "repository": {"type": "keyword"},
            "username": {"type": "keyword"},
            "relative_path": {"type": "keyword"},
            "element_type": {"type": "keyword"},
            "name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "language": {"type": "keyword"},
            "line_start": {"type": "integer"},
            "line_end": {"type": "integer"},
            "raw_code": {
                "type": "text",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 32766,  # Max for keyword, allows regexp on full code
                    }
                }
            },
            "signature": {"type": "text"},
            "docstring": {"type": "text"},
            "summary": {"type": "text"},
            "level": {"type": "integer"},
            "parent_id": {"type": "keyword"},
            "decorators": {"type": "keyword"},
            "visibility": {"type": "keyword"},
            "is_async": {"type": "boolean"},
            "is_test": {"type": "boolean"},  # Whether element is test code
            "file_hash": {"type": "keyword"},  # For change detection (stored on all elements)
            "content_hash": {"type": "keyword"},  # SHA256 of raw_code for element-level change detection
            "element_count": {"type": "integer"},  # Total elements in file (only on file elements)
            "indexed_at": {"type": "date"},
            "cluster_id": {"type": "keyword"},  # Feature cluster ID
            "cluster_label": {"type": "keyword"},  # Human-readable cluster label
            "member_count": {"type": "integer"},  # Number of elements in feature (for feature docs)
            "member_ids": {"type": "keyword"},  # Element IDs of members (for feature docs)
            "parent_feature_label": {"type": "keyword"},  # Parent feature label (for subfeatures)
            "parent_feature_summary": {"type": "text"},  # Parent feature summary (for subfeatures)
            "term": {"type": "keyword"},  # Glossary term
            "total_count": {"type": "integer"},  # Glossary term occurrence count
            "file_paths": {"type": "keyword"},  # Array of file paths
            "feature_associations": {"type": "object"},  # Feature linking data
            "updated_at": {"type": "date"},  # Update timestamp
            "embedding": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine",
            },
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
}


class ElasticsearchRepository:
    """Elasticsearch repository for code element indexing and search."""

    def __init__(self, config: MagaldiConfig | None = None):
        self._config = config or get_config()
        self._client: Elasticsearch | None = None

    def _get_client(self) -> Elasticsearch:
        """Get or create Elasticsearch client."""
        if self._client is None:
            es_config = self._config.elasticsearch
            self._client = Elasticsearch(
                hosts=[{
                    "host": es_config.host,
                    "port": es_config.port,
                    "scheme": es_config.scheme,
                }],
            )
            # Ensure index exists
            self._ensure_index()
        return self._client

    def _ensure_index(self) -> None:
        """Create index if it doesn't exist."""
        client = self._client
        if client and not client.indices.exists(index=INDEX_NAME):
            client.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)

    def close(self) -> None:
        """Close Elasticsearch client."""
        if self._client:
            self._client.close()
            self._client = None

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

    def store_embedding(self, element_id: str, embedding: list[float]) -> bool:
        """Store embedding vector for an element.

        Args:
            element_id: Element ID to update.
            embedding: Vector embedding (1024 dimensions).

        Returns:
            True on success.
        """
        try:
            client = self._get_client()
            client.update(
                index=INDEX_NAME,
                id=element_id,
                body={"doc": {"embedding": embedding}},
            )
            return True
        except NotFoundError:
            return False

    def get_embedding(self, element_id: str) -> list[float] | None:
        """Get embedding vector for an element."""
        doc = self.get_document(element_id)
        if doc:
            return doc.get("embedding")
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

        if scope:
            must_clauses.append({"term": {"scope": scope}})
        if repository:
            must_clauses.append({"term": {"repository": repository}})
        if username:
            must_clauses.append({"term": {"username": username}})
        if element_types:
            must_clauses.append({"terms": {"element_type": element_types}})

        client = self._get_client()
        result = client.search(
            index=INDEX_NAME,
            body={"query": {"bool": {"must": must_clauses}}, "size": size},
        )

        return [hit["_source"] for hit in result["hits"]["hits"]]

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

    def search_by_vector(
        self,
        embedding: list[float],
        scope: str | None = None,
        repository: str | None = None,
        username: str | None = None,
        element_types: list[str] | None = None,
        size: int = 10,
        min_score: float = 0.3,
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

        Returns:
            List of matching documents with scores.
        """
        filter_clauses: list[dict[str, Any]] = [
            # Only search documents that have embeddings
            {"exists": {"field": "embedding"}}
        ]

        if scope:
            filter_clauses.append({"term": {"scope": scope}})
        if repository:
            filter_clauses.append({"term": {"repository": repository}})
        if username:
            filter_clauses.append({"term": {"username": username}})
        if element_types:
            filter_clauses.append({"terms": {"element_type": element_types}})

        query: dict[str, Any] = {
            "script_score": {
                "query": {"bool": {"filter": filter_clauses}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": embedding},
                },
            }
        }

        client = self._get_client()
        result = client.search(
            index=INDEX_NAME,
            body={"query": query, "size": size, "min_score": min_score + 1.0},
        )

        return [
            {**hit["_source"], "_score": hit["_score"] - 1.0}
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
        if username:
            filter_clauses.append({"term": {"username": username}})
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

        es_query: dict[str, Any] = {
            "bool": {
                "must": [must_clause],
            }
        }

        if filter_clauses:
            es_query["bool"]["filter"] = filter_clauses

        client = self._get_client()
        result = client.search(
            index=INDEX_NAME,
            body={"query": es_query, "size": size},
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
        """Search raw_code field using Elasticsearch regexp query.

        Uses Lucene regexp syntax (not Python re). Key differences:
        - . matches any character (no need to escape)
        - * is zero or more of previous
        - .* matches any string
        - No lookahead/lookbehind support

        Note: Patterns are automatically wrapped with .* for substring matching
        (Lucene regexp requires matching the entire field by default).

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

        if username:
            filter_clauses.append({"term": {"username": username}})
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
        """Search raw_code field using Elasticsearch wildcard query.

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

        if username:
            filter_clauses.append({"term": {"username": username}})
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

    def get_all_embeddings(
        self,
        scope: str,
        repository: str,
        username: str,
        element_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch all elements with embeddings for clustering.

        Args:
            scope: Scope to filter by.
            repository: Repository to filter by.
            username: Username to filter by.
            element_types: Filter by element types (e.g., ["function", "method"]).

        Returns:
            List of dicts with element_id, embedding, element_type, name, relative_path.
        """
        must_clauses: list[dict[str, Any]] = [
            {"term": {"scope": scope}},
            {"term": {"repository": repository}},
            {"term": {"username": username}},
            {"exists": {"field": "embedding"}},
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
                "_source": ["element_id", "embedding", "element_type", "name", "relative_path"],
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

    def update_cluster_assignments(
        self,
        assignments: list[dict[str, Any]],
    ) -> int:
        """Bulk update cluster assignments for elements.

        Args:
            assignments: List of {element_id, cluster_id, cluster_label}.

        Returns:
            Number of elements updated.
        """
        if not assignments:
            return 0

        client = self._get_client()

        # Build bulk update body
        bulk_body: list[dict[str, Any]] = []
        for assignment in assignments:
            bulk_body.append({
                "update": {
                    "_index": INDEX_NAME,
                    "_id": assignment["element_id"],
                }
            })
            bulk_body.append({
                "doc": {
                    "cluster_id": assignment["cluster_id"],
                    "cluster_label": assignment.get("cluster_label"),
                }
            })

        response = client.bulk(body=bulk_body, refresh=True)

        # Count successful updates
        updated = 0
        for item in response.get("items", []):
            if item.get("update", {}).get("result") in ["updated", "noop"]:
                updated += 1

        return updated

    def get_clusters(
        self,
        scope: str,
        repository: str,
        username: str,
    ) -> list[dict[str, Any]]:
        """Get all clusters with their elements.

        Args:
            scope: Scope to filter by.
            repository: Repository to filter by.
            username: Username to filter by.

        Returns:
            List of cluster dicts with cluster_id, cluster_label, element_count, elements.
        """
        client = self._get_client()

        # Aggregate by cluster_id
        response = client.search(
            index=INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"scope": scope}},
                            {"term": {"repository": repository}},
                            {"term": {"username": username}},
                            {"exists": {"field": "cluster_id"}},
                        ]
                    }
                },
                "size": 0,
                "aggs": {
                    "clusters": {
                        "terms": {
                            "field": "cluster_id",
                            "size": 1000,
                        },
                        "aggs": {
                            "cluster_label": {
                                "terms": {
                                    "field": "cluster_label",
                                    "size": 1,
                                }
                            },
                            "sample_elements": {
                                "top_hits": {
                                    "size": 5,
                                    "_source": ["element_id", "name", "element_type", "relative_path"],
                                }
                            },
                        },
                    }
                },
            },
        )

        clusters = []
        for bucket in response["aggregations"]["clusters"]["buckets"]:
            cluster_id = bucket["key"]
            label_buckets = bucket["cluster_label"]["buckets"]
            cluster_label = label_buckets[0]["key"] if label_buckets else None

            sample_elements = [
                hit["_source"] for hit in bucket["sample_elements"]["hits"]["hits"]
            ]

            clusters.append({
                "cluster_id": cluster_id,
                "cluster_label": cluster_label,
                "element_count": bucket["doc_count"],
                "elements": sample_elements,
            })

        return clusters

    def clear_cluster_assignments(
        self,
        scope: str,
        repository: str,
        username: str,
    ) -> int:
        """Clear all cluster assignments for a repository.

        Args:
            scope: Scope to filter by.
            repository: Repository to filter by.
            username: Username to filter by.

        Returns:
            Number of elements updated.
        """
        client = self._get_client()

        response = client.update_by_query(
            index=INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"scope": scope}},
                            {"term": {"repository": repository}},
                            {"term": {"username": username}},
                            {"exists": {"field": "cluster_id"}},
                        ]
                    }
                },
                "script": {
                    "source": "ctx._source.remove('cluster_id'); ctx._source.remove('cluster_label')",
                    "lang": "painless",
                },
            },
            refresh=True,
        )

        return response.get("updated", 0)

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
    ) -> bool:
        """Index a feature document.

        Args:
            feature_id: Unique feature ID.
            scope: Repository scope.
            repository: Repository name.
            username: Username/branch.
            cluster_id: Cluster ID from HDBSCAN.
            label: Feature label (e.g., "authentication").
            summary: Generated summary of the feature.
            embedding: Embedding vector for the feature.
            member_ids: List of element IDs belonging to this feature.

        Returns:
            True on success.
        """
        from datetime import datetime

        doc: dict[str, Any] = {
            "element_id": feature_id,
            "hash_id": generate_hash_id(feature_id),
            "scope": scope,
            "repository": repository,
            "username": username,
            "element_type": "feature",
            "name": label,
            "cluster_id": cluster_id,
            "cluster_label": label,
            "summary": summary,
            "member_count": len(member_ids),
            "member_ids": member_ids,
            "indexed_at": datetime.now().isoformat(),
            "level": -1,  # Features are above files in hierarchy
        }

        if embedding is not None:
            doc["embedding"] = embedding

        client = self._get_client()
        client.index(index=INDEX_NAME, id=feature_id, document=doc)
        return True

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
        """Index a subfeature document.

        Args:
            subfeature_id: Unique subfeature ID.
            scope: Repository scope.
            repository: Repository name.
            username: Username/branch.
            cluster_id: Sub-cluster ID from HDBSCAN.
            label: Subfeature label.
            summary: Generated summary of the subfeature.
            embedding: Embedding vector for the subfeature.
            member_ids: List of element IDs belonging to this subfeature.
            parent_feature_label: Label of the parent feature.
            parent_feature_summary: Summary of the parent feature.

        Returns:
            True on success.
        """
        from datetime import datetime

        doc: dict[str, Any] = {
            "element_id": subfeature_id,
            "hash_id": generate_hash_id(subfeature_id),
            "scope": scope,
            "repository": repository,
            "username": username,
            "element_type": "subfeature",
            "name": label,
            "cluster_id": cluster_id,
            "cluster_label": label,
            "summary": summary,
            "member_count": len(member_ids),
            "member_ids": member_ids,
            "parent_feature_label": parent_feature_label,
            "parent_feature_summary": parent_feature_summary,
            "indexed_at": datetime.now().isoformat(),
            "level": -2,  # Subfeatures are below features in hierarchy
        }

        if embedding is not None:
            doc["embedding"] = embedding

        client = self._get_client()
        client.index(index=INDEX_NAME, id=subfeature_id, document=doc)
        return True

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
            "feature_associations": [],
            "indexed_at": datetime.now().isoformat(),
            "level": -3,  # Glossary terms are at the highest conceptual level
        }

        client = self._get_client()
        client.index(index=INDEX_NAME, id=glossary_id, document=doc)
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
            query=query,
            size=1000,
            sort=[{"total_count": "desc"}],
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
            response = client.get(index=INDEX_NAME, id=glossary_id)
            if response.get("found"):
                source = response.get("_source", {})
                return {
                    "glossary_id": glossary_id,
                    "term": source.get("term"),
                    "total_count": source.get("total_count"),
                    "element_ids": source.get("element_ids", []),
                    "file_paths": source.get("file_paths", []),
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
            query=es_query,
            size=100,
            sort=[{"total_count": "desc"}],
        )

        results: list[dict[str, Any]] = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            results.append({
                "glossary_id": hit.get("_id"),
                "term": source.get("term"),
                "total_count": source.get("total_count"),
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

        client.update(
            index=INDEX_NAME,
            id=glossary_id,
            doc={
                "feature_associations": feature_associations,
                "updated_at": datetime.now().isoformat(),
            },
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

        response = client.delete_by_query(
            index=INDEX_NAME,
            query={
                "bool": {
                    "must": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                        {"term": {"element_type": "glossary"}},
                    ]
                }
            },
            refresh=True,
        )

        return response.get("deleted", 0)

    def delete_features(
        self,
        scope: str,
        repository: str,
        username: str,
    ) -> int:
        """Delete all feature documents for a repository.

        Args:
            scope: Scope to filter by.
            repository: Repository to filter by.
            username: Username to filter by.

        Returns:
            Number of features deleted.
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
                            {"term": {"element_type": "feature"}},
                        ]
                    }
                }
            },
            refresh=True,
        )

        return response.get("deleted", 0)

    def delete_subfeatures(
        self,
        scope: str,
        repository: str,
        username: str,
    ) -> int:
        """Delete all subfeature documents for a repository.

        Args:
            scope: Scope to filter by.
            repository: Repository to filter by.
            username: Username to filter by.

        Returns:
            Number of subfeatures deleted.
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
                            {"term": {"element_type": "subfeature"}},
                        ]
                    }
                }
            },
            refresh=True,
        )

        return response.get("deleted", 0)

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
            try:
                client.clear_scroll(scroll_id=scroll_id)
            except Exception:
                pass

        return results

    def get_features(
        self,
        scope: str,
        repository: str,
        username: str = "main",
    ) -> list[dict[str, Any]]:
        """Get all features for a repository.

        Args:
            scope: Repository scope.
            repository: Repository name.
            username: User branch.

        Returns:
            List of features with summaries and member counts.
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
                            {"term": {"element_type": "feature"}},
                        ]
                    }
                },
                "size": 100,
                "sort": [{"member_count": "desc"}],
                "_source": [
                    "element_id",
                    "cluster_id",
                    "cluster_label",
                    "summary",
                    "member_count",
                    "member_ids",
                ],
            },
        )

        features = []
        for hit in result.get("hits", {}).get("hits", []):
            source = hit["_source"]
            features.append({
                "feature_id": source.get("element_id"),
                "label": source.get("cluster_label"),
                "summary": source.get("summary", ""),
                "member_count": source.get("member_count", 0),
                "member_ids": source.get("member_ids", []),
            })

        return features

    def get_subfeatures(
        self,
        scope: str,
        repository: str,
        username: str = "main",
        parent_feature_label: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get all subfeatures for a repository.

        Args:
            scope: Repository scope.
            repository: Repository name.
            username: User branch.
            parent_feature_label: Optional filter by parent feature label.

        Returns:
            List of subfeatures with parent feature info.
        """
        client = self._get_client()

        filters = [
            {"term": {"scope": scope}},
            {"term": {"repository": repository}},
            {"term": {"username": username}},
            {"term": {"element_type": "subfeature"}},
        ]

        if parent_feature_label:
            filters.append({"term": {"parent_feature_label": parent_feature_label}})

        result = client.search(
            index=INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "filter": filters
                    }
                },
                "size": 500,
                "sort": [{"member_count": "desc"}],
                "_source": [
                    "element_id",
                    "cluster_id",
                    "cluster_label",
                    "summary",
                    "member_count",
                    "member_ids",
                    "parent_feature_label",
                    "parent_feature_summary",
                ],
            },
        )

        subfeatures = []
        for hit in result.get("hits", {}).get("hits", []):
            source = hit["_source"]
            subfeatures.append({
                "subfeature_id": source.get("element_id"),
                "label": source.get("cluster_label"),
                "summary": source.get("summary", ""),
                "member_count": source.get("member_count", 0),
                "member_ids": source.get("member_ids", []),
                "parent_feature_label": source.get("parent_feature_label", ""),
                "parent_feature_summary": source.get("parent_feature_summary", ""),
            })

        return subfeatures

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
                    actual_count = self._es.count_elements_by_file_hash(
                        scope, repository, username, file_hash
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
        self._elements: dict[str, CodeElement] = {}

    def store_element(self, element: CodeElement, file_hash: str | None = None) -> None:
        """Store a code element (index to ES)."""
        self._elements[element.element_id] = element
        self.index_element(element, file_hash=file_hash)

    def get_element(self, element_id: str) -> CodeElement | None:
        """Get element by ID (from cache or reconstruct from ES)."""
        if element_id in self._elements:
            return self._elements[element_id]

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
