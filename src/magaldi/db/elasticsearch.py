"""Elasticsearch repository implementation for Magaldi.

Handles indexing of code elements and storage of embedding vectors.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from elasticsearch import Elasticsearch, NotFoundError

from magaldi.config import MagaldiConfig, get_config
from magaldi.parser.code_parser import CodeElement


# Index name for code elements
INDEX_NAME = "magaldi-code-elements"

# Index mapping with dense_vector for embeddings
INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "element_id": {"type": "keyword"},
            "scope": {"type": "keyword"},
            "repository": {"type": "keyword"},
            "username": {"type": "keyword"},
            "relative_path": {"type": "keyword"},
            "element_type": {"type": "keyword"},
            "name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "language": {"type": "keyword"},
            "line_start": {"type": "integer"},
            "line_end": {"type": "integer"},
            "raw_code": {"type": "text"},
            "signature": {"type": "text"},
            "docstring": {"type": "text"},
            "summary": {"type": "text"},
            "level": {"type": "integer"},
            "parent_id": {"type": "keyword"},
            "decorators": {"type": "keyword"},
            "visibility": {"type": "keyword"},
            "is_async": {"type": "boolean"},
            "indexed_at": {"type": "date"},
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
        self, element: CodeElement, indexed_at: datetime | None = None
    ) -> bool:
        """Index a code element.

        Args:
            element: Code element to index.
            indexed_at: Timestamp for indexing.

        Returns:
            True on success.
        """
        if indexed_at is None:
            indexed_at = datetime.now()

        doc = {
            "element_id": element.element_id,
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
            "indexed_at": indexed_at.isoformat(),
        }

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

    def search_by_vector(
        self,
        embedding: list[float],
        scope: str | None = None,
        repository: str | None = None,
        username: str | None = None,
        element_types: list[str] | None = None,
        size: int = 10,
        min_score: float = 0.7,
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
        filter_clauses: list[dict[str, Any]] = []

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
                "query": {"bool": {"filter": filter_clauses}} if filter_clauses else {"match_all": {}},
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


class ElasticsearchEmbeddingStore(ElasticsearchRepository):
    """Elasticsearch-backed embedding store that also reads from MySQL for summaries."""

    def __init__(self, config: MagaldiConfig | None = None):
        super().__init__(config)
        # Import here to avoid circular imports
        from magaldi.db.mysql import MySQLSummaryStore
        self._mysql_store = MySQLSummaryStore(config)

    def store_element(self, element: CodeElement) -> None:
        """Store a code element (index to ES)."""
        self.index_element(element)

    def get_element(self, element_id: str) -> CodeElement | None:
        """Get element by ID from MySQL."""
        return self._mysql_store.get_element(element_id)

    def get_summary(self, element_id: str) -> str | None:
        """Get summary from MySQL."""
        return self._mysql_store.get_summary(element_id)

    def get_file_summary(self, element: CodeElement) -> str | None:
        """Get file summary from MySQL."""
        summaries = self._mysql_store.get_parent_summaries(element)
        return summaries.get("file")

    def get_class_summary(self, element: CodeElement) -> str | None:
        """Get class summary from MySQL."""
        summaries = self._mysql_store.get_parent_summaries(element)
        return summaries.get("class")
