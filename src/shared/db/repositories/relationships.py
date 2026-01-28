"""Repository for code relationships (knowledge graph edges).

Handles CRUD operations and querying for relationships between code elements,
with support for multi-user branching (user relationships shadow main).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .base import (
    EXTERNAL_REFS_INDEX_NAME,
    RELATIONSHIPS_INDEX_NAME,
    ElasticsearchBase,
)

if TYPE_CHECKING:
    from magaldi_core.analysis.relationships import CodeRelationship, ExternalReference


class RelationshipsRepository(ElasticsearchBase):
    """Repository for querying and storing code relationships."""

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def index_relationship(self, relationship: CodeRelationship) -> None:
        """Index a single relationship."""
        client = self._get_client()
        doc = {
            "relationship_id": relationship.relationship_id,
            "relationship_key": relationship.relationship_key,
            "source_hash": relationship.source_hash,
            "target_hash": relationship.target_hash,
            "relationship_type": relationship.relationship_type.value,
            "scope": relationship.scope,
            "repository": relationship.repository,
            "username": relationship.username,
            "priority": relationship.priority,
            "confidence": relationship.confidence,
            "line": relationship.line,
            "details": relationship.details,
            "indexed_at": datetime.now(UTC).isoformat(),
        }
        client.index(
            index=RELATIONSHIPS_INDEX_NAME,
            id=relationship.relationship_id,
            document=doc,
        )

    def bulk_index_relationships(
        self, relationships: list[CodeRelationship]
    ) -> dict[str, int]:
        """Bulk index multiple relationships.

        Returns:
            Dict with 'indexed' and 'errors' counts.
        """
        if not relationships:
            return {"indexed": 0, "errors": 0}

        client = self._get_client()
        operations = []
        now = datetime.now(UTC).isoformat()

        for rel in relationships:
            operations.append(
                {"index": {"_index": RELATIONSHIPS_INDEX_NAME, "_id": rel.relationship_id}}
            )
            operations.append({
                "relationship_id": rel.relationship_id,
                "relationship_key": rel.relationship_key,
                "source_hash": rel.source_hash,
                "target_hash": rel.target_hash,
                "relationship_type": rel.relationship_type.value,
                "scope": rel.scope,
                "repository": rel.repository,
                "username": rel.username,
                "priority": rel.priority,
                "confidence": rel.confidence,
                "line": rel.line,
                "details": rel.details,
                "indexed_at": now,
            })

        result = client.bulk(operations=operations, refresh=True)
        errors = sum(1 for item in result["items"] if "error" in item.get("index", {}))
        return {"indexed": len(relationships) - errors, "errors": errors}

    def delete_relationships_for_user(
        self, scope: str, repository: str, username: str
    ) -> int:
        """Delete all relationships for a specific user.

        Used before re-indexing a user's branch.

        Returns:
            Number of deleted relationships.
        """
        client = self._get_client()
        result = client.delete_by_query(
            index=RELATIONSHIPS_INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"scope": scope}},
                            {"term": {"repository": repository}},
                            {"term": {"username": username}},
                        ]
                    }
                }
            },
            refresh=True,
        )
        return result.get("deleted", 0)

    # =========================================================================
    # Query Operations (with user priority)
    # =========================================================================

    def get_relationships(
        self,
        scope: str,
        repository: str,
        username: str,
        source_hash: str | None = None,
        target_hash: str | None = None,
        relationship_types: list[str] | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get relationships, preferring user's over main's.

        Uses ES collapse to deduplicate by relationship_key,
        keeping highest priority (user > main).

        Args:
            scope: Repository scope
            repository: Repository name
            username: User to query for (will also include main)
            source_hash: Filter by source element hash_id
            target_hash: Filter by target element hash_id
            relationship_types: Filter by relationship types
            limit: Maximum results

        Returns:
            List of relationship documents (user's shadow main's)
        """
        client = self._get_client()

        # Build query
        filters = [
            {"term": {"scope": scope}},
            {"term": {"repository": repository}},
            {"terms": {"username": [username, "main"]}},
        ]

        if relationship_types:
            filters.append({"terms": {"relationship_type": relationship_types}})

        should_clauses = []
        if source_hash:
            should_clauses.append({"term": {"source_hash": source_hash}})
        if target_hash:
            should_clauses.append({"term": {"target_hash": target_hash}})

        query: dict[str, Any] = {"bool": {"filter": filters}}
        if should_clauses:
            query["bool"]["should"] = should_clauses
            query["bool"]["minimum_should_match"] = 1

        body = {
            "query": query,
            # Collapse by relationship_key, keep highest priority
            "collapse": {
                "field": "relationship_key",
            },
            "sort": [
                {"priority": {"order": "desc"}},  # User first (1), then main (0)
                {"indexed_at": {"order": "desc"}},
            ],
            "size": limit,
        }

        result = client.search(index=RELATIONSHIPS_INDEX_NAME, body=body)
        return [hit["_source"] for hit in result["hits"]["hits"]]

    def get_relationships_by_type(
        self,
        scope: str,
        repository: str,
        username: str,
        relationship_type: str,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get all relationships of a specific type.

        Args:
            scope: Repository scope
            repository: Repository name
            username: User to query for
            relationship_type: Type of relationship (e.g., "belongs_to_group")
            limit: Maximum results

        Returns:
            List of relationship documents
        """
        return self.get_relationships(
            scope=scope,
            repository=repository,
            username=username,
            relationship_types=[relationship_type],
            limit=limit,
        )

    def get_outgoing_relationships(
        self,
        source_hash: str,
        scope: str,
        repository: str,
        username: str,
        relationship_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get all relationships FROM an element."""
        return self.get_relationships(
            scope=scope,
            repository=repository,
            username=username,
            source_hash=source_hash,
            relationship_types=relationship_types,
        )

    def get_incoming_relationships(
        self,
        target_hash: str,
        scope: str,
        repository: str,
        username: str,
        relationship_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get all relationships TO an element."""
        return self.get_relationships(
            scope=scope,
            repository=repository,
            username=username,
            target_hash=target_hash,
            relationship_types=relationship_types,
        )

    # =========================================================================
    # External References
    # =========================================================================

    def index_external_ref(self, ref: ExternalReference) -> None:
        """Index an external reference."""
        client = self._get_client()
        doc = {
            "ref_id": ref.ref_id,
            "ref_type": ref.ref_type.value,
            "name": ref.name,
            "name_text": ref.name,  # For text search
            "description": ref.description,
            "scope": ref.scope,
            "repository": ref.repository,
            "username": ref.username,
            "usages_count": ref.usages_count,
            "indexed_at": datetime.now(UTC).isoformat(),
        }
        client.index(
            index=EXTERNAL_REFS_INDEX_NAME,
            id=ref.ref_id,
            document=doc,
        )

    def bulk_index_external_refs(
        self, refs: list[ExternalReference]
    ) -> dict[str, int]:
        """Bulk index external references."""
        if not refs:
            return {"indexed": 0, "errors": 0}

        client = self._get_client()
        operations = []
        now = datetime.now(UTC).isoformat()

        for ref in refs:
            operations.append(
                {"index": {"_index": EXTERNAL_REFS_INDEX_NAME, "_id": ref.ref_id}}
            )
            operations.append({
                "ref_id": ref.ref_id,
                "ref_type": ref.ref_type.value,
                "name": ref.name,
                "name_text": ref.name,
                "description": ref.description,
                "scope": ref.scope,
                "repository": ref.repository,
                "username": ref.username,
                "usages_count": ref.usages_count,
                "indexed_at": now,
            })

        result = client.bulk(operations=operations, refresh=True)
        errors = sum(1 for item in result["items"] if "error" in item.get("index", {}))
        return {"indexed": len(refs) - errors, "errors": errors}

    def get_external_refs(
        self,
        scope: str,
        repository: str,
        ref_type: str | None = None,
        search: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get external references with optional filtering."""
        client = self._get_client()

        filters = [
            {"term": {"scope": scope}},
            {"term": {"repository": repository}},
        ]

        if ref_type:
            filters.append({"term": {"ref_type": ref_type}})

        query: dict[str, Any] = {"bool": {"filter": filters}}

        if search:
            query["bool"]["must"] = {"match": {"name_text": search}}

        result = client.search(
            index=EXTERNAL_REFS_INDEX_NAME,
            body={
                "query": query,
                "sort": [{"usages_count": {"order": "desc"}}],
                "size": limit,
            },
        )
        return [hit["_source"] for hit in result["hits"]["hits"]]

    def delete_external_refs_for_user(
        self, scope: str, repository: str, username: str
    ) -> int:
        """Delete all external refs for a specific user."""
        client = self._get_client()
        result = client.delete_by_query(
            index=EXTERNAL_REFS_INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"scope": scope}},
                            {"term": {"repository": repository}},
                            {"term": {"username": username}},
                        ]
                    }
                }
            },
            refresh=True,
        )
        return result.get("deleted", 0)
