"""Feature repository for managing features, subfeatures, and clustering.

Handles indexing and retrieval of feature documents and cluster management.
"""

from __future__ import annotations

import contextlib
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .base import INDEX_NAME, RepositoryBase, generate_hash_id

if TYPE_CHECKING:
    from shared.db.bulk_buffer import BulkIndexBuffer


class FeatureRepository:
    """Repository for feature and subfeature operations."""

    def __init__(self, base: RepositoryBase):
        self._base = base
        self._bulk_buffer: BulkIndexBuffer | None = None

    def _get_client(self) -> Any:
        """Get search client from base."""
        return self._base._get_client()

    def _get_bulk_timeout(self) -> int:
        """Get bulk operation timeout from config."""
        return self._base._config.search_backend.bulk_timeout  # type: ignore[no-any-return]

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
            connected_features: Optional list of connected features from soft clustering.
                Each dict should have: feature_id, label, affinity.

        Returns:
            True on success.
        """
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
            doc["summary_embedding"] = embedding

        # Add soft clustering connected features
        if connected_features:
            doc["connected_features"] = connected_features

        if self._bulk_buffer is not None:
            self._bulk_buffer.add_index(feature_id, doc)
        else:
            client = self._get_client()
            client.index_document(INDEX_NAME, feature_id, doc)
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
            doc["summary_embedding"] = embedding

        if self._bulk_buffer is not None:
            self._bulk_buffer.add_index(subfeature_id, doc)
        else:
            client = self._get_client()
            client.index_document(INDEX_NAME, subfeature_id, doc)
        return True

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
                    "hash_id",
                    "cluster_id",
                    "cluster_label",
                    "summary",
                    "member_count",
                    "member_ids",
                    "connected_features",
                ],
            },
        )

        features = []
        for hit in result.get("hits", {}).get("hits", []):
            source = hit["_source"]
            feature: dict[str, Any] = {
                "feature_id": source.get("element_id"),
                "hash_id": source.get("hash_id"),
                "label": source.get("cluster_label"),
                "summary": source.get("summary", ""),
                "member_count": source.get("member_count", 0),
                "member_ids": source.get("member_ids", []),
            }
            # Include connected features if present (soft clustering)
            if "connected_features" in source:
                feature["connected_features"] = source["connected_features"]
            features.append(feature)

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
                    "hash_id",
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
                "hash_id": source.get("hash_id"),
                "label": source.get("cluster_label"),
                "summary": source.get("summary", ""),
                "member_count": source.get("member_count", 0),
                "member_ids": source.get("member_ids", []),
                "parent_feature_label": source.get("parent_feature_label", ""),
                "parent_feature_summary": source.get("parent_feature_summary", ""),
            })

        return subfeatures

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
                            {"term": {"element_type": "feature"}},
                        ]
                    }
                }
            },
            refresh=True,
            timeout=f"{bulk_timeout}s",
            request_timeout=bulk_timeout,
        )

        return response.get("deleted", 0)  # type: ignore[no-any-return]

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
                            {"term": {"element_type": "subfeature"}},
                        ]
                    }
                }
            },
            refresh=True,
            timeout=f"{bulk_timeout}s",
            request_timeout=bulk_timeout,
        )

        return response.get("deleted", 0)  # type: ignore[no-any-return]

    def update_cluster_assignments(
        self,
        assignments: list[dict[str, Any]],
    ) -> int:
        """Bulk update cluster assignments for elements.

        Args:
            assignments: List of dicts with:
                - element_id: Element ID
                - cluster_id: Primary cluster ID (for backward compat)
                - cluster_label: Primary cluster label (optional)
                - feature_memberships: Optional list of soft memberships
                    Each membership: {feature_id, label, score, is_primary}

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

            doc: dict[str, Any] = {
                "cluster_id": assignment["cluster_id"],
                "cluster_label": assignment.get("cluster_label"),
            }

            # Add soft memberships if present
            if "feature_memberships" in assignment:
                doc["feature_memberships"] = assignment["feature_memberships"]

            bulk_body.append({"doc": doc})

        response = client.bulk(operations=bulk_body, refresh=True)

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
        bulk_timeout = self._get_bulk_timeout()

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
            timeout=f"{bulk_timeout}s",
            request_timeout=bulk_timeout,
        )

        return response.get("updated", 0)  # type: ignore[no-any-return]

    def get_features_for_element(
        self,
        element_id: str,
        scope: str | None = None,
        repository: str | None = None,
        username: str = "main",
    ) -> list[dict[str, Any]]:
        """Get all features and subfeatures that contain a specific element.

        Args:
            element_id: The element_id to search for in member_ids.
            scope: Optional scope filter.
            repository: Optional repository filter.
            username: User branch (default: "main").

        Returns:
            List of features/subfeatures containing this element.
        """
        client = self._get_client()

        filters: list[dict[str, Any]] = [
            {"terms": {"element_type": ["feature", "subfeature"]}},
            {"term": {"member_ids": element_id}},
            {"term": {"username": username}},
        ]

        if scope:
            filters.append({"term": {"scope": scope}})
        if repository:
            filters.append({"term": {"repository": repository}})

        result = client.search(
            index=INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "filter": filters
                    }
                },
                "size": 50,
                "sort": [{"member_count": "desc"}],
                "_source": [
                    "element_id",
                    "hash_id",
                    "element_type",
                    "cluster_label",
                    "summary",
                    "member_count",
                    "parent_feature_label",
                    "parent_feature_summary",
                    "scope",
                    "repository",
                ],
            },
        )

        features = []
        for hit in result.get("hits", {}).get("hits", []):
            source = hit["_source"]
            feature: dict[str, Any] = {
                "feature_id": source.get("element_id"),
                "hash_id": source.get("hash_id"),
                "element_type": source.get("element_type"),
                "label": source.get("cluster_label"),
                "summary": source.get("summary", ""),
                "member_count": source.get("member_count", 0),
                "scope": source.get("scope"),
                "repository": source.get("repository"),
            }
            # Include parent info for subfeatures
            if source.get("element_type") == "subfeature":
                feature["parent_feature_label"] = source.get("parent_feature_label")
                feature["parent_feature_summary"] = source.get("parent_feature_summary")
            features.append(feature)

        return features

    def update_element_feature_memberships(
        self,
        scope: str,
        repository: str,
        username: str,
        cluster_id_to_feature: dict[str, tuple[str, str]],
    ) -> int:
        """Update element feature_memberships with proper feature_ids and labels.

        After features are summarized/labeled, this updates the placeholder
        feature_memberships on elements with actual feature IDs and labels.

        Args:
            scope: Repository scope.
            repository: Repository name.
            username: Username/branch.
            cluster_id_to_feature: Mapping of cluster_id -> (feature_id, label).

        Returns:
            Number of elements updated.
        """
        if not cluster_id_to_feature:
            return 0

        client = self._get_client()

        # Query elements with feature_memberships in this repo
        result = client.search(
            index=INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"scope": scope}},
                            {"term": {"repository": repository}},
                            {"term": {"username": username}},
                            {"exists": {"field": "feature_memberships"}},
                        ]
                    }
                },
                "size": 10000,
                "_source": ["element_id", "feature_memberships"],
            },
            scroll="2m",
        )

        bulk_body: list[dict[str, Any]] = []
        scroll_id = result.get("_scroll_id")

        try:
            while True:
                hits = result.get("hits", {}).get("hits", [])
                if not hits:
                    break

                for hit in hits:
                    source = hit["_source"]
                    element_id = source["element_id"]
                    memberships = source.get("feature_memberships", [])

                    # Update memberships with proper feature_id and label
                    updated_memberships = []
                    for m in memberships:
                        # Extract cluster_id from placeholder like "feature_48"
                        placeholder_id = m.get("feature_id", "")
                        if placeholder_id.startswith("feature_"):
                            cluster_id = placeholder_id.replace("feature_", "")
                        else:
                            cluster_id = placeholder_id

                        if cluster_id in cluster_id_to_feature:
                            feature_id, label = cluster_id_to_feature[cluster_id]
                            updated_memberships.append({
                                "feature_id": feature_id,
                                "label": label,
                                "score": m.get("score", 0),
                                "is_primary": m.get("is_primary", False),
                            })
                        else:
                            # Keep original if no mapping found
                            updated_memberships.append(m)

                    # Add to bulk update
                    bulk_body.append({
                        "update": {
                            "_index": INDEX_NAME,
                            "_id": element_id,
                        }
                    })
                    bulk_body.append({
                        "doc": {"feature_memberships": updated_memberships}
                    })

                # Get next batch
                if scroll_id:
                    result = client.scroll(scroll_id=scroll_id, scroll="2m")
                else:
                    break

        finally:
            if scroll_id:
                with contextlib.suppress(Exception):
                    client.clear_scroll(scroll_id=scroll_id)

        if not bulk_body:
            return 0

        # Execute bulk update
        response = client.bulk(operations=bulk_body, refresh=True)

        # Count successful updates
        updated = 0
        for item in response.get("items", []):
            if item.get("update", {}).get("result") in ["updated", "noop"]:
                updated += 1

        return updated
