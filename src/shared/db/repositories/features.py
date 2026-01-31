"""Feature repository for managing features, subfeatures, and clustering.

Handles indexing and retrieval of feature documents and cluster management.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .base import INDEX_NAME, ElasticsearchBase, generate_hash_id


class FeatureRepository:
    """Repository for feature and subfeature operations."""

    def __init__(self, base: ElasticsearchBase):
        self._base = base

    def _get_client(self) -> Any:
        """Get Elasticsearch client from base."""
        return self._base._get_client()

    def _get_bulk_timeout(self) -> int:
        """Get bulk operation timeout from config."""
        return self._base._config.elasticsearch.bulk_timeout

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

        client = self._get_client()
        client.index(index=INDEX_NAME, id=subfeature_id, document=doc)
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

        return response.get("deleted", 0)

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

        return response.get("updated", 0)
