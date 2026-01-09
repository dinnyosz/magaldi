"""Vector map and clustering API routes."""

from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, Path, Query

from magaldi_web.dependencies import get_es_repository
from magaldi_web.models import (
    Cluster,
    ClusterMember,
    ClusterRepresentative,
    ClustersResponse,
    VectorMapResponse,
    VectorPoint,
)
from shared.db.elasticsearch import ElasticsearchRepository, INDEX_NAME

router = APIRouter()


def reduce_dimensions(
    embeddings: list[list[float]],
    n_components: int = 2,
    algorithm: str = "umap",
) -> np.ndarray:
    """Reduce high-dimensional embeddings to 2D/3D for visualization.

    Args:
        embeddings: List of embedding vectors
        n_components: Target dimensions (2 or 3)
        algorithm: Reduction algorithm ('umap' or 'tsne')

    Returns:
        Numpy array of reduced coordinates
    """
    X = np.array(embeddings)

    if len(X) < 5:
        # Not enough points for dimensionality reduction
        # Return simple 2D/3D projection
        if n_components == 2:
            return X[:, :2] if X.shape[1] >= 2 else np.zeros((len(X), 2))
        else:
            return X[:, :3] if X.shape[1] >= 3 else np.zeros((len(X), 3))

    if algorithm == "umap":
        try:
            import umap

            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=min(15, len(X) - 1),
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )
            return reducer.fit_transform(X)
        except ImportError:
            # Fall back to t-SNE if UMAP not installed
            algorithm = "tsne"

    if algorithm == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30, len(X) - 1)
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            metric="cosine",
            random_state=42,
            n_iter=1000,
        )
        return reducer.fit_transform(X)

    raise ValueError(f"Unknown algorithm: {algorithm}")


@router.get("/repos/{scope}/{repository}/vector-map", response_model=VectorMapResponse)
async def get_vector_map(
    scope: str = Path(..., description="Repository scope"),
    repository: str = Path(..., description="Repository name"),
    username: str = Query(default="main", description="Username/branch"),
    element_types: list[str] = Query(
        default=["class", "function", "method"],
        description="Element types to include",
    ),
    dimensions: int = Query(default=2, ge=2, le=3, description="Output dimensions"),
    algorithm: Literal["umap", "tsne"] = Query(default="umap", description="Reduction algorithm"),
    limit: int = Query(default=5000, ge=1, le=10000, description="Max elements"),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> VectorMapResponse:
    """Get 2D/3D coordinates for visualizing the embedding space."""
    client = es_repo._get_client()

    # Fetch elements with embeddings
    result = client.search(
        index=INDEX_NAME,
        body={
            "size": limit,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                        {"terms": {"element_type": element_types}},
                        {"exists": {"field": "embedding"}},
                    ],
                },
            },
            "_source": [
                "element_id",
                "name",
                "element_type",
                "relative_path",
                "line_start",
                "summary",
                "embedding",
            ],
        },
    )

    hits = result.get("hits", {}).get("hits", [])
    if not hits:
        return VectorMapResponse(
            points=[],
            bounds={"x": [0, 0], "y": [0, 0]},
            algorithm=algorithm,
            dimensions=dimensions,
            element_count=0,
        )

    # Extract embeddings and metadata
    embeddings = []
    metadata = []

    for hit in hits:
        source = hit["_source"]
        embedding = source.get("embedding")
        if embedding:
            embeddings.append(embedding)
            metadata.append(
                {
                    "element_id": source["element_id"],
                    "name": source["name"],
                    "element_type": source["element_type"],
                    "file_path": source["relative_path"],
                    "line": source["line_start"],
                    "summary": (source.get("summary") or "")[:100],
                }
            )

    if not embeddings:
        return VectorMapResponse(
            points=[],
            bounds={"x": [0, 0], "y": [0, 0]},
            algorithm=algorithm,
            dimensions=dimensions,
            element_count=0,
        )

    # Reduce dimensions
    coords = reduce_dimensions(embeddings, n_components=dimensions, algorithm=algorithm)

    # Build points
    points = []
    for i, meta in enumerate(metadata):
        point = VectorPoint(
            x=float(coords[i][0]),
            y=float(coords[i][1]),
            z=float(coords[i][2]) if dimensions == 3 else None,
            element_id=meta["element_id"],
            name=meta["name"],
            element_type=meta["element_type"],
            file_path=meta["file_path"],
            line=meta["line"],
            summary=meta["summary"],
        )
        points.append(point)

    # Calculate bounds
    bounds = {
        "x": [float(coords[:, 0].min()), float(coords[:, 0].max())],
        "y": [float(coords[:, 1].min()), float(coords[:, 1].max())],
    }
    if dimensions == 3:
        bounds["z"] = [float(coords[:, 2].min()), float(coords[:, 2].max())]

    return VectorMapResponse(
        points=points,
        bounds=bounds,
        algorithm=algorithm,
        dimensions=dimensions,
        element_count=len(points),
    )


@router.get("/repos/{scope}/{repository}/clusters", response_model=ClustersResponse)
async def get_clusters(
    scope: str = Path(..., description="Repository scope"),
    repository: str = Path(..., description="Repository name"),
    username: str = Query(default="main", description="Username/branch"),
    limit: int = Query(default=50, ge=1, le=100, description="Max features to return"),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> ClustersResponse:
    """Get pre-extracted HDBSCAN feature clusters from the codebase."""
    client = es_repo._get_client()

    # Get features (element_type = 'feature') which are already extracted via HDBSCAN
    features_result = client.search(
        index=INDEX_NAME,
        body={
            "size": limit,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                        {"term": {"element_type": "feature"}},
                        {"exists": {"field": "embedding"}},
                    ],
                },
            },
            "_source": [
                "element_id",
                "name",
                "element_type",
                "relative_path",
                "summary",
                "embedding",
            ],
        },
    )

    hits = features_result.get("hits", {}).get("hits", [])

    if not hits:
        return ClustersResponse(clusters=[], total_elements=0)

    # Build clusters from pre-extracted HDBSCAN features
    # Each feature document represents a cluster with its member elements
    clusters = []

    for idx, hit in enumerate(hits):
        source = hit["_source"]

        # Feature documents have: name (feature name), summary (description),
        # and we need to get the member elements for this feature
        feature_id = source["element_id"]
        feature_name = source["name"]
        feature_summary = source.get("summary")

        # Get member elements for this feature (stored as children or via feature_id reference)
        # For now, create a cluster with the feature as its own representative
        # TODO: Query for elements that belong to this feature cluster
        clusters.append(
            Cluster(
                cluster_id=idx,
                size=1,  # Will be updated when we have member counts
                representative=ClusterRepresentative(
                    name=feature_name,
                    element_type="feature",
                    file_path=source.get("relative_path", ""),
                    summary=feature_summary,
                ),
                members=[
                    ClusterMember(
                        element_id=feature_id,
                        name=feature_name,
                        element_type="feature",
                    )
                ],
            )
        )

    return ClustersResponse(
        clusters=clusters,
        total_elements=len(clusters),
    )
