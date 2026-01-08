"""Feature clustering using HDBSCAN on code element embeddings.

Groups semantically similar functions/methods into feature clusters
with optional Ollama-based labeling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import hdbscan
import numpy as np

from magaldi.summarization.summarization import OllamaClient


class ClusteringError(Exception):
    """Raised when clustering fails."""

    pass


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ClusterConfig:
    """Configuration for HDBSCAN clustering."""

    # HDBSCAN parameters
    min_cluster_size: int = 5
    min_samples: int = 3

    # Element types to cluster
    element_types: list[str] = field(default_factory=lambda: ["function", "method"])

    # Ollama settings for labeling
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:3b"
    label_temperature: float = 0.3
    label_max_tokens: int = 32
    label_timeout: int = 30


@dataclass
class ClusterResult:
    """Result for a single cluster."""

    cluster_id: int
    label: str | None
    element_ids: list[str]
    element_names: list[str]
    centroid: list[float] | None = None

    @property
    def size(self) -> int:
        """Number of elements in this cluster."""
        return len(self.element_ids)


@dataclass
class ClusteringResult:
    """Overall clustering result."""

    clusters: list[ClusterResult]
    outlier_count: int
    outlier_element_ids: list[str]
    total_elements: int

    @property
    def cluster_count(self) -> int:
        """Number of clusters found (excluding outliers)."""
        return len(self.clusters)


# =============================================================================
# CLUSTER LABELING PROMPT
# =============================================================================

LABEL_PROMPT = """Given these function/method names from a code cluster, generate a short label (1-5 words) that describes the common feature or functionality they share.

Function names:
{names}

Generate ONLY a short label like: "user authentication", "database query handling", "REST API endpoints", "file processing utilities", "input validation", etc.

Label:"""


# =============================================================================
# FEATURE CLUSTERER
# =============================================================================


class FeatureClusterer:
    """Clusters code elements by semantic similarity using HDBSCAN."""

    def __init__(
        self,
        config: ClusterConfig | None = None,
        ollama_client: OllamaClient | None = None,
    ):
        """Initialize clusterer.

        Args:
            config: Clustering configuration.
            ollama_client: Optional Ollama client for labeling.
        """
        self.config = config or ClusterConfig()
        self._ollama = ollama_client

    def _get_ollama(self) -> OllamaClient:
        """Get or create Ollama client."""
        if self._ollama is None:
            self._ollama = OllamaClient(
                url=self.config.ollama_url,
                model=self.config.ollama_model,
            )
        return self._ollama

    def cluster(
        self,
        elements: list[dict[str, Any]],
    ) -> ClusteringResult:
        """Run HDBSCAN clustering on element embeddings.

        Args:
            elements: List of dicts with element_id, embedding, name, element_type.

        Returns:
            ClusteringResult with clusters and outliers.

        Raises:
            ClusteringError: If clustering fails or no embeddings provided.
        """
        if not elements:
            raise ClusteringError("No elements provided for clustering")

        # Extract embeddings as numpy array
        embeddings = []
        valid_elements = []
        for elem in elements:
            emb = elem.get("embedding")
            if emb is not None:
                embeddings.append(emb)
                valid_elements.append(elem)

        if not embeddings:
            raise ClusteringError("No embeddings found in elements")

        embedding_array = np.array(embeddings, dtype=np.float32)

        # Run HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            min_samples=self.config.min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )

        labels = clusterer.fit_predict(embedding_array)

        # Group elements by cluster label
        cluster_map: dict[int, list[dict[str, Any]]] = {}
        outliers: list[str] = []

        for idx, label in enumerate(labels):
            elem = valid_elements[idx]
            if label == -1:
                # Outlier
                outliers.append(elem["element_id"])
            else:
                if label not in cluster_map:
                    cluster_map[label] = []
                cluster_map[label].append(elem)

        # Build ClusterResult objects
        clusters: list[ClusterResult] = []
        for cluster_label, cluster_elements in sorted(cluster_map.items()):
            # Compute centroid
            cluster_embeddings = np.array(
                [e["embedding"] for e in cluster_elements], dtype=np.float32
            )
            centroid = cluster_embeddings.mean(axis=0).tolist()

            cluster = ClusterResult(
                cluster_id=cluster_label,
                label=None,  # Will be set by label_clusters
                element_ids=[e["element_id"] for e in cluster_elements],
                element_names=[e.get("name", "") for e in cluster_elements],
                centroid=centroid,
            )
            clusters.append(cluster)

        return ClusteringResult(
            clusters=clusters,
            outlier_count=len(outliers),
            outlier_element_ids=outliers,
            total_elements=len(valid_elements),
        )

    def label_clusters(
        self,
        result: ClusteringResult,
        max_names_per_prompt: int = 15,
    ) -> ClusteringResult:
        """Generate labels for clusters using Ollama.

        Args:
            result: Clustering result to label.
            max_names_per_prompt: Max function names to include in prompt.

        Returns:
            Updated ClusteringResult with labels.
        """
        ollama = self._get_ollama()

        for cluster in result.clusters:
            # Get sample of function names
            names = cluster.element_names[:max_names_per_prompt]
            names_str = "\n".join(f"- {name}" for name in names if name)

            if not names_str:
                cluster.label = f"cluster_{cluster.cluster_id}"
                continue

            prompt = LABEL_PROMPT.format(names=names_str)

            try:
                raw_label = ollama.generate(
                    prompt=prompt,
                    temperature=self.config.label_temperature,
                    max_tokens=self.config.label_max_tokens,
                    timeout=self.config.label_timeout,
                )

                # Clean label
                label = self._clean_label(raw_label)
                cluster.label = label or f"cluster_{cluster.cluster_id}"

            except Exception:
                # Fall back to numbered label
                cluster.label = f"cluster_{cluster.cluster_id}"

        return result

    def _clean_label(self, raw_label: str) -> str:
        """Clean and normalize generated label.

        Args:
            raw_label: Raw label from Ollama.

        Returns:
            Cleaned label string.
        """
        label = raw_label.strip().lower()

        # Remove quotes
        label = label.strip("\"'")

        # Remove common prefixes
        prefixes = ["label:", "the label is", "this cluster is about"]
        for prefix in prefixes:
            if label.startswith(prefix):
                label = label[len(prefix) :].strip()

        # Replace spaces with underscores for consistency
        label = label.replace(" ", "_")

        # Remove special characters except underscores
        label = "".join(c for c in label if c.isalnum() or c == "_")

        # Limit length
        if len(label) > 50:
            label = label[:50]

        return label


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def run_clustering(
    elements: list[dict[str, Any]],
    config: ClusterConfig | None = None,
    label_clusters: bool = True,
) -> ClusteringResult:
    """Run clustering pipeline on elements.

    Args:
        elements: List of elements with embeddings.
        config: Optional clustering config.
        label_clusters: Whether to generate labels with Ollama.

    Returns:
        ClusteringResult with clusters.
    """
    clusterer = FeatureClusterer(config)

    # Run HDBSCAN
    result = clusterer.cluster(elements)

    # Optionally label clusters
    if label_clusters and result.clusters:
        result = clusterer.label_clusters(result)

    return result
