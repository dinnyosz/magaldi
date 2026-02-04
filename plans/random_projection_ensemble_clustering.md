# Random Projection Ensemble Clustering for High-Dimensional Embeddings

A method for finding **overlapping/soft cluster memberships** in high-dimensional vector spaces (e.g., 1024-d embeddings) using random projections, ensemble clustering, and cooccurrence analysis.

---

## The Problem

- High-dimensional embeddings (1024-d) suffer from the **curse of dimensionality** — distances become less meaningful, density-based methods struggle
- Traditional clustering assigns each point to exactly one cluster
- In embedding spaces, points often legitimately belong to **multiple concepts/clusters**

### Magaldi-Specific Problems

Current Magaldi feature extraction uses single HDBSCAN clustering with hard assignments:
- A `validate_and_save_user()` function belongs to ONLY "validation" OR "user_management" — not both
- Utility functions that span multiple domains get arbitrarily assigned to one feature
- No way to discover which features are related to each other
- Subfeatures are isolated within their parent feature — no cross-feature connections

---

## Magaldi Integration Overview

### Current Architecture

```
src/shared/ai/clustering/clusterer.py:193 - FeatureClusterer.cluster()

embeddings (1024d) → Single HDBSCAN → Hard labels (one feature per element)
```

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  LEVEL 1: Soft Feature Clustering                                           │
│                                                                             │
│  30k elements → Random Projection Ensemble → Cooccurrence → NMF            │
│                                                                             │
│  Result: Each element has membership scores across ~500 features           │
│  validate_user() → { auth: 0.4, validation: 0.35, user_mgmt: 0.25 }       │
│                                                                             │
│  BONUS: Feature Affinity Matrix (500×500)                                  │
│  auth ↔ user_mgmt: 0.7 (strongly connected)                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LEVEL 2: Soft Subfeature Clustering (per feature)                         │
│                                                                             │
│  For each feature with >20 members:                                        │
│  - Take members weighted by their feature membership score                 │
│  - Run soft clustering again → subfeatures                                 │
│                                                                             │
│  Feature "auth" (150 weighted members):                                    │
│  ├── Subfeature "token_validation" (elements with soft scores)            │
│  ├── Subfeature "password_hashing" (elements with soft scores)            │
│  └── Subfeature "session_management" (elements with soft scores)          │
│                                                                             │
│  BONUS: Subfeature Affinity Matrix (per feature)                           │
│  token_validation ↔ session_management: 0.6                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LEVEL 3: Cross-Feature Subfeature Connections                             │
│                                                                             │
│  auth::token_validation ↔ user_mgmt::jwt_handling: 0.8                     │
│  validation::input_sanitization ↔ security::xss_prevention: 0.9           │
│                                                                             │
│  These emerge from elements that have high membership in BOTH              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Scale Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Elements | 1k - 30k | Functions/methods per repository |
| Features | 500+ | High-level capability groupings |
| Subfeatures | ~2,500 | ~5 subfeatures per large feature |
| Membership threshold | 0.01 | Below this, don't store (just numerical noise) |
| Affinity threshold | 0.001 | For connected features (show all, tune later) |

---

## What is Soft Membership?

**Current (Hard):**
```
function validate_user() → Feature: "validation"      (100%)
function save_user()     → Feature: "persistence"     (100%)
```

**Proposed (Soft):**
```
function validate_and_save_user() → Feature: "validation"  (45%)
                                  → Feature: "persistence" (35%)
                                  → Feature: "user_mgmt"   (20%)
```

Each function gets a **probability distribution** over all features. The numbers sum to ~100%.

**Primary cluster** = the one with highest score (validation @ 45% in this example). Kept for backward compatibility.

**Why low threshold (0.01) is safe:** The algorithm naturally self-filters unrelated memberships. If element X has nothing to do with feature Y, it won't cooccur with Y's members during ensemble clustering, so NMF gives it near-zero weight. The threshold just filters numerical noise from floating point math — not semantic noise.

---

## Connected Features (Feature Affinity Matrix)

From soft memberships, we compute **feature-to-feature affinity**:

```python
# W = soft membership matrix (n_elements × n_features)
# Each row: one function's membership distribution
# Each column: one feature

# Feature affinity matrix:
feature_affinity = W.T @ W  # (n_features × n_features)

# feature_affinity[i,j] = how much features i and j share elements
```

**Example:**
```
              validation  persistence  user_mgmt  auth
validation       1.0         0.4         0.6      0.2
persistence      0.4         1.0         0.5      0.1
user_mgmt        0.6         0.5         1.0      0.7
auth             0.2         0.1         0.7      1.0
```

This tells us:
- "validation" and "user_mgmt" are strongly connected (0.6)
- "auth" and "user_mgmt" are strongly connected (0.7)
- "auth" and "persistence" are weakly connected (0.1)

---

## The Algorithm

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. RANDOM PROJECTIONS (Johnson-Lindenstrauss)                      │
│     1024-d → 50-d, repeated N times with different random matrices  │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. ENSEMBLE CLUSTERING                                             │
│     Run HDBSCAN on each projection, track cluster co-membership     │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. BUILD COOCCURRENCE MATRIX (SPARSE for scale)                    │
│     cooccurrence[i,j] = P(points i,j in same cluster)               │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. EXTRACT SOFT MEMBERSHIPS via NMF                                │
│     cooccurrence ≈ W @ H, where W = membership matrix               │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. COMPUTE FEATURE AFFINITY                                        │
│     feature_affinity = W.T @ W                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Random Projection Theory

**Johnson-Lindenstrauss Lemma:** For n points, projecting to O(log n / ε²) dimensions preserves pairwise distances within factor (1 ± ε).

This means 1024 → 50-100 dimensions via random projection is mathematically justified — structure is preserved.

```python
from sklearn.random_projection import GaussianRandomProjection

rp = GaussianRandomProjection(n_components=50)
reduced = rp.fit_transform(embeddings)  # (n_points, 50)
```

---

## Step 2 & 3: Ensemble Clustering + Cooccurrence Matrix

Run clustering on many random projections, aggregate how often pairs land together.

**IMPORTANT: Use sparse matrix for 30k elements** — dense would be 7GB.

```python
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.random_projection import GaussianRandomProjection
from hdbscan import HDBSCAN

def build_cooccurrence_matrix_sparse(
    embeddings,
    n_runs=50,
    n_components=50,
    min_cluster_size=15,
    cooccurrence_threshold=0.05,  # Only store if > 5% cooccurrence
):
    """
    Build SPARSE cooccurrence matrix from ensemble of random projection clusterings.

    Returns:
        cooccurrence: sparse (n_points, n_points) matrix where entry [i,j] =
                      probability that points i and j land in the same cluster
    """
    n_points = embeddings.shape[0]

    # Use lil_matrix for efficient incremental construction
    cooccurrence = lil_matrix((n_points, n_points), dtype=np.float32)

    for run in range(n_runs):
        # Random projection
        rp = GaussianRandomProjection(n_components=n_components)
        reduced = rp.fit_transform(embeddings)

        # Cluster in reduced space
        labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(reduced)

        # Count co-occurrences (excluding noise points labeled -1)
        for cluster_id in set(labels) - {-1}:
            indices = np.where(labels == cluster_id)[0]
            for i in indices:
                for j in indices:
                    if i <= j:  # Only upper triangle + diagonal
                        cooccurrence[i, j] += 1

    # Normalize to probabilities
    cooccurrence = cooccurrence / n_runs

    # Convert to CSR for efficient arithmetic, threshold small values
    cooccurrence = csr_matrix(cooccurrence)
    cooccurrence.data[cooccurrence.data < cooccurrence_threshold] = 0
    cooccurrence.eliminate_zeros()

    # Make symmetric
    cooccurrence = cooccurrence + cooccurrence.T - csr_matrix(np.diag(cooccurrence.diagonal()))

    return cooccurrence

# Usage
cooccurrence = build_cooccurrence_matrix_sparse(embeddings, n_runs=50)
```

**Memory comparison for 30k elements:**
- Dense: 30k × 30k × 4 bytes = **3.6 GB**
- Sparse (5% density): ~**180 MB**

---

## Step 4: Extract Soft Memberships via NMF

NMF decomposes a matrix V into two smaller non-negative matrices:

```
V (n × m)  ≈  W (n × k)  ×  H (k × m)

where all entries in W and H are ≥ 0
```

**Why non-negative matters:**

The constraint forces **additive, parts-based** representations. Each point is a sum of positive contributions — no cancellation.

| Method | Interpretation |
|--------|---------------|
| PCA/SVD | "Point = +0.7×component_A − 0.3×component_B" (hard to interpret) |
| NMF | "Point = 0.7×topic_A + 0.4×topic_B" (natural soft membership) |

```python
from sklearn.decomposition import NMF

def nmf_soft_clustering(cooccurrence, n_clusters=500, threshold=0.01):
    """
    NMF decomposition for soft cluster memberships.

    Args:
        cooccurrence: Sparse cooccurrence matrix
        n_clusters: Number of features to extract
        threshold: Minimum membership score to keep (0.01 = 1%, filters numerical noise)

    Returns:
        memberships: dict[element_idx, list[tuple[cluster_idx, score]]]
        feature_affinity: (n_clusters, n_clusters) matrix
    """
    # Convert sparse to dense for NMF (or use sparse NMF variant)
    cooccurrence_dense = cooccurrence.toarray()

    nmf = NMF(
        n_components=n_clusters,
        init='nndsvd',
        max_iter=500,
        random_state=42
    )

    W = nmf.fit_transform(cooccurrence_dense)  # (n_elements, n_clusters)

    # Normalize rows to sum to 1 (proper probabilities)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_normalized = W / row_sums

    # Extract memberships above threshold
    memberships = {}
    for elem_idx in range(W_normalized.shape[0]):
        scores = W_normalized[elem_idx]
        above_threshold = [(cluster_idx, float(score))
                          for cluster_idx, score in enumerate(scores)
                          if score >= threshold]
        if above_threshold:
            # Sort by score descending
            above_threshold.sort(key=lambda x: -x[1])
            memberships[elem_idx] = above_threshold

    # Compute feature affinity matrix
    feature_affinity = W_normalized.T @ W_normalized  # (n_clusters, n_clusters)

    return memberships, feature_affinity

# Usage
memberships, feature_affinity = nmf_soft_clustering(cooccurrence, n_clusters=500, threshold=0.01)

# Element 42's memberships:
print(memberships[42])  # [(3, 0.45), (7, 0.32), (12, 0.18)]

# Features connected to feature 3:
connected = [(i, feature_affinity[3, i]) for i in range(500) if feature_affinity[3, i] > 0.3]
```

---

## Storage Schema (Elasticsearch)

All data lives in the **single `magaldi-elements` index** with different `element_type` values.

### On Code Elements (function, method, class)

```python
# Add to INDEX_MAPPING in src/shared/db/repositories/base.py

# Soft feature memberships
"feature_memberships": {
    "type": "nested",
    "properties": {
        "feature_id": {"type": "keyword"},
        "label": {"type": "keyword"},
        "score": {"type": "float"},
        "is_primary": {"type": "boolean"},
    },
},

# Soft subfeature memberships
"subfeature_memberships": {
    "type": "nested",
    "properties": {
        "subfeature_id": {"type": "keyword"},
        "label": {"type": "keyword"},
        "parent_feature_id": {"type": "keyword"},
        "parent_feature_label": {"type": "keyword"},
        "score": {"type": "float"},
        "is_primary": {"type": "boolean"},
    },
},
```

### Example Code Element Document

```json
{
  "element_id": "myorg:myrepo:main:src/auth/validate.py:function:validate_and_save_user:42",
  "name": "validate_and_save_user",
  "element_type": "function",

  "cluster_id": "feature_auth",
  "cluster_label": "authentication",

  "feature_memberships": [
    {"feature_id": "feature_auth", "label": "authentication", "score": 0.45, "is_primary": true},
    {"feature_id": "feature_validation", "label": "input_validation", "score": 0.32, "is_primary": false},
    {"feature_id": "feature_persistence", "label": "data_persistence", "score": 0.18, "is_primary": false}
  ],

  "subfeature_memberships": [
    {"subfeature_id": "sub_token_val", "label": "token_validation", "parent_feature_id": "feature_auth", "parent_feature_label": "authentication", "score": 0.40, "is_primary": true},
    {"subfeature_id": "sub_input_san", "label": "input_sanitization", "parent_feature_id": "feature_validation", "parent_feature_label": "input_validation", "score": 0.28, "is_primary": true}
  ]
}
```

### On Feature Elements (element_type: "feature")

```python
# Connected features (from feature affinity matrix)
"connected_features": {
    "type": "nested",
    "properties": {
        "feature_id": {"type": "keyword"},
        "label": {"type": "keyword"},
        "affinity": {"type": "float"},
    },
},

# Cross-feature subfeature connections
"connected_subfeatures_cross": {
    "type": "nested",
    "properties": {
        "subfeature_id": {"type": "keyword"},
        "label": {"type": "keyword"},
        "parent_feature_id": {"type": "keyword"},
        "parent_feature_label": {"type": "keyword"},
        "affinity": {"type": "float"},
    },
},
```

### Example Feature Document

```json
{
  "element_id": "myorg:myrepo:main:feature:authentication",
  "element_type": "feature",
  "name": "authentication",
  "cluster_label": "authentication",
  "summary": "Handles user authentication, token validation, and session management...",
  "member_count": 150,

  "connected_features": [
    {"feature_id": "feature_user_mgmt", "label": "user_management", "affinity": 0.72},
    {"feature_id": "feature_session", "label": "session_handling", "affinity": 0.58},
    {"feature_id": "feature_security", "label": "security", "affinity": 0.45}
  ],

  "connected_subfeatures_cross": [
    {"subfeature_id": "sub_jwt_handling", "label": "jwt_handling", "parent_feature_id": "feature_user_mgmt", "parent_feature_label": "user_management", "affinity": 0.81}
  ]
}
```

### On Subfeature Elements (element_type: "subfeature")

```python
# Connected subfeatures (within same feature + cross-feature)
"connected_subfeatures": {
    "type": "nested",
    "properties": {
        "subfeature_id": {"type": "keyword"},
        "label": {"type": "keyword"},
        "parent_feature_id": {"type": "keyword"},
        "parent_feature_label": {"type": "keyword"},
        "affinity": {"type": "float"},
    },
},
```

---

## Query Patterns

| Query | Elasticsearch |
|-------|---------------|
| Features of element X | Just read `element.feature_memberships` |
| Elements in feature Y | `nested: feature_memberships.feature_id = Y` |
| Elements with score > 0.3 | `nested: feature_id = Y AND score > 0.3` |
| Elements in BOTH feature A and B | Bool query with two nested clauses |
| Primary members of feature Y | `nested: feature_id = Y AND is_primary = true` |
| Features connected to feature X | Read `feature.connected_features` |

---

## MCP Tool Enhancements

| Tool | Enhancement |
|------|-------------|
| `search_features` | Return `connected_features` in response |
| `get_feature_members` | Include membership scores, show shared features |
| **NEW: `find_associated`** | Given element, return features + their connections |
| `list_features` | Show feature graph / connection counts |

---

## Web UI Enhancements

1. **Feature Graph View** — Visualize connected features as a network
2. **Multiple Feature Badges** — Elements show all their features with scores
3. **"Related Features" Sidebar** — When viewing a feature, show connections
4. **Cross-Feature Subfeature Links** — Navigate between related subfeatures

---

## Storage Estimates

| Data | Count | Size |
|------|-------|------|
| Feature memberships | 30k elements × ~15 features (at 0.01 threshold) | ~450k nested docs |
| Subfeature memberships | 30k elements × ~8 subfeatures | ~240k nested docs |
| Feature affinity | 500 × 500 | ~250k entries (~1 MB) |
| Cooccurrence (sparse) | 30k × 30k @ 5% | ~180 MB |

All manageable within existing Elasticsearch infrastructure. The low threshold (0.01) captures weak-but-real connections; queries can filter higher when needed.

---

## Complete Pipeline Class

```python
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import NMF
from hdbscan import HDBSCAN
from dataclasses import dataclass

@dataclass
class SoftClusteringConfig:
    """Configuration for soft clustering pipeline."""
    n_projection_runs: int = 50
    projection_dims: int = 50
    min_cluster_size: int = 15
    n_features: int = 500
    membership_threshold: float = 0.01  # Low threshold - algorithm self-filters unrelated
    cooccurrence_threshold: float = 0.05
    affinity_threshold: float = 0.001


@dataclass
class FeatureMembership:
    """Soft membership of an element in a feature."""
    feature_idx: int
    score: float
    is_primary: bool


@dataclass
class FeatureAffinity:
    """Connection between two features."""
    feature_idx: int
    affinity: float


class SoftClusteringPipeline:
    """
    Complete pipeline for soft/overlapping clustering of high-dimensional embeddings.
    Designed for Magaldi feature extraction at scale (1k-30k elements, 500+ features).
    """

    def __init__(self, config: SoftClusteringConfig | None = None):
        self.config = config or SoftClusteringConfig()
        self.cooccurrence_ = None
        self.memberships_ = None  # W matrix from NMF
        self.feature_affinity_ = None
        self.element_memberships_ = None  # dict[element_idx, list[FeatureMembership]]

    def fit(self, embeddings: np.ndarray, element_ids: list[str] | None = None):
        """
        Fit the pipeline: build cooccurrence matrix, extract soft memberships.

        Args:
            embeddings: (n_elements, 1024) embedding matrix
            element_ids: Optional list of element IDs for result mapping
        """
        n_elements = embeddings.shape[0]

        # Step 1-3: Build sparse cooccurrence matrix
        self.cooccurrence_ = self._build_cooccurrence_sparse(embeddings)

        # Step 4: NMF for soft memberships
        self.memberships_, self.feature_affinity_ = self._extract_memberships()

        # Step 5: Build element membership mapping
        self.element_memberships_ = self._build_element_memberships(element_ids)

        return self

    def _build_cooccurrence_sparse(self, embeddings: np.ndarray) -> csr_matrix:
        """Build sparse cooccurrence matrix from ensemble clustering."""
        n_points = embeddings.shape[0]
        cooccurrence = lil_matrix((n_points, n_points), dtype=np.float32)

        for run in range(self.config.n_projection_runs):
            rp = GaussianRandomProjection(n_components=self.config.projection_dims)
            reduced = rp.fit_transform(embeddings)

            labels = HDBSCAN(min_cluster_size=self.config.min_cluster_size).fit_predict(reduced)

            for cluster_id in set(labels) - {-1}:
                indices = np.where(labels == cluster_id)[0]
                for i in indices:
                    for j in indices:
                        if i <= j:
                            cooccurrence[i, j] += 1

        cooccurrence = cooccurrence / self.config.n_projection_runs
        cooccurrence = csr_matrix(cooccurrence)
        cooccurrence.data[cooccurrence.data < self.config.cooccurrence_threshold] = 0
        cooccurrence.eliminate_zeros()
        cooccurrence = cooccurrence + cooccurrence.T - csr_matrix(
            np.diag(cooccurrence.diagonal())
        )

        return cooccurrence

    def _extract_memberships(self) -> tuple[np.ndarray, np.ndarray]:
        """Extract soft memberships via NMF."""
        cooccurrence_dense = self.cooccurrence_.toarray()

        nmf = NMF(
            n_components=self.config.n_features,
            init='nndsvd',
            max_iter=500,
            random_state=42
        )

        W = nmf.fit_transform(cooccurrence_dense)

        # Normalize rows
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W_normalized = W / row_sums

        # Compute feature affinity
        feature_affinity = W_normalized.T @ W_normalized

        return W_normalized, feature_affinity

    def _build_element_memberships(
        self,
        element_ids: list[str] | None
    ) -> dict[str | int, list[FeatureMembership]]:
        """Build per-element membership lists."""
        result = {}

        for elem_idx in range(self.memberships_.shape[0]):
            scores = self.memberships_[elem_idx]

            # Find all features above threshold
            above_threshold = [
                (feat_idx, float(score))
                for feat_idx, score in enumerate(scores)
                if score >= self.config.membership_threshold
            ]

            if not above_threshold:
                continue

            # Sort by score descending
            above_threshold.sort(key=lambda x: -x[1])

            # Mark primary (highest score)
            memberships = [
                FeatureMembership(
                    feature_idx=feat_idx,
                    score=score,
                    is_primary=(i == 0)
                )
                for i, (feat_idx, score) in enumerate(above_threshold)
            ]

            key = element_ids[elem_idx] if element_ids else elem_idx
            result[key] = memberships

        return result

    def get_element_memberships(self, element_id: str | int) -> list[FeatureMembership]:
        """Get soft memberships for a specific element."""
        return self.element_memberships_.get(element_id, [])

    def get_feature_members(
        self,
        feature_idx: int,
        min_score: float | None = None
    ) -> list[tuple[str | int, float, bool]]:
        """Get all elements belonging to a feature with their scores."""
        min_score = min_score or self.config.membership_threshold

        members = []
        for elem_id, memberships in self.element_memberships_.items():
            for m in memberships:
                if m.feature_idx == feature_idx and m.score >= min_score:
                    members.append((elem_id, m.score, m.is_primary))
                    break

        return sorted(members, key=lambda x: -x[1])

    def get_connected_features(
        self,
        feature_idx: int,
        min_affinity: float | None = None
    ) -> list[FeatureAffinity]:
        """Get features connected to the given feature."""
        min_affinity = min_affinity or self.config.affinity_threshold

        affinities = self.feature_affinity_[feature_idx]
        connected = [
            FeatureAffinity(feature_idx=i, affinity=float(affinities[i]))
            for i in range(len(affinities))
            if i != feature_idx and affinities[i] >= min_affinity
        ]

        return sorted(connected, key=lambda x: -x.affinity)


# Usage example
if __name__ == "__main__":
    np.random.seed(42)
    embeddings = np.random.randn(1000, 1024)
    element_ids = [f"element_{i}" for i in range(1000)]

    config = SoftClusteringConfig(
        n_projection_runs=50,
        projection_dims=50,
        n_features=50,  # Smaller for demo
        membership_threshold=0.01,
    )

    pipeline = SoftClusteringPipeline(config)
    pipeline.fit(embeddings, element_ids)

    # Get memberships for an element
    memberships = pipeline.get_element_memberships("element_42")
    print(f"Element 42 belongs to {len(memberships)} features:")
    for m in memberships[:3]:
        print(f"  Feature {m.feature_idx}: {m.score:.2%} {'(primary)' if m.is_primary else ''}")

    # Get connected features
    if memberships:
        primary_feature = memberships[0].feature_idx
        connected = pipeline.get_connected_features(primary_feature)
        print(f"\nFeature {primary_feature} is connected to {len(connected)} other features:")
        for c in connected[:3]:
            print(f"  Feature {c.feature_idx}: affinity {c.affinity:.2f}")
```

---

## Choosing Parameters

| Parameter | Guidance |
|-----------|----------|
| `n_projection_runs` | 30-100. More = smoother cooccurrence, diminishing returns past ~50 |
| `projection_dims` | 30-100. Lower = faster, higher = more faithful to original structure |
| `min_cluster_size` | Depends on your data. Start with sqrt(n_points) |
| `n_features` (NMF) | Use reconstruction error elbow plot, or domain knowledge (~500 for large repos) |
| `membership_threshold` | 0.01 (1%) — algorithm self-filters unrelated, this just removes numerical noise |
| `affinity_threshold` | 0.001 initially, tune after observing connections |

**Finding optimal n_features:**

```python
from sklearn.decomposition import NMF

errors = []
k_range = range(50, 600, 50)

for k in k_range:
    nmf = NMF(n_components=k, init='nndsvd', max_iter=500)
    W = nmf.fit_transform(cooccurrence)
    errors.append(nmf.reconstruction_err_)

# Plot and look for elbow
import matplotlib.pyplot as plt
plt.plot(k_range, errors, 'bo-')
plt.xlabel('n_features')
plt.ylabel('Reconstruction Error')
plt.title('NMF Elbow Plot')
plt.show()
```

---

## Implementation Plan

### Phase 1: Core Algorithm
- [ ] Implement `SoftClusteringPipeline` class
- [ ] Add sparse cooccurrence matrix support
- [ ] Add NMF with membership extraction
- [ ] Add feature affinity computation
- [ ] Unit tests with synthetic data

### Phase 2: Magaldi Integration
- [ ] Update `INDEX_MAPPING` with new fields
- [ ] Modify `FeatureClusterer` to use soft clustering
- [ ] Update `process_features` to store soft memberships
- [ ] Update `process_subfeatures` for two-level soft clustering
- [ ] Backward compat: keep `cluster_id`/`cluster_label` as primary

### Phase 3: MCP Tools
- [ ] Update `search_features` to return connected features
- [ ] Update `get_feature_members` to include scores
- [ ] Add new `find_associated` tool
- [ ] Update `list_features` with connection info

### Phase 4: Web UI
- [ ] Feature graph visualization
- [ ] Multiple feature badges on elements
- [ ] Related features sidebar
- [ ] Cross-feature navigation

---

## Summary

1. **Random projections** reduce dimensionality while preserving structure (J-L lemma)
2. **Ensemble clustering** across many projections builds a robust **cooccurrence matrix**
3. **Sparse storage** enables scaling to 30k elements
4. **NMF** decomposes cooccurrence into natural **soft cluster memberships**
5. **Feature affinity matrix** reveals which features are connected
6. Each element gets a probability distribution over features — **overlapping membership solved**
7. **Two-level hierarchy** preserved: features → subfeatures, both with soft assignments
8. **Cross-feature subfeature connections** emerge naturally from shared element memberships

This approach combines theoretical guarantees (J-L), ensemble robustness, and interpretable soft assignments (NMF). It's particularly well-suited for code embedding spaces where functions genuinely belong to multiple semantic categories.
