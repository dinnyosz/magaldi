"""Search client protocol and shared types.

Defines the SearchClient Protocol that all backends must implement,
plus backend-agnostic exceptions and helper functions.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class NotFoundError(Exception):
    """Raised when a document is not found in the search backend."""


def _retry_on_overload(
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
):
    """Decorator that retries on HTTP 429 / circuit-breaker errors.

    OpenSearch and Elasticsearch return TransportError(429) when the cluster
    is under memory pressure (circuit breaker) or rate-limited.  The built-in
    ``retry_on_timeout`` only handles socket timeouts, not 429 responses.

    Uses exponential backoff with jitter, matching the pattern in
    ``shared.ai.llm_client._retry_with_backoff``.
    """

    def decorator(fn):  # noqa: ANN001, ANN202
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            last_error: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    status = getattr(e, "status_code", None)
                    if status != 429:
                        raise

                    last_error = e
                    if attempt < max_retries:
                        delay = min(
                            base_delay * (2**attempt) + random.uniform(0, 1),
                            max_delay,
                        )
                        logger.warning(
                            "Search backend 429 (attempt %d/%d), retrying in %.1fs: %s",
                            attempt + 1,
                            max_retries + 1,
                            delay,
                            e,
                        )
                        time.sleep(delay)

            raise last_error  # type: ignore[misc]

        return wrapper

    return decorator


@runtime_checkable
class SearchClient(Protocol):
    """Protocol defining all search backend operations.

    Implementations wrap a specific client library (opensearch-py, elasticsearch-py)
    and normalize the API surface so repositories are backend-agnostic.
    """

    # --- Index management ---

    def index_exists(self, index: str) -> bool: ...

    def index_create(self, index: str, body: dict[str, Any]) -> None: ...

    # --- Document CRUD ---

    def index_document(
        self, index: str, doc_id: str, document: dict[str, Any]
    ) -> dict[str, Any]: ...

    def get_document(
        self, index: str, doc_id: str, source: list[str] | bool | None = None
    ) -> dict[str, Any]: ...

    def update_document(
        self, index: str, doc_id: str, body: dict[str, Any]
    ) -> dict[str, Any]: ...

    def exists_document(self, index: str, doc_id: str) -> bool: ...

    # --- Search / count ---

    def search(self, index: str, body: dict[str, Any], **kwargs: Any) -> dict[str, Any]: ...

    def count(self, index: str, body: dict[str, Any]) -> dict[str, Any]: ...

    def scroll(self, scroll_id: str, scroll: str) -> dict[str, Any]: ...

    def clear_scroll(self, scroll_id: str) -> None: ...

    # --- Batch operations ---

    def mget(
        self, index: str, ids: list[str], source: list[str] | bool | None = None
    ) -> dict[str, Any]: ...

    def bulk(self, operations: list[dict[str, Any]], refresh: bool = False) -> dict[str, Any]: ...

    def bulk_helpers(
        self,
        actions: list[dict[str, Any]],
        raise_on_error: bool = True,
        refresh: bool | str = False,
    ) -> tuple[int, list[Any]]: ...

    # --- Bulk mutation ---

    def delete_by_query(
        self,
        index: str,
        body: dict[str, Any],
        refresh: bool = True,
        timeout: str | None = None,
        request_timeout: int | None = None,
    ) -> dict[str, Any]: ...

    def update_by_query(
        self,
        index: str,
        body: dict[str, Any],
        refresh: bool = True,
        timeout: str | None = None,
        request_timeout: int | None = None,
    ) -> dict[str, Any]: ...

    # --- Cluster ---

    def cluster_health(self) -> dict[str, Any]: ...

    def indices_refresh(self, index: str) -> None: ...

    def indices_stats(self, index: str) -> dict[str, Any]: ...

    def indices_put_settings(self, index: str, body: dict[str, Any]) -> None: ...

    # --- Lifecycle ---

    def close(self) -> None: ...


def get_index_mapping(backend_type: str, dims: int = 1024) -> dict[str, Any]:
    """Return the vector field mapping snippet for the given backend.

    Args:
        backend_type: "opensearch" or "elasticsearch".
        dims: Embedding dimensions.

    Returns:
        Mapping dict for a single dense_vector / knn_vector field.
    """
    if backend_type == "opensearch":
        return {
            "type": "knn_vector",
            "dimension": dims,
            "method": {
                "name": "hnsw",
                "space_type": "cosinesimil",
                "engine": "faiss",
            },
        }
    # elasticsearch
    return {
        "type": "dense_vector",
        "dims": dims,
        "index": True,
        "similarity": "cosine",
    }


def build_vector_query(
    backend_type: str,
    field: str,
    vector: list[float],
    filters: list[dict[str, Any]],
    size: int = 10,
    min_score: float = 0.3,
) -> tuple[dict[str, Any], float]:
    """Build a backend-specific vector similarity query.

    Args:
        backend_type: "opensearch" or "elasticsearch".
        field: Name of the vector field (e.g. "summary_embedding").
        vector: Query vector.
        filters: List of filter clauses (term, terms, exists, etc.).
        size: Maximum results.
        min_score: Minimum cosine similarity (0-1 scale).

    Returns:
        Tuple of (query_body, adjusted_min_score) ready for search().
    """
    if backend_type == "opensearch":
        # OpenSearch native knn query with pre-filter
        query: dict[str, Any] = {
            "knn": {
                field: {
                    "vector": vector,
                    "k": size,
                    "filter": {"bool": {"filter": filters}} if filters else {"match_all": {}},
                }
            }
        }
        return query, min_score
    # Elasticsearch: script_score with cosineSimilarity
    query = {
        "script_score": {
            "query": {"bool": {"filter": filters}},
            "script": {
                "source": f"cosineSimilarity(params.query_vector, '{field}') + 1.0",
                "params": {"query_vector": vector},
            },
        }
    }
    # ES script_score adds +1.0, so adjust min_score
    return query, min_score + 1.0
