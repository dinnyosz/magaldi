"""OpenSearch backend implementation.

Wraps opensearch-py client to conform to the SearchClient protocol.
"""

from __future__ import annotations

from typing import Any

from opensearchpy import NotFoundError as OSNotFoundError
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk as os_bulk

from shared.db.backends.base import NotFoundError, _retry_on_overload


class OpenSearchClient:
    """SearchClient implementation backed by OpenSearch."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        scheme: str = "http",
        timeout: int = 30,
        retry_on_timeout: bool = True,
        max_retries: int = 3,
    ) -> None:
        self._client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_compress=True,
            use_ssl=(scheme == "https"),
            verify_certs=False,
            ssl_show_warn=False,
            timeout=timeout,
            retry_on_timeout=retry_on_timeout,
            max_retries=max_retries,
        )

    # --- Index management ---

    def index_exists(self, index: str) -> bool:
        return bool(self._client.indices.exists(index=index))

    def index_create(self, index: str, body: dict[str, Any]) -> None:
        # OpenSearch uses 'body' parameter
        self._client.indices.create(index=index, body=body)

    # --- Document CRUD ---

    @_retry_on_overload()
    def index_document(
        self, index: str, doc_id: str, document: dict[str, Any]
    ) -> dict[str, Any]:
        return dict(self._client.index(index=index, id=doc_id, body=document))

    @_retry_on_overload()
    def get_document(
        self, index: str, doc_id: str, source: list[str] | bool | None = None
    ) -> dict[str, Any]:
        try:
            kwargs: dict[str, Any] = {"index": index, "id": doc_id}
            if source is not None:
                kwargs["_source"] = source
            return dict(self._client.get(**kwargs))
        except OSNotFoundError as exc:
            raise NotFoundError(str(exc)) from exc

    @_retry_on_overload()
    def update_document(
        self, index: str, doc_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            return dict(self._client.update(index=index, id=doc_id, body=body))
        except OSNotFoundError as exc:
            raise NotFoundError(str(exc)) from exc

    @_retry_on_overload()
    def exists_document(self, index: str, doc_id: str) -> bool:
        return bool(self._client.exists(index=index, id=doc_id))

    # --- Search / count ---

    @_retry_on_overload()
    def search(self, index: str, body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return dict(self._client.search(index=index, body=body, **kwargs))

    @_retry_on_overload()
    def count(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        return dict(self._client.count(index=index, body=body))

    @_retry_on_overload()
    def scroll(self, scroll_id: str, scroll: str) -> dict[str, Any]:
        return dict(self._client.scroll(scroll_id=scroll_id, scroll=scroll))

    def clear_scroll(self, scroll_id: str) -> None:
        self._client.clear_scroll(scroll_id=scroll_id)

    # --- Batch operations ---

    @_retry_on_overload()
    def mget(
        self, index: str, ids: list[str], source: list[str] | bool | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"ids": ids}
        kwargs: dict[str, Any] = {"index": index, "body": body}
        if source is not None:
            kwargs["_source"] = source
        return dict(self._client.mget(**kwargs))

    @_retry_on_overload()
    def bulk(self, operations: list[dict[str, Any]], refresh: bool = False) -> dict[str, Any]:
        return dict(self._client.bulk(body=operations, refresh=refresh))

    @_retry_on_overload()
    def bulk_helpers(
        self,
        actions: list[dict[str, Any]],
        raise_on_error: bool = True,
        refresh: bool | str = False,
    ) -> tuple[int, list[Any]]:
        success, errors = os_bulk(
            self._client, actions, raise_on_error=raise_on_error, refresh=refresh
        )
        return success, errors

    # --- Bulk mutation ---

    @_retry_on_overload()
    def delete_by_query(
        self,
        index: str,
        body: dict[str, Any],
        refresh: bool = True,
        timeout: str | None = None,
        request_timeout: int | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"index": index, "body": body, "refresh": refresh}
        if timeout is not None:
            kwargs["timeout"] = timeout
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        return dict(self._client.delete_by_query(**kwargs))

    @_retry_on_overload()
    def update_by_query(
        self,
        index: str,
        body: dict[str, Any],
        refresh: bool = True,
        timeout: str | None = None,
        request_timeout: int | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"index": index, "body": body, "refresh": refresh}
        if timeout is not None:
            kwargs["timeout"] = timeout
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        return dict(self._client.update_by_query(**kwargs))

    # --- Cluster ---

    def cluster_health(self) -> dict[str, Any]:
        return dict(self._client.cluster.health())

    def indices_refresh(self, index: str) -> None:
        self._client.indices.refresh(index=index)

    def indices_stats(self, index: str) -> dict[str, Any]:
        return dict(self._client.indices.stats(index=index))

    def indices_put_settings(self, index: str, body: dict[str, Any]) -> None:
        self._client.indices.put_settings(index=index, body=body)

    # --- Lifecycle ---

    def close(self) -> None:
        self._client.close()
