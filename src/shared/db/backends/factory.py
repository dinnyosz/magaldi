"""Search backend factory.

Creates the appropriate SearchClient based on configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.config import SearchBackendConfig
    from shared.db.backends.base import SearchClient


def create_client(config: SearchBackendConfig) -> SearchClient:
    """Create a SearchClient from configuration.

    Args:
        config: Search backend configuration with type, host, port, etc.

    Returns:
        Configured SearchClient instance.

    Raises:
        ValueError: If backend type is not supported.
    """
    if config.type == "opensearch":
        from shared.db.backends.opensearch import OpenSearchClient

        return OpenSearchClient(
            host=config.host,
            port=config.port,
            scheme=config.scheme,
            timeout=config.timeout,
            retry_on_timeout=config.retry_on_timeout,
            max_retries=config.max_retries,
            pool_maxsize=config.pool_maxsize,
        )
    elif config.type == "elasticsearch":
        from shared.db.backends.elasticsearch import ElasticSearchClient

        return ElasticSearchClient(
            host=config.host,
            port=config.port,
            scheme=config.scheme,
            timeout=config.timeout,
            retry_on_timeout=config.retry_on_timeout,
            max_retries=config.max_retries,
            pool_maxsize=config.pool_maxsize,
        )
    else:
        raise ValueError(
            f"Unsupported search backend type: {config.type!r}. "
            f"Supported: 'opensearch', 'elasticsearch'"
        )
