"""Search backend abstractions for Magaldi.

Provides a Protocol-based interface for search backends (OpenSearch, Elasticsearch).
"""

from shared.db.backends.base import NotFoundError, SearchClient
from shared.db.backends.factory import create_client

__all__ = [
    "SearchClient",
    "NotFoundError",
    "create_client",
]
