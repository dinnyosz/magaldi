"""Storage module for Magaldi (Phase 4).

This module handles:
- Storing parsed files and elements to Elasticsearch
- Creating summarization and embedding jobs
"""

from magaldi.storage.storage import (
    DatabaseRepository,
    InMemoryDatabaseRepository,
    InMemorySearchRepository,
    SearchRepository,
    StorageError,
    StorageResult,
    create_embedding_jobs,
    create_summarization_jobs,
    handle_deletions,
    index_elements,
    should_embed,
    store_parsed_files,
    store_storage_result,
)

__all__ = [
    # Protocols
    "DatabaseRepository",
    "SearchRepository",
    # Implementations
    "InMemoryDatabaseRepository",
    "InMemorySearchRepository",
    # Functions
    "handle_deletions",
    "store_parsed_files",
    "index_elements",
    "create_summarization_jobs",
    "create_embedding_jobs",
    "should_embed",
    "store_storage_result",
    # Data classes
    "StorageResult",
    "StorageError",
]
