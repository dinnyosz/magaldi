"""Embedding module for Magaldi (Phase 6).

This module handles:
- Generating vector embeddings using Ollama
- Building hierarchical context for embeddings
- Storing embeddings in Elasticsearch
"""

from magaldi.embedding.embedding import (
    EmbeddingConfig,
    EmbeddingError,
    EmbeddingJobRepository,
    EmbeddingResult,
    EmbeddingStore,
    InMemoryEmbeddingJobRepository,
    InMemoryEmbeddingStore,
    OllamaEmbedClient,
    build_embedding_text,
    estimate_tokens,
    normalize_vector,
    process_embedding_job,
    validate_context_length,
    validate_vector,
)

__all__ = [
    # Protocols
    "EmbeddingJobRepository",
    "EmbeddingStore",
    # Implementations
    "InMemoryEmbeddingJobRepository",
    "InMemoryEmbeddingStore",
    "OllamaEmbedClient",
    # Functions
    "build_embedding_text",
    "estimate_tokens",
    "validate_context_length",
    "normalize_vector",
    "validate_vector",
    "process_embedding_job",
    # Data classes
    "EmbeddingConfig",
    "EmbeddingResult",
    "EmbeddingError",
]
