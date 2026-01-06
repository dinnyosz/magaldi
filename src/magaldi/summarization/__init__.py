"""Summarization module for Magaldi (Phase 5).

This module handles:
- Generating natural language summaries using Ollama
- Hierarchical processing (files -> classes -> functions)
- Job management and dependency resolution
"""

from magaldi.summarization.summarization import (
    InMemoryJobRepository,
    InMemorySummaryStore,
    JobRepository,
    OllamaClient,
    SummarizationConfig,
    SummarizationError,
    SummarizationResult,
    SummaryStore,
    build_prompt,
    clean_summary,
    generate_summary,
    process_summarization_job,
    truncate_code,
    update_dependencies_after_completion,
)

__all__ = [
    # Protocols
    "JobRepository",
    "SummaryStore",
    # Implementations
    "InMemoryJobRepository",
    "InMemorySummaryStore",
    "OllamaClient",
    # Functions
    "build_prompt",
    "clean_summary",
    "truncate_code",
    "generate_summary",
    "process_summarization_job",
    "update_dependencies_after_completion",
    # Data classes
    "SummarizationConfig",
    "SummarizationResult",
    "SummarizationError",
]
