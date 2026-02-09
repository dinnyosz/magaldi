"""Helper functions for element processing.

Contains the core processing logic for individual elements.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Callable

from shared.ai.context_size import compute_element_num_ctx
from shared.ai.embedding import (
    build_caller_embedding_text,
    build_code_embedding_text,
    build_summary_embedding_text,
    normalize_vector,
    validate_vector,
)
from shared.ai.summarization import build_prompt, clean_summary

from .models import ProcessedElement, ProcessingConfig, _get_model_display_name

if TYPE_CHECKING:
    from magaldi_core.code_parser import CodeElement
    from magaldi_core.job_tracker import SummaryCache
    from shared.ai.embedding import CodeEmbeddingClient
    from shared.ai.summarization import SummarizationLLMClient
    from shared.db.store import Repository
    from .status import WorkerStatus

logger = logging.getLogger(__name__)


def should_embed(element: "CodeElement") -> bool:
    """Determine if element should be embedded.

    Args:
        element: Code element to check.

    Returns:
        True if element should be embedded.
    """
    # All code elements get embedded (including imports for semantic search)
    if element.element_type in (
        "file", "class", "interface", "type_alias", "trait", "enum",
        "function", "method", "constant", "variable", "import"
    ):
        return True

    return False


# Element types that get handcrafted summaries (no LLM needed)
# These still get embedded for semantic search
_HANDCRAFTED_SUMMARY_TYPES = frozenset({"import"})


def _generate_import_summary(element: "CodeElement") -> str:
    """Generate a handcrafted summary for import elements.

    Works across languages by using the raw code directly,
    which the embedding model understands semantically.

    Args:
        element: An import element with raw_code.

    Returns:
        A simple summary string suitable for embedding.
    """
    code = (element.signature or element.raw_code or "").strip()
    if not code:
        return f"Imports {element.name}" if element.name else "Import statement"

    # The code itself is the best description - embedding model understands it
    # Just add context prefix for clarity
    return f"Import: {code}"


# =============================================================================
# ELEMENT PROCESSING HELPERS
# =============================================================================


def _summarize_element(
    element: "CodeElement",
    summary_cache: "SummaryCache",
    llm_client: "SummarizationLLMClient",
    config: ProcessingConfig,
) -> tuple[str, int, int]:
    """Generate summary for an element.

    Args:
        element: Element to summarize.
        summary_cache: Cache with parent summaries.
        llm_client: LLM client for text generation.
        config: Processing configuration.

    Returns:
        Tuple of (summary, prompt_tokens, response_tokens).
    """
    # Get parent summaries for context
    parent_summaries = summary_cache.get_parent_summaries(element)

    # Build prompt with context
    prompt = build_prompt(element, parent_summaries, config.max_code_tokens)
    prompt_tokens = len(prompt) // 4

    # Generate summary (select model based on element type)
    model_config = config.get_model_for_element_type(element.element_type)
    # Compute per-element context size for optimal KV cache efficiency
    num_ctx = compute_element_num_ctx(
        element.element_type,
        len(element.raw_code or ""),
    )
    raw_summary = llm_client.generate(
        prompt=prompt,
        temperature=config.summarize_temperature,
        max_tokens=config.summarize_max_tokens,
        timeout=config.summarize_timeout,
        model=model_config.name,
        num_ctx=num_ctx,
    )
    response_tokens = len(raw_summary) // 4

    # Clean and return
    return clean_summary(raw_summary), prompt_tokens, response_tokens


def _embed_element(
    element: "CodeElement",
    summary_cache: "SummaryCache",
    embed_client: "CodeEmbeddingClient",
    config: ProcessingConfig,
    on_stage_change: Callable[[str], None] | None = None,
) -> tuple[list[float], list[float], list[float], float, float, float]:
    """Generate all three embeddings for an element.

    Args:
        element: Element to embed.
        summary_cache: Cache with summaries for context.
        embed_client: Embedding client.
        config: Processing configuration.
        on_stage_change: Optional callback to update status stage.

    Returns:
        Tuple of (summary_embedding, code_embedding, caller_embedding,
                  summary_embed_time, code_embed_time, caller_embed_time).

    Raises:
        ValueError: If embedding validation fails.
    """
    # Summary embedding
    if on_stage_change:
        on_stage_change("summ_embed")
    summary_text = build_summary_embedding_text(element, summary_cache, config.embed_max_context)
    summary_start = time.time()
    summary_embedding = embed_client.embed_single(summary_text, timeout=config.embed_timeout)
    summary_embed_time = time.time() - summary_start

    # Validate dimensions
    if not validate_vector(summary_embedding, config.embed_dimensions):
        raise ValueError(
            f"Invalid summary embedding: expected {config.embed_dimensions} dims, "
            f"got {len(summary_embedding)}"
        )
    summary_embedding = normalize_vector(summary_embedding)

    # Code embedding
    if on_stage_change:
        on_stage_change("code_embed")
    code_text = build_code_embedding_text(element, config.embed_max_context)
    code_start = time.time()
    code_embedding = embed_client.embed_single(code_text, timeout=config.embed_timeout)
    code_embed_time = time.time() - code_start

    # Validate dimensions
    if not validate_vector(code_embedding, config.embed_dimensions):
        raise ValueError(
            f"Invalid code embedding: expected {config.embed_dimensions} dims, "
            f"got {len(code_embedding)}"
        )
    code_embedding = normalize_vector(code_embedding)

    # Caller embedding (passport + outbound calls for asymmetric resolution)
    # Only elements with calls benefit — without calls, text is identical to summary
    caller_embed_time = 0.0
    if element.calls:
        if on_stage_change:
            on_stage_change("caller_embed")
        caller_text = build_caller_embedding_text(element, summary_cache, config.embed_max_context)
        caller_start = time.time()
        caller_embedding = embed_client.embed_single(caller_text, timeout=config.embed_timeout)
        caller_embed_time = time.time() - caller_start

        if not validate_vector(caller_embedding, config.embed_dimensions):
            raise ValueError(
                f"Invalid caller embedding: expected {config.embed_dimensions} dims, "
                f"got {len(caller_embedding)}"
            )
        caller_embedding = normalize_vector(caller_embedding)
    else:
        # No calls — reuse summary embedding (identical text)
        caller_embedding = summary_embedding

    return summary_embedding, code_embedding, caller_embedding, summary_embed_time, code_embed_time, caller_embed_time


def _index_element(
    element: "CodeElement",
    summary: str,
    summary_embedding: list[float] | None,
    code_embedding: list[float] | None,
    caller_embedding: list[float] | None,
    repo: "Repository",
    file_hash: str | None = None,
    element_count: int | None = None,
) -> bool:
    """Index element to search backend with summary and all embeddings.

    Args:
        element: Element to index.
        summary: Generated summary.
        summary_embedding: Summary embedding vector (or None if not embedded).
        code_embedding: Code embedding vector (or None if not embedded).
        caller_embedding: Caller embedding vector (or None if not embedded).
        repo: Search repository.
        file_hash: File hash for all elements.
        element_count: Total element count in file (only for file-level elements).

    Returns:
        True on success.
    """
    # Index the element
    repo.index_element(element, indexed_at=datetime.now(), file_hash=file_hash, element_count=element_count)

    # Store summary
    repo.store_summary(element.element_id, summary)

    # Store embeddings if present (using type-specific methods)
    if summary_embedding is not None:
        repo.store_summary_embedding(element.element_id, summary_embedding)
    if code_embedding is not None:
        repo.store_code_embedding(element.element_id, code_embedding)
    if caller_embedding is not None:
        repo.store_caller_embedding(element.element_id, caller_embedding)

    # Store imports for file elements
    if element.element_type == "file" and element.imports:
        imports_data = [
            {"name": imp.name, "module": imp.module, "alias": imp.alias, "line": imp.line}
            for imp in element.imports
        ]
        repo.store_imports(element.element_id, imports_data)

    # Store calls for function/method/file elements
    if element.element_type in ("function", "method", "file") and element.calls:
        calls_data = [
            {
                "name": call.name,
                "receiver": call.receiver,
                "line": call.line,
                "resolved_id": call.resolved_id,
                "category": call.category,
            }
            for call in element.calls
        ]
        repo.store_calls(element.element_id, calls_data)

    return True


def _process_single_element(
    element: "CodeElement",
    summary_cache: "SummaryCache",
    llm_client: "SummarizationLLMClient | None",
    embed_client: "CodeEmbeddingClient | None",
    config: ProcessingConfig,
    file_hashes: dict[str, str] | None,
    element_counts: dict[str, int] | None,
    repo: "Repository",
    worker_id: int,
    worker_status: "WorkerStatus",
    on_status_change: Callable[[], None] | None = None,
) -> ProcessedElement:
    """Process a single element: summarize -> embed -> index.

    Args:
        element: Element to process.
        summary_cache: Cache for summaries.
        llm_client: LLM client for summarization (None if skip_ai).
        embed_client: Embedding client (None if skip_ai).
        config: Processing configuration.
        file_hashes: Optional dict mapping relative_path to file hash.
        element_counts: Optional dict mapping relative_path to element count.
        repo: Search repository for indexing.
        worker_id: Worker thread ID.
        worker_status: Status tracker for workers.
        on_status_change: Optional callback when worker status changes.

    Returns:
        ProcessedElement with timing info and success/error status.
    """
    start_wall = time.time()
    summarize_time = 0.0
    embed_time = 0.0
    summary_embed_time = 0.0
    code_embed_time = 0.0
    prompt_tokens = 0
    response_tokens = 0
    num_ctx = 0

    # Build hierarchical display name: [type] .../path/file.py → Class → method
    def build_display_name(max_path_len: int = 60) -> str:
        parts = []
        # Add path (truncated from left if too long)
        path = element.relative_path
        if len(path) > max_path_len:
            path = "..." + path[-(max_path_len - 3):]
        if element.element_type == "file":
            # For file elements, show the path as the name
            parts.append(path)
        else:
            parts.append(path)
            # Add parent class if method
            if element.parent_id:
                parent = summary_cache.get_element(element.parent_id)
                if parent and parent.element_type == "class":
                    parts.append(parent.name)
            # Add element name
            parts.append(element.name)
        # Prefix with element type (use angle brackets to avoid Rich markup interpretation)
        return f"<{element.element_type}> " + " → ".join(parts)

    display_name = build_display_name()
    # Get model for this element type
    element_model = config.get_model_for_element_type(element.element_type)

    # Track current stage start time for elapsed display
    stage_start_time = time.time()

    def update_status(stage: str, model: str = "", ctx_size: str = "") -> None:
        nonlocal stage_start_time
        stage_start_time = time.time()
        worker_status.set(worker_id, display_name, stage, model, ctx_size, stage_start_time)
        if on_status_change:
            on_status_change()

    try:
        # Step 1: Summarize
        # Compute context tier for display
        num_ctx = compute_element_num_ctx(
            element.element_type,
            len(element.raw_code or ""),
        )
        # Format tier compactly: 2048 -> "2K", 32768 -> "32K"
        ctx_display = f"{num_ctx // 1024}K" if num_ctx >= 1024 else str(num_ctx)
        # Display tiered model name for Ollama (e.g., "qwen3:4b-instruct-4k")
        model_display = _get_model_display_name(element_model, num_ctx)
        update_status("summarizing", model_display, ctx_display)
        if config.skip_ai:
            summary = f"{element.element_type.title()}: {element.name}"
        elif element.element_type in _HANDCRAFTED_SUMMARY_TYPES:
            # Use handcrafted summary (no LLM call needed)
            summary = _generate_import_summary(element)
        else:
            api_start = time.time()
            summary, prompt_tokens, response_tokens = _summarize_element(element, summary_cache, llm_client, config)
            summarize_time = time.time() - api_start

            if prompt_tokens > num_ctx:
                logger.warning(
                    "Tier overflow: %s %s prompt=%d tier=%d",
                    element.element_type, element.name, prompt_tokens, num_ctx,
                )

        # Cache summary for children
        summary_cache.add_summary(element.element_id, summary)

        # Step 2: Embed (if applicable) - generate summary, code, and caller embeddings
        summary_embedding: list[float] | None = None
        code_embedding: list[float] | None = None
        caller_embedding: list[float] | None = None
        caller_embed_time = 0.0
        if should_embed(element):
            if config.skip_ai:
                # Skip embeddings entirely - don't generate zero vectors
                # (search backend rejects zero-magnitude vectors)
                update_status("summ_embed", config.embed_model.name, "-")
                update_status("code_embed", config.embed_model.name, "-")
                # Leave embeddings as None
            else:
                # Generate all embeddings (returns tuple with timing)
                # Pass callback to update status between embedding phases
                def on_embed_stage(stage: str) -> None:
                    update_status(stage, config.embed_model.name, "-")
                summary_embedding, code_embedding, caller_embedding, summary_embed_time, code_embed_time, caller_embed_time = _embed_element(
                    element, summary_cache, embed_client, config, on_embed_stage
                )
                embed_time = summary_embed_time + code_embed_time + caller_embed_time

        # Step 3: Index (only after summarize+embed complete)
        update_status("indexing")
        # Store file_hash on ALL elements (not just file elements) for change detection
        # This allows us to delete all elements by file_hash if needed
        file_hash = file_hashes.get(element.relative_path) if file_hashes else None
        # Store element_count only on FILE elements for completeness verification
        element_count = None
        if element.element_type == "file" and element_counts:
            element_count = element_counts.get(element.relative_path)

        _index_element(element, summary, summary_embedding, code_embedding, caller_embedding, repo, file_hash, element_count)

        worker_status.clear(worker_id)
        wall_time = time.time() - start_wall

        return ProcessedElement(
            element_id=element.element_id,
            success=True,
            wall_time=wall_time,
            summarize_time=summarize_time,
            embed_time=embed_time,
            summary_embed_time=summary_embed_time,
            code_embed_time=code_embed_time,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            assigned_tier=num_ctx,
        )

    except Exception as e:
        worker_status.clear(worker_id)
        wall_time = time.time() - start_wall
        return ProcessedElement(
            element_id=element.element_id,
            success=False,
            wall_time=wall_time,
            summarize_time=summarize_time,
            embed_time=embed_time,
            summary_embed_time=summary_embed_time,
            code_embed_time=code_embed_time,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            assigned_tier=num_ctx,
            error=str(e),
        )
