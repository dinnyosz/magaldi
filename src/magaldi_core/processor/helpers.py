"""Helper functions for element processing.

Contains the core processing logic for individual elements.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

from shared.ai.context_size import compute_element_num_ctx
from shared.ai.embedding import (
    build_caller_embedding_text,
    build_code_embedding_text,
    build_summary_embedding_text,
    normalize_vector,
    validate_vector,
)
from shared.ai.prompts import get_max_tokens_for_element_type
from shared.ai.summarization import build_prompt, clean_summary
from shared.text_utils import humanize_name

from .models import ProcessedElement, ProcessingConfig, _get_model_display_name

if TYPE_CHECKING:
    from magaldi_core.code_parser import CodeElement
    from magaldi_core.job_tracker import SummaryCache
    from shared.ai.embedding import CodeEmbeddingClient
    from shared.ai.summarization import SummarizationLLMClient
    from shared.db.store import Repository

    from .status import WorkerStatus

logger = logging.getLogger(__name__)


def should_embed(element: CodeElement) -> bool:
    """Determine if element should be embedded.

    Args:
        element: Code element to check.

    Returns:
        True if element should be embedded.
    """
    # All code elements get embedded (including imports for semantic search)
    return element.element_type in ("file", "class", "interface", "type_alias", "trait", "enum", "function", "method", "constant", "variable", "import")


# Element types that get handcrafted summaries (no LLM needed)
# These still get embedded for semantic search
_HANDCRAFTED_SUMMARY_TYPES = frozenset({"import"})


def _generate_import_summary(element: CodeElement) -> str:
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


# Small function/method handcrafted summaries
# Functions/methods below a line threshold skip LLM and use code/docstring directly

# Section header patterns that mark end of docstring description paragraph
_DOCSTRING_SECTION_HEADERS = re.compile(
    r"^\s*("
    r"Args?:|Returns?:|Raises?:|Yields?:|Note:|Notes?:|"  # Google
    r"Example:|Examples:|Todo:|Attributes?:|See Also:|References?:|Warnings?:|"  # Google cont.
    r"Parameters?[\s:]*$|Returns?\s*$|Raises?\s*$|"  # NumPy (followed by ---)
    r":param\s|:type\s|:returns?:|:rtype:|:raises?:|"  # Sphinx
    r"@param\s|@returns?\s|@throws?\s|@type\s"  # JSDoc/PHPDoc
    r")",
    re.IGNORECASE,
)


def _extract_docstring_description(docstring: str) -> str:
    """Extract the description paragraph from a docstring.

    Takes everything before the first section header (Args:, Returns:,
    :param, @param, etc.), joining lines into a single string.

    Args:
        docstring: Raw docstring text.

    Returns:
        Description text with lines joined by spaces.
    """
    lines = docstring.strip().split("\n")
    description_lines: list[str] = []

    for line in lines:
        # Stop at section headers
        if _DOCSTRING_SECTION_HEADERS.match(line):
            break
        description_lines.append(line.strip())

    # Join non-empty lines with spaces
    text = " ".join(line for line in description_lines if line)
    return text.strip()

def _get_element_line_count(element: CodeElement) -> int:
    """Get non-empty line count for an element.

    Uses code_metrics if available (populated by parser), otherwise
    computes from raw_code, falling back to line_start/line_end.

    Args:
        element: Code element.

    Returns:
        Non-empty line count.
    """
    if element.code_metrics and "line_count" in element.code_metrics:
        return element.code_metrics["line_count"]  # type: ignore[no-any-return]
    if element.raw_code:
        return sum(1 for line in element.raw_code.split("\n") if line.strip())
    if element.line_end and element.line_start:
        return max(1, element.line_end - element.line_start + 1)  # type: ignore[no-any-return]
    return 0


def _is_small_function(element: CodeElement, threshold: int) -> bool:
    """Check if a function/method is small enough for a handcrafted summary.

    Args:
        element: Code element to check.
        threshold: Max non-empty lines for handcrafted summary (0 to disable).

    Returns:
        True if element is a function/method with <= threshold lines.
    """
    if threshold <= 0:
        return False
    if element.element_type not in ("function", "method"):
        return False
    return _get_element_line_count(element) <= threshold


def _generate_small_function_summary(element: CodeElement) -> str:
    """Generate a handcrafted summary for a small function/method.

    For small functions, the code itself is the best explanation. Priority:
    1. Description paragraph from docstring (developer's own description)
    2. Signature (for trivial getters/setters without docstring)
    3. Raw code (fallback — code IS the explanation)

    Args:
        element: A small function or method element.

    Returns:
        A summary string suitable for embedding.
    """
    # Try docstring description paragraph
    if element.docstring:
        desc = _extract_docstring_description(element.docstring)
        # Strip trailing period for consistency with clean_summary
        desc = desc.rstrip(".")
        if desc:
            return desc

    # Try signature
    if element.signature:
        return element.signature.strip()  # type: ignore[no-any-return]

    # Fallback to raw code
    code = (element.raw_code or "").strip()
    if code:
        if len(code) <= 200:
            return code
        # Truncate to first few non-empty lines
        lines = [line for line in code.split("\n") if line.strip()][:3]
        return "\n".join(lines)

    return f"{element.element_type.title()}: {element.name}"


# =============================================================================
# TEST ELEMENT HANDCRAFTED SUMMARIES
# =============================================================================


def _generate_test_function_summary(element: CodeElement) -> str:
    """Generate a handcrafted summary for a test function/method.

    Test functions are self-documenting via naming conventions.
    Priority:
    1. Description paragraph from docstring (developer's own explanation)
    2. Humanized function name (e.g., "test user login with expired token")

    Args:
        element: A test function or method element.

    Returns:
        A summary string suitable for embedding.
    """
    if element.docstring:
        desc = _extract_docstring_description(element.docstring)
        desc = desc.rstrip(".")
        if desc:
            return desc
    return humanize_name(element.name)


def _generate_test_class_summary(element: CodeElement) -> str:
    """Generate a handcrafted summary for a test class.

    Test class names describe what they test (e.g., TestUserAuthentication).
    Priority:
    1. Description paragraph from docstring
    2. Humanized class name (e.g., "test user authentication")

    Args:
        element: A test class element.

    Returns:
        A summary string suitable for embedding.
    """
    if element.docstring:
        desc = _extract_docstring_description(element.docstring)
        desc = desc.rstrip(".")
        if desc:
            return desc
    return humanize_name(element.name)


def _generate_test_file_summary(element: CodeElement) -> str:
    """Generate a handcrafted summary for a test file.

    Priority:
    1. Module docstring description
    2. Humanized file name (e.g., test_user_auth.py -> "test user auth")

    Args:
        element: A test file element.

    Returns:
        A summary string suitable for embedding.
    """
    if element.docstring:
        desc = _extract_docstring_description(element.docstring)
        desc = desc.rstrip(".")
        if desc:
            return desc
    # Use humanized file stem (strip extension)
    file_stem = element.name.rsplit(".", 1)[0] if "." in element.name else element.name
    return humanize_name(file_stem)


# =============================================================================
# HANDCRAFTED SUMMARY DISPATCH
# =============================================================================


# Minimum docstring description length to qualify for docstring-as-summary.
# Shorter descriptions are too terse to be useful summaries.
_MIN_DOCSTRING_DESC_LENGTH = 10


def _get_craft_reason(element: CodeElement, config: ProcessingConfig) -> str | None:
    """Determine why an element should use a handcrafted summary, if at all.

    Returns a reason string for handcrafted elements, or None if the element
    should be summarized by the LLM. Priority order:
    1. Test elements → "test"
    2. Imports → "import"
    3. Elements with meaningful docstrings (when use_docstrings enabled) → "docstring"
    4. Small functions/methods → "small"

    Docstring is checked before small-function so that a short function with
    a good docstring uses the human-written description rather than a generic
    "small function" template.

    Args:
        element: Code element to check.
        config: Processing configuration with thresholds.

    Returns:
        Craft reason string, or None if element needs LLM summarization.
    """
    if element.is_test:
        return "test"
    if element.element_type in _HANDCRAFTED_SUMMARY_TYPES:
        return "import"
    if config.use_docstrings and element.docstring:
        desc = _extract_docstring_description(element.docstring)
        if len(desc) >= _MIN_DOCSTRING_DESC_LENGTH:
            return "docstring"
    if element.element_type in ("function", "method") and _is_small_function(element, config.handcrafted_max_lines):
        return "small"
    return None


def _should_handcraft(element: CodeElement, config: ProcessingConfig) -> bool:
    """Check if an element should use a handcrafted summary instead of LLM.

    Centralizes the handcrafted/LLM decision so it can be used both in
    processing and ETA tracking. Delegates to _get_craft_reason() for
    the actual decision logic.

    Args:
        element: Code element to check.
        config: Processing configuration with thresholds.

    Returns:
        True if element should skip LLM and use a handcrafted summary.
    """
    return _get_craft_reason(element, config) is not None


def _generate_docstring_summary(element: CodeElement) -> str:
    """Generate a summary from an element's docstring description paragraph.

    Uses the developer-written docstring as the summary, which is often
    more accurate than an LLM-generated one. Strips trailing period for
    consistency with clean_summary().

    Args:
        element: Code element with a docstring.

    Returns:
        A summary string from the docstring description.
    """
    desc = _extract_docstring_description(element.docstring or "")
    desc = desc.rstrip(".")
    if desc:
        return desc
    # Fallback: shouldn't happen if _get_craft_reason checked length
    return f"{element.element_type.title()}: {element.name}"


def _generate_handcrafted_summary(
    element: CodeElement, craft_reason: str | None = None
) -> str:
    """Generate a handcrafted summary based on element type and craft reason.

    Dispatches to the appropriate per-type generator. Each element type
    that supports handcrafting has its own method with type-specific logic.

    Args:
        element: Code element to generate summary for.
        craft_reason: Why this element is handcrafted ("test", "import",
            "small", "docstring"). If None, derives reason from element.

    Returns:
        A summary string suitable for embedding.
    """
    # Test elements get their own generators (self-documenting via naming)
    if craft_reason == "test" or (craft_reason is None and element.is_test):
        if element.element_type in ("function", "method"):
            return _generate_test_function_summary(element)
        if element.element_type == "class":
            return _generate_test_class_summary(element)
        if element.element_type == "file":
            return _generate_test_file_summary(element)
        # Fallback for test variables/constants/etc.
        return humanize_name(element.name)

    # Docstring-based summary (any element type with a meaningful docstring)
    if craft_reason == "docstring":
        return _generate_docstring_summary(element)

    # Non-test elements
    if craft_reason == "import" or (
        craft_reason is None and element.element_type in _HANDCRAFTED_SUMMARY_TYPES
    ):
        return _generate_import_summary(element)
    if craft_reason == "small" or (
        craft_reason is None and element.element_type in ("function", "method")
    ):
        return _generate_small_function_summary(element)
    # Fallback for future types
    return f"{element.element_type.title()}: {element.name}"


# =============================================================================
# ELEMENT PROCESSING HELPERS
# =============================================================================


def _summarize_element(
    element: CodeElement,
    summary_cache: SummaryCache,
    llm_client: SummarizationLLMClient,
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

    # Generate summary (select model and max_tokens based on element type)
    model_config = config.get_model_for_element_type(element.element_type)
    max_tokens = get_max_tokens_for_element_type(
        element.element_type, default=config.summarize_max_tokens
    )
    # Compute per-element context size for optimal KV cache efficiency
    num_ctx = compute_element_num_ctx(
        element.element_type,
        len(element.raw_code or ""),
    )
    raw_summary = llm_client.generate(
        prompt=prompt,
        temperature=config.summarize_temperature,
        top_p=config.summarize_top_p,
        max_tokens=max_tokens,
        timeout=config.summarize_timeout,
        model=model_config.name,
        num_ctx=num_ctx,
        top_k=config.summarize_top_k,
        min_p=config.summarize_min_p,
        presence_penalty=config.summarize_presence_penalty,
        repetition_penalty=config.summarize_repetition_penalty,
    )
    response_tokens = len(raw_summary) // 4

    # Clean and return
    return clean_summary(raw_summary), prompt_tokens, response_tokens


def _embed_element(
    element: CodeElement,
    summary_cache: SummaryCache,
    embed_client: CodeEmbeddingClient,
    config: ProcessingConfig,
    on_stage_change: Callable[[str], None] | None = None,
    cached_embeddings: dict[str, list[float] | None] | None = None,
) -> tuple[list[float], list[float], list[float] | None, float, float, float]:
    """Generate embeddings for an element, reusing cached vectors when available.

    Args:
        element: Element to embed.
        summary_cache: Cache with summaries for context.
        embed_client: Embedding client.
        config: Processing configuration.
        on_stage_change: Optional callback to update status stage.
        cached_embeddings: Optional dict with pre-existing embedding vectors
            (keys: summary_embedding, code_embedding, caller_embedding).
            Skips API calls for any embedding type already present.

    Returns:
        Tuple of (summary_embedding, code_embedding, caller_embedding,
                  summary_embed_time, code_embed_time, caller_embed_time).

    Raises:
        ValueError: If embedding validation fails.
    """
    cached = cached_embeddings or {}

    # Summary embedding
    summary_embed_time = 0.0
    cached_summary = cached.get("summary_embedding")
    if cached_summary is not None:
        summary_embedding = cached_summary
    else:
        if on_stage_change:
            on_stage_change("summ_embed")
        summary_text = build_summary_embedding_text(element, summary_cache, config.embed_max_context)
        summary_start = time.time()
        summary_embedding = embed_client.embed_single(summary_text, timeout=config.embed_timeout)
        summary_embed_time = time.time() - summary_start

        if not validate_vector(summary_embedding, config.embed_dimensions):
            raise ValueError(
                f"Invalid summary embedding: expected {config.embed_dimensions} dims, "
                f"got {len(summary_embedding)}"
            )
        summary_embedding = normalize_vector(summary_embedding)

    # Code embedding
    code_embed_time = 0.0
    cached_code = cached.get("code_embedding")
    if cached_code is not None:
        code_embedding = cached_code
    else:
        if on_stage_change:
            on_stage_change("code_embed")
        code_text = build_code_embedding_text(element, config.embed_max_context)
        code_start = time.time()
        code_embedding = embed_client.embed_single(code_text, timeout=config.embed_timeout)
        code_embed_time = time.time() - code_start

        if not validate_vector(code_embedding, config.embed_dimensions):
            raise ValueError(
                f"Invalid code embedding: expected {config.embed_dimensions} dims, "
                f"got {len(code_embedding)}"
            )
        code_embedding = normalize_vector(code_embedding)

    # Caller embedding (passport + outbound calls for asymmetric resolution)
    # Only elements with calls need caller_embedding
    caller_embed_time = 0.0
    caller_embedding: list[float] | None = None
    if element.calls:
        cached_caller = cached.get("caller_embedding")
        if cached_caller is not None:
            caller_embedding = cached_caller
        else:
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

    return summary_embedding, code_embedding, caller_embedding, summary_embed_time, code_embed_time, caller_embed_time


def _index_element(
    element: CodeElement,
    summary: str,
    summary_embedding: list[float] | None,
    code_embedding: list[float] | None,
    caller_embedding: list[float] | None,
    repo: Repository,
    file_hash: str | None = None,
    element_count: int | None = None,
    craft_reason: str | None = None,
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
        craft_reason: Why this element was handcrafted (or None for LLM).

    Returns:
        True on success.
    """
    # Index the element
    repo.index_element(element, indexed_at=datetime.now(), file_hash=file_hash, element_count=element_count)

    # Store summary (with craft reason if applicable)
    repo.store_summary(element.element_id, summary, craft_reason=craft_reason)

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
    element: CodeElement,
    summary_cache: SummaryCache,
    llm_client: SummarizationLLMClient | None,
    embed_client: CodeEmbeddingClient | None,
    config: ProcessingConfig,
    file_hashes: dict[str, str] | None,
    element_counts: dict[str, int] | None,
    repo: Repository,
    worker_id: int,
    worker_status: WorkerStatus,
    on_status_change: Callable[[], None] | None = None,
    cached_embeddings: dict[str, list[float] | None] | None = None,
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
        cached_embeddings: Optional dict with pre-existing embedding vectors
            (keys: summary_embedding, code_embedding, caller_embedding).
            Skips API calls for any embedding type already present.

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
        # Get model for this element type (needs num_ctx for 1024-tier fallback)
        element_model = config.get_model_for_element_type(element.element_type, num_ctx)
        # Format tier compactly: 2048 -> "2K", 32768 -> "32K"
        ctx_display = f"{num_ctx // 1024}K" if num_ctx >= 1024 else str(num_ctx)
        # Display tiered model name for Ollama (e.g., "qwen3:4b-instruct-4k")
        model_display = _get_model_display_name(element_model, num_ctx)
        update_status("summarizing", model_display, ctx_display)
        craft_reason = _get_craft_reason(element, config)
        if config.skip_ai:
            summary = f"{element.element_type.title()}: {element.name}"
        elif craft_reason is not None:
            # Handcrafted summary: per-type generator, no LLM call needed
            summary = _generate_handcrafted_summary(element, craft_reason)
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
                    element, summary_cache, embed_client, config, on_embed_stage,
                    cached_embeddings=cached_embeddings,
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

        _index_element(element, summary, summary_embedding, code_embedding, caller_embedding, repo, file_hash, element_count, craft_reason=craft_reason)

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
            model_name=model_display,
            craft_reason=craft_reason,
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
            model_name=model_display,
            craft_reason=craft_reason,
            error=str(e),
        )
