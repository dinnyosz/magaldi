"""AI-powered glossary extraction from feature summaries."""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from shared.ai.llm_client import LLMClient, LLMError

if TYPE_CHECKING:
    from shared.config import MagaldiConfig


# =============================================================================
# PROGRESS STATE CLASSES
# =============================================================================


@dataclass
class GlossaryTimingStats:
    """Timing statistics for glossary extraction."""

    start_time: float = 0.0
    total_api_time: float = 0.0
    features_processed: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def elapsed(self) -> float:
        """Elapsed wall time since start."""
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time

    @property
    def avg_api_time(self) -> float:
        """Average API time per feature."""
        if self.features_processed == 0:
            return 0.0
        return self.total_api_time / self.features_processed

    def record_api_call(self, api_time: float) -> None:
        """Record an API call timing."""
        with self._lock:
            self.total_api_time += api_time
            self.features_processed += 1

    def eta_seconds(self, completed: int, total: int, num_workers: int) -> float:
        """Estimate time remaining based on current progress."""
        if completed == 0 or self.elapsed == 0:
            return 0.0
        rate = completed / self.elapsed
        remaining = total - completed
        return remaining / rate if rate > 0 else 0.0


class GlossaryWorkerStatus:
    """Thread-safe tracking of worker status."""

    def __init__(self) -> None:
        self._status: dict[int, tuple[str, str]] = {}  # worker_id -> (feature_label, model)
        self._lock = threading.Lock()

    def set_status(self, worker_id: int, feature_label: str, model: str) -> None:
        """Set worker status."""
        with self._lock:
            self._status[worker_id] = (feature_label, model)

    def clear_status(self, worker_id: int) -> None:
        """Clear worker status (worker is idle)."""
        with self._lock:
            if worker_id in self._status:
                del self._status[worker_id]

    def get_all(self) -> dict[int, tuple[str, str]]:
        """Get all worker statuses."""
        with self._lock:
            return dict(self._status)


@dataclass
class GlossaryProgressState:
    """Progress state for glossary extraction."""

    total: int
    completed: int
    failed: int
    terms_extracted: int
    timing: GlossaryTimingStats
    workers: GlossaryWorkerStatus
    num_workers: int


# =============================================================================
# GLOSSARY EXTRACTION PROMPTS (Optimized for Prefix Caching)
# =============================================================================
# System messages are STATIC and get cached by Ollama's KV cache.
# User messages contain VARIABLE content (feature info, term context).

# Phase 1: Extract term names
GLOSSARY_EXTRACTION_SYSTEM_PROMPT = """Extract domain glossary terms from a code feature.

Extract terms for:
- Actors: entities that perform actions (user, admin, client, customer)
- Objects: domain entities (order, invoice, product, account)
- Processes: business operations (registration, authentication, checkout)
- States: conditions or statuses (pending, active, expired)

Rules:
- Extract 2-5 terms per feature
- Use lowercase, singular form (1-2 words)
- Only domain/business terms, NOT programming terms (function, class, handler, service)
- NOT technical terms (cache, queue, thread, buffer)

Return JSON array of term names only:
["term1", "term2", "term3"]

Return [] if no domain terms found."""

GLOSSARY_EXTRACTION_USER_PROMPT = """Feature: {label}
Description: {summary}"""

# Legacy single-prompt template (kept for backwards compatibility)
GLOSSARY_EXTRACTION_PROMPT = """Extract domain glossary terms from this code feature.

Feature: {label}
Description: {summary}

Extract terms for:
- Actors: entities that perform actions (user, admin, client, customer)
- Objects: domain entities (order, invoice, product, account)
- Processes: business operations (registration, authentication, checkout)
- States: conditions or statuses (pending, active, expired)

Rules:
- Extract 2-5 terms per feature
- Use lowercase, singular form (1-2 words)
- Only domain/business terms, NOT programming terms (function, class, handler, service)
- NOT technical terms (cache, queue, thread, buffer)

Return JSON array of term names only:
["term1", "term2", "term3"]

Return [] if no domain terms found.

JSON:"""

# Phase 2: Generate holistic summary
GLOSSARY_SUMMARY_SYSTEM_PROMPT = """You are generating a glossary entry for a domain term found in a codebase.

Write a glossary entry with:

1. **Definition** (1 sentence): A clear, concise definition of what the term represents in this codebase.

2. **Details** (2-4 sentences): Synthesize a holistic description of the term's role and meaning in the system. Do NOT describe each feature separately - instead, write about the term itself: what it represents, its purpose, and its significance in the domain.

IMPORTANT: Write about the TERM, not about the features. The features are context to help you understand the term, but your entry should read like a dictionary/glossary definition.

Focus on the business/domain meaning, not technical implementation details.
Use markdown formatting (bold for emphasis, bullet points if listing multiple aspects)."""

GLOSSARY_SUMMARY_USER_PROMPT = """Term: {term}

This term appears in the following features/capabilities of the codebase:

{features_context}"""

# Legacy single-prompt template (kept for backwards compatibility)
GLOSSARY_SUMMARY_PROMPT = """You are generating a glossary entry for a domain term found in a codebase.

Term: {term}

This term appears in the following features/capabilities of the codebase:

{features_context}

Write a glossary entry with:

1. **Definition** (1 sentence): A clear, concise definition of what "{term}" represents in this codebase.

2. **Details** (2-4 sentences): Synthesize a holistic description of the term's role and meaning in the system. Do NOT describe each feature separately - instead, write about the term itself: what it represents, its purpose, and its significance in the domain.

IMPORTANT: Write about the TERM, not about the features. The features are context to help you understand the term, but your entry should read like a dictionary/glossary definition.

Focus on the business/domain meaning, not technical implementation details.
Use markdown formatting (bold for emphasis, bullet points if listing multiple aspects).

Entry:"""


@dataclass
class GlossaryItem:
    """A glossary item extracted from a feature."""

    name: str
    description: str
    source_feature_id: str
    source_feature_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.source_feature_id and self.source_feature_id not in self.source_feature_ids:
            self.source_feature_ids = [self.source_feature_id]


def build_glossary_prompt(summary: str, label: str) -> str:
    """Build the prompt for glossary extraction (legacy single-prompt format).

    Args:
        summary: The feature summary text to extract terms from.
        label: The feature label for context.

    Returns:
        Formatted prompt string for the LLM.
    """
    return GLOSSARY_EXTRACTION_PROMPT.format(label=label, summary=summary)


def build_glossary_messages(summary: str, label: str) -> list[dict[str, str]]:
    """Build messages for glossary extraction (optimized for prefix caching).

    Args:
        summary: The feature summary text to extract terms from.
        label: The feature label for context.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    user_content = GLOSSARY_EXTRACTION_USER_PROMPT.format(label=label, summary=summary)
    return [
        {"role": "system", "content": GLOSSARY_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_llm_response(response: str) -> list[str]:
    """Parse LLM response to extract glossary term names.

    Handles:
    - Plain JSON arrays of strings ["term1", "term2"]
    - JSON wrapped in markdown code blocks (```json ... ```)
    - Legacy format with objects [{"name": "term1", ...}]

    Args:
        response: Raw response string from the LLM.

    Returns:
        List of term name strings.
        Returns empty list if parsing fails or no valid items found.
    """
    response = response.strip()

    # Handle markdown code blocks
    if "```" in response:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if match:
            response = match.group(1).strip()

    try:
        data = json.loads(response)
        if isinstance(data, list):
            terms = []
            for item in data:
                if isinstance(item, str):
                    # New format: ["term1", "term2"]
                    terms.append(item)
                elif isinstance(item, dict) and "name" in item:
                    # Legacy format: [{"name": "term1", ...}]
                    terms.append(item["name"])
            return terms
    except json.JSONDecodeError:
        pass

    return []


async def call_llm_for_glossary(
    summary: str,
    label: str,
    config: MagaldiConfig | None = None,
) -> list[str]:
    """Call LLM to extract glossary term names from a summary.

    Uses message-based format (system + user) optimized for Ollama's KV cache
    prefix caching. The system message contains static instructions that get
    cached, while the user message has variable content.

    Args:
        summary: The feature summary text to extract terms from.
        label: The feature label for context.
        config: Optional MagaldiConfig. If None, uses default config.

    Returns:
        List of term name strings.
        Returns empty list if LLM call fails or returns invalid data.
    """
    if config is None:
        from shared.config import MagaldiConfig

        config = MagaldiConfig()

    # Build messages optimized for prefix caching
    messages = build_glossary_messages(summary, label)

    # Compute dynamic context size based on prompt length
    from shared.ai.context_size import compute_aggregation_num_ctx
    user_content = GLOSSARY_EXTRACTION_USER_PROMPT.format(label=label, summary=summary)
    prompt_chars = len(GLOSSARY_EXTRACTION_SYSTEM_PROMPT) + len(user_content)
    num_ctx = compute_aggregation_num_ctx(prompt_chars, task_type="glossary_extract")

    # Get model config from LLM config
    model_config = config.llm.get_summarize_model()
    client = LLMClient(
        model=model_config.get_litellm_model(),
        api_base=model_config.get_api_base(),
        api_key=model_config.api_key,
    )

    try:
        response = client.generate_from_messages(
            messages=messages,
            temperature=config.llm.summarize_temperature,
            top_p=config.llm.summarize_top_p,
            max_tokens=128,  # Just term names, not full descriptions
            num_ctx=num_ctx,
        )
    except LLMError:
        return []

    return parse_llm_response(response)


async def extract_glossary_from_feature(
    feature: dict[str, Any],
    config: MagaldiConfig | None = None,
) -> list[GlossaryItem]:
    """Extract glossary items from a single feature.

    Args:
        feature: Feature dict with feature_id, label, summary.
        config: Optional MagaldiConfig. If None, uses default config.

    Returns:
        List of GlossaryItem extracted from the feature.
    """
    feature_id = feature.get("feature_id") or feature.get("subfeature_id", "")
    label = feature.get("label", "")
    summary = feature.get("summary", "")

    if not summary:
        return []

    term_names = await call_llm_for_glossary(summary, label, config)

    items = []
    for name in term_names:
        name = name.lower().strip()
        if name:
            items.append(
                GlossaryItem(
                    name=name,
                    description="",  # Will be generated in Phase 2
                    source_feature_id=feature_id,
                )
            )

    return items


def normalize_term(name: str) -> str:
    """Normalize a glossary term name.

    Handles pluralization and common variations.

    Args:
        name: The term name to normalize.

    Returns:
        Normalized term name (lowercase, singular form).
    """
    name = name.lower().strip()

    # Simple depluralization for common patterns
    if name.endswith("ies") and len(name) > 3:
        singular = name[:-3] + "y"
        if len(singular) > 2:
            name = singular
    elif name.endswith("sses"):
        # classes -> class
        name = name[:-2]
    elif name.endswith("xes") or name.endswith("ches") or name.endswith("shes"):
        # boxes -> box, matches -> match, flashes -> flash
        name = name[:-2]
    elif name.endswith("es") and len(name) > 3 and not name.endswith("sses"):
        # processes -> process (but not classes which is handled above)
        if not name[:-2].endswith("s"):
            name = name[:-1]  # processes -> processe -> process (actually need -2)
            # Re-check: processes -> process requires removing 'es'
            # But "ses" ending needs special handling
            pass
    elif name.endswith("s") and len(name) > 3 and not name.endswith("ss"):
        name = name[:-1]

    return name


def merge_glossary_items(items: list[GlossaryItem]) -> list[GlossaryItem]:
    """Merge glossary items with same/similar names.

    Items with the same normalized name are merged:
    - Feature IDs are combined
    - The longest description is kept

    Args:
        items: List of GlossaryItem to merge.

    Returns:
        List of merged GlossaryItem, sorted by name.
    """
    grouped: dict[str, list[GlossaryItem]] = {}

    for item in items:
        normalized = normalize_term(item.name)
        if normalized not in grouped:
            grouped[normalized] = []
        grouped[normalized].append(item)

    merged = []
    for normalized_name, group in grouped.items():
        all_feature_ids: list[str] = []
        for item in group:
            for fid in item.source_feature_ids:
                if fid not in all_feature_ids:
                    all_feature_ids.append(fid)

        best_description = max(
            (item.description for item in group),
            key=len,
        )

        merged.append(
            GlossaryItem(
                name=normalized_name,
                description=best_description,
                source_feature_id=all_feature_ids[0] if all_feature_ids else "",
                source_feature_ids=all_feature_ids,
            )
        )

    merged.sort(key=lambda x: x.name)
    return merged


def build_features_context(
    feature_ids: list[str],
    features_by_id: dict[str, dict[str, Any]],
) -> str:
    """Build context string for summary generation from connected features.

    Args:
        feature_ids: List of feature IDs this term appears in.
        features_by_id: Dict mapping feature_id to feature data.

    Returns:
        Formatted string with feature labels and summaries.
    """
    lines = []
    for fid in feature_ids:
        feature = features_by_id.get(fid, {})
        label = feature.get("label", "Unknown")
        summary = feature.get("summary", "")
        if summary:
            lines.append(f"- {label}: {summary}")
        else:
            lines.append(f"- {label}")
    return "\n".join(lines) if lines else "No feature context available."


def generate_glossary_summary_sync(
    term: str,
    feature_ids: list[str],
    features_by_id: dict[str, dict[str, Any]],
    config: MagaldiConfig | None = None,
) -> tuple[str, float]:
    """Generate a holistic summary for a glossary term based on all its connected features.

    Uses message-based format (system + user) optimized for Ollama's KV cache
    prefix caching. The system message contains static instructions that get
    cached, while the user message has variable content.

    Args:
        term: The glossary term name.
        feature_ids: List of feature IDs where this term appears.
        features_by_id: Dict mapping feature_id to feature data.
        config: Optional MagaldiConfig.

    Returns:
        Tuple of (summary string, api_time in seconds).
    """
    if config is None:
        from shared.config import MagaldiConfig
        config = MagaldiConfig()

    features_context = build_features_context(feature_ids, features_by_id)

    # Build messages optimized for prefix caching
    user_content = GLOSSARY_SUMMARY_USER_PROMPT.format(
        term=term,
        features_context=features_context,
    )
    messages = [
        {"role": "system", "content": GLOSSARY_SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Compute dynamic context size based on prompt length
    from shared.ai.context_size import compute_aggregation_num_ctx
    prompt_chars = len(GLOSSARY_SUMMARY_SYSTEM_PROMPT) + len(user_content)
    num_ctx = compute_aggregation_num_ctx(prompt_chars, task_type="glossary_summary")

    # Get model config from LLM config
    model_config = config.llm.get_summarize_model()
    client = LLMClient(
        model=model_config.get_litellm_model(),
        api_base=model_config.get_api_base(),
        api_key=model_config.api_key,
    )

    start_time = time.time()
    try:
        response = client.generate_from_messages(
            messages=messages,
            temperature=config.llm.summarize_temperature,
            top_p=config.llm.summarize_top_p,
            max_tokens=512,  # Allow for definition + details
            num_ctx=num_ctx,
        )
        api_time = time.time() - start_time
        return response.strip(), api_time
    except LLMError:
        api_time = time.time() - start_time
        return "", api_time


def extract_glossary_from_feature_sync(
    feature: dict[str, Any],
    config: MagaldiConfig | None = None,
) -> tuple[list[GlossaryItem], float]:
    """Synchronous version of extract_glossary_from_feature for thread pool.

    Args:
        feature: Feature dict with feature_id, label, summary.
        config: Optional MagaldiConfig. If None, uses default config.

    Returns:
        Tuple of (list of GlossaryItem, api_time in seconds).
    """
    feature_id = feature.get("feature_id") or feature.get("subfeature_id", "")
    label = feature.get("label", "")
    summary = feature.get("summary", "")

    if not summary:
        return [], 0.0

    start_time = time.time()
    term_names = call_llm_for_glossary_sync(summary, label, config)
    api_time = time.time() - start_time

    items = []
    for name in term_names:
        name = name.lower().strip()
        if name:
            items.append(
                GlossaryItem(
                    name=name,
                    description="",  # Will be generated in Phase 2
                    source_feature_id=feature_id,
                )
            )

    return items, api_time


def call_llm_for_glossary_sync(
    summary: str,
    label: str,
    config: MagaldiConfig | None = None,
) -> list[str]:
    """Synchronous LLM call for glossary term extraction.

    Uses message-based format (system + user) optimized for Ollama's KV cache
    prefix caching. The system message contains static instructions that get
    cached, while the user message has variable content.

    Args:
        summary: The feature summary text to extract terms from.
        label: The feature label for context.
        config: Optional MagaldiConfig. If None, uses default config.

    Returns:
        List of term name strings.
    """
    if config is None:
        from shared.config import MagaldiConfig

        config = MagaldiConfig()

    # Build messages optimized for prefix caching
    messages = build_glossary_messages(summary, label)

    # Compute dynamic context size based on prompt length
    from shared.ai.context_size import compute_aggregation_num_ctx
    user_content = GLOSSARY_EXTRACTION_USER_PROMPT.format(label=label, summary=summary)
    prompt_chars = len(GLOSSARY_EXTRACTION_SYSTEM_PROMPT) + len(user_content)
    num_ctx = compute_aggregation_num_ctx(prompt_chars, task_type="glossary_extract")

    # Get model config from LLM config
    model_config = config.llm.get_summarize_model()
    client = LLMClient(
        model=model_config.get_litellm_model(),
        api_base=model_config.get_api_base(),
        api_key=model_config.api_key,
    )

    try:
        response = client.generate_from_messages(
            messages=messages,
            temperature=config.llm.summarize_temperature,
            top_p=config.llm.summarize_top_p,
            max_tokens=128,  # Just term names, not full descriptions
            num_ctx=num_ctx,
        )
    except LLMError:
        return []

    return parse_llm_response(response)


def extract_glossary_from_features_concurrent(
    features: list[dict[str, Any]],
    config: MagaldiConfig | None = None,
    num_workers: int = 4,
    on_progress: Callable[[GlossaryProgressState], None] | None = None,
    on_status_change: Callable[[], None] | None = None,
    worker_status: GlossaryWorkerStatus | None = None,
    timing_stats: GlossaryTimingStats | None = None,
    on_phase_change: Callable[[str], None] | None = None,
) -> list[GlossaryItem]:
    """Extract and merge glossary items from multiple features using concurrent workers.

    Two-phase process:
    1. Extract term names from each feature (concurrent)
    2. Generate holistic summaries for each merged term (concurrent)

    Args:
        features: List of feature/subfeature dicts with feature_id, label, summary.
        config: Optional config for LLM client.
        num_workers: Number of concurrent workers.
        on_progress: Callback for progress updates.
        on_status_change: Callback when worker status changes.
        worker_status: Shared worker status tracker.
        timing_stats: Shared timing statistics.
        on_phase_change: Callback when phase changes (phase name).

    Returns:
        List of merged GlossaryItem with generated summaries.
    """
    if not features:
        return []

    if worker_status is None:
        worker_status = GlossaryWorkerStatus()
    if timing_stats is None:
        timing_stats = GlossaryTimingStats()

    timing_stats.start_time = time.time()
    all_items: list[GlossaryItem] = []
    items_lock = threading.Lock()
    progress_lock = threading.Lock()
    counters = {"completed": 0, "failed": 0, "terms_extracted": 0}

    # Get model name for display
    if config is None:
        from shared.config import MagaldiConfig
        config = MagaldiConfig()
    model_name = config.llm.get_summarize_model().name

    # Build features lookup for summary generation
    features_by_id: dict[str, dict[str, Any]] = {}
    for feature in features:
        fid = feature.get("feature_id") or feature.get("subfeature_id", "")
        if fid:
            features_by_id[fid] = feature

    # =========================================================================
    # PHASE 1: Extract term names from features
    # =========================================================================
    if on_phase_change:
        on_phase_change("Extracting terms from features")

    def process_feature(
        worker_id: int,
        feature: dict[str, Any],
    ) -> tuple[list[GlossaryItem], float, bool]:
        """Process a single feature in a worker thread."""
        label = feature.get("label", "")[:40]

        worker_status.set_status(worker_id, label, model_name)
        if on_status_change:
            on_status_change()

        try:
            items, api_time = extract_glossary_from_feature_sync(feature, config)
            return items, api_time, True
        except Exception:
            return [], 0.0, False
        finally:
            worker_status.clear_status(worker_id)
            if on_status_change:
                on_status_change()

    total = len(features)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for i, feature in enumerate(features):
            worker_id = i % num_workers
            future = executor.submit(process_feature, worker_id, feature)
            futures[future] = feature

        for future in as_completed(futures):
            items, api_time, success = future.result()

            with progress_lock:
                counters["completed"] += 1
                if not success:
                    counters["failed"] += 1
                else:
                    counters["terms_extracted"] += len(items)
                    timing_stats.record_api_call(api_time)

            with items_lock:
                all_items.extend(items)

            if on_progress:
                state = GlossaryProgressState(
                    total=total,
                    completed=counters["completed"],
                    failed=counters["failed"],
                    terms_extracted=counters["terms_extracted"],
                    timing=timing_stats,
                    workers=worker_status,
                    num_workers=num_workers,
                )
                on_progress(state)

    # Merge items by term name
    merged_items = merge_glossary_items(all_items)

    # Emit final Phase 1 progress state before switching phases
    if on_progress:
        state = GlossaryProgressState(
            total=total,
            completed=counters["completed"],
            failed=counters["failed"],
            terms_extracted=counters["terms_extracted"],
            timing=timing_stats,
            workers=worker_status,
            num_workers=num_workers,
        )
        on_progress(state)

    # =========================================================================
    # PHASE 2: Generate holistic summaries for each merged term
    # =========================================================================
    if on_phase_change:
        on_phase_change("Generating summaries for terms")

    # Reset progress counters and timing for phase 2
    counters["completed"] = 0
    counters["failed"] = 0
    timing_stats.start_time = time.time()
    timing_stats.total_api_time = 0.0
    timing_stats.features_processed = 0
    total_terms = len(merged_items)

    def generate_summary_for_term(
        worker_id: int,
        item: GlossaryItem,
    ) -> tuple[GlossaryItem, float, bool]:
        """Generate summary for a merged glossary term."""
        worker_status.set_status(worker_id, item.name, model_name)
        if on_status_change:
            on_status_change()

        try:
            summary, api_time = generate_glossary_summary_sync(
                term=item.name,
                feature_ids=item.source_feature_ids,
                features_by_id=features_by_id,
                config=config,
            )
            if summary:
                item.description = summary
            return item, api_time, True
        except Exception:
            return item, 0.0, False
        finally:
            worker_status.clear_status(worker_id)
            if on_status_change:
                on_status_change()

    final_items: list[GlossaryItem] = []
    final_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for i, item in enumerate(merged_items):
            worker_id = i % num_workers
            future = executor.submit(generate_summary_for_term, worker_id, item)
            futures[future] = item

        for future in as_completed(futures):
            item, api_time, success = future.result()

            with progress_lock:
                counters["completed"] += 1
                if not success:
                    counters["failed"] += 1
                else:
                    timing_stats.record_api_call(api_time)

            with final_lock:
                final_items.append(item)

            if on_progress:
                state = GlossaryProgressState(
                    total=total_terms,
                    completed=counters["completed"],
                    failed=counters["failed"],
                    terms_extracted=len(final_items),
                    timing=timing_stats,
                    workers=worker_status,
                    num_workers=num_workers,
                )
                on_progress(state)

    final_items.sort(key=lambda x: x.name)
    return final_items


async def extract_glossary_from_features(
    features: list[dict[str, Any]],
    config: MagaldiConfig | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[GlossaryItem]:
    """Extract and merge glossary items from multiple features.

    Processes each feature through the LLM, then merges duplicates.
    This is a simple sequential version - use extract_glossary_from_features_concurrent
    for multi-threaded execution with progress display.

    Args:
        features: List of feature/subfeature dicts with feature_id, label, summary.
        config: Optional config for LLM client.
        progress_callback: Optional callback(current, total) for progress updates.

    Returns:
        List of merged GlossaryItem.
    """
    if not features:
        return []

    all_items: list[GlossaryItem] = []
    total = len(features)

    for i, feature in enumerate(features):
        if progress_callback:
            progress_callback(i + 1, total)

        items = await extract_glossary_from_feature(feature, config)
        all_items.extend(items)

    return merge_glossary_items(all_items)
