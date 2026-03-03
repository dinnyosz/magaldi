"""AI-powered glossary extraction from feature summaries."""

from __future__ import annotations

import bisect
import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from shared.ai.context_size import TIER_SCALING_EXPONENT
from shared.ai.llm_client import LLMClient, LLMError
from shared.parallel_processor import ThrottleContext, ThrottleDisplayInfo, run_throttled_tier
from shared.throttling import ThroughputTracker

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
    throughput_tracker: ThroughputTracker = field(default_factory=lambda: ThroughputTracker(window_seconds=300.0))

    # Per-tier tracking for accurate ETA (matches Feature/SubfeatureTimingStats)
    total_time_by_tier: dict[int, float] = field(default_factory=dict)
    count_by_tier: dict[int, int] = field(default_factory=dict)
    totals_by_tier: dict[int, int] = field(default_factory=dict)

    def set_totals_by_tier(self, totals: dict[int, int]) -> None:
        """Set total counts by tier for ETA calculation."""
        with self._lock:
            self.totals_by_tier = dict(totals)

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

    def record_api_call(self, api_time: float, tier: int = 0) -> None:
        """Record an API call timing."""
        with self._lock:
            self.total_api_time += api_time
            self.features_processed += 1

            # Track per-tier timing
            if tier > 0:
                self.total_time_by_tier[tier] = self.total_time_by_tier.get(tier, 0.0) + api_time
                self.count_by_tier[tier] = self.count_by_tier.get(tier, 0) + 1

    def record_task_runtime(self, runtime: float, concurrent_workers: float = 1.0) -> None:
        """Record task wall-clock runtime for throttling."""
        self.throughput_tracker.record_completion(runtime, concurrent_workers)

    def get_throughput_stats_with_concurrency(self) -> tuple[float, float, int, float, float]:
        """Get throughput statistics with concurrency context."""
        return self.throughput_tracker.get_stats_with_concurrency()  # type: ignore[no-any-return]

    def _get_avg_for_tier(self, tier: int, global_avg: float) -> float:
        """Get average time for a tier with fallback."""
        # Exact tier match
        if tier in self.count_by_tier and self.count_by_tier[tier] > 0:
            return self.total_time_by_tier[tier] / self.count_by_tier[tier]

        # Find closest tier with data
        tiers_with_data = [t for t in self.count_by_tier if self.count_by_tier[t] > 0]
        if tiers_with_data:
            closest = min(tiers_with_data, key=lambda t: abs(t - tier))
            base_avg = self.total_time_by_tier[closest] / self.count_by_tier[closest]
            # Scale by tier ratio - use empirically-derived exponent for sub-linear scaling
            tier_ratio = (tier / closest) ** TIER_SCALING_EXPONENT if closest > 0 else 1.0
            return base_avg * tier_ratio

        return global_avg

    def eta_seconds(self, completed: int, total: int, _num_workers: int) -> float:
        """Estimate time remaining based on current progress."""
        if completed == 0 or self.elapsed == 0:
            return 0.0
        rate = completed / self.elapsed
        remaining = total - completed
        return remaining / rate if rate > 0 else 0.0

    def get_eta_breakdown_with_avg(self, _num_workers: int = 1) -> list[tuple[str, int, float, bool, int, int]]:
        """Get average time per tier for display.

        Returns:
            List of (type, tier, avg_seconds, is_fallback, done, total) tuples,
            matching the format used by processor.py TimingStats.
            Type is always "glossary" for this class.
        """
        with self._lock:
            if not self.totals_by_tier:
                return []

            global_avg = self.total_api_time / self.features_processed if self.features_processed > 0 else 0.0

            breakdown = []
            for tier, total in self.totals_by_tier.items():
                done = self.count_by_tier.get(tier, 0)
                avg = self._get_avg_for_tier(tier, global_avg)
                # is_fallback: True if we don't have actual data for this tier
                is_fallback = tier not in self.count_by_tier or self.count_by_tier[tier] == 0
                breakdown.append(("glossary", tier, avg, is_fallback, done, total))

            # Sort by tier descending (largest first)
            breakdown.sort(key=lambda x: -x[1])
            return breakdown


class GlossaryWorkerStatus:
    """Thread-safe tracking of worker status."""

    def __init__(self) -> None:
        self._status: dict[int, tuple[str, str, str, float]] = {}  # worker_id -> (feature_label, model, ctx_size, start_time)
        self._lock = threading.Lock()

    def set_status(self, worker_id: int, feature_label: str, model: str, ctx_size: str = "") -> None:
        """Set worker status."""
        with self._lock:
            self._status[worker_id] = (feature_label, model, ctx_size, time.time())

    def clear_status(self, worker_id: int) -> None:
        """Clear worker status (worker is idle)."""
        with self._lock:
            if worker_id in self._status:
                del self._status[worker_id]

    def get_all(self) -> dict[int, tuple[str, str, str]]:
        """Get all worker statuses (without start_time for display)."""
        with self._lock:
            return {k: (v[0], v[1], v[2]) for k, v in self._status.items()}

    def active_count(self) -> int:
        """Get number of active workers."""
        with self._lock:
            return len(self._status)

    def get_max_active_runtime(self) -> float:
        """Get maximum runtime of currently active workers."""
        with self._lock:
            if not self._status:
                return 0.0
            now = time.time()
            return max(now - v[3] for v in self._status.values())


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
    allowed_workers: int = 0  # Current throttle-allowed workers (0 = use num_workers)
    current_max: float = 0.0  # Max runtime of active workers (for throttle display)
    avg_base_time: float = 0.0  # Historical base time per worker
    completion_count: int = 0  # Number of completions used for avg_base_time
    peak_concurrency: int | None = None  # Concurrency level with peak throughput
    all_levels: dict[int, tuple[float, int]] | None = None  # Per-level throughput data
    exploration_status: str | None = None  # Lifecycle status for constant feedback
    gss_probe: int | None = None  # GSS probe target (if active)
    gss_lo: int | None = None  # GSS bracket lower bound
    gss_hi: int | None = None  # GSS bracket upper bound
    gss_signal: str | None = None  # Signal-aware action
    exploration_target: int | None = None  # Level being explored
    prob_map_data: dict[int, float] | None = None  # Probability map data
    explore_cap: int | None = None  # Effective base_workers after budget cap


# =============================================================================
# GLOSSARY EXTRACTION PROMPTS (Optimized for Prefix Caching)
# =============================================================================
# System messages are STATIC and get cached by Ollama's KV cache.
# User messages contain VARIABLE content (feature info, term context).

# Phase 1: Extract term names
GLOSSARY_EXTRACTION_SYSTEM_PROMPT = """Extract DOMAIN-SPECIFIC glossary terms from a code feature.

ONLY extract terms that are:
- Specific to THIS codebase's domain (e.g., "call chain", "context tier", "dead code analysis")
- Compound terms or phrases with clear meaning (e.g., "entry point", "feature clustering")
- Named concepts unique to the system (e.g., "element extraction", "tiered model")

NEVER extract:
- Generic programming verbs: get, set, add, delete, update, check, clear, create, run, execute, process, handle, manage, load, save, fetch, store, retrieve, convert, parse, validate, render, format, filter, sort, merge, search, find, match, compare, count, list, read, write, send, receive, start, stop, reset, init, close, open, call, return, throw, catch
- Generic nouns: data, element, result, value, error, item, object, entity, record, entry, node, instance, type, kind, state, status, action, event, task, job, request, response, input, output, parameter, argument, option, setting, config, property, attribute, field, key, id, name, label, path, file, line, index, offset, size, count, length, number, string, array, list, map, set, dict, queue, stack, buffer, cache, pool, batch, chunk, block, group, collection, container
- Single-word terms that could appear in ANY codebase
- Test-related terms: mock, stub, fixture, assert, test case
- Implementation details: function, class, method, variable, loop, thread, callback

Good examples: "call hierarchy", "dead code detection", "context window optimization", "feature embedding"
Bad examples: "get", "process", "data", "element", "handle error", "load config"

Rules:
- Extract 0-4 meaningful domain terms per feature
- Use lowercase, 2-3 word phrases preferred
- If no domain-specific terms exist, return []

Return JSON array: ["term1", "term2"] or []"""

GLOSSARY_EXTRACTION_USER_PROMPT = """Feature: {label}
Description: {summary}"""

# Legacy single-prompt template (kept for backwards compatibility)
GLOSSARY_EXTRACTION_PROMPT = """Extract DOMAIN-SPECIFIC glossary terms from this code feature.

Feature: {label}
Description: {summary}

ONLY extract terms that are:
- Specific to THIS codebase's domain (e.g., "call chain", "context tier", "dead code analysis")
- Compound terms or phrases with clear meaning (e.g., "entry point", "feature clustering")
- Named concepts unique to the system (e.g., "element extraction", "tiered model")

NEVER extract generic programming terms like: get, set, add, delete, process, handle, data, element, result, error, config, state, action, task, job, request, response, etc.

Rules:
- Extract 0-4 meaningful domain terms per feature
- Use lowercase, 2-3 word phrases preferred
- If no domain-specific terms exist, return []

Return JSON array: ["term1", "term2"] or []

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


# Blocklist of generic terms that should never appear in the glossary
# These are common programming terms that don't add domain-specific value
GENERIC_TERM_BLOCKLIST = frozenset([
    # Generic verbs
    "get", "set", "add", "delete", "update", "check", "clear", "create", "run",
    "execute", "process", "handle", "manage", "load", "save", "fetch", "store",
    "retrieve", "convert", "parse", "validate", "render", "format", "filter",
    "sort", "merge", "search", "find", "match", "compare", "count", "list",
    "read", "write", "send", "receive", "start", "stop", "reset", "init",
    "close", "open", "call", "return", "throw", "catch", "build", "make",
    "remove", "insert", "append", "pop", "push", "pull", "extract", "detect",
    "analyze", "generate", "compute", "calculate", "evaluate", "acquire",
    "release", "register", "complete", "fail", "succeed", "finish", "begin",
    # Generic nouns
    "data", "element", "result", "value", "error", "item", "object", "entity",
    "record", "entry", "node", "instance", "type", "kind", "state", "status",
    "action", "event", "task", "job", "request", "response", "input", "output",
    "parameter", "argument", "option", "setting", "config", "configuration",
    "property", "attribute", "field", "key", "id", "name", "label", "path",
    "file", "line", "index", "offset", "size", "count", "length", "number",
    "string", "array", "list", "map", "set", "dict", "queue", "stack", "buffer",
    "cache", "pool", "batch", "chunk", "block", "group", "collection", "container",
    "client", "server", "handler", "manager", "factory", "builder", "helper",
    "util", "utility", "service", "provider", "consumer", "producer", "worker",
    "context", "scope", "session", "connection", "transaction", "operation",
    "model", "schema", "format", "pattern", "rule", "policy", "strategy",
    "method", "function", "class", "module", "package", "interface", "protocol",
    # States
    "pending", "running", "completed", "failed", "active", "inactive", "ready",
    "waiting", "done", "success", "failure", "valid", "invalid", "enabled",
    "disabled", "available", "unavailable",
    # Test-related
    "test", "mock", "stub", "fixture", "assert", "assertion", "expect",
    # Other common terms
    "info", "debug", "warning", "message", "content", "body", "header",
    "metadata", "payload", "token", "hash", "digest", "signature", "version",
    "timestamp", "duration", "timeout", "interval", "limit", "threshold",
    "default", "fallback", "override", "extension", "plugin", "hook", "callback",
    "listener", "observer", "subscriber", "publisher", "emitter", "dispatcher",
])


def _is_valid_glossary_term(term: str) -> bool:
    """Check if a term is valid for the glossary (not in blocklist).

    Args:
        term: The term to check.

    Returns:
        True if the term should be included, False if it should be filtered out.
    """
    term_lower = term.lower().strip()

    # Reject empty or very short terms
    if len(term_lower) < 3:
        return False

    # Reject single-word terms that are in the blocklist
    if term_lower in GENERIC_TERM_BLOCKLIST:
        return False

    # For multi-word terms, reject if ALL words are generic
    words = term_lower.split()
    if len(words) > 1:
        non_generic_words = [w for w in words if w not in GENERIC_TERM_BLOCKLIST]
        if not non_generic_words:
            return False

    return True


def parse_llm_response(response: str) -> list[str]:
    """Parse LLM response to extract glossary term names.

    Handles:
    - Plain JSON arrays of strings ["term1", "term2"]
    - JSON wrapped in markdown code blocks (```json ... ```)
    - Legacy format with objects [{"name": "term1", ...}]

    Also filters out generic programming terms using a blocklist.

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
                    if _is_valid_glossary_term(item):
                        terms.append(item)
                elif isinstance(item, dict) and "name" in item and _is_valid_glossary_term(item["name"]):
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
    client = LLMClient.from_model_config(model_config)

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

    # Words ending in "us" are Latin origin (status, bonus, cactus) - don't depluralize
    if name.endswith("us"):
        return name

    # Words ending in "is" are also Latin origin (analysis, basis) - don't depluralize
    if name.endswith("is"):
        return name

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
    elif name.endswith("ses") and len(name) > 4:
        # processes -> process, analyses stays as "analyses" (handled by "is" check above for singular)
        name = name[:-2]
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
    client = LLMClient.from_model_config(model_config)

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
    client = LLMClient.from_model_config(model_config)

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
    # Incremental indexing parameters
    repo: Any | None = None,
    scope: str | None = None,
    repository: str | None = None,
    username: str | None = None,
    on_indexed: Callable[[str], None] | None = None,
) -> list[GlossaryItem]:
    """Extract and merge glossary items from multiple features using concurrent workers.

    Two-phase process:
    1. Extract term names from each feature (concurrent)
    2. Generate holistic summaries for each merged term (concurrent, with optional incremental indexing)

    Args:
        features: List of feature/subfeature dicts with feature_id, label, summary.
        config: Optional config for LLM client.
        num_workers: Number of concurrent workers.
        on_progress: Callback for progress updates.
        on_status_change: Callback when worker status changes.
        worker_status: Shared worker status tracker.
        timing_stats: Shared timing statistics.
        on_phase_change: Callback when phase changes (phase name).
        repo: Optional Search repository for incremental indexing.
        scope: Scope for indexing (required if repo provided).
        repository: Repository name for indexing (required if repo provided).
        username: Username for indexing (required if repo provided).
        on_indexed: Optional callback when an item is indexed (receives term name).

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

    # Get model config for display
    if config is None:
        from shared.config import MagaldiConfig
        config = MagaldiConfig()
    model_config = config.llm.get_summarize_model()

    def get_tiered_model_display(num_ctx: int) -> str:
        """Get tiered model name for Ollama models."""
        if model_config.provider == "ollama":
            from shared.ai.ollama_models import get_tiered_model_name
            return get_tiered_model_name(model_config.name, num_ctx)  # type: ignore[no-any-return]
        return model_config.name  # type: ignore[no-any-return]

    def format_ctx(num_ctx: int) -> str:
        """Format context size for display."""
        return f"{num_ctx // 1024}K" if num_ctx >= 1024 else str(num_ctx)

    # Worker ID pool - reuse IDs 0 to num_workers-1
    available_worker_ids: list[int] = list(range(num_workers))
    worker_id_lock = threading.Lock()

    def acquire_worker_id() -> int:
        """Get an available worker ID from the pool."""
        with worker_id_lock:
            if available_worker_ids:
                return available_worker_ids.pop(0)
            return 0  # Fallback

    def release_worker_id(wid: int) -> None:
        """Return a worker ID to the pool (maintains sorted order)."""
        with worker_id_lock:
            if wid not in available_worker_ids:
                # Insert in sorted position to keep lower IDs at front
                bisect.insort(available_worker_ids, wid)

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
        feature: dict[str, Any],
    ) -> tuple[list[GlossaryItem], float, bool]:
        """Process a single feature in a worker thread."""
        worker_id = acquire_worker_id()
        try:
            label = feature.get("label", "")[:40]
            summary = feature.get("summary", "")

            # Compute context size for display
            from shared.ai.context_size import compute_aggregation_num_ctx
            prompt_chars = len(GLOSSARY_EXTRACTION_SYSTEM_PROMPT) + len(
                GLOSSARY_EXTRACTION_USER_PROMPT.format(label=label, summary=summary)
            )
            num_ctx = compute_aggregation_num_ctx(prompt_chars, task_type="glossary_extract")

            worker_status.set_status(worker_id, label, get_tiered_model_display(num_ctx), format_ctx(num_ctx))
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
        finally:
            release_worker_id(worker_id)

    total = len(features)

    # Pre-compute context tier for each feature
    from shared.ai.context_size import (
        TIER_MAX_WORKERS,
        compute_aggregation_num_ctx,
        get_tier_timeout,
        iter_by_tier,
    )

    def estimate_feature_tier(feature: dict[str, Any]) -> int:
        """Estimate context tier for a feature based on prompt size."""
        label = feature.get("label", "")[:40]
        summary = feature.get("summary", "")
        user_content = GLOSSARY_EXTRACTION_USER_PROMPT.format(label=label, summary=summary)
        prompt_chars = len(GLOSSARY_EXTRACTION_SYSTEM_PROMPT) + len(user_content)
        return compute_aggregation_num_ctx(prompt_chars, task_type="glossary_extract")  # type: ignore[no-any-return]

    # Group features by tier and process each tier with appropriate max_workers
    tier_groups = list(iter_by_tier(features, estimate_feature_tier))
    max_pool_workers = num_workers if num_workers > 0 else max(TIER_MAX_WORKERS.values())

    # Build feature_id -> tier mapping for tracking
    feature_to_tier: dict[str, int] = {}
    tier_counts: dict[int, int] = {}
    for tier, _, tier_features in tier_groups:
        tier_counts[tier] = len(tier_features)
        for feature in tier_features:
            fid = feature.get("feature_id") or feature.get("subfeature_id", "")
            if fid:
                feature_to_tier[fid] = tier

    # Set tier totals for ETA calculation
    timing_stats.set_totals_by_tier(tier_counts)

    # Mutable state for current tier workers
    state = {"current_workers": max_pool_workers}

    def on_complete_phase1(feature: dict, result: tuple, _avg_workers: float, _runtime: float) -> None:
        """Handle completed feature extraction."""
        items, api_time, success = result

        # Look up tier for this feature
        fid = feature.get("feature_id") or feature.get("subfeature_id", "")
        tier = feature_to_tier.get(fid, 0)

        # Throughput is recorded by run_throttled_tier in the throttle context

        with progress_lock:
            counters["completed"] += 1
            if not success:
                counters["failed"] += 1
            else:
                counters["terms_extracted"] += len(items)
                timing_stats.record_api_call(api_time, tier=tier)

        with items_lock:
            all_items.extend(items)

    def on_tick_phase1(throttle_info: ThrottleDisplayInfo) -> None:
        """Update progress display."""
        if on_progress:
            progress_state = GlossaryProgressState(
                total=total,
                completed=counters["completed"],
                failed=counters["failed"],
                terms_extracted=counters["terms_extracted"],
                timing=timing_stats,
                workers=worker_status,
                num_workers=state["current_workers"],
                allowed_workers=throttle_info.allowed_workers,
                current_max=throttle_info.current_max,
                avg_base_time=throttle_info.avg_base_time,
                completion_count=throttle_info.completion_count,
                peak_concurrency=throttle_info.peak_concurrency,
                all_levels=throttle_info.all_levels,
                exploration_status=throttle_info.exploration_status,
                gss_probe=throttle_info.gss_probe,
                gss_lo=throttle_info.gss_lo,
                gss_hi=throttle_info.gss_hi,
                gss_signal=throttle_info.gss_signal,
                exploration_target=throttle_info.exploration_target,
                prob_map_data=throttle_info.prob_map_data,
                explore_cap=throttle_info.explore_cap,
            )
            on_progress(progress_state)

    # Create throttle context
    throttle_ctx = ThrottleContext(
        tier_timeout=get_tier_timeout(2048, max_pool_workers),
        base_workers=max_pool_workers,
        throughput_tracker=timing_stats.throughput_tracker,
    )

    for _tier, _tier_max_workers, tier_features in tier_groups:
        # Use full max workers like Phase 4 - time-based throttling handles scaling
        state["current_workers"] = max_pool_workers

        # Reset worker ID pool for this tier
        with worker_id_lock:
            available_worker_ids.clear()
            available_worker_ids.extend(range(max_pool_workers))

        run_throttled_tier(
            items=list(tier_features),
            tier=_tier,
            effective_workers=max_pool_workers,
            process_fn=process_feature,
            throttle_ctx=throttle_ctx,
            get_max_runtime=worker_status.get_max_active_runtime,
            on_complete=on_complete_phase1,
            on_tick=on_tick_phase1,
        )

    # Merge items by term name
    merged_items = merge_glossary_items(all_items)

    # Emit final Phase 1 progress state before switching phases
    if on_progress:
        progress_state = GlossaryProgressState(
            total=total,
            completed=counters["completed"],
            failed=counters["failed"],
            terms_extracted=counters["terms_extracted"],
            timing=timing_stats,
            workers=worker_status,
            num_workers=num_workers,
        )
        on_progress(progress_state)

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

    # Reset worker ID pool for phase 2
    with worker_id_lock:
        available_worker_ids.clear()
        available_worker_ids.extend(range(num_workers))

    def generate_summary_for_term(
        item: GlossaryItem,
    ) -> tuple[GlossaryItem, float, bool]:
        """Generate summary for a merged glossary term."""
        worker_id = acquire_worker_id()
        try:
            # Compute context size for display
            from shared.ai.context_size import compute_aggregation_num_ctx
            features_context = build_features_context(item.source_feature_ids, features_by_id)
            user_content = GLOSSARY_SUMMARY_USER_PROMPT.format(
                term=item.name,
                features_context=features_context,
            )
            prompt_chars = len(GLOSSARY_SUMMARY_SYSTEM_PROMPT) + len(user_content)
            num_ctx = compute_aggregation_num_ctx(prompt_chars, task_type="glossary_summary")

            worker_status.set_status(worker_id, item.name, get_tiered_model_display(num_ctx), format_ctx(num_ctx))
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
        finally:
            release_worker_id(worker_id)

    final_items: list[GlossaryItem] = []
    final_lock = threading.Lock()

    # Pre-compute context tier for each merged item
    def estimate_term_tier(item: GlossaryItem) -> int:
        """Estimate context tier for a glossary term based on prompt size."""
        features_context = build_features_context(item.source_feature_ids, features_by_id)
        user_content = GLOSSARY_SUMMARY_USER_PROMPT.format(
            term=item.name,
            features_context=features_context,
        )
        prompt_chars = len(GLOSSARY_SUMMARY_SYSTEM_PROMPT) + len(user_content)
        return compute_aggregation_num_ctx(prompt_chars, task_type="glossary_summary")  # type: ignore[no-any-return]

    # Group items by tier and process each tier with appropriate max_workers
    tier_groups = list(iter_by_tier(merged_items, estimate_term_tier))

    # Build term_name -> tier mapping for tracking
    term_to_tier: dict[str, int] = {}
    tier_counts_phase2: dict[int, int] = {}
    for tier, _, tier_items in tier_groups:
        tier_counts_phase2[tier] = len(tier_items)
        for item in tier_items:
            term_to_tier[item.name] = tier

    # Reset tier totals for Phase 2
    timing_stats.total_time_by_tier = {}
    timing_stats.count_by_tier = {}
    timing_stats.set_totals_by_tier(tier_counts_phase2)

    # Mutable state for current tier workers (Phase 2)
    phase2_state = {"current_workers": max_pool_workers}

    def on_complete_phase2(
        item: GlossaryItem, result: tuple, _avg_workers: float, _runtime: float
    ) -> None:
        """Handle completed summary generation."""
        completed_item, api_time, success = result

        # Look up tier for this term
        tier = term_to_tier.get(item.name, 0)

        # Throughput is recorded by run_throttled_tier in the throttle context

        with progress_lock:
            counters["completed"] += 1
            if not success:
                counters["failed"] += 1
            else:
                timing_stats.record_api_call(api_time, tier=tier)

        with final_lock:
            final_items.append(completed_item)

        # Incremental indexing: index each item as it completes
        if success and repo is not None and scope and repository and username:
            glossary_id = f"{scope}:{repository}:{username}:glossary:{completed_item.name}"

            # Build feature associations
            feature_associations = []
            for fid in completed_item.source_feature_ids:
                if fid in features_by_id:
                    feature_data = features_by_id[fid]
                    feature_associations.append({
                        "feature_id": fid,
                        "feature_label": feature_data.get("label", ""),
                        "frequency": 1,
                        "total_members": 0,
                        "percentage": 0.0,
                    })

            repo.index_glossary(
                glossary_id=glossary_id,
                scope=scope,
                repository=repository,
                username=username,
                term=completed_item.name,
                total_count=len(completed_item.source_feature_ids),
                element_ids=completed_item.source_feature_ids,
                file_paths=[],
                description=completed_item.description,
                feature_associations=feature_associations,
            )

            if on_indexed:
                on_indexed(completed_item.name)

    def on_tick_phase2(throttle_info: ThrottleDisplayInfo) -> None:
        """Update progress display for Phase 2."""
        if on_progress:
            progress_state = GlossaryProgressState(
                total=total_terms,
                completed=counters["completed"],
                failed=counters["failed"],
                terms_extracted=len(final_items),
                timing=timing_stats,
                workers=worker_status,
                num_workers=phase2_state["current_workers"],
                allowed_workers=throttle_info.allowed_workers,
                current_max=throttle_info.current_max,
                avg_base_time=throttle_info.avg_base_time,
                completion_count=throttle_info.completion_count,
                peak_concurrency=throttle_info.peak_concurrency,
                all_levels=throttle_info.all_levels,
                exploration_status=throttle_info.exploration_status,
                gss_probe=throttle_info.gss_probe,
                gss_lo=throttle_info.gss_lo,
                gss_hi=throttle_info.gss_hi,
                gss_signal=throttle_info.gss_signal,
                exploration_target=throttle_info.exploration_target,
                prob_map_data=throttle_info.prob_map_data,
                explore_cap=throttle_info.explore_cap,
            )
            on_progress(progress_state)

    # Create throttle context for Phase 2
    # Note: reusing the same throughput_tracker for continuity
    throttle_ctx_phase2 = ThrottleContext(
        tier_timeout=get_tier_timeout(2048, max_pool_workers),
        base_workers=max_pool_workers,
        throughput_tracker=timing_stats.throughput_tracker,
    )

    for _tier, _tier_max_workers, tier_items in tier_groups:
        # Use full max workers like Phase 4 - time-based throttling handles scaling
        phase2_state["current_workers"] = max_pool_workers

        # Reset worker ID pool for this tier
        with worker_id_lock:
            available_worker_ids.clear()
            available_worker_ids.extend(range(max_pool_workers))

        run_throttled_tier(
            items=list(tier_items),
            tier=_tier,
            effective_workers=max_pool_workers,
            process_fn=generate_summary_for_term,
            throttle_ctx=throttle_ctx_phase2,
            get_max_runtime=worker_status.get_max_active_runtime,
            on_complete=on_complete_phase2,
            on_tick=on_tick_phase2,
        )

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
