"""LLM-based variable scoring for code discovery.

This module scores variables using a local LLM to determine whether they are
useful for code discovery. Variables scoring below threshold are dropped before
the summarization/embedding phase.

Replaces the hand-curated heuristic filter (usefulness_filter.py) with a
semantic approach that can handle novel patterns without maintenance.
"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from shared.ai.llm_client import LLMClient

from magaldi_core.variable_scoring.models import (
    ScoringProgressState,
    ScoringResult,
    ScoringWorkerStatus,
    VariableScore,
    VariableScoringConfig,
)
from magaldi_core.variable_scoring.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
)

logger = logging.getLogger(__name__)

# Fixed token overheads for context window budgeting.
# token_budget represents the total context window; these are subtracted
# to get the available space for variable content.
_SYSTEM_PROMPT_TOKENS = len(SYSTEM_PROMPT) // 4  # ~226
_HEADER_TOKENS = 10  # "/no_think\nClassify these variables:"
_OUTPUT_BASE_TOKENS = 30  # Base overhead for LLM response formatting
_OUTPUT_PER_VAR_TOKENS = 10  # ~10 tokens per variable in output (N. KEEP/DROP\n)

# Fixed overhead = system prompt + header + output base
_FIXED_OVERHEAD_TOKENS = _SYSTEM_PROMPT_TOKENS + _HEADER_TOKENS + _OUTPUT_BASE_TOKENS


def _estimate_tokens(raw_code: str, file_path: str) -> int:
    """Estimate token count for a single variable entry in the prompt.

    Uses ~4 chars per token for code + overhead for line number, file path,
    and formatting.
    """
    return len(raw_code) // 4 + len(file_path) // 4 + 20


def _build_batches(
    variables: list[tuple[str, str, str, str]],
    token_budget: int,
) -> list[list[tuple[int, str, str, str, str]]]:
    """Build dynamic batches of variables based on estimated token count.

    Variables are sorted by estimated token size (smallest first) so that
    small variables pack tightly into full batches instead of being stranded
    in tiny batches after a large variable forces a break.

    Args:
        variables: List of (element_id, file_path, name, raw_code) tuples.
        token_budget: Total context window size (system + content + output).
            Fixed overhead for the system prompt, header, and output base
            is subtracted internally. Each variable also reserves
            ~10 tokens for its output line (N. KEEP/DROP).

    Returns:
        List of batches, where each batch is a list of
        (index, element_id, file_path, name, raw_code) tuples.
        Index is 1-based within each batch.
    """
    # Subtract fixed overhead (system prompt + header + output base) from
    # the total budget. What remains is available for variable content +
    # per-variable output tokens.
    available = max(50, token_budget - _FIXED_OVERHEAD_TOKENS)

    # Sort by estimated token size (smallest first) for better packing.
    # Without sorting, a large variable in the middle of small ones forces
    # a batch break, leaving a half-empty batch of small variables.
    sorted_vars = sorted(
        variables, key=lambda v: _estimate_tokens(v[3], v[1])
    )

    batches: list[list[tuple[int, str, str, str, str]]] = []
    current_batch: list[tuple[int, str, str, str, str]] = []
    current_tokens = 0
    batch_idx = 1

    for element_id, file_path, name, raw_code in sorted_vars:
        # Each variable costs its content tokens + output tokens (~10)
        var_tokens = _estimate_tokens(raw_code, file_path) + _OUTPUT_PER_VAR_TOKENS

        if current_tokens + var_tokens > available and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
            batch_idx = 1

        current_batch.append((batch_idx, element_id, file_path, name, raw_code))
        current_tokens += var_tokens
        batch_idx += 1

    if current_batch:
        batches.append(current_batch)

    return batches


def _strip_think_tags(output: str) -> str:
    """Strip <think>...</think> blocks from LLM output.

    Qwen3 and other thinking models may emit reasoning in <think> tags
    even when thinking is disabled (e.g. server doesn't honor
    chat_template_kwargs). The reasoning text can contain numbered patterns
    that confuse the score parser, or consume the token budget so actual
    scores never appear.
    """
    return re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()


def _parse_scores(output: str, batch_size: int) -> list[VariableScore | None]:
    """Parse LLM output into binary keep/drop decisions.

    Expected format:
        1. KEEP
        2. DROP

    Handles various formatting quirks from LLMs (extra whitespace,
    punctuation after number, case variations).

    Args:
        output: Raw LLM output text.
        batch_size: Expected number of scores.

    Returns:
        List of VariableScore (or None for unparseable lines).
    """
    # Strip thinking blocks before parsing (defense against thinking models)
    output = _strip_think_tags(output)

    scores: list[VariableScore | None] = [None] * batch_size
    # Match "N. KEEP" or "N. DROP" (case-insensitive, flexible separators)
    pattern = re.compile(r"(\d+)\s*[.):]\s*(keep|drop)", re.IGNORECASE)

    for match in pattern.finditer(output):
        idx = int(match.group(1))
        if 1 <= idx <= batch_size:
            decision = match.group(2).upper()
            scores[idx - 1] = VariableScore(keep=(decision == "KEEP"))

    return scores


def _score_batch(
    batch: list[tuple[int, str, str, str, str]],
    llm_client: LLMClient,
    config: VariableScoringConfig,
    num_ctx: int,
    debug_log: list[tuple[str, str]] | None = None,
) -> tuple[dict[str, VariableScore], int, int]:
    """Score a single batch of variables using the LLM.

    Args:
        batch: List of (index, element_id, file_path, name, raw_code) tuples.
        llm_client: LLM client with generate_from_messages method.
        config: Scoring configuration.
        num_ctx: Context window size for the LLM call.
        debug_log: If provided and empty, appends (user_prompt, llm_output) for
            the first batch that completes successfully (for debug display).

    Returns:
        Tuple of (scores_dict, prompt_tokens, response_tokens).
    """
    # Build the prompt
    prompt_vars = [(idx, fp, name, code) for idx, _eid, fp, name, code in batch]
    user_prompt = build_user_prompt(prompt_vars, token_budget=config.token_budget)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Estimate prompt tokens (system + user message, ~4 chars/token)
    prompt_chars = len(SYSTEM_PROMPT) + len(user_prompt)
    prompt_tokens = prompt_chars // 4

    # Output budget: ~10 tokens per variable (number + KEEP/DROP + newline)
    output_budget = len(batch) * 10 + 30

    try:
        output = llm_client.generate_from_messages(
            messages=messages,
            temperature=config.temperature,
            max_tokens=output_budget,
            timeout=config.timeout,
            num_ctx=num_ctx,
        )
    except Exception:
        logger.warning(
            "LLM scoring failed for batch of %d variables, defaulting to keep",
            len(batch),
            exc_info=True,
        )
        # Default: keep on error (false positives safer than false negatives)
        return (
            {eid: VariableScore(keep=True) for _, eid, _, _, _ in batch},
            prompt_tokens,
            0,
        )

    # Estimate response tokens
    response_tokens = len(output) // 4

    # Capture debug output for every batch (written to log file)
    if debug_log is not None:
        debug_log.append((user_prompt, output))

    # Parse scores
    parsed = _parse_scores(output, len(batch))

    result: dict[str, VariableScore] = {}
    for i, (_, element_id, _, _, _) in enumerate(batch):
        if parsed[i] is not None:
            result[element_id] = parsed[i]
        else:
            # Unparseable: default to keep (safe fallback)
            result[element_id] = VariableScore(keep=True)

    return result, prompt_tokens, response_tokens


def _snap_to_context_tier(token_budget: int) -> int:
    """Snap a token budget to the smallest CONTEXT_TIER that fits.

    Since token_budget represents the total context window, we simply find
    the smallest tier >= token_budget. If token_budget already equals a
    tier (e.g. 2048), this is a no-op.
    """
    from shared.ai.context_size import CONTEXT_TIERS

    for tier in CONTEXT_TIERS:
        if token_budget <= tier:
            return tier  # type: ignore[no-any-return]
    return CONTEXT_TIERS[-1]  # type: ignore[no-any-return]


def score_variables(
    variables: list[tuple[str, str, str, str]],
    llm_client: LLMClient,
    config: VariableScoringConfig | None = None,
    max_workers: int = 12,
    on_progress: Callable[[ScoringProgressState], None] | None = None,
    progress_state: ScoringProgressState | None = None,
    worker_status: ScoringWorkerStatus | None = None,
    on_batch_logged: Callable[[str, str], None] | None = None,
) -> ScoringResult:
    """Score variables using the LLM and return scoring results.

    Uses runtime-aware throttling to adapt worker count based on LLM response
    times, preventing GPU saturation and timeouts.

    Args:
        variables: List of (element_id, file_path, name, raw_code) tuples.
        llm_client: LLM client with generate_from_messages method.
        config: Scoring configuration (defaults to VariableScoringConfig()).
        max_workers: Maximum parallel workers for batch processing.
        on_progress: Optional callback called after each batch completes.
        progress_state: Optional shared progress state for live display.
        worker_status: Optional shared worker status for live display.
        on_batch_logged: Optional callback with (user_prompt, llm_output) after each batch.

    Returns:
        ScoringResult with scores and statistics.
    """
    if config is None:
        config = VariableScoringConfig()

    start = time.time()
    result = ScoringResult(total_variables=len(variables))

    if not variables:
        return result

    # Build batches
    batches = _build_batches(variables, config.token_budget)
    result.batch_count = len(batches)

    # Initialize progress state if provided
    if progress_state is not None:
        progress_state.total_variables = len(variables)
        progress_state.total_batches = len(batches)
        progress_state.start_time = start
        if worker_status is not None:
            progress_state.workers = worker_status

    all_scores: dict[str, VariableScore] = {}

    # Shared debug log: first successful batch captures (prompt, response)
    debug_log: list[tuple[str, str]] = []

    # Batch sampling: collect up to N batches with 3 random items each
    import random as _random

    _SAMPLE_BATCHES = 5
    _SAMPLE_PER_BATCH = 3
    batch_samples: list[list[tuple[str, str, str, VariableScore]]] = []
    _batch_sample_count = 0  # total batches seen (for reservoir sampling)

    def _collect_batch_sample(
        batch: list[tuple[int, str, str, str, str]],
        batch_scores: dict[str, VariableScore],
    ) -> None:
        """Reservoir-sample batches and pick random items from each."""
        nonlocal _batch_sample_count
        _batch_sample_count += 1

        # Build sample items: (file_path, name, raw_code, score)
        items = [
            (fp, name, code, batch_scores.get(eid, VariableScore()))
            for _, eid, fp, name, code in batch
        ]
        sampled = _random.sample(items, min(_SAMPLE_PER_BATCH, len(items)))

        # Reservoir sampling: always keep first N, then replace with probability
        if len(batch_samples) < _SAMPLE_BATCHES:
            batch_samples.append(sampled)
        else:
            j = _random.randint(0, _batch_sample_count - 1)
            if j < _SAMPLE_BATCHES:
                batch_samples[j] = sampled

    def _update_progress(batch: list, batch_scores: dict[str, VariableScore], batch_time: float, is_error: bool = False) -> None:
        """Update progress state after a batch completes."""
        if progress_state is None:
            return
        progress_state.completed_batches += 1
        progress_state.completed_variables += len(batch)
        progress_state.batch_times.append(batch_time)
        if is_error:
            progress_state.errors += 1
        else:
            # Count kept/dropped in real time
            for _eid, score in batch_scores.items():
                if score.passes_threshold(config.threshold):
                    progress_state.kept += 1
                else:
                    progress_state.dropped += 1
        if on_progress:
            on_progress(progress_state)

    # Compute context tier once from budget (all batches share the same window)
    num_ctx = _snap_to_context_tier(config.token_budget)

    # Process batches in parallel with throttling
    effective_workers = min(max_workers, len(batches))
    if progress_state is not None:
        progress_state.num_workers = effective_workers

    # Accumulate token counts across all batches
    total_prompt_tokens = 0
    total_response_tokens = 0

    _notified_count = 0

    def _notify_batch_logged() -> None:
        """Flush any new debug_log entries to on_batch_logged callback."""
        nonlocal _notified_count
        if not on_batch_logged:
            return
        while _notified_count < len(debug_log):
            prompt, output = debug_log[_notified_count]
            on_batch_logged(prompt, output)
            _notified_count += 1

    if effective_workers <= 1:
        # Sequential processing for single batch
        for batch_num, batch in enumerate(batches):
            if worker_status is not None:
                worker_status.set(0, batch_num + 1, len(batch))
            batch_start = time.time()
            batch_scores, batch_prompt_tok, batch_resp_tok = _score_batch(batch, llm_client, config, num_ctx, debug_log)
            batch_time = time.time() - batch_start
            total_prompt_tokens += batch_prompt_tok
            total_response_tokens += batch_resp_tok
            all_scores.update(batch_scores)
            _collect_batch_sample(batch, batch_scores)
            _notify_batch_logged()
            if worker_status is not None:
                worker_status.clear(0)
            _update_progress(batch, batch_scores, batch_time)
    else:
        import threading

        from shared.parallel_processor import ThrottleContext, run_throttled_tier
        from shared.throttling import ThroughputTracker

        # Throughput tracker with 3min window (matches other processors)
        throughput_tracker = ThroughputTracker(window_seconds=180.0)

        # Map thread IDs to stable worker display IDs
        _thread_id_lock = threading.Lock()
        _thread_to_wid: dict[int, int] = {}
        _next_wid = 0

        def _get_worker_id() -> int:
            """Assign a stable display ID to each thread."""
            nonlocal _next_wid
            tid = threading.get_ident()
            with _thread_id_lock:
                if tid not in _thread_to_wid:
                    _thread_to_wid[tid] = _next_wid
                    _next_wid += 1
                return _thread_to_wid[tid]

        # Items for run_throttled_tier: (batch, batch_num, num_ctx) tuples
        batch_items = [
            (batch, batch_num, num_ctx)
            for batch_num, batch in enumerate(batches)
        ]

        def process_fn(
            item: tuple[list, int, int],
        ) -> tuple[dict[str, VariableScore], int, int]:
            """Process a single batch with worker status tracking."""
            batch, batch_num, num_ctx = item
            wid = _get_worker_id()
            if worker_status is not None:
                worker_status.set(wid, batch_num + 1, len(batch))
            try:
                return _score_batch(batch, llm_client, config, num_ctx, debug_log)
            finally:
                if worker_status is not None:
                    worker_status.clear(wid)

        def _sync_throttle_to_progress() -> None:
            """Sync throttle state from ThrottleContext to progress_state."""
            if progress_state is not None:
                progress_state.throttle_decision = throttle_ctx.last_decision
                progress_state.allowed_workers = (
                    throttle_ctx._last_recommended_workers or 1
                )

        def on_complete(
            item: tuple[list, int, int],
            batch_result: tuple[dict[str, VariableScore], int, int],
            _avg_workers: float,
            runtime: float,
        ) -> None:
            """Handle batch completion: update scores and progress."""
            nonlocal total_prompt_tokens, total_response_tokens
            batch = item[0]
            batch_scores, batch_prompt_tok, batch_resp_tok = batch_result
            total_prompt_tokens += batch_prompt_tok
            total_response_tokens += batch_resp_tok
            _sync_throttle_to_progress()
            try:
                all_scores.update(batch_scores)
                _collect_batch_sample(batch, batch_scores)
                _notify_batch_logged()
                _update_progress(batch, batch_scores, runtime)
            except Exception:
                result.errors += 1
                error_scores = {}
                for _, eid, _, _, _ in batch:
                    error_scores[eid] = VariableScore(keep=True)
                all_scores.update(error_scores)
                _update_progress(batch, error_scores, runtime, is_error=True)

        def on_tick(_info: object) -> None:
            """Update progress display on poll timeouts."""
            _sync_throttle_to_progress()
            if on_progress and progress_state is not None:
                on_progress(progress_state)

        def get_max_runtime() -> float:
            if worker_status is not None:
                return worker_status.get_max_active_runtime()  # type: ignore[no-any-return]
            return 0.0

        throttle_ctx = ThrottleContext(
            tier_timeout=0,  # Set by run_throttled_tier
            base_workers=effective_workers,
            throughput_tracker=throughput_tracker,
            tier=num_ctx,
        )

        run_throttled_tier(
            items=batch_items,
            tier=num_ctx,
            effective_workers=effective_workers,
            process_fn=process_fn,
            throttle_ctx=throttle_ctx,
            get_max_runtime=get_max_runtime,
            on_complete=on_complete,
            on_tick=on_tick,
        )

    # Apply threshold (final tally from actual scores)
    result.kept = 0
    result.dropped = 0
    for _eid, score in all_scores.items():
        if score.passes_threshold(config.threshold):
            result.kept += 1
        else:
            result.dropped += 1

    result.scores = all_scores
    result.elapsed = time.time() - start
    result.prompt_tokens = total_prompt_tokens
    result.response_tokens = total_response_tokens
    result.debug_log = debug_log
    result.batch_samples = batch_samples

    return result
