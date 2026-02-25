"""LLM-based variable scoring for code discovery.

This module scores variables using a local LLM to determine whether they are
useful for code discovery. Variables scoring below threshold are dropped before
the summarization/embedding phase.

Replaces the hand-curated heuristic filter (usefulness_filter.py) with a
semantic approach that can handle novel patterns without maintenance.
"""

from __future__ import annotations

import contextlib
import logging
import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

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

    Each batch fits within the token budget. Variables are accumulated until
    the budget would be exceeded, then a new batch is started.

    Args:
        variables: List of (element_id, file_path, name, raw_code) tuples.
        token_budget: Maximum content tokens per batch.

    Returns:
        List of batches, where each batch is a list of
        (index, element_id, file_path, name, raw_code) tuples.
        Index is 1-based within each batch.
    """
    batches: list[list[tuple[int, str, str, str, str]]] = []
    current_batch: list[tuple[int, str, str, str, str]] = []
    current_tokens = 0
    batch_idx = 1

    for element_id, file_path, name, raw_code in variables:
        var_tokens = _estimate_tokens(raw_code, file_path)

        if current_tokens + var_tokens > token_budget and current_batch:
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


def _parse_scores(output: str, batch_size: int) -> list[VariableScore | None]:
    """Parse LLM output into variable scores.

    Expected format:
        1. 9,2,1,8
        2. 2,9,1,9

    Handles various formatting quirks from LLMs.

    Args:
        output: Raw LLM output text.
        batch_size: Expected number of scores.

    Returns:
        List of VariableScore (or None for unparseable lines).
    """
    scores: list[VariableScore | None] = [None] * batch_size
    score_pattern = re.compile(r"(\d+)\.\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")

    for match in score_pattern.finditer(output):
        idx = int(match.group(1))
        if 1 <= idx <= batch_size:
            with contextlib.suppress(ValueError, IndexError):
                scores[idx - 1] = VariableScore(
                    config_value=min(10, max(1, int(match.group(2)))),
                    architectural_role=min(10, max(1, int(match.group(3)))),
                    data_definition=min(10, max(1, int(match.group(4)))),
                    general_usefulness=min(10, max(1, int(match.group(5)))),
                )

    return scores


def _score_batch(
    batch: list[tuple[int, str, str, str, str]],
    llm_client: object,
    config: VariableScoringConfig,
    num_ctx: int,
    debug_log: list[tuple[str, str]] | None = None,
) -> dict[str, VariableScore]:
    """Score a single batch of variables using the LLM.

    Args:
        batch: List of (index, element_id, file_path, name, raw_code) tuples.
        llm_client: LLM client with generate_from_messages method.
        config: Scoring configuration.
        num_ctx: Context window size for the LLM call.
        debug_log: If provided and empty, appends (user_prompt, llm_output) for
            the first batch that completes successfully (for debug display).

    Returns:
        Dict mapping element_id to VariableScore.
    """
    # Build the prompt
    prompt_vars = [(idx, fp, name, code) for idx, _eid, fp, name, code in batch]
    user_prompt = build_user_prompt(prompt_vars)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Output budget: ~20 tokens per variable (number + 4 scores + commas + newline + slack)
    output_budget = len(batch) * 20 + 50

    try:
        output = llm_client.generate_from_messages(
            messages=messages,
            temperature=config.temperature,
            max_tokens=output_budget,
            num_ctx=num_ctx,
        )
    except Exception:
        logger.warning("LLM scoring failed for batch of %d variables, defaulting to keep", len(batch))
        # Default: score 5 on general_usefulness (pass threshold)
        return {
            eid: VariableScore(general_usefulness=5)
            for _, eid, _, _, _ in batch
        }

    # Capture debug output for the first successful batch
    if debug_log is not None and len(debug_log) == 0:
        debug_log.append((user_prompt, output))

    # Parse scores
    parsed = _parse_scores(output, len(batch))

    result: dict[str, VariableScore] = {}
    for i, (_, element_id, _, _, _) in enumerate(batch):
        if parsed[i] is not None:
            result[element_id] = parsed[i]
        else:
            # Unparseable: default to score 5 (keep)
            result[element_id] = VariableScore(general_usefulness=5)

    return result


def _get_context_tier(batch: list[tuple[int, str, str, str, str]]) -> int:
    """Calculate the context tier for a batch."""
    from shared.ai.context_size import CONTEXT_TIERS

    content_chars = sum(len(code) + len(fp) + 20 for _, _, fp, _, code in batch)
    output_tokens = len(batch) * 20 + 50
    total_tokens = content_chars // 4 + 700 + output_tokens  # 700 = system prompt (~660 tokens)
    num_ctx = CONTEXT_TIERS[0]  # Default to smallest
    for tier in CONTEXT_TIERS:
        if total_tokens < tier:
            num_ctx = tier
            break
    return num_ctx


def score_variables(
    variables: list[tuple[str, str, str, str]],
    llm_client: object,
    config: VariableScoringConfig | None = None,
    max_workers: int = 12,
    on_progress: Callable[[ScoringProgressState], None] | None = None,
    progress_state: ScoringProgressState | None = None,
    worker_status: ScoringWorkerStatus | None = None,
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

    # Process batches in parallel with throttling
    effective_workers = min(max_workers, len(batches))
    if progress_state is not None:
        progress_state.num_workers = effective_workers

    if effective_workers <= 1:
        # Sequential processing for single batch
        for batch_num, batch in enumerate(batches):
            num_ctx = _get_context_tier(batch)
            if worker_status is not None:
                worker_status.set(0, batch_num + 1, len(batch))
            batch_start = time.time()
            batch_scores = _score_batch(batch, llm_client, config, num_ctx, debug_log)
            batch_time = time.time() - batch_start
            all_scores.update(batch_scores)
            if worker_status is not None:
                worker_status.clear(0)
            _update_progress(batch, batch_scores, batch_time)
    else:
        import threading

        from shared.parallel_processor import ThrottleContext, run_throttled_tier
        from shared.throttling import ThroughputTracker

        tier_for_timeout = _get_context_tier(batches[0])

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
            (batch, batch_num, _get_context_tier(batch))
            for batch_num, batch in enumerate(batches)
        ]

        def process_fn(
            item: tuple[list, int, int],
        ) -> dict[str, VariableScore]:
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
            batch_scores: dict[str, VariableScore],
            _avg_workers: float,
        ) -> None:
            """Handle batch completion: update scores and progress."""
            batch = item[0]
            _sync_throttle_to_progress()
            try:
                all_scores.update(batch_scores)
                _update_progress(batch, batch_scores, 0.0)
            except Exception:
                result.errors += 1
                error_scores = {}
                for _, eid, _, _, _ in batch:
                    error_scores[eid] = VariableScore(general_usefulness=5)
                all_scores.update(error_scores)
                _update_progress(batch, error_scores, 0.0, is_error=True)

        def on_tick(_info: object) -> None:
            """Update progress display on poll timeouts."""
            _sync_throttle_to_progress()
            if on_progress and progress_state is not None:
                on_progress(progress_state)

        def get_max_runtime() -> float:
            if worker_status is not None:
                return worker_status.get_max_active_runtime()
            return 0.0

        throttle_ctx = ThrottleContext(
            tier_timeout=0,  # Set by run_throttled_tier
            base_workers=effective_workers,
            throughput_tracker=throughput_tracker,
            tier=tier_for_timeout,
        )

        run_throttled_tier(
            items=batch_items,
            tier=tier_for_timeout,
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
    result.debug_log = debug_log

    return result
