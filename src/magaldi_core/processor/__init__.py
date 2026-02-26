"""Unified element processor - atomic summarize -> embed -> index flow.

Processes elements level-by-level:
- Level 0: Files
- Level 1: Classes
- Level 2: Functions/Methods
- Level 3: Variables

Only indexes after full processing, ensuring search backend presence = completion.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shared.config import MagaldiConfig
    from shared.throttling import ThrottleDecision

from magaldi_core.code_parser import CodeElement, ParsedFile
from magaldi_core.job_tracker import RedisJobTracker, SummaryCache
from shared.ai.context_size import CONTEXT_TIERS, HANDCRAFTED_TIER, compute_element_num_ctx
from shared.ai.embedding import CodeEmbeddingClient
from shared.ai.summarization import SummarizationLLMClient
from shared.db.store import Repository

from .dependency_tracker import DependencyTracker
from .helpers import (
    _HANDCRAFTED_SUMMARY_TYPES,
    _embed_element,
    _extract_docstring_description,
    _generate_handcrafted_summary,
    _generate_import_summary,
    _generate_small_function_summary,
    _get_element_line_count,
    _index_element,
    _is_small_function,
    _process_single_element,
    _should_handcraft,
    _summarize_element,
    should_embed,
)

# Import from submodules
from .models import (
    ProcessedElement,
    ProcessingConfig,
    ProcessingResult,
    _get_model_display_name,
)
from .status import ParallelismStats, ProgressState, WorkerStatus
from .timing import TimingStats

# Re-export all public classes for backward compatibility
__all__ = [
    # Main function
    "process_elements",
    # Config and result classes
    "ProcessingConfig",
    "ProcessingResult",
    "ProcessedElement",
    # Timing and status
    "TimingStats",
    "WorkerStatus",
    "ParallelismStats",
    "ProgressState",
    # Dependency tracking
    "DependencyTracker",
    # Helpers
    "should_embed",
    "_HANDCRAFTED_SUMMARY_TYPES",
    "_generate_import_summary",
    "_extract_docstring_description",
    "_get_element_line_count",
    "_is_small_function",
    "_generate_small_function_summary",
    "_should_handcraft",
    "_generate_handcrafted_summary",
    "_summarize_element",
    "_embed_element",
    "_index_element",
    "_process_single_element",
    "_get_model_display_name",
]

logger = logging.getLogger(__name__)


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================


def process_elements(
    parsed_files: list[ParsedFile],
    scope: str,
    repository: str,
    username: str,
    repo: Repository,
    config: ProcessingConfig | None = None,
    on_progress: Callable[[ProgressState], None] | None = None,
    file_hashes: dict[str, str] | None = None,
    on_status_change: Callable[[], None] | None = None,
    worker_status: WorkerStatus | None = None,
    timing_stats: TimingStats | None = None,
    magaldi_config: MagaldiConfig | None = None,
) -> ProcessingResult:
    """Process elements: summarize -> embed -> index (atomic per element).

    Uses DependencyTracker and ThreadPoolExecutor for parallel processing
    while respecting parent-child dependencies.

    Args:
        parsed_files: List of parsed files from Phase 3.
        scope: Repository scope.
        repository: Repository name.
        username: Username/branch.
        repo: Search repository for indexing.
        config: Processing configuration.
        on_progress: Optional callback(ProgressState) for progress updates.
        file_hashes: Optional dict mapping relative_path to file hash.
        on_status_change: Optional callback when any worker status changes.
        worker_status: Optional shared WorkerStatus (created if not provided).
        timing_stats: Optional shared TimingStats (created if not provided).
        magaldi_config: Optional Magaldi config for Redis job tracking.

    Returns:
        ProcessingResult with counts and errors.
    """
    if config is None:
        config = ProcessingConfig()

    result = ProcessingResult(scope=scope, repository=repository, username=username)

    # Deduplicate parsed_files by path (keep first occurrence)
    seen_paths: set[str] = set()
    unique_parsed_files: list[ParsedFile] = []
    for pf in parsed_files:
        if pf.file_info.relative_path not in seen_paths:
            seen_paths.add(pf.file_info.relative_path)
            unique_parsed_files.append(pf)

    parsed_files = unique_parsed_files

    # Collect all elements and compute element counts per file
    all_elements: list[CodeElement] = []
    element_counts: dict[str, int] = {}
    for pf in parsed_files:
        all_elements.extend(pf.elements)
        element_counts[pf.file_info.relative_path] = len(pf.elements)

    if not all_elements:
        return result

    # Smart delete: only remove stale elements (those no longer in code)
    # For each file, compare existing indexed elements with newly parsed elements
    new_element_ids = {e.element_id for e in all_elements}
    stale_element_ids: list[str] = []

    for pf in parsed_files:
        existing_ids = repo.get_element_ids_by_file(
            scope, repository, username, pf.file_info.relative_path
        )
        # Stale = indexed but not in new code
        stale_ids = existing_ids - new_element_ids
        stale_element_ids.extend(stale_ids)

    # Get content hashes and summary state for change detection
    # Only skip if content unchanged AND summary exists (handles interrupted runs)
    all_element_ids = list(new_element_ids)
    existing_states = repo.get_element_processing_state(all_element_ids)

    # RELOCATED ELEMENT MATCHING (must happen BEFORE deleting stale elements!)
    # When an element's line number changes, its ID changes but content_hash stays same.
    # We find these "relocated" elements by matching content_hash from stale → new elements.
    # IMPORTANT: Search per-file to avoid cross-file false matches (same code in different files).
    elements_not_found_by_id = [
        elem for elem in all_elements
        if elem.element_id not in existing_states and elem.content_hash
    ]
    relocated_states: dict[str, dict] = {}  # content_hash -> state (includes actual data)
    relocated_old_ids: set[str] = set()  # old element IDs that were relocated (don't delete)
    if elements_not_found_by_id:
        # Group by file path for per-file searching
        by_file: dict[str, list[str]] = {}
        for elem in elements_not_found_by_id:
            if elem.content_hash:
                by_file.setdefault(elem.relative_path, []).append(elem.content_hash)

        # Search each file separately to avoid cross-file false matches
        for rel_path, hashes in by_file.items():
            unique_hashes = list(set(hashes))
            file_relocated = repo.find_elements_by_content_hash(
                scope, repository, username, unique_hashes, relative_path=rel_path
            )
            relocated_states.update(file_relocated)
            # Track old element IDs that matched - these are relocated, not deleted
            for state in file_relocated.values():
                old_id = state.get("old_element_id")
                if old_id:
                    relocated_old_ids.add(old_id)

    # Now delete stale elements, EXCLUDING relocated ones (they'll be updated, not deleted)
    truly_stale_ids = [eid for eid in stale_element_ids if eid not in relocated_old_ids]
    if truly_stale_ids:
        repo.delete_elements(truly_stale_ids)
        result.elements_deleted = len(truly_stale_ids)

    # Filter: only process elements that are new, changed, or missing summary
    # Also track which files had skipped elements (need file_hash update)
    elements_to_process = []
    skipped_by_file: dict[str, int] = {}  # relative_path -> count of skipped elements
    skipped_with_summary = 0
    skipped_no_summary = 0
    new_elements = 0
    state_none_count = 0
    state_found_count = 0
    relocated_copied = 0
    non_ai_count = 0  # No longer used - imports now get handcrafted summaries + embeddings
    for elem in all_elements:
        state = existing_states.get(elem.element_id)
        is_relocated = False

        # Fallback: check if element was found by content_hash (relocated element)
        if state is None and elem.content_hash:
            relocated_data = relocated_states.get(elem.content_hash)
            if relocated_data is not None:
                is_relocated = True
                state = relocated_data

        if state is not None:
            state_found_count += 1
            content_unchanged = state.get("content_hash") == elem.content_hash
            has_summary = state.get("has_summary", False)
            has_summary_embedding = state.get("has_summary_embedding", False)
            has_code_embedding = state.get("has_code_embedding", False)
            has_caller_embedding = state.get("has_caller_embedding", False)

            # Check if element is fully processed
            # For embeddable elements, require all three embeddings
            is_embeddable = should_embed(elem)
            is_fully_processed = has_summary and (
                not is_embeddable or (has_summary_embedding and has_code_embedding and has_caller_embedding)
            )

            if content_unchanged and is_fully_processed:
                if is_relocated:
                    # Relocated element: copy data to new element with new ID
                    # This preserves the summary while updating line numbers
                    file_hash = file_hashes.get(elem.relative_path) if file_hashes else None
                    element_count = None
                    if elem.element_type == "file" and element_counts:
                        element_count = element_counts.get(elem.relative_path)
                    _index_element(
                        elem,
                        state.get("summary"),
                        state.get("summary_embedding"),
                        state.get("code_embedding"),
                        state.get("caller_embedding"),
                        repo,
                        file_hash,
                        element_count,
                    )
                    # Delete the old element now that we've indexed the new one
                    old_element_id = state.get("old_element_id")
                    if old_element_id:
                        repo.delete_elements([old_element_id])
                    relocated_copied += 1
                    result.elements_skipped += 1
                    skipped_by_file[elem.relative_path] = skipped_by_file.get(elem.relative_path, 0) + 1
                    skipped_with_summary += 1
                    continue
                else:
                    # Element exists with same ID and content - skip entirely
                    result.elements_skipped += 1
                    skipped_by_file[elem.relative_path] = skipped_by_file.get(elem.relative_path, 0) + 1
                    skipped_with_summary += 1
                    continue
            elif content_unchanged and not is_fully_processed:
                skipped_no_summary += 1
        else:
            state_none_count += 1
            new_elements += 1
        # Element is new OR content changed OR missing summary/embeddings - needs processing
        elements_to_process.append(elem)

    # Total = elements that need processing (excludes unchanged)
    total = len(elements_to_process)
    # Track totals for display
    unchanged_count = result.elements_skipped
    total_found = len(all_elements)  # All elements found from parsing

    # Count elements per file to identify files where ALL elements were skipped
    elements_per_file: dict[str, int] = {}
    for elem in all_elements:
        elements_per_file[elem.relative_path] = elements_per_file.get(elem.relative_path, 0) + 1

    # Find files where ALL elements were skipped and update their file_hash in ES
    # This prevents files from appearing "modified" on every run when only
    # file metadata changed but element content stayed the same
    files_to_update: dict[str, str] = {}
    if file_hashes and skipped_by_file:
        for rel_path, skipped_count in skipped_by_file.items():
            if skipped_count == elements_per_file.get(rel_path, 0) and rel_path in file_hashes:
                # All elements in this file were skipped - update file_hash
                files_to_update[rel_path] = file_hashes[rel_path]

        if files_to_update:
            # Pass element_counts so FILE elements also get updated element_count
            repo.update_file_hashes(
                scope, repository, username, files_to_update, elements_per_file
            )

    if not elements_to_process:
        # All elements unchanged - fire progress callback showing 0/0 with skipped count
        if on_progress:
            if timing_stats is None:
                timing_stats = TimingStats()
            if worker_status is None:
                worker_status = WorkerStatus()
            progress_state = ProgressState(
                total=0,  # Nothing to process
                completed=0,
                skipped=unchanged_count,  # All were unchanged
                failed=0,
                timing=timing_stats,
                workers=worker_status,
                num_workers=config.num_workers if config.num_workers > 0 else 8,
                recent_errors=[],
                parallelism=None,
                total_found=total_found,
                non_ai_skipped=non_ai_count,
            )
            on_progress(progress_state)
        return result

    # Summary cache for hierarchical context
    summary_cache = SummaryCache()

    # Populate cache with all elements (for parent lookup)
    for elem in all_elements:
        summary_cache.add_element(elem)

    # Initialize LLM clients (only if not skipping AI)
    llm_client: SummarizationLLMClient | None = None
    embed_client: CodeEmbeddingClient | None = None

    if not config.skip_ai:
        summarize_cfg = config.summarize_model
        llm_client = SummarizationLLMClient(
            url=summarize_cfg.get_api_base() or "",
            model=summarize_cfg.name,
            provider=summarize_cfg.provider,
            api_key=summarize_cfg.api_key,
        )
        embed_cfg = config.embed_model
        embed_client = CodeEmbeddingClient(
            url=embed_cfg.get_api_base() or "",
            model=embed_cfg.name,
            provider=embed_cfg.provider,
            api_key=embed_cfg.api_key,
        )

    # Pre-compute context sizes for all elements (for tier batching)
    # This enables DependencyTracker to group elements by context tier
    element_context_sizes: dict[str, int] = {}
    for elem in elements_to_process:
        char_count = len(elem.raw_code or "")
        element_context_sizes[elem.element_id] = compute_element_num_ctx(
            elem.element_type, char_count
        )

    # Initialize tracking structures (use provided or create new)
    # Pass per-element context sizes for tier batching (minimizes model reloads)
    # Pass num_workers as upper limit (0 or None = use tier defaults)
    max_num_workers = config.num_workers if config.num_workers > 0 else None
    dependency_tracker = DependencyTracker(
        elements_to_process,
        element_context_sizes,
        max_num_workers=max_num_workers,
        timeout=config.summarize_timeout,  # For throttle calculations
    )
    if timing_stats is None:
        timing_stats = TimingStats()
    timing_stats.phase_start = time.time()
    if worker_status is None:
        worker_status = WorkerStatus()

    # Count elements by type for per-type ETA
    totals_by_type: dict[str, int] = {}
    for elem in elements_to_process:
        totals_by_type[elem.element_type] = totals_by_type.get(elem.element_type, 0) + 1
    timing_stats.set_totals_by_type(totals_by_type)

    # Count elements by (type, tier) for tier-aware ETA
    # Handcrafted elements get HANDCRAFTED_TIER (0) instead of a context tier
    totals_by_type_tier: dict[tuple[str, int], int] = {}
    for elem in elements_to_process:
        if _should_handcraft(elem, config):
            key = (elem.element_type, HANDCRAFTED_TIER)
        else:
            ctx_size = element_context_sizes.get(elem.element_id, 2048)
            # Snap to standard tier
            tier = 2048
            for t in CONTEXT_TIERS:
                if ctx_size <= t:
                    tier = t
                    break
            else:
                tier = CONTEXT_TIERS[-1]
            key = (elem.element_type, tier)
        totals_by_type_tier[key] = totals_by_type_tier.get(key, 0) + 1
    timing_stats.set_totals_by_type_tier(totals_by_type_tier)

    # Track completed/failed counts for progress
    completed_count = 0  # Only count actually processed elements
    failed_count = 0
    recent_errors: list[tuple[str, str]] = []  # Track recent errors for display

    # Initialize Redis job tracker if config provided
    redis_tracker: RedisJobTracker | None = None
    if magaldi_config is not None:
        try:
            redis_tracker = RedisJobTracker(magaldi_config, scope, repository, username, should_embed)
            redis_tracker.clear_queues()  # Clear stale data before adding new jobs
            redis_tracker.add_pending_jobs(elements_to_process)
        except Exception:
            # Redis unavailable - continue without tracking
            logger.debug("Redis unavailable for job tracking, continuing without tracking", exc_info=True)
            redis_tracker = None

    # Worker ID pool - use configured num_workers or default
    max_workers = config.num_workers if config.num_workers > 0 else DependencyTracker.DEFAULT_WORKERS
    available_worker_ids: list[int] = list(range(max_workers))
    worker_id_lock = threading.Lock()

    def acquire_worker_id() -> int:
        """Get an available worker ID from the pool."""
        with worker_id_lock:
            if available_worker_ids:
                return available_worker_ids.pop(0)
            return 0  # Fallback

    def release_worker_id(wid: int) -> None:
        """Return a worker ID to the pool (sorted to prefer low IDs)."""
        import bisect
        with worker_id_lock:
            if wid not in available_worker_ids:
                bisect.insort(available_worker_ids, wid)

    def process_wrapper(element: CodeElement) -> ProcessedElement:
        """Wrapper to assign worker ID and call _process_single_element."""
        wid = acquire_worker_id()
        # Mark as running in Redis before processing
        if redis_tracker:
            try:
                redis_tracker.mark_running(element.element_id, should_embed(element))
            except Exception:
                logger.debug("Failed to mark element as running in Redis", exc_info=True)
        try:
            return _process_single_element(
                element=element,
                summary_cache=summary_cache,
                llm_client=llm_client,
                embed_client=embed_client,
                config=config,
                file_hashes=file_hashes,
                element_counts=element_counts,
                repo=repo,
                worker_id=wid,
                worker_status=worker_status,
                on_status_change=on_status_change,
            )
        finally:
            release_worker_id(wid)

    # Process elements in parallel using ThreadPoolExecutor
    # DependencyTracker manages concurrency with tier batching and runtime-based throttling
    executor = ThreadPoolExecutor(max_workers=max_workers)
    future_to_element: dict = {}

    # Track allowed workers at task start for averaging with end value
    future_to_allowed_at_start: dict[Any, int] = {}

    # Track current throttle decision for display
    current_throttle: ThrottleDecision | None = None
    # Interval for wait() timeout - allows periodic display updates
    HISTORY_RECORD_INTERVAL = 2.0  # Refresh display every 2 seconds

    try:
        while not dependency_tracker.is_complete():
            # Get current max from active workers (for emergency throttling)
            current_max_runtime = worker_status.get_max_active_runtime()

            # Get current time and active worker count for throttle decisions
            time.time()
            active_workers = worker_status.active_count()

            # Get throughput-based stats for adaptive throttling (with concurrency context)
            throughput, avg_runtime, completion_count, avg_concurrency, high_load_avg = (
                timing_stats.get_throughput_stats_with_concurrency()
            )
            peak_concurrency = timing_stats.get_peak_concurrency()
            remaining = dependency_tracker.pending_count()
            exploration_target = timing_stats.get_exploration_target(max_workers, remaining)
            all_levels = timing_stats.get_all_throughput_levels()
            current_throttle = dependency_tracker.compute_throttle_decision(
                current_max_runtime, active_workers, throughput, avg_runtime, completion_count,
                avg_concurrency, high_load_avg, peak_concurrency=peak_concurrency,
                exploration_target=exploration_target,
                all_levels=all_levels if all_levels else None,
            )

            # Always use recommended_workers as the limit (includes ramp-up logic)
            # Even when not throttling, we ramp up gradually to avoid overwhelming
            throttle_limit = current_throttle.recommended_workers

            # Clear post_warmup flag AFTER getting throttle_limit (not in display calls)
            # This ensures the flag is only consumed once per warmup, for the main calculation
            dependency_tracker.clear_post_warmup()

            # Get elements that are ready (parents completed)
            # DependencyTracker applies tier-specific limits internally
            # Pass throttle_limit to further reduce concurrency if needed
            ready_elements = dependency_tracker.get_ready_elements(
                max_count=max_workers * 2,
                throttle_limit=throttle_limit,
            )

            # Reset throughput history on tier/model change so stale data
            # from a different tier doesn't pollute throttling decisions
            if dependency_tracker.did_tier_just_change():
                timing_stats.reset_throughput()

            # Submit new tasks for ready elements
            # Track allowed workers at start for throughput calculation
            for element in ready_elements:
                future = executor.submit(process_wrapper, element)
                future_to_element[future] = element
                future_to_allowed_at_start[future] = throttle_limit

            if not future_to_element:
                # No futures pending and not complete - shouldn't happen
                # Store diagnostic info in result for the CLI to display
                result.errors.append(
                    f"Processing stalled: no ready elements. "
                    f"pending={dependency_tracker.pending_count()}, "
                    f"tier_changing={dependency_tracker.is_tier_changing()}"
                )
                break

            # Wait for at least one to complete, or timeout for periodic updates
            done, _ = wait(future_to_element.keys(), timeout=HISTORY_RECORD_INTERVAL, return_when=FIRST_COMPLETED)

            # If timeout with no completions, just update display
            # NOTE: We no longer record to history here because fresh_active
            # (current count) gives wrong base_time for tasks that started earlier.
            if not done:
                fresh_current_max = worker_status.get_max_active_runtime()
                fresh_active = worker_status.active_count()

                if on_progress:
                    fresh_throughput, fresh_avg, fresh_count, fresh_avg_conc, fresh_high_load = (
                        timing_stats.get_throughput_stats_with_concurrency()
                    )
                    fresh_peak = timing_stats.get_peak_concurrency()
                    fresh_remaining = dependency_tracker.pending_count()
                    fresh_explore = timing_stats.get_exploration_target(max_workers, fresh_remaining)
                    fresh_all_levels = timing_stats.get_all_throughput_levels()
                    fresh_throttle = dependency_tracker.compute_throttle_decision(
                        fresh_current_max, fresh_active, fresh_throughput, fresh_avg, fresh_count,
                        fresh_avg_conc, fresh_high_load, is_display_call=True,
                        peak_concurrency=fresh_peak,
                        exploration_target=fresh_explore,
                        all_levels=fresh_all_levels if fresh_all_levels else None,
                    )
                    progress_state = ProgressState(
                        total=total,
                        completed=completed_count,
                        skipped=result.elements_skipped,
                        failed=failed_count,
                        timing=timing_stats,
                        workers=worker_status,
                        num_workers=max_workers,
                        recent_errors=list(recent_errors),
                        parallelism=dependency_tracker.get_parallelism_stats(
                            max_workers, fresh_throttle
                        ),
                        total_found=total_found,
                        non_ai_skipped=non_ai_count,
                    )
                    on_progress(progress_state)
                continue

            for future in done:
                element = future_to_element.pop(future)
                processed = future.result()

                # Get element's tier for accurate ETA tracking
                element_tier = dependency_tracker._get_tier(element.element_id)

                # Use peak allowed workers for throughput calculation
                # max(start, end) reflects the peak concurrency the element competed with
                allowed_at_start = future_to_allowed_at_start.pop(future, throttle_limit)
                allowed_at_end = throttle_limit
                avg_workers = max(allowed_at_start, allowed_at_end)

                # Record timing with element type, tier, and avg_workers (for throughput)
                # Handcrafted elements use HANDCRAFTED_TIER (0) for their own ETA column
                record_tier = HANDCRAFTED_TIER if _should_handcraft(element, config) else element_tier
                timing_stats.record(
                    processed.wall_time,
                    processed.summarize_time,
                    processed.embed_time,
                    element.element_type,
                    was_embedded=should_embed(element),
                    summary_embed_time=processed.summary_embed_time,
                    code_embed_time=processed.code_embed_time,
                    tier=record_tier,
                    avg_workers=avg_workers,
                    prompt_tokens=processed.prompt_tokens,
                    response_tokens=processed.response_tokens,
                    assigned_tier=processed.assigned_tier,
                )
                timing_stats.record_task_runtime(processed.wall_time, avg_workers)

                was_embedded = should_embed(element)
                if processed.success:
                    dependency_tracker.mark_complete(element.element_id)
                    result.elements_processed += 1
                    result.indexed += 1
                    if not config.skip_ai:
                        result.summarized += 1
                        if was_embedded:
                            result.embedded += 1
                    completed_count += 1
                    # Update Redis job status
                    if redis_tracker:
                        try:
                            redis_tracker.mark_completed(element.element_id, was_embedded)
                        except Exception:
                            # Don't fail processing if Redis update fails
                            logger.debug("Failed to mark element as completed in Redis", exc_info=True)
                else:
                    dependency_tracker.mark_failed(element.element_id)
                    result.elements_failed += 1
                    failed_count += 1
                    error_msg = f"Failed to process {element.element_id}: {processed.error}"
                    result.errors.append(error_msg)
                    result.failed_elements.append((element.element_id, processed.error or "Unknown error"))
                    # Add to recent errors for display (keep last 3)
                    short_name = f"{element.element_type}:{element.name}"
                    recent_errors.append((short_name, processed.error or "Unknown error"))
                    if len(recent_errors) > 3:
                        recent_errors.pop(0)

                    # Update Redis job status
                    if redis_tracker:
                        try:
                            redis_tracker.mark_failed(element.element_id, processed.error or "Unknown", was_embedded)
                        except Exception:
                            logger.debug("Failed to mark element as failed in Redis", exc_info=True)

                # Report progress with throttle info
                if on_progress:
                    # Refresh throttle decision with current throughput values for accurate display
                    fresh_current_max = worker_status.get_max_active_runtime()
                    fresh_throughput, fresh_avg, fresh_count, fresh_avg_conc, fresh_high_load = (
                        timing_stats.get_throughput_stats_with_concurrency()
                    )
                    fresh_active = worker_status.active_count()
                    fresh_peak = timing_stats.get_peak_concurrency()
                    fresh_remaining = dependency_tracker.pending_count()
                    fresh_explore = timing_stats.get_exploration_target(max_workers, fresh_remaining)
                    fresh_all_levels = timing_stats.get_all_throughput_levels()
                    fresh_throttle = dependency_tracker.compute_throttle_decision(
                        fresh_current_max, fresh_active, fresh_throughput, fresh_avg, fresh_count,
                        fresh_avg_conc, fresh_high_load, peak_concurrency=fresh_peak,
                        exploration_target=fresh_explore,
                        all_levels=fresh_all_levels if fresh_all_levels else None,
                    )
                    progress_state = ProgressState(
                        total=total,
                        completed=completed_count,
                        skipped=result.elements_skipped,
                        failed=failed_count,
                        timing=timing_stats,
                        workers=worker_status,
                        num_workers=max_workers,
                        recent_errors=list(recent_errors),
                        parallelism=dependency_tracker.get_parallelism_stats(
                            max_workers, fresh_throttle
                        ),
                        total_found=total_found,
                        non_ai_skipped=non_ai_count,
                    )
                    on_progress(progress_state)

    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C - cancel pending futures and stop executor
        for future in future_to_element:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        # Clear worker status display first (so Live display is clean)
        for wid in range(max_workers):
            worker_status.clear(wid)
        # Clean up Redis tracker
        if redis_tracker:
            try:
                redis_tracker.close()
            except Exception:
                logger.debug("Failed to close Redis tracker on interrupt", exc_info=True)
        # Re-raise so CLI can handle the wait message and exit
        raise
    else:
        # Normal completion - shutdown and wait
        executor.shutdown(wait=True)

    # Clean up Redis tracker
    if redis_tracker:
        try:
            redis_tracker.close()
        except Exception:
            logger.debug("Failed to close Redis tracker", exc_info=True)

    return result
