"""Data models for variable scoring."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.throttling import ThrottleDecision


@dataclass
class VariableScore:
    """Multi-dimensional score for a variable's usefulness for code discovery."""

    config_value: int = 1
    architectural_role: int = 1
    data_definition: int = 1
    general_usefulness: int = 1

    @property
    def max_score(self) -> int:
        """Return the highest dimension score."""
        return max(
            self.config_value,
            self.architectural_role,
            self.data_definition,
            self.general_usefulness,
        )

    def passes_threshold(self, threshold: int = 5) -> bool:
        """Check if any dimension scores at or above threshold."""
        return self.max_score >= threshold

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return scores as a tuple for compact display."""
        return (
            self.config_value,
            self.architectural_role,
            self.data_definition,
            self.general_usefulness,
        )


@dataclass
class VariableScoringConfig:
    """Configuration for variable scoring."""

    threshold: int = 5
    temperature: float = 0.1
    token_budget: int = 1200
    max_retries: int = 2
    timeout: int = 180  # 3 minutes — matches ProcessingConfig.summarize_timeout
    debug_log: bool = False  # Write all batch prompts + LLM responses to logs/


@dataclass
class ScoringResult:
    """Result of the variable scoring phase."""

    total_variables: int = 0
    kept: int = 0
    dropped: int = 0
    heuristic_dropped: int = 0  # Dropped by pre-filter (before LLM)
    cached_scores: int = 0  # Reused from previous run cache
    llm_scored: int = 0  # Actually sent to LLM
    batch_count: int = 0
    elapsed: float = 0.0
    errors: int = 0
    scores: dict[str, VariableScore] = field(default_factory=dict)
    # Token usage tracking (estimated, same as Phase 5)
    prompt_tokens: int = 0  # Total input tokens across all batches
    response_tokens: int = 0  # Total output tokens across all batches
    model_name: str = ""  # Model used for scoring
    # Debug: (user_prompt, llm_output) for first successful batch
    debug_log: list[tuple[str, str]] = field(default_factory=list)
    # Debug: batch-level samples for display
    # Each entry is a list of (file_path, name, raw_code, score) from one batch
    batch_samples: list[list[tuple[str, str, str, VariableScore]]] = field(
        default_factory=list
    )


@dataclass
class ScoringWorkerStatus:
    """Track what each worker is doing during variable scoring."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    # worker_id -> (batch_num, batch_size, start_time)
    _status: dict[int, tuple[int, int, float]] = field(default_factory=dict)

    def set(self, worker_id: int, batch_num: int, batch_size: int) -> None:
        """Mark worker as processing a batch."""
        with self._lock:
            self._status[worker_id] = (batch_num, batch_size, time.time())

    def clear(self, worker_id: int) -> None:
        """Mark worker as idle."""
        with self._lock:
            self._status.pop(worker_id, None)

    def get_all(self) -> dict[int, tuple[int, int, float]]:
        """Get all worker statuses."""
        with self._lock:
            return dict(self._status)

    def active_count(self) -> int:
        """Get number of active workers."""
        with self._lock:
            return len(self._status)

    def get_max_active_runtime(self) -> float:
        """Get the max runtime of any currently active worker."""
        now = time.time()
        with self._lock:
            if not self._status:
                return 0.0
            return max(now - start for _, _, start in self._status.values())


@dataclass
class ScoringProgressState:
    """Progress state for variable scoring live display."""

    total_variables: int = 0
    total_batches: int = 0
    completed_batches: int = 0
    completed_variables: int = 0
    kept: int = 0
    dropped: int = 0
    errors: int = 0
    num_workers: int = 1
    start_time: float = field(default_factory=time.time)
    workers: ScoringWorkerStatus = field(default_factory=ScoringWorkerStatus)
    # Per-batch timing for throughput calculation
    batch_times: list[float] = field(default_factory=list)
    # Throttle state for display
    throttle_decision: ThrottleDecision | None = None
    allowed_workers: int | None = None

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        return time.time() - self.start_time

    @property
    def avg_batch_time(self) -> float:
        """Average batch processing time."""
        if not self.batch_times:
            return 0.0
        return sum(self.batch_times) / len(self.batch_times)

    @property
    def avg_variable_time(self) -> float:
        """Average wall time per variable."""
        if self.completed_variables == 0:
            return 0.0
        return self.elapsed / self.completed_variables

    def eta_seconds(self) -> float | None:
        """Estimate time remaining based on batch throughput."""
        if self.completed_batches == 0:
            return None
        remaining_batches = self.total_batches - self.completed_batches
        if remaining_batches <= 0:
            return 0.0
        # Use allowed_workers (throttled count) for ETA if available
        workers = self.allowed_workers if self.allowed_workers is not None else self.num_workers
        effective_workers = min(workers, remaining_batches)
        if effective_workers <= 0:
            return None
        return (remaining_batches / effective_workers) * self.avg_batch_time
