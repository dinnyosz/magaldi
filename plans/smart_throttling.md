# Smart Throttling Implementation Plan

**Status: IMPLEMENTED**

## Overview

Implement runtime-aware throttling for AI processing phases (4-7) that monitors the max runtime of active threads and adjusts concurrency dynamically to prevent timeouts.

## Current State

- All phases use `ThreadPoolExecutor` with `TIER_MAX_WORKERS` limits
- `WorkerStatus` tracks per-worker: `element_name`, `stage`, `model`, `ctx_size`, `start_time`
- Fixed timeouts: 180s summarize, 120s embed, 30s labeling
- No runtime-based throttling exists

## Design

### 1. RuntimeHistory Class

New class to track max runtimes in sliding 10-second windows:

```python
@dataclass
class RuntimeWindow:
    """A 10-second window of runtime data."""
    timestamp: float  # Window start time
    max_runtime: float  # Max runtime observed in this window

class RuntimeHistory:
    """Tracks historical max runtimes for throttling decisions."""

    def __init__(self, window_seconds: float = 10.0, history_windows: int = 6):
        self.window_seconds = window_seconds  # 10 seconds per window
        self.history_windows = history_windows  # Keep 6 windows = 60 seconds
        self.windows: deque[RuntimeWindow] = deque(maxlen=history_windows)
        self.current_window_start: float = 0
        self.current_window_max: float = 0
        self._lock = Lock()

    def record_runtime(self, runtime: float) -> None:
        """Record a completed task's runtime."""
        now = time.time()
        with self._lock:
            if now - self.current_window_start >= self.window_seconds:
                # Rotate to new window
                if self.current_window_max > 0:
                    self.windows.append(RuntimeWindow(
                        timestamp=self.current_window_start,
                        max_runtime=self.current_window_max
                    ))
                self.current_window_start = now
                self.current_window_max = runtime
            else:
                self.current_window_max = max(self.current_window_max, runtime)

    def get_historical_max(self) -> float:
        """Get the max runtime across all historical windows."""
        with self._lock:
            if not self.windows and self.current_window_max == 0:
                return 0.0
            historical_max = max((w.max_runtime for w in self.windows), default=0)
            return max(historical_max, self.current_window_max)
```

### 2. Enhanced TimingStats

Add runtime tracking to existing `TimingStats` classes:

```python
class TimingStats:
    # Existing fields...
    runtime_history: RuntimeHistory = field(default_factory=RuntimeHistory)

    def record_task_completion(self, runtime: float) -> None:
        """Record a completed task's runtime for throttling."""
        self.runtime_history.record_runtime(runtime)
```

### 3. Active Runtime Calculation

Add method to calculate max runtime of currently active workers:

```python
def get_max_active_runtime(self) -> float:
    """Get the max runtime of currently running workers."""
    now = time.time()
    max_runtime = 0.0
    with self._lock:
        for status in self.worker_status.values():
            if status.start_time:
                runtime = now - status.start_time
                max_runtime = max(max_runtime, runtime)
    return max_runtime
```

### 4. Throttling Logic

New function to determine if throttling should be applied:

```python
class ThrottleDecision:
    should_throttle: bool
    current_max: float  # Max runtime of active workers
    historical_max: float  # Max from 10s windows
    recommended_workers: int  # Suggested worker count
    reason: str

def compute_throttle_decision(
    stats: TimingStats,
    tier_timeout: float,  # e.g., 180s for summarize
    base_workers: int,  # Original TIER_MAX_WORKERS value
) -> ThrottleDecision:
    """Determine if throttling should be applied based on runtimes."""

    current_max = stats.get_max_active_runtime()
    historical_max = stats.runtime_history.get_historical_max()

    # Use the higher of current or historical max
    effective_max = max(current_max, historical_max)

    # Throttle thresholds (percentage of timeout)
    if effective_max == 0:
        return ThrottleDecision(False, 0, 0, base_workers, "No data")

    ratio = effective_max / tier_timeout

    if ratio >= 0.8:  # >= 80% of timeout
        # Critical: reduce to 25% workers
        workers = max(1, base_workers // 4)
        return ThrottleDecision(True, current_max, historical_max, workers,
                               f"Critical: {ratio:.0%} of timeout")
    elif ratio >= 0.5:  # >= 50% of timeout
        # Warning: reduce to 50% workers
        workers = max(1, base_workers // 2)
        return ThrottleDecision(True, current_max, historical_max, workers,
                               f"Warning: {ratio:.0%} of timeout")
    elif ratio >= 0.3:  # >= 30% of timeout
        # Caution: reduce to 75% workers
        workers = max(1, int(base_workers * 0.75))
        return ThrottleDecision(True, current_max, historical_max, workers,
                               f"Caution: {ratio:.0%} of timeout")
    else:
        return ThrottleDecision(False, current_max, historical_max, base_workers,
                               "Normal")
```

### 5. Display Changes

Update progress display to show runtime metrics:

**Current display:**
```
Processing: 45/100 [████████░░] | Summarizing... | Workers: 8
```

**New display:**
```
Processing: 45/100 [████████░░] | Summarizing... | Workers: 4/8 | Max: 45.2s (hist: 52.1s)
```

Format: `Workers: {active}/{max} | Max: {current_max}s (hist: {historical_max}s)`

When throttled, add indicator:
```
Processing: 45/100 [████████░░] | Summarizing... | Workers: 4/8 ⚠ | Max: 45.2s (hist: 52.1s)
```

### 6. Integration Points

#### Phase 4 (AI Processing) - `processor.py`

```python
# In process_batch_parallel()
while pending_elements:
    # Check throttle decision before submitting new work
    decision = compute_throttle_decision(stats, timeout, TIER_MAX_WORKERS[tier])
    effective_workers = decision.recommended_workers

    # Submit up to effective_workers tasks
    while len(futures) < effective_workers and pending_elements:
        element = pending_elements.pop(0)
        future = executor.submit(process_element, element)
        futures.append(future)

    # On task completion, record runtime
    for future in as_completed(futures, timeout=1):
        runtime = time.time() - future_start_times[future]
        stats.record_task_completion(runtime)
```

#### Phases 5-7 (Feature/Subfeature/Glossary)

Same pattern - wrap existing ThreadPoolExecutor usage with throttle checks.

## Implementation Steps

1. **Create `src/shared/throttling.py`**
   - `RuntimeWindow` dataclass
   - `RuntimeHistory` class
   - `ThrottleDecision` dataclass
   - `compute_throttle_decision()` function

2. **Update `src/magaldi_core/processor.py`**
   - Import throttling module
   - Add `runtime_history` to stats tracking
   - Integrate throttle decision into batch processing loop
   - Update progress display format

3. **Update `src/magaldi_core/features.py`**
   - Same throttling integration for feature labeling/processing

4. **Update `src/magaldi_core/subfeatures.py`**
   - Same throttling integration for subfeature extraction

5. **Update `src/magaldi_core/glossary.py`**
   - Same throttling integration for glossary generation

6. **Add tests**
   - Unit tests for RuntimeHistory
   - Unit tests for compute_throttle_decision
   - Integration test for throttling behavior

## Configuration

Add to config (optional future enhancement):
```yaml
throttling:
  enabled: true
  window_seconds: 10
  history_windows: 6
  thresholds:
    critical: 0.8  # 80% of timeout
    warning: 0.5   # 50% of timeout
    caution: 0.3   # 30% of timeout
```

## Benefits

1. **Prevents cascading timeouts**: When tasks run slow, reduces concurrency to let them complete
2. **Historical awareness**: Doesn't over-recover when one slow task completes but similar tasks remain slow
3. **Visibility**: Shows max runtime in display so user can see why throttling is happening
4. **Gradual response**: Three throttle levels (caution/warning/critical) for proportional response

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Over-throttling | Use 60s history window to smooth spikes |
| Under-throttling | Use historical max, not just current |
| Display clutter | Compact format, only show when relevant |
| Thread safety | Use locks in RuntimeHistory |
