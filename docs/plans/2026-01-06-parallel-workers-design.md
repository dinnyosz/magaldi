# Parallel Workers Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add parallel processing with rich progress display showing worker status, timing stats, and ETA.

**Architecture:** ThreadPoolExecutor with N workers, each processing complete element pipeline (summarize → embed → index). Dependency tracking ensures children wait for parent summaries.

**Tech Stack:** concurrent.futures, threading.Lock, Rich Live display

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Main Thread                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ Dependency  │  │   Worker    │  │  Rich Live      │ │
│  │  Tracker    │──│    Pool     │──│   Display       │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
│         │               │                   │          │
│         ▼               ▼                   ▼          │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Thread-Safe State                   │   │
│  │  - Summary Cache (for child context)            │   │
│  │  - Timing Stats (wall/api times)                │   │
│  │  - Worker Status (current element per worker)   │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**Processing flow per level:**
1. DependencyTracker identifies ready elements (parent done or no parent)
2. Workers pull ready elements from queue
3. Each worker: summarize → embed → index (atomic)
4. On completion: update tracker, cache summary, update stats
5. Display refreshes showing worker status + progress + ETA

---

## Data Structures

### Timing Statistics

```python
@dataclass
class TimingStats:
    """Thread-safe timing statistics."""

    _lock: threading.Lock = field(default_factory=threading.Lock)

    wall_times: list[float] = field(default_factory=list)
    api_times: list[float] = field(default_factory=list)
    phase_start: float = 0.0

    def record(self, wall_time: float, api_time: float) -> None:
        with self._lock:
            self.wall_times.append(wall_time)
            self.api_times.append(api_time)

    @property
    def avg_wall_time(self) -> float:
        with self._lock:
            return sum(self.wall_times) / len(self.wall_times) if self.wall_times else 0.0

    @property
    def avg_api_time(self) -> float:
        with self._lock:
            return sum(self.api_times) / len(self.api_times) if self.api_times else 0.0

    @property
    def elapsed(self) -> float:
        return time.time() - self.phase_start

    def eta_seconds(self, completed: int, total: int) -> float | None:
        """Calculate ETA based on average wall time."""
        if completed == 0:
            return None
        remaining = total - completed
        return remaining * self.avg_wall_time
```

### Worker Status

```python
@dataclass
class WorkerStatus:
    """Track what each worker is doing."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _status: dict[int, tuple[str, str]] = field(default_factory=dict)  # worker_id -> (element_name, stage)

    def set(self, worker_id: int, element_name: str, stage: str) -> None:
        with self._lock:
            self._status[worker_id] = (element_name, stage)

    def clear(self, worker_id: int) -> None:
        with self._lock:
            self._status.pop(worker_id, None)

    def get_all(self) -> dict[int, tuple[str, str]]:
        with self._lock:
            return dict(self._status)
```

### Progress State

```python
@dataclass
class ProgressState:
    """Combined state for display updates."""

    total: int
    completed: int
    skipped: int
    failed: int
    timing: TimingStats
    workers: WorkerStatus
```

### Processed Element Result

```python
@dataclass
class ProcessedElement:
    """Result from processing a single element."""

    element_id: str
    success: bool
    wall_time: float
    api_time: float
    error: str | None = None
```

---

## Display Design

### Live Display Layout

```
Phase 4: Processing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45% 360/800 [2:15 elapsed, ~2:45 ETA]

Workers:
  [1] summarize  MyClass.process_data
  [2] embed      utils.helper_function
  [3] index      Config.__init__
  [4] idle

Avg: 1.2s wall | 0.8s API
```

### Final Output (after completion)

```
Phase 4: Processing
  360 processed | 440 skipped | 0 failed | 800 indexed
  Avg: 1.2s wall | 0.8s API | Total: 4:30
```

### Implementation

```python
def _build_live_display(self, state: ProgressState) -> RenderableType:
    """Build Rich renderable for live display."""
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeElapsedColumn
    from rich.console import Group

    # Progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Processing[/]"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        TextColumn(self._format_eta(state)),
    )
    task = progress.add_task("processing", total=state.total, completed=state.completed)

    # Worker table
    worker_table = Table(show_header=False, box=None, padding=(0, 1))
    worker_table.add_column("Worker", style="dim")
    worker_table.add_column("Stage", style="cyan")
    worker_table.add_column("Element")

    for worker_id, (element, stage) in sorted(state.workers.get_all().items()):
        worker_table.add_row(f"[{worker_id}]", stage, element)

    # Fill idle workers
    for i in range(1, self.num_workers + 1):
        if i not in state.workers.get_all():
            worker_table.add_row(f"[{i}]", "idle", "")

    # Stats line
    stats = f"Avg: {state.timing.avg_wall_time:.1f}s wall | {state.timing.avg_api_time:.1f}s API"

    return Group(progress, "", worker_table, "", stats)

def _format_eta(self, state: ProgressState) -> str:
    """Format ETA string."""
    eta = state.timing.eta_seconds(state.completed, state.total)
    if eta is None:
        return ""
    minutes, seconds = divmod(int(eta), 60)
    return f"[~{minutes}:{seconds:02d} ETA]"
```

---

## Dependency Tracker

```python
class DependencyTracker:
    """Track element dependencies for parallel processing.

    Rules:
    - Level 0 (files): Always ready
    - Level 1 (classes): Ready when parent file done
    - Level 2 (methods/functions): Ready when parent class done (or file if no class)
    - Level 3 (variables): Ready when parent done
    """

    def __init__(self, elements: list[CodeElement]) -> None:
        self._lock = threading.Lock()
        self._elements = {e.element_id: e for e in elements}
        self._completed: set[str] = set()
        self._in_progress: set[str] = set()

        # Build parent lookup: element_id -> parent_element_id
        self._parents: dict[str, str | None] = {}
        for e in elements:
            self._parents[e.element_id] = e.parent_id

    def get_ready_elements(self, max_count: int = 10) -> list[CodeElement]:
        """Get elements ready for processing (parent done, not started)."""
        with self._lock:
            ready = []
            for eid, element in self._elements.items():
                if eid in self._completed or eid in self._in_progress:
                    continue

                parent_id = self._parents.get(eid)
                if parent_id is None or parent_id in self._completed:
                    ready.append(element)
                    if len(ready) >= max_count:
                        break

            # Mark as in-progress
            for e in ready:
                self._in_progress.add(e.element_id)

            return ready

    def mark_complete(self, element_id: str) -> None:
        """Mark element as completed."""
        with self._lock:
            self._in_progress.discard(element_id)
            self._completed.add(element_id)

    def mark_failed(self, element_id: str) -> None:
        """Mark element as failed (won't block children)."""
        with self._lock:
            self._in_progress.discard(element_id)
            self._completed.add(element_id)  # Treat as done so children can proceed

    def is_complete(self) -> bool:
        """Check if all elements are processed."""
        with self._lock:
            return len(self._completed) == len(self._elements)

    def pending_count(self) -> int:
        """Count elements not yet completed."""
        with self._lock:
            return len(self._elements) - len(self._completed)
```

---

## Worker Implementation

```python
def _worker(
    self,
    worker_id: int,
    element: CodeElement,
    summary_cache: _SummaryCache,
    worker_status: WorkerStatus,
    timing_stats: TimingStats,
) -> ProcessedElement:
    """Process single element through full pipeline."""
    import time

    start_wall = time.time()
    api_time = 0.0

    try:
        # Step 1: Summarize
        worker_status.set(worker_id, element.name, "summarize")
        api_start = time.time()
        summary = self._summarize_element(element, summary_cache)
        api_time += time.time() - api_start
        summary_cache.add_summary(element.element_id, summary)

        # Step 2: Embed (if applicable)
        embedding: list[float] | None = None
        if should_embed(element):
            worker_status.set(worker_id, element.name, "embed")
            api_start = time.time()
            embedding = self._embed_element(element, summary_cache)
            api_time += time.time() - api_start

        # Step 3: Index
        worker_status.set(worker_id, element.name, "index")
        file_hash = self._file_hashes.get(element.relative_path)
        self._index_element(element, summary, embedding, file_hash)

        wall_time = time.time() - start_wall
        timing_stats.record(wall_time, api_time)
        worker_status.clear(worker_id)

        return ProcessedElement(
            element_id=element.element_id,
            success=True,
            wall_time=wall_time,
            api_time=api_time,
        )

    except Exception as e:
        wall_time = time.time() - start_wall
        worker_status.clear(worker_id)

        return ProcessedElement(
            element_id=element.element_id,
            success=False,
            wall_time=wall_time,
            api_time=api_time,
            error=str(e),
        )
```

### Orchestration

```python
def process_elements_parallel(
    self,
    parsed_files: list[ParsedFile],
    on_progress: Callable[[ProgressState], None] | None = None,
) -> ProcessingResult:
    """Process elements with parallel workers."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Collect all elements
    all_elements = [e for pf in parsed_files for e in pf.elements]

    # Initialize tracking
    tracker = DependencyTracker(all_elements)
    summary_cache = _SummaryCache()
    worker_status = WorkerStatus()
    timing_stats = TimingStats()
    timing_stats.phase_start = time.time()

    # Populate cache with elements for parent lookup
    for e in all_elements:
        summary_cache.add_element(e)

    result = ProcessingResult(...)

    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
        futures: dict[Future, CodeElement] = {}

        while not tracker.is_complete():
            # Submit ready elements
            ready = tracker.get_ready_elements(max_count=self.num_workers)
            for i, element in enumerate(ready):
                worker_id = (len(futures) % self.num_workers) + 1
                future = executor.submit(
                    self._worker, worker_id, element,
                    summary_cache, worker_status, timing_stats
                )
                futures[future] = element

            # Wait for at least one completion
            if futures:
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)

                for future in done:
                    element = futures.pop(future)
                    proc_result = future.result()

                    if proc_result.success:
                        tracker.mark_complete(element.element_id)
                        result.indexed += 1
                    else:
                        tracker.mark_failed(element.element_id)
                        result.failed_elements.append(
                            (element.element_id, proc_result.error)
                        )

                    result.elements_processed += 1

                    # Update progress
                    if on_progress:
                        state = ProgressState(
                            total=len(all_elements),
                            completed=result.elements_processed,
                            skipped=result.elements_skipped,
                            failed=len(result.failed_elements),
                            timing=timing_stats,
                            workers=worker_status,
                        )
                        on_progress(state)

    return result
```

---

## CLI Integration

### New Options

```python
@main.command()
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--user", "-u", required=True, help="Username/branch")
@click.option("--skip-ai", is_flag=True, help="Skip AI processing")
@click.option("--dry-run", is_flag=True, help="Use in-memory storage")
@click.option("--ollama-url", default=None, help="Ollama API URL")
@click.option("--workers", "-w", default=4, type=int, help="Number of parallel workers (default: 4)")
@click.option("--force-clean", is_flag=True, help="Delete all indexed data for this repo/user before parsing")
def parse(repo_path: str, user: str, skip_ai: bool, dry_run: bool,
          ollama_url: str | None, workers: int, force_clean: bool) -> None:
```

### Force Clean Implementation

```python
# After discovery, before change detection
if force_clean and not dry_run:
    console.print("[yellow]Force clean:[/] Deleting existing index data...")
    from magaldi.db.elasticsearch import ElasticsearchRepository
    es_repo = ElasticsearchRepository(config)
    deleted = es_repo.delete_by_repository(
        scope=discovery_result.scope,
        repository=discovery_result.repository,
        username=user,
    )
    console.print(f"  Deleted {deleted} documents")
```

### ProcessingConfig Update

```python
@dataclass
class ProcessingConfig:
    summarize_model: str = "qwen2.5-coder:7b"
    embed_model: str = "snowflake-arctic-embed2"
    ollama_url: str = "http://localhost:11434"
    skip_ai: bool = False
    num_workers: int = 4  # New field

    # ... rest unchanged
```

---

## New ES Method

```python
def delete_by_repository(self, scope: str, repository: str, username: str) -> int:
    """Delete all elements for a repository/user combination.

    Returns:
        Number of documents deleted.
    """
    client = self._get_client()
    response = client.delete_by_query(
        index=INDEX_NAME,
        body={
            "query": {
                "bool": {
                    "must": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                    ]
                }
            }
        },
        refresh=True,
    )
    return response.get("deleted", 0)
```

---

## Error Handling

- **Per-worker isolation:** Errors in one worker don't affect others
- **Failed elements:** Tracked separately, children still process (parent marked "done")
- **Ollama failures:** First connection error fails fast with clear message
- **Display failures:** At end, show first 10 failed elements with errors

```python
if result.failed_elements:
    console.print(f"\n[red]Failed elements ({len(result.failed_elements)}):[/]")
    for elem_id, error in result.failed_elements[:10]:
        console.print(f"  • {elem_id}: {error}")
    if len(result.failed_elements) > 10:
        console.print(f"  ... and {len(result.failed_elements) - 10} more")
```

---

## Implementation Tasks

### Task 1: Add TimingStats and WorkerStatus dataclasses

**Files:**
- Modify: `src/magaldi/processing/processor.py`

Add the thread-safe `TimingStats`, `WorkerStatus`, `ProgressState`, and `ProcessedElement` dataclasses at the top of the file.

### Task 2: Implement DependencyTracker

**Files:**
- Modify: `src/magaldi/processing/processor.py`

Add `DependencyTracker` class with `get_ready_elements()`, `mark_complete()`, `mark_failed()`, `is_complete()` methods.

### Task 3: Add delete_by_repository to ES

**Files:**
- Modify: `src/magaldi/db/elasticsearch.py`
- Test: `tests/integration/test_elasticsearch.py`

Add the `delete_by_repository` method and test it.

### Task 4: Refactor processor for parallel execution

**Files:**
- Modify: `src/magaldi/processing/processor.py`

Refactor `process_elements` to use ThreadPoolExecutor with DependencyTracker. Update `ProcessingConfig` to include `num_workers`. Update `ProcessingResult` to include `failed_elements`.

### Task 5: Update CLI with new options and Rich Live display

**Files:**
- Modify: `src/magaldi/cli.py`

Add `--workers` and `--force-clean` options. Implement Rich Live display with worker status table. Add force-clean logic before change detection.

### Task 6: E2E Testing

**Files:**
- Test: `tests/integration/test_cli_e2e.py`

Test parallel processing, force-clean option, and verify correct output.
