# Atomic Element Processing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor processing so each element is fully processed (summarize → embed → index) before moving to the next, ensuring ES presence means completion.

**Architecture:** Replace separate phases 4/5/6 with unified processing that handles elements level-by-level (files→classes→methods→variables). Only index to ES after summarization and embedding complete. Check ES for existing elements to enable resumable runs.

**Tech Stack:** Python, Elasticsearch, Ollama (qwen2.5-coder:7b for summarization, snowflake-arctic-embed2 for embeddings)

---

### Task 1: Add Element Completion Check to ES Repository

**Files:**
- Modify: `src/magaldi/db/elasticsearch.py`

**Step 1: Add `element_exists` method**

Add this method to `ElasticsearchRepository` class after `get_document`:

```python
def element_exists(self, element_id: str) -> bool:
    """Check if element exists in ES (meaning it's fully processed).

    Args:
        element_id: Element ID to check.

    Returns:
        True if element exists (fully processed).
    """
    try:
        client = self._get_client()
        return client.exists(index=INDEX_NAME, id=element_id)
    except Exception:
        return False
```

**Step 2: Add `get_existing_element_ids` for batch checking**

```python
def get_existing_element_ids(self, element_ids: list[str]) -> set[str]:
    """Check which elements already exist in ES.

    Args:
        element_ids: List of element IDs to check.

    Returns:
        Set of element IDs that exist.
    """
    if not element_ids:
        return set()

    client = self._get_client()

    # Use mget for efficient batch lookup
    response = client.mget(index=INDEX_NAME, ids=element_ids, _source=False)

    return {
        doc["_id"] for doc in response["docs"] if doc.get("found", False)
    }
```

**Step 3: Run existing tests**

Run: `pytest tests/ -v -k elasticsearch`
Expected: All existing tests pass

**Step 4: Commit**

```bash
git add src/magaldi/db/elasticsearch.py
git commit -m "feat: add element existence check methods to ES repository"
```

---

### Task 2: Create Unified Element Processor Module

**Files:**
- Create: `src/magaldi/processing/__init__.py`
- Create: `src/magaldi/processing/processor.py`

**Step 1: Create package init**

```python
"""Unified element processing - summarize, embed, index."""

from magaldi.processing.processor import (
    ProcessingConfig,
    ProcessingResult,
    process_elements,
)

__all__ = ["ProcessingConfig", "ProcessingResult", "process_elements"]
```

**Step 2: Create processor module with data classes**

```python
"""Unified element processor - atomic summarize → embed → index flow.

Processes elements level-by-level:
- Level 0: Files
- Level 1: Classes
- Level 2: Functions/Methods
- Level 3: Variables

Only indexes to ES after full processing, ensuring ES presence = completion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from magaldi.config import MagaldiConfig
from magaldi.db.elasticsearch import ElasticsearchRepository
from magaldi.embedding.embedding import EmbeddingConfig, OllamaEmbedClient
from magaldi.parser.code_parser import CodeElement, ParsedFile
from magaldi.summarization.summarization import OllamaClient, SummarizationConfig


@dataclass
class ProcessingConfig:
    """Configuration for unified processing."""

    summarize_model: str = "qwen2.5-coder:7b"
    embed_model: str = "snowflake-arctic-embed2"
    ollama_url: str = "http://localhost:11434"
    skip_ai: bool = False


@dataclass
class ProcessingResult:
    """Result of unified processing."""

    scope: str
    repository: str
    username: str

    # Counts
    elements_processed: int = 0
    elements_skipped: int = 0  # Already in ES
    elements_failed: int = 0

    # By type
    summarized: int = 0
    embedded: int = 0
    indexed: int = 0

    # Errors
    errors: list[str] = field(default_factory=list)
```

**Step 3: Add helper to determine if element needs embedding**

```python
def should_embed(element: CodeElement) -> bool:
    """Determine if element should be embedded.

    Args:
        element: Code element to check.

    Returns:
        True if element should be embedded.
    """
    # Always embed files, classes, functions, methods
    if element.element_type in ("file", "class", "function", "method"):
        return True

    # For variables: only embed significant ones
    if element.element_type == "variable":
        name = element.name
        # Embed uppercase constants
        if name.isupper():
            return True
        # Embed if has docstring
        if element.docstring:
            return True

    return False
```

**Step 4: Add main processing function signature**

```python
def process_elements(
    parsed_files: list[ParsedFile],
    config: ProcessingConfig,
    es_repo: ElasticsearchRepository,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> ProcessingResult:
    """Process all elements: summarize → embed → index.

    Processes elements level-by-level to ensure parent summaries
    are available when processing children.

    Args:
        parsed_files: Parsed files from Phase 3.
        config: Processing configuration.
        es_repo: Elasticsearch repository for indexing.
        on_progress: Optional callback(completed, total, current_element_name).

    Returns:
        ProcessingResult with counts and errors.
    """
    # Collect all elements
    all_elements: list[CodeElement] = []
    file_hashes: dict[str, str] = {}  # element_id -> file_hash for file elements

    for pf in parsed_files:
        file_hashes[pf.elements[0].element_id] = pf.file_info.hash if pf.elements else ""
        all_elements.extend(pf.elements)

    # Check which elements already exist in ES
    element_ids = [e.element_id for e in all_elements]
    existing_ids = es_repo.get_existing_element_ids(element_ids)

    # Filter to only process new elements
    elements_to_process = [e for e in all_elements if e.element_id not in existing_ids]

    result = ProcessingResult(
        scope=all_elements[0].scope if all_elements else "",
        repository=all_elements[0].repository if all_elements else "",
        username=all_elements[0].username if all_elements else "",
        elements_skipped=len(existing_ids),
    )

    if not elements_to_process:
        return result

    # Group by level for hierarchical processing
    by_level: dict[int, list[CodeElement]] = {}
    for elem in elements_to_process:
        by_level.setdefault(elem.level, []).append(elem)

    # Initialize AI clients if not skipping
    ollama_summarize = None
    ollama_embed = None
    summary_cache: dict[str, str] = {}  # element_id -> summary

    if not config.skip_ai:
        sum_config = SummarizationConfig(
            ollama_url=config.ollama_url,
            model=config.summarize_model,
        )
        emb_config = EmbeddingConfig(
            ollama_url=config.ollama_url,
            model=config.embed_model,
        )
        ollama_summarize = OllamaClient(sum_config.ollama_url, sum_config.model)
        ollama_embed = OllamaEmbedClient(emb_config.ollama_url, emb_config.model)

    total = len(elements_to_process)
    completed = 0

    # Process level by level (0, 1, 2, 3)
    for level in sorted(by_level.keys()):
        for element in by_level[level]:
            try:
                name = element.name
                if on_progress:
                    on_progress(completed, total, name)

                summary = None
                embedding = None

                # Step 1: Summarize (if AI enabled)
                if ollama_summarize and not config.skip_ai:
                    summary = _summarize_element(
                        element, ollama_summarize, sum_config, summary_cache
                    )
                    if summary:
                        summary_cache[element.element_id] = summary
                        result.summarized += 1

                # Step 2: Embed (if AI enabled and element needs it)
                if ollama_embed and not config.skip_ai and should_embed(element):
                    embedding = _embed_element(
                        element, ollama_embed, emb_config, summary, summary_cache
                    )
                    if embedding:
                        result.embedded += 1

                # Step 3: Index to ES (only after summarize + embed complete)
                file_hash = file_hashes.get(element.element_id)
                _index_element(element, es_repo, summary, embedding, file_hash)
                result.indexed += 1
                result.elements_processed += 1

            except Exception as e:
                result.errors.append(f"{element.element_id}: {e}")
                result.elements_failed += 1

            completed += 1
            if on_progress:
                on_progress(completed, total, "")

    return result
```

**Step 5: Add summarization helper**

```python
def _summarize_element(
    element: CodeElement,
    client: OllamaClient,
    config: SummarizationConfig,
    summary_cache: dict[str, str],
) -> str | None:
    """Generate summary for an element.

    Args:
        element: Element to summarize.
        client: Ollama client.
        config: Summarization config.
        summary_cache: Cache of parent summaries.

    Returns:
        Summary text or None if failed.
    """
    from magaldi.summarization.summarization import build_prompt

    # Get parent summary for context
    parent_summary = None
    if element.parent_id and element.parent_id in summary_cache:
        parent_summary = summary_cache[element.parent_id]

    prompt = build_prompt(element, parent_summary)

    try:
        response = client.generate(prompt, config)
        return response.strip() if response else None
    except Exception:
        return None
```

**Step 6: Add embedding helper**

```python
def _embed_element(
    element: CodeElement,
    client: OllamaEmbedClient,
    config: EmbeddingConfig,
    summary: str | None,
    summary_cache: dict[str, str],
) -> list[float] | None:
    """Generate embedding for an element.

    Args:
        element: Element to embed.
        client: Ollama embedding client.
        config: Embedding config.
        summary: Element's summary (if available).
        summary_cache: Cache of parent summaries.

    Returns:
        Embedding vector or None if failed.
    """
    from magaldi.embedding.embedding import build_embedding_context

    # Get parent summaries for context
    parent_summaries = {}
    if element.parent_id and element.parent_id in summary_cache:
        parent_summaries["parent"] = summary_cache[element.parent_id]

    context = build_embedding_context(element, summary, parent_summaries)

    try:
        embedding = client.embed(context, config)
        return embedding if embedding else None
    except Exception:
        return None
```

**Step 7: Add indexing helper**

```python
def _index_element(
    element: CodeElement,
    es_repo: ElasticsearchRepository,
    summary: str | None,
    embedding: list[float] | None,
    file_hash: str | None,
) -> None:
    """Index element to ES with summary and embedding.

    Args:
        element: Element to index.
        es_repo: Elasticsearch repository.
        summary: Summary text (optional).
        embedding: Embedding vector (optional).
        file_hash: File hash for file-level elements.
    """
    # First index the base element
    es_repo.index_element(element, file_hash=file_hash)

    # Then update with summary if available
    if summary:
        es_repo.store_summary(element.element_id, summary)

    # Then update with embedding if available
    if embedding:
        es_repo.store_embedding(element.element_id, embedding)
```

**Step 8: Commit**

```bash
git add src/magaldi/processing/
git commit -m "feat: add unified element processor module"
```

---

### Task 3: Update CLI to Use Unified Processing

**Files:**
- Modify: `src/magaldi/cli.py`

**Step 1: Add import for new processor**

Add to imports section:

```python
from magaldi.processing.processor import (
    ProcessingConfig,
    process_elements,
)
```

**Step 2: Replace phases 4, 5, 6 with unified processing**

Replace the `run_storage`, `run_summarization`, and `run_embedding` functions and update the parse command to use unified processing:

```python
def run_processing(
    parsing_result: ParsingResult,
    manifest: ChangeManifest,
    config: MagaldiConfig,
    dry_run: bool,
    skip_ai: bool,
) -> tuple[int, int, int]:
    """Run unified processing: summarize → embed → index.

    Returns:
        Tuple of (processed, skipped, indexed).
    """
    if dry_run:
        # Dry run: just count what would be processed
        total = parsing_result.total_elements
        console.print(f"  [dim]Dry run: would process {total} elements[/]")
        return (0, 0, 0)

    from magaldi.db.elasticsearch import ElasticsearchRepository

    es_repo = ElasticsearchRepository(config)

    proc_config = ProcessingConfig(
        summarize_model=config.ollama.summarize_model,
        embed_model=config.ollama.embed_model,
        ollama_url=config.ollama.url,
        skip_ai=skip_ai,
    )

    total = parsing_result.total_elements

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Processing[/]"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("processing", total=total)

        def on_progress(completed: int, total: int, name: str) -> None:
            progress.update(task, completed=completed, total=total)

        result = process_elements(
            parsing_result.parsed_files,
            proc_config,
            es_repo,
            on_progress,
        )

    return (result.elements_processed, result.elements_skipped, result.indexed)
```

**Step 3: Update parse command flow**

Modify the `parse` function to use unified processing:

```python
@main.command()
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--user", "-u", required=True, help="Username/branch (use 'main' for primary parse)")
@click.option("--skip-ai", is_flag=True, help="Skip summarization and embedding (index only)")
@click.option("--dry-run", is_flag=True, help="Parse only, don't store to ES")
@click.option("--ollama-url", default=None, help="Ollama API URL (default: from config)")
def parse(
    repo_path: str, user: str, skip_ai: bool, dry_run: bool, ollama_url: str | None
) -> None:
    """Parse a repository and index its code elements."""
    config = load_config(skip_validation=dry_run)
    if ollama_url:
        config.ollama.url = ollama_url

    if dry_run:
        console.print("[yellow]Dry run mode:[/] Parse only, no ES storage\n")

    try:
        # Phase 1: Discovery
        console.print("[bold blue]Phase 1:[/] Discovery")
        discovery_result = run_discovery(repo_path, user)
        print_discovery_result(discovery_result)

        # Phase 2: Change Detection
        console.print("\n[bold blue]Phase 2:[/] Change Detection")
        manifest = run_change_detection(discovery_result, config, dry_run)
        print_change_manifest(manifest)

        if manifest.files_to_parse == 0:
            console.print("\n[green]No files to parse. Repository is up to date.[/]")
            return

        # Phase 3: Parsing
        console.print("\n[bold blue]Phase 3:[/] Parsing")
        parsing_result = run_parsing(manifest)
        print_parsing_result(parsing_result)

        # Phase 4: Unified Processing (summarize → embed → index)
        console.print("\n[bold blue]Phase 4:[/] Processing")
        processed, skipped, indexed = run_processing(
            parsing_result, manifest, config, dry_run, skip_ai
        )
        print_processing_result(processed, skipped, indexed, skip_ai)

        print_summary(discovery_result, manifest, processed, skipped, indexed)

    except DiscoveryError as e:
        console.print(f"\n[red]Discovery error:[/] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error:[/] {e}")
        if "--dry-run" not in sys.argv:
            console.print("[dim]Hint: Use --dry-run to test without ES[/]")
        sys.exit(1)
```

**Step 4: Add new print function for processing result**

```python
def print_processing_result(processed: int, skipped: int, indexed: int, skip_ai: bool) -> None:
    """Print processing phase results."""
    parts = []
    if processed:
        parts.append(f"[green]{processed} processed[/]")
    if skipped:
        parts.append(f"[dim]{skipped} skipped (already in ES)[/]")
    if indexed:
        parts.append(f"{indexed} indexed")
    if skip_ai:
        parts.append("[yellow]AI skipped[/]")
    console.print(f"  {' | '.join(parts)}")
```

**Step 5: Update print_summary function**

```python
def print_summary(
    discovery: DiscoveryResult,
    manifest: ChangeManifest,
    processed: int,
    skipped: int,
    indexed: int,
) -> None:
    """Print final summary."""
    console.print("\n" + "=" * 60)
    console.print("[bold green]Parse Complete[/]")
    console.print("=" * 60)

    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Repository", f"{discovery.scope}/{discovery.repository}")
    table.add_row("User", discovery.username)
    table.add_row("Files parsed", str(manifest.files_to_parse))
    table.add_row("Elements processed", str(processed))
    if skipped:
        table.add_row("Elements skipped", str(skipped))
    table.add_row("Elements indexed", str(indexed))

    console.print(table)
    console.print()
```

**Step 6: Remove old phase 5/6 runner functions**

Delete these functions:
- `run_storage`
- `run_summarization`
- `run_embedding`
- `print_storage_result`

**Step 7: Commit**

```bash
git add src/magaldi/cli.py
git commit -m "refactor: replace phases 4-6 with unified atomic processing"
```

---

### Task 4: Handle Deletions in Change Detection

**Files:**
- Modify: `src/magaldi/db/elasticsearch.py`

**Step 1: Add method to delete elements by file**

This already exists but verify it's called during change detection for deleted/modified files:

```python
def delete_elements_by_file(
    self, scope: str, repository: str, username: str, relative_path: str
) -> int:
    """Delete all elements for a file from ES.

    Args:
        scope: Scope.
        repository: Repository.
        username: Username.
        relative_path: File path.

    Returns:
        Count of deleted elements.
    """
    return self.delete_by_file(scope, repository, username, relative_path)
```

**Step 2: Update processing to handle deletions**

Add to `process_elements` at the start:

```python
# Handle deletions first - remove old elements for modified files
for pf in parsed_files:
    # Delete existing elements for this file before reprocessing
    es_repo.delete_by_file(
        pf.elements[0].scope if pf.elements else "",
        pf.elements[0].repository if pf.elements else "",
        pf.elements[0].username if pf.elements else "",
        pf.file_info.relative_path,
    )
```

**Step 3: Commit**

```bash
git add src/magaldi/db/elasticsearch.py src/magaldi/processing/processor.py
git commit -m "fix: handle file deletions before reprocessing"
```

---

### Task 5: Test End-to-End Flow

**Step 1: Clear ES index for fresh test**

```bash
curl -X DELETE http://localhost:9200/magaldi-code-elements
```

**Step 2: Run full parse**

```bash
.venv/bin/magaldi parse . --user main
```

Expected: All elements processed, none skipped

**Step 3: Run again without changes**

```bash
.venv/bin/magaldi parse . --user main
```

Expected: "No files to parse. Repository is up to date."

**Step 4: Modify a file and run again**

```bash
echo "# test" >> src/magaldi/cli.py
.venv/bin/magaldi parse . --user main
```

Expected: Only cli.py elements reprocessed

**Step 5: Interrupt mid-run and resume**

```bash
# Start parse, Ctrl+C during processing
.venv/bin/magaldi parse . --user main
# ^C

# Run again
.venv/bin/magaldi parse . --user main
```

Expected: Picks up where it left off, shows "X skipped (already in ES)"

**Step 6: Revert test change and commit**

```bash
git checkout src/magaldi/cli.py
git add -A
git commit -m "test: verify atomic processing flow"
```

---

## Summary

This plan converts the sequential phase-based approach to atomic per-element processing:

1. **ES presence = completion** - Only index after summarize + embed done
2. **Level-ordered processing** - Files → Classes → Methods → Variables
3. **Resumable** - Check ES for existing elements, skip completed ones
4. **Handles interruption** - Ctrl+C leaves no partial state in ES
