# Watch Command

## Overview

A new CLI command `magaldi watch` that continuously monitors a repository for file changes and automatically re-parses modified files. This enables a "hot reload" development experience - no need to manually re-run `parse` after each code change.

## User Experience

```bash
# Basic usage - watch current directory
magaldi watch . --user main

# With parse flags
magaldi watch /path/to/repo --user dev --skip-ai --features

# Watch with debounce (wait for changes to settle)
magaldi watch . --user main --debounce 2.0
```

## Flags

### Inherited from `parse` (where they make sense):
| Flag | Include | Reason |
|------|---------|--------|
| `--user` | ✅ Yes | Required for indexing |
| `--skip-ai` | ✅ Yes | May want fast parsing without AI |
| `--skip-tests` | ✅ Yes | Filter test files |
| `--skip-resolve` | ✅ Yes | Skip call resolution for speed |
| `--llm-url` | ✅ Yes | Custom LLM endpoint |
| `--workers` | ✅ Yes | Parallel processing |
| `--features` | ✅ Yes | Run feature extraction after changes |
| `--glossary` | ✅ Yes | Run glossary extraction after changes |
| `--dry-run` | ❌ No | Watch mode requires persistence |
| `--force-clean` | ❌ No | Only makes sense for initial parse |

### New flags for `watch`:
| Flag | Default | Description |
|------|---------|-------------|
| `--debounce` | 1.0 | Seconds to wait after last change before processing |
| `--initial-scan` | True | Run change detection on startup |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Watch Command                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   Watchdog   │────▶│ Change Queue │────▶│   Processor  │    │
│  │   Observer   │     │  (debounced) │     │    Thread    │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│         │                                          │            │
│         │ File events                              │            │
│         ▼                                          ▼            │
│  ┌──────────────┐                         ┌──────────────┐     │
│  │   Filter:    │                         │  Run Parse   │     │
│  │ - Extensions │                         │  Pipeline:   │     │
│  │ - Excludes   │                         │ - Detection  │     │
│  │ - .git, etc  │                         │ - Parsing    │     │
│  └──────────────┘                         │ - Processing │     │
│                                           └──────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation

### 1. Add `watchdog` dependency

```toml
# pyproject.toml
dependencies = [
    ...
    "watchdog>=4.0.0",
]
```

### 2. Create `src/shared/cli/watch.py`

```python
"""Watch command - continuous file monitoring and parsing."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import click
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from shared.cli._shared import console, main
from shared.config import load_config

if TYPE_CHECKING:
    from magaldi_core.discovery import DiscoveryResult


class MagaldiFileHandler(FileSystemEventHandler):
    """Handle file system events and queue changes."""

    def __init__(
        self,
        change_queue: queue.Queue,
        discovery_result: DiscoveryResult,
        supported_extensions: set[str],
    ):
        self.change_queue = change_queue
        self.discovery_result = discovery_result
        self.supported_extensions = supported_extensions
        self.repo_path = discovery_result.repo_path

    def _should_process(self, path: str) -> bool:
        """Check if file should be processed."""
        rel_path = Path(path).relative_to(self.repo_path)

        # Check extension
        if Path(path).suffix not in self.supported_extensions:
            return False

        # Check excluded directories
        for part in rel_path.parts:
            for exclude in self.discovery_result.exclude_directories:
                if fnmatch.fnmatch(part, exclude):
                    return False

        # Check excluded files
        for exclude in self.discovery_result.exclude_files:
            if fnmatch.fnmatch(rel_path.name, exclude):
                return False

        return True

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._should_process(event.src_path):
            self.change_queue.put(("modified", event.src_path))

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._should_process(event.src_path):
            self.change_queue.put(("created", event.src_path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._should_process(event.src_path):
            self.change_queue.put(("deleted", event.src_path))


def process_changes(
    changed_files: set[str],
    deleted_files: set[str],
    discovery_result: DiscoveryResult,
    config: MagaldiConfig,
    options: dict,
) -> None:
    """Process accumulated file changes."""
    # Build a minimal ChangeManifest from the changed files
    # Run through parsing and processing pipeline
    # Similar to parse command but only for changed files
    ...


@main.command()
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False))
@click.option("--user", "-u", required=True, help="Username/branch")
@click.option("--skip-ai", is_flag=True, help="Skip AI processing")
@click.option("--features", is_flag=True, help="Run feature extraction")
@click.option("--glossary", is_flag=True, help="Run glossary extraction")
@click.option("--skip-tests", is_flag=True, help="Skip test files")
@click.option("--skip-resolve", is_flag=True, help="Skip call resolution")
@click.option("--llm-url", default=None, help="LLM API URL")
@click.option("--workers", "-w", default=0, type=int, help="Max workers")
@click.option("--debounce", default=1.0, type=float, help="Debounce seconds")
@click.option("--no-initial-scan", is_flag=True, help="Skip initial scan")
def watch(
    repo_path: str,
    user: str,
    skip_ai: bool,
    features: bool,
    glossary: bool,
    skip_tests: bool,
    skip_resolve: bool,
    llm_url: str | None,
    workers: int,
    debounce: float,
    no_initial_scan: bool,
) -> None:
    """Watch a repository for changes and auto-parse."""
    ...
```

### 3. Processing Loop

```python
def watch_loop(
    change_queue: queue.Queue,
    discovery_result: DiscoveryResult,
    config: MagaldiConfig,
    options: dict,
    debounce: float,
) -> None:
    """Main watch loop - collect changes and process in batches."""

    changed_files: set[str] = set()
    deleted_files: set[str] = set()
    last_change_time = 0.0

    while True:
        try:
            # Wait for changes (with timeout for debounce check)
            event_type, path = change_queue.get(timeout=0.1)

            if event_type == "deleted":
                deleted_files.add(path)
                changed_files.discard(path)
            else:
                changed_files.add(path)
                deleted_files.discard(path)

            last_change_time = time.time()

        except queue.Empty:
            # Check if debounce period has passed
            if changed_files or deleted_files:
                if time.time() - last_change_time >= debounce:
                    # Process batch
                    console.print(f"\n[cyan]Processing {len(changed_files)} changed, {len(deleted_files)} deleted files...[/]")

                    process_changes(
                        changed_files.copy(),
                        deleted_files.copy(),
                        discovery_result,
                        config,
                        options,
                    )

                    changed_files.clear()
                    deleted_files.clear()

                    console.print("[green]✓ Ready, watching for changes...[/]")
```

## Files to Create/Modify

1. **`pyproject.toml`** - Add `watchdog>=4.0.0` dependency
2. **`src/shared/cli/watch.py`** - New watch command implementation
3. **`src/shared/cli/__init__.py`** - Import watch command
4. **`src/magaldi_core/change_detection.py`** - Add helper to create manifest from file list

## Output Example

```
$ magaldi watch . --user main

[bold blue]Magaldi Watch Mode[/]
  Repository: magaldi/magaldi
  User: main
  Watching: /Users/dev/code/magaldi

[bold blue]Initial Scan[/]
  Discovered 156 files
  No changes detected

[green]✓ Ready, watching for changes...[/]

[dim]14:32:15[/] Modified: src/shared/cli/watch.py
[dim]14:32:16[/] Modified: src/shared/cli/__init__.py

[cyan]Processing 2 changed files...[/]
  Phase 3: Parsing
    Parsed 2 files, 15 elements
  Phase 4: Processing
    Processed 15 elements (2.3s)

[green]✓ Ready, watching for changes...[/]

^C
[yellow]Stopping watch...[/]
```

## Edge Cases

1. **Rapid changes** - Debounce prevents processing on every keystroke
2. **Large batches** - Process all accumulated changes together
3. **Deleted then recreated** - Handle as modification
4. **Git operations** - Ignore .git directory changes
5. **Build outputs** - Respect exclude patterns from discovery
6. **Keyboard interrupt** - Clean shutdown of observer

## Testing

1. Unit tests for `MagaldiFileHandler._should_process()`
2. Integration test with temporary directory and file modifications
3. Test debounce behavior
4. Test batch processing of multiple files
