"""File and element selection for benchmarks.

This module contains functions for creating file manifests and selecting
files/elements for benchmarking.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from magaldi_core.change_detection import ChangeManifest
    from magaldi_core.code_parser import ParsingResult
    from magaldi_core.discovery import DiscoveryResult


def create_full_manifest(discovery_result: DiscoveryResult) -> ChangeManifest:
    """Create a ChangeManifest with all discovered files as 'new'.

    This skips change detection and treats all files as new.
    """
    import hashlib
    import os
    from datetime import datetime as dt
    from pathlib import Path

    from magaldi_core.change_detection import ChangeManifest, FileInfo
    from magaldi_core.discovery import SUPPORTED_EXTENSIONS, _is_excluded_dir, _is_excluded_file

    new_files = []
    repo_path = discovery_result.repo_path

    for root, dirs, files in os.walk(repo_path):
        # Filter directories
        dirs[:] = [
            d for d in dirs
            if not _is_excluded_dir(
                d,
                discovery_result.exclude_directories + ["node_modules", ".git", "__pycache__", ".venv", "venv"]
            )
        ]

        for file_name in files:
            abs_path = Path(root) / file_name
            rel_path = str(abs_path.relative_to(repo_path))

            # Check extension
            ext = abs_path.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            # Check exclusions
            if _is_excluded_file(file_name, discovery_result.exclude_files):
                continue

            # Compute hash
            try:
                with open(abs_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
            except Exception:
                continue

            new_files.append(FileInfo(
                relative_path=rel_path,
                absolute_path=abs_path,
                language=SUPPORTED_EXTENSIONS[ext],
                hash=file_hash,
            ))

    return ChangeManifest(
        scope=discovery_result.scope,
        repository=discovery_result.repository,
        username=discovery_result.username,
        timestamp=dt.now(),
        total_files_scanned=len(new_files),
        new_files=new_files,
        modified_files=[],
        deleted_files=[],
        unchanged_count=0,
        skipped_count=0,
    )


def select_benchmark_files(
    parsing_result: ParsingResult,
    forced_path: str | None,
    num_files: int = 5,
    max_per_type: int = 10,
) -> list[dict] | None:
    """Select files for benchmarking.

    Args:
        parsing_result: Result from parsing phase.
        forced_path: Specific file path to use, or None for random selection.
        num_files: Number of files to select (default 5).
        max_per_type: Maximum elements per type per file (default 10).

    Returns:
        List of dicts with 'path' and 'elements' keys, or None if no valid files found.
    """
    import random

    def cap_elements_by_type(elements: list, max_per_type: int) -> list:
        """Cap elements to max_per_type per element_type per file."""
        by_type: dict[str, list] = defaultdict(list)
        for elem in elements:
            by_type[elem.element_type].append(elem)

        result = []
        for _elem_type, type_elements in by_type.items():
            if len(type_elements) > max_per_type:
                # Randomly sample to get variety
                result.extend(random.sample(type_elements, max_per_type))
            else:
                result.extend(type_elements)

        # Sort by line number to maintain order
        result.sort(key=lambda e: e.line_start)
        return result

    parsed_files = parsing_result.parsed_files

    if forced_path:
        # Find the specific file (single file mode)
        for pf in parsed_files:
            path = pf.file_info.relative_path
            if path == forced_path or path.endswith(forced_path):
                elements = cap_elements_by_type(list(pf.elements), max_per_type)
                return [{
                    "path": path,
                    "elements": elements,
                }]
        return None

    # Random selection: prefer files with 3+ elements
    candidates = [
        pf for pf in parsed_files
        if len(pf.elements) >= 3
    ]

    if not candidates:
        # Fall back to any file with elements
        candidates = [pf for pf in parsed_files if len(pf.elements) > 0]

    if not candidates:
        return None

    # Select up to num_files randomly
    selected_files = random.sample(candidates, min(num_files, len(candidates)))

    result = []
    for pf in selected_files:
        elements = cap_elements_by_type(list(pf.elements), max_per_type)
        result.append({
            "path": pf.file_info.relative_path,
            "elements": elements,
        })

    return result
