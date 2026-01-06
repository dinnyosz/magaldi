"""Phase 2: Change Detection - Hash computation and diff logic.

This module identifies which files have changed since the last parse:
1. Enumerate files (reuses discovery logic)
2. Compute SHA256 hashes (parallel)
3. Compare to database state
4. Generate change manifest
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

from magaldi.config import get_config
from magaldi.parser.discovery import (
    SUPPORTED_EXTENSIONS,
    DiscoveryResult,
    RepoConfig,
    _is_excluded_dir,
    _is_excluded_file,
)


class ChangeDetectionError(Exception):
    """Raised when change detection fails."""

    pass


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class FileInfo:
    """Information about a single file."""

    relative_path: str
    absolute_path: Path
    language: str
    hash: str | None = None
    previous_hash: str | None = None
    file_size: int = 0
    line_count: int = 0


@dataclass
class ChangeManifest:
    """Result of change detection phase."""

    # Context
    scope: str
    repository: str
    username: str
    timestamp: datetime

    # Counts
    total_files_scanned: int

    # File lists
    new_files: list[FileInfo] = field(default_factory=list)
    modified_files: list[FileInfo] = field(default_factory=list)
    deleted_files: list[FileInfo] = field(default_factory=list)
    unchanged_count: int = 0
    skipped_count: int = 0  # User branch: same as main

    @property
    def files_to_parse(self) -> int:
        """Number of files that need parsing."""
        return len(self.new_files) + len(self.modified_files)

    @property
    def files_to_remove(self) -> int:
        """Number of files to mark as deleted."""
        return len(self.deleted_files)


# =============================================================================
# DATABASE INTERFACE (Protocol for dependency injection)
# =============================================================================


@dataclass
class DBFileState:
    """Database record for a file state."""

    relative_path: str
    file_hash: str | None
    is_deleted: bool = False


class FileStateRepository(Protocol):
    """Interface for file state database operations."""

    def get_file_states(
        self, scope: str, repository: str, username: str
    ) -> dict[str, DBFileState]:
        """Get all file states for a scope/repo/user as a dict keyed by relative_path."""
        ...

    def main_branch_exists(self, scope: str, repository: str) -> bool:
        """Check if main branch has been parsed."""
        ...


# =============================================================================
# IN-MEMORY IMPLEMENTATION (for testing / standalone use)
# =============================================================================


class InMemoryFileStateRepository:
    """In-memory implementation for testing."""

    def __init__(self) -> None:
        self._states: dict[tuple[str, str, str], dict[str, DBFileState]] = {}

    def get_file_states(
        self, scope: str, repository: str, username: str
    ) -> dict[str, DBFileState]:
        key = (scope, repository, username)
        return self._states.get(key, {})

    def set_file_states(
        self, scope: str, repository: str, username: str, states: dict[str, DBFileState]
    ) -> None:
        key = (scope, repository, username)
        self._states[key] = states

    def main_branch_exists(self, scope: str, repository: str) -> bool:
        key = (scope, repository, "main")
        return key in self._states and len(self._states[key]) > 0


# =============================================================================
# 2.1 FILE ENUMERATION
# =============================================================================


def enumerate_files(
    repo_path: Path,
    exclude_directories: list[str],
    exclude_files: list[str],
) -> list[FileInfo]:
    """Enumerate all parseable files in repository.

    Args:
        repo_path: Path to repository root.
        exclude_directories: Directory names to exclude.
        exclude_files: File patterns to exclude.

    Returns:
        List of FileInfo objects for each parseable file.
    """
    files: list[FileInfo] = []

    for root, dirs, filenames in os.walk(repo_path):
        root_path = Path(root)

        # Filter out excluded directories (modifying in-place prevents descent)
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and not _is_excluded_dir(d, exclude_directories)
        ]

        for filename in filenames:
            # Skip hidden files
            if filename.startswith("."):
                continue

            # Skip excluded file patterns
            if _is_excluded_file(filename, exclude_files):
                continue

            file_path = root_path / filename

            # Skip symlinks
            if file_path.is_symlink():
                continue

            # Check extension
            ext = file_path.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            relative_path = file_path.relative_to(repo_path)
            language = SUPPORTED_EXTENSIONS[ext]

            files.append(
                FileInfo(
                    relative_path=str(relative_path),
                    absolute_path=file_path,
                    language=language,
                )
            )

    return files


# =============================================================================
# 2.2 HASH COMPUTATION
# =============================================================================


def compute_file_hash(file_path: Path, chunk_size: int = 65536) -> str | None:
    """Compute SHA256 hash of a file.

    Args:
        file_path: Path to file.
        chunk_size: Size of chunks to read (default 64KB).

    Returns:
        Hex string of SHA256 hash, or None if file unreadable.
    """
    sha256 = hashlib.sha256()

    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except (OSError, IOError):
        return None


def compute_hashes_parallel(
    files: list[FileInfo],
    max_workers: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[FileInfo]:
    """Compute hashes for multiple files in parallel.

    Args:
        files: List of FileInfo objects to hash.
        max_workers: Maximum worker threads (default from config or 4).
        on_progress: Optional callback(completed, total) for progress updates.

    Returns:
        Same list with hash field populated.
    """
    if max_workers is None:
        try:
            config = get_config()
            max_workers = config.parser.hash_workers
        except RuntimeError:
            max_workers = 4

    if not files:
        return files

    def hash_file(file_info: FileInfo) -> FileInfo:
        file_info.hash = compute_file_hash(file_info.absolute_path)
        try:
            file_info.file_size = file_info.absolute_path.stat().st_size
            with open(file_info.absolute_path, "r", encoding="utf-8", errors="replace") as f:
                file_info.line_count = sum(1 for _ in f)
        except (OSError, IOError):
            pass
        return file_info

    total = len(files)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(hash_file, f): f for f in files}
        for future in as_completed(futures):
            completed += 1
            if on_progress:
                on_progress(completed, total)

    return files


# =============================================================================
# 2.3 COMPARE LOGIC
# =============================================================================


def compare_main_branch(
    files: list[FileInfo],
    db_states: dict[str, DBFileState],
) -> tuple[list[FileInfo], list[FileInfo], list[FileInfo], int]:
    """Compare files against main branch database state.

    Args:
        files: List of local files with hashes.
        db_states: Database file states keyed by relative_path.

    Returns:
        Tuple of (new_files, modified_files, deleted_files, unchanged_count).
    """
    new_files: list[FileInfo] = []
    modified_files: list[FileInfo] = []
    unchanged_count = 0

    local_paths = set()

    for file_info in files:
        local_paths.add(file_info.relative_path)
        db_state = db_states.get(file_info.relative_path)

        if db_state is None:
            # New file
            new_files.append(file_info)
        elif db_state.file_hash == file_info.hash:
            # Unchanged
            unchanged_count += 1
        else:
            # Modified
            file_info.previous_hash = db_state.file_hash
            modified_files.append(file_info)

    # Find deleted files (in DB but not on disk)
    deleted_files: list[FileInfo] = []
    for rel_path, db_state in db_states.items():
        if rel_path not in local_paths and not db_state.is_deleted:
            deleted_files.append(
                FileInfo(
                    relative_path=rel_path,
                    absolute_path=Path(),  # No path for deleted
                    language="",  # Unknown for deleted
                    hash=None,
                    previous_hash=db_state.file_hash,
                )
            )

    return new_files, modified_files, deleted_files, unchanged_count


def compare_user_branch(
    files: list[FileInfo],
    main_states: dict[str, DBFileState],
    user_states: dict[str, DBFileState],
) -> tuple[list[FileInfo], list[FileInfo], list[FileInfo], int, int]:
    """Compare files for user branch against main and previous user state.

    Args:
        files: List of local files with hashes.
        main_states: Main branch file states.
        user_states: User's previous file states.

    Returns:
        Tuple of (new_files, modified_files, deleted_files, unchanged_count, skipped_count).
    """
    new_files: list[FileInfo] = []
    modified_files: list[FileInfo] = []
    unchanged_count = 0
    skipped_count = 0

    local_paths = set()

    for file_info in files:
        local_paths.add(file_info.relative_path)

        main_state = main_states.get(file_info.relative_path)
        user_state = user_states.get(file_info.relative_path)

        # First check if same as main - if so, skip (use main's data)
        if main_state and main_state.file_hash == file_info.hash:
            skipped_count += 1
            continue

        # Different from main (or not in main) - check user's previous state
        if user_state is None:
            # New for this user
            new_files.append(file_info)
        elif user_state.file_hash == file_info.hash:
            # Unchanged from user's previous parse
            unchanged_count += 1
        else:
            # Modified since user's previous parse
            file_info.previous_hash = user_state.file_hash
            modified_files.append(file_info)

    # Find files deleted by user (in main but not on user's disk, not already marked)
    deleted_files: list[FileInfo] = []
    for rel_path, main_state in main_states.items():
        if rel_path not in local_paths:
            user_state = user_states.get(rel_path)
            # Only mark as deleted if not already marked
            if user_state is None or not user_state.is_deleted:
                deleted_files.append(
                    FileInfo(
                        relative_path=rel_path,
                        absolute_path=Path(),
                        language="",
                        hash=None,
                        previous_hash=main_state.file_hash,
                    )
                )

    return new_files, modified_files, deleted_files, unchanged_count, skipped_count


# =============================================================================
# MAIN DETECT CHANGES FUNCTION
# =============================================================================


def detect_changes(
    discovery_result: DiscoveryResult,
    file_state_repo: FileStateRepository | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> ChangeManifest:
    """Run change detection on a discovered repository.

    Args:
        discovery_result: Result from Phase 1 Discovery.
        file_state_repo: Database interface (uses in-memory if None).
        on_progress: Optional callback(completed, total) for progress updates.

    Returns:
        ChangeManifest with all change information.

    Raises:
        ChangeDetectionError: If detection fails (e.g., main branch required but missing).
    """
    if file_state_repo is None:
        file_state_repo = InMemoryFileStateRepository()

    scope = discovery_result.scope
    repository = discovery_result.repository
    username = discovery_result.username
    repo_path = discovery_result.repo_path

    # Check if user branch requires main to exist
    if username != "main" and not file_state_repo.main_branch_exists(scope, repository):
        raise ChangeDetectionError(
            f"Main branch not found for {scope}/{repository}.\n"
            "Run 'magaldi parse --user main' first."
        )

    # 2.1 Enumerate files
    files = enumerate_files(
        repo_path,
        discovery_result.exclude_directories,
        discovery_result.exclude_files,
    )

    total_scanned = len(files)

    # 2.2 Compute hashes
    files = compute_hashes_parallel(files, on_progress=on_progress)

    # Filter out files that couldn't be hashed
    files = [f for f in files if f.hash is not None]

    # 2.3 Compare to database
    if username == "main":
        db_states = file_state_repo.get_file_states(scope, repository, "main")
        new_files, modified_files, deleted_files, unchanged_count = compare_main_branch(
            files, db_states
        )
        skipped_count = 0
    else:
        main_states = file_state_repo.get_file_states(scope, repository, "main")
        user_states = file_state_repo.get_file_states(scope, repository, username)
        new_files, modified_files, deleted_files, unchanged_count, skipped_count = (
            compare_user_branch(files, main_states, user_states)
        )

    return ChangeManifest(
        scope=scope,
        repository=repository,
        username=username,
        timestamp=datetime.now(),
        total_files_scanned=total_scanned,
        new_files=new_files,
        modified_files=modified_files,
        deleted_files=deleted_files,
        unchanged_count=unchanged_count,
        skipped_count=skipped_count,
    )
