"""Phase 1: Discovery - Path validation, config loading, language enumeration.

This module handles the initial discovery phase of the parser pipeline:
1. Validate repository path
2. Load magaldi.yaml configuration
3. Resolve username from CLI/env/config
4. Enumerate languages and count files
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class DiscoveryError(Exception):
    """Raised when discovery phase fails."""

    pass


# =============================================================================
# CONSTANTS
# =============================================================================

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".php": "php",
    ".rs": "rust",
    ".java": "java",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".txt": "text",
    ".nfo": "text",
    ".sh": "bash",
    ".bash": "bash",
    ".bats": "bash",
}

# Filenames (without extension) that map to a language
SUPPORTED_FILENAMES: dict[str, str] = {
    "Dockerfile": "dockerfile",
}

# Shebang tokens for detecting language from extensionless files
SHEBANG_PATTERNS: dict[str, str] = {
    "bash": "bash",
    "sh": "bash",
    "zsh": "bash",
    "python3": "python",
    "python": "python",
    "node": "javascript",
    "php": "php",
}

DEFAULT_EXCLUDE_DIRECTORIES: list[str] = [
    "node_modules",
    "vendor",
    "__pycache__",
    ".git",
    "dist",
    "build",
    ".venv",
    "venv",
    "target",
    ".next",
    ".cache",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "eggs",
    "*.egg-info",
]

DEFAULT_EXCLUDE_FILES: list[str] = [
    "*.min.js",
    "*.min.css",
    "*.map",
    "*.lock",
    "*-lock.yaml",
    "*-lock.json",
    "package-lock.json",
    "yarn.lock",
    "composer.lock",
    "Cargo.lock",
    "poetry.lock",
    "Pipfile.lock",
]


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class LanguageStats:
    """Statistics for a single language."""

    files: int = 0
    lines: int = 0


@dataclass
class RepoConfig:
    """Repository configuration from magaldi.yaml."""

    scope: str
    name: str
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    user: str | None = None
    exclude_directories: list[str] = field(default_factory=list)
    exclude_files: list[str] = field(default_factory=list)


@dataclass
class DiscoveryResult:
    """Result of the discovery phase, passed to Phase 2."""

    # Identity
    scope: str
    repository: str
    username: str
    repo_path: Path

    # Config
    description: str | None
    tags: list[str]
    exclude_directories: list[str]
    exclude_files: list[str]

    # Languages
    languages: dict[str, LanguageStats]
    total_files: int
    total_lines: int


# =============================================================================
# 1.1 PATH VALIDATION
# =============================================================================


def validate_path(path: str | Path) -> Path:
    """Validate that the given path is a readable directory.

    Args:
        path: Path to validate (string or Path object).

    Returns:
        Resolved absolute Path object.

    Raises:
        DiscoveryError: If path doesn't exist, isn't a directory, or isn't readable.
    """
    path = Path(path)

    if not path.exists():
        raise DiscoveryError(f"Path does not exist: {path}")

    if not path.is_dir():
        raise DiscoveryError(f"Path is not a directory: {path}")

    # Check readability by trying to list contents
    try:
        list(path.iterdir())
    except PermissionError as e:
        raise DiscoveryError(f"Permission denied reading directory: {path}") from e

    return path.resolve()


# =============================================================================
# 1.2 CONFIG LOADING
# =============================================================================


def load_repo_config(repo_path: Path) -> RepoConfig:
    """Load repository configuration from magaldi.yaml.

    Args:
        repo_path: Path to repository root.

    Returns:
        RepoConfig object with loaded configuration.

    Raises:
        DiscoveryError: If config file is missing or invalid.
    """
    config_path = repo_path / "magaldi.yaml"

    if not config_path.exists():
        raise DiscoveryError(
            f"Configuration file not found: {config_path}\n"
            "Create a magaldi.yaml file with at least a 'scope' field."
        )

    try:
        with open(config_path, encoding="utf-8") as f:
            raw_config: dict[str, Any] = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise DiscoveryError(f"Invalid YAML in {config_path}: {e}") from e

    # Validate required fields
    if "scope" not in raw_config or not raw_config["scope"]:
        raise DiscoveryError(
            f"Required field 'scope' missing in {config_path}\n"
            "The scope is used to group related repositories together."
        )

    # Build config with defaults
    scope = raw_config["scope"]
    # Support both "name" and "repository" fields (repository is an alias)
    name = raw_config.get("name") or raw_config.get("repository") or repo_path.name
    description = raw_config.get("description")
    tags = raw_config.get("tags", [])
    user = raw_config.get("user")

    # Merge exclusions with defaults
    user_exclude_dirs = raw_config.get("exclude_directories", [])
    user_exclude_files = raw_config.get("exclude_files", [])

    exclude_directories = _merge_unique(DEFAULT_EXCLUDE_DIRECTORIES, user_exclude_dirs)
    exclude_files = _merge_unique(DEFAULT_EXCLUDE_FILES, user_exclude_files)

    return RepoConfig(
        scope=scope,
        name=name,
        description=description,
        tags=tags,
        user=user,
        exclude_directories=exclude_directories,
        exclude_files=exclude_files,
    )


def _merge_unique(defaults: list[str], user_values: list[str]) -> list[str]:
    """Merge two lists, keeping unique values and preserving order."""
    seen: set[str] = set()
    result: list[str] = []

    for item in defaults + user_values:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


# =============================================================================
# 1.3 USERNAME RESOLUTION
# =============================================================================


def resolve_username(cli_user: str | None, config: RepoConfig) -> str:
    """Resolve username from CLI, environment, or config.

    Priority (highest to lowest):
    1. CLI argument
    2. MAGALDI_USER environment variable
    3. user field in magaldi.yaml

    Args:
        cli_user: Username from CLI argument (may be None).
        config: Repository configuration.

    Returns:
        Resolved username.

    Raises:
        DiscoveryError: If no username can be resolved.
    """
    # Try CLI first
    if cli_user and cli_user.strip():
        return cli_user.strip()

    # Try environment variable
    env_user = os.environ.get("MAGALDI_USER")
    if env_user and env_user.strip():
        return env_user.strip()

    # Try config file
    if config.user and config.user.strip():
        return config.user.strip()

    # No username found
    raise DiscoveryError(
        "Username is required but not provided.\n"
        "Provide via: --user CLI argument, MAGALDI_USER env var, or 'user' in magaldi.yaml"
    )


# =============================================================================
# 1.4 LANGUAGE ENUMERATION
# =============================================================================


def _detect_shebang_language(file_path: Path) -> str | None:
    """Detect language from shebang line for extensionless files.

    Reads only the first 128 bytes in binary mode to minimize I/O
    and handle binary files gracefully.
    """
    try:
        with open(file_path, "rb") as f:
            first_bytes = f.read(128)
        if not first_bytes.startswith(b"#!"):
            return None
        first_line = first_bytes.split(b"\n", 1)[0].decode("ascii", errors="replace")
        for token, lang in SHEBANG_PATTERNS.items():
            if token in first_line:
                return lang
    except OSError:
        pass
    return None


def _detect_file_language(file_path: Path) -> str | None:
    """Detect language from file extension, filename, or shebang.

    Checks SUPPORTED_EXTENSIONS first, then SUPPORTED_FILENAMES for
    files like Dockerfile, then shebang for extensionless files.
    """
    ext = file_path.suffix.lower()
    if ext in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[ext]

    # Check filename match (e.g., Dockerfile, Dockerfile.prod)
    name = file_path.name
    for pattern, lang in SUPPORTED_FILENAMES.items():
        if name == pattern or name.startswith(f"{pattern}."):
            return lang

    # Check shebang for extensionless files (only when no extension)
    if not ext:
        return _detect_shebang_language(file_path)

    return None


def enumerate_languages(
    repo_path: Path,
    config: RepoConfig,
) -> dict[str, LanguageStats]:
    """Enumerate languages and count files/lines in the repository.

    Args:
        repo_path: Path to repository root.
        config: Repository configuration with exclusions.

    Returns:
        Dictionary mapping language names to LanguageStats.
    """
    stats: dict[str, LanguageStats] = {}

    for file_path in _walk_files(repo_path, config):
        language = _detect_file_language(file_path)
        if language is None:
            continue
        line_count = _count_lines(file_path)

        if language not in stats:
            stats[language] = LanguageStats()

        stats[language].files += 1
        stats[language].lines += line_count

    return stats


def _walk_files(repo_path: Path, config: RepoConfig) -> list[Path]:
    """Walk repository and return list of non-excluded files.

    Args:
        repo_path: Path to repository root.
        config: Repository configuration with exclusions.

    Returns:
        List of file paths (not directories, not excluded).
    """
    files: list[Path] = []

    for root, dirs, filenames in os.walk(repo_path):
        root_path = Path(root)

        # Filter out excluded directories (modifying dirs in-place prevents descent)
        dirs[:] = [
            d for d in dirs
            if not _is_excluded_dir(d, config.exclude_directories)
        ]

        # Process files
        for filename in filenames:
            if _is_excluded_file(filename, config.exclude_files):
                continue

            file_path = root_path / filename

            # Skip symlinks
            if file_path.is_symlink():
                continue

            files.append(file_path)

    return files


def _is_excluded_dir(dirname: str, exclude_patterns: list[str]) -> bool:
    """Check if a directory name matches any exclusion pattern."""
    return any(fnmatch.fnmatch(dirname, pattern) for pattern in exclude_patterns)


def _is_excluded_file(filename: str, exclude_patterns: list[str]) -> bool:
    """Check if a filename matches any exclusion pattern."""
    return any(fnmatch.fnmatch(filename, pattern) for pattern in exclude_patterns)


def _count_lines(file_path: Path) -> int:
    """Count lines in a file, handling encoding errors gracefully."""
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


# =============================================================================
# MAIN DISCOVER FUNCTION
# =============================================================================


def discover(
    repo_path: str | Path,
    username: str | None = None,
    skip_tests: bool = False,
) -> DiscoveryResult:
    """Run the discovery phase on a repository.

    This is the main entry point for Phase 1. It:
    1. Validates the repository path
    2. Loads magaldi.yaml configuration
    3. Resolves the username
    4. Enumerates languages and counts files

    Args:
        repo_path: Path to repository root.
        username: Username from CLI (optional, can come from env/config).
        skip_tests: If True, exclude test directories and files.

    Returns:
        DiscoveryResult with all discovery data.

    Raises:
        DiscoveryError: If any validation or loading fails.
    """
    # 1.1 Validate path
    validated_path = validate_path(repo_path)

    # 1.2 Load config
    config = load_repo_config(validated_path)

    # 1.2a If skip_tests, add test directories to exclusions
    if skip_tests:
        test_dirs = ["tests", "test", "__tests__", "spec", "specs"]
        test_files = ["test_*.py", "*_test.py", "*.test.js", "*.spec.js", "*.test.ts", "*.spec.ts"]
        config = RepoConfig(
            scope=config.scope,
            name=config.name,
            description=config.description,
            tags=config.tags,
            user=config.user,
            exclude_directories=_merge_unique(config.exclude_directories, test_dirs),
            exclude_files=_merge_unique(config.exclude_files, test_files),
        )

    # 1.3 Resolve username
    resolved_username = resolve_username(username, config)

    # 1.4 Enumerate languages
    languages = enumerate_languages(validated_path, config)

    # Calculate totals
    total_files = sum(stats.files for stats in languages.values())
    total_lines = sum(stats.lines for stats in languages.values())

    return DiscoveryResult(
        scope=config.scope,
        repository=config.name,
        username=resolved_username,
        repo_path=validated_path,
        description=config.description,
        tags=config.tags,
        exclude_directories=config.exclude_directories,
        exclude_files=config.exclude_files,
        languages=languages,
        total_files=total_files,
        total_lines=total_lines,
    )
