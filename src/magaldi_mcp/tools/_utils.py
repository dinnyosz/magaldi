"""Shared utilities for MCP tool implementations."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _auto_detect_repo_config() -> tuple[str | None, str | None]:
    """Auto-detect scope and repository from magaldi.yaml in cwd or parent directories.

    Walks up from current working directory looking for magaldi.yaml.

    Returns:
        Tuple of (scope, repository) or (None, None) if not found.
    """
    cwd = Path.cwd()

    # Walk up directory tree looking for magaldi.yaml
    for directory in [cwd, *cwd.parents]:
        config_path = directory / "magaldi.yaml"
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                scope = config.get("scope")
                repository = config.get("repository") or config.get("name") or directory.name
                if scope:
                    return scope, repository
            except (yaml.YAMLError, OSError):
                logger.debug("Failed to read magaldi.yaml from %s", directory, exc_info=True)
                continue

    return None, None


def _resolve_scope_repo(
    scope: str | None, repository: str | None
) -> tuple[str | None, str | None]:
    """Resolve scope and repository, auto-detecting from magaldi.yaml if not provided.

    Args:
        scope: Explicit scope or None to auto-detect.
        repository: Explicit repository or None to auto-detect.

    Returns:
        Tuple of (scope, repository) with auto-detected values filled in.
    """
    if scope is None or repository is None:
        auto_scope, auto_repo = _auto_detect_repo_config()
        scope = scope or auto_scope
        repository = repository or auto_repo
    return scope, repository


def _escape_for_lucene_regexp(name: str) -> str:
    """Escape special characters for Lucene regexp queries.

    Lucene regexp uses different syntax than standard regex.
    Special chars that need escaping: . ? + * | { } [ ] ( ) " \\ # @ & < >  ~

    Args:
        name: The string to escape.

    Returns:
        Escaped string safe for Lucene regexp.
    """
    # Lucene regexp special characters
    special_chars = r'.?+*|{}[]()"\#@&<>~'
    result = []
    for char in name:
        if char in special_chars:
            result.append("\\")
        result.append(char)
    return "".join(result)
