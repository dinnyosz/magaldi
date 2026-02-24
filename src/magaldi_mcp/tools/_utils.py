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
    """Resolve scope and repository, always preferring auto-detected values.

    Auto-detected values from magaldi.yaml always take priority over
    explicitly provided values, because LLMs frequently guess wrong
    (e.g., passing scope="project" instead of the actual scope).

    Args:
        scope: Explicit scope (ignored if magaldi.yaml found).
        repository: Explicit repository (ignored if magaldi.yaml found).

    Returns:
        Tuple of (scope, repository) with auto-detected values preferred.
    """
    auto_scope, auto_repo = _auto_detect_repo_config()
    # Always prefer auto-detected values — LLMs often guess wrong
    if auto_scope:
        if scope and scope != auto_scope:
            logger.debug("Overriding scope=%r with auto-detected=%r", scope, auto_scope)
        scope = auto_scope
    if auto_repo:
        if repository and repository != auto_repo:
            logger.debug(
                "Overriding repository=%r with auto-detected=%r", repository, auto_repo
            )
        repository = auto_repo
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
