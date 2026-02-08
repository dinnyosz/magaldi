"""Pattern-based search tools."""

from __future__ import annotations

from typing import Any

from shared.db.store import Repository

from ._utils import _resolve_scope_repo


def pattern_search(
    repo: Repository,
    pattern: str,
    mode: str,
    scope: str | None = None,
    repository: str | None = None,
    username: str | None = None,
    slop: int = 5,
    glob: str | None = None,
    limit: int = 50,
    include_tests: bool = True,
) -> dict[str, Any]:
    """Search code using native pattern matching.

    Three modes available:
    - regexp: Lucene regexp syntax (e.g., "add_column.*Model")
    - wildcard: Simple wildcards (e.g., "*column*Model*")
    - proximity: Terms near each other (e.g., "add column Model")

    Args:
        repo: Search repository.
        pattern: Search pattern (syntax depends on mode).
        mode: One of "regexp", "wildcard", "proximity".
        scope: Filter by scope (auto-detected from magaldi.yaml if not provided).
        repository: Filter by repository (auto-detected from magaldi.yaml if not provided).
        username: User branch to search (optional).
        slop: For proximity mode: max positions between terms.
        glob: File path glob filter (e.g., '*.py').
        limit: Maximum results to return.
        include_tests: Whether to include test results.

    Returns:
        Dict with code_results, test_results, and totals.

    Raises:
        ValueError: If mode is not one of the valid options.
    """
    # Auto-detect scope/repository from magaldi.yaml if not provided
    scope, repository = _resolve_scope_repo(scope, repository)
    if not scope or not repository:
        raise ValueError(
            "scope and repository are required. Either provide them explicitly "
            "or create a magaldi.yaml file in your project root."
        )

    valid_modes = ("regexp", "wildcard", "proximity")
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

    if mode == "regexp":
        results = repo.search_by_regexp(
            pattern=pattern,
            scope=scope,
            repository=repository,
            username=username,
            glob=glob,
            size=limit,
            include_tests=include_tests,
        )
    elif mode == "wildcard":
        results = repo.search_by_wildcard(
            pattern=pattern,
            scope=scope,
            repository=repository,
            username=username,
            glob=glob,
            size=limit,
            include_tests=include_tests,
        )
    else:  # proximity
        results = repo.search_by_proximity(
            terms=pattern,
            slop=slop,
            scope=scope,
            repository=repository,
            username=username,
            glob=glob,
            size=limit,
            include_tests=include_tests,
        )

    # Format results
    code_results: list[dict[str, Any]] = []
    test_results: list[dict[str, Any]] = []

    for result in results:
        is_test = result.get("is_test", False)

        entry = {
            "element_id": result.get("element_id"),
            "hash_id": result.get("hash_id"),
            "file": result.get("relative_path"),
            "name": result.get("name"),
            "element_type": result.get("element_type"),
            "line_start": result.get("line_start"),
            "raw_code": result.get("raw_code"),
            "is_test": is_test,
        }

        if is_test:
            test_results.append(entry)
        else:
            code_results.append(entry)

    return {
        "code_results": code_results,
        "test_results": test_results,
        "totals": {
            "code": len(code_results),
            "tests": len(test_results),
        },
        "mode": mode,
        "pattern": pattern,
    }
