"""File discovery tools."""

from __future__ import annotations

from typing import Any

from shared.db.store import Repository

from ._utils import _resolve_scope_repo


def find_files(
    es: Repository,
    pattern: str,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Find indexed files by glob pattern.

    Searches file elements in Elasticsearch - no filesystem access needed.

    Args:
        es: Elasticsearch repository.
        pattern: Glob pattern (e.g., '**/*.py', 'src/**/*.ts').
        scope: Filter by scope.
        repository: Filter by repository.
        username: User branch to search.
        limit: Maximum files to return.

    Returns:
        List of matching files with basic info.
    """
    # Auto-detect scope/repository from magaldi.yaml if not provided
    scope, repository = _resolve_scope_repo(scope, repository)

    client = es._get_client()

    # Convert glob pattern to ES wildcard
    if "*" not in pattern and "?" not in pattern:
        # No wildcards: exact filename match anywhere in tree
        es_pattern = f"*/{pattern}"
    else:
        # Has wildcards: convert ** to * (ES wildcard doesn't have **)
        es_pattern = pattern.replace("**", "*")

    # Build ES query with wildcard filter (filtering happens in ES, not client-side)
    filters = [
        {"term": {"element_type": "file"}},
        {"term": {"username": username}},
        {"wildcard": {"relative_path": es_pattern}},
    ]
    if scope:
        filters.append({"term": {"scope": scope}})
    if repository:
        filters.append({"term": {"repository": repository}})

    # Fetch file elements - ES filters directly, no need for extra results
    es_result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {"bool": {"filter": filters}},
            "_source": ["element_id", "relative_path", "language", "line_end"],
            "size": limit,
        },
    )

    hits = es_result.get("hits", {}).get("hits", [])
    matches = []

    for hit in hits:
        source = hit["_source"]
        matches.append(
            {
                "path": source.get("relative_path", ""),
                "language": source.get("language"),
                "lines": source.get("line_end", 0),
            }
        )

    return matches
