"""Code quality analysis tools."""

from __future__ import annotations

from typing import Any

from shared.db.repositories.base import INDEX_NAME
from shared.db.store import Repository

from ._utils import _resolve_scope_repo


def find_complex_functions(
    repo: Repository,
    scope: str | None = None,
    repository: str | None = None,
    username: str | None = None,
    min_complexity: int = 10,
    limit: int = 20,
    include_tests: bool = False,
) -> dict[str, Any]:
    """Find functions/methods with high cyclomatic complexity.

    Args:
        repo: Search repository.
        scope: Repository scope (auto-detected from magaldi.yaml if not provided).
        repository: Repository name (auto-detected from magaldi.yaml if not provided).
        username: User branch (defaults to "main").
        min_complexity: Minimum cyclomatic complexity threshold.
        limit: Maximum results.
        include_tests: Whether to include test functions.

    Returns:
        Dict with functions list and statistics.
    """
    scope, repository = _resolve_scope_repo(scope, repository)
    if not scope or not repository:
        raise ValueError(
            "scope and repository are required. Either provide them explicitly "
            "or create a magaldi.yaml file in your project root."
        )
    username = username or "main"
    limit = max(1, min(limit, 100))

    must_clauses = [
        {"term": {"scope": scope}},
        {"term": {"repository": repository}},
        {"term": {"username": username}},
        {"terms": {"element_type": ["function", "method"]}},
        {"range": {"complexity.cyclomatic": {"gte": min_complexity}}},
    ]

    if not include_tests:
        must_clauses.append({"term": {"is_test": False}})

    client = repo._get_client()
    result = client.search(
        index=INDEX_NAME,
        body={
            "query": {"bool": {"must": must_clauses}},
            "sort": [{"complexity.cyclomatic": {"order": "desc"}}],
            "size": limit,
            "_source": [
                "element_id",
                "hash_id",
                "name",
                "element_type",
                "relative_path",
                "line_start",
                "complexity",
                "code_metrics",
                "summary",
                "is_test",
            ],
        },
    )

    functions = []
    for hit in result.get("hits", {}).get("hits", []):
        source = hit["_source"]
        complexity = source.get("complexity", {})
        functions.append({
            "element_id": source.get("element_id"),
            "hash_id": source.get("hash_id"),
            "name": source.get("name"),
            "type": source.get("element_type"),
            "file": source.get("relative_path"),
            "line": source.get("line_start"),
            "cyclomatic": complexity.get("cyclomatic", 0),
            "nesting_depth": complexity.get("nesting_depth", 0),
            "branch_count": complexity.get("branch_count", 0),
            "line_count": source.get("code_metrics", {}).get("line_count", 0),
            "summary": source.get("summary"),
            "is_test": source.get("is_test", False),
        })

    return {
        "functions": functions,
        "count": len(functions),
        "min_complexity": min_complexity,
    }


def find_security_issues(
    repo: Repository,
    scope: str | None = None,
    repository: str | None = None,
    username: str | None = None,
    severity: str = "high",
    kind: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Find potential security issues in code.

    Args:
        repo: Search repository.
        scope: Repository scope (auto-detected from magaldi.yaml if not provided).
        repository: Repository name (auto-detected from magaldi.yaml if not provided).
        username: User branch (defaults to "main").
        severity: Minimum severity level ("critical", "high", "medium", "low", "info", "all").
        kind: Filter by issue kind (optional).
        limit: Maximum results.

    Returns:
        Dict with issues list and statistics.
    """
    scope, repository = _resolve_scope_repo(scope, repository)
    if not scope or not repository:
        raise ValueError(
            "scope and repository are required. Either provide them explicitly "
            "or create a magaldi.yaml file in your project root."
        )
    username = username or "main"
    limit = max(1, min(limit, 100))

    # Severity levels in order
    severity_levels = ["critical", "high", "medium", "low", "info"]
    if severity != "all":
        min_idx = severity_levels.index(severity) if severity in severity_levels else 1
        allowed_severities = severity_levels[: min_idx + 1]
    else:
        allowed_severities = severity_levels

    must_clauses = [
        {"term": {"scope": scope}},
        {"term": {"repository": repository}},
        {"term": {"username": username}},
        {"terms": {"element_type": ["function", "method"]}},
        {
            "nested": {
                "path": "security_issues",
                "query": {
                    "bool": {
                        "must": [
                            {"terms": {"security_issues.severity": allowed_severities}}
                        ]
                    }
                },
            }
        },
    ]

    client = repo._get_client()
    result = client.search(
        index=INDEX_NAME,
        body={
            "query": {"bool": {"must": must_clauses}},
            "size": limit,
            "_source": [
                "element_id",
                "hash_id",
                "name",
                "element_type",
                "relative_path",
                "line_start",
                "security_issues",
                "summary",
            ],
        },
    )

    issues = []
    by_severity: dict[str, int] = {}
    by_kind: dict[str, int] = {}

    for hit in result.get("hits", {}).get("hits", []):
        source = hit["_source"]
        element_issues = source.get("security_issues", [])

        for issue in element_issues:
            issue_severity = issue.get("severity", "info")
            issue_kind = issue.get("kind", "unknown")

            # Filter by severity
            if severity != "all" and issue_severity not in allowed_severities:
                continue

            # Filter by kind
            if kind and issue_kind != kind:
                continue

            issues.append({
                "element_id": source.get("element_id"),
                "hash_id": source.get("hash_id"),
                "function_name": source.get("name"),
                "element_type": source.get("element_type"),
                "file": source.get("relative_path"),
                "function_line": source.get("line_start"),
                "issue_line": issue.get("line"),
                "severity": issue_severity,
                "kind": issue_kind,
                "message": issue.get("message"),
            })

            by_severity[issue_severity] = by_severity.get(issue_severity, 0) + 1
            by_kind[issue_kind] = by_kind.get(issue_kind, 0) + 1

    # Sort by severity (critical first)
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    issues.sort(key=lambda x: severity_order.get(x["severity"], 5))

    return {
        "issues": issues[:limit],
        "count": len(issues),
        "by_severity": by_severity,
        "by_kind": by_kind,
    }


def find_undocumented(
    repo: Repository,
    scope: str | None = None,
    repository: str | None = None,
    username: str | None = None,
    max_coverage: float = 0.5,
    public_only: bool = True,
    limit: int = 30,
    include_tests: bool = False,
) -> dict[str, Any]:
    """Find functions/methods missing documentation.

    Args:
        repo: Search repository.
        scope: Repository scope (auto-detected from magaldi.yaml if not provided).
        repository: Repository name (auto-detected from magaldi.yaml if not provided).
        username: User branch (defaults to "main").
        max_coverage: Maximum documentation coverage (0-1) to include.
        public_only: Only include public functions/methods.
        limit: Maximum results.
        include_tests: Whether to include test functions.

    Returns:
        Dict with functions list and statistics.
    """
    scope, repository = _resolve_scope_repo(scope, repository)
    if not scope or not repository:
        raise ValueError(
            "scope and repository are required. Either provide them explicitly "
            "or create a magaldi.yaml file in your project root."
        )
    username = username or "main"
    limit = max(1, min(limit, 100))

    must_clauses = [
        {"term": {"scope": scope}},
        {"term": {"repository": repository}},
        {"term": {"username": username}},
        {"terms": {"element_type": ["function", "method"]}},
    ]

    # Filter by coverage
    should_no_docstring = {"term": {"docstring_quality.has_docstring": False}}
    should_low_coverage = {"range": {"docstring_quality.coverage": {"lte": max_coverage}}}
    must_clauses.append({"bool": {"should": [should_no_docstring, should_low_coverage]}})

    if public_only:
        must_clauses.append({"term": {"visibility": "public"}})

    if not include_tests:
        must_clauses.append({"term": {"is_test": False}})

    client = repo._get_client()
    result = client.search(
        index=INDEX_NAME,
        body={
            "query": {"bool": {"must": must_clauses}},
            "sort": [{"docstring_quality.coverage": {"order": "asc"}}],
            "size": limit,
            "_source": [
                "element_id",
                "hash_id",
                "name",
                "element_type",
                "relative_path",
                "line_start",
                "visibility",
                "docstring_quality",
                "code_metrics",
                "summary",
            ],
        },
    )

    functions = []
    for hit in result.get("hits", {}).get("hits", []):
        source = hit["_source"]
        doc_quality = source.get("docstring_quality", {})
        functions.append({
            "element_id": source.get("element_id"),
            "hash_id": source.get("hash_id"),
            "name": source.get("name"),
            "type": source.get("element_type"),
            "file": source.get("relative_path"),
            "line": source.get("line_start"),
            "visibility": source.get("visibility"),
            "has_docstring": doc_quality.get("has_docstring", False),
            "has_params": doc_quality.get("has_params", False),
            "has_return": doc_quality.get("has_return", False),
            "coverage": doc_quality.get("coverage", 0),
            "param_count": source.get("code_metrics", {}).get("param_count", 0),
            "summary": source.get("summary"),
        })

    return {
        "functions": functions,
        "count": len(functions),
        "max_coverage": max_coverage,
    }


def find_env_usage(
    repo: Repository,
    scope: str | None = None,
    repository: str | None = None,
    username: str | None = None,
    env_name: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Find environment variable usage across the codebase.

    Args:
        repo: Search repository.
        scope: Repository scope (auto-detected from magaldi.yaml if not provided).
        repository: Repository name (auto-detected from magaldi.yaml if not provided).
        username: User branch (defaults to "main").
        env_name: Filter by specific env var name (optional).
        limit: Maximum results.

    Returns:
        Dict with usages list and statistics.
    """
    scope, repository = _resolve_scope_repo(scope, repository)
    if not scope or not repository:
        raise ValueError(
            "scope and repository are required. Either provide them explicitly "
            "or create a magaldi.yaml file in your project root."
        )
    username = username or "main"
    limit = max(1, min(limit, 100))

    must_clauses = [
        {"term": {"scope": scope}},
        {"term": {"repository": repository}},
        {"term": {"username": username}},
        {"terms": {"element_type": ["function", "method"]}},
    ]

    # Only include elements with env_vars
    nested_query: dict[str, Any] = {
        "nested": {
            "path": "env_vars",
            "query": {"bool": {"must": [{"exists": {"field": "env_vars.name"}}]}},
        }
    }

    if env_name:
        nested_query["nested"]["query"]["bool"]["must"].append(
            {"term": {"env_vars.name": env_name}}
        )

    must_clauses.append(nested_query)

    client = repo._get_client()
    result = client.search(
        index=INDEX_NAME,
        body={
            "query": {"bool": {"must": must_clauses}},
            "size": limit,
            "_source": [
                "element_id",
                "hash_id",
                "name",
                "element_type",
                "relative_path",
                "line_start",
                "env_vars",
                "summary",
            ],
        },
    )

    usages = []
    env_var_counts: dict[str, int] = {}

    for hit in result.get("hits", {}).get("hits", []):
        source = hit["_source"]
        element_env_vars = source.get("env_vars", [])

        for ev in element_env_vars:
            ev_name = ev.get("name", "")

            # Filter by name if specified
            if env_name and ev_name != env_name:
                continue

            usages.append({
                "element_id": source.get("element_id"),
                "hash_id": source.get("hash_id"),
                "function_name": source.get("name"),
                "element_type": source.get("element_type"),
                "file": source.get("relative_path"),
                "function_line": source.get("line_start"),
                "env_name": ev_name,
                "env_line": ev.get("line"),
                "access_type": ev.get("access_type"),
            })

            env_var_counts[ev_name] = env_var_counts.get(ev_name, 0) + 1

    # Sort by env var name
    usages.sort(key=lambda x: x["env_name"])

    return {
        "usages": usages[:limit],
        "count": len(usages),
        "unique_vars": len(env_var_counts),
        "by_var": env_var_counts,
    }


def find_async_code(
    repo: Repository,
    scope: str | None = None,
    repository: str | None = None,
    username: str | None = None,
    pattern: str = "all",
    limit: int = 30,
    include_tests: bool = False,
) -> dict[str, Any]:
    """Find async/concurrent code patterns.

    Args:
        repo: Search repository.
        scope: Repository scope (auto-detected from magaldi.yaml if not provided).
        repository: Repository name (auto-detected from magaldi.yaml if not provided).
        username: User branch (defaults to "main").
        pattern: Type of pattern ("async", "threading", "locking", "all").
        limit: Maximum results.
        include_tests: Whether to include test functions.

    Returns:
        Dict with functions list and statistics.
    """
    scope, repository = _resolve_scope_repo(scope, repository)
    if not scope or not repository:
        raise ValueError(
            "scope and repository are required. Either provide them explicitly "
            "or create a magaldi.yaml file in your project root."
        )
    username = username or "main"
    limit = max(1, min(limit, 100))

    must_clauses = [
        {"term": {"scope": scope}},
        {"term": {"repository": repository}},
        {"term": {"username": username}},
        {"terms": {"element_type": ["function", "method"]}},
    ]

    # Build pattern-specific filter
    if pattern == "async":
        must_clauses.append({"term": {"concurrency.is_async": True}})
    elif pattern == "threading":
        must_clauses.append({"term": {"concurrency.uses_threads": True}})
    elif pattern == "locking":
        must_clauses.append({"term": {"concurrency.uses_locks": True}})
    else:  # "all"
        must_clauses.append({
            "bool": {
                "should": [
                    {"term": {"concurrency.is_async": True}},
                    {"term": {"concurrency.uses_threads": True}},
                    {"term": {"concurrency.uses_locks": True}},
                ],
                "minimum_should_match": 1,
            }
        })

    if not include_tests:
        must_clauses.append({"term": {"is_test": False}})

    client = repo._get_client()
    result = client.search(
        index=INDEX_NAME,
        body={
            "query": {"bool": {"must": must_clauses}},
            "size": limit,
            "_source": [
                "element_id",
                "hash_id",
                "name",
                "element_type",
                "relative_path",
                "line_start",
                "is_async",
                "concurrency",
                "summary",
            ],
        },
    )

    functions = []
    pattern_counts: dict[str, int] = {}

    for hit in result.get("hits", {}).get("hits", []):
        source = hit["_source"]
        conc = source.get("concurrency", {})
        patterns_found = conc.get("patterns", [])

        functions.append({
            "element_id": source.get("element_id"),
            "hash_id": source.get("hash_id"),
            "name": source.get("name"),
            "type": source.get("element_type"),
            "file": source.get("relative_path"),
            "line": source.get("line_start"),
            "is_async": conc.get("is_async", False),
            "uses_threads": conc.get("uses_threads", False),
            "uses_locks": conc.get("uses_locks", False),
            "patterns": patterns_found,
            "summary": source.get("summary"),
        })

        for p in patterns_found:
            pattern_counts[p] = pattern_counts.get(p, 0) + 1

    return {
        "functions": functions,
        "count": len(functions),
        "pattern_filter": pattern,
        "by_pattern": pattern_counts,
    }
