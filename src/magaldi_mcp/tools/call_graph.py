"""Call graph analysis tools."""

from __future__ import annotations

from typing import Any

from shared.db.store import Repository


def get_call_graph(
    repo: Repository,
    element_id: str,
    direction: str = "both",
) -> dict[str, Any]:
    """Get callers and/or callees of a function/method.

    Analyzes indexed code in search backend - no filesystem access needed.

    Args:
        repo: Search repository.
        element_id: Function/method element ID.
        direction: 'callers', 'callees', or 'both'.

    Returns:
        Call graph with callers and callees lists.
    """
    doc = repo.get_document_by_id_or_hash(element_id)
    if not doc:
        raise ValueError(f"Element not found: {element_id}")

    name = doc.get("name")
    element_type = doc.get("element_type")
    defining_file = doc.get("relative_path")
    line_start = doc.get("line_start")
    scope = doc.get("scope")
    repository = doc.get("repository")
    username = doc.get("username", "main")
    # Get actual element_id from doc (calls.resolved_id stores element_id format)
    target_element_id = doc.get("element_id")

    result: dict[str, Any] = {
        "element": {
            "name": name,
            "type": element_type,
            "file": defining_file,
            "line": line_start,
            "element_id": target_element_id,
            "hash_id": doc.get("hash_id"),
        },
        "callers": [],
        "callees": [],
    }

    # Find callers (who calls this function) using indexed call data
    if direction in ("callers", "both"):
        callers = repo.find_elements_calling(
            target_id=target_element_id,
            scope=scope,
            repository=repository,
            username=username,
            limit=30,
        )
        for caller in callers:
            result["callers"].append(
                {
                    "element_id": caller.get("element_id"),
                    "hash_id": caller.get("hash_id"),
                    "name": caller.get("name"),
                    "type": caller.get("element_type"),
                    "file": caller.get("relative_path"),
                    "line": caller.get("line_start"),
                    "summary": caller.get("summary", ""),
                }
            )

    # Find callees (what this function calls) using indexed call data
    if direction in ("callees", "both"):
        calls = repo.get_calls(element_id)
        for call in calls:
            callee_entry: dict[str, Any] = {
                "name": call.get("name"),
                "receiver": call.get("receiver"),
                "line": call.get("line"),
            }
            # Include resolved target info if available
            resolved_id = call.get("resolved_id")
            if resolved_id:
                callee_entry["element_id"] = resolved_id
                # Get resolved element details
                resolved_doc = repo.get_document(resolved_id)
                if resolved_doc:
                    callee_entry["type"] = resolved_doc.get("element_type")
                    callee_entry["file"] = resolved_doc.get("relative_path")
                    callee_entry["target_line"] = resolved_doc.get("line_start")
                    callee_entry["summary"] = resolved_doc.get("summary", "")
            result["callees"].append(callee_entry)

    # Include pre-computed semantic relationships if available
    semantic_related = doc.get("semantic_related", [])
    if semantic_related:
        related_list: list[dict[str, Any]] = []
        for rel in semantic_related:
            rel_id = rel.get("element_id")
            if not rel_id:
                continue
            rel_doc = repo.get_document(rel_id)
            if rel_doc:
                related_list.append({
                    "element_id": rel_id,
                    "hash_id": rel.get("hash_id", rel_doc.get("hash_id", "")),
                    "name": rel_doc.get("name"),
                    "type": rel_doc.get("element_type"),
                    "file": rel_doc.get("relative_path"),
                    "line": rel_doc.get("line_start"),
                    "summary": rel_doc.get("summary", ""),
                    "score": rel.get("score", 0.0),
                })
        if related_list:
            result["semantic_related"] = related_list

    return result


def find_callers(
    repo: Repository,
    element_id: str,
    scope: str | None = None,
    repository: str | None = None,
    username: str | None = None,
    limit: int = 30,
    include_tests: bool = True,
) -> dict[str, Any]:
    """Find all functions that call the specified element.

    Uses indexed call data to find callers via calls.resolved_id.

    Args:
        repo: Search repository.
        element_id: Target element ID to find callers of.
        scope: Filter by scope.
        repository: Filter by repository.
        username: Filter by username branch.
        limit: Maximum results to return.
        include_tests: Whether to include test functions as callers.

    Returns:
        Dict with target info and lists of callers grouped by code/tests.
    """
    # Get target element info (supports both element_id and hash_id)
    doc = repo.get_document_by_id_or_hash(element_id)
    if not doc:
        raise ValueError(f"Element not found: {element_id}")

    # Use target's scope/repo if not specified
    scope = scope or doc.get("scope")
    repository = repository or doc.get("repository")
    username = username or doc.get("username", "main")

    # Get the actual element_id from doc (calls.resolved_id stores element_id format)
    target_element_id = doc.get("element_id")

    # Find elements calling the target
    callers = repo.find_elements_calling(
        target_id=target_element_id,
        scope=scope,
        repository=repository,
        username=username,
        limit=limit * 2 if include_tests else limit,  # Get extra to filter
    )

    # Group by is_test
    code_results: list[dict[str, Any]] = []
    test_results: list[dict[str, Any]] = []

    for caller in callers:
        is_test = caller.get("is_test", False)

        # Skip tests if not included
        if is_test and not include_tests:
            continue

        entry = {
            "element_id": caller.get("element_id"),
            "hash_id": caller.get("hash_id"),
            "name": caller.get("name"),
            "type": caller.get("element_type"),
            "file": caller.get("relative_path"),
            "line": caller.get("line_start"),
            "summary": caller.get("summary", ""),
            "is_test": is_test,
        }

        if is_test:
            test_results.append(entry)
        else:
            code_results.append(entry)

        # Respect limit
        if len(code_results) + len(test_results) >= limit:
            break

    return {
        "target": {
            "element_id": element_id,
            "hash_id": doc.get("hash_id"),
            "name": doc.get("name"),
            "type": doc.get("element_type"),
            "file": doc.get("relative_path"),
            "line": doc.get("line_start"),
        },
        "code_results": code_results[:limit],
        "test_results": test_results[:limit] if include_tests else [],
        "total_code": len(code_results),
        "total_tests": len(test_results) if include_tests else 0,
    }


def find_call_chain(
    repo: Repository,
    element_id: str,
    direction: str = "callees",
    max_depth: int = 5,
    scope: str | None = None,
    repository: str | None = None,
    username: str | None = None,
) -> dict[str, Any]:
    """Trace call chains from an element.

    Args:
        repo: Search repository.
        element_id: Starting element ID.
        direction: "callees" (what does it call), "callers" (what calls it),
                   or "both".
        max_depth: Maximum depth to traverse (default 5).
        scope: Filter by scope.
        repository: Filter by repository.
        username: Filter by username branch.

    Returns:
        Tree structure with depth levels showing call relationships.
    """
    # Validate max_depth
    max_depth = max(1, min(max_depth, 10))

    # Get root element info (supports both element_id and hash_id)
    doc = repo.get_document_by_id_or_hash(element_id)
    if not doc:
        raise ValueError(f"Element not found: {element_id}")

    # Use element's scope/repo if not specified
    scope = scope or doc.get("scope")
    repository = repository or doc.get("repository")
    username = username or doc.get("username", "main")
    # Get actual element_id from doc (calls.resolved_id stores element_id format)
    root_element_id = doc.get("element_id")

    root_node = {
        "element_id": root_element_id,
        "hash_id": doc.get("hash_id"),
        "name": doc.get("name"),
        "type": doc.get("element_type"),
        "file": doc.get("relative_path"),
        "line": doc.get("line_start"),
        "depth": 0,
    }

    result: dict[str, Any] = {
        "root": root_node,
        "direction": direction,
        "max_depth": max_depth,
    }

    visited: set[str] = {root_element_id}

    def build_callers_tree(node_id: str, depth: int) -> list[dict[str, Any]]:
        """Recursively build tree of callers."""
        if depth >= max_depth:
            return []

        callers = repo.find_elements_calling(
            target_id=node_id,
            scope=scope,
            repository=repository,
            username=username,
            limit=10,  # Limit branching
        )

        children = []
        for caller in callers:
            caller_id = caller.get("element_id")
            if caller_id in visited:
                # Cycle detected, mark but don't recurse
                children.append(
                    {
                        "element_id": caller_id,
                        "hash_id": caller.get("hash_id"),
                        "name": caller.get("name"),
                        "type": caller.get("element_type"),
                        "file": caller.get("relative_path"),
                        "line": caller.get("line_start"),
                        "depth": depth + 1,
                        "cycle": True,
                    }
                )
                continue

            visited.add(caller_id)
            node = {
                "element_id": caller_id,
                "hash_id": caller.get("hash_id"),
                "name": caller.get("name"),
                "type": caller.get("element_type"),
                "file": caller.get("relative_path"),
                "line": caller.get("line_start"),
                "depth": depth + 1,
                "callers": build_callers_tree(caller_id, depth + 1),
            }
            children.append(node)

        return children

    def build_callees_tree(node_id: str, depth: int) -> list[dict[str, Any]]:
        """Recursively build tree of callees."""
        if depth >= max_depth:
            return []

        calls = repo.get_calls(node_id)
        children = []

        for call in calls:
            resolved_id = call.get("resolved_id")
            if not resolved_id:
                # Unresolved call - just record the name
                children.append(
                    {
                        "name": call.get("name"),
                        "receiver": call.get("receiver"),
                        "line": call.get("line"),
                        "depth": depth + 1,
                        "unresolved": True,
                    }
                )
                continue

            if resolved_id in visited:
                # Cycle detected
                children.append(
                    {
                        "element_id": resolved_id,
                        "name": call.get("name"),
                        "line": call.get("line"),
                        "depth": depth + 1,
                        "cycle": True,
                    }
                )
                continue

            visited.add(resolved_id)

            # Get resolved element info
            resolved_doc = repo.get_document(resolved_id)
            if resolved_doc:
                node = {
                    "element_id": resolved_id,
                    "hash_id": resolved_doc.get("hash_id"),
                    "name": resolved_doc.get("name"),
                    "type": resolved_doc.get("element_type"),
                    "file": resolved_doc.get("relative_path"),
                    "line": resolved_doc.get("line_start"),
                    "depth": depth + 1,
                    "callees": build_callees_tree(resolved_id, depth + 1),
                }
                children.append(node)
            else:
                # Resolved ID but no document found
                children.append(
                    {
                        "element_id": resolved_id,
                        "name": call.get("name"),
                        "line": call.get("line"),
                        "depth": depth + 1,
                        "missing": True,
                    }
                )

        return children

    if direction in ("callers", "both"):
        result["callers"] = build_callers_tree(root_element_id, 0)
    if direction in ("callees", "both"):
        # Reset visited for callees direction if doing both
        if direction == "both":
            visited = {root_element_id}
        result["callees"] = build_callees_tree(root_element_id, 0)

    return result


# Entry point decorators to exclude (functions with these are likely called externally)
_ENTRY_POINT_DECORATORS = {
    # Web frameworks
    "app.route",
    "route",
    "get",
    "post",
    "put",
    "delete",
    "patch",
    "head",
    "options",
    "router.get",
    "router.post",
    "router.put",
    "router.delete",
    "router.patch",
    "api_view",
    "action",
    "endpoint",
    # FastAPI
    "app.get",
    "app.post",
    "app.put",
    "app.delete",
    "app.patch",
    "app.websocket",
    "websocket",
    "depends",
    "Depends",
    # CLI
    "click.command",
    "command",
    "click.group",
    "group",
    "typer.command",
    "typer.callback",
    "argparse",
    # Testing
    "pytest.fixture",
    "fixture",
    "pytest.mark",
    "mark.",
    "parametrize",
    "pytest.parametrize",
    # Class internals
    "property",
    "staticmethod",
    "classmethod",
    "abstractmethod",
    "abstractproperty",
    "cached_property",
    "functools.cached_property",
    # Async/background
    "celery.task",
    "task",
    "dramatiq.actor",
    "actor",
    "asyncio",
    "async_generator",
    # Event handlers
    "on_event",
    "lifespan",
    "startup",
    "shutdown",
    "event_handler",
    "listener",
    "subscriber",
    # Serialization/validation
    "validator",
    "field_validator",
    "model_validator",
    "serializer",
    "deserializer",
    # Registration patterns
    "register",
    "registry",
    "handler",
    "callback",
    "hook",
}

# Names to exclude (entry points, magic methods, common public API patterns)
_EXCLUDED_NAMES = {
    # Entry points
    "main",
    "__main__",
    "run",
    "start",
    "execute",
    "cli",
    # Magic methods
    "__init__",
    "__new__",
    "__del__",
    "__post_init__",
    "__str__",
    "__repr__",
    "__hash__",
    "__eq__",
    "__ne__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__cmp__",
    "__add__",
    "__sub__",
    "__mul__",
    "__truediv__",
    "__floordiv__",
    "__mod__",
    "__pow__",
    "__and__",
    "__or__",
    "__xor__",
    "__radd__",
    "__rsub__",
    "__rmul__",
    "__rtruediv__",
    "__iadd__",
    "__isub__",
    "__imul__",
    "__itruediv__",
    "__neg__",
    "__pos__",
    "__abs__",
    "__invert__",
    "__iter__",
    "__next__",
    "__getitem__",
    "__setitem__",
    "__delitem__",
    "__len__",
    "__call__",
    "__enter__",
    "__exit__",
    "__contains__",
    "__bool__",
    "__index__",
    "__int__",
    "__float__",
    "__getattr__",
    "__setattr__",
    "__delattr__",
    "__getattribute__",
    "__get__",
    "__set__",
    "__delete__",  # Descriptors
    "__class_getitem__",
    "__init_subclass__",
    "__set_name__",
    "__reduce__",
    "__reduce_ex__",
    "__getstate__",
    "__setstate__",
    "__copy__",
    "__deepcopy__",
    "__format__",
    "__sizeof__",
    "__bytes__",
    "__aiter__",
    "__anext__",
    "__await__",  # Async magic
    "__aenter__",
    "__aexit__",
    # Testing
    "setUp",
    "tearDown",
    "setUpClass",
    "tearDownClass",
    "setUpModule",
    "tearDownModule",
    # Common callback/handler names
    "on_start",
    "on_stop",
    "on_error",
    "on_success",
    "handle",
    "process",
    "callback",
}


def find_dead_code(
    repo: Repository,
    scope: str,
    repository: str,
    username: str | None = None,
    include_tests: bool = False,
) -> dict[str, Any]:
    """Find functions/methods that are never called.

    Excludes:
    - Entry points (decorated with @app.route, @click.command, etc.)
    - Functions named 'main', '__main__'
    - Test functions (if include_tests=False)
    - Magic methods (__init__, __str__, etc.)

    Args:
        repo: Search repository.
        scope: Repository scope (required).
        repository: Repository name (required).
        username: User branch (defaults to "main").
        include_tests: Whether to include test functions in dead code check.

    Returns:
        Dict with potentially_dead list and statistics.
    """
    username = username or "main"

    client = repo._get_client()

    # Get all functions and methods in the repository
    filters = [
        {"term": {"scope": scope}},
        {"term": {"repository": repository}},
        {"term": {"username": username}},
        {"terms": {"element_type": ["function", "method"]}},
    ]

    if not include_tests:
        filters.append({"term": {"is_test": False}})

    search_result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {"bool": {"filter": filters}},
            "_source": [
                "element_id",
                "hash_id",
                "name",
                "element_type",
                "relative_path",
                "line_start",
                "decorators",
                "is_test",
                "summary",
                "visibility",
                "level",
            ],
            "size": 2000,
        },
    )

    hits = search_result.get("hits", {}).get("hits", [])

    # Filter out entry points and find uncalled functions
    potentially_dead: list[dict[str, Any]] = []
    excluded_count = 0
    called_count = 0

    for hit in hits:
        source = hit["_source"]
        element_id = source.get("element_id")
        name = source.get("name")
        element_type = source.get("element_type")
        decorators = source.get("decorators", []) or []
        level = source.get("level", 2)  # Default to function level

        # Skip excluded names
        if name in _EXCLUDED_NAMES:
            excluded_count += 1
            continue

        # Skip public module-level functions (they're likely public API)
        # Level 2 = function, level 3 = method/nested
        # Public = doesn't start with underscore
        if element_type == "function" and level == 2 and not name.startswith("_"):
            excluded_count += 1
            continue

        # Skip if has entry point decorator
        is_entry_point = False
        for dec in decorators:
            # Decorator might be "app.route" or just the decorator name in the list
            dec_lower = dec.lower() if isinstance(dec, str) else ""
            for ep in _ENTRY_POINT_DECORATORS:
                if ep in dec_lower or dec_lower.endswith(ep):
                    is_entry_point = True
                    break
            if is_entry_point:
                break

        if is_entry_point:
            excluded_count += 1
            continue

        # Check if anything calls this element
        callers = repo.find_elements_calling(
            target_id=element_id,
            scope=scope,
            repository=repository,
            username=username,
            limit=1,  # Just need to know if there's at least one
        )

        if not callers:
            potentially_dead.append(
                {
                    "element_id": element_id,
                    "hash_id": source.get("hash_id"),
                    "name": name,
                    "type": source.get("element_type"),
                    "file": source.get("relative_path"),
                    "line": source.get("line_start"),
                    "summary": source.get("summary", ""),
                    "is_test": source.get("is_test", False),
                }
            )
        else:
            called_count += 1

    return {
        "potentially_dead": potentially_dead,
        "stats": {
            "total_functions": len(hits),
            "excluded_entry_points": excluded_count,
            "called": called_count,
            "potentially_dead": len(potentially_dead),
        },
    }


def find_entry_points(
    repo: Repository,
    scope: str,
    repository: str,
    username: str | None = None,
) -> dict[str, Any]:
    """Find entry points: HTTP handlers, CLI commands, test fixtures, main functions.

    Detection patterns:
    - Decorator patterns: @app.route, @click.command, @pytest.fixture, etc.
    - Named: main, __main__
    - Called externally with no internal callers

    Args:
        repo: Search repository.
        scope: Repository scope (required).
        repository: Repository name (required).
        username: User branch (defaults to "main").

    Returns:
        Entry points grouped by type (http, cli, test, main, other).
    """
    username = username or "main"

    # Decorator categories
    http_decorators = {
        "route",
        "app.route",
        "get",
        "post",
        "put",
        "delete",
        "patch",
        "api_view",
        "action",
        "api.route",
        "blueprint.route",
    }
    cli_decorators = {
        "click.command",
        "command",
        "click.group",
        "group",
        "click.option",
        "argument",
    }
    test_decorators = {"pytest.fixture", "fixture", "pytest.mark"}
    async_decorators = {"celery.task", "task", "dramatiq.actor", "actor"}

    client = repo._get_client()

    # Get all functions and methods
    filters = [
        {"term": {"scope": scope}},
        {"term": {"repository": repository}},
        {"term": {"username": username}},
        {"terms": {"element_type": ["function", "method"]}},
    ]

    search_result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {"bool": {"filter": filters}},
            "_source": [
                "element_id",
                "hash_id",
                "name",
                "element_type",
                "relative_path",
                "line_start",
                "decorators",
                "is_test",
                "summary",
                "http_routes",
                "cli_commands",
            ],
            "size": 2000,
        },
    )

    hits = search_result.get("hits", {}).get("hits", [])

    # Categorize entry points
    http_handlers: list[dict[str, Any]] = []
    cli_commands: list[dict[str, Any]] = []
    test_fixtures: list[dict[str, Any]] = []
    main_functions: list[dict[str, Any]] = []
    async_tasks: list[dict[str, Any]] = []
    other_entry_points: list[dict[str, Any]] = []

    for hit in hits:
        source = hit["_source"]
        name = source.get("name")
        decorators = source.get("decorators", []) or []

        entry = {
            "element_id": source.get("element_id"),
            "hash_id": source.get("hash_id"),
            "name": name,
            "type": source.get("element_type"),
            "file": source.get("relative_path"),
            "line": source.get("line_start"),
            "summary": source.get("summary", ""),
            "decorators": decorators,
            "http_routes": source.get("http_routes", []),
            "cli_commands": source.get("cli_commands", []),
        }

        # Check for main function
        if name in ("main", "__main__"):
            main_functions.append(entry)
            continue

        # Check decorators
        decorator_matched = False
        for dec in decorators:
            dec_lower = dec.lower() if isinstance(dec, str) else ""

            # HTTP handlers
            for pattern in http_decorators:
                if pattern in dec_lower:
                    http_handlers.append(entry)
                    decorator_matched = True
                    break

            if decorator_matched:
                break

            # CLI commands
            for pattern in cli_decorators:
                if pattern in dec_lower:
                    cli_commands.append(entry)
                    decorator_matched = True
                    break

            if decorator_matched:
                break

            # Test fixtures
            for pattern in test_decorators:
                if pattern in dec_lower:
                    test_fixtures.append(entry)
                    decorator_matched = True
                    break

            if decorator_matched:
                break

            # Async tasks
            for pattern in async_decorators:
                if pattern in dec_lower:
                    async_tasks.append(entry)
                    decorator_matched = True
                    break

            if decorator_matched:
                break

    return {
        "http": http_handlers,
        "cli": cli_commands,
        "test": test_fixtures,
        "main": main_functions,
        "async_tasks": async_tasks,
        "other": other_entry_points,
        "stats": {
            "total_http": len(http_handlers),
            "total_cli": len(cli_commands),
            "total_test": len(test_fixtures),
            "total_main": len(main_functions),
            "total_async": len(async_tasks),
            "total": len(http_handlers)
            + len(cli_commands)
            + len(test_fixtures)
            + len(main_functions)
            + len(async_tasks),
        },
    }
