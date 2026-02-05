"""Dependency analysis tools."""

from __future__ import annotations

from typing import Any

from shared.db.elasticsearch import ElasticsearchRepository


def _find_file_element(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    username: str,
    relative_path: str,
) -> dict[str, Any] | None:
    """Find the file element for a given path."""
    client = es._get_client()
    result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                        {"term": {"relative_path": relative_path}},
                        {"term": {"element_type": "file"}},
                    ]
                }
            },
            "size": 1,
        },
    )

    hits = result.get("hits", {}).get("hits", [])
    return hits[0]["_source"] if hits else None


def find_dependencies(
    es: ElasticsearchRepository,
    file_path: str | None = None,
    element_id: str | None = None,
    scope: str | None = None,
    repository: str | None = None,
    username: str | None = None,
) -> dict[str, Any]:
    """Get imports for a file.

    Shows what a file depends on - both internal (from within the repo)
    and external (third-party packages) imports.

    Args:
        es: Elasticsearch repository.
        file_path: Relative file path (e.g., "src/utils.py").
        element_id: Or provide file element ID directly.
        scope: Filter by scope (required if using file_path).
        repository: Filter by repository (required if using file_path).
        username: User branch (defaults to "main").

    Returns:
        Dict with:
        - internal_imports: Imports from within the repo
        - external_imports: Imports from external packages
        - all_imports: Combined list
        - file_info: File element details

    Raises:
        ValueError: If neither file_path nor element_id provided, or file not found.
    """
    username = username or "main"

    # Get the file element (supports both element_id and hash_id)
    if element_id:
        file_doc = es.get_document_by_id_or_hash(element_id)
        if not file_doc:
            raise ValueError(f"Element not found: {element_id}")
        if file_doc.get("element_type") != "file":
            raise ValueError(f"Element is not a file: {element_id}")
        scope = file_doc.get("scope")
        repository = file_doc.get("repository")
    elif file_path:
        if not scope or not repository:
            raise ValueError("scope and repository required when using file_path")
        file_doc = _find_file_element(es, scope, repository, username, file_path)
        if not file_doc:
            raise ValueError(f"File not found: {file_path}")
        element_id = file_doc.get("element_id")
    else:
        raise ValueError("Either file_path or element_id required")

    # Get imports from the file element
    imports = es.get_imports(element_id)

    # Classify imports as internal vs external
    internal_imports: list[dict[str, Any]] = []
    external_imports: list[dict[str, Any]] = []

    # Get all file paths in the repo to detect internal imports
    client = es._get_client()
    file_result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                        {"term": {"element_type": "file"}},
                    ]
                }
            },
            "_source": ["relative_path"],
            "size": 2000,
        },
    )
    repo_files = {
        hit["_source"]["relative_path"] for hit in file_result.get("hits", {}).get("hits", [])
    }

    # Also create a set of potential module names from file paths
    # e.g., "src/utils.py" -> "src.utils", "utils"
    repo_modules: set[str] = set()
    for path in repo_files:
        # Remove .py extension and convert to module name
        if path.endswith(".py"):
            module_path = path[:-3].replace("/", ".").replace("\\", ".")
            repo_modules.add(module_path)
            # Also add the last component
            parts = module_path.split(".")
            if parts:
                repo_modules.add(parts[-1])

    for imp in imports:
        module = imp.get("module", "")
        name = imp.get("name", "")

        import_entry = {
            "name": name,
            "module": module,
            "alias": imp.get("alias"),
            "line": imp.get("line"),
        }

        # Check if this is an internal import
        is_internal = False

        # Check if module starts with "." (relative import)
        if module and module.startswith(".") or module in repo_modules:
            is_internal = True
        # Check common patterns for internal imports
        elif module:
            # Check if any part of the module path matches a repo module
            parts = module.split(".")
            for i in range(len(parts)):
                partial = ".".join(parts[: i + 1])
                if partial in repo_modules:
                    is_internal = True
                    break

        if is_internal:
            internal_imports.append(import_entry)
        else:
            external_imports.append(import_entry)

    return {
        "file_info": {
            "path": file_doc.get("relative_path"),
            "element_id": element_id,
        },
        "internal_imports": internal_imports,
        "external_imports": external_imports,
        "all_imports": imports,
        "stats": {
            "total": len(imports),
            "internal": len(internal_imports),
            "external": len(external_imports),
        },
    }


def find_dependents(
    es: ElasticsearchRepository,
    module: str,
    scope: str,
    repository: str,
    username: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Find files that import a module.

    Shows what depends on a given module - useful for impact analysis
    before making changes to a module.

    Args:
        es: Elasticsearch repository.
        module: Module name to search for (e.g., "utils", "shared.config", "./utils").
        scope: Repository scope (required).
        repository: Repository name (required).
        username: User branch (defaults to "main").
        limit: Maximum files to return.

    Returns:
        Dict with:
        - module: The searched module
        - dependents: List of files that import the module
        - total: Number of dependents found
    """
    username = username or "main"

    # Find files that import this module
    dependents = es.find_elements_importing(
        module=module,
        scope=scope,
        repository=repository,
        username=username,
        limit=limit,
    )

    formatted_dependents = []
    for dep in dependents:
        formatted_dependents.append(
            {
                "file": dep.get("relative_path"),
                "element_id": dep.get("element_id"),
                "hash_id": dep.get("hash_id"),
                "language": dep.get("language"),
            }
        )

    return {
        "module": module,
        "dependents": formatted_dependents,
        "total": len(formatted_dependents),
    }


def dependency_graph(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    username: str | None = None,
    internal_only: bool = True,
) -> dict[str, Any]:
    """Build a module-level dependency graph.

    Creates a directed graph of module dependencies within the repository.
    Useful for understanding code architecture and detecting circular dependencies.

    Args:
        es: Elasticsearch repository.
        scope: Repository scope (required).
        repository: Repository name (required).
        username: User branch (defaults to "main").
        internal_only: Only include internal imports (default True).

    Returns:
        Dict with:
        - nodes: List of module names (files)
        - edges: List of {from, to} dependency links
        - cycles: List of circular dependency chains (if any)
        - stats: Graph statistics
    """
    username = username or "main"

    client = es._get_client()

    # Get all file elements with their imports
    result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                        {"term": {"element_type": "file"}},
                    ]
                }
            },
            "_source": ["element_id", "relative_path", "imports"],
            "size": 2000,
        },
    )

    hits = result.get("hits", {}).get("hits", [])

    # Build set of all file paths for internal import detection
    file_paths = {hit["_source"]["relative_path"] for hit in hits}

    # Build module name mapping from file paths
    path_to_module: dict[str, str] = {}
    module_to_path: dict[str, str] = {}

    for path in file_paths:
        if path.endswith(".py"):
            # Convert path to module name
            module_name = path[:-3].replace("/", ".").replace("\\", ".")
            path_to_module[path] = module_name
            module_to_path[module_name] = path
            # Also map just the filename without extension
            parts = module_name.split(".")
            if parts:
                module_to_path[parts[-1]] = path

    # Build nodes and edges
    nodes: list[str] = list(file_paths)
    edges: list[dict[str, str]] = []
    edge_set: set[tuple[str, str]] = set()

    for hit in hits:
        source = hit["_source"]
        from_path = source.get("relative_path")
        imports = source.get("imports", []) or []

        for imp in imports:
            module = imp.get("module", "")
            if not module:
                continue

            # Determine the target path
            to_path = None

            # Handle relative imports
            if module.startswith("."):
                # Relative import - resolve relative to current file
                from_dir = "/".join(from_path.split("/")[:-1])
                rel_module = module.lstrip(".")
                if rel_module:
                    # e.g., ".utils" from "src/main.py" -> "src/utils.py"
                    if from_dir:
                        to_path = f"{from_dir}/{rel_module.replace('.', '/')}.py"
                    else:
                        to_path = f"{rel_module.replace('.', '/')}.py"
                else:
                    # Just "." means current package's __init__.py
                    if from_dir:
                        to_path = f"{from_dir}/__init__.py"
            else:
                # Absolute import - try to find matching file
                # Try direct module path
                possible_paths = [
                    f"{module.replace('.', '/')}.py",
                    f"{module.replace('.', '/')}/__init__.py",
                ]
                for pp in possible_paths:
                    if pp in file_paths:
                        to_path = pp
                        break

                # Also check if module name directly maps to a known module
                if not to_path and module in module_to_path:
                    to_path = module_to_path[module]

            # Skip if target not found or if filtering external imports
            if not to_path or to_path not in file_paths:
                if internal_only:
                    continue
                # For external imports when internal_only=False, we'd need to handle differently
                # For now, skip external imports entirely
                continue

            # Add edge if not already present
            edge_key = (from_path, to_path)
            if edge_key not in edge_set:
                edge_set.add(edge_key)
                edges.append({"from": from_path, "to": to_path})

    # Detect cycles using DFS
    cycles = _detect_cycles(nodes, edges)

    return {
        "nodes": nodes,
        "edges": edges,
        "cycles": cycles,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "cycle_count": len(cycles),
            "has_cycles": len(cycles) > 0,
        },
    }


def _detect_cycles(nodes: list[str], edges: list[dict[str, str]]) -> list[list[str]]:
    """Detect circular dependencies in a directed graph.

    Uses DFS to find cycles.

    Args:
        nodes: List of node names.
        edges: List of {from, to} edges.

    Returns:
        List of cycles, where each cycle is a list of nodes forming the cycle.
    """
    # Build adjacency list
    graph: dict[str, list[str]] = {node: [] for node in nodes}
    for edge in edges:
        from_node = edge["from"]
        to_node = edge["to"]
        if from_node in graph:
            graph[from_node].append(to_node)

    cycles: list[list[str]] = []
    visited: set[str] = set()
    rec_stack: set[str] = set()
    path: list[str] = []

    def dfs(node: str) -> None:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in rec_stack:
                # Found a cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                # Avoid duplicate cycles (by normalizing)
                normalized = _normalize_cycle(cycle)
                if normalized not in [_normalize_cycle(c) for c in cycles]:
                    cycles.append(cycle)

        path.pop()
        rec_stack.remove(node)

    for node in nodes:
        if node not in visited:
            dfs(node)

    return cycles


def _normalize_cycle(cycle: list[str]) -> tuple[str, ...]:
    """Normalize a cycle for comparison by starting from the lexicographically smallest element."""
    if not cycle:
        return ()
    # Remove the duplicate end element if present
    if len(cycle) > 1 and cycle[0] == cycle[-1]:
        cycle = cycle[:-1]
    # Find the smallest element's index
    min_idx = cycle.index(min(cycle))
    # Rotate to start from smallest
    normalized = cycle[min_idx:] + cycle[:min_idx]
    return tuple(normalized)
