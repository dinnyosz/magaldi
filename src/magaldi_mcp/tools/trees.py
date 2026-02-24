"""CLI command and HTTP route tree tools."""

from __future__ import annotations

from typing import Any

from shared.db.store import Repository

from ._utils import _resolve_scope_repo


def get_command_tree(
    repo: Repository,
    scope: str | None = None,
    repository: str | None = None,
    username: str | None = None,
) -> dict[str, Any]:
    """Get CLI command tree with full command paths.

    Returns the hierarchical structure of CLI commands (Click, Typer, etc.)
    with their full invocation paths like "magaldi web serve".

    Args:
        repo: Search repository.
        scope: Repository scope (auto-detected from magaldi.yaml if not provided).
        repository: Repository name (auto-detected from magaldi.yaml if not provided).
        username: User branch (defaults to "main").

    Returns:
        Dict with 'tree' (hierarchical structure) and 'commands' (flat list).
    """
    from magaldi_core.analysis.relationships import ExternalRefType, RelationshipType
    from shared.db.repositories.relationships import RelationshipsRepository

    scope, repository = _resolve_scope_repo(scope, repository)
    if not scope or not repository:
        raise ValueError(
            "scope and repository are required. Either provide them explicitly "
            "or create a magaldi.yaml file in your project root."
        )
    username = username or "main"
    rel_repo = RelationshipsRepository(repo.config)

    # Get CLI entry point external ref
    cli_refs = rel_repo.get_external_refs(
        scope, repository,
        ref_type=ExternalRefType.CLI_ENTRY_POINT.value,
    )
    entry_point_name = ""
    if cli_refs:
        entry_point_name = cli_refs[0].get("name", "")

    # Get all BELONGS_TO_GROUP relationships
    relationships = rel_repo.get_relationships_by_type(
        scope, repository, username, RelationshipType.BELONGS_TO_GROUP.value
    )

    if not relationships:
        return {
            "tree": {
                "name": entry_point_name or "root",
                "is_group": True,
                "children": [],
            },
            "commands": [],
            "entry_point": entry_point_name,
        }

    # Collect all unique source hashes to fetch element details
    source_hashes = {r["source_hash"] for r in relationships}
    target_hashes = {r["target_hash"] for r in relationships}
    all_hashes = source_hashes | target_hashes

    # Fetch element details
    hash_to_element = {}
    if all_hashes:
        client = repo._get_client()
        hash_query = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                        {"terms": {"hash_id": list(all_hashes)}},
                    ]
                }
            },
            "size": len(all_hashes),
            "_source": ["hash_id", "element_id", "name", "relative_path", "line_start", "summary"],
        }
        result = client.search(index="magaldi-code-elements", body=hash_query)
        for hit in result.get("hits", {}).get("hits", []):
            source = hit["_source"]
            hash_to_element[source["hash_id"]] = source

    # Build children map
    children_map: dict[str, list[dict]] = {}
    commands_flat: list[dict[str, Any]] = []

    for rel in relationships:
        parent_hash = rel["target_hash"]
        details = rel.get("details", {})

        if parent_hash not in children_map:
            children_map[parent_hash] = []

        elem = hash_to_element.get(rel["source_hash"], {})

        child_info = {
            "name": details.get("command_name", elem.get("name", "")),
            "hash_id": rel["source_hash"],
            "full_command": details.get("full_command", ""),
            "is_group": details.get("is_group", False),
            "file": elem.get("relative_path", ""),
            "line": elem.get("line_start"),
            "summary": elem.get("summary"),
        }
        children_map[parent_hash].append(child_info)

        # Add to flat list if it's a command (not a group)
        if not details.get("is_group", False):
            commands_flat.append({
                "full_command": details.get("full_command", ""),
                "function_name": elem.get("name", ""),
                "hash_id": rel["source_hash"],
                "file": elem.get("relative_path", ""),
                "line": elem.get("line_start"),
                "summary": elem.get("summary"),
            })

    # Find root groups (groups that are targets but not sources)
    root_hashes = target_hashes - source_hashes

    # Build tree recursively
    def build_subtree(hash_id: str) -> dict[str, Any]:
        elem = hash_to_element.get(hash_id, {})
        children = children_map.get(hash_id, [])

        node = {
            "name": elem.get("name", ""),
            "hash_id": hash_id,
            "is_group": True,  # If it's in children_map, it's a group
            "file": elem.get("relative_path"),
            "line": elem.get("line_start"),
            "children": [],
        }

        for child in children:
            if child["is_group"] and child["hash_id"] in children_map:
                # Recurse for groups
                node["children"].append(build_subtree(child["hash_id"]))
            else:
                # Leaf command
                node["children"].append({
                    "name": child["name"],
                    "hash_id": child["hash_id"],
                    "full_command": child["full_command"],
                    "is_group": child["is_group"],
                    "file": child["file"],
                    "line": child["line"],
                    "summary": child["summary"],
                })

        return node

    # Build root node
    root = {
        "name": entry_point_name or "root",
        "is_group": True,
        "children": [],
    }

    for root_hash in root_hashes:
        root["children"].append(build_subtree(root_hash))

    return {
        "tree": root,
        "commands": commands_flat,
        "entry_point": entry_point_name,
    }


def get_route_tree(
    repo: Repository,
    scope: str | None = None,
    repository: str | None = None,
    username: str | None = None,
) -> dict[str, Any]:
    """Get HTTP route tree with full URL paths.

    Returns the hierarchical structure of HTTP routes grouped by router module,
    with their full URL paths like "/api/v1/users/{id}".

    Args:
        repo: Search repository.
        scope: Repository scope (auto-detected from magaldi.yaml if not provided).
        repository: Repository name (auto-detected from magaldi.yaml if not provided).
        username: User branch (defaults to "main").

    Returns:
        Dict with 'routers' (list of router info with their routes) and 'routes' (flat list).
    """
    from magaldi_core.analysis.relationships import ExternalRefType, RelationshipType
    from shared.db.repositories.relationships import RelationshipsRepository

    scope, repository = _resolve_scope_repo(scope, repository)
    if not scope or not repository:
        raise ValueError(
            "scope and repository are required. Either provide them explicitly "
            "or create a magaldi.yaml file in your project root."
        )
    username = username or "main"
    rel_repo = RelationshipsRepository(repo.config)

    # Get all HTTP router external refs
    router_refs = rel_repo.get_external_refs(
        scope, repository,
        ref_type=ExternalRefType.HTTP_ROUTER.value,
    )

    # Get all BELONGS_TO_ROUTER relationships
    relationships = rel_repo.get_relationships_by_type(
        scope, repository, username, RelationshipType.BELONGS_TO_ROUTER.value
    )

    if not relationships and not router_refs:
        return {
            "routers": [],
            "routes": [],
            "api_prefix": "",
        }

    # Group relationships by router (target_hash is the router ref_id)
    routes_by_router: dict[str, list[dict]] = {}
    routes_flat: list[dict[str, Any]] = []

    # Collect all source hashes to fetch element details
    source_hashes = {r["source_hash"] for r in relationships}

    # Fetch element details
    hash_to_element = {}
    if source_hashes:
        client = repo._get_client()
        hash_query = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                        {"terms": {"hash_id": list(source_hashes)}},
                    ]
                }
            },
            "size": len(source_hashes),
            "_source": ["hash_id", "element_id", "name", "relative_path", "line_start", "summary"],
        }
        result = client.search(index="magaldi-code-elements", body=hash_query)
        for hit in result.get("hits", {}).get("hits", []):
            source = hit["_source"]
            hash_to_element[source["hash_id"]] = source

    # Process relationships
    for rel in relationships:
        router_id = rel["target_hash"]
        details = rel.get("details", {})
        elem = hash_to_element.get(rel["source_hash"], {})

        route_info = {
            "method": details.get("method", "GET"),
            "path": details.get("path", ""),
            "full_url": details.get("full_url", ""),
            "hash_id": rel["source_hash"],
            "function_name": elem.get("name", ""),
            "file": elem.get("relative_path", ""),
            "line": elem.get("line_start"),
            "summary": elem.get("summary"),
            "path_params": details.get("path_params", []),
        }

        if router_id not in routes_by_router:
            routes_by_router[router_id] = []
        routes_by_router[router_id].append(route_info)
        routes_flat.append(route_info)

    # Build router info with their routes
    routers: list[dict[str, Any]] = []
    api_prefix = ""

    for ref in router_refs:
        ref_id = ref.get("ref_id", "")
        metadata = ref.get("metadata", {})
        api_prefix = metadata.get("api_prefix", api_prefix)

        router_routes = routes_by_router.get(ref_id, [])
        # Sort routes by path
        router_routes.sort(key=lambda r: (r["path"], r["method"]))

        routers.append({
            "name": ref.get("name", ""),
            "file": metadata.get("file_path", ""),
            "route_count": len(router_routes),
            "common_prefix": metadata.get("common_prefix", ""),
            "routes": router_routes,
        })

    # Sort routers by name
    routers.sort(key=lambda r: r["name"])

    # Sort flat routes by full_url
    routes_flat.sort(key=lambda r: (r["full_url"], r["method"]))

    return {
        "routers": routers,
        "routes": routes_flat,
        "api_prefix": api_prefix,
    }
