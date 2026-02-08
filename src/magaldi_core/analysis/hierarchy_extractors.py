"""Hierarchy extractors for CLI commands and HTTP routes.

This module extracts parent-child relationships from code elements to build:
- CLI command trees (Click groups → commands)
- HTTP route trees (routers → routes with full URL paths)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from magaldi_core.analysis.relationships import (
    CodeRelationship,
    CommandTreeNode,
    ExternalReference,
    ExternalRefType,
    RelationshipType,
    generate_relationship_id,
    generate_relationship_key,
    get_priority_for_username,
)


@dataclass
class ElementInfo:
    """Minimal element info needed for hierarchy extraction."""

    hash_id: str
    element_id: str
    name: str
    file_path: str
    line: int
    decorators: list[str]
    decorator_details: list[dict[str, Any]]
    summary: str | None = None


# =============================================================================
# CLI HIERARCHY EXTRACTOR
# =============================================================================


class CliHierarchyExtractor:
    """Extract Click/Typer CLI command hierarchies.

    Builds BELONGS_TO_GROUP relationships between commands and their parent groups,
    and tracks the full command path (e.g., "magaldi web serve").
    """

    # Suffixes that indicate a group definition
    GROUP_SUFFIXES = {".group"}

    # Suffixes that indicate a command definition
    COMMAND_SUFFIXES = {".command"}

    def __init__(
        self,
        scope: str,
        repository: str,
        username: str,
        cli_entry_point: str | None = None,
    ):
        """Initialize extractor.

        Args:
            scope: Repository scope
            repository: Repository name
            username: Username for the branch
            cli_entry_point: CLI entry point name (e.g., "magaldi" from pyproject.toml)
        """
        self.scope = scope
        self.repository = repository
        self.username = username
        self.cli_entry_point = cli_entry_point or ""
        self.priority = get_priority_for_username(username)

    def extract(self, elements: list[ElementInfo]) -> tuple[
        list[CodeRelationship],
        list[ExternalReference],
    ]:
        """Extract CLI hierarchy relationships from elements.

        Args:
            elements: List of code elements with decorator info

        Returns:
            Tuple of (relationships, external_references)
        """
        relationships: list[CodeRelationship] = []
        external_refs: list[ExternalReference] = []

        # Step 1: Find all groups (functions with .group decorator)
        # Maps: function_name -> ElementInfo
        groups: dict[str, ElementInfo] = {}

        # Step 2: Find all commands/groups with their parent reference
        # List of (element, parent_name, is_group, command_name)
        commands: list[tuple[ElementInfo, str, bool, str]] = []

        for element in elements:
            for dec_detail in element.decorator_details:
                dec_name = dec_detail.get("name", "")
                dec_args = dec_detail.get("args", "")

                # Check if this defines a group
                if any(dec_name.endswith(suffix) for suffix in self.GROUP_SUFFIXES):
                    groups[element.name] = element

                    # Extract parent reference (prefix before .group)
                    parent_name = self._extract_parent_name(dec_name)
                    if parent_name:
                        # Extract group name from args or use function name
                        group_name = self._extract_name_from_args(dec_args) or element.name
                        commands.append((element, parent_name, True, group_name))

                # Check if this defines a command
                elif any(dec_name.endswith(suffix) for suffix in self.COMMAND_SUFFIXES):
                    parent_name = self._extract_parent_name(dec_name)
                    if parent_name:
                        # Extract command name from args or use function name
                        command_name = self._extract_name_from_args(dec_args) or element.name
                        commands.append((element, parent_name, False, command_name))

        # Step 3: Build relationships
        for element, parent_name, is_group, cmd_name in commands:
            parent_element = groups.get(parent_name)

            if parent_element:
                # Build the group path
                group_path = self._build_group_path(parent_name, groups)
                full_command = self._build_full_command(group_path, cmd_name)

                # Create relationship
                rel_type = RelationshipType.BELONGS_TO_GROUP

                relationship = CodeRelationship(
                    relationship_id=generate_relationship_id(
                        element.hash_id,
                        parent_element.hash_id,
                        rel_type,
                        self.scope,
                        self.repository,
                        self.username,
                    ),
                    relationship_key=generate_relationship_key(
                        element.hash_id,
                        parent_element.hash_id,
                        rel_type,
                        self.scope,
                        self.repository,
                    ),
                    source_hash=element.hash_id,
                    target_hash=parent_element.hash_id,
                    relationship_type=rel_type,
                    scope=self.scope,
                    repository=self.repository,
                    username=self.username,
                    priority=self.priority,
                    confidence=1.0,
                    line=element.line,
                    details={
                        "command_name": cmd_name,
                        "is_group": is_group,
                        "parent_group": parent_name,
                        "group_path": group_path,
                        "full_command": full_command,
                    },
                )
                relationships.append(relationship)

        # Step 4: Create external ref for CLI entry point if provided
        if self.cli_entry_point:
            ref = ExternalReference(
                ref_id=f"cli_entry:{self.scope}:{self.repository}:{self.cli_entry_point}",
                ref_type=ExternalRefType.CLI_ENTRY_POINT,
                name=self.cli_entry_point,
                description=f"CLI entry point for {self.scope}/{self.repository}",
                scope=self.scope,
                repository=self.repository,
                username=self.username,
            )
            external_refs.append(ref)

        return relationships, external_refs

    def _extract_parent_name(self, decorator_name: str) -> str | None:
        """Extract parent group name from decorator.

        Examples:
            "web.command" -> "web"
            "main.group" -> "main"
            "click.command" -> None (no parent reference)
        """
        if "." in decorator_name:
            parts = decorator_name.rsplit(".", 1)
            prefix = parts[0]
            # Ignore standard prefixes that aren't group references
            if prefix not in ("click", "typer", "app"):
                return prefix
        return None

    def _extract_name_from_args(self, args: str | None) -> str | None:
        """Extract command/group name from decorator arguments.

        Examples:
            '("serve")' -> "serve"
            '("parse", help="...")' -> "parse"
            None -> None
        """
        if not args:
            return None

        # Strip outer parentheses
        cleaned = args.strip()
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = cleaned[1:-1].strip()

        # Extract first quoted string
        match = re.match(r'["\']([^"\']+)["\']', cleaned)
        if match:
            return match.group(1)

        return None

    def _build_group_path(
        self, group_name: str, groups: dict[str, ElementInfo]
    ) -> list[str]:
        """Build the path from root to this group.

        Follows parent references up the chain.
        """
        path = [group_name]
        visited = {group_name}

        current = group_name
        while current in groups:
            element = groups[current]
            # Find parent reference in decorators
            parent = None
            for dec_detail in element.decorator_details:
                dec_name = dec_detail.get("name", "")
                if any(dec_name.endswith(suffix) for suffix in self.GROUP_SUFFIXES):
                    parent = self._extract_parent_name(dec_name)
                    break

            if parent and parent not in visited:
                path.insert(0, parent)
                visited.add(parent)
                current = parent
            else:
                break

        return path

    def _build_full_command(self, group_path: list[str], command_name: str) -> str:
        """Build full command string.

        Examples:
            ["main", "web"], "serve" -> "magaldi web serve"
            ["main"], "parse" -> "magaldi parse"
        """
        parts = []

        # Add CLI entry point if available
        if self.cli_entry_point:
            parts.append(self.cli_entry_point)

        # Add group path (skip "main" if it's the root and we have entry point)
        for group in group_path:
            if group == "main" and self.cli_entry_point:
                continue  # Skip "main" when we have entry point name
            parts.append(group)

        # Add command name
        parts.append(command_name)

        return " ".join(parts)

    def build_command_tree(
        self, elements: list[ElementInfo]
    ) -> CommandTreeNode:
        """Build a tree structure of CLI commands.

        Returns:
            Root CommandTreeNode with full hierarchy
        """
        relationships, _ = self.extract(elements)

        # Build lookup maps
        # hash_id -> element
        element_by_hash: dict[str, ElementInfo] = {e.hash_id: e for e in elements}

        # hash_id -> relationship details
        rel_by_source: dict[str, dict[str, Any]] = {}
        for rel in relationships:
            rel_by_source[rel.source_hash] = rel.details

        # Find root groups (groups with no parent relationship)
        all_sources = {rel.source_hash for rel in relationships}
        all_targets = {rel.target_hash for rel in relationships}
        root_hashes = all_targets - all_sources

        # Build children map
        children_map: dict[str, list[str]] = {}
        for rel in relationships:
            parent_hash = rel.target_hash
            if parent_hash not in children_map:
                children_map[parent_hash] = []
            children_map[parent_hash].append(rel.source_hash)

        # Create root node
        root = CommandTreeNode(
            name=self.cli_entry_point or "root",
            hash_id=None,
            full_path=self.cli_entry_point or "",
            is_group=True,
        )

        # Build tree recursively
        def build_node(hash_id: str, parent_path: str) -> CommandTreeNode:
            element = element_by_hash.get(hash_id)
            details = rel_by_source.get(hash_id, {})

            name = details.get("command_name", element.name if element else "unknown")
            is_group = details.get("is_group", False)
            full_path = details.get("full_command", f"{parent_path} {name}".strip())

            node = CommandTreeNode(
                name=name,
                hash_id=hash_id,
                full_path=full_path,
                is_group=is_group,
                description=element.summary if element else None,
                file_path=element.file_path if element else None,
                line=element.line if element else None,
            )

            # Add children
            for child_hash in children_map.get(hash_id, []):
                child_node = build_node(child_hash, full_path)
                node.children.append(child_node)

            return node

        # Add root groups as children of root node
        for root_hash in root_hashes:
            child = build_node(root_hash, self.cli_entry_point or "")
            root.children.append(child)

        return root


# =============================================================================
# ROUTE HIERARCHY EXTRACTOR
# =============================================================================


@dataclass
class RouteElementInfo:
    """Element info enriched with HTTP route data."""

    hash_id: str
    element_id: str
    name: str
    file_path: str
    line: int
    http_routes: list[dict[str, Any]]
    summary: str | None = None


class RouteHierarchyExtractor:
    """Extract FastAPI/Flask route hierarchies.

    Builds BELONGS_TO_ROUTER relationships and tracks full URL paths.
    Groups routes by their source file (router module).
    """

    # Common API prefixes to detect
    COMMON_PREFIXES = ["/api/v1", "/api/v2", "/api"]

    def __init__(
        self,
        scope: str,
        repository: str,
        username: str,
        api_prefix: str = "/api/v1",
    ):
        """Initialize extractor.

        Args:
            scope: Repository scope
            repository: Repository name
            username: Username for the branch
            api_prefix: API prefix (e.g., "/api/v1")
        """
        self.scope = scope
        self.repository = repository
        self.username = username
        self.api_prefix = api_prefix
        self.priority = get_priority_for_username(username)

    def extract(self, elements: list[RouteElementInfo]) -> tuple[
        list[CodeRelationship],
        list[ExternalReference],
    ]:
        """Extract route hierarchy relationships from elements.

        Groups routes by their source file (which typically corresponds to a router).
        Creates external references for each router module.

        Args:
            elements: List of elements with HTTP route data

        Returns:
            Tuple of (relationships, external_references)
        """
        relationships: list[CodeRelationship] = []
        external_refs: list[ExternalReference] = []

        # Group elements by file (router module)
        routes_by_file: dict[str, list[RouteElementInfo]] = {}
        for element in elements:
            if element.http_routes:
                if element.file_path not in routes_by_file:
                    routes_by_file[element.file_path] = []
                routes_by_file[element.file_path].append(element)

        # Create external refs for each router and relationships for routes
        for file_path, routes in routes_by_file.items():
            router_name = self._extract_router_name(file_path)
            router_ref_id = f"router:{self.scope}:{self.repository}:{router_name}"

            # Detect common prefix for routes in this file
            common_prefix = self._detect_common_prefix(routes)

            # Create external reference for the router
            router_ref = ExternalReference(
                ref_id=router_ref_id,
                ref_type=ExternalRefType.HTTP_ROUTER,
                name=router_name,
                description=f"HTTP router from {file_path}",
                scope=self.scope,
                repository=self.repository,
                username=self.username,
                metadata={
                    "file_path": file_path,
                    "route_count": len(routes),
                    "common_prefix": common_prefix,
                    "api_prefix": self.api_prefix,
                },
            )
            external_refs.append(router_ref)

            # Create relationships for each route to its router
            for element in routes:
                for route in element.http_routes:
                    route_path = route.get("path", "")
                    method = route.get("method", "GET")
                    full_url = self._build_full_url(route_path)

                    relationship = CodeRelationship(
                        relationship_id=generate_relationship_id(
                            element.hash_id,
                            router_ref_id,  # Target is the router external ref
                            RelationshipType.BELONGS_TO_ROUTER,
                            self.scope,
                            self.repository,
                            self.username,
                        ),
                        relationship_key=generate_relationship_key(
                            element.hash_id,
                            router_ref_id,
                            RelationshipType.BELONGS_TO_ROUTER,
                            self.scope,
                            self.repository,
                        ),
                        source_hash=element.hash_id,
                        target_hash=router_ref_id,
                        relationship_type=RelationshipType.BELONGS_TO_ROUTER,
                        scope=self.scope,
                        repository=self.repository,
                        username=self.username,
                        priority=self.priority,
                        confidence=1.0,
                        line=element.line,
                        details={
                            "method": method,
                            "path": route_path,
                            "full_url": full_url,
                            "router_name": router_name,
                            "framework": route.get("framework"),
                            "path_params": route.get("path_params", []),
                        },
                    )
                    relationships.append(relationship)

        return relationships, external_refs

    def _extract_router_name(self, file_path: str) -> str:
        """Extract router name from file path.

        Examples:
            "src/magaldi_web/routes/repos.py" -> "repos"
            "routes/admin.py" -> "admin"
            "api/v1/users.py" -> "users"
        """
        # Get filename without extension
        from pathlib import Path
        name = Path(file_path).stem

        # Skip common non-descriptive names
        if name in ("__init__", "main", "app", "routes", "views"):
            # Use parent directory name
            parent = Path(file_path).parent.name
            if parent and parent not in (".", "src"):
                return parent
        return name

    def _detect_common_prefix(self, routes: list[RouteElementInfo]) -> str:
        """Detect common URL prefix for routes in a file.

        Returns the longest common prefix path.
        """
        if not routes:
            return ""

        # Collect all route paths
        all_paths = []
        for element in routes:
            for route in element.http_routes:
                path = route.get("path", "")
                if path:
                    all_paths.append(path)

        if not all_paths:
            return ""

        # Find common prefix
        if len(all_paths) == 1:
            # Single route - use its first path segment
            parts = all_paths[0].strip("/").split("/")
            return "/" + parts[0] if parts and parts[0] else ""

        # Find longest common prefix
        prefix_parts = all_paths[0].strip("/").split("/")
        for path in all_paths[1:]:
            path_parts = path.strip("/").split("/")
            # Compare parts
            common_len = 0
            for i, (a, b) in enumerate(zip(prefix_parts, path_parts)):
                # Stop at path parameters
                if "{" in a or "{" in b:
                    break
                if a == b:
                    common_len = i + 1
                else:
                    break
            prefix_parts = prefix_parts[:common_len]

        if prefix_parts:
            return "/" + "/".join(prefix_parts)
        return ""

    def _build_full_url(self, route_path: str) -> str:
        """Build full URL including API prefix.

        Examples:
            "/repos" -> "/api/v1/repos"
            "/admin/health" -> "/api/v1/admin/health"
        """
        # Ensure route_path starts with /
        if route_path and not route_path.startswith("/"):
            route_path = "/" + route_path

        # Combine with API prefix
        if self.api_prefix:
            prefix = self.api_prefix.rstrip("/")
            return prefix + route_path
        return route_path


# =============================================================================
# HELPER: Load route elements from ES
# =============================================================================


def elements_to_route_info(es_docs: list[dict[str, Any]]) -> list[RouteElementInfo]:
    """Convert Search documents to RouteElementInfo objects."""
    result = []
    for doc in es_docs:
        # Handle both _source wrapper and direct doc
        source = doc.get("_source", doc)

        # Skip if no http_routes
        http_routes = source.get("http_routes") or []
        if not http_routes:
            continue

        result.append(RouteElementInfo(
            hash_id=source.get("hash_id", ""),
            element_id=source.get("element_id", ""),
            name=source.get("name", ""),
            file_path=source.get("relative_path", ""),
            line=source.get("line_start", 0),
            http_routes=http_routes,
            summary=source.get("summary"),
        ))

    return result


# =============================================================================
# HELPER: Load elements from ES for extraction
# =============================================================================


def elements_to_element_info(es_docs: list[dict[str, Any]]) -> list[ElementInfo]:
    """Convert Search documents to ElementInfo objects."""
    result = []
    for doc in es_docs:
        # Handle both _source wrapper and direct doc
        source = doc.get("_source", doc)

        # Skip if no decorators
        decorators = source.get("decorators") or []
        decorator_details = source.get("decorator_details") or []

        if not decorators and not decorator_details:
            continue

        result.append(ElementInfo(
            hash_id=source.get("hash_id", ""),
            element_id=source.get("element_id", ""),
            name=source.get("name", ""),
            file_path=source.get("relative_path", ""),
            line=source.get("line_start", 0),
            decorators=decorators,
            decorator_details=decorator_details,
            summary=source.get("summary"),
        ))

    return result
