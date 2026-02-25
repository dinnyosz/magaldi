"""Formatters for dependency analysis results.

All formatters use compact output format for token efficiency.
"""

from __future__ import annotations

from typing import Any

from magaldi_mcp.formatters.base import ResultFormatter


def _compact_import(imp: dict[str, Any]) -> str:
    """Format a single import as a compact line.

    Returns:
        Line like '  module.name:L42' or '  module.name as alias:L42'
    """
    module = imp.get("module", "")
    name = imp.get("name", "")
    alias = imp.get("alias", "")
    line_num = imp.get("line", "?")

    if module and name:
        path = f"{module}.{name}"
    elif module:
        path = module
    else:
        path = name

    if alias:
        return f"  {path} as {alias}:L{line_num}"
    return f"  {path}:L{line_num}"


class DependencyFormatter(ResultFormatter):
    """Compact formatter for find_dependencies results."""

    def can_format(self, result: Any) -> bool:
        """Check if result is dependencies."""
        if not isinstance(result, dict):
            return False
        return "internal_imports" in result and "external_imports" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format dependencies with compact import lines."""
        file_info = result.get("file_info", {})
        stats = result.get("stats", {})
        lines = [
            f"# deps: {file_info.get('path', '?')}",
            f"# {stats.get('total', 0)} total ({stats.get('internal', 0)} int, {stats.get('external', 0)} ext)",
        ]

        internal = result.get("internal_imports", [])
        if internal:
            lines.append(f"\nInternal ({len(internal)}):")
            for imp in internal:
                lines.append(_compact_import(imp))

        external = result.get("external_imports", [])
        if external:
            lines.append(f"\nExternal ({len(external)}):")
            for imp in external:
                lines.append(_compact_import(imp))

        return "\n".join(lines)


class DependentFormatter(ResultFormatter):
    """Compact formatter for find_dependents results."""

    def can_format(self, result: Any) -> bool:
        """Check if result is dependents."""
        if not isinstance(result, dict):
            return False
        return "module" in result and "dependents" in result and "total" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format dependents as compact file list."""
        module = result.get("module", "?")
        dependents = result.get("dependents", [])
        total = result.get("total", 0)
        lines = [f"# dependents_of: {module} ({total})"]

        if dependents:
            for dep in dependents:
                lines.append(f"  {dep.get('file', '?')}")
        else:
            lines.append("  (none)")

        return "\n".join(lines)


class DependencyGraphFormatter(ResultFormatter):
    """Compact formatter for dependency_graph results."""

    def can_format(self, result: Any) -> bool:
        """Check if result is a dependency graph."""
        if not isinstance(result, dict):
            return False
        return "nodes" in result and "edges" in result and "cycles" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format dependency graph with compact stats and edges."""
        stats = result.get("stats", {})
        cycles = result.get("cycles", [])
        edges = result.get("edges", [])

        lines = [
            f"# dep_graph: {stats.get('node_count', 0)} nodes,"
            f" {stats.get('edge_count', 0)} edges,"
            f" {stats.get('cycle_count', 0)} cycles",
        ]

        if cycles:
            lines.append(f"\nCycles ({len(cycles)}):")
            for cycle in cycles[:10]:
                lines.append(f"  {' → '.join(cycle)}")
            if len(cycles) > 10:
                lines.append(f"  ...+{len(cycles) - 10} more")

        if edges:
            lines.append(f"\nEdges ({len(edges)}):")
            for edge in edges[:50]:
                lines.append(f"  {edge.get('from', '?')} → {edge.get('to', '?')}")
            if len(edges) > 50:
                lines.append(f"  ...+{len(edges) - 50} more")

        return "\n".join(lines)
