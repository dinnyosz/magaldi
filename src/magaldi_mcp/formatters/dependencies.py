"""Formatters for dependency analysis results."""

from __future__ import annotations

from typing import Any

from magaldi_mcp.formatters.base import ResultFormatter


class DependencyFormatter(ResultFormatter):
    """Formatter for find_dependencies results."""

    def can_format(self, result: Any) -> bool:
        """Check if result is dependencies."""
        if not isinstance(result, dict):
            return False
        return "internal_imports" in result and "external_imports" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format dependencies."""
        file_info = result.get("file_info", {})
        stats = result.get("stats", {})
        lines = [f"Dependencies for {file_info.get('path', '?')}\n"]
        lines.append(f"Stats: {stats.get('total', 0)} total ({stats.get('internal', 0)} internal, {stats.get('external', 0)} external)")
        lines.append("")

        internal = result.get("internal_imports", [])
        if internal:
            lines.append(f"Internal Imports ({len(internal)}):")
            for imp in internal:
                module = imp.get("module", "")
                name = imp.get("name", "")
                alias = imp.get("alias", "")
                line_num = imp.get("line", "?")
                if alias:
                    lines.append(f"  from {module} import {name} as {alias} (line {line_num})")
                elif module:
                    lines.append(f"  from {module} import {name} (line {line_num})")
                else:
                    lines.append(f"  import {name} (line {line_num})")
            lines.append("")

        external = result.get("external_imports", [])
        if external:
            lines.append(f"External Imports ({len(external)}):")
            for imp in external:
                module = imp.get("module", "")
                name = imp.get("name", "")
                alias = imp.get("alias", "")
                line_num = imp.get("line", "?")
                if alias:
                    lines.append(f"  from {module} import {name} as {alias} (line {line_num})")
                elif module:
                    lines.append(f"  from {module} import {name} (line {line_num})")
                else:
                    lines.append(f"  import {name} (line {line_num})")

        return "\n".join(lines)


class DependentFormatter(ResultFormatter):
    """Formatter for find_dependents results."""

    def can_format(self, result: Any) -> bool:
        """Check if result is dependents."""
        if not isinstance(result, dict):
            return False
        return "module" in result and "dependents" in result and "total" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format dependents."""
        module = result.get("module", "?")
        dependents = result.get("dependents", [])
        total = result.get("total", 0)
        lines = [f"Files importing '{module}' ({total}):\n"]

        for dep in dependents:
            lines.append(f"  {dep.get('file', '?')}")

        if not dependents:
            lines.append("  (no dependents found)")

        return "\n".join(lines)


class DependencyGraphFormatter(ResultFormatter):
    """Formatter for dependency_graph results."""

    def can_format(self, result: Any) -> bool:
        """Check if result is a dependency graph."""
        if not isinstance(result, dict):
            return False
        return "nodes" in result and "edges" in result and "cycles" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format dependency graph."""
        stats = result.get("stats", {})
        lines = ["Dependency Graph"]
        lines.append(f"  Nodes: {stats.get('node_count', 0)}")
        lines.append(f"  Edges: {stats.get('edge_count', 0)}")
        lines.append(f"  Cycles: {stats.get('cycle_count', 0)}")
        lines.append("")

        cycles = result.get("cycles", [])
        if cycles:
            lines.append(f"Circular Dependencies ({len(cycles)}):")
            for i, cycle in enumerate(cycles[:10], 1):  # Show up to 10 cycles
                lines.append(f"  {i}. {' -> '.join(cycle)}")
            if len(cycles) > 10:
                lines.append(f"  ... and {len(cycles) - 10} more")
            lines.append("")

        edges = result.get("edges", [])
        if edges:
            lines.append(f"Dependencies ({len(edges)}):")
            for edge in edges[:50]:  # Show up to 50 edges
                lines.append(f"  {edge.get('from', '?')} -> {edge.get('to', '?')}")
            if len(edges) > 50:
                lines.append(f"  ... and {len(edges) - 50} more")

        return "\n".join(lines)
