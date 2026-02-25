"""Formatters for analysis results (call graphs, dead code, entry points).

All element references use the shared compact format from _compact module
for consistent output across all tools.
"""

from __future__ import annotations

from typing import Any

from magaldi_mcp.formatters._compact import (
    SUMMARY_SHORT,
    compact_element,
    compact_element_header,
    compact_ref,
    compact_unresolved,
    file_group,
    indented_summary,
)
from magaldi_mcp.formatters.base import ResultFormatter


class CallGraphFormatter(ResultFormatter):
    """Compact formatter for get_call_graph results.

    Uses file-grouped layout for callers/callees and compact element refs.
    """

    def can_format(self, result: Any) -> bool:
        """Check if result is a call graph."""
        if not isinstance(result, dict):
            return False
        return "callers" in result and "callees" in result and "element" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format call graph with compact element references."""
        elem = result.get("element", {})
        callers = result.get("callers", [])
        callees = result.get("callees", [])
        semantic = result.get("semantic_related", [])

        resolved_callees = [c for c in callees if c.get("element_id")]
        unresolved_callees = [c for c in callees if not c.get("element_id")]

        lines = [
            f"# call_graph: {compact_element_header(elem)}",
            f"# {len(callers)} callers, {len(callees)} callees"
            + (f", {len(semantic)} similar" if semantic else ""),
        ]

        if callers:
            lines.append(f"\nCallers ({len(callers)}):")
            for fp, _, elements in file_group(callers):
                lines.append(f"  {fp}")
                for c in elements:
                    lines.append(compact_ref(c, arrow="←", indent=4))

        if resolved_callees:
            lines.append(f"\nCallees ({len(resolved_callees)}):")
            for c in resolved_callees:
                # Use target_line for resolved callees
                ref = dict(c)
                if "target_line" in ref:
                    ref["line"] = ref["target_line"]
                lines.append(compact_ref(ref, arrow="→", indent=2))

        if unresolved_callees:
            lines.append(f"\nUnresolved ({len(unresolved_callees)}):")
            for c in unresolved_callees:
                lines.append(compact_unresolved(c))

        if semantic:
            lines.append(f"\nSimilar ({len(semantic)}):")
            for s in semantic:
                lines.append(compact_ref(s, arrow="~", indent=2, score=True))

        return "\n".join(lines)


class FindCallersFormatter(ResultFormatter):
    """Compact formatter for find_callers results.

    Merges code/test callers into file-grouped layout, marks tests with [T].
    """

    def can_format(self, result: Any) -> bool:
        """Check if result is find_callers result."""
        if not isinstance(result, dict):
            return False
        return "target" in result and "code_results" in result and "test_results" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format find_callers with compact element references."""
        target = result.get("target", {})
        code_results = result.get("code_results", [])
        test_results = result.get("test_results", [])

        total = len(code_results) + len(test_results)
        lines = [
            f"# callers_of: {compact_element_header(target)}",
            f"# {len(code_results)} code, {len(test_results)} test",
        ]

        if code_results:
            lines.append(f"\nCode ({len(code_results)}):")
            for fp, _, elements in file_group(code_results):
                lines.append(f"  {fp}")
                for c in elements:
                    lines.append(compact_ref(c, arrow="←", indent=4))

        if test_results:
            lines.append(f"\nTests ({len(test_results)}):")
            for fp, _, elements in file_group(test_results):
                lines.append(f"  {fp}")
                for c in elements:
                    lines.append(compact_ref(c, arrow="←", indent=4))

        if not total:
            lines.append("\nNo callers found.")

        return "\n".join(lines)


class CallChainFormatter(ResultFormatter):
    """Compact formatter for find_call_chain results.

    Uses ASCII tree with compact element refs and arrow indicators.
    """

    def can_format(self, result: Any) -> bool:
        """Check if result is a call chain."""
        if not isinstance(result, dict):
            return False
        return "root" in result and "direction" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format call chain with compact tree nodes."""
        root = result.get("root", {})
        direction = result.get("direction")
        max_depth = result.get("max_depth", 5)
        lines = [
            f"# call_chain: {compact_element_header(root)}",
            f"# direction:{direction} depth:{max_depth}",
        ]

        def format_tree(nodes: list, indent: int = 1, arrow: str = "→") -> None:
            for node in nodes:
                prefix = "  " * indent
                if node.get("cycle"):
                    lines.append(f"{prefix}⟳ {node.get('name')}")
                elif node.get("unresolved"):
                    receiver = f"{node.get('receiver')}." if node.get('receiver') else ""
                    lines.append(f"{prefix}→ ?:{receiver}{node.get('name')}()")
                elif node.get("missing"):
                    lines.append(f"{prefix}→ ?:{node.get('name')}")
                else:
                    lines.append(compact_ref(node, arrow=arrow, indent=indent * 2))
                    if node.get("callers"):
                        format_tree(node["callers"], indent + 1, arrow="←")
                    if node.get("callees"):
                        format_tree(node["callees"], indent + 1, arrow="→")

        if result.get("callers"):
            lines.append("\nCallers:")
            format_tree(result["callers"], 1, arrow="←")

        if result.get("callees"):
            lines.append("\nCallees:")
            format_tree(result["callees"], 1, arrow="→")

        return "\n".join(lines)


class DeadCodeFormatter(ResultFormatter):
    """Compact formatter for find_dead_code results.

    File-grouped dead elements with compact element format.
    """

    def can_format(self, result: Any) -> bool:
        """Check if result is dead code analysis."""
        if not isinstance(result, dict):
            return False
        return "potentially_dead" in result and "stats" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format dead code with compact element references."""
        dead = result.get("potentially_dead", [])
        stats = result.get("stats", {})

        lines = [
            f"# dead_code: {stats.get('potentially_dead', 0)} dead"
            f" / {stats.get('total_functions', 0)} total"
            f" ({stats.get('called', 0)} called,"
            f" {stats.get('excluded_entry_points', 0)} entry points)",
        ]

        if dead:
            lines.append("")
            for fp, _, elements in file_group(dead):
                lines.append(fp)
                for d in elements:
                    lines.append(compact_element(d))
                    summary = indented_summary(d, max_len=SUMMARY_SHORT)
                    if summary:
                        lines.append(summary)
        else:
            lines.append("\nNo potentially dead code found!")

        return "\n".join(lines)


class EntryPointsFormatter(ResultFormatter):
    """Compact formatter for find_entry_points results.

    Compact element format grouped by entry point category.
    """

    def can_format(self, result: Any) -> bool:
        """Check if result is entry points."""
        if not isinstance(result, dict):
            return False
        return "http" in result and "cli" in result and "test" in result and "main" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format entry points with compact element references."""
        stats = result.get("stats", {})
        lines = [f"# entry_points: {stats.get('total', 0)} total"]

        categories = [
            ("HTTP", result.get("http", [])),
            ("CLI", result.get("cli", [])),
            ("Test", result.get("test", [])),
            ("Main", result.get("main", [])),
            ("Async", result.get("async_tasks", [])),
        ]

        for title, entries in categories:
            if entries:
                lines.append(f"\n{title} ({len(entries)}):")
                for e in entries:
                    lines.append(compact_element(e))
                    if e.get('decorators'):
                        decs = ", ".join(e['decorators'][:3])
                        if len(e['decorators']) > 3:
                            decs += "..."
                        lines.append(f"    @{decs}")

        return "\n".join(lines)


class ExplainElementFormatter(ResultFormatter):
    """Compact formatter for explain_element results.

    Uses compact element format throughout all sections. Significantly
    reduces output tokens compared to the verbose original.
    """

    def can_format(self, result: Any) -> bool:
        """Check if result is explain_element."""
        if not isinstance(result, dict):
            return False
        return "element" in result and "similar_code" in result and "parent" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format explain_element with compact references everywhere."""
        elem = result.get("element", {})
        lines = [f"# explain: {compact_element_header(elem)}"]

        # Core info
        if elem.get('signature'):
            lines.append(f"  sig: {elem['signature']}")
        if elem.get('summary'):
            lines.append(f"  {elem['summary']}")
        if elem.get('docstring'):
            docstring = elem['docstring']
            if len(docstring) > SUMMARY_SHORT:
                docstring = docstring[:SUMMARY_SHORT] + "..."
            lines.append(f"  doc: {docstring}")
        if elem.get('decorators'):
            lines.append(f"  @{', '.join(elem['decorators'])}")

        # Parent context
        parent = result.get("parent")
        if parent:
            lines.append(f"\nParent: {compact_element_header(parent)}")
            summary = indented_summary(parent, indent=2, max_len=SUMMARY_SHORT)
            if summary:
                lines.append(summary)

        # Callers
        callers = result.get("callers", [])
        if callers:
            lines.append(f"\nCallers ({len(callers)}):")
            for c in callers:
                lines.append(compact_ref(c, arrow="←"))

        # Callees
        callees = result.get("callees", [])
        if callees:
            resolved = [c for c in callees if c.get("element_id")]
            unresolved = [c for c in callees if not c.get("element_id")]
            lines.append(f"\nCallees ({len(callees)}):")
            for c in resolved:
                ref = dict(c)
                if "target_line" in ref:
                    ref["line"] = ref["target_line"]
                lines.append(compact_ref(ref, arrow="→"))
            for c in unresolved:
                lines.append(compact_unresolved(c))

        # Imports (compact count + list)
        imports = result.get("imports", [])
        if imports:
            lines.append(f"\nImports ({len(imports)}):")
            for imp in imports[:10]:
                module = imp.get("module", "")
                name = imp.get("name", "")
                alias = imp.get("alias", "")
                if alias:
                    lines.append(f"  {module}.{name} as {alias}")
                elif module:
                    lines.append(f"  {module}.{name}")
                else:
                    lines.append(f"  {name}")
            if len(imports) > 10:
                lines.append(f"  ... +{len(imports) - 10} more")

        # Similar code
        similar = result.get("similar_code", [])
        if similar:
            lines.append(f"\nSimilar ({len(similar)}):")
            for s in similar:
                # Use similarity field for score display
                ref = dict(s)
                if "similarity" in ref:
                    ref["score"] = ref["similarity"]
                lines.append(compact_ref(ref, arrow="~", score=True))

        return "\n".join(lines)
