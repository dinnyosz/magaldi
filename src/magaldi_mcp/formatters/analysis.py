"""Formatters for analysis results (call graphs, dead code, entry points)."""

from __future__ import annotations

from typing import Any

from magaldi_mcp.formatters.base import ResultFormatter, terse_summary


class CallGraphFormatter(ResultFormatter):
    """Formatter for get_call_graph results."""

    def can_format(self, result: Any) -> bool:
        """Check if result is a call graph."""
        if not isinstance(result, dict):
            return False
        return "callers" in result and "callees" in result and "element" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format call graph."""
        elem = result.get("element", {})
        lines = [f"Call graph for [{elem.get('type')}] {elem.get('name')} ({elem.get('file')}:{elem.get('line')})\n"]

        if result.get("callers"):
            lines.append(f"Callers ({len(result['callers'])}):")
            for c in result["callers"]:
                loc = f"{c.get('file')}:{c.get('line')}" if c.get('file') else "?"
                lines.append(f"  [{c.get('type', '?')}] {c.get('name')} ({loc})")
                if c.get('summary'):
                    summary = terse_summary(c['summary'], max_length=100)
                    lines.append(f"    {summary}")
            lines.append("")

        if result.get("callees"):
            lines.append(f"Callees ({len(result['callees'])}):")
            for c in result["callees"]:
                if c.get("element_id"):
                    loc = f"{c.get('file')}:{c.get('target_line')}" if c.get('file') else "?"
                    lines.append(f"  [{c.get('type', '?')}] {c.get('name')} ({loc})")
                    if c.get('summary'):
                        summary = terse_summary(c['summary'], max_length=100)
                        lines.append(f"    {summary}")
                else:
                    receiver = f"{c.get('receiver')}." if c.get('receiver') else ""
                    lines.append(f"  {receiver}{c.get('name')}() @ line {c.get('line')} (unresolved)")
            lines.append("")

        return "\n".join(lines)


class FindCallersFormatter(ResultFormatter):
    """Formatter for find_callers results."""

    def can_format(self, result: Any) -> bool:
        """Check if result is find_callers result."""
        if not isinstance(result, dict):
            return False
        return "target" in result and "code_results" in result and "test_results" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format find_callers results."""
        target = result.get("target", {})
        lines = [f"Callers of [{target.get('type')}] {target.get('name')} ({target.get('file')}:{target.get('line')})\n"]

        code_results = result.get("code_results", [])
        test_results = result.get("test_results", [])

        if code_results:
            lines.append(f"Code Callers ({len(code_results)}):")
            for c in code_results:
                loc = f"{c.get('file')}:{c.get('line')}" if c.get('file') else "?"
                lines.append(f"  [{c.get('type', '?')}] {c.get('name')} ({loc})")
                if c.get('summary'):
                    summary = terse_summary(c['summary'], max_length=100)
                    lines.append(f"    {summary}")
            lines.append("")

        if test_results:
            lines.append(f"Test Callers ({len(test_results)}):")
            for c in test_results:
                loc = f"{c.get('file')}:{c.get('line')}" if c.get('file') else "?"
                lines.append(f"  [{c.get('type', '?')}] {c.get('name')} ({loc})")

        return "\n".join(lines)


class CallChainFormatter(ResultFormatter):
    """Formatter for find_call_chain results."""

    def can_format(self, result: Any) -> bool:
        """Check if result is a call chain."""
        if not isinstance(result, dict):
            return False
        return "root" in result and "direction" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format call chain."""
        root = result.get("root", {})
        direction = result.get("direction")
        max_depth = result.get("max_depth", 5)
        lines = [f"Call chain for [{root.get('type')}] {root.get('name')} ({root.get('file')}:{root.get('line')})"]
        lines.append(f"Direction: {direction}, Max depth: {max_depth}\n")

        def format_tree(nodes: list, indent: int = 0) -> None:
            for node in nodes:
                prefix = "  " * indent
                if node.get("cycle"):
                    lines.append(f"{prefix}[CYCLE] {node.get('name')}")
                elif node.get("unresolved"):
                    receiver = f"{node.get('receiver')}." if node.get('receiver') else ""
                    lines.append(f"{prefix}[unresolved] {receiver}{node.get('name')}()")
                elif node.get("missing"):
                    lines.append(f"{prefix}[missing] {node.get('name')}")
                else:
                    loc = f"{node.get('file')}:{node.get('line')}" if node.get('file') else "?"
                    lines.append(f"{prefix}[{node.get('type', '?')}] {node.get('name')} ({loc})")
                    # Recurse into children
                    if node.get("callers"):
                        format_tree(node["callers"], indent + 1)
                    if node.get("callees"):
                        format_tree(node["callees"], indent + 1)

        if result.get("callers"):
            lines.append("Callers:")
            format_tree(result["callers"], 1)
            lines.append("")

        if result.get("callees"):
            lines.append("Callees:")
            format_tree(result["callees"], 1)

        return "\n".join(lines)


class DeadCodeFormatter(ResultFormatter):
    """Formatter for find_dead_code results."""

    def can_format(self, result: Any) -> bool:
        """Check if result is dead code analysis."""
        if not isinstance(result, dict):
            return False
        return "potentially_dead" in result and "stats" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format dead code analysis."""
        dead = result.get("potentially_dead", [])
        stats = result.get("stats", {})
        lines = ["Dead Code Analysis\n"]
        lines.append("Stats:")
        lines.append(f"  Total functions: {stats.get('total_functions', 0)}")
        lines.append(f"  Excluded (entry points): {stats.get('excluded_entry_points', 0)}")
        lines.append(f"  Called: {stats.get('called', 0)}")
        lines.append(f"  Potentially dead: {stats.get('potentially_dead', 0)}")
        lines.append("")

        if dead:
            lines.append(f"Potentially Dead Code ({len(dead)}):")
            for d in dead:
                loc = f"{d.get('file')}:{d.get('line')}" if d.get('file') else "?"
                lines.append(f"  [{d.get('type', '?')}] {d.get('name')} ({loc})")
                if d.get('summary'):
                    summary = terse_summary(d['summary'], max_length=100)
                    lines.append(f"    {summary}")
        else:
            lines.append("No potentially dead code found!")

        return "\n".join(lines)


class EntryPointsFormatter(ResultFormatter):
    """Formatter for find_entry_points results."""

    def can_format(self, result: Any) -> bool:
        """Check if result is entry points."""
        if not isinstance(result, dict):
            return False
        return "http" in result and "cli" in result and "test" in result and "main" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format entry points."""
        stats = result.get("stats", {})
        lines = [f"Entry Points (Total: {stats.get('total', 0)})\n"]

        def format_entries(title: str, entries: list) -> None:
            if entries:
                lines.append(f"{title} ({len(entries)}):")
                for e in entries:
                    loc = f"{e.get('file')}:{e.get('line')}" if e.get('file') else "?"
                    lines.append(f"  [{e.get('type', '?')}] {e.get('name')} ({loc})")
                    if e.get('decorators'):
                        decs = ", ".join(e['decorators'][:3])
                        if len(e['decorators']) > 3:
                            decs += "..."
                        lines.append(f"    decorators: {decs}")
                lines.append("")

        format_entries("HTTP Handlers", result.get("http", []))
        format_entries("CLI Commands", result.get("cli", []))
        format_entries("Test Fixtures", result.get("test", []))
        format_entries("Main Functions", result.get("main", []))
        format_entries("Async Tasks", result.get("async_tasks", []))

        return "\n".join(lines)


class ExplainElementFormatter(ResultFormatter):
    """Formatter for explain_element results."""

    def can_format(self, result: Any) -> bool:
        """Check if result is explain_element."""
        if not isinstance(result, dict):
            return False
        return "element" in result and "similar_code" in result and "parent" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format explain_element results."""
        elem = result.get("element", {})
        lines = [f"Element Overview: [{elem.get('type')}] {elem.get('name')}"]
        lines.append(f"  File: {elem.get('file')}:{elem.get('line')}")
        if elem.get('signature'):
            lines.append(f"  Signature: {elem['signature']}")
        if elem.get('summary'):
            lines.append(f"  Summary: {elem['summary']}")
        if elem.get('docstring'):
            docstring = elem['docstring']
            if len(docstring) > 200:
                docstring = docstring[:200] + "..."
            lines.append(f"  Docstring: {docstring}")
        if elem.get('decorators'):
            lines.append(f"  Decorators: {', '.join(elem['decorators'])}")
        lines.append("")

        # Parent context
        parent = result.get("parent")
        if parent:
            lines.append(f"Parent: [{parent.get('type')}] {parent.get('name')} ({parent.get('file')}:{parent.get('line')})")
            if parent.get('summary'):
                lines.append(f"  {terse_summary(parent['summary'], max_length=100)}")
            lines.append("")

        # Callers
        callers = result.get("callers", [])
        if callers:
            lines.append(f"Callers ({len(callers)}):")
            for c in callers:
                loc = f"{c.get('file')}:{c.get('line')}" if c.get('file') else "?"
                lines.append(f"  [{c.get('type', '?')}] {c.get('name')} ({loc})")
                if c.get('summary'):
                    summary = terse_summary(c['summary'], max_length=80)
                    lines.append(f"    {summary}")
            lines.append("")

        # Callees
        callees = result.get("callees", [])
        if callees:
            lines.append(f"Callees ({len(callees)}):")
            for c in callees:
                if c.get("element_id"):
                    loc = f"{c.get('file')}:{c.get('target_line')}" if c.get('file') else "?"
                    lines.append(f"  [{c.get('type', '?')}] {c.get('name')} ({loc})")
                    if c.get('summary'):
                        summary = terse_summary(c['summary'], max_length=80)
                        lines.append(f"    {summary}")
                else:
                    receiver = f"{c.get('receiver')}." if c.get('receiver') else ""
                    lines.append(f"  {receiver}{c.get('name')}() @ line {c.get('line')} (unresolved)")
            lines.append("")

        # Imports (for file elements)
        imports = result.get("imports", [])
        if imports:
            lines.append(f"Imports ({len(imports)}):")
            for imp in imports[:10]:
                module = imp.get("module", "")
                name = imp.get("name", "")
                alias = imp.get("alias", "")
                if alias:
                    lines.append(f"  from {module} import {name} as {alias}")
                elif module:
                    lines.append(f"  from {module} import {name}")
                else:
                    lines.append(f"  import {name}")
            if len(imports) > 10:
                lines.append(f"  ... and {len(imports) - 10} more")
            lines.append("")

        # Similar code
        similar = result.get("similar_code", [])
        if similar:
            lines.append(f"Similar Code ({len(similar)}):")
            for s in similar:
                loc = f"{s.get('file')}:{s.get('line')}" if s.get('file') else "?"
                sim_score = s.get('similarity', 0)
                lines.append(f"  [{s.get('type', '?')}] {s.get('name')} ({loc}) - {sim_score:.1%} similar")
                if s.get('summary'):
                    summary = terse_summary(s['summary'], max_length=80)
                    lines.append(f"    {summary}")

        return "\n".join(lines)
