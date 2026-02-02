"""Formatters for element details and lists."""

from __future__ import annotations

from typing import Any

from magaldi_mcp.formatters.base import ResultFormatter


class ElementDetailsFormatter(ResultFormatter):
    """Formatter for single element details."""

    def can_format(self, result: Any) -> bool:
        """Check if result is a single element with details."""
        if not isinstance(result, dict):
            return False
        # Element details have element_id, type, and file but NOT callers/callees (explain_element)
        return (
            "element_id" in result
            and "type" in result
            and "callers" not in result
            and "code_results" not in result
        )

    def format(self, result: dict[str, Any]) -> str:
        """Format element details."""
        lines = [f"[{result.get('type')}] {result.get('name')}"]
        lines.append(f"  File: {result.get('file')}:{result.get('line_start')}-{result.get('line_end')}")
        if result.get('signature'):
            lines.append(f"  Signature: {result['signature']}")
        if result.get('summary'):
            lines.append(f"  Summary: {result['summary']}")
        if result.get('code'):
            lines.append(f"  Code:\n{result['code']}")
        return "\n".join(lines)


class RepoListFormatter(ResultFormatter):
    """Formatter for repository list."""

    def can_format(self, result: Any) -> bool:
        """Check if result is a list of repositories."""
        if not isinstance(result, list) or not result:
            return False
        first = result[0]
        return "element_count" in first and "scope" in first

    def format(self, result: list[dict[str, Any]]) -> str:
        """Format repository list."""
        lines = ["Indexed repositories:\n"]
        for r in result:
            lines.append(
                f"  {r.get('scope')}/{r.get('repository')}: "
                f"{r.get('element_count')} elements, {r.get('file_count')} files"
            )
        return "\n".join(lines)


class FileListFormatter(ResultFormatter):
    """Formatter for file list (find_files)."""

    def can_format(self, result: Any) -> bool:
        """Check if result is a list of files."""
        if not isinstance(result, list) or not result:
            return False
        first = result[0]
        return "path" in first and "element_id" not in first

    def format(self, result: list[dict[str, Any]]) -> str:
        """Format file list."""
        lines = [f"Found {len(result)} files:\n"]
        for r in result:
            size = r.get('size') or r.get('lines', 0)
            unit = 'bytes' if r.get('size') else 'lines'
            lines.append(f"  {r.get('path')} ({size} {unit})")
        return "\n".join(lines)


class GrepResultsFormatter(ResultFormatter):
    """Formatter for grep results."""

    def can_format(self, result: Any) -> bool:
        """Check if result is grep results."""
        if not isinstance(result, list) or not result:
            return False
        first = result[0]
        return "file" in first and "content" in first and "line" in first and "element_id" not in first

    def format(self, result: list[dict[str, Any]]) -> str:
        """Format grep results."""
        lines = [f"Found {len(result)} matches:\n"]
        for r in result:
            lines.append(f"{r.get('file')}:{r.get('line')}")
            for ctx in r.get("context_before", []):
                lines.append(f"  | {ctx}")
            lines.append(f"  > {r.get('content')}")
            for ctx in r.get("context_after", []):
                lines.append(f"  | {ctx}")
            lines.append("")
        return "\n".join(lines)


class UsageResultsFormatter(ResultFormatter):
    """Formatter for find_usages results."""

    def can_format(self, result: Any) -> bool:
        """Check if result is usage results."""
        if not isinstance(result, list) or not result:
            return False
        first = result[0]
        return "file" in first and "content" in first and "context_before" in first and "element_id" not in first

    def format(self, result: list[dict[str, Any]]) -> str:
        """Format usage results."""
        lines = [f"Found {len(result)} usages:\n"]
        for r in result:
            lines.append(f"{r.get('file')}:{r.get('line')}")
            for ctx in r.get("context_before", []):
                lines.append(f"  | {ctx}")
            lines.append(f"  > {r.get('content')}")
            for ctx in r.get("context_after", []):
                lines.append(f"  | {ctx}")
            lines.append("")
        return "\n".join(lines)


class ImplementationResultsFormatter(ResultFormatter):
    """Formatter for find_implementations results."""

    def can_format(self, result: Any) -> bool:
        """Check if result is implementation results."""
        if not isinstance(result, list) or not result:
            return False
        first = result[0]
        return "class_name" in first and "definition" in first

    def format(self, result: list[dict[str, Any]]) -> str:
        """Format implementation results."""
        lines = [f"Found {len(result)} implementations:\n"]
        for r in result:
            lines.append(f"[class] {r.get('class_name')} ({r.get('file')}:{r.get('line')})")
            lines.append(f"  {r.get('definition')}")
            lines.append("")
        return "\n".join(lines)


class GlossaryListFormatter(ResultFormatter):
    """Formatter for glossary list results."""

    def can_format(self, result: Any) -> bool:
        """Check if result is glossary list."""
        if not isinstance(result, list) or not result:
            return False
        first = result[0]
        return "term" in first and "total_count" in first

    def format(self, result: list[dict[str, Any]]) -> str:
        """Format glossary list."""
        lines = [f"Found {len(result)} glossary terms:\n"]
        for r in result:
            lines.append(f"  {r.get('term')}: {r.get('total_count')} occurrences")
            if r.get('feature_associations'):
                assocs = r['feature_associations'][:3]  # Show top 3
                for a in assocs:
                    lines.append(f"    -> {a.get('feature_label')} ({a.get('percentage'):.0f}%)")
        return "\n".join(lines)


class RepoStatsFormatter(ResultFormatter):
    """Formatter for repository stats."""

    def can_format(self, result: Any) -> bool:
        """Check if result is repo stats."""
        if not isinstance(result, dict):
            return False
        return "elements_by_type" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format repository stats."""
        lines = ["Repository stats:"]
        lines.append(f"  Total elements: {result.get('total_elements')}")
        lines.append(f"  Total lines: {result.get('total_lines')}")
        lines.append(f"  Features: {result.get('feature_count')}")
        lines.append(f"  By type: {result.get('elements_by_type')}")
        return "\n".join(lines)


class FileReadFormatter(ResultFormatter):
    """Formatter for file read result."""

    def can_format(self, result: Any) -> bool:
        """Check if result is a file read result."""
        if not isinstance(result, dict):
            return False
        return "content" in result and "path" in result and "lines_returned" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format file read result."""
        lines = [f"File: {result.get('path')} ({result.get('lines_returned')}/{result.get('total_lines')} lines)"]
        lines.append("```")
        lines.append(result.get("content", ""))
        lines.append("```")
        return "\n".join(lines)


class FeatureMembersFormatter(ResultFormatter):
    """Formatter for feature members result."""

    def can_format(self, result: Any) -> bool:
        """Check if result is feature members."""
        if not isinstance(result, dict):
            return False
        return "members" in result and "glossary_terms" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format feature members."""
        members = result.get("members", [])
        glossary_terms = result.get("glossary_terms", [])

        lines = []

        # Format members
        if members:
            lines.append(f"Feature Members ({len(members)}):\n")
            for m in members:
                loc = f"{m.get('file', '?')}:{m.get('line', '?')}" if m.get('file') else "N/A"
                lines.append(f"[{m.get('type', '?')}] {m.get('name', '?')} ({loc})")
                if m.get('signature'):
                    lines.append(f"  {m['signature']}")
                if m.get('summary'):
                    summary = m['summary']
                    if len(summary) > 200:
                        summary = summary[:200] + "..."
                    lines.append(f"  {summary}")
                lines.append("")
        else:
            lines.append("No members.\n")

        # Format glossary terms
        if glossary_terms:
            lines.append(f"Glossary Terms ({len(glossary_terms)}):\n")
            for t in glossary_terms:
                lines.append(f"  {t.get('term')}: {t.get('frequency')} occurrences ({t.get('percentage', 0):.1f}%)")
            lines.append("")

        return "\n".join(lines)
