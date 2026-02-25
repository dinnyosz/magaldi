"""Formatters for element details and lists.

All element references use the shared compact format from _compact module.
"""

from __future__ import annotations

from typing import Any

from magaldi_mcp.formatters._compact import (
    abbrev_type,
    compact_element,
    indented_summary,
    line_range,
    truncate_hash,
)
from magaldi_mcp.formatters.base import ResultFormatter


class ElementDetailsFormatter(ResultFormatter):
    """Compact formatter for single element details."""

    def can_format(self, result: Any) -> bool:
        """Check if result is a single element with details."""
        if not isinstance(result, dict):
            return False
        return (
            "element_id" in result
            and "type" in result
            and "callers" not in result
            and "code_results" not in result
        )

    def format(self, result: dict[str, Any]) -> str:
        """Format element details using compact format."""
        abbrev = abbrev_type(result.get("type"))
        name = result.get("name", "?")
        lr = line_range(result)
        hid = truncate_hash(result.get("hash_id"))
        fp = result.get("file", "?")

        lines = [f"# {abbrev}:{name}:{lr}|{hid}"]
        lines.append(f"  file: {fp}")
        if result.get('signature'):
            lines.append(f"  sig: {result['signature']}")
        if result.get('summary'):
            lines.append(f"  {result['summary']}")
        if result.get('document_sections'):
            lines.append("  sections:")
            for s in result['document_sections']:
                indent = "  " * s.get("level", 1)
                lines.append(f"    {indent}{s.get('title')} (L{s.get('line_start')}-{s.get('line_end')})")
        if result.get('semantic_related'):
            lines.append(f"  similar ({len(result['semantic_related'])}):")
            for r in result['semantic_related']:
                score_pct = int(r.get('score', 0) * 100)
                rhid = truncate_hash(r.get('hash_id', r.get('element_id', '?')))
                lines.append(f"    {rhid} {score_pct}%")
        if result.get('code'):
            lines.append("```")
            lines.append(result['code'])
            lines.append("```")
        return "\n".join(lines)


class RepoListFormatter(ResultFormatter):
    """Compact formatter for repository list."""

    def can_format(self, result: Any) -> bool:
        """Check if result is a list of repositories."""
        if not isinstance(result, list) or not result:
            return False
        first = result[0]
        return "element_count" in first and "scope" in first

    def format(self, result: list[dict[str, Any]]) -> str:
        """Format repository list."""
        lines = [f"# repos: {len(result)}"]
        for r in result:
            lines.append(
                f"  {r.get('scope')}/{r.get('repository')}:"
                f" {r.get('element_count')} elem, {r.get('file_count')} files"
            )
        return "\n".join(lines)


class FileListFormatter(ResultFormatter):
    """Compact formatter for file list (find_files)."""

    def can_format(self, result: Any) -> bool:
        """Check if result is a list of files."""
        if not isinstance(result, list) or not result:
            return False
        first = result[0]
        return "path" in first and "element_id" not in first

    def format(self, result: list[dict[str, Any]]) -> str:
        """Format file list."""
        lines = [f"# files: {len(result)}"]
        for r in result:
            size = r.get('size') or r.get('lines', 0)
            unit = 'b' if r.get('size') else 'L'
            lines.append(f"  {r.get('path')} ({size}{unit})")
        return "\n".join(lines)


class GrepResultsFormatter(ResultFormatter):
    """Compact formatter for grep/pattern_search results."""

    def can_format(self, result: Any) -> bool:
        """Check if result is grep results."""
        if not isinstance(result, list) or not result:
            return False
        first = result[0]
        return "file" in first and "content" in first and "line" in first and "element_id" not in first

    def format(self, result: list[dict[str, Any]]) -> str:
        """Format grep results with compact file:line prefix."""
        lines = [f"# matches: {len(result)}"]
        for r in result:
            lines.append(f"{r.get('file')}:L{r.get('line')}")
            for ctx in r.get("context_before", []):
                lines.append(f"  | {ctx}")
            lines.append(f"  > {r.get('content')}")
            for ctx in r.get("context_after", []):
                lines.append(f"  | {ctx}")
        return "\n".join(lines)


class UsageResultsFormatter(ResultFormatter):
    """Compact formatter for find_usages results."""

    def can_format(self, result: Any) -> bool:
        """Check if result is usage results."""
        if not isinstance(result, list) or not result:
            return False
        first = result[0]
        return "file" in first and "content" in first and "context_before" in first and "element_id" not in first

    def format(self, result: list[dict[str, Any]]) -> str:
        """Format usage results with compact file:line prefix."""
        lines = [f"# usages: {len(result)}"]
        for r in result:
            lines.append(f"{r.get('file')}:L{r.get('line')}")
            for ctx in r.get("context_before", []):
                lines.append(f"  | {ctx}")
            lines.append(f"  > {r.get('content')}")
            for ctx in r.get("context_after", []):
                lines.append(f"  | {ctx}")
        return "\n".join(lines)


class ImplementationResultsFormatter(ResultFormatter):
    """Compact formatter for find_implementations results."""

    def can_format(self, result: Any) -> bool:
        """Check if result is implementation results."""
        if not isinstance(result, list) or not result:
            return False
        first = result[0]
        return "class_name" in first and "definition" in first

    def format(self, result: list[dict[str, Any]]) -> str:
        """Format implementation results with compact element format."""
        lines = [f"# implementations: {len(result)}"]
        for r in result:
            fp = r.get('file', '?')
            line = r.get('line', '?')
            lines.append(f"  cls:{r.get('class_name')}:L{line}  {fp}")
            lines.append(f"    {r.get('definition')}")
        return "\n".join(lines)


class GlossaryListFormatter(ResultFormatter):
    """Compact formatter for glossary list results."""

    def can_format(self, result: Any) -> bool:
        """Check if result is glossary list."""
        if not isinstance(result, list) or not result:
            return False
        first = result[0]
        return "term" in first and "total_count" in first

    def format(self, result: list[dict[str, Any]]) -> str:
        """Format glossary list."""
        lines = [f"# glossary: {len(result)} terms"]
        for r in result:
            lines.append(f"  {r.get('term')}: {r.get('total_count')}x")
            if r.get('feature_associations'):
                for a in r['feature_associations'][:3]:
                    lines.append(f"    → {a.get('feature_label')} ({a.get('percentage'):.0f}%)")
        return "\n".join(lines)


class RepoStatsFormatter(ResultFormatter):
    """Compact formatter for repository stats."""

    def can_format(self, result: Any) -> bool:
        """Check if result is repo stats."""
        if not isinstance(result, dict):
            return False
        return "elements_by_type" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format repository stats."""
        by_type = result.get('elements_by_type', {})
        type_parts = [f"{k}:{v}" for k, v in by_type.items()] if isinstance(by_type, dict) else [str(by_type)]
        lines = [
            f"# stats: {result.get('total_elements')} elem, {result.get('total_lines')} lines, {result.get('feature_count')} features",
            f"  {', '.join(type_parts)}",
        ]
        return "\n".join(lines)


class FileReadFormatter(ResultFormatter):
    """Compact formatter for file read result."""

    def can_format(self, result: Any) -> bool:
        """Check if result is a file read result."""
        if not isinstance(result, dict):
            return False
        return "content" in result and "path" in result and "lines_returned" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format file read result."""
        lines = [
            f"# file: {result.get('path')} ({result.get('lines_returned')}/{result.get('total_lines')}L)",
            "```",
            result.get("content", ""),
            "```",
        ]
        return "\n".join(lines)


class FeatureMembersFormatter(ResultFormatter):
    """Compact formatter for feature members result.

    Uses shared compact element format for member listings.
    """

    def can_format(self, result: Any) -> bool:
        """Check if result is feature members."""
        if not isinstance(result, dict):
            return False
        return "members" in result and "glossary_terms" in result

    def format(self, result: dict[str, Any]) -> str:
        """Format feature members with compact element format."""
        members = result.get("members", [])
        glossary_terms = result.get("glossary_terms", [])

        lines: list[str] = []

        if members:
            lines.append(f"Members ({len(members)}):\n")
            for m in members:
                lines.append(compact_element(m, indent=0))
                if m.get('signature'):
                    lines.append(f"  {m['signature']}")
                summary = indented_summary(m, indent=2)
                if summary:
                    lines.append(summary)
                lines.append("")
        else:
            lines.append("No members.\n")

        if glossary_terms:
            lines.append(f"Glossary ({len(glossary_terms)}):")
            for t in glossary_terms:
                lines.append(f"  {t.get('term')}: {t.get('frequency')} ({t.get('percentage', 0):.1f}%)")

        return "\n".join(lines)
