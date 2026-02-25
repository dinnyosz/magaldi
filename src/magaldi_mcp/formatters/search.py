"""Formatters for search results.

All element references use the shared compact format from _compact module
for consistent output across all tools.
"""

from __future__ import annotations

from typing import Any

from magaldi_mcp.formatters._compact import (
    compact_element,
    file_group,
    indented_summary,
)
from magaldi_mcp.formatters.base import ResultFormatter


class CodeSearchListFormatter(ResultFormatter):
    """Compact formatter for code search results (list of elements).

    Uses file-grouped layout with compact element format, consistent
    with GroupedSearchFormatter output.
    """

    def can_format(self, result: Any) -> bool:
        """Check if result is a list of code elements."""
        if not isinstance(result, list) or not result:
            return False
        first = result[0]
        return "element_id" in first and "type" in first and "feature_id" not in first

    def format(self, result: list[dict[str, Any]]) -> str:
        """Format code search results with compact file-grouped output."""
        if not result:
            return "No results found."

        lines = [
            f"# search: {len(result)} results",
            "# type:name:L<start>[-end]:score|<hash8>",
        ]

        for fp, avg, elements in file_group(result):
            lines.append(f"\n{fp}  avg:{avg:.2f}")
            for r in elements:
                lines.append(compact_element(r, score=True))
                summary = indented_summary(r, indent=4)
                if summary:
                    lines.append(summary)

        return "\n".join(lines)


class FeatureSearchListFormatter(ResultFormatter):
    """Formatter for feature search results (list of features)."""

    def can_format(self, result: Any) -> bool:
        """Check if result is a list of features."""
        if not isinstance(result, list) or not result:
            return False
        return "feature_id" in result[0] or "subfeature_id" in result[0]

    def format(self, result: list[dict[str, Any]]) -> str:
        """Format feature search results."""
        lines = [f"Found {len(result)} features:\n"]
        for r in result:
            # Use hash_id (stable) over feature_id/subfeature_id
            hash_id = r.get('hash_id', '')
            id_suffix = f" | id:{hash_id}" if hash_id else ""
            # Determine type - only show if mixed or subfeature
            elem_type = r.get('type', 'feature')
            type_prefix = f"[{elem_type}] " if elem_type == 'subfeature' else ""
            lines.append(f"{type_prefix}{r.get('label', '?')} ({r.get('member_count', 0)} members){id_suffix}")
            if r.get('summary'):
                summary = r['summary']
                if len(summary) > 200:
                    summary = summary[:200] + "..."
                lines.append(f"  {summary}")
            lines.append("")
        return "\n".join(lines)


class GroupedSearchFormatter(ResultFormatter):
    """Compact file-grouped formatter for search results.

    Uses shared _compact utilities for consistent element rendering.
    Groups elements by file, sorted by average relevance score.
    """

    def can_format(self, result: Any) -> bool:
        """Check if result has code_results and test_results."""
        if not isinstance(result, dict):
            return False
        return "code_results" in result and "test_results" in result and "target" not in result

    def format(self, result: dict[str, Any]) -> str:
        """Format results as compact file-grouped output."""
        code_results = result.get("code_results", [])
        test_results = result.get("test_results", [])

        all_results = code_results + test_results
        total = len(all_results)

        if not all_results:
            return "No results found."

        lines: list[str] = [
            f"# search_code: {total} results (scored 0-1, desc)",
            "# type:name:L<start>[-end]:score|<hash8>",
        ]

        for fp, avg, elements in file_group(all_results):
            lines.append(f"\n{fp}  avg:{avg:.2f}")
            for r in elements:
                lines.append(compact_element(r, score=True))
                summary = indented_summary(r, indent=4)
                if summary:
                    lines.append(summary)

        return "\n".join(lines)
