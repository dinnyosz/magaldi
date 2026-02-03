"""Formatters for search results."""

from __future__ import annotations

from typing import Any

from magaldi_mcp.formatters.base import ResultFormatter


class CodeSearchListFormatter(ResultFormatter):
    """Formatter for code search results (list of elements)."""

    def can_format(self, result: Any) -> bool:
        """Check if result is a list of code elements."""
        if not isinstance(result, list) or not result:
            return False
        first = result[0]
        return "element_id" in first and "type" in first and "feature_id" not in first

    def format(self, result: list[dict[str, Any]]) -> str:
        """Format code search results."""
        lines = [f"Found {len(result)} results:\n"]
        for r in result:
            line_start = r.get('line', '?')
            line_end = r.get('line_end', '')
            line_range = f"{line_start}-{line_end}" if line_end else str(line_start)
            loc = f"{r.get('file', '?')}:{line_range}" if r.get('file') else "N/A"
            hash_id = r.get('hash_id', '')
            id_suffix = f" | id:{hash_id}" if hash_id else ""
            lines.append(f"[{r.get('type', '?')}] {r.get('name', '?')} ({loc}){id_suffix}")
            if r.get('signature'):
                lines.append(f"  {r['signature']}")
            if r.get('summary'):
                summary = r['summary']
                if len(summary) > 200:
                    summary = summary[:200] + "..."
                lines.append(f"  {summary}")
            if r.get('code'):
                lines.append("  ```")
                lines.append(r['code'])
                lines.append("  ```")
            lines.append("")
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
    """Formatter for grouped search results (code_results and test_results)."""

    def can_format(self, result: Any) -> bool:
        """Check if result has code_results and test_results."""
        if not isinstance(result, dict):
            return False
        return "code_results" in result and "test_results" in result and "target" not in result

    def format(self, result: dict[str, Any]) -> str:
        """Format grouped search results."""
        code_results = result.get("code_results", [])
        test_results = result.get("test_results", [])
        total_code = result.get("total_code", len(code_results))
        total_tests = result.get("total_tests", len(test_results))

        lines = []

        # Format code results
        if code_results:
            lines.append(f"Code Results ({total_code}):\n")
            for r in code_results:
                line_start = r.get('line', '?')
                line_end = r.get('line_end', '')
                line_range = f"{line_start}-{line_end}" if line_end else str(line_start)
                loc = f"{r.get('file', '?')}:{line_range}" if r.get('file') else "N/A"
                hash_id = r.get('hash_id', '')
                id_suffix = f" | id:{hash_id}" if hash_id else ""
                lines.append(f"[{r.get('type', '?')}] {r.get('name', '?')} ({loc}){id_suffix}")
                if r.get('signature'):
                    lines.append(f"  {r['signature']}")
                if r.get('summary'):
                    summary = r['summary']
                    if len(summary) > 200:
                        summary = summary[:200] + "..."
                    lines.append(f"  {summary}")
                if r.get('code'):
                    lines.append("  ```")
                    lines.append(r['code'])
                    lines.append("  ```")
                # For grep results
                if r.get('content') and r.get('match'):
                    for ctx in r.get("context_before", []):
                        lines.append(f"  | {ctx}")
                    lines.append(f"  > {r.get('content')}")
                    for ctx in r.get("context_after", []):
                        lines.append(f"  | {ctx}")
                lines.append("")
        else:
            lines.append("No code results.\n")

        # Format test results
        if test_results:
            lines.append(f"Test Results ({total_tests}):\n")
            for r in test_results:
                line_start = r.get('line', '?')
                line_end = r.get('line_end', '')
                line_range = f"{line_start}-{line_end}" if line_end else str(line_start)
                loc = f"{r.get('file', '?')}:{line_range}" if r.get('file') else "N/A"
                hash_id = r.get('hash_id', '')
                id_suffix = f" | id:{hash_id}" if hash_id else ""
                lines.append(f"[{r.get('type', '?')}] {r.get('name', '?')} ({loc}){id_suffix}")
                if r.get('signature'):
                    lines.append(f"  {r['signature']}")
                if r.get('summary'):
                    summary = r['summary']
                    if len(summary) > 200:
                        summary = summary[:200] + "..."
                    lines.append(f"  {summary}")
                if r.get('code'):
                    lines.append("  ```")
                    lines.append(r['code'])
                    lines.append("  ```")
                # For grep results
                if r.get('content') and r.get('match'):
                    for ctx in r.get("context_before", []):
                        lines.append(f"  | {ctx}")
                    lines.append(f"  > {r.get('content')}")
                    for ctx in r.get("context_after", []):
                        lines.append(f"  | {ctx}")
                lines.append("")
        elif result.get("total_tests", 0) > 0:
            lines.append(f"(Tests excluded: {total_tests})")

        return "\n".join(lines)
