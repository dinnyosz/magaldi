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
    """Compact file-grouped formatter for search results.

    Groups elements by file, sorted by average relevance score.
    Elements within each file sorted by score descending.
    """

    TYPE_ABBREV: dict[str, str] = {
        "function": "fn",
        "method": "mth",
        "class": "cls",
        "constant": "const",
        "variable": "var",
        "interface": "iface",
        "enum": "enum",
        "type_alias": "type",
        "import": "imp",
        "file": "file",
        "trait": "trait",
    }

    def can_format(self, result: Any) -> bool:
        """Check if result has code_results and test_results."""
        if not isinstance(result, dict):
            return False
        return "code_results" in result and "test_results" in result and "target" not in result

    def format(self, result: dict[str, Any]) -> str:
        """Format results as compact file-grouped output."""
        code_results = result.get("code_results", [])
        test_results = result.get("test_results", [])

        # Merge all results
        all_results = code_results + test_results
        total = len(all_results)

        if not all_results:
            return "No results found."

        # Group by file
        file_groups: dict[str, list[dict[str, Any]]] = {}
        for r in all_results:
            file_path = r.get("file") or r.get("relative_path") or "unknown"
            file_groups.setdefault(file_path, []).append(r)

        # Sort elements within each file by score desc
        for elements in file_groups.values():
            elements.sort(key=lambda e: e.get("score", 0.0), reverse=True)

        # Sort file groups by avg score desc
        def _avg_score(elements: list[dict[str, Any]]) -> float:
            scores = [e.get("score", 0.0) for e in elements]
            return sum(scores) / len(scores) if scores else 0.0

        sorted_files = sorted(
            file_groups.items(), key=lambda item: _avg_score(item[1]), reverse=True,
        )

        lines: list[str] = [
            f"# search_code: {total} results (scored 0-1, desc)",
            "# type:name:L<start>[-end]:score|<hash8>",
        ]

        for file_path, elements in sorted_files:
            avg = _avg_score(elements)
            lines.append(f"\n{file_path}  avg:{avg:.2f}")

            for r in elements:
                lines.append(self._format_element(r))

                # Non-brief: summary and signature indented under element
                if r.get("summary"):
                    summary = r["summary"]
                    if len(summary) > 200:
                        summary = summary[:200] + "..."
                    lines.append(f"    {summary}")
                if r.get("signature"):
                    lines.append(f"    {r['signature']}")
                if r.get("code") or r.get("raw_code"):
                    code = r.get("code") or r.get("raw_code")
                    lines.append("    ```")
                    for code_line in code.splitlines():
                        lines.append(f"    {code_line}")
                    lines.append("    ```")

        return "\n".join(lines)

    def _format_element(self, r: dict[str, Any]) -> str:
        """Format a single element as a compact line."""
        elem_type = r.get("type") or r.get("element_type") or "?"
        abbrev = self.TYPE_ABBREV.get(elem_type, elem_type)
        name = r.get("name", "?")
        line_start = r.get("line") or r.get("line_start") or "?"
        line_end = r.get("line_end")
        line_range = f"L{line_start}-{line_end}" if line_end else f"L{line_start}"
        score = r.get("score", 0.0)
        hash_id = r.get("hash_id", "")
        return f"  {abbrev}:{name}:{line_range}:{score:.2f}|{hash_id}"
