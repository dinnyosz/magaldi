"""Shared compact formatting utilities for MCP tool output.

All formatters should use these utilities to ensure consistent element
representation across every tool. The canonical element line format is:

    {abbrev}:{name}:L{start}[-{end}]|{hash8}

When a score is relevant (search, similarity):

    {abbrev}:{name}:L{start}[-{end}]:{score:.2f}|{hash8}

Directional references (callers/callees) use arrow prefixes:

    ← {abbrev}:{name}:L{start}[-{end}]|{hash8}
    → {abbrev}:{name}:L{start}[-{end}]|{hash8}

Unresolved callees:

    → ?:{receiver}.{name}():L{line}
"""

from __future__ import annotations

from typing import Any

# Canonical type abbreviation map. Every formatter MUST use this.
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

# Standard truncation lengths
SUMMARY_MAX = 200
SUMMARY_SHORT = 100
HASH_LEN = 8


def abbrev_type(elem_type: str | None) -> str:
    """Abbreviate an element type using the canonical map.

    Unknown types pass through unchanged.
    """
    return TYPE_ABBREV.get(elem_type or "?", elem_type or "?")


def truncate_hash(hash_id: str | None) -> str:
    """Truncate a hash ID to HASH_LEN characters."""
    if not hash_id:
        return ""
    return hash_id[:HASH_LEN]


def line_range(r: dict[str, Any]) -> str:
    """Build a line range string like 'L42' or 'L42-78'.

    Handles both 'line'/'line_start' and 'line_end' field name variants.
    """
    start = r.get("line") or r.get("line_start") or "?"
    end = r.get("line_end")
    if end and end != start:
        return f"L{start}-{end}"
    return f"L{start}"


def file_path(r: dict[str, Any]) -> str:
    """Extract file path, handling field name variants."""
    return r.get("file") or r.get("relative_path") or "unknown"


def compact_element(r: dict[str, Any], *, indent: int = 2, score: bool = False) -> str:
    """Format a single element as one compact line.

    Args:
        r: Element dict with name, type, line, hash_id, etc.
        indent: Number of leading spaces.
        score: Whether to include the relevance score.

    Returns:
        Compact line like '  fn:process:L42-78:0.85|ab12cd34'
        or without score: '  fn:process:L42-78|ab12cd34'
    """
    prefix = " " * indent
    abbrev = abbrev_type(r.get("type") or r.get("element_type"))
    name = r.get("name", "?")
    lr = line_range(r)
    hid = truncate_hash(r.get("hash_id"))

    if score:
        s = r.get("score", 0.0)
        return f"{prefix}{abbrev}:{name}:{lr}:{s:.2f}|{hid}"
    return f"{prefix}{abbrev}:{name}:{lr}|{hid}"


def compact_ref(
    r: dict[str, Any],
    *,
    arrow: str = "",
    indent: int = 2,
    score: bool = False,
) -> str:
    """Format an element reference with an optional directional arrow.

    Args:
        r: Element dict.
        arrow: '←' for callers, '→' for callees, '' for neutral.
        indent: Number of leading spaces.
        score: Whether to include the relevance score.

    Returns:
        Line like '  ← fn:process:L42|ab12cd34'
    """
    prefix = " " * indent
    abbrev = abbrev_type(r.get("type") or r.get("element_type"))
    name = r.get("name", "?")
    lr = line_range(r)
    hid = truncate_hash(r.get("hash_id"))

    arrow_prefix = f"{arrow} " if arrow else ""

    if score:
        s = r.get("score", 0.0)
        return f"{prefix}{arrow_prefix}{abbrev}:{name}:{lr}:{s:.2f}|{hid}"
    return f"{prefix}{arrow_prefix}{abbrev}:{name}:{lr}|{hid}"


def compact_unresolved(r: dict[str, Any], *, indent: int = 2) -> str:
    """Format an unresolved callee reference.

    Returns:
        Line like '  → ?:Receiver.method():L42'
    """
    prefix = " " * indent
    receiver = f"{r.get('receiver')}." if r.get("receiver") else ""
    name = r.get("name", "?")
    line = r.get("line", "?")
    return f"{prefix}→ ?:{receiver}{name}():L{line}"


def compact_element_header(r: dict[str, Any]) -> str:
    """Format an element as a compact header line (no indent, for section titles).

    Returns:
        Line like 'fn:process:L42-78|ab12cd34'
    """
    return compact_element(r, indent=0)


def truncate_summary(summary: str | None, max_len: int = SUMMARY_MAX) -> str | None:
    """Truncate a summary string to max_len characters."""
    if not summary:
        return None
    if len(summary) > max_len:
        return summary[:max_len] + "..."
    return summary


def indented_summary(
    r: dict[str, Any], *, indent: int = 4, max_len: int = SUMMARY_MAX,
) -> str | None:
    """Return an indented summary line or None if no summary.

    Args:
        r: Element dict with optional 'summary' key.
        indent: Number of leading spaces.
        max_len: Truncation length.

    Returns:
        Indented summary like '    Processes data and returns result...'
        or None if no summary.
    """
    summary = truncate_summary(r.get("summary"), max_len)
    if summary:
        prefix = " " * indent
        return f"{prefix}{summary}"
    return None


def file_group(
    results: list[dict[str, Any]],
) -> list[tuple[str, float, list[dict[str, Any]]]]:
    """Group results by file path, sorted by average score descending.

    Returns:
        List of (file_path, avg_score, elements) tuples.
        Elements within each group are sorted by score descending.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        fp = file_path(r)
        groups.setdefault(fp, []).append(r)

    # Sort elements within each group by score desc
    for elements in groups.values():
        elements.sort(key=lambda e: e.get("score", 0.0), reverse=True)

    def _avg_score(elements: list[dict[str, Any]]) -> float:
        scores = [e.get("score", 0.0) for e in elements]
        return sum(scores) / len(scores) if scores else 0.0

    return sorted(
        [(fp, _avg_score(elems), elems) for fp, elems in groups.items()],
        key=lambda t: t[1],
        reverse=True,
    )
