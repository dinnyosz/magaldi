"""Term extraction from code element names."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


COMMON_TERMS: set[str] = {
    # Verbs
    "get", "set", "add", "remove", "delete", "update", "create",
    "find", "fetch", "load", "save", "init", "handle", "process",
    "validate", "check", "is", "has", "can", "should", "do", "run",
    "start", "stop", "on", "before", "after", "pre", "post",

    # Architectural suffixes
    "service", "controller", "handler", "manager", "factory",
    "repository", "provider", "helper", "util", "utils",
    "impl", "interface", "abstract", "base", "default",
    "client", "server", "worker", "job", "task",

    # Common patterns
    "by", "for", "with", "from", "to", "and", "or", "the", "all",
    "id", "ids", "name", "type", "data", "info", "item", "items",
    "list", "array", "map", "dict", "set", "config", "options", "params",
    "request", "response", "result", "error", "exception", "status",
    "test", "spec", "mock", "stub", "fake",

    # Type-related
    "str", "string", "int", "integer", "bool", "boolean", "float",
    "none", "null", "void", "any", "object", "class", "func", "function",
    "method", "attr", "attribute", "prop", "property", "field", "key", "value",

    # Common single words
    "new", "old", "tmp", "temp", "async", "sync", "callback", "promise",
}


def split_name(name: str) -> list[str]:
    """Split an element name into component terms.

    Handles CamelCase, PascalCase, snake_case, and mixed formats.

    Args:
        name: Element name to split.

    Returns:
        List of lowercase terms.
    """
    if not name:
        return []

    # First, split on underscores and hyphens
    parts = re.split(r"[_\-]", name)

    terms: list[str] = []
    for part in parts:
        if not part:
            continue
        # Split CamelCase: insert space before uppercase letters
        # Handle sequences of uppercase (acronyms) by keeping them together
        camel_split = re.sub(r"([a-z])([A-Z])", r"\1 \2", part)
        camel_split = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", camel_split)

        for term in camel_split.split():
            lower_term = term.lower()
            if lower_term:
                terms.append(lower_term)

    return terms


def extract_terms(name: str, min_length: int = 2) -> list[str]:
    """Extract domain-specific terms from an element name.

    Splits the name, filters common programming terms, and returns
    unique domain-specific terms in order of appearance.

    Args:
        name: Element name to extract terms from.
        min_length: Minimum term length (default: 2).

    Returns:
        List of unique domain terms in order of appearance.
    """
    raw_terms = split_name(name)

    seen: set[str] = set()
    result: list[str] = []

    for term in raw_terms:
        # Skip if too short
        if len(term) < min_length:
            continue
        # Skip if common term
        if term in COMMON_TERMS:
            continue
        # Skip if already seen
        if term in seen:
            continue

        seen.add(term)
        result.append(term)

    return result


@dataclass
class GlossaryEntry:
    """A glossary term with its occurrences."""

    term: str
    total_count: int = 0
    element_ids: list[str] = field(default_factory=list)
    file_paths: list[str] = field(default_factory=list)


def aggregate_glossary_terms(
    elements: list[dict[str, Any]],
) -> dict[str, GlossaryEntry]:
    """Aggregate glossary terms from a list of elements.

    Args:
        elements: List of element dicts with 'element_id', 'name', 'relative_path'.

    Returns:
        Dict mapping term to GlossaryEntry with aggregated data.
    """
    entries: dict[str, GlossaryEntry] = {}

    for element in elements:
        element_id = element.get("element_id", "")
        name = element.get("name", "")
        file_path = element.get("relative_path", "")

        terms = extract_terms(name)

        for term in terms:
            if term not in entries:
                entries[term] = GlossaryEntry(term=term)

            entry = entries[term]
            entry.total_count += 1
            entry.element_ids.append(element_id)

            if file_path and file_path not in entry.file_paths:
                entry.file_paths.append(file_path)

    return entries
