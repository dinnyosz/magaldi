"""Term extraction from code element names."""

from __future__ import annotations

import re


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
