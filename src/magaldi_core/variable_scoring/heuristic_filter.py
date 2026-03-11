"""Conservative heuristic pre-filtering for variable scoring.

Drops obvious low-value variables before LLM scoring to reduce costs.
Conservative by design: false KEEPs are fine (LLM scores them low),
false DROPs are bad (we lose elements permanently).

Rules applied (in order):
1. Single-character names (a-z, _) -> DROP
2. Known throwaway names with simple assignments -> DROP
3. Generic function-call results (lowercase, not framework names) -> DROP
"""

from __future__ import annotations

import re

# Names that are almost always throwaway locals.
# Only dropped when the raw_code is a simple single-line assignment.
_THROWAWAY_NAMES: frozenset[str] = frozenset({
    "i", "j", "k", "n", "m", "x", "y", "z",
    "tmp", "temp",
    "result", "results", "res",
    "response", "resp",
    "data", "val", "value", "values",
    "err", "error",
    "ret", "rv", "retval",
    "output", "out",
    "buf", "buffer",
    "line", "lines",
    "item", "items", "elem",
    "obj",
    "key", "keys",
    "idx", "index",
    "count", "total", "num",
    "msg", "message",
    "path", "name", "text", "body",
    "row", "col",
    "args", "kwargs",
    "match", "matches",
    "parts", "tokens",
    "entry", "entries",
    "chunk", "node",
})

# Names that suggest framework/infrastructure instances — never drop these
# even if they look like simple function call results.
_FRAMEWORK_PATTERNS: frozenset[str] = frozenset({
    "app", "application",
    "logger", "log",
    "engine",
    "client", "session",
    "connection", "conn",
    "db", "database",
    "router", "route",
    "server",
    "cache",
    "queue",
    "pool",
    "registry",
    "factory",
    "handler",
    "middleware",
    "dispatcher",
    "scheduler",
    "manager",
    "provider",
    "adapter",
    "driver",
    "socket", "sock",
    "transport",
    "channel",
    "broker",
    "emitter",
    "watcher",
    "monitor",
    "tracer",
    "serializer",
    "parser",
    "compiler",
    "scanner",
    "lexer",
    "resolver",
    "validator",
    "sanitizer",
    "formatter",
    "renderer",
    "template",
    "blueprint",
    "namespace",
    "context", "ctx",
})


def _normalize_raw_code(name: str, raw_code: str) -> str:
    """Strip language-specific prefixes so raw_code starts with the variable name.

    Different languages produce raw_code with various prefixes:
    - Rust: ``let data = ...`` or ``let mut data = ...``
    - PHP: ``$data = ...`` (name stored without ``$``)
    - Java: ``private static final int data = ...``
    - Python/JS/TS/Bash: already start with the name

    Returns the stripped code, or the original if no normalization applies.
    """
    code = raw_code.strip()

    # Fast path: already starts with name (Python, JS/TS, Bash)
    if code.startswith(name):
        return code

    # PHP: $name → strip the leading $
    if code.startswith(f"${name}"):
        return code[1:]

    # Rust/Java/etc: find `name` after language prefixes.
    # Look for the variable name followed by an assignment, type annotation,
    # or statement terminator.
    for suffix in (" =", "=", ":", ";", "\n"):
        idx = code.find(f"{name}{suffix}")
        if idx >= 0:
            return code[idx:]

    return code


def _is_simple_assignment(name: str, raw_code: str) -> bool:
    """Check if raw_code is a simple single-line assignment.

    Multi-line code, dict/list literals, or complex expressions are NOT simple.
    """
    code = _normalize_raw_code(name, raw_code)
    if not code.startswith(name):
        return False

    # Multi-line → likely complex (dict literal, list comprehension, etc.)
    lines = code.split("\n")
    if len(lines) > 2:
        return False

    # Pattern: name = expr or name: Type = expr
    return bool(re.match(
        rf"^{re.escape(name)}\s*(?::\s*[\w\[\], |<>?]+\s*)?=\s*.+",
        code,
    ))


def _is_function_call_result(raw_code: str, name: str) -> bool:
    """Check if raw_code is ``name = func(...)`` or ``name = obj.method(...)``.

    Only matches single-line assignments where the RHS is a plain call.
    """
    code = _normalize_raw_code(name, raw_code)
    # Must be single-line
    if "\n" in code:
        return False

    # Pattern: name = func(...) or name: Type = func(...)
    # Allow optional type annotation (including generics like Map<K, V>)
    return bool(re.match(
        rf"^{re.escape(name)}\s*(?::\s*[\w\[\], |<>?]+\s*)?=\s*[\w.]+\([^)]*\)\s*;?\s*$",
        code,
    ))


def _has_framework_name(name: str) -> bool:
    """Check if the variable name suggests a framework/infrastructure object."""
    lower = name.lower()
    # Exact match
    if lower in _FRAMEWORK_PATTERNS:
        return True
    # Suffix match: e.g. "redis_client", "db_connection", "my_logger"
    for pattern in _FRAMEWORK_PATTERNS:
        if lower.endswith(f"_{pattern}") or lower.startswith(f"{pattern}_"):
            return True
    return False


def should_drop_variable(name: str, raw_code: str) -> tuple[bool, str]:
    """Determine if a variable should be dropped before LLM scoring.

    Args:
        name: Variable name.
        raw_code: Full raw code of the variable declaration.

    Returns:
        Tuple of (should_drop, reason).
    """
    # ALWAYS KEEP: UPPER_CASE names (constants, config, enums)
    if name.isupper() and len(name) > 1:
        return False, ""

    # ALWAYS KEEP: dunder names (__all__, __version__, etc.)
    if name.startswith("__") and name.endswith("__"):
        return False, ""

    # Rule 1: Single-character names
    if len(name) <= 1:
        return True, "single_letter"

    # Rule 2: Known throwaway names with simple assignments
    if name.lower() in _THROWAWAY_NAMES and _is_simple_assignment(name, raw_code):
        return True, "throwaway_name"

    # Rule 3: Generic function-call results
    if _is_function_call_result(raw_code, name):
        # Keep framework/infrastructure instances
        if _has_framework_name(name):
            return False, ""
        # Keep if name has any uppercase (likely intentional: MyClass, etc.)
        if any(c.isupper() for c in name):
            return False, ""
        # Drop generic lowercase function results
        if name.lower() in _THROWAWAY_NAMES:
            return True, "throwaway_call_result"
        # For non-throwaway names, only drop very short ones
        if len(name) <= 4:
            return True, "short_call_result"

    # Default: KEEP — let the LLM decide
    return False, ""


def apply_heuristic_filter(
    variables: list[tuple[str, str, str, str]],
) -> tuple[list[tuple[str, str, str, str]], dict[str, str]]:
    """Apply heuristic pre-filtering to variables before LLM scoring.

    Conservative: only drops variables that are almost certainly low-value.

    Args:
        variables: List of ``(element_id, file_path, name, raw_code)`` tuples.

    Returns:
        Tuple of ``(kept_variables, drop_reasons)`` where *drop_reasons* maps
        ``element_id -> reason`` for dropped variables.
    """
    kept: list[tuple[str, str, str, str]] = []
    drops: dict[str, str] = {}

    for element_id, file_path, name, raw_code in variables:
        drop, reason = should_drop_variable(name, raw_code)
        if drop:
            drops[element_id] = reason
        else:
            kept.append((element_id, file_path, name, raw_code))

    return kept, drops
