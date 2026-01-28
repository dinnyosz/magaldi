"""Concurrency and environment variable detection.

This module provides functions for detecting:
- Environment variable usage
- Async/await patterns
- Threading and multiprocessing usage
- Lock and synchronization primitives
"""

from __future__ import annotations

import re
from typing import Any

# =============================================================================
# ENVIRONMENT VARIABLE PATTERNS
# =============================================================================

ENV_VAR_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "python": [
        (r"os\.environ\[(['\"])(\w+)\1\]", "dict_access"),
        (r"os\.environ\.get\(['\"](\w+)['\"]", "get"),
        (r"os\.getenv\(['\"](\w+)['\"]", "getenv"),
        (r"environ\.get\(['\"](\w+)['\"]", "get"),
        (r"environ\[(['\"])(\w+)\1\]", "dict_access"),
    ],
    "javascript": [
        (r"process\.env\.(\w+)", "property"),
        (r"process\.env\[(['\"])(\w+)\1\]", "bracket"),
    ],
    "typescript": [
        (r"process\.env\.(\w+)", "property"),
        (r"process\.env\[(['\"])(\w+)\1\]", "bracket"),
        (r"Deno\.env\.get\(['\"](\w+)['\"]", "get"),
    ],
    "php": [
        (r"\$_ENV\[(['\"])(\w+)\1\]", "superglobal"),
        (r"getenv\(['\"](\w+)['\"]", "getenv"),
        (r"\$_SERVER\[(['\"])(\w+)\1\]", "server"),
        (r"env\(['\"](\w+)['\"]", "helper"),
    ],
    "rust": [
        (r"env::var\(['\"](\w+)['\"]", "var"),
        (r"env::var_os\(['\"](\w+)['\"]", "var_os"),
        (r"std::env::var\(['\"](\w+)['\"]", "var"),
    ],
}


def detect_env_vars(
    _calls: list[Any] | None,  # Reserved for future call-based detection
    raw_code: str,
    language: str,
) -> list[dict[str, Any]]:
    """Detect environment variable usage in code.

    Args:
        calls: List of Call objects (not currently used, but available for future).
        raw_code: The source code to analyze.
        language: Programming language.

    Returns:
        List of env var usages: [{name, line, access_type}]
    """
    if not raw_code:
        return []

    env_vars: list[dict[str, Any]] = []
    patterns = ENV_VAR_PATTERNS.get(language, [])
    lines = raw_code.split("\n")

    for pattern_str, access_type in patterns:
        try:
            pattern = re.compile(pattern_str)
            for line_num, line in enumerate(lines, start=1):
                for match in pattern.finditer(line):
                    # Extract variable name from different match group positions
                    groups = match.groups()
                    var_name = None
                    for g in groups:
                        if g and g not in ("'", '"'):
                            var_name = g
                            break

                    if var_name:
                        env_vars.append({
                            "name": var_name,
                            "line": line_num,
                            "access_type": access_type,
                        })
        except re.error:
            continue

    # Deduplicate by name and line
    seen = set()
    unique = []
    for ev in env_vars:
        key = (ev["name"], ev["line"])
        if key not in seen:
            seen.add(key)
            unique.append(ev)

    return unique


# =============================================================================
# CONCURRENCY PATTERNS
# =============================================================================

CONCURRENCY_PATTERNS: dict[str, dict[str, list[str]]] = {
    "python": {
        "async": ["async def", "await ", "asyncio."],
        "threading": [
            "threading.Thread",
            "threading.Lock",
            "threading.RLock",
            "threading.Semaphore",
            "threading.Event",
            "threading.Condition",
            "threading.Barrier",
        ],
        "multiprocessing": [
            "multiprocessing.Process",
            "multiprocessing.Pool",
            "multiprocessing.Queue",
            "multiprocessing.Lock",
            "multiprocessing.Manager",
        ],
        "concurrent": [
            "concurrent.futures",
            "ThreadPoolExecutor",
            "ProcessPoolExecutor",
        ],
    },
    "javascript": {
        "async": ["async ", "await ", "Promise"],
        "workers": ["Worker(", "SharedWorker", "postMessage"],
    },
    "typescript": {
        "async": ["async ", "await ", "Promise"],
        "workers": ["Worker(", "SharedWorker", "postMessage"],
    },
    "php": {
        "async": ["async ", "await ", "Promise"],
        "parallel": ["parallel\\", "Thread", "Threaded", "Pool"],
    },
    "rust": {
        "async": ["async fn", "async move", ".await", "tokio::", "async-std::"],
        "threading": [
            "std::thread",
            "thread::spawn",
            "Mutex",
            "RwLock",
            "Arc<",
            "Condvar",
        ],
        "channels": ["mpsc::", "channel()", "Sender", "Receiver"],
    },
}

# Lock patterns for detecting synchronization
LOCK_PATTERNS: dict[str, list[str]] = {
    "python": ["Lock()", ".acquire()", ".release()", "with lock:", "RLock()", "Semaphore("],
    "javascript": ["Mutex", "Atomics.", "SharedArrayBuffer"],
    "typescript": ["Mutex", "Atomics.", "SharedArrayBuffer"],
    "php": ["flock(", "sem_acquire", "sem_release"],
    "rust": [".lock()", ".read()", ".write()", "Mutex::new", "RwLock::new"],
}


def detect_concurrency(
    _calls: list[Any] | None,  # Reserved for future call-based detection
    raw_code: str,
    decorators: list[str] | None,
    is_async: bool,
    language: str,
) -> dict[str, Any]:
    """Detect concurrency patterns in code.

    Args:
        calls: List of Call objects.
        raw_code: The source code to analyze.
        decorators: List of decorator names.
        is_async: Whether the element is marked as async.
        language: Programming language.

    Returns:
        Dict with is_async, uses_locks, uses_threads, patterns.
    """
    if not raw_code:
        return {
            "is_async": is_async,
            "uses_locks": False,
            "uses_threads": False,
            "patterns": [],
        }

    patterns_found: set[str] = set()
    uses_locks = False
    uses_threads = False

    # Check language-specific concurrency patterns
    lang_patterns = CONCURRENCY_PATTERNS.get(language, {})
    for pattern_type, indicators in lang_patterns.items():
        for indicator in indicators:
            if indicator in raw_code:
                patterns_found.add(pattern_type)
                if pattern_type in ("threading", "multiprocessing", "workers", "parallel"):
                    uses_threads = True
                break

    # Check for lock usage
    lock_indicators = LOCK_PATTERNS.get(language, [])
    for indicator in lock_indicators:
        if indicator in raw_code:
            uses_locks = True
            patterns_found.add("locking")
            break

    # Check decorators for async indicators
    if decorators:
        async_decorators = {"asynccontextmanager", "async_generator", "coroutine"}
        for dec in decorators:
            if dec in async_decorators or "async" in dec.lower():
                patterns_found.add("async")
                break

    # Include is_async from element metadata
    if is_async:
        patterns_found.add("async")

    return {
        "is_async": is_async or "async" in patterns_found,
        "uses_locks": uses_locks,
        "uses_threads": uses_threads,
        "patterns": sorted(patterns_found),
    }


def summarize_concurrency(elements_concurrency: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize concurrency patterns across multiple elements.

    Args:
        elements_concurrency: List of concurrency dicts from multiple elements.

    Returns:
        Summary dict with counts and patterns.
    """
    async_count = 0
    lock_count = 0
    thread_count = 0
    all_patterns: set[str] = set()

    for conc in elements_concurrency:
        if conc.get("is_async"):
            async_count += 1
        if conc.get("uses_locks"):
            lock_count += 1
        if conc.get("uses_threads"):
            thread_count += 1
        all_patterns.update(conc.get("patterns", []))

    return {
        "async_count": async_count,
        "lock_count": lock_count,
        "thread_count": thread_count,
        "patterns": sorted(all_patterns),
    }
