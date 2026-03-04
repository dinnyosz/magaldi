"""Text utilities for code identifier processing.

Provides functions for converting code identifiers (snake_case, camelCase, etc.)
into human-readable text suitable for embedding models and display.
"""

from __future__ import annotations

import re

# Regex for splitting camelCase/PascalCase boundaries:
# - Between lowercase and uppercase: "parseHTML" -> "parse" + "HTML"
# - Between uppercase run and uppercase+lowercase: "HTMLParser" -> "HTML" + "Parser"
_CAMEL_SPLIT_RE = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])"  # aB or 2B boundary
    r"|(?<=[A-Z])(?=[A-Z][a-z])"  # ABc boundary (acronym end)
)


def humanize_name(name: str) -> str:
    """Convert a code identifier to human-readable lowercase text.

    Splits snake_case, camelCase, PascalCase, and SCREAMING_SNAKE_CASE
    into space-separated lowercase words. Preserves natural reading order
    for embedding model consumption.

    Args:
        name: Code identifier (e.g., "test_parseHTMLContent").

    Returns:
        Human-readable string (e.g., "test parse html content").

    Examples:
        >>> humanize_name("test_parse_html")
        'test parse html'
        >>> humanize_name("parseHTMLContent")
        'parse html content'
        >>> humanize_name("TestUserAuthentication")
        'test user authentication'
        >>> humanize_name("MAX_RETRY_COUNT")
        'max retry count'
        >>> humanize_name("test_parseHTMLContent")
        'test parse html content'
        >>> humanize_name("__init__")
        'init'
        >>> humanize_name("")
        ''
    """
    if not name:
        return ""

    # Step 1: Split on underscores
    parts = name.split("_")

    # Step 2: Split each part on camelCase/PascalCase boundaries
    words: list[str] = []
    for part in parts:
        if not part:
            continue
        sub_parts = _CAMEL_SPLIT_RE.split(part)
        words.extend(sub_parts)

    # Step 3: Lowercase and join
    return " ".join(w.lower() for w in words if w)
