"""Base classes for result formatters."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any


def short_id(hash_id: str, length: int = 12) -> str:
    """Truncate hash_id for display."""
    return hash_id[:length] if hash_id else ""


def terse_summary(summary: str, max_length: int = 200) -> str:
    """Make summary more concise by stripping verbose prefixes."""
    if not summary:
        return ""

    # Strip common verbose patterns from AI-generated summaries
    patterns = [
        r"^The `[^`]+` (?:function|method|class) is used to ",
        r"^The `[^`]+` (?:function|method|class) ",
        r"^This (?:function|method|class) is used to ",
        r"^This (?:function|method|class) ",
    ]

    result = summary
    for pattern in patterns:
        result = re.sub(pattern, "", result, count=1, flags=re.IGNORECASE)

    # Capitalize first letter if we stripped something
    if result and result != summary:
        result = result[0].upper() + result[1:] if len(result) > 1 else result.upper()

    # Truncate if needed
    if len(result) > max_length:
        result = result[:max_length] + "..."

    return result


class ResultFormatter(ABC):
    """Abstract base class for result formatters.

    Each formatter handles a specific type of result structure.
    """

    @abstractmethod
    def can_format(self, result: Any) -> bool:
        """Check if this formatter can handle the given result.

        Args:
            result: The result to check.

        Returns:
            True if this formatter can handle the result.
        """
        pass

    @abstractmethod
    def format(self, result: Any) -> str:
        """Format the result as a string.

        Args:
            result: The result to format.

        Returns:
            Formatted string representation.
        """
        pass
