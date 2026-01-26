"""Base classes for result formatters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


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
