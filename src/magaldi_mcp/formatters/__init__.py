"""Formatters for MCP tool results."""

from magaldi_mcp.formatters.base import ResultFormatter
from magaldi_mcp.formatters.registry import FormatterRegistry, format_result

__all__ = [
    "format_result",
    "FormatterRegistry",
    "ResultFormatter",
]
