"""Formatter registry that tries formatters in sequence."""

from __future__ import annotations

import json
from typing import Any

from magaldi_mcp.formatters.analysis import (
    CallChainFormatter,
    CallGraphFormatter,
    DeadCodeFormatter,
    EntryPointsFormatter,
    ExplainElementFormatter,
    FindCallersFormatter,
)
from magaldi_mcp.formatters.base import ResultFormatter
from magaldi_mcp.formatters.dependencies import (
    DependencyFormatter,
    DependencyGraphFormatter,
    DependentFormatter,
)
from magaldi_mcp.formatters.elements import (
    ElementDetailsFormatter,
    FeatureMembersFormatter,
    FileListFormatter,
    FileReadFormatter,
    GlossaryListFormatter,
    GrepResultsFormatter,
    ImplementationResultsFormatter,
    RepoListFormatter,
    RepoStatsFormatter,
    UsageResultsFormatter,
)
from magaldi_mcp.formatters.search import (
    CodeSearchListFormatter,
    FeatureSearchListFormatter,
    GroupedSearchFormatter,
)


class FormatterRegistry:
    """Registry of result formatters.

    Formatters are tried in order and the first matching one is used.
    """

    def __init__(self) -> None:
        self._formatters: list[ResultFormatter] = []
        self._register_default_formatters()

    def _register_default_formatters(self) -> None:
        """Register all default formatters in priority order."""
        # More specific formatters should come first

        # Search results
        self._formatters.append(GroupedSearchFormatter())  # Has code_results/test_results
        self._formatters.append(FindCallersFormatter())  # Has target + code_results/test_results
        self._formatters.append(FeatureSearchListFormatter())  # List with feature_id
        self._formatters.append(CodeSearchListFormatter())  # List with element_id/type

        # Element details
        self._formatters.append(ExplainElementFormatter())  # Has element + similar_code + parent
        self._formatters.append(ElementDetailsFormatter())  # Has element_id + type (but not callers)
        self._formatters.append(FeatureMembersFormatter())  # Has members + glossary_terms
        self._formatters.append(FileReadFormatter())  # Has content + path + lines_returned

        # Repository/file lists
        self._formatters.append(RepoListFormatter())  # List with element_count + scope
        self._formatters.append(FileListFormatter())  # List with path (no element_id)
        self._formatters.append(RepoStatsFormatter())  # Has elements_by_type

        # Grep/usage results
        self._formatters.append(UsageResultsFormatter())  # List with file + content + context_before
        self._formatters.append(GrepResultsFormatter())  # List with file + content + line
        self._formatters.append(ImplementationResultsFormatter())  # List with class_name + definition
        self._formatters.append(GlossaryListFormatter())  # List with term + total_count

        # Analysis results
        self._formatters.append(CallGraphFormatter())  # Has callers + callees + element
        self._formatters.append(CallChainFormatter())  # Has root + direction
        self._formatters.append(DeadCodeFormatter())  # Has potentially_dead + stats
        self._formatters.append(EntryPointsFormatter())  # Has http + cli + test + main

        # Dependency results
        self._formatters.append(DependencyFormatter())  # Has internal_imports + external_imports
        self._formatters.append(DependentFormatter())  # Has module + dependents + total
        self._formatters.append(DependencyGraphFormatter())  # Has nodes + edges + cycles

    def register(self, formatter: ResultFormatter) -> None:
        """Register a formatter at the beginning of the list (highest priority)."""
        self._formatters.insert(0, formatter)

    def format(self, result: Any) -> str:
        """Format a result using the first matching formatter.

        Falls back to JSON for unrecognized results.
        """
        for formatter in self._formatters:
            if formatter.can_format(result):
                formatted: str = formatter.format(result)
                return formatted

        # Fallback to JSON
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2, default=str)
        return str(result)


# Global registry instance
_registry = FormatterRegistry()


def format_result(result: Any) -> str:
    """Format a tool result using the global formatter registry.

    This is the main entry point for formatting results.

    Args:
        result: The result to format.

    Returns:
        Formatted string representation.
    """
    return _registry.format(result)
