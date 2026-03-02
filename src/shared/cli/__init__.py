"""Magaldi CLI - Command Line Interface.

This package provides the CLI commands for Magaldi.

Commands are organized in separate modules:
- _shared: Common utilities, console setup, and main CLI group
- _runners: Phase runner functions
- _printers: Output formatting functions
- parse: Parse command
- watch: Watch command (continuous file monitoring)
- extract: Shared utilities for feature/glossary extraction
- feature_commands: Extract-features command
- glossary_commands: Extract-glossary command
- web: Web server commands
- benchmarks: Benchmark-models command (package)
"""

from __future__ import annotations

# Import command modules to register their commands with main
# The import side-effect registers @main.command decorators
from shared.cli import (
    benchmarks,  # noqa: F401
    feature_commands,  # noqa: F401
    glossary_commands,  # noqa: F401
    llm,  # noqa: F401
    parse,  # noqa: F401
    watch,  # noqa: F401
    web,  # noqa: F401
)
from shared.cli._printers import (
    print_change_manifest,
    print_discovery_result,
    print_feature_result,
    print_parsing_result,
    print_processing_result,
    print_summary,
)
from shared.cli._runners import (
    run_change_detection,
    run_discovery,
    run_parsing,
    run_processing,
)

# Import the main CLI group first
# Re-export commonly used items
from shared.cli._shared import check_model_availability, console, format_duration, main
from shared.cli.feature_commands import run_feature_extraction
from shared.cli.glossary_commands import run_glossary_extraction

__all__ = [
    # Main CLI
    "main",
    # Console
    "console",
    "format_duration",
    "check_model_availability",
    # Runners
    "run_discovery",
    "run_change_detection",
    "run_parsing",
    "run_processing",
    "run_feature_extraction",
    "run_glossary_extraction",
    # Printers
    "print_discovery_result",
    "print_change_manifest",
    "print_parsing_result",
    "print_feature_result",
    "print_processing_result",
    "print_summary",
]
