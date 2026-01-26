"""Magaldi CLI - Command Line Interface.

This package provides the CLI commands for Magaldi.

Commands are organized in separate modules:
- _shared: Common utilities, console setup, and main CLI group
- _runners: Phase runner functions
- _printers: Output formatting functions
- parse: Parse command
- extract: Extract-features and extract-glossary commands
- web: Web server commands
- benchmark: Benchmark-models command
"""

from __future__ import annotations

# Import the main CLI group first
from shared.cli._shared import main

# Import command modules to register their commands with main
# The import side-effect registers @main.command decorators
from shared.cli import parse  # noqa: F401
from shared.cli import extract  # noqa: F401
from shared.cli import web  # noqa: F401
from shared.cli import benchmark  # noqa: F401

# Re-export commonly used items
from shared.cli._shared import console, format_duration, check_model_availability
from shared.cli._runners import (
    run_discovery,
    run_change_detection,
    run_parsing,
    run_processing,
)
from shared.cli._printers import (
    print_discovery_result,
    print_change_manifest,
    print_parsing_result,
    print_feature_result,
    print_processing_result,
    print_summary,
)
from shared.cli.extract import run_feature_extraction, run_glossary_extraction

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
