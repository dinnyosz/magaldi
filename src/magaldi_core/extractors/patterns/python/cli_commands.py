"""Python CLI framework command patterns and detection.

Pattern definitions and detection functions for CLI commands
in Python CLI frameworks:
- Click
- Typer
"""

from __future__ import annotations

from magaldi_core.extractors.types import CliCommand, DecoratorInfo

# =============================================================================
# CLI COMMAND PATTERNS
# =============================================================================

# CLI command decorator patterns (exact matches)
# Maps decorator name -> framework
_PYTHON_CLI_COMMAND_PATTERNS: dict[str, str] = {
    "click.command": "click",
    "click.group": "click",
    "app.command": "typer",
    "typer.command": "typer",
}

# CLI decorator suffixes that indicate a command (for patterns like main.command, web.group)
_PYTHON_CLI_COMMAND_SUFFIXES: set[str] = {".command", ".group"}

# CLI option decorator patterns
_PYTHON_CLI_OPTION_PATTERNS: set[str] = {
    "click.option",
    "click.argument",
    "typer.Option",
    "typer.Argument",
}

# CLI option suffixes (for patterns like main.option)
_PYTHON_CLI_OPTION_SUFFIXES: set[str] = {".option", ".argument"}


# =============================================================================
# DETECTION FUNCTION
# =============================================================================


def detect_python_cli_commands(
    decorators: list[DecoratorInfo],
    function_name: str,
) -> list[CliCommand]:
    """Detect CLI commands from Python decorator information.

    Args:
        decorators: List of decorator information extracted from a function.
        function_name: Name of the decorated function.

    Returns:
        List of detected CLI commands.
    """
    commands = []

    for dec in decorators:
        framework = None

        # Check exact pattern match first
        if dec.name in _PYTHON_CLI_COMMAND_PATTERNS:
            framework = _PYTHON_CLI_COMMAND_PATTERNS[dec.name]
        else:
            # Check for suffix patterns like main.command, web.group, etc.
            for suffix in _PYTHON_CLI_COMMAND_SUFFIXES:
                if dec.name.endswith(suffix):
                    framework = "click"  # Assume click for .command/.group patterns
                    break

        if framework:
            # Extract command name from decorator args if available
            # e.g., @web.command("serve") -> command_name = "serve"
            command_name = function_name
            if dec.args:
                # Extract first string argument as command name
                args_stripped = dec.args.strip().strip("()")
                if args_stripped.startswith(("'", '"')):
                    # Extract quoted string
                    quote_char = args_stripped[0]
                    end_quote = args_stripped.find(quote_char, 1)
                    if end_quote > 0:
                        command_name = args_stripped[1:end_quote]

            # Extract options from sibling decorators
            options = _extract_cli_options(decorators)

            commands.append(
                CliCommand(
                    name=command_name,
                    options=options,
                    framework=framework,
                )
            )

    return commands


def _extract_cli_options(decorators: list[DecoratorInfo]) -> list[dict]:
    """Extract CLI options from decorators.

    Args:
        decorators: List of decorator information.

    Returns:
        List of option dictionaries with name, type, and required fields.
    """
    options = []

    for dec in decorators:
        # Check for click/typer options - also handle suffix patterns
        is_option = (
            dec.name in _PYTHON_CLI_OPTION_PATTERNS
            or any(dec.name.endswith(suffix) for suffix in _PYTHON_CLI_OPTION_SUFFIXES)
        )

        if is_option:
            option_name = ""
            if dec.args:
                # Extract the first argument as the option name
                args_parts = dec.args.split(",")
                if args_parts:
                    option_name = args_parts[0].strip().strip("\"'")

            options.append(
                {
                    "name": option_name,
                    "type": None,
                    "required": "required=True" in (dec.full or ""),
                }
            )

    return options
