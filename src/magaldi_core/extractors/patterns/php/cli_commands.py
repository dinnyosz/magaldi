"""PHP CLI framework command patterns and detection.

Pattern definitions and detection functions for CLI commands
in PHP CLI frameworks:
- Symfony Console (#[AsCommand] attribute)
- Laravel Artisan ($signature property)
"""

from __future__ import annotations

import re

from tree_sitter import Node, Tree

from magaldi_core.extractors.base import get_node_text, walk_tree
from magaldi_core.extractors.types import CliCommand


# =============================================================================
# CONSTANTS
# =============================================================================

# Symfony Console command attribute names
_SYMFONY_COMMAND_ATTRIBUTES = {"AsCommand"}

# Laravel Artisan command property patterns
_LARAVEL_COMMAND_PROPERTIES = {"signature"}


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================


def extract_php_cli_commands(tree: Tree, lines: list[str]) -> list[CliCommand]:
    """Extract CLI commands from PHP code.

    Detects patterns like:
    - Symfony: #[AsCommand(name: 'app:command')]
    - Laravel: protected $signature = 'command:name {arg}';

    Args:
        tree: Parsed tree-sitter tree.
        lines: Source code lines.

    Returns:
        List of detected CLI commands.
    """
    commands: list[CliCommand] = []

    for node in walk_tree(tree.root_node):
        if node.type == "class_declaration":
            # Check for Symfony AsCommand attribute on class
            symfony_cmd = _extract_symfony_command(node)
            if symfony_cmd:
                commands.append(symfony_cmd)
                continue

            # Check for Laravel Artisan command (signature property)
            laravel_cmd = _extract_laravel_command(node)
            if laravel_cmd:
                commands.append(laravel_cmd)

    return commands


# =============================================================================
# SYMFONY CONSOLE
# =============================================================================


def _extract_symfony_command(class_node: Node) -> CliCommand | None:
    """Extract Symfony Console command from class with #[AsCommand] attribute.

    Args:
        class_node: A class_declaration node.

    Returns:
        CliCommand if this is a Symfony command class, None otherwise.
    """
    command_name = None
    description = None

    # Look for #[AsCommand] attribute on the class
    for child in class_node.children:
        if child.type == "attribute_list":
            for attr_group in child.children:
                if attr_group.type == "attribute_group":
                    name, desc = _extract_as_command_attribute(attr_group)
                    if name:
                        command_name = name
                        description = desc

    if not command_name:
        return None

    # Extract options from configure() method if present
    options = _extract_symfony_options(class_node)

    return CliCommand(
        name=command_name,
        options=options,
        framework="symfony-console",
    )


def _extract_as_command_attribute(attr_group: Node) -> tuple[str | None, str | None]:
    """Extract name and description from #[AsCommand] attribute.

    Args:
        attr_group: An attribute_group node.

    Returns:
        Tuple of (command_name, description).
    """
    for child in attr_group.children:
        if child.type == "attribute":
            attr_name = None
            args_node = None

            for attr_child in child.children:
                if attr_child.type == "name" or attr_child.type == "qualified_name":
                    attr_name = get_node_text(attr_child)
                    if "\\" in attr_name:
                        attr_name = attr_name.split("\\")[-1]
                elif attr_child.type == "arguments":
                    args_node = attr_child

            if attr_name in _SYMFONY_COMMAND_ATTRIBUTES and args_node:
                return _parse_as_command_args(args_node)

    return None, None


def _parse_as_command_args(args_node: Node) -> tuple[str | None, str | None]:
    """Parse arguments from AsCommand attribute.

    Handles: #[AsCommand(name: 'app:cmd', description: 'Desc')]

    Args:
        args_node: An arguments node.

    Returns:
        Tuple of (command_name, description).
    """
    command_name = None
    description = None

    for arg in args_node.children:
        if arg.type == "argument":
            arg_name = None
            arg_value = None

            for arg_child in arg.children:
                if arg_child.type == "name":
                    arg_name = get_node_text(arg_child)
                elif arg_child.type in ("string", "encapsed_string"):
                    arg_value = _get_string_content(arg_child)

            if arg_name == "name" and arg_value:
                command_name = arg_value
            elif arg_name == "description" and arg_value:
                description = arg_value
            elif arg_name is None and arg_value and command_name is None:
                # First positional argument is the name
                command_name = arg_value

    return command_name, description


def _extract_symfony_options(class_node: Node) -> list[dict]:
    """Extract command options from Symfony configure() method.

    Args:
        class_node: A class_declaration node.

    Returns:
        List of option dictionaries.
    """
    options = []

    for child in class_node.children:
        if child.type == "declaration_list":
            for decl_child in child.children:
                if decl_child.type == "method_declaration":
                    method_name = _get_method_name(decl_child)
                    if method_name == "configure":
                        options = _extract_options_from_configure(decl_child)

    return options


def _extract_options_from_configure(method_node: Node) -> list[dict]:
    """Extract options from configure() method body.

    Looks for ->addOption() and ->addArgument() calls.

    Args:
        method_node: A method_declaration node.

    Returns:
        List of option dictionaries.
    """
    options = []

    for node in walk_tree(method_node):
        if node.type == "member_call_expression":
            method_name = None
            for child in node.children:
                if child.type == "name":
                    method_name = get_node_text(child)

            if method_name in ("addOption", "addArgument"):
                option = _extract_option_from_call(node, method_name)
                if option:
                    options.append(option)

    return options


def _extract_option_from_call(call_node: Node, call_type: str) -> dict | None:
    """Extract option info from addOption/addArgument call.

    Args:
        call_node: A member_call_expression node.
        call_type: Either 'addOption' or 'addArgument'.

    Returns:
        Option dictionary or None.
    """
    for child in call_node.children:
        if child.type == "arguments":
            args = []
            for arg in child.children:
                if arg.type == "argument":
                    for arg_child in arg.children:
                        if arg_child.type in ("string", "encapsed_string"):
                            args.append(_get_string_content(arg_child))

            if args:
                return {
                    "name": args[0],
                    "type": "argument" if call_type == "addArgument" else "option",
                    "required": False,
                }

    return None


# =============================================================================
# LARAVEL ARTISAN
# =============================================================================


def _extract_laravel_command(class_node: Node) -> CliCommand | None:
    """Extract Laravel Artisan command from class with $signature property.

    Args:
        class_node: A class_declaration node.

    Returns:
        CliCommand if this is a Laravel command class, None otherwise.
    """
    signature = None
    description = None

    for child in class_node.children:
        if child.type == "declaration_list":
            for decl_child in child.children:
                if decl_child.type == "property_declaration":
                    prop_name, prop_value = _extract_property(decl_child)
                    if prop_name == "signature" and prop_value:
                        signature = prop_value
                    elif prop_name == "description" and prop_value:
                        description = prop_value

    if not signature:
        return None

    # Parse command name and options from signature
    command_name, options = _parse_laravel_signature(signature)

    return CliCommand(
        name=command_name,
        options=options,
        framework="laravel-artisan",
    )


def _extract_property(prop_node: Node) -> tuple[str | None, str | None]:
    """Extract property name and value from property_declaration.

    Args:
        prop_node: A property_declaration node.

    Returns:
        Tuple of (property_name, property_value).
    """
    prop_name = None
    prop_value = None

    for child in prop_node.children:
        if child.type == "property_element":
            for prop_child in child.children:
                if prop_child.type == "variable_name":
                    name_text = get_node_text(prop_child)
                    prop_name = name_text.lstrip("$")
                elif prop_child.type in ("string", "encapsed_string"):
                    prop_value = _get_string_content(prop_child)

    return prop_name, prop_value


def _parse_laravel_signature(signature: str) -> tuple[str, list[dict]]:
    """Parse Laravel Artisan command signature.

    Format: 'command:name {arg} {--option}'

    Args:
        signature: The signature string.

    Returns:
        Tuple of (command_name, options_list).
    """
    options = []

    # Extract command name (everything before first space or {)
    match = re.match(r"([^\s{]+)", signature)
    command_name = match.group(1) if match else signature

    # Extract arguments: {arg} or {arg?} or {arg=default}
    for arg_match in re.finditer(r"\{(\w+)(\?|=[^}]*)?\}", signature):
        arg_name = arg_match.group(1)
        if not arg_name.startswith("--"):
            options.append({
                "name": arg_name,
                "type": "argument",
                "required": arg_match.group(2) is None,
            })

    # Extract options: {--option} or {--option=} or {--option=default}
    for opt_match in re.finditer(r"\{--(\w+)(=[^}]*)?\}", signature):
        opt_name = opt_match.group(1)
        options.append({
            "name": f"--{opt_name}",
            "type": "option",
            "required": False,
        })

    return command_name, options


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_method_name(method_node: Node) -> str | None:
    """Get method name from method_declaration node."""
    for child in method_node.children:
        if child.type == "name":
            return get_node_text(child)
    return None


def _get_string_content(node: Node) -> str:
    """Extract string content from a string node."""
    for child in node.children:
        if child.type == "string_content":
            return get_node_text(child)
    # Fallback: strip quotes from the whole text
    text = get_node_text(node)
    return text.strip("'\"")
