"""Code metrics analysis functions.

This module provides functions for computing code quality metrics:
- Cyclomatic complexity
- Code size metrics (lines, parameters, characters)
- Docstring quality analysis
"""

from __future__ import annotations

import re
from typing import Any

from tree_sitter import Node

from magaldi_core.extractors.base import find_nodes, get_node_text

# =============================================================================
# CYCLOMATIC COMPLEXITY
# =============================================================================

# Decision point node types by language
DECISION_NODES: dict[str, set[str]] = {
    "python": {
        "if_statement",
        "elif_clause",
        "for_statement",
        "while_statement",
        "try_statement",
        "except_clause",
        "with_statement",
        "match_statement",
        "case_clause",
        "conditional_expression",  # ternary
        "list_comprehension",
        "dictionary_comprehension",
        "set_comprehension",
        "generator_expression",
    },
    "javascript": {
        "if_statement",
        "for_statement",
        "for_in_statement",
        "while_statement",
        "do_statement",
        "switch_statement",
        "case",  # switch case
        "try_statement",
        "catch_clause",
        "ternary_expression",
    },
    "typescript": {
        "if_statement",
        "for_statement",
        "for_in_statement",
        "while_statement",
        "do_statement",
        "switch_statement",
        "case",
        "try_statement",
        "catch_clause",
        "ternary_expression",
    },
    "php": {
        "if_statement",
        "for_statement",
        "foreach_statement",
        "while_statement",
        "do_statement",
        "switch_statement",
        "case_statement",
        "try_statement",
        "catch_clause",
    },
    "rust": {
        "if_expression",
        "if_let_expression",
        "for_expression",
        "while_expression",
        "while_let_expression",
        "loop_expression",
        "match_expression",
        "match_arm",
    },
}

# Boolean operator patterns (add to complexity for short-circuit evaluation)
BOOLEAN_OPERATORS: dict[str, set[str]] = {
    "python": {"and", "or"},
    "javascript": {"&&", "||"},
    "typescript": {"&&", "||"},
    "php": {"&&", "||", "and", "or"},
    "rust": {"&&", "||"},
}


def _count_boolean_operators(code: str, language: str) -> int:
    """Count boolean operators that add to cyclomatic complexity."""
    operators = BOOLEAN_OPERATORS.get(language, set())
    count = 0
    for op in operators:
        # Use word boundary for word operators, direct match for symbols
        if op.isalpha():
            count += len(re.findall(rf"\b{op}\b", code))
        else:
            count += code.count(op)
    return count


def _calculate_nesting_depth(node: Node, current_depth: int = 0) -> int:
    """Calculate maximum nesting depth of control structures."""
    max_depth = current_depth

    # Node types that increase nesting
    nesting_types = {
        "if_statement", "if_expression", "if_let_expression",
        "for_statement", "for_expression", "for_in_statement", "foreach_statement",
        "while_statement", "while_expression", "while_let_expression",
        "do_statement", "loop_expression",
        "try_statement", "with_statement",
        "match_statement", "match_expression", "switch_statement",
        "function_definition", "method_definition", "arrow_function",
        "lambda",
    }

    for child in node.children:
        child_depth = current_depth
        if child.type in nesting_types:
            child_depth = current_depth + 1

        nested_depth = _calculate_nesting_depth(child, child_depth)
        max_depth = max(max_depth, nested_depth)

    return max_depth


def compute_complexity(node: Node, language: str) -> dict[str, Any]:
    """Compute cyclomatic complexity metrics for a function/method node.

    Cyclomatic complexity = 1 + number of decision points + boolean operators

    Args:
        node: AST node of the function/method body.
        language: Programming language.

    Returns:
        Dict with cyclomatic, nesting_depth, and branch_count.
    """
    decision_types = DECISION_NODES.get(language, DECISION_NODES.get("python", set()))

    # Count decision points
    branch_count = 0
    for node_type in decision_types:
        branch_count += len(list(find_nodes(node, node_type)))

    # Count boolean operators
    code = get_node_text(node)
    bool_ops = _count_boolean_operators(code, language)

    # Cyclomatic complexity = 1 + branches + boolean operators
    cyclomatic = 1 + branch_count + bool_ops

    # Calculate nesting depth
    nesting_depth = _calculate_nesting_depth(node)

    return {
        "cyclomatic": cyclomatic,
        "nesting_depth": nesting_depth,
        "branch_count": branch_count,
    }


# =============================================================================
# CODE METRICS
# =============================================================================


def compute_code_metrics(raw_code: str, params: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Compute basic code size metrics.

    Args:
        raw_code: The raw source code of the element.
        params: List of parameter dicts (from CodeElement.parameters).

    Returns:
        Dict with line_count, param_count, char_count.
    """
    lines = raw_code.split("\n") if raw_code else []
    # Filter out empty lines for meaningful line count
    non_empty_lines = [line for line in lines if line.strip()]

    return {
        "line_count": len(non_empty_lines),
        "param_count": len(params) if params else 0,
        "char_count": len(raw_code) if raw_code else 0,
    }


# =============================================================================
# DOCSTRING QUALITY
# =============================================================================

# Common docstring param patterns
PARAM_PATTERNS = [
    re.compile(r":param\s+(\w+):", re.IGNORECASE),  # Sphinx
    re.compile(r"@param\s+(\w+)", re.IGNORECASE),  # JSDoc/PHPDoc
    re.compile(r"Args:\s*\n((?:\s+\w+.*\n)+)", re.IGNORECASE),  # Google style
    re.compile(r"Parameters\s*\n\s*-+\s*\n((?:\s+\w+.*\n)+)", re.IGNORECASE),  # NumPy
]

RETURN_PATTERNS = [
    re.compile(r":returns?:", re.IGNORECASE),  # Sphinx
    re.compile(r"@returns?\s", re.IGNORECASE),  # JSDoc
    re.compile(r"Returns:\s*\n", re.IGNORECASE),  # Google style
    re.compile(r"Returns\s*\n\s*-+", re.IGNORECASE),  # NumPy
]


def _has_param_docs(docstring: str) -> bool:
    """Check if docstring documents parameters."""
    return any(pattern.search(docstring) for pattern in PARAM_PATTERNS)


def _has_return_docs(docstring: str) -> bool:
    """Check if docstring documents return value."""
    return any(pattern.search(docstring) for pattern in RETURN_PATTERNS)


def analyze_docstring(
    docstring: str | None,
    params: list[dict[str, Any]] | None,
    return_type: str | None,
) -> dict[str, Any]:
    """Analyze docstring quality and completeness.

    Args:
        docstring: The docstring text (or None).
        params: List of parameter dicts.
        return_type: Return type annotation (or None).

    Returns:
        Dict with has_docstring, has_params, has_return, coverage.
    """
    has_docstring = bool(docstring and docstring.strip())

    if not has_docstring:
        return {
            "has_docstring": False,
            "has_params": False,
            "has_return": False,
            "coverage": 0.0,
        }

    # Type narrowing: at this point docstring is guaranteed to be non-empty str
    assert docstring is not None
    has_params = _has_param_docs(docstring)
    has_return = _has_return_docs(docstring)

    # Calculate coverage score
    # - Has docstring: base 0.5
    # - Has param docs (if params exist): +0.25
    # - Has return docs (if return type exists): +0.25
    coverage = 0.5

    param_count = len(params) if params else 0
    needs_param_docs = param_count > 0
    needs_return_docs = return_type is not None and return_type != "None"

    if needs_param_docs:
        if has_params:
            coverage += 0.25
    else:
        coverage += 0.25  # No params to document

    if needs_return_docs:
        if has_return:
            coverage += 0.25
    else:
        coverage += 0.25  # No return to document

    return {
        "has_docstring": True,
        "has_params": has_params,
        "has_return": has_return,
        "coverage": round(coverage, 2),
    }
