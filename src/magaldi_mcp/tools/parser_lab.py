"""Parser Lab tools for Magaldi self-improvement.

These tools allow testing and improving Magaldi's parsing capabilities.
RESTRICTED: Only available when working on magaldi/magaldi scope.
"""

from __future__ import annotations

import re
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from magaldi_core.extractors.types import ExtractedElement
from magaldi_core.tree_sitter_manager import get_manager

# =============================================================================
# SCOPE RESTRICTION
# =============================================================================


def _check_magaldi_scope() -> None:
    """Verify we're operating on magaldi/magaldi.

    For now, we check if we're in the magaldi project directory.
    In production, this would check the MCP request context.
    """
    # TODO: In MCP context, check scope/repository from request
    # For now, allow all (trust the tool description warning)
    pass


# =============================================================================
# LANGUAGE DETECTION
# =============================================================================

EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "javascript",
    ".php": "php",
    ".rs": "rust",
}


def _detect_language(file_path: str | None, code: str | None) -> str | None:
    """Detect language from file path or code content."""
    if file_path:
        ext = Path(file_path).suffix.lower()
        return EXTENSION_TO_LANGUAGE.get(ext)

    if code:
        # Simple heuristics for language detection from code
        if "def " in code and "import " in code:
            return "python"
        if "function " in code or "const " in code or "=>" in code:
            return "javascript"
        if "fn " in code and "let " in code:
            return "rust"
        if "<?php" in code or "$this->" in code:
            return "php"

    return None


# =============================================================================
# GAP ANALYSIS
# =============================================================================


@dataclass
class GapInfo:
    """Information about a parsing gap (AST node not captured)."""

    node_type: str
    text: str
    line: int
    suggestion: str = ""


@dataclass
class ASTSummary:
    """Summary of AST node types found."""

    node_types: dict[str, int] = field(default_factory=dict)
    total_nodes: int = 0


def _count_ast_nodes(node: Any, summary: ASTSummary) -> None:
    """Recursively count AST node types."""
    summary.total_nodes += 1
    node_type = node.type
    summary.node_types[node_type] = summary.node_types.get(node_type, 0) + 1

    for child in node.children:
        _count_ast_nodes(child, summary)


def _find_uncaptured_nodes(
    node: Any,
    captured_ranges: set[tuple[int, int]],
    language: str,
) -> list[GapInfo]:
    """Find AST nodes that weren't captured by extraction.

    Args:
        node: Root AST node.
        captured_ranges: Set of (start_byte, end_byte) ranges that were captured.
        language: Language name for suggestions.

    Returns:
        List of GapInfo for potentially uncaptured nodes.
    """
    gaps = []

    # Node types we care about (potential extraction targets)
    interesting_types = {
        "python": {
            "function_definition",
            "class_definition",
            "decorated_definition",
            "import_statement",
            "import_from_statement",
            "call",
            "assignment",
        },
        "javascript": {
            "function_declaration",
            "class_declaration",
            "arrow_function",
            "call_expression",
            "import_statement",
            "variable_declaration",
        },
        "typescript": {
            "function_declaration",
            "class_declaration",
            "arrow_function",
            "call_expression",
            "import_statement",
            "variable_declaration",
            "type_alias_declaration",
            "interface_declaration",
        },
        "tsx": {
            "function_declaration",
            "class_declaration",
            "arrow_function",
            "call_expression",
            "import_statement",
            "variable_declaration",
            "type_alias_declaration",
            "interface_declaration",
            "jsx_element",
            "jsx_self_closing_element",
        },
        "php": {
            "function_definition",
            "class_declaration",
            "method_declaration",
            "function_call_expression",
        },
        "rust": {
            "function_item",
            "struct_item",
            "impl_item",
            "enum_item",
            "call_expression",
            "use_declaration",
        },
    }

    target_types = interesting_types.get(language, set())

    def check_node(n: Any) -> None:
        if n.type in target_types:
            # Check if this node's range overlaps with any captured range
            is_captured = any(
                start <= n.start_byte < end or start < n.end_byte <= end
                for start, end in captured_ranges
            )
            if not is_captured:
                text = n.text.decode("utf-8")[:100]  # First 100 chars
                gaps.append(
                    GapInfo(
                        node_type=n.type,
                        text=text,
                        line=n.start_point[0] + 1,
                        suggestion=f"Add pattern for {n.type} in queries/{language}/elements.scm",
                    )
                )

        for child in n.children:
            check_node(child)

    check_node(node)
    return gaps


# =============================================================================
# PARSER LAB ANALYZE
# =============================================================================


def parser_lab_analyze(
    file_path: str | None = None,
    code: str | None = None,
    context7_query: str | None = None,
    language: str | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Analyze parsing of code from any source.

    Args:
        file_path: Path to file to parse (can be external).
        code: Inline code snippet to parse.
        context7_query: Query for Context7 (not implemented yet).
        language: Language override (auto-detected if not provided).
        debug: Include full AST tree in output.

    Returns:
        Dict with elements, ast_summary, gaps, and optionally ast_debug.
    """
    _check_magaldi_scope()

    # Get source code
    if file_path:
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}
        code = path.read_text(encoding="utf-8")
    elif context7_query:
        # Context7 MCP is available but MCP tools can't call other MCP tools directly.
        # Return a delegation response for the AI agent to handle.
        return {
            "delegation": "context7",
            "instructions": (
                "To use Context7 for code examples, follow these steps:\n"
                "1. Call mcp__plugin_context7_context7__resolve-library-id with:\n"
                f"   - libraryName: extract the library/framework name from '{context7_query}'\n"
                f"   - query: '{context7_query}'\n"
                "2. Call mcp__plugin_context7_context7__query-docs with:\n"
                "   - libraryId: use the ID from step 1\n"
                f"   - query: '{context7_query}'\n"
                "3. Extract code snippets from the response\n"
                "4. Call parser_lab_analyze again with:\n"
                "   - code: the extracted code snippet\n"
                f"   - language: '{language or 'auto-detect'}'"
            ),
            "context7_query": context7_query,
            "language": language,
        }
    elif not code:
        return {"error": "Must provide file_path, code, or context7_query"}

    # Detect language
    if not language:
        language = _detect_language(file_path, code)
        if not language:
            return {"error": "Could not detect language. Please specify explicitly."}

    # Get manager and parse
    manager = get_manager()

    if not manager.is_language_supported(language):
        return {"error": f"Language not supported: {language}"}

    # Parse with tree-sitter
    tree = manager.parse(code.encode("utf-8"), language)

    # Extract using existing extractor
    extractor = manager.get_extractor(language)
    lines = code.split("\n")
    elements: list[ExtractedElement] = []
    if extractor:
        elements = extractor.extract_elements(tree, lines)

    # Build AST summary
    ast_summary = ASTSummary()
    _count_ast_nodes(tree.root_node, ast_summary)

    # Run SCM queries for comparison
    query_results: dict[str, int] = {}
    try:
        for query_name in manager.list_queries(language):
            result = manager.run_query(code, language, query_name)
            query_results[query_name] = len(result)
    except Exception as e:
        query_results["error"] = str(e)

    # Find gaps (nodes not captured by extractor)
    captured_ranges: set[tuple[int, int]] = set()
    for elem in elements:
        if elem.node:
            captured_ranges.add((elem.node.start_byte, elem.node.end_byte))

    gaps = _find_uncaptured_nodes(tree.root_node, captured_ranges, language)

    # Extract framework-specific patterns
    framework_patterns: dict[str, Any] = {}
    if language == "php":
        from magaldi_core.extractors.patterns.slim import extract_slim_routes, extract_slim_route_groups
        slim_routes = extract_slim_routes(tree, lines)
        slim_groups = extract_slim_route_groups(tree, lines)
        if slim_routes or slim_groups:
            framework_patterns["slim"] = {
                "routes": [
                    {"method": r.method, "path": r.path, "path_params": r.path_params}
                    for r in slim_routes
                ],
                "route_groups": [
                    {
                        "prefix": g.prefix,
                        "routes": [
                            {"method": r.method, "path": r.path, "path_params": r.path_params}
                            for r in g.routes
                        ],
                        "line_start": g.line_start,
                        "line_end": g.line_end,
                    }
                    for g in slim_groups
                ],
            }

    # Build response
    result: dict[str, Any] = {
        "language": language,
        "source_lines": len(lines),
        "elements": [
            {
                "type": e.element_type,
                "name": e.name,
                "line_start": e.line_start,
                "line_end": e.line_end,
                "decorators": e.decorators or [],
                "decorator_details": [
                    {"name": d.name, "args": d.args, "full": d.full}
                    for d in (e.decorator_details or [])
                ],
                "is_async": getattr(e, "is_async", False),
            }
            for e in elements
        ],
        "element_count": len(elements),
        "ast_summary": {
            "node_types": dict(
                sorted(ast_summary.node_types.items(), key=lambda x: -x[1])[:20]
            ),
            "total_nodes": ast_summary.total_nodes,
        },
        "query_results": query_results,
        "gaps": [
            {
                "node_type": g.node_type,
                "text": g.text[:50] + ("..." if len(g.text) > 50 else ""),
                "line": g.line,
                "suggestion": g.suggestion,
            }
            for g in gaps[:10]  # Limit to 10 gaps
        ],
        "gap_count": len(gaps),
    }

    # Add framework patterns if any were detected
    if framework_patterns:
        result["framework_patterns"] = framework_patterns

    if debug:
        # Include AST tree representation
        def node_to_dict(n: Any, depth: int = 0) -> dict[str, Any]:
            if depth > 5:  # Limit depth
                return {"type": n.type, "truncated": True}
            return {
                "type": n.type,
                "text": n.text.decode("utf-8")[:50] if n.child_count == 0 else None,
                "start": list(n.start_point),
                "end": list(n.end_point),
                "children": [node_to_dict(c, depth + 1) for c in n.children]
                if depth < 3
                else f"[{n.child_count} children]",
            }

        result["ast_debug"] = node_to_dict(tree.root_node)

    return result


# =============================================================================
# PARSER LAB CREATE TEST
# =============================================================================


def parser_lab_create_test(
    name: str,
    language: str,
    code: str,
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Create a test case for expected parsing behavior.

    Args:
        name: Test name (e.g., 'drf_api_view_decorator').
        language: Language of the code.
        code: Code snippet to test.
        expected: Expected parsing results.

    Returns:
        Dict with test_file path and preview of generated test.
    """
    _check_magaldi_scope()

    # Sanitize name for file/class naming
    safe_name = re.sub(r"[^a-z0-9_]", "_", name.lower())
    class_name = "".join(word.capitalize() for word in safe_name.split("_"))

    # Generate test code
    test_code = f'''"""Auto-generated parser test for {name}."""

import pytest
from pathlib import Path

from magaldi_core.code_parser import {language.capitalize()}Parser
from magaldi_core.change_detection import FileInfo


class Test{class_name}:
    """Tests for {name} extraction."""

    def test_{safe_name}(self):
        code = {repr(code)}

        parser = {language.capitalize()}Parser()
        file_info = FileInfo(
            relative_path="test.{_get_extension(language)}",
            absolute_path=Path("/fake/test.{_get_extension(language)}"),
            language="{language}",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
'''

    # Add assertions based on expected
    if "elements" in expected:
        for exp_elem in expected["elements"]:
            elem_type = exp_elem.get("type", "function")
            elem_name = exp_elem.get("name", "unknown")
            test_code += f'''
        # Check for {elem_name}
        {safe_name}_elem = next(
            (e for e in elements if e.name == "{elem_name}" and e.element_type == "{elem_type}"),
            None
        )
        assert {safe_name}_elem is not None, f"Expected {elem_type} '{elem_name}' not found"
'''
            if "decorators" in exp_elem:
                for deco in exp_elem["decorators"]:
                    test_code += f'        assert "{deco}" in {safe_name}_elem.decorators, "Missing decorator {deco}"\n'

            if exp_elem.get("is_async"):
                test_code += f"        assert {safe_name}_elem.is_async is True\n"

    if "element_count" in expected:
        test_code += f'''
        # Check element count
        assert len(elements) == {expected["element_count"]}, f"Expected {expected["element_count"]} elements, got {{len(elements)}}"
'''

    if expected.get("has_routes"):
        test_code += '''
        # Check for HTTP routes
        has_routes = any(e.http_routes for e in elements if hasattr(e, "http_routes"))
        assert has_routes, "Expected HTTP routes to be detected"
'''

    # Write test file
    test_dir = Path("tests/extractors")
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / f"test_{language}_{safe_name}.py"
    test_file.write_text(test_code, encoding="utf-8")

    return {
        "test_file": str(test_file),
        "test_function": f"test_{safe_name}",
        "test_class": f"Test{class_name}",
        "preview": test_code[:500] + ("..." if len(test_code) > 500 else ""),
        "status": "created",
    }


def _get_extension(language: str) -> str:
    """Get file extension for a language."""
    return {
        "python": "py",
        "javascript": "js",
        "typescript": "ts",
        "tsx": "tsx",
        "php": "php",
        "rust": "rs",
    }.get(language, "txt")


# =============================================================================
# PARSER LAB RUN TESTS
# =============================================================================


def parser_lab_run_tests(
    filter: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run parser tests.

    Args:
        filter: Pytest filter expression.
        verbose: Include full test output.

    Returns:
        Dict with test results.
    """
    _check_magaldi_scope()

    # Build pytest command
    cmd = [".venv/bin/pytest", "tests/"]

    if filter:
        cmd.extend(["-k", filter])

    cmd.extend(["--tb=short", "-q"])

    if verbose:
        cmd.append("-v")

    # Run pytest
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=Path(__file__).parent.parent.parent.parent,  # Project root
        )

        # Parse output
        output = result.stdout + result.stderr

        # Extract counts
        passed = 0
        failed = 0
        errors = 0

        # Look for summary line like "5 passed, 2 failed in 1.23s"
        summary_match = re.search(
            r"(\d+) passed(?:.*?(\d+) failed)?(?:.*?(\d+) error)?",
            output,
        )
        if summary_match:
            passed = int(summary_match.group(1) or 0)
            failed = int(summary_match.group(2) or 0) if summary_match.group(2) else 0
            errors = int(summary_match.group(3) or 0) if summary_match.group(3) else 0

        # Extract failure details
        failures = []
        failure_sections = re.findall(
            r"FAILED (.*?) - (.*?)(?=\n(?:FAILED|=|$))",
            output,
            re.DOTALL,
        )
        for test_name, message in failure_sections[:5]:  # Limit to 5
            failures.append(
                {
                    "test": test_name.strip(),
                    "message": message.strip()[:200],
                }
            )

        return {
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "failures": failures,
            "output": output if verbose else output[:500],
            "return_code": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {"error": "Test execution timed out after 120 seconds"}
    except Exception as e:
        return {"error": f"Failed to run tests: {e}"}


# =============================================================================
# PARSER LAB SUGGEST FIX
# =============================================================================


def parser_lab_suggest_fix(
    gap_description: str,
    language: str,
    failing_test: str | None = None,  # noqa: ARG001 - reserved for future use
) -> dict[str, Any]:
    """Suggest extractor modifications to fix a parsing gap.

    Args:
        gap_description: Description of what's not being extracted.
        language: Language of the code.
        failing_test: Optional path to failing test for context.

    Returns:
        Dict with suggested file modifications.
    """
    _check_magaldi_scope()

    suggestions: list[dict[str, Any]] = []

    # Determine which files to modify based on the gap
    gap_lower = gap_description.lower()

    # SCM query suggestions
    query_dir = Path(f"src/magaldi_core/queries/{language}")

    if any(word in gap_lower for word in ["decorator", "route", "api", "endpoint"]):
        suggestions.append(
            {
                "file": str(query_dir / "elements.scm"),
                "section": "DECORATORS",
                "suggestion": textwrap.dedent(f"""
                    Add a pattern for the decorator. Example:

                    ; {gap_description}
                    (decorated_definition
                      (decorator
                        (call
                          function: (identifier) @decorator.name
                          arguments: (argument_list) @decorator.args))
                      definition: (function_definition) @decorated.function) @your_pattern
                """).strip(),
                "explanation": "Add a new decorator pattern to capture the specific decorator syntax.",
            }
        )

    if any(word in gap_lower for word in ["call", "method", "function"]):
        suggestions.append(
            {
                "file": str(query_dir / "calls.scm"),
                "section": "SPECIAL CALLS",
                "suggestion": textwrap.dedent(f"""
                    Add a pattern for the call type. Example:

                    ; {gap_description}
                    (call
                      function: (attribute
                        object: (_) @call.receiver
                        attribute: (identifier) @call.method)
                      arguments: (argument_list) @call.args) @your_call_pattern
                """).strip(),
                "explanation": "Add a new call pattern to capture the specific call syntax.",
            }
        )

    if any(word in gap_lower for word in ["import", "require", "use"]):
        suggestions.append(
            {
                "file": str(query_dir / "imports.scm"),
                "section": "IMPORTS",
                "suggestion": textwrap.dedent(f"""
                    Add a pattern for the import type. Example:

                    ; {gap_description}
                    (import_from_statement
                      module_name: (dotted_name) @import.module
                      name: (_) @import.name) @your_import_pattern
                """).strip(),
                "explanation": "Add a new import pattern to capture the specific import syntax.",
            }
        )

    # Extractor file suggestions
    extractor_file = f"src/magaldi_core/extractors/{language}.py"
    if Path(extractor_file).exists():
        suggestions.append(
            {
                "file": extractor_file,
                "section": "Post-processing",
                "suggestion": "After adding the SCM query pattern, you may need to add post-processing logic in the extractor to convert the query matches to ExtractedElement objects.",
                "explanation": "The extractor handles complex logic like building parent-child relationships.",
            }
        )

    # Analysis file suggestions (for routes, CLI commands, etc.)
    if any(word in gap_lower for word in ["route", "endpoint", "api", "http"]):
        suggestions.append(
            {
                "file": "src/magaldi_core/analysis/api_detection.py",
                "section": "HTTP route detection",
                "suggestion": f"Add detection pattern for the framework/decorator mentioned in: {gap_description}",
                "explanation": "API detection handles framework-specific patterns like Flask @route, FastAPI @app.get, etc.",
            }
        )

    return {
        "gap_description": gap_description,
        "language": language,
        "suggestions": suggestions,
        "workflow": [
            "1. Create a test case with parser_lab_create_test",
            "2. Run parser_lab_run_tests to verify it fails",
            "3. Add the suggested SCM query pattern",
            "4. Update extractor if needed for post-processing",
            "5. Run parser_lab_run_tests to verify it passes",
        ],
    }
