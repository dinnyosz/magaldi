"""Parser Lab tool schemas: self-improvement tools for Magaldi parser.

These tools are restricted to magaldi/magaldi scope only and allow testing,
analyzing, and improving the parser's extraction capabilities.
"""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import READONLY_ANNOTATIONS, WRITE_ANNOTATIONS

PARSER_LAB_TOOLS = [
    Tool(
        name="parser_lab_analyze",
        description="Analyze code parsing: extracted elements, AST summary, gap analysis.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "File to parse"},
                "code": {"type": "string", "description": "Code snippet to parse"},
                "context7_query": {
                    "type": "string",
                    "description": "Context7 query for example code",
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript", "typescript", "tsx", "php", "rust", "java", "bash"],
                },
                "debug": {"type": "boolean", "default": False, "description": "Include full AST"},
            },
            "required": [],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="parser_lab_create_test",
        description="Create pytest test for expected parsing behavior (TDD).",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Test name (e.g., 'drf_api_view_decorator')"},
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript", "typescript", "tsx", "php", "rust", "java", "bash"],
                },
                "code": {"type": "string", "description": "Code to test parsing on"},
                "expected": {
                    "type": "object",
                    "properties": {
                        "elements": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "name": {"type": "string"},
                                    "decorators": {"type": "array", "items": {"type": "string"}},
                                    "is_async": {"type": "boolean"},
                                    "visibility": {"type": "string"},
                                    "has_docstring": {"type": "boolean"},
                                },
                                "required": ["type", "name"],
                            },
                        },
                        "element_count": {"type": "integer"},
                        "has_routes": {"type": "boolean"},
                        "routes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "method": {"type": "string"},
                                    "path": {"type": "string"},
                                    "decorator": {"type": "string"},
                                },
                            },
                        },
                        "calls": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "receiver": {"type": "string"},
                                    "in_element": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
            "required": ["name", "language", "code", "expected"],
        },
        annotations=WRITE_ANNOTATIONS,
    ),
    Tool(
        name="parser_lab_run_tests",
        description="Run parser tests and return results.",
        inputSchema={
            "type": "object",
            "properties": {
                "filter": {"type": "string", "description": "Pytest filter expression"},
                "verbose": {"type": "boolean", "default": False},
            },
            "required": [],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="parser_lab_suggest_fix",
        description="Suggest extractor fix for a parsing gap.",
        inputSchema={
            "type": "object",
            "properties": {
                "gap_description": {"type": "string", "description": "What\'s not extracted correctly"},
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript", "typescript", "tsx", "php", "rust", "java", "bash"],
                },
                "failing_test": {"type": "string", "description": "Path to failing test"},
            },
            "required": ["gap_description", "language"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
]
