"""Parser Lab tool schemas: self-improvement tools for Magaldi parser.

These tools are restricted to magaldi/magaldi scope only and allow testing,
analyzing, and improving the parser's extraction capabilities.
"""

from mcp.types import Tool

PARSER_LAB_TOOLS = [
    Tool(
        name="parser_lab_analyze",
        description=(
            "PARSER LAB: Analyze parsing of code from any source. "
            "Parse a file, code snippet, or fetch examples via Context7 query. "
            "Returns extracted elements, AST summary, and gap analysis. "
            "Use debug=true for full AST tree. "
            "RESTRICTED: Only available when working on magaldi/magaldi."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to file to parse (can be external to magaldi)",
                },
                "code": {
                    "type": "string",
                    "description": "Inline code snippet to parse",
                },
                "context7_query": {
                    "type": "string",
                    "description": "Query for Context7 to fetch example code (e.g., 'Django class-based views')",
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript", "typescript", "tsx", "php", "rust"],
                    "description": "Language of the code (auto-detected if file_path provided)",
                },
                "debug": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include full AST tree in output (verbose)",
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="parser_lab_create_test",
        description=(
            "PARSER LAB: Create a test case for expected parsing behavior. "
            "Generates a pytest test file in tests/extractors/ that validates "
            "the parser extracts the expected elements from the given code. "
            "Use TDD: create the test BEFORE fixing the parser. "
            "RESTRICTED: Only available when working on magaldi/magaldi."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Test name (e.g., 'drf_api_view_decorator'). Used for file and class naming.",
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript", "typescript", "tsx", "php", "rust"],
                    "description": "Language of the code to test",
                },
                "code": {
                    "type": "string",
                    "description": "Code snippet to test parsing on",
                },
                "expected": {
                    "type": "object",
                    "description": "Expected parsing results",
                    "properties": {
                        "elements": {
                            "type": "array",
                            "description": "Expected elements to be extracted",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "description": "Element type (function, class, method, etc.)"},
                                    "name": {"type": "string", "description": "Element name"},
                                    "decorators": {"type": "array", "items": {"type": "string"}},
                                    "is_async": {"type": "boolean"},
                                    "visibility": {"type": "string"},
                                    "has_docstring": {"type": "boolean"},
                                },
                                "required": ["type", "name"],
                            },
                        },
                        "element_count": {
                            "type": "integer",
                            "description": "Expected total element count (optional)",
                        },
                        "has_routes": {
                            "type": "boolean",
                            "description": "Whether HTTP routes should be detected",
                        },
                        "routes": {
                            "type": "array",
                            "description": "Expected HTTP routes",
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
                            "description": "Expected calls within a function",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "receiver": {"type": "string"},
                                    "in_element": {"type": "string", "description": "Name of element containing the call"},
                                },
                            },
                        },
                    },
                },
            },
            "required": ["name", "language", "code", "expected"],
        },
    ),
    Tool(
        name="parser_lab_run_tests",
        description=(
            "PARSER LAB: Run parser tests. "
            "Executes pytest on parser/extractor tests and returns structured results. "
            "Use filter to run specific tests (e.g., 'test_python_drf'). "
            "RESTRICTED: Only available when working on magaldi/magaldi."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "filter": {
                    "type": "string",
                    "description": "Pytest filter expression (e.g., 'test_drf' or 'tests/extractors/test_python_drf.py')",
                },
                "verbose": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include full test output",
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="parser_lab_suggest_fix",
        description=(
            "PARSER LAB: Suggest extractor modifications to fix a parsing gap. "
            "Given a description of what's missing and the language, suggests "
            "which files to modify and provides code hints. "
            "RESTRICTED: Only available when working on magaldi/magaldi."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "gap_description": {
                    "type": "string",
                    "description": "Description of what's not being extracted correctly",
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript", "typescript", "tsx", "php", "rust"],
                    "description": "Language of the code with the gap",
                },
                "failing_test": {
                    "type": "string",
                    "description": "Optional: path to failing test for additional context",
                },
            },
            "required": ["gap_description", "language"],
        },
    ),
]
