"""File tool schemas: find_files, get_file_structure."""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import READONLY_ANNOTATIONS

FILE_TOOLS = [
    Tool(
        name="find_files",
        description="Search for files by glob pattern in the indexed codebase.",
        inputSchema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern"},
                "scope": {"type": "string", "description": "Filter by scope"},
                "repository": {"type": "string", "description": "Filter by repo"},
                "limit": {"type": "integer", "default": 50},
            },
            "required": ["pattern"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="get_file_structure",
        description="Get the structure of a file (classes, functions, methods, imports) without reading it.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Relative file path"},
                "scope": {"type": "string", "description": "Filter by scope"},
                "repository": {"type": "string", "description": "Filter by repo"},
            },
            "required": ["file_path"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
]
