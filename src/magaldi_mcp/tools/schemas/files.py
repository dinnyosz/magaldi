"""File tool schemas: find_files, get_file_structure."""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import READONLY_ANNOTATIONS

FILE_TOOLS = [
    Tool(
        name="find_files",
        description="Find files by glob pattern in the indexed codebase.",
        inputSchema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern"},
                "limit": {"type": "integer", "default": 30},
            },
            "required": ["pattern"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="get_file_structure",
        description="Outline a file's classes, functions, methods, and imports.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Relative file path"},
            },
            "required": ["file_path"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
]
