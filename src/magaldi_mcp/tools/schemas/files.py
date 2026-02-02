"""File tool schemas: find_files, get_file_structure."""

from mcp.types import Tool

FILE_TOOLS = [
    Tool(
        name="find_files",
        description="FIND FILES: Search for files by glob pattern. "
        "USE THIS instead of built-in Glob - searches indexed codebase. "
        "Discovers file structure (e.g., '**/*.py', 'src/**/*.ts'). "
        "Then use get_file_structure to see what's inside a file.",
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
    ),
    Tool(
        name="get_file_structure",
        description="FILE OVERVIEW: Get the structure of a file (classes, functions, methods, imports). "
        "Use to understand what's in a file without reading it.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Relative file path"},
                "scope": {"type": "string", "description": "Filter by scope"},
                "repository": {"type": "string", "description": "Filter by repo"},
            },
            "required": ["file_path"],
        },
    ),
]
