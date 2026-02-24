"""Dependency tool schemas: find_dependencies, find_dependents, dependency_graph."""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import READONLY_ANNOTATIONS

DEPENDENCY_TOOLS = [
    Tool(
        name="find_dependencies",
        description="Get imports for a file. Use find_dependents for the reverse.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Relative file path"},
                "hash_id": {"type": "string", "description": "Or file element ID"},
                "username": {"type": "string"},
            },
            "required": [],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="find_dependents",
        description="Find files that import a module. For impact analysis.",
        inputSchema={
            "type": "object",
            "properties": {
                "module": {"type": "string", "description": "Module name (e.g., 'shared.config')"},
                "username": {"type": "string"},
                "limit": {"type": "integer", "default": 30},
            },
            "required": ["module"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="dependency_graph",
        description="Build module-level dependency graph. Detects circular dependencies.",
        inputSchema={
            "type": "object",
            "properties": {
                "username": {"type": "string"},
                "internal_only": {"type": "boolean", "default": True},
            },
            "required": [],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
]
