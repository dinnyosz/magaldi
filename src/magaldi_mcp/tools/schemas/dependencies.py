"""Dependency tool schemas: find_dependencies, find_dependents, dependency_graph."""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import READONLY_ANNOTATIONS

DEPENDENCY_TOOLS = [
    Tool(
        name="find_dependencies",
        description="Get imports for a file (internal and external). "
        "Use find_dependents for the reverse.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Relative file path (e.g., 'src/utils.py')"},
                "hash_id": {"type": "string", "description": "Or provide file element ID directly (hash_id)"},
                "scope": {"type": "string", "description": "Repository scope (required if using file_path)"},
                "repository": {"type": "string", "description": "Repository name (required if using file_path)"},
                "username": {"type": "string", "description": "User branch"},
            },
            "required": [],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="find_dependents",
        description="Find files that import a module. "
        "Useful for impact analysis before changing a module.",
        inputSchema={
            "type": "object",
            "properties": {
                "module": {"type": "string", "description": "Module name (e.g., 'utils', 'shared.config', './utils')"},
                "scope": {"type": "string", "description": "Repository scope (required)"},
                "repository": {"type": "string", "description": "Repository name (required)"},
                "username": {"type": "string", "description": "User branch"},
                "limit": {"type": "integer", "default": 50, "description": "Maximum files to return"},
            },
            "required": ["module", "scope", "repository"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="dependency_graph",
        description="Build a module-level dependency graph. "
        "Useful for understanding architecture and detecting circular dependencies.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string", "description": "Repository scope (required)"},
                "repository": {"type": "string", "description": "Repository name (required)"},
                "username": {"type": "string", "description": "User branch"},
                "internal_only": {"type": "boolean", "default": True, "description": "Only include internal imports"},
            },
            "required": ["scope", "repository"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
]
