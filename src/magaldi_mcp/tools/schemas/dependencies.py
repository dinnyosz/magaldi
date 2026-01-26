"""Dependency tool schemas: find_dependencies, find_dependents, dependency_graph."""

from mcp.types import Tool

DEPENDENCY_TOOLS = [
    Tool(
        name="find_dependencies",
        description="FIND DEPENDENCIES: Get imports for a file. "
        "Shows what a file depends on - both internal (from within the repo) "
        "and external (third-party packages) imports.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Relative file path (e.g., 'src/utils.py')"},
                "element_id": {"type": "string", "description": "Or provide file element ID directly"},
                "scope": {"type": "string", "description": "Repository scope (required if using file_path)"},
                "repository": {"type": "string", "description": "Repository name (required if using file_path)"},
                "username": {"type": "string", "description": "User branch"},
            },
            "required": [],
        },
    ),
    Tool(
        name="find_dependents",
        description="FIND DEPENDENTS: Find files that import a module. "
        "Shows what depends on a given module - useful for impact analysis "
        "before making changes to a module.",
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
    ),
    Tool(
        name="dependency_graph",
        description="DEPENDENCY GRAPH: Build a module-level dependency graph. "
        "Creates a directed graph of module dependencies within the repository. "
        "Useful for understanding code architecture and detecting circular dependencies.",
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
    ),
]
