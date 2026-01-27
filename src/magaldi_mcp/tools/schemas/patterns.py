"""Pattern detection tool schemas."""

from mcp.types import Tool

PATTERN_TOOLS = [
    Tool(
        name="list_patterns",
        description="LIST PATTERNS: Show all detected design patterns in a repository. "
        "Returns pattern types (singleton, builder, factory, repository) with counts and example classes.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string", "description": "Repository scope (required)"},
                "repository": {"type": "string", "description": "Repository name (required)"},
                "username": {"type": "string", "description": "User branch"},
            },
            "required": ["scope", "repository"],
        },
    ),
    Tool(
        name="find_by_pattern",
        description="FIND BY PATTERN: Find all classes implementing a specific design pattern. "
        "Supports: singleton, builder, factory, repository.",
        inputSchema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "enum": ["singleton", "builder", "factory", "repository"],
                    "description": "Pattern type to search for",
                },
                "scope": {"type": "string", "description": "Repository scope (required)"},
                "repository": {"type": "string", "description": "Repository name (required)"},
                "username": {"type": "string", "description": "User branch"},
                "min_confidence": {
                    "type": "number",
                    "default": 0.6,
                    "description": "Minimum confidence score (0.0-1.0)",
                },
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["pattern", "scope", "repository"],
        },
    ),
]
