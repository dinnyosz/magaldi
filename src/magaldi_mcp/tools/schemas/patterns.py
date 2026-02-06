"""Pattern detection tool schemas."""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import READONLY_ANNOTATIONS

PATTERN_TOOLS = [
    Tool(
        name="list_patterns",
        description="Show all detected design patterns in a repository (singleton, builder, factory, etc).",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
                "repository": {"type": "string"},
                "username": {"type": "string", "description": "User branch"},
            },
            "required": ["scope", "repository"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="find_by_pattern",
        description="Find all classes implementing a specific design pattern.",
        inputSchema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "enum": ["singleton", "builder", "factory", "repository"],
                    "description": "Pattern type to search for",
                },
                "scope": {"type": "string"},
                "repository": {"type": "string"},
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
        annotations=READONLY_ANNOTATIONS,
    ),
]
