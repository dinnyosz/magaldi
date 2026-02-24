"""Pattern detection tool schemas."""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import READONLY_ANNOTATIONS

PATTERN_TOOLS = [
    Tool(
        name="list_patterns",
        description="List detected design patterns in a repo (singleton, factory, etc).",
        inputSchema={
            "type": "object",
            "properties": {
                "username": {"type": "string"},
            },
            "required": [],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="find_by_pattern",
        description="Find classes implementing a specific design pattern.",
        inputSchema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "enum": ["singleton", "builder", "factory", "repository"],
                },
                "username": {"type": "string"},
                "min_confidence": {"type": "number", "default": 0.6},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["pattern"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
]
