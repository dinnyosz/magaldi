"""Hierarchy tool schemas: CLI command tree, HTTP route tree."""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import READONLY_ANNOTATIONS

HIERARCHY_TOOLS = [
    Tool(
        name="get_command_tree",
        description="Get hierarchical structure of CLI commands (Click, Typer, etc.) "
        "with full invocation paths.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string", "description": "Repository scope (required)"},
                "repository": {"type": "string", "description": "Repository name (required)"},
                "username": {"type": "string", "description": "User branch (defaults to 'main')"},
            },
            "required": ["scope", "repository"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="get_route_tree",
        description="Get hierarchical structure of HTTP routes (FastAPI, Flask, etc.) "
        "grouped by router module with full URL paths.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string", "description": "Repository scope (required)"},
                "repository": {"type": "string", "description": "Repository name (required)"},
                "username": {"type": "string", "description": "User branch (defaults to 'main')"},
            },
            "required": ["scope", "repository"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
]
