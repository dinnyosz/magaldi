"""Hierarchy tool schemas: CLI command tree, HTTP route tree."""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import READONLY_ANNOTATIONS

HIERARCHY_TOOLS = [
    Tool(
        name="get_command_tree",
        description="Get CLI command hierarchy (Click, Typer, etc).",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
                "repository": {"type": "string"},
                "username": {"type": "string"},
            },
            "required": ["scope", "repository"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="get_route_tree",
        description="Get HTTP route hierarchy (FastAPI, Flask, etc).",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
                "repository": {"type": "string"},
                "username": {"type": "string"},
            },
            "required": ["scope", "repository"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
]
