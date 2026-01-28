"""Hierarchy tool schemas: CLI command tree, HTTP route tree."""

from mcp.types import Tool

HIERARCHY_TOOLS = [
    Tool(
        name="get_command_tree",
        description="CLI COMMAND TREE: Get hierarchical structure of CLI commands (Click, Typer, etc.) "
        "with their full invocation paths like 'magaldi web serve'. Use this to understand CLI structure, "
        "document commands, or navigate to specific command handlers.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string", "description": "Repository scope (required)"},
                "repository": {"type": "string", "description": "Repository name (required)"},
                "username": {"type": "string", "description": "User branch (defaults to 'main')"},
            },
            "required": ["scope", "repository"],
        },
    ),
    Tool(
        name="get_route_tree",
        description="HTTP ROUTE TREE: Get hierarchical structure of HTTP routes (FastAPI, Flask, etc.) "
        "grouped by router module with full URL paths like '/api/v1/users/{id}'. Use this to understand "
        "API structure, document endpoints, or navigate to specific route handlers.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string", "description": "Repository scope (required)"},
                "repository": {"type": "string", "description": "Repository name (required)"},
                "username": {"type": "string", "description": "User branch (defaults to 'main')"},
            },
            "required": ["scope", "repository"],
        },
    ),
]
