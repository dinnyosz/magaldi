"""MCP Labs tools for experimentation and self-review."""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import READONLY_ANNOTATIONS

LABS_TOOLS = [
    Tool(
        name="mcp_self_review",
        description="Analyze recent tool usage to suggest query improvements.",
        inputSchema={
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "description": "Recent conversation context with tool calls and reasoning.",
                },
                "include_analytics": {"type": "boolean", "default": True},
                "focus_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tool names to focus on",
                },
            },
            "required": ["context"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
]
