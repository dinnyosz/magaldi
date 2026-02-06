"""MCP Labs tools for experimentation and self-review."""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import READONLY_ANNOTATIONS

LABS_TOOLS = [
    Tool(
        name="mcp_self_review",
        description="Analyze recent magaldi tool usage to suggest improvements. "
        "Identifies deviation patterns, query refinements, and missing information.",
        inputSchema={
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "description": "The recent conversation context to analyze. Include the tool calls and subsequent reasoning/actions taken.",
                },
                "include_analytics": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to include MCP analytics data (tool usage patterns, transitions).",
                },
                "focus_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of specific tool names to focus the review on.",
                },
            },
            "required": ["context"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
]
