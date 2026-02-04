"""MCP Labs tools for experimentation and self-review."""

from mcp.types import Tool

LABS_TOOLS = [
    Tool(
        name="mcp_self_review",
        description="""MCP SELF REVIEW: Analyze recent magaldi tool usage to evaluate effectiveness.

Reviews the conversation context and MCP analytics data to determine:
1. Which magaldi tool calls were useful (information was used in subsequent reasoning)
2. Which tool calls were not useful (results were ignored or Claude deviated)
3. Suggestions for how tool responses could be improved

This helps identify:
- Tool calls that returned too much/too little information
- Missing fields that would have been helpful
- Response format issues
- Patterns in successful vs unsuccessful tool usage

Use this tool periodically during complex tasks to improve tool effectiveness.""",
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
    ),
]
