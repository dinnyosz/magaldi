"""MCP Labs tools for experimentation and self-review."""

from mcp.types import Tool

LABS_TOOLS = [
    Tool(
        name="mcp_self_review",
        description="""MCP SELF REVIEW: Analyze recent magaldi tool usage to SUGGEST improvements.

Analyzes tool call sequences to identify potential improvement opportunities:
1. Deviation patterns - when magaldi results led to fallback to Read/Grep/Glob
2. Query refinements - when initial search didn't find what was needed
3. Missing information - what was searched for elsewhere after magaldi calls

IMPORTANT: This tool provides SUGGESTIONS only. The user should review each
suggestion and decide whether it's worth implementing. Not all patterns
indicate problems - some may be expected workflow.

Output includes:
- deviation_patterns: When builtin tools were used after magaldi (potential gaps)
- usage_analysis: Which results were actually referenced later
- improvement_suggestions: Specific, actionable ideas for magaldi enhancements

Use periodically to identify magaldi improvement opportunities.""",
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
