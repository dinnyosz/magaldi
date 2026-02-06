"""Shared MCP tool annotations and schema helpers."""

from mcp.types import ToolAnnotations

# Most tools are read-only queries against the index
READONLY_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)

# Tools that write files (generate_skill, generate_config)
WRITE_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)

# Tools that support max_tokens and filename parameters for output control
TOOLS_WITH_OUTPUT_CONTROL = {
    "search_code",
    "search_features",
    "find_usages",
    "pattern_search",
    "find_callers",
    "find_call_chain",
    "explain_element",
    "find_dead_code",
    "find_entry_points",
}

# Shared schema properties for output control (added to high-volume tools)
OUTPUT_CONTROL_PROPERTIES = {
    "max_tokens": {
        "type": "integer",
        "description": "Approximate max tokens in response. Results are dropped (not cut mid-result) to fit budget.",
    },
    "filename": {
        "type": "string",
        "description": "Save full results to this file and return a summary instead.",
    },
}
