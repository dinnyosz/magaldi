"""Shared MCP tool annotations."""

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
