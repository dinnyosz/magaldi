"""Tool schemas for MCP server."""

from magaldi_mcp.tools.schemas.analysis import ANALYSIS_TOOLS
from magaldi_mcp.tools.schemas.dependencies import DEPENDENCY_TOOLS
from magaldi_mcp.tools.schemas.features import FEATURE_TOOLS
from magaldi_mcp.tools.schemas.files import FILE_TOOLS
from magaldi_mcp.tools.schemas.glossary import GLOSSARY_TOOLS
from magaldi_mcp.tools.schemas.inspect import INSPECT_TOOLS
from magaldi_mcp.tools.schemas.meta import META_TOOLS
from magaldi_mcp.tools.schemas.patterns import PATTERN_TOOLS
from magaldi_mcp.tools.schemas.search import SEARCH_TOOLS

# Combine all tool schemas
ALL_TOOL_SCHEMAS = (
    SEARCH_TOOLS +
    INSPECT_TOOLS +
    FILE_TOOLS +
    FEATURE_TOOLS +
    ANALYSIS_TOOLS +
    GLOSSARY_TOOLS +
    DEPENDENCY_TOOLS +
    PATTERN_TOOLS +
    META_TOOLS
)

__all__ = [
    "ALL_TOOL_SCHEMAS",
    "SEARCH_TOOLS",
    "INSPECT_TOOLS",
    "FILE_TOOLS",
    "FEATURE_TOOLS",
    "ANALYSIS_TOOLS",
    "GLOSSARY_TOOLS",
    "DEPENDENCY_TOOLS",
    "PATTERN_TOOLS",
    "META_TOOLS",
]
