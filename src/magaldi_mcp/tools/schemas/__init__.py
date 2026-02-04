"""Tool schemas for MCP server."""

from magaldi_mcp.tools.schemas.analysis import ANALYSIS_TOOLS
from magaldi_mcp.tools.schemas.dependencies import DEPENDENCY_TOOLS
from magaldi_mcp.tools.schemas.features import FEATURE_TOOLS
from magaldi_mcp.tools.schemas.files import FILE_TOOLS
from magaldi_mcp.tools.schemas.glossary import GLOSSARY_TOOLS
from magaldi_mcp.tools.schemas.hierarchy import HIERARCHY_TOOLS
from magaldi_mcp.tools.schemas.inspect import INSPECT_TOOLS
from magaldi_mcp.tools.schemas.labs import LABS_TOOLS
from magaldi_mcp.tools.schemas.meta import META_TOOLS
from magaldi_mcp.tools.schemas.metrics import METRICS_TOOLS
from magaldi_mcp.tools.schemas.parser_lab import PARSER_LAB_TOOLS
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
    METRICS_TOOLS +
    HIERARCHY_TOOLS +
    META_TOOLS +
    PARSER_LAB_TOOLS +
    LABS_TOOLS
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
    "METRICS_TOOLS",
    "HIERARCHY_TOOLS",
    "META_TOOLS",
    "PARSER_LAB_TOOLS",
    "LABS_TOOLS",
]
