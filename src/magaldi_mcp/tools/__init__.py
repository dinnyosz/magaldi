"""MCP Tools module.

This module exports tool implementations and tool schemas for the MCP server.
"""

# Re-export all tool implementations from the original tools module
# Export tool schemas
from magaldi_mcp.tools.schemas import ALL_TOOL_SCHEMAS
from magaldi_mcp.tools_impl import (
    batch_get_elements,
    dependency_graph,
    explain_element,
    find_call_chain,
    find_callers,
    find_dead_code,
    find_dependencies,
    find_dependents,
    find_entry_points,
    find_files,
    find_implementations,
    find_similar,
    find_usages,
    generate_skill,
    get_call_graph,
    get_children,
    get_context,
    get_element,
    get_feature_members,
    get_file_structure,
    get_glossary_term,
    get_repo_stats,
    grep_code,
    list_features,
    list_glossary,
    list_repos,
    pattern_search,
    search_code,
    search_features,
    search_glossary,
)

__all__ = [
    # Tool implementations
    "batch_get_elements",
    "dependency_graph",
    "explain_element",
    "find_call_chain",
    "find_callers",
    "find_dead_code",
    "find_dependencies",
    "find_dependents",
    "find_entry_points",
    "find_files",
    "find_implementations",
    "find_similar",
    "find_usages",
    "generate_skill",
    "get_call_graph",
    "get_children",
    "get_context",
    "get_element",
    "get_feature_members",
    "get_file_structure",
    "get_glossary_term",
    "get_repo_stats",
    "grep_code",
    "list_features",
    "list_glossary",
    "list_repos",
    "pattern_search",
    "search_code",
    "search_features",
    "search_glossary",
    # Tool schemas
    "ALL_TOOL_SCHEMAS",
]
