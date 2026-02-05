"""MCP Tools module.

This module exports tool implementations and tool schemas for the MCP server.

Tools are organized into category modules:
- search: Semantic and keyword search (search_code, search_features, find_similar)
- elements: Element inspection and navigation (get_element, get_context, get_children, etc.)
- repository: Repository and feature discovery (list_repos, list_features, get_repo_stats, etc.)
- files: File discovery (find_files)
- usages: Usage and implementation finding (find_usages, find_implementations)
- glossary: Domain term discovery (list_glossary, get_glossary_term, search_glossary)
- patterns: Pattern-based search (pattern_search)
"""

# Export tool schemas
from magaldi_mcp.tools.elements import (
    batch_get_elements,
    get_children,
    get_context,
    get_element,
    get_file_structure,
)
from magaldi_mcp.tools.files import find_files
from magaldi_mcp.tools.glossary import (
    get_glossary_term,
    list_glossary,
    search_glossary,
)

# Parser Lab tools
from magaldi_mcp.tools.parser_lab import (
    parser_lab_analyze,
    parser_lab_create_test,
    parser_lab_run_tests,
    parser_lab_suggest_fix,
)
from magaldi_mcp.tools.patterns import pattern_search
from magaldi_mcp.tools.repository import (
    get_feature_members,
    get_repo_stats,
    list_features,
    list_repos,
)
from magaldi_mcp.tools.schemas import ALL_TOOL_SCHEMAS

# Import from new modular structure
from magaldi_mcp.tools.search import (
    find_similar,
    search_code,
    search_features,
)
from magaldi_mcp.tools.usages import (
    find_implementations,
    find_usages,
)

# Import remaining tools from tools_impl (not yet migrated to separate modules)
from magaldi_mcp.tools_impl import (
    dependency_graph,
    explain_element,
    find_async_code,
    find_by_pattern,
    find_call_chain,
    find_callers,
    find_complex_functions,
    find_dead_code,
    find_dependencies,
    find_dependents,
    find_entry_points,
    find_env_usage,
    find_security_issues,
    find_undocumented,
    generate_config,
    generate_skill,
    get_call_graph,
    get_command_tree,
    get_route_tree,
    list_patterns,
    mcp_self_review,
)

__all__ = [
    # Search tools
    "search_code",
    "search_features",
    "find_similar",
    # Element navigation tools
    "get_element",
    "get_context",
    "get_children",
    "batch_get_elements",
    "get_file_structure",
    # Repository/feature tools
    "list_repos",
    "list_features",
    "get_repo_stats",
    "get_feature_members",
    # File tools
    "find_files",
    # Usage tools
    "find_usages",
    "find_implementations",
    # Glossary tools
    "list_glossary",
    "get_glossary_term",
    "search_glossary",
    # Pattern tools
    "pattern_search",
    # Call graph tools (from tools_impl)
    "get_call_graph",
    "find_callers",
    "find_call_chain",
    "find_dead_code",
    "find_entry_points",
    # Dependency tools (from tools_impl)
    "find_dependencies",
    "find_dependents",
    "dependency_graph",
    # Quality tools (from tools_impl)
    "find_complex_functions",
    "find_security_issues",
    "find_undocumented",
    "find_env_usage",
    "find_async_code",
    # Pattern/design tools (from tools_impl)
    "list_patterns",
    "find_by_pattern",
    # Config tools (from tools_impl)
    "generate_skill",
    "generate_config",
    # Tree tools (from tools_impl)
    "get_command_tree",
    "get_route_tree",
    # Analysis tools (from tools_impl)
    "explain_element",
    # MCP review (from tools_impl)
    "mcp_self_review",
    # Parser Lab tools
    "parser_lab_analyze",
    "parser_lab_create_test",
    "parser_lab_run_tests",
    "parser_lab_suggest_fix",
    # Tool schemas
    "ALL_TOOL_SCHEMAS",
]
