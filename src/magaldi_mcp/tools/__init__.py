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
- call_graph: Call graph analysis (get_call_graph, find_callers, find_call_chain, etc.)
- dependencies: Dependency analysis (find_dependencies, find_dependents, dependency_graph)
- quality: Code quality tools (find_complex_functions, find_security_issues, etc.)
- design_patterns: Design pattern detection (list_patterns, find_by_pattern)
- trees: CLI command and HTTP route trees (get_command_tree, get_route_tree)
- analysis: Element analysis (explain_element)
- config: Configuration generation (generate_skill, generate_config)
- mcp_review: MCP self-review (mcp_self_review)
"""

# Export tool schemas
# Analysis tools
from magaldi_mcp.tools.analysis import explain_element

# Call graph tools
from magaldi_mcp.tools.call_graph import (
    find_call_chain,
    find_callers,
    find_dead_code,
    find_entry_points,
    get_call_graph,
)

# Config tools
from magaldi_mcp.tools.config import generate_config, generate_skill

# Dependencies tools
from magaldi_mcp.tools.dependencies import (
    dependency_graph,
    find_dependencies,
    find_dependents,
)

# Design pattern tools
from magaldi_mcp.tools.design_patterns import find_by_pattern, list_patterns

# Element navigation tools
from magaldi_mcp.tools.elements import (
    batch_get_elements,
    get_children,
    get_context,
    get_element,
    get_file_structure,
)

# File tools
from magaldi_mcp.tools.files import find_files

# Glossary tools
from magaldi_mcp.tools.glossary import (
    get_glossary_term,
    list_glossary,
    search_glossary,
)

# MCP review tools
from magaldi_mcp.tools.mcp_review import mcp_self_review

# Parser Lab tools
from magaldi_mcp.tools.parser_lab import (
    parser_lab_analyze,
    parser_lab_create_test,
    parser_lab_run_tests,
    parser_lab_suggest_fix,
)

# Pattern tools
from magaldi_mcp.tools.patterns import pattern_search

# Prompt Lab tools
from magaldi_mcp.tools.prompt_lab import prompt_lab_improve

# Quality tools
from magaldi_mcp.tools.quality import (
    find_async_code,
    find_complex_functions,
    find_env_usage,
    find_security_issues,
    find_undocumented,
)

# Repository/feature tools
from magaldi_mcp.tools.repository import (
    get_feature_members,
    get_repo_stats,
    list_features,
    list_repos,
)
from magaldi_mcp.tools.schemas import ALL_TOOL_SCHEMAS

# Search tools
from magaldi_mcp.tools.search import (
    find_similar,
    search_code,
    search_features,
)

# Tree tools
from magaldi_mcp.tools.trees import get_command_tree, get_route_tree

# Usage tools
from magaldi_mcp.tools.usages import (
    find_implementations,
    find_usages,
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
    # Call graph tools
    "get_call_graph",
    "find_callers",
    "find_call_chain",
    "find_dead_code",
    "find_entry_points",
    # Dependency tools
    "find_dependencies",
    "find_dependents",
    "dependency_graph",
    # Quality tools
    "find_complex_functions",
    "find_security_issues",
    "find_undocumented",
    "find_env_usage",
    "find_async_code",
    # Design pattern tools
    "list_patterns",
    "find_by_pattern",
    # Config tools
    "generate_skill",
    "generate_config",
    # Tree tools
    "get_command_tree",
    "get_route_tree",
    # Analysis tools
    "explain_element",
    # MCP review
    "mcp_self_review",
    # Prompt Lab tools
    "prompt_lab_improve",
    # Parser Lab tools
    "parser_lab_analyze",
    "parser_lab_create_test",
    "parser_lab_run_tests",
    "parser_lab_suggest_fix",
    # Tool schemas
    "ALL_TOOL_SCHEMAS",
]
