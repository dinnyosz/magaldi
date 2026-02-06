"""Search tool schemas: search_code, search_features, find_similar."""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import READONLY_ANNOTATIONS

SEARCH_TOOLS = [
    Tool(
        name="search_code",
        description="Semantic search for functions, classes, methods by what they do. "
        "Returns AI summaries so you understand code without reading it.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What the code does (e.g., 'handle authentication', 'parse JSON', 'validate email')",
                },
                "element_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter: file, class, interface, trait, enum, type_alias, function, method, constant, variable, import",
                },
                "include_code": {
                    "type": "boolean",
                    "description": "Include source code in results",
                    "default": False,
                },
                "brief": {
                    "type": "boolean",
                    "description": "Minimal output (name, file, line only) - use for broad exploration",
                    "default": False,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default: 20)",
                    "default": 20,
                },
                "repository": {"type": "string", "description": "Filter by repo"},
                "scope": {"type": "string", "description": "Filter by scope"},
                "include_tests": {
                    "type": "boolean",
                    "description": "Include test elements in results. Default: true.",
                    "default": True,
                },
                "include_related": {
                    "type": "boolean",
                    "description": "Include related files that call/use found elements. Default: true.",
                    "default": True,
                },
            },
            "required": ["query"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="search_features",
        description="Search for high-level features (groups of related functions). "
        "Returns pre-clustered capabilities in the codebase.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What capability you need (e.g., 'authentication', 'caching', 'validation')",
                },
                "limit": {"type": "integer", "default": 20},
                "repository": {"type": "string"},
                "scope": {"type": "string"},
                "glossary_term": {
                    "type": "string",
                    "description": "Filter to features where this glossary term appears in members",
                },
                "min_percentage": {
                    "type": "number",
                    "description": "Minimum percentage of members containing the term (0-100)",
                    "default": 0,
                },
            },
            "required": ["query"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="find_similar",
        description="Given an element ID, find similar implementations and related patterns.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string", "description": "Element ID (hash_id from search results)"},
                "limit": {"type": "integer", "default": 10},
                "same_repo_only": {"type": "boolean", "default": False},
                "include_tests": {
                    "type": "boolean",
                    "description": "Include test elements in results. Default: true.",
                    "default": True,
                },
            },
            "required": ["hash_id"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
]
