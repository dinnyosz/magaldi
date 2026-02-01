"""Search tool schemas: search_code, search_features, find_similar."""

from mcp.types import Tool

SEARCH_TOOLS = [
    Tool(
        name="search_code",
        description="FIND CODE: Search for functions, classes, methods by what they do. "
        "Uses pre-indexed semantic embeddings - finds 'login' when you search 'authentication'. "
        "Returns AI summaries so you understand code without reading it. "
        "Use include_code=true to see implementation. Use brief=true for exploration.",
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
                    "description": "Filter: file, class, function, method",
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
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="search_features",
        description="FIND CAPABILITIES: Search for high-level features (groups of related functions). "
        "Pre-clustered by AI - 'authentication' returns all auth-related functions grouped together. "
        "Use to understand what the codebase CAN DO, not just specific implementations.",
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
    ),
    Tool(
        name="find_similar",
        description="FIND RELATED CODE: Given an element ID, find similar implementations. "
        "Use after search_code to find related patterns or alternative approaches.",
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
    ),
]
