"""Search tool schemas: search_code, search_features, find_similar."""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import OUTPUT_CONTROL_PROPERTIES, READONLY_ANNOTATIONS

SEARCH_TOOLS = [
    Tool(
        name="search_code",
        description="Semantic search for code by what it does. Returns AI summaries.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What the code does (e.g., 'handle authentication')",
                },
                "element_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter: file, class, interface, trait, enum, type_alias, function, method, constant, variable, import",
                    "default": ["function", "method"],
                },
                "include_code": {"type": "boolean", "default": False},
                "brief": {
                    "type": "boolean",
                    "description": "False for summaries and signatures",
                    "default": True,
                },
                "limit": {"type": "integer", "default": 20},
                "include_tests": {"type": "boolean", "default": False},
                **OUTPUT_CONTROL_PROPERTIES,
            },
            "required": ["query"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="search_features",
        description="Search for high-level features (groups of related functions).",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Capability to find (e.g., 'authentication', 'caching')",
                },
                "limit": {"type": "integer", "default": 20},
                "glossary_term": {
                    "type": "string",
                    "description": "Filter to features containing this glossary term",
                },
                "min_percentage": {
                    "type": "number",
                    "description": "Min % of members with the term (0-100)",
                    "default": 0,
                },
                **OUTPUT_CONTROL_PROPERTIES,
            },
            "required": ["query"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="find_similar",
        description="Find similar implementations to a given element.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string"},
                "limit": {"type": "integer", "default": 5},
                "same_repo_only": {"type": "boolean", "default": False},
                "include_tests": {"type": "boolean", "default": True},
            },
            "required": ["hash_id"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
]
