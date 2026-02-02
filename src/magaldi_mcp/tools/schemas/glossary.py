"""Glossary tool schemas: list_glossary, get_glossary_term, search_glossary."""

from mcp.types import Tool

GLOSSARY_TOOLS = [
    Tool(
        name="list_glossary",
        description="LIST GLOSSARY: List all glossary terms for a repository. "
        "Shows domain concepts extracted from code element names. "
        "Use to discover what terminology exists in the codebase.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string", "description": "Filter by scope"},
                "repository": {"type": "string", "description": "Filter by repo"},
                "min_count": {"type": "integer", "default": 1, "description": "Minimum occurrence count"},
            },
            "required": [],
        },
    ),
    Tool(
        name="get_glossary_term",
        description="GET GLOSSARY TERM: Get full details for a specific glossary term. "
        "Shows element IDs, file paths, and feature associations for the term.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
                "repository": {"type": "string"},
                "term": {"type": "string", "description": "The glossary term to look up"},
            },
            "required": ["term"],
        },
    ),
    Tool(
        name="search_glossary",
        description="SEARCH GLOSSARY: Search glossary terms by partial match. "
        "Use to find all terms related to a concept (e.g., 'user' matches 'user', 'username').",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
                "repository": {"type": "string"},
                "query": {"type": "string", "description": "Partial term to search for"},
            },
            "required": ["query"],
        },
    ),
]
