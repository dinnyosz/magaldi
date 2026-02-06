"""Glossary tool schemas: list_glossary, get_glossary_term, search_glossary."""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import READONLY_ANNOTATIONS

GLOSSARY_TOOLS = [
    Tool(
        name="list_glossary",
        description="List all glossary terms (domain concepts extracted from code element names).",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
                "repository": {"type": "string"},
                "min_count": {"type": "integer", "default": 1, "description": "Minimum occurrence count"},
            },
            "required": [],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="get_glossary_term",
        description="Get full details for a specific glossary term (element IDs, file paths, features).",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
                "repository": {"type": "string"},
                "term": {"type": "string", "description": "The glossary term to look up"},
            },
            "required": ["term"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="search_glossary",
        description="Search glossary terms by partial match (e.g., 'user' matches 'user', 'username').",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
                "repository": {"type": "string"},
                "query": {"type": "string", "description": "Partial term to search for"},
            },
            "required": ["query"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
]
