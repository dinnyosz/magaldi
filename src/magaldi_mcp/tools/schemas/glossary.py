"""Glossary tool schemas: list_glossary, get_glossary_term, search_glossary."""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import READONLY_ANNOTATIONS

GLOSSARY_TOOLS = [
    Tool(
        name="list_glossary",
        description="List all glossary terms (domain concepts from code names).",
        inputSchema={
            "type": "object",
            "properties": {
                "min_count": {"type": "integer", "default": 1},
                "limit": {"type": "integer", "default": 50},
            },
            "required": [],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="get_glossary_term",
        description="Get details for a glossary term (elements, files, features).",
        inputSchema={
            "type": "object",
            "properties": {
                "term": {"type": "string"},
            },
            "required": ["term"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="search_glossary",
        description="Search glossary terms by partial match.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
]
