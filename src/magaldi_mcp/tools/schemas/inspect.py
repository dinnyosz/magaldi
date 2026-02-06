"""Inspect tool schemas: get_element, batch_get_elements, get_context, get_children."""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import READONLY_ANNOTATIONS

INSPECT_TOOLS = [
    Tool(
        name="get_element",
        description="Get details of a specific element by ID. "
        "Use include_code=true to see source code.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string", "description": "Element ID (hash_id from search results)"},
                "include_code": {"type": "boolean", "default": False},
                "brief": {
                    "type": "boolean",
                    "default": True,
                    "description": "Return core fields only (default). Set false for calls, complexity, etc.",
                },
            },
            "required": ["hash_id"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="batch_get_elements",
        description="Get several elements at once by their IDs. "
        "More efficient than multiple get_element calls.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of element IDs (hash_ids from search results)",
                },
                "include_code": {"type": "boolean", "default": False},
            },
            "required": ["hash_ids"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="get_context",
        description="See where an element fits - its parent class, siblings, and children.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string", "description": "Element ID (hash_id from search results)"},
                "include_children": {"type": "boolean", "default": True},
                "include_siblings": {"type": "boolean", "default": False},
            },
            "required": ["hash_id"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="get_children",
        description="Get all child elements (methods in a class, etc).",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string", "description": "Parent element ID (hash_id from search results)"},
            },
            "required": ["hash_id"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
]
