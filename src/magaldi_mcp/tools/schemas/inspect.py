"""Inspect tool schemas: get_element, batch_get_elements, get_context, get_children."""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import READONLY_ANNOTATIONS

INSPECT_TOOLS = [
    Tool(
        name="get_element",
        description="Get details of a code element by ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string"},
                "include_code": {"type": "boolean", "default": False},
                "brief": {
                    "type": "boolean",
                    "default": True,
                    "description": "Core fields only (default). False for calls, complexity, etc.",
                },
            },
            "required": ["hash_id"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="batch_get_elements",
        description="Get multiple elements by ID in one call.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "include_code": {"type": "boolean", "default": False},
            },
            "required": ["hash_ids"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="get_context",
        description="Get an element's parent, siblings, and children.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string"},
                "include_children": {"type": "boolean", "default": True},
                "include_siblings": {"type": "boolean", "default": False},
            },
            "required": ["hash_id"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="get_children",
        description="Get child elements (methods in a class, etc).",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string"},
            },
            "required": ["hash_id"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
]
