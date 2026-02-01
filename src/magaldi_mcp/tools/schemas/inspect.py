"""Inspect tool schemas: get_element, batch_get_elements, get_context, get_children."""

from mcp.types import Tool

INSPECT_TOOLS = [
    Tool(
        name="get_element",
        description="INSPECT ELEMENT: Get full details of a specific element by ID. "
        "Use after search_code to see complete info. Use include_code=true for source.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string", "description": "Element ID (hash_id from search results)"},
                "include_code": {"type": "boolean", "default": False},
            },
            "required": ["hash_id"],
        },
    ),
    Tool(
        name="batch_get_elements",
        description="INSPECT MULTIPLE: Get several elements at once by their IDs. "
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
    ),
    Tool(
        name="get_context",
        description="UNDERSTAND CONTEXT: See where an element fits - its parent class, "
        "siblings, and children. Use to understand code structure around a function.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string", "description": "Element ID (hash_id from search results)"},
                "include_children": {"type": "boolean", "default": True},
                "include_siblings": {"type": "boolean", "default": False},
            },
            "required": ["hash_id"],
        },
    ),
    Tool(
        name="get_children",
        description="LIST CHILDREN: Get all child elements (methods in a class, etc). "
        "Use to explore what a class contains.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string", "description": "Parent element ID (hash_id from search results)"},
            },
            "required": ["hash_id"],
        },
    ),
]
