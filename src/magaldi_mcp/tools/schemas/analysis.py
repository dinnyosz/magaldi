"""Analysis tool schemas: pattern_search, find_usages, find_implementations, call graph tools."""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import OUTPUT_CONTROL_PROPERTIES, READONLY_ANNOTATIONS

ANALYSIS_TOOLS = [
    Tool(
        name="pattern_search",
        description="Search code by text pattern, regex, or grep-like matching. "
        "Three modes: regexp (Lucene syntax), wildcard (* and ?), proximity (terms near each other).",
        inputSchema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern. Syntax depends on mode.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["regexp", "wildcard", "proximity"],
                    "description": "regexp: Lucene regex (e.g., 'add_column.*Model'). "
                    "wildcard: Simple wildcards (e.g., '*column*'). "
                    "proximity: Terms near each other (e.g., 'add column Model').",
                },
                "scope": {"type": "string"},
                "repository": {"type": "string"},
                "username": {"type": "string", "description": "User branch"},
                "slop": {
                    "type": "integer",
                    "default": 5,
                    "description": "For proximity mode: max word distance",
                },
                "glob": {"type": "string", "description": "File filter (e.g., '*.py')"},
                "limit": {"type": "integer", "default": 50},
                "include_tests": {"type": "boolean", "default": True},
                **OUTPUT_CONTROL_PROPERTIES,
            },
            "required": ["pattern", "mode"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="find_usages",
        description="Find where a function/class/method is called or referenced. "
        "Automatically filters definitions and includes context. "
        "Use after search_code to trace usage sites.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string", "description": "Element ID (hash_id from search results)"},
                "limit": {"type": "integer", "default": 30},
                **OUTPUT_CONTROL_PROPERTIES,
            },
            "required": ["hash_id"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="find_implementations",
        description="Find classes that inherit from or implement a protocol/base class. "
        "Shows subclasses and interface implementations.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string", "description": "Element ID of the protocol/base class (hash_id)"},
                "class_name": {"type": "string", "description": "Or just the class name to search for"},
                "scope": {"type": "string"},
                "repository": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": [],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="get_call_graph",
        description="Get callers and callees for a function. "
        "Pre-computed call graph for instant dependency analysis. "
        "Also includes semantically related functions when available.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string", "description": "Function/method element ID (hash_id)"},
                "direction": {"type": "string", "enum": ["callers", "callees", "both"], "default": "both"},
            },
            "required": ["hash_id"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="find_callers",
        description="Find all functions that call a given function/method. "
        "Returns callers grouped by code/tests. "
        "Use for impact analysis before modifying code.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string", "description": "Target element ID to find callers of (hash_id)"},
                "scope": {"type": "string"},
                "repository": {"type": "string"},
                "username": {"type": "string", "description": "User branch"},
                "limit": {"type": "integer", "default": 30},
                "include_tests": {"type": "boolean", "default": True},
                **OUTPUT_CONTROL_PROPERTIES,
            },
            "required": ["hash_id"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="find_call_chain",
        description="Trace call chains from an element recursively. "
        "Use for impact analysis before refactoring.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string", "description": "Starting element ID (hash_id)"},
                "direction": {
                    "type": "string",
                    "enum": ["callers", "callees", "both"],
                    "default": "callees",
                    "description": "callers: what calls this, callees: what this calls, both: both directions",
                },
                "max_depth": {"type": "integer", "default": 5, "description": "Max depth to traverse (1-10)"},
                "scope": {"type": "string"},
                "repository": {"type": "string"},
                "username": {"type": "string", "description": "User branch"},
                **OUTPUT_CONTROL_PROPERTIES,
            },
            "required": ["hash_id"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="find_dead_code",
        description="Find functions/methods that are never called. "
        "Excludes entry points, magic methods, and main functions.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
                "repository": {"type": "string"},
                "username": {"type": "string", "description": "User branch"},
                "include_tests": {"type": "boolean", "default": False},
                **OUTPUT_CONTROL_PROPERTIES,
            },
            "required": [],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="find_entry_points",
        description="Find HTTP handlers, CLI commands, test fixtures, and main functions. "
        "Returns grouped by type.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
                "repository": {"type": "string"},
                "username": {"type": "string", "description": "User branch"},
                **OUTPUT_CONTROL_PROPERTIES,
            },
            "required": [],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
]
