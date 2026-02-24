"""Analysis tool schemas: pattern_search, find_usages, find_implementations, call graph tools."""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import OUTPUT_CONTROL_PROPERTIES, READONLY_ANNOTATIONS

ANALYSIS_TOOLS = [
    Tool(
        name="pattern_search",
        description="Search code by text/regex/wildcard pattern. Three modes: regexp, wildcard, proximity.",
        inputSchema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["regexp", "wildcard", "proximity"],
                    "description": "regexp: Lucene regex. wildcard: * and ?. proximity: terms near each other.",
                },
                "username": {"type": "string"},
                "slop": {
                    "type": "integer",
                    "default": 5,
                    "description": "Proximity mode: max word distance",
                },
                "glob": {"type": "string", "description": "File filter (e.g., '*.py')"},
                "limit": {"type": "integer", "default": 20},
                "include_tests": {"type": "boolean", "default": True},
                **OUTPUT_CONTROL_PROPERTIES,
            },
            "required": ["pattern", "mode"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="find_usages",
        description="Find where a function/class is called. Filters out definitions.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
                **OUTPUT_CONTROL_PROPERTIES,
            },
            "required": ["hash_id"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="find_implementations",
        description="Find subclasses or implementations of a protocol/base class.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string", "description": "Element ID of the base class"},
                "class_name": {"type": "string", "description": "Or search by class name"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": [],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="get_call_graph",
        description="Get callers and callees for a function. Includes semantic relations.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string"},
                "direction": {"type": "string", "enum": ["callers", "callees", "both"], "default": "both"},
            },
            "required": ["hash_id"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="find_callers",
        description="Find all functions that call a given function. Grouped by code/tests.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string"},
                "username": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
                "include_tests": {"type": "boolean", "default": True},
                **OUTPUT_CONTROL_PROPERTIES,
            },
            "required": ["hash_id"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="find_call_chain",
        description="Trace call chains recursively. For impact analysis.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string"},
                "direction": {
                    "type": "string",
                    "enum": ["callers", "callees", "both"],
                    "default": "callees",
                },
                "max_depth": {"type": "integer", "default": 3, "description": "1-10"},
                "username": {"type": "string"},
                **OUTPUT_CONTROL_PROPERTIES,
            },
            "required": ["hash_id"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="find_dead_code",
        description="Find functions never called. Excludes entry points and magic methods.",
        inputSchema={
            "type": "object",
            "properties": {
                "username": {"type": "string"},
                "include_tests": {"type": "boolean", "default": False},
                "limit": {"type": "integer", "default": 30},
                **OUTPUT_CONTROL_PROPERTIES,
            },
            "required": [],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="find_entry_points",
        description="Find HTTP handlers, CLI commands, test fixtures, main functions.",
        inputSchema={
            "type": "object",
            "properties": {
                "username": {"type": "string"},
                "limit": {"type": "integer", "default": 30},
                **OUTPUT_CONTROL_PROPERTIES,
            },
            "required": [],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
]
