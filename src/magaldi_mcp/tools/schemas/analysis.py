"""Analysis tool schemas: pattern_search, find_usages, find_implementations, call graph tools."""

from mcp.types import Tool

ANALYSIS_TOOLS = [
    Tool(
        name="pattern_search",
        description="PATTERN SEARCH: ES-native pattern matching on code. "
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
                "scope": {"type": "string", "description": "Filter by scope (required)"},
                "repository": {"type": "string", "description": "Filter by repo (required)"},
                "username": {"type": "string", "description": "User branch to search"},
                "slop": {
                    "type": "integer",
                    "default": 5,
                    "description": "For proximity mode: max word distance",
                },
                "glob": {"type": "string", "description": "File filter (e.g., '*.py')"},
                "limit": {"type": "integer", "default": 50},
                "include_tests": {"type": "boolean", "default": True},
            },
            "required": ["pattern", "mode", "scope", "repository"],
        },
    ),
    Tool(
        name="find_usages",
        description="FIND USAGES: Find where a function/class/method is called or referenced. "
        "USE THIS instead of grepping for 'functionName(' - automatically filters definitions, "
        "includes context lines, and understands code structure.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string", "description": "Element ID (hash_id from search results)"},
                "limit": {"type": "integer", "default": 30},
            },
            "required": ["hash_id"],
        },
    ),
    Tool(
        name="find_implementations",
        description="FIND IMPLEMENTATIONS: Find classes that inherit from or implement a protocol/base class. "
        "USE THIS instead of grepping for 'class.*BaseClass' - understands inheritance patterns.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string", "description": "Element ID of the protocol/base class (hash_id)"},
                "class_name": {"type": "string", "description": "Or just the class name to search for"},
                "scope": {"type": "string", "description": "Filter by scope"},
                "repository": {"type": "string", "description": "Filter by repo"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": [],
        },
    ),
    Tool(
        name="get_call_graph",
        description="CALL GRAPH: Get callers (who calls this) and callees (what it calls) for a function. "
        "Pre-computed from indexed code - instant dependency analysis for refactoring impact.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string", "description": "Function/method element ID (hash_id)"},
                "direction": {"type": "string", "enum": ["callers", "callees", "both"], "default": "both"},
            },
            "required": ["hash_id"],
        },
    ),
    Tool(
        name="find_callers",
        description="FIND CALLERS: Find all functions that call a given function/method. "
        "Uses indexed call data for instant results. Returns callers grouped by code/tests.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string", "description": "Target element ID to find callers of (hash_id)"},
                "scope": {"type": "string", "description": "Filter by scope"},
                "repository": {"type": "string", "description": "Filter by repository"},
                "username": {"type": "string", "description": "Filter by username branch"},
                "limit": {"type": "integer", "default": 30, "description": "Max results"},
                "include_tests": {"type": "boolean", "default": True, "description": "Include test functions"},
            },
            "required": ["hash_id"],
        },
    ),
    Tool(
        name="find_call_chain",
        description="CALL CHAIN: Trace call chains from an element. "
        "Shows what a function calls (callees) or what calls it (callers) recursively. "
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
                "scope": {"type": "string", "description": "Filter by scope"},
                "repository": {"type": "string", "description": "Filter by repository"},
                "username": {"type": "string", "description": "Filter by username branch"},
            },
            "required": ["hash_id"],
        },
    ),
    Tool(
        name="find_dead_code",
        description="DEAD CODE: Find functions/methods that are never called. "
        "Excludes entry points (routes, CLI commands), magic methods, and main functions. "
        "Use for codebase cleanup.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string", "description": "Repository scope (required)"},
                "repository": {"type": "string", "description": "Repository name (required)"},
                "username": {"type": "string", "description": "Username branch"},
                "include_tests": {"type": "boolean", "default": False, "description": "Include test functions in check"},
            },
            "required": ["scope", "repository"],
        },
    ),
    Tool(
        name="find_entry_points",
        description="ENTRY POINTS: Find HTTP handlers, CLI commands, test fixtures, main functions. "
        "Detects entry points by decorator patterns (@route, @command, @fixture) and naming. "
        "Returns grouped by type.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string", "description": "Repository scope (required)"},
                "repository": {"type": "string", "description": "Repository name (required)"},
                "username": {"type": "string", "description": "Username branch"},
            },
            "required": ["scope", "repository"],
        },
    ),
]
