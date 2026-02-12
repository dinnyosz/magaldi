"""Metrics tool schemas: complexity, security, documentation, env vars, concurrency."""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import READONLY_ANNOTATIONS

METRICS_TOOLS = [
    Tool(
        name="find_complex_functions",
        description="Find functions with high cyclomatic complexity.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
                "repository": {"type": "string"},
                "username": {"type": "string"},
                "min_complexity": {"type": "integer", "default": 10},
                "limit": {"type": "integer", "default": 20},
                "include_tests": {"type": "boolean", "default": False},
            },
            "required": ["scope", "repository"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="find_security_issues",
        description="Find potential security issues (hardcoded secrets, SQL injection, etc).",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
                "repository": {"type": "string"},
                "username": {"type": "string"},
                "severity": {
                    "type": "string",
                    "enum": ["critical", "high", "medium", "low", "info", "all"],
                    "default": "high",
                },
                "kind": {"type": "string", "description": "Filter by issue kind (e.g., 'sql_injection')"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["scope", "repository"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="find_undocumented",
        description="Find functions/methods missing documentation.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
                "repository": {"type": "string"},
                "username": {"type": "string"},
                "max_coverage": {"type": "number", "default": 0.5, "description": "Max doc coverage (0-1)"},
                "public_only": {"type": "boolean", "default": True},
                "limit": {"type": "integer", "default": 30},
                "include_tests": {"type": "boolean", "default": False},
            },
            "required": ["scope", "repository"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="find_env_usage",
        description="Find environment variable usage across the codebase.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
                "repository": {"type": "string"},
                "username": {"type": "string"},
                "env_name": {"type": "string", "description": "Filter by env var name"},
                "limit": {"type": "integer", "default": 30},
            },
            "required": ["scope", "repository"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="find_async_code",
        description="Find async/concurrent code patterns.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
                "repository": {"type": "string"},
                "username": {"type": "string"},
                "pattern": {
                    "type": "string",
                    "enum": ["async", "threading", "locking", "all"],
                    "default": "all",
                },
                "limit": {"type": "integer", "default": 30},
                "include_tests": {"type": "boolean", "default": False},
            },
            "required": ["scope", "repository"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
]
