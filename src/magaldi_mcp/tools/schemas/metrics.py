"""Metrics tool schemas: complexity, security, documentation, env vars, concurrency."""

from mcp.types import Tool

METRICS_TOOLS = [
    Tool(
        name="find_complex_functions",
        description="FIND COMPLEX: Find functions/methods with high cyclomatic complexity. "
        "Use for identifying code that may need refactoring or additional testing.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string", "description": "Repository scope (required)"},
                "repository": {"type": "string", "description": "Repository name (required)"},
                "username": {"type": "string", "description": "Username branch"},
                "min_complexity": {
                    "type": "integer",
                    "default": 10,
                    "description": "Minimum cyclomatic complexity threshold",
                },
                "limit": {"type": "integer", "default": 20},
                "include_tests": {"type": "boolean", "default": False},
            },
            "required": ["scope", "repository"],
        },
    ),
    Tool(
        name="find_security_issues",
        description="FIND SECURITY: Find potential security issues in code. "
        "Detects hardcoded secrets, SQL injection, command injection, etc. "
        "Use for security audits and code review.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string", "description": "Repository scope (required)"},
                "repository": {"type": "string", "description": "Repository name (required)"},
                "username": {"type": "string", "description": "Username branch"},
                "severity": {
                    "type": "string",
                    "enum": ["critical", "high", "medium", "low", "info", "all"],
                    "default": "high",
                    "description": "Minimum severity level to include",
                },
                "kind": {
                    "type": "string",
                    "description": "Filter by issue kind (e.g., 'sql_injection', 'hardcoded_secret')",
                },
                "limit": {"type": "integer", "default": 50},
            },
            "required": ["scope", "repository"],
        },
    ),
    Tool(
        name="find_undocumented",
        description="FIND UNDOCUMENTED: Find functions/methods missing documentation. "
        "Use for improving code documentation coverage.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string", "description": "Repository scope (required)"},
                "repository": {"type": "string", "description": "Repository name (required)"},
                "username": {"type": "string", "description": "Username branch"},
                "max_coverage": {
                    "type": "number",
                    "default": 0.5,
                    "description": "Maximum documentation coverage (0-1) to include",
                },
                "public_only": {
                    "type": "boolean",
                    "default": True,
                    "description": "Only include public functions/methods",
                },
                "limit": {"type": "integer", "default": 30},
                "include_tests": {"type": "boolean", "default": False},
            },
            "required": ["scope", "repository"],
        },
    ),
    Tool(
        name="find_env_usage",
        description="FIND ENV VARS: Find environment variable usage across the codebase. "
        "Use for understanding configuration dependencies and deployment requirements.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string", "description": "Repository scope (required)"},
                "repository": {"type": "string", "description": "Repository name (required)"},
                "username": {"type": "string", "description": "Username branch"},
                "env_name": {
                    "type": "string",
                    "description": "Filter by specific env var name (optional)",
                },
                "limit": {"type": "integer", "default": 50},
            },
            "required": ["scope", "repository"],
        },
    ),
    Tool(
        name="find_async_code",
        description="FIND ASYNC: Find async/concurrent code patterns. "
        "Detects async functions, threading, multiprocessing, locks. "
        "Use for understanding concurrency patterns in the codebase.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string", "description": "Repository scope (required)"},
                "repository": {"type": "string", "description": "Repository name (required)"},
                "username": {"type": "string", "description": "Username branch"},
                "pattern": {
                    "type": "string",
                    "enum": ["async", "threading", "locking", "all"],
                    "default": "all",
                    "description": "Type of concurrency pattern to find",
                },
                "limit": {"type": "integer", "default": 30},
                "include_tests": {"type": "boolean", "default": False},
            },
            "required": ["scope", "repository"],
        },
    ),
]
