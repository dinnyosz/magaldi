"""Meta tool schemas: list_repos, get_repo_stats, generate_skill, explain_element."""

from mcp.types import Tool

META_TOOLS = [
    Tool(
        name="list_repos",
        description="LIST REPOSITORIES: See all indexed codebases.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
            },
            "required": [],
        },
    ),
    Tool(
        name="get_repo_stats",
        description="REPO OVERVIEW: Get statistics (element counts, languages, etc).",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
                "repository": {"type": "string"},
            },
            "required": [],
        },
    ),
    Tool(
        name="generate_skill",
        description="GENERATE SKILL: Create a SKILL.md file that teaches LLMs how to use this MCP effectively. "
        "The skill file documents best practices, workflows, and anti-patterns for token-efficient code discovery. "
        "BEFORE CALLING: Ask the user (1) scope: project or global? (defaults to MCP installation type), "
        "and (2) whether to add magaldi tools to allowed tools (recommended: yes).",
        inputSchema={
            "type": "object",
            "properties": {
                "project_root": {"type": "string", "description": "Project root directory (required for scope='project')"},
                "skill_name": {"type": "string", "default": "magaldi", "description": "Name of the skill directory"},
                "scope": {
                    "type": "string",
                    "enum": ["project", "global"],
                    "description": "Where to install. Defaults to MCP installation scope if not specified.",
                },
                "add_allowed_tools": {
                    "type": "boolean",
                    "default": False,
                    "description": "Add all magaldi MCP tools to allowed tools (no permission prompts)",
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="explain_element",
        description="EXPLAIN ELEMENT: Comprehensive overview of a code element. "
        "Returns element details (name, type, signature, summary), callers (top 5), "
        "callees (all direct calls), imports (if file), similar code (top 3), and parent context. "
        "Use to understand any element in one call.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string", "description": "Element ID to explain (hash_id)"},
            },
            "required": ["hash_id"],
        },
    ),
    Tool(
        name="generate_config",
        description="GENERATE CONFIG: Create a magaldi.yaml config file for a repository. "
        "Auto-detects repository name from directory and scope from parent directory. "
        "Creates the config at the repository root (not in .magaldi/). "
        "Use this before indexing a new repository.",
        inputSchema={
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to the repository root directory",
                },
                "scope": {
                    "type": "string",
                    "description": "Override auto-detected scope (optional)",
                },
                "repository": {
                    "type": "string",
                    "description": "Override auto-detected repository name (optional)",
                },
            },
            "required": ["repo_path"],
        },
    ),
]
