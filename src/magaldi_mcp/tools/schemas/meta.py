"""Meta tool schemas: list_repos, get_repo_stats, generate_skill, explain_element."""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import READONLY_ANNOTATIONS, WRITE_ANNOTATIONS

META_TOOLS = [
    Tool(
        name="list_repos",
        description="List all indexed codebases.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
            },
            "required": [],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="get_repo_stats",
        description="Get repository statistics (element counts, languages, etc).",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
                "repository": {"type": "string"},
            },
            "required": [],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="generate_skill",
        description="Create a SKILL.md file that teaches LLMs how to use this MCP. "
        "Ask the user for scope (project/global) and whether to add allowed tools.",
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
        annotations=WRITE_ANNOTATIONS,
    ),
    Tool(
        name="explain_element",
        description="Comprehensive overview of a code element in one call. "
        "Returns details, callers, callees, imports, similar code, and parent context.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string", "description": "Element ID to explain (hash_id)"},
            },
            "required": ["hash_id"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="generate_config",
        description="Create a magaldi.yaml config file for a repository. "
        "Auto-detects repository name and scope from directory structure.",
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
        annotations=WRITE_ANNOTATIONS,
    ),
]
