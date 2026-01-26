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
            "required": ["scope", "repository"],
        },
    ),
    Tool(
        name="generate_skill",
        description="GENERATE SKILL: Create a SKILL.md file that teaches LLMs how to use this MCP effectively. "
        "The skill file documents best practices, workflows, and anti-patterns for token-efficient code discovery. "
        "ASK THE USER whether to install globally (~/.claude/skills) or project-local (.claude/skills).",
        inputSchema={
            "type": "object",
            "properties": {
                "project_root": {"type": "string", "description": "Project root directory (required for scope='project')"},
                "skill_name": {"type": "string", "default": "magaldi", "description": "Name of the skill directory"},
                "scope": {
                    "type": "string",
                    "enum": ["project", "global"],
                    "default": "project",
                    "description": "Where to install: 'project' (this project only) or 'global' (all projects)",
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
                "element_id": {"type": "string", "description": "Element ID to explain"},
            },
            "required": ["element_id"],
        },
    ),
]
