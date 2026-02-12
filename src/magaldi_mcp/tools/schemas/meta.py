"""Meta tool schemas: list_repos, get_repo_stats, generate_skill, explain_element."""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import (
    OUTPUT_CONTROL_PROPERTIES,
    READONLY_ANNOTATIONS,
    WRITE_ANNOTATIONS,
)

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
        description="Create a SKILL.md that teaches LLMs to use this MCP.",
        inputSchema={
            "type": "object",
            "properties": {
                "project_root": {"type": "string", "description": "Project root (for scope='project')"},
                "skill_name": {"type": "string", "default": "magaldi"},
                "scope": {
                    "type": "string",
                    "enum": ["project", "global"],
                },
                "add_allowed_tools": {
                    "type": "boolean",
                    "default": False,
                    "description": "Auto-allow all magaldi tools (no prompts)",
                },
            },
            "required": [],
        },
        annotations=WRITE_ANNOTATIONS,
    ),
    Tool(
        name="explain_element",
        description="Full overview: details, callers, callees, imports, similar, parent.",
        inputSchema={
            "type": "object",
            "properties": {
                "hash_id": {"type": "string"},
                **OUTPUT_CONTROL_PROPERTIES,
            },
            "required": ["hash_id"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="generate_config",
        description="Create magaldi.yaml config. Auto-detects repo name and scope.",
        inputSchema={
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Repository root directory"},
                "scope": {"type": "string", "description": "Override auto-detected scope"},
                "repository": {"type": "string", "description": "Override auto-detected name"},
            },
            "required": ["repo_path"],
        },
        annotations=WRITE_ANNOTATIONS,
    ),
]
