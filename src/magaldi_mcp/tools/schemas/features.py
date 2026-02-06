"""Feature tool schemas: list_features, get_feature_members."""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import READONLY_ANNOTATIONS

FEATURE_TOOLS = [
    Tool(
        name="list_features",
        description="List all extracted features/capabilities in a repo. "
        "Use get_feature_members to see actual code in each feature.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
                "repository": {"type": "string"},
                "brief": {
                    "type": "boolean",
                    "default": True,
                    "description": "Omit summaries (default). Set false to include feature summaries.",
                },
            },
            "required": ["scope", "repository"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
    Tool(
        name="get_feature_members",
        description="Get all functions that belong to a feature.",
        inputSchema={
            "type": "object",
            "properties": {
                "feature_id": {"type": "string"},
            },
            "required": ["feature_id"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
]
