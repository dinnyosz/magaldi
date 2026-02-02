"""Feature tool schemas: list_features, get_feature_members."""

from mcp.types import Tool

FEATURE_TOOLS = [
    Tool(
        name="list_features",
        description="LIST ALL FEATURES: See all extracted features/capabilities in a repo. "
        "Use to get a high-level understanding of what the codebase does. "
        "Then use get_feature_members to see the actual code.",
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
        name="get_feature_members",
        description="FEATURE DETAILS: Get all functions that belong to a feature. "
        "Shows the actual implementations grouped under a feature.",
        inputSchema={
            "type": "object",
            "properties": {
                "feature_id": {"type": "string"},
            },
            "required": ["feature_id"],
        },
    ),
]
