"""MCP Prompt Lab tools for RALPH prompt optimization."""

from mcp.types import Tool

from magaldi_mcp.tools.schemas._annotations import READONLY_ANNOTATIONS

PROMPT_LAB_TOOLS = [
    Tool(
        name="prompt_lab_improve",
        description=(
            "RALPH prompt optimizer: iteratively improves summarization prompts for a code element. "
            "Generates a summary, evaluates it with LLM-as-judge, uses feedback to rewrite the prompt, "
            "and repeats until the score converges. Returns per-iteration scores, the best prompt, "
            "and improvement reasoning."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "element_id": {
                    "type": "string",
                    "description": "Element ID or hash prefix to optimize prompts for.",
                },
                "max_iterations": {
                    "type": "integer",
                    "default": 5,
                    "description": "Max RALPH iterations (including baseline). Default: 5.",
                },
                "target_score": {
                    "type": "number",
                    "default": 8.0,
                    "description": "Stop when avg score >= this (1-10). Default: 8.0.",
                },
            },
            "required": ["element_id"],
        },
        annotations=READONLY_ANNOTATIONS,
    ),
]
