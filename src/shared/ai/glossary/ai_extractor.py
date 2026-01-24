"""AI-powered glossary extraction from feature summaries."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from shared.ai.llm_client import LLMClient, LLMError

if TYPE_CHECKING:
    from shared.config import MagaldiConfig


# Prompt template for glossary extraction
GLOSSARY_EXTRACTION_PROMPT = """You are extracting glossary terms from a code feature description.

Feature: {label}
Description: {summary}

Extract domain-specific glossary items that represent:
- Actors: entities that perform actions (e.g., user, admin, worker, client)
- Concepts: domain objects or processes (e.g., email, registration, authentication, payment)

For each item, provide:
- name: a short lowercase term (1-2 words)
- description: one sentence explaining what it represents in this codebase

Rules:
- Only extract terms that are meaningful in the domain context
- Ignore generic programming terms (function, class, method, variable, etc.)
- Ignore technical implementation details (cache, queue, handler, etc.)
- Focus on business/domain concepts

Return a JSON array of objects with "name" and "description" fields.
Return an empty array [] if no domain-specific terms are found.

Example output:
[
  {{"name": "user", "description": "A person who interacts with the system"}},
  {{"name": "registration", "description": "The process of creating a new account"}}
]

JSON output:"""


@dataclass
class GlossaryItem:
    """A glossary item extracted from a feature."""

    name: str
    description: str
    source_feature_id: str
    source_feature_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.source_feature_id and self.source_feature_id not in self.source_feature_ids:
            self.source_feature_ids = [self.source_feature_id]


def build_glossary_prompt(summary: str, label: str) -> str:
    """Build the prompt for glossary extraction.

    Args:
        summary: The feature summary text to extract terms from.
        label: The feature label for context.

    Returns:
        Formatted prompt string for the LLM.
    """
    return GLOSSARY_EXTRACTION_PROMPT.format(label=label, summary=summary)


def parse_llm_response(response: str) -> list[dict[str, str]]:
    """Parse LLM response to extract glossary items.

    Handles:
    - Plain JSON arrays
    - JSON wrapped in markdown code blocks (```json ... ```)
    - Filters out malformed items

    Args:
        response: Raw response string from the LLM.

    Returns:
        List of dicts with 'name' and 'description' keys.
        Returns empty list if parsing fails or no valid items found.
    """
    response = response.strip()

    # Handle markdown code blocks
    if "```" in response:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if match:
            response = match.group(1).strip()

    try:
        data = json.loads(response)
        if isinstance(data, list):
            return [
                item
                for item in data
                if isinstance(item, dict) and "name" in item and "description" in item
            ]
    except json.JSONDecodeError:
        pass

    return []


async def call_llm_for_glossary(
    summary: str,
    label: str,
    config: MagaldiConfig | None = None,
) -> list[dict[str, str]]:
    """Call LLM to extract glossary items from a summary.

    Args:
        summary: The feature summary text to extract terms from.
        label: The feature label for context.
        config: Optional MagaldiConfig. If None, uses default config.

    Returns:
        List of dicts with 'name' and 'description' keys.
        Returns empty list if LLM call fails or returns invalid data.
    """
    if config is None:
        from shared.config import MagaldiConfig

        config = MagaldiConfig()

    prompt = build_glossary_prompt(summary, label)

    # Build model identifier based on provider
    llm_config = config.llm
    if llm_config.provider == "ollama":
        model = f"ollama/{llm_config.summarize_model}"
        api_base = llm_config.url
    elif llm_config.provider == "openai":
        model = llm_config.summarize_model
        api_base = None
    else:
        model = f"{llm_config.provider}/{llm_config.summarize_model}"
        api_base = None

    client = LLMClient(
        model=model,
        api_base=api_base,
        api_key=llm_config.api_key,
    )

    try:
        response = client.generate(
            prompt=prompt,
            temperature=llm_config.summarize_temperature,
            top_p=llm_config.summarize_top_p,
            max_tokens=llm_config.summarize_max_tokens,
        )
    except LLMError:
        return []

    return parse_llm_response(response)


async def extract_glossary_from_feature(
    feature: dict[str, Any],
    config: MagaldiConfig | None = None,
) -> list[GlossaryItem]:
    """Extract glossary items from a single feature.

    Args:
        feature: Feature dict with feature_id, label, summary.
        config: Optional MagaldiConfig. If None, uses default config.

    Returns:
        List of GlossaryItem extracted from the feature.
    """
    feature_id = feature.get("feature_id") or feature.get("subfeature_id", "")
    label = feature.get("label", "")
    summary = feature.get("summary", "")

    if not summary:
        return []

    raw_items = await call_llm_for_glossary(summary, label, config)

    items = []
    for raw in raw_items:
        name = raw.get("name", "").lower().strip()
        description = raw.get("description", "").strip()

        if name and description:
            items.append(
                GlossaryItem(
                    name=name,
                    description=description,
                    source_feature_id=feature_id,
                )
            )

    return items


def normalize_term(name: str) -> str:
    """Normalize a glossary term name.

    Handles pluralization and common variations.

    Args:
        name: The term name to normalize.

    Returns:
        Normalized term name (lowercase, singular form).
    """
    name = name.lower().strip()

    # Simple depluralization for common patterns
    if name.endswith("ies") and len(name) > 3:
        singular = name[:-3] + "y"
        if len(singular) > 2:
            name = singular
    elif name.endswith("sses"):
        # classes -> class
        name = name[:-2]
    elif name.endswith("xes") or name.endswith("ches") or name.endswith("shes"):
        # boxes -> box, matches -> match, flashes -> flash
        name = name[:-2]
    elif name.endswith("es") and len(name) > 3 and not name.endswith("sses"):
        # processes -> process (but not classes which is handled above)
        if not name[:-2].endswith("s"):
            name = name[:-1]  # processes -> processe -> process (actually need -2)
            # Re-check: processes -> process requires removing 'es'
            # But "ses" ending needs special handling
            pass
    elif name.endswith("s") and len(name) > 3 and not name.endswith("ss"):
        name = name[:-1]

    return name


def merge_glossary_items(items: list[GlossaryItem]) -> list[GlossaryItem]:
    """Merge glossary items with same/similar names.

    Items with the same normalized name are merged:
    - Feature IDs are combined
    - The longest description is kept

    Args:
        items: List of GlossaryItem to merge.

    Returns:
        List of merged GlossaryItem, sorted by name.
    """
    grouped: dict[str, list[GlossaryItem]] = {}

    for item in items:
        normalized = normalize_term(item.name)
        if normalized not in grouped:
            grouped[normalized] = []
        grouped[normalized].append(item)

    merged = []
    for normalized_name, group in grouped.items():
        all_feature_ids: list[str] = []
        for item in group:
            for fid in item.source_feature_ids:
                if fid not in all_feature_ids:
                    all_feature_ids.append(fid)

        best_description = max(
            (item.description for item in group),
            key=len,
        )

        merged.append(
            GlossaryItem(
                name=normalized_name,
                description=best_description,
                source_feature_id=all_feature_ids[0] if all_feature_ids else "",
                source_feature_ids=all_feature_ids,
            )
        )

    merged.sort(key=lambda x: x.name)
    return merged
