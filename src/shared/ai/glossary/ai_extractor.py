"""AI-powered glossary extraction from feature summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


async def call_llm_for_glossary(summary: str, label: str) -> list[dict[str, str]]:
    """Call LLM to extract glossary items from a summary.

    Args:
        summary: The feature summary text to extract terms from.
        label: The feature label for context.

    Returns:
        List of dicts with 'name' and 'description' keys.

    This is a placeholder - will be implemented in Task 2.
    """
    # Placeholder: parameters will be used by actual LLM implementation
    _ = summary, label
    return []


async def extract_glossary_from_feature(
    feature: dict[str, Any],
) -> list[GlossaryItem]:
    """Extract glossary items from a single feature.

    Args:
        feature: Feature dict with feature_id, label, summary.

    Returns:
        List of GlossaryItem extracted from the feature.
    """
    feature_id = feature.get("feature_id") or feature.get("subfeature_id", "")
    label = feature.get("label", "")
    summary = feature.get("summary", "")

    if not summary:
        return []

    raw_items = await call_llm_for_glossary(summary, label)

    items = []
    for raw in raw_items:
        name = raw.get("name", "").lower().strip()
        description = raw.get("description", "").strip()

        if name and description:
            items.append(GlossaryItem(
                name=name,
                description=description,
                source_feature_id=feature_id,
            ))

    return items
