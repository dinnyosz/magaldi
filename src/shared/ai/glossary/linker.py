# src/shared/ai/glossary/linker.py
"""Feature-glossary linking logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shared.ai.glossary.extractor import extract_terms


@dataclass
class FeatureAssociation:
    """Association between a glossary term and a feature."""

    term: str
    feature_id: str
    feature_label: str
    frequency: int
    total_members: int
    percentage: float


def compute_feature_associations(
    feature: dict[str, Any],
    element_names: dict[str, str],
    glossary_terms: set[str],
) -> list[FeatureAssociation]:
    """Compute glossary associations for a feature.

    Args:
        feature: Feature dict with 'feature_id', 'label', 'member_ids'.
        element_names: Dict mapping element_id to element name.
        glossary_terms: Set of valid glossary terms.

    Returns:
        List of FeatureAssociation sorted by frequency descending.
    """
    feature_id = feature.get("feature_id", "")
    feature_label = feature.get("label", "")
    member_ids = feature.get("member_ids", [])

    if not member_ids:
        return []

    total_members = len(member_ids)

    # Count occurrences of each glossary term across members
    term_counts: dict[str, int] = {}

    for member_id in member_ids:
        name = element_names.get(member_id, "")
        terms = extract_terms(name)

        for term in terms:
            # Only count if term is in the glossary
            if term in glossary_terms:
                term_counts[term] = term_counts.get(term, 0) + 1

    # Build associations
    associations: list[FeatureAssociation] = []

    for term, frequency in term_counts.items():
        percentage = (frequency / total_members) * 100

        associations.append(FeatureAssociation(
            term=term,
            feature_id=feature_id,
            feature_label=feature_label,
            frequency=frequency,
            total_members=total_members,
            percentage=round(percentage, 1),
        ))

    # Sort by frequency descending
    associations.sort(key=lambda a: a.frequency, reverse=True)

    return associations


def link_glossary_to_features(
    glossary_terms: set[str],
    features: list[dict[str, Any]],
    element_names: dict[str, str],
) -> dict[str, list[FeatureAssociation]]:
    """Link glossary terms to features.

    Args:
        glossary_terms: Set of glossary terms.
        features: List of feature dicts.
        element_names: Dict mapping element_id to element name.

    Returns:
        Dict mapping glossary term to list of feature associations.
    """
    term_to_features: dict[str, list[FeatureAssociation]] = {
        term: [] for term in glossary_terms
    }

    for feature in features:
        associations = compute_feature_associations(
            feature, element_names, glossary_terms
        )

        for assoc in associations:
            term_to_features[assoc.term].append(assoc)

    return term_to_features
