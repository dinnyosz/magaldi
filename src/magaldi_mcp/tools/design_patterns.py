"""Design pattern detection tools."""

from __future__ import annotations

from typing import Any

from shared.db.store import Repository
from shared.db.repositories.base import INDEX_NAME


def list_patterns(
    repo: Repository,
    scope: str,
    repository: str,
    username: str = "main",
) -> dict[str, Any]:
    """List all detected design patterns in a repository.

    Returns pattern types (singleton, builder, factory, repository) with
    counts and example classes.

    Args:
        repo: Search repository.
        scope: Repository scope (required).
        repository: Repository name (required).
        username: User branch.

    Returns:
        Dict with:
        - patterns: List of pattern summaries with count and examples
        - total_classes_with_patterns: Total count of classes with patterns
    """
    must_clauses = [
        {"term": {"username": username}},
        {"term": {"scope": scope}},
        {"term": {"repository": repository}},
        {"exists": {"field": "detected_patterns"}},
        {"term": {"element_type": "class"}},
    ]

    client = repo._get_client()

    # Get pattern counts via aggregation
    agg_result = client.search(
        index=INDEX_NAME,
        body={
            "query": {"bool": {"must": must_clauses}},
            "size": 0,
            "aggs": {
                "patterns": {
                    "terms": {
                        # Use .keyword subfield for aggregations on text fields
                        "field": "detected_patterns.keyword",
                        "size": 100,
                    }
                }
            },
        },
    )

    pattern_counts = {
        bucket["key"]: bucket["doc_count"]
        for bucket in agg_result.get("aggregations", {}).get("patterns", {}).get("buckets", [])
    }

    # Get example classes for each pattern
    patterns_summary = []
    total_classes = 0

    for pattern_name, count in pattern_counts.items():
        total_classes += count

        # Get a few example classes for this pattern
        examples_result = client.search(
            index=INDEX_NAME,
            body={
                "query": {
                    "bool": {"must": must_clauses + [{"term": {"detected_patterns": pattern_name}}]}
                },
                "size": 3,
                "_source": ["name", "relative_path", "line_start", "pattern_confidence"],
            },
        )

        examples = []
        for hit in examples_result.get("hits", {}).get("hits", []):
            source = hit["_source"]
            confidence = source.get("pattern_confidence", {}).get(pattern_name, 0)
            examples.append(
                {
                    "name": source.get("name"),
                    "file": source.get("relative_path"),
                    "line": source.get("line_start"),
                    "confidence": confidence,
                }
            )

        patterns_summary.append(
            {
                "pattern": pattern_name,
                "count": count,
                "examples": examples,
            }
        )

    return {
        "patterns": patterns_summary,
        "total_classes_with_patterns": total_classes,
    }


def find_by_pattern(
    repo: Repository,
    pattern: str,
    scope: str,
    repository: str,
    username: str = "main",
    min_confidence: float = 0.6,
    limit: int = 20,
) -> dict[str, Any]:
    """Find all classes implementing a specific design pattern.

    Supports: singleton, builder, factory, repository.

    Args:
        repo: Search repository.
        pattern: Pattern type to search for.
        scope: Repository scope (required).
        repository: Repository name (required).
        username: User branch.
        min_confidence: Minimum confidence score (0.0-1.0).
        limit: Maximum results.

    Returns:
        Dict with:
        - classes: List of matching classes with confidence scores
        - count: Total number of matches
        - pattern: The pattern searched for
    """
    # Validate limit
    limit = max(1, min(limit, 100))

    must_clauses = [
        {"term": {"username": username}},
        {"term": {"scope": scope}},
        {"term": {"repository": repository}},
        {"term": {"detected_patterns": pattern}},
        {"term": {"element_type": "class"}},
    ]

    client = repo._get_client()
    result = client.search(
        index=INDEX_NAME,
        body={
            "query": {"bool": {"must": must_clauses}},
            "size": limit * 2,  # Fetch extra for confidence filtering
            "_source": [
                "element_id",
                "hash_id",
                "name",
                "relative_path",
                "line_start",
                "detected_patterns",
                "pattern_confidence",
                "summary",
            ],
        },
    )

    classes = []
    for hit in result.get("hits", {}).get("hits", []):
        source = hit["_source"]
        confidence = source.get("pattern_confidence", {}).get(pattern, 0)

        # Filter by minimum confidence
        if confidence < min_confidence:
            continue

        classes.append(
            {
                "element_id": source.get("element_id"),
                "hash_id": source.get("hash_id"),
                "name": source.get("name"),
                "file": source.get("relative_path"),
                "line": source.get("line_start"),
                "confidence": confidence,
                "summary": source.get("summary"),
            }
        )

        if len(classes) >= limit:
            break

    # Sort by confidence descending
    classes.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "pattern": pattern,
        "classes": classes,
        "count": len(classes),
    }
