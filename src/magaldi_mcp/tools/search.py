"""Search-related MCP tools."""

from __future__ import annotations

import logging
from typing import Any

from shared.ai.embedding import CodeEmbeddingClient
from shared.db.store import Repository

from ._utils import _resolve_scope_repo

logger = logging.getLogger(__name__)


def search_code(
    repo: Repository,
    embed_client: CodeEmbeddingClient | None,
    query: str,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    element_types: list[str] | None = None,
    language: str | None = None,
    limit: int = 20,
    include_code: bool = False,
    brief: bool = False,
    include_tests: bool = True,
) -> dict[str, Any]:
    """Semantic search for code elements.

    Tries vector search first, falls back to keyword search if Ollama unavailable.

    Args:
        repo: Search repository.
        embed_client: Embedding client for query (optional, falls back to keyword).
        query: Natural language search query.
        scope: Filter by scope.
        repository: Filter by repository.
        username: User branch to search.
        element_types: Filter by element types.
        language: Filter by programming language.
        limit: Maximum results.
        include_code: Include source code in results (for detailed inspection).
        brief: Minimal output - just name, type, file, line (for exploration).
        include_tests: Include test results (default True).

    Returns:
        Dict with code_results and test_results lists, grouped by is_test field.
    """
    # Auto-detect scope/repository from magaldi.yaml if not provided
    scope, repository = _resolve_scope_repo(scope, repository)

    # Validate limit
    limit = max(1, min(limit, 50))

    # Try vector search first, fall back to keyword search
    results = []
    if embed_client is not None:
        try:
            query_embedding = embed_client.embed_single(query)
            results = repo.search_by_vector(
                embedding=query_embedding,
                scope=scope,
                repository=repository,
                username=username,
                element_types=element_types,
                size=limit,
            )
        except Exception:
            logger.debug("Vector search failed, falling back to keyword search", exc_info=True)

    # Fallback to keyword search if vector search failed or unavailable
    if not results:
        results = repo.search_by_keyword(
            query=query,
            scope=scope,
            repository=repository,
            username=username,
            element_types=element_types,
            size=limit,
        )

    # Group results by is_test
    code_results = []
    test_results = []

    for result in results:
        # Filter by language if specified
        if language and result.get("language") != language:
            continue

        is_test = result.get("is_test", False)

        # Skip tests if not included
        if is_test and not include_tests:
            continue

        # Build qualified name: ClassName.method_name for methods
        name = result.get("name")
        if result.get("element_type") == "method" and result.get("parent_id"):
            # Try to get parent class name from parent_id
            parent_doc = repo.get_document(result["parent_id"])
            if parent_doc and parent_doc.get("element_type") == "class":
                name = f"{parent_doc.get('name')}.{name}"

        entry: dict[str, Any] = {
            "name": name,
            "type": result.get("element_type"),
            "file": result.get("relative_path"),
            "line": result.get("line_start"),
            "line_end": result.get("line_end"),
            "score": round(result.get("_score", 0.0), 2),
            "element_id": result.get("element_id"),
            "hash_id": result.get("hash_id", "")[:8],
            "is_test": is_test,
        }

        # Brief mode: just the basics for exploration
        if not brief:
            entry["summary"] = result.get("summary", "")
            # Only include signature if present and non-empty
            sig = result.get("signature")
            if sig:
                entry["signature"] = sig
            # Include code if requested
            if include_code and result.get("raw_code"):
                entry["code"] = result["raw_code"]

        if is_test:
            test_results.append(entry)
        else:
            code_results.append(entry)

    return {
        "code_results": code_results[:limit],
        "test_results": test_results[:limit] if include_tests else [],
        "total_code": len(code_results),
        "total_tests": len(test_results) if include_tests else 0,
    }


def search_features(
    repo: Repository,
    embed_client: CodeEmbeddingClient | None,
    query: str,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    limit: int = 20,
    glossary_term: str | None = None,
    min_percentage: float = 0.0,
) -> list[dict[str, Any]]:
    """Search for features/capabilities.

    Tries vector search first, falls back to keyword search if Ollama unavailable.

    Args:
        repo: Search repository.
        embed_client: Embedding client for query (optional, falls back to keyword).
        query: Search query for features.
        scope: Filter by scope.
        repository: Filter by repository.
        username: User branch.
        limit: Maximum results.
        glossary_term: Optional term to filter by. Only returns features where this
            term appears in the feature's members.
        min_percentage: Minimum percentage of feature members that must contain the
            glossary_term (0-100). Default 0 means any occurrence.

    Returns:
        List of matching features.
    """
    scope, repository = _resolve_scope_repo(scope, repository)
    limit = max(1, min(limit, 50))

    # Try vector search first, fall back to keyword search
    results = []
    if embed_client is not None:
        try:
            query_embedding = embed_client.embed_single(query)
            results = repo.search_by_vector(
                embedding=query_embedding,
                scope=scope,
                repository=repository,
                username=username,
                element_types=["feature", "subfeature"],
                size=limit,
            )
        except Exception:
            logger.debug("Vector search for features failed, falling back to keyword search", exc_info=True)

    # Fallback to keyword search if vector search failed or unavailable
    if not results:
        results = repo.search_by_keyword(
            query=query,
            scope=scope,
            repository=repository,
            username=username,
            element_types=["feature", "subfeature"],
            size=limit,
        )

    # Apply glossary term filter if provided
    if glossary_term is not None:
        glossary_entry = repo.get_glossary_term(scope, repository, glossary_term, username)
        if glossary_entry is None:
            # Term not found, return empty results
            return []

        # Build set of feature_ids that meet the min_percentage threshold
        feature_associations = glossary_entry.get("feature_associations", [])
        valid_feature_ids = {
            assoc["feature_id"]
            for assoc in feature_associations
            if assoc.get("percentage", 0) >= min_percentage
        }

        # Filter results to only include features in the valid set
        results = [r for r in results if r.get("element_id") in valid_feature_ids]

    formatted = []
    for result in results:
        entry = {
            "label": result.get("cluster_label", result.get("name")),
            "summary": result.get("summary", ""),
            "member_count": result.get("member_count", 0),
            "feature_id": result.get("element_id"),  # For follow-up queries
            "type": result.get("element_type", "feature"),  # "feature" or "subfeature"
        }
        # Include parent feature info for subfeatures
        if result.get("element_type") == "subfeature":
            entry["parent_feature_label"] = result.get("parent_feature_label", "")
            entry["parent_feature_summary"] = result.get("parent_feature_summary", "")
        formatted.append(entry)

    return formatted


def find_similar(
    repo: Repository,
    element_id: str,
    limit: int = 10,
    same_repo_only: bool = False,
    include_tests: bool = True,
) -> dict[str, Any]:
    """Find code elements similar to a given element.

    Args:
        repo: Search repository.
        element_id: Source element ID.
        limit: Maximum results.
        same_repo_only: Only search within same repository.
        include_tests: Include test results (default True).

    Returns:
        Dict with code_results and test_results lists, grouped by is_test field.
    """
    limit = max(1, min(limit, 50))

    # Get source element (supports both element_id and hash_id)
    doc = repo.get_document_by_id_or_hash(element_id)
    if not doc:
        raise ValueError(f"Element not found: {element_id}")

    # Try summary_embedding first, fall back to code_embedding
    embedding = doc.get("summary_embedding") or doc.get("code_embedding")
    if not embedding:
        raise ValueError(f"Element has no embedding: {element_id}")

    # Build search filters
    scope = doc.get("scope") if same_repo_only else None
    repository = doc.get("repository") if same_repo_only else None

    # Search excluding self
    results = repo.search_by_vector(
        embedding=embedding,
        scope=scope,
        repository=repository,
        username=doc.get("username", "main"),
        size=limit + 1,  # Get one extra to filter out self
    )

    # Group results by is_test
    code_results = []
    test_results = []

    for result in results:
        # Skip self
        if result.get("element_id") == element_id:
            continue

        is_test = result.get("is_test", False)

        # Skip tests if not included
        if is_test and not include_tests:
            continue

        entry: dict[str, Any] = {
            "name": result.get("name"),
            "type": result.get("element_type"),
            "file": result.get("relative_path"),
            "line": result.get("line_start"),
            "summary": result.get("summary", ""),
            "element_id": result.get("element_id"),
            "hash_id": result.get("hash_id"),
            "is_test": is_test,
        }

        if is_test:
            test_results.append(entry)
        else:
            code_results.append(entry)

    return {
        "code_results": code_results[:limit],
        "test_results": test_results[:limit] if include_tests else [],
        "total_code": len(code_results),
        "total_tests": len(test_results) if include_tests else 0,
    }
