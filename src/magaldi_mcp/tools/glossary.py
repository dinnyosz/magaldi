"""Glossary tools for domain term discovery."""

from __future__ import annotations

from typing import Any

from shared.db.store import Repository

from ._utils import _resolve_scope_repo


def list_glossary(
    repo: Repository,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    min_count: int = 1,
) -> list[dict[str, Any]]:
    """List all glossary terms for a repository.

    Glossary terms are domain concepts extracted from element names
    (e.g., 'user', 'email', 'order') that appear across the codebase.

    Args:
        repo: Search repository.
        scope: Repository scope (auto-detected from magaldi.yaml if not provided).
        repository: Repository name (auto-detected from magaldi.yaml if not provided).
        username: User branch (default: main).
        min_count: Minimum occurrence count to include (default: 1).

    Returns:
        List of glossary terms sorted by count, each with:
        - term: The domain term
        - total_count: Number of elements containing this term
        - description: AI-generated description explaining the term's meaning
        - file_paths: Files where term appears
        - feature_associations: Linked features (if glossary was linked)
    """
    # Auto-detect scope/repository from magaldi.yaml if not provided
    scope, repository = _resolve_scope_repo(scope, repository)
    if not scope or not repository:
        raise ValueError(
            "scope and repository are required. Either provide them explicitly "
            "or create a magaldi.yaml file in your project root."
        )
    return repo.get_glossary_terms(
        scope=scope,
        repository=repository,
        username=username,
        min_count=min_count,
    )


def get_glossary_term(
    repo: Repository,
    scope: str | None = None,
    repository: str | None = None,
    term: str = "",
    username: str = "main",
) -> dict[str, Any] | None:
    """Get full details for a specific glossary term.

    Args:
        repo: Search repository.
        scope: Repository scope (auto-detected from magaldi.yaml if not provided).
        repository: Repository name (auto-detected from magaldi.yaml if not provided).
        term: The glossary term to retrieve.
        username: User branch (default: main).

    Returns:
        Glossary entry with:
        - term: The domain term
        - total_count: Number of elements containing this term
        - description: AI-generated description explaining the term's meaning
        - element_ids: List of element IDs containing this term
        - file_paths: Files where term appears
        - feature_associations: Linked features with frequency/percentage
        Or None if term not found.
    """
    # Auto-detect scope/repository from magaldi.yaml if not provided
    scope, repository = _resolve_scope_repo(scope, repository)
    if not scope or not repository:
        raise ValueError(
            "scope and repository are required. Either provide them explicitly "
            "or create a magaldi.yaml file in your project root."
        )
    if not term:
        raise ValueError("term is required")
    return repo.get_glossary_term(
        scope=scope,
        repository=repository,
        term=term,
        username=username,
    )


def search_glossary(
    repo: Repository,
    scope: str | None = None,
    repository: str | None = None,
    query: str = "",
    username: str = "main",
) -> list[dict[str, Any]]:
    """Search glossary terms by partial match.

    Finds terms that contain the query string. For example,
    searching 'user' might return 'user', 'username', 'userid'.

    Args:
        repo: Search repository.
        scope: Repository scope (auto-detected from magaldi.yaml if not provided).
        repository: Repository name (auto-detected from magaldi.yaml if not provided).
        query: Partial term to search for.
        username: User branch (default: main).

    Returns:
        List of matching glossary terms with:
        - term: The domain term
        - total_count: Number of occurrences
        - description: AI-generated description explaining the term's meaning
    """
    # Auto-detect scope/repository from magaldi.yaml if not provided
    scope, repository = _resolve_scope_repo(scope, repository)
    if not scope or not repository:
        raise ValueError(
            "scope and repository are required. Either provide them explicitly "
            "or create a magaldi.yaml file in your project root."
        )
    if not query:
        raise ValueError("query is required")
    return repo.search_glossary(
        scope=scope,
        repository=repository,
        query=query,
        username=username,
    )
