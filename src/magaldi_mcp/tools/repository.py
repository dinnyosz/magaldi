"""Repository and feature discovery tools."""

from __future__ import annotations

from typing import Any

from shared.db.store import Repository

from ._utils import _resolve_scope_repo


def list_repos(
    repo: Repository,
    scope: str | None = None,
) -> list[dict[str, Any]]:
    """List all indexed repositories.

    Args:
        repo: Search repository.
        scope: Filter by scope (auto-detected from magaldi.yaml if not provided).

    Returns:
        List of repositories with statistics.
    """
    scope, _ = _resolve_scope_repo(scope, None)
    return repo.get_indexed_repositories(scope=scope)  # type: ignore[no-any-return]


def list_features(
    repo: Repository,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    brief: bool = True,
) -> list[dict[str, Any]]:
    """List all features and subfeatures for a repository.

    Args:
        repo: Search repository.
        scope: Repository scope (auto-detected from magaldi.yaml if not provided).
        repository: Repository name (auto-detected from magaldi.yaml if not provided).
        username: User branch.
        brief: Return only core fields (default True). Set False to include summaries.

    Returns:
        List of features and subfeatures with parent info.
    """
    scope, repository = _resolve_scope_repo(scope, repository)
    if not scope or not repository:
        raise ValueError(
            "scope and repository are required. Either provide them explicitly "
            "or create a magaldi.yaml file in your project root."
        )
    # Get features
    features = repo.get_features(scope, repository, username)
    for f in features:
        f["type"] = "feature"

    # Get subfeatures
    subfeatures = repo.get_subfeatures(scope, repository, username)
    for sf in subfeatures:
        sf["type"] = "subfeature"

    # Combine and sort by member count
    all_features = features + subfeatures
    all_features.sort(key=lambda x: x.get("member_count", 0), reverse=True)

    # In brief mode, remove summaries to save tokens
    if brief:
        for f in all_features:
            f.pop("summary", None)

    return all_features  # type: ignore[no-any-return]


def get_repo_stats(
    repo: Repository,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
) -> dict[str, Any]:
    """Get statistics for a repository.

    Args:
        repo: Search repository.
        scope: Repository scope (auto-detected from magaldi.yaml if not provided).
        repository: Repository name (auto-detected from magaldi.yaml if not provided).
        username: User branch.

    Returns:
        Repository statistics.
    """
    # Auto-detect scope/repository from magaldi.yaml if not provided
    scope, repository = _resolve_scope_repo(scope, repository)
    if not scope or not repository:
        raise ValueError(
            "scope and repository are required. Either provide them explicitly "
            "or create a magaldi.yaml file in your project root."
        )
    return repo.get_repository_stats(scope, repository, username)  # type: ignore[no-any-return]


def get_feature_members(
    repo: Repository,
    feature_id: str,
) -> dict[str, Any]:
    """Get all members of a feature or subfeature cluster.

    Args:
        repo: Search repository.
        feature_id: Feature or subfeature ID.

    Returns:
        Dict with 'members' list and 'glossary_terms' list.
    """
    # Get feature/subfeature document (supports both element_id and hash_id)
    feature = repo.get_document_by_id_or_hash(feature_id)
    if not feature:
        raise ValueError(f"Feature/subfeature not found: {feature_id}")

    member_ids = feature.get("member_ids", [])
    if not member_ids:
        return {"members": [], "glossary_terms": []}

    # Fetch member documents
    members = []
    for member_id in member_ids:
        doc = repo.get_document(member_id)
        if doc:
            members.append(
                {
                    "element_id": doc.get("element_id"),
                    "hash_id": doc.get("hash_id"),
                    "name": doc.get("name"),
                    "type": doc.get("element_type"),
                    "file": doc.get("relative_path"),
                    "line": doc.get("line_start"),
                    "summary": doc.get("summary", ""),
                    "signature": doc.get("signature", ""),
                }
            )

    # Parse feature_id to get scope, repository, username
    # Format: scope:repo:username:feature:N or scope:repo:username:subfeature:N
    parts = feature_id.split(":")
    glossary_terms = []
    if len(parts) >= 3:
        scope = parts[0]
        repository = parts[1]
        username = parts[2]

        # Get all glossary terms for this repo
        all_terms = repo.get_glossary_terms(scope, repository, username)

        # Filter to terms that have associations with this feature
        for term_entry in all_terms:
            for assoc in term_entry.get("feature_associations", []):
                if assoc.get("feature_id") == feature_id:
                    glossary_terms.append(
                        {
                            "term": term_entry.get("term"),
                            "frequency": assoc.get("frequency"),
                            "percentage": assoc.get("percentage"),
                        }
                    )
                    break  # Found association for this feature

    return {
        "members": members,
        "glossary_terms": sorted(
            glossary_terms,
            key=lambda x: x.get("percentage", 0),
            reverse=True,
        ),
    }
