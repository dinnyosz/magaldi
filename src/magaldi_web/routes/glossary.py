"""Glossary API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from magaldi_web.dependencies import get_es_repository
from magaldi_web.models import (
    FeatureAssociationSummary,
    GlossaryListResponse,
    GlossaryTermResponse,
    GlossaryTermSummary,
)
from shared.db.elasticsearch import ElasticsearchRepository

router = APIRouter()


@router.get("/repos/{scope}/{repository}/glossary", response_model=GlossaryListResponse)
async def list_glossary_terms(
    scope: str = Path(..., description="Repository scope"),
    repository: str = Path(..., description="Repository name"),
    username: str = Query(default="main", description="Username/branch"),
    min_count: int = Query(default=1, ge=1, description="Minimum occurrence count"),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> GlossaryListResponse:
    """List all glossary terms for a repository."""
    terms = es_repo.get_glossary_terms(scope, repository, username)

    # Filter by min_count
    filtered_terms = [t for t in terms if t.get("total_count", 0) >= min_count]

    # Sort by count descending
    filtered_terms.sort(key=lambda t: t.get("total_count", 0), reverse=True)

    return GlossaryListResponse(
        terms=[
            GlossaryTermSummary(
                term=t.get("term", ""),
                total_count=t.get("total_count", 0),
                file_count=len(t.get("file_paths", [])),
                feature_count=len(t.get("feature_associations", [])),
            )
            for t in filtered_terms
        ],
        total=len(filtered_terms),
    )


@router.get(
    "/repos/{scope}/{repository}/glossary/{term}", response_model=GlossaryTermResponse
)
async def get_glossary_term(
    scope: str = Path(..., description="Repository scope"),
    repository: str = Path(..., description="Repository name"),
    term: str = Path(..., description="Glossary term"),
    username: str = Query(default="main", description="Username/branch"),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> GlossaryTermResponse:
    """Get detailed information about a glossary term."""
    entry = es_repo.get_glossary_term(scope, repository, term, username)

    if not entry:
        raise HTTPException(status_code=404, detail=f"Glossary term '{term}' not found")

    return GlossaryTermResponse(
        term=entry.get("term", ""),
        total_count=entry.get("total_count", 0),
        element_ids=entry.get("element_ids", []),
        file_paths=entry.get("file_paths", []),
        feature_associations=[
            FeatureAssociationSummary(
                feature_id=a.get("feature_id", ""),
                feature_label=a.get("feature_label", ""),
                frequency=a.get("frequency", 0),
                total_members=a.get("total_members", 0),
                percentage=a.get("percentage", 0.0),
            )
            for a in entry.get("feature_associations", [])
        ],
    )


@router.get(
    "/repos/{scope}/{repository}/glossary/search/{query}",
    response_model=GlossaryListResponse,
)
async def search_glossary_terms(
    scope: str = Path(..., description="Repository scope"),
    repository: str = Path(..., description="Repository name"),
    query: str = Path(..., description="Search query (partial match)"),
    username: str = Query(default="main", description="Username/branch"),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> GlossaryListResponse:
    """Search glossary terms by partial match."""
    terms = es_repo.search_glossary(scope, repository, query, username)

    return GlossaryListResponse(
        terms=[
            GlossaryTermSummary(
                term=t.get("term", ""),
                total_count=t.get("total_count", 0),
                file_count=len(t.get("file_paths", [])),
                feature_count=len(t.get("feature_associations", [])),
            )
            for t in terms
        ],
        total=len(terms),
    )
