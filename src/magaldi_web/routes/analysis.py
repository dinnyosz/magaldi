"""Analysis API routes for call graphs, dependencies, and code quality."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from magaldi_web.dependencies import get_es_repository
from magaldi_web.models import (
    CallerInfo,
    CalleeInfo,
    CallChainNode,
    CallChainResponse,
    CallGraphResponse,
    DeadCodeItem,
    DeadCodeResponse,
    DependenciesResponse,
    DependencyGraphResponse,
    DependentInfo,
    DependentsResponse,
    EntryPointItem,
    EntryPointsResponse,
    ExplainElementResponse,
    ImportInfo,
    SimilarCodeItem,
)
from shared.db.elasticsearch import ElasticsearchRepository

router = APIRouter()


# =============================================================================
# CALL ANALYSIS ROUTES
# =============================================================================


@router.get("/analysis/callers/{hash_id}", response_model=CallGraphResponse)
async def get_element_callers(
    hash_id: str = Path(..., description="Element hash ID"),
    limit: int = Query(default=30, ge=1, le=100),
    include_tests: bool = Query(default=True),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> CallGraphResponse:
    """Get callers and callees for an element."""
    source = es_repo.get_document_by_hash_id(hash_id)
    if not source:
        raise HTTPException(status_code=404, detail="Element not found")

    element_id = source["element_id"]
    scope = source["scope"]
    repository = source["repository"]
    username = source.get("username", "main")

    element = {
        "element_id": element_id,
        "hash_id": source.get("hash_id"),
        "name": source["name"],
        "element_type": source["element_type"],
        "file_path": source.get("relative_path"),
        "line": source.get("line_start"),
    }

    # Get callers
    callers: list[CallerInfo] = []
    caller_results = es_repo.find_elements_calling(
        target_id=element_id,
        scope=scope,
        repository=repository,
        username=username,
        limit=limit,
    )
    for c in caller_results:
        is_test = c.get("is_test", False)
        if not include_tests and is_test:
            continue
        callers.append(
            CallerInfo(
                element_id=c.get("element_id"),
                hash_id=c.get("hash_id"),
                name=c.get("name"),
                element_type=c.get("element_type"),
                file_path=c.get("relative_path", ""),
                line=c.get("line_start", 0),
                summary=c.get("summary"),
            )
        )

    # Get callees
    callees: list[CalleeInfo] = []
    calls = es_repo.get_calls(element_id)
    for call in calls:
        callee_entry = CalleeInfo(
            name=call.get("name"),
            receiver=call.get("receiver"),
            line=call.get("line"),
        )
        resolved_id = call.get("resolved_id")
        if resolved_id:
            resolved_doc = es_repo.get_document(resolved_id)
            if resolved_doc:
                callee_entry.element_id = resolved_id
                callee_entry.hash_id = resolved_doc.get("hash_id")
                callee_entry.element_type = resolved_doc.get("element_type")
                callee_entry.file_path = resolved_doc.get("relative_path")
                callee_entry.target_line = resolved_doc.get("line_start")
                callee_entry.summary = resolved_doc.get("summary")
        callees.append(callee_entry)

    return CallGraphResponse(element=element, callers=callers, callees=callees)


@router.get("/analysis/call-chain/{hash_id}", response_model=CallChainResponse)
async def get_call_chain(
    hash_id: str = Path(..., description="Element hash ID"),
    direction: str = Query(default="both", pattern="^(callers|callees|both)$"),
    max_depth: int = Query(default=3, ge=1, le=5),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> CallChainResponse:
    """Get call chain for an element."""
    from magaldi_mcp.tools import find_call_chain

    source = es_repo.get_document_by_hash_id(hash_id)
    if not source:
        raise HTTPException(status_code=404, detail="Element not found")

    element_id = source["element_id"]

    result = find_call_chain(
        es=es_repo,
        element_id=element_id,
        direction=direction,
        max_depth=max_depth,
    )

    def convert_nodes(nodes: list[dict]) -> list[CallChainNode]:
        return [
            CallChainNode(
                element_id=n.get("element_id"),
                hash_id=n.get("hash_id"),
                name=n.get("name", ""),
                element_type=n.get("type"),
                file_path=n.get("file"),
                line=n.get("line"),
                summary=n.get("summary"),
                callers=convert_nodes(n.get("callers", [])),
                callees=convert_nodes(n.get("callees", [])),
                cycle=n.get("cycle", False),
                unresolved=n.get("unresolved", False),
                missing=n.get("missing", False),
            )
            for n in nodes
        ]

    return CallChainResponse(
        root=result.get("root", {}),
        direction=result.get("direction", direction),
        max_depth=result.get("max_depth", max_depth),
        callers=convert_nodes(result.get("callers", [])),
        callees=convert_nodes(result.get("callees", [])),
    )


@router.get("/analysis/dead-code", response_model=DeadCodeResponse)
async def get_dead_code(
    scope: str = Query(..., description="Repository scope"),
    repository: str = Query(..., description="Repository name"),
    username: str = Query(default="main"),
    include_tests: bool = Query(default=False),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> DeadCodeResponse:
    """Find potentially dead code."""
    from magaldi_mcp.tools import find_dead_code

    result = find_dead_code(
        es=es_repo,
        scope=scope,
        repository=repository,
        username=username,
        include_tests=include_tests,
    )

    items = [
        DeadCodeItem(
            element_id=d.get("element_id"),
            hash_id=d.get("hash_id"),
            name=d.get("name"),
            element_type=d.get("type", ""),
            file_path=d.get("file", ""),
            line=d.get("line", 0),
            summary=d.get("summary"),
            is_test=d.get("is_test", False),
        )
        for d in result.get("potentially_dead", [])
    ]

    return DeadCodeResponse(potentially_dead=items, stats=result.get("stats", {}))


@router.get("/analysis/entry-points", response_model=EntryPointsResponse)
async def get_entry_points(
    scope: str = Query(..., description="Repository scope"),
    repository: str = Query(..., description="Repository name"),
    username: str = Query(default="main"),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> EntryPointsResponse:
    """Find entry points (HTTP handlers, CLI commands, etc.)."""
    from magaldi_mcp.tools import find_entry_points

    result = find_entry_points(
        es=es_repo,
        scope=scope,
        repository=repository,
        username=username,
    )

    def convert_entries(entries: list[dict]) -> list[EntryPointItem]:
        return [
            EntryPointItem(
                element_id=e.get("element_id"),
                hash_id=e.get("hash_id"),
                name=e.get("name"),
                element_type=e.get("type", ""),
                file_path=e.get("file", ""),
                line=e.get("line", 0),
                summary=e.get("summary"),
                decorators=e.get("decorators", []),
            )
            for e in entries
        ]

    return EntryPointsResponse(
        http=convert_entries(result.get("http", [])),
        cli=convert_entries(result.get("cli", [])),
        test=convert_entries(result.get("test", [])),
        main=convert_entries(result.get("main", [])),
        async_tasks=convert_entries(result.get("async_tasks", [])),
        stats=result.get("stats", {}),
    )


# =============================================================================
# DEPENDENCY ANALYSIS ROUTES
# =============================================================================


@router.get("/analysis/dependencies/{hash_id}", response_model=DependenciesResponse)
async def get_dependencies(
    hash_id: str = Path(..., description="File element hash ID"),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> DependenciesResponse:
    """Get imports for a file element."""
    from magaldi_mcp.tools import find_dependencies

    source = es_repo.get_document_by_hash_id(hash_id)
    if not source:
        raise HTTPException(status_code=404, detail="Element not found")

    if source.get("element_type") != "file":
        raise HTTPException(status_code=400, detail="Element is not a file")

    element_id = source["element_id"]

    result = find_dependencies(
        es=es_repo,
        element_id=element_id,
    )

    def convert_imports(imports: list[dict], is_internal: bool) -> list[ImportInfo]:
        return [
            ImportInfo(
                name=i.get("name", ""),
                module=i.get("module", ""),
                alias=i.get("alias"),
                line=i.get("line", 0),
                is_internal=is_internal,
            )
            for i in imports
        ]

    return DependenciesResponse(
        file_info=result.get("file_info", {}),
        internal_imports=convert_imports(result.get("internal_imports", []), True),
        external_imports=convert_imports(result.get("external_imports", []), False),
        stats=result.get("stats", {}),
    )


@router.get("/analysis/dependents", response_model=DependentsResponse)
async def get_dependents(
    module: str = Query(..., description="Module name to search for"),
    scope: str = Query(..., description="Repository scope"),
    repository: str = Query(..., description="Repository name"),
    username: str = Query(default="main"),
    limit: int = Query(default=50, ge=1, le=200),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> DependentsResponse:
    """Find files that import a module."""
    from magaldi_mcp.tools import find_dependents

    result = find_dependents(
        es=es_repo,
        module=module,
        scope=scope,
        repository=repository,
        username=username,
        limit=limit,
    )

    dependents = [
        DependentInfo(
            element_id=d.get("element_id"),
            hash_id=d.get("hash_id"),
            file_path=d.get("file", ""),
            name=d.get("name", ""),
        )
        for d in result.get("dependents", [])
    ]

    return DependentsResponse(
        module=module,
        dependents=dependents,
        total=result.get("total", len(dependents)),
    )


@router.get("/analysis/dependency-graph", response_model=DependencyGraphResponse)
async def get_dependency_graph(
    scope: str = Query(..., description="Repository scope"),
    repository: str = Query(..., description="Repository name"),
    username: str = Query(default="main"),
    internal_only: bool = Query(default=True),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> DependencyGraphResponse:
    """Get module dependency graph."""
    from magaldi_mcp.tools import dependency_graph

    result = dependency_graph(
        es=es_repo,
        scope=scope,
        repository=repository,
        username=username,
        internal_only=internal_only,
    )

    return DependencyGraphResponse(
        nodes=result.get("nodes", []),
        edges=result.get("edges", []),
        cycles=result.get("cycles", []),
        stats=result.get("stats", {}),
    )


# =============================================================================
# COMPREHENSIVE ANALYSIS ROUTES
# =============================================================================


@router.get("/analysis/explain/{hash_id}", response_model=ExplainElementResponse)
async def explain_element(
    hash_id: str = Path(..., description="Element hash ID"),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> ExplainElementResponse:
    """Get comprehensive explanation of an element."""
    from magaldi_mcp.tools import explain_element as explain_element_tool

    source = es_repo.get_document_by_hash_id(hash_id)
    if not source:
        raise HTTPException(status_code=404, detail="Element not found")

    element_id = source["element_id"]

    result = explain_element_tool(es=es_repo, element_id=element_id)

    # Convert callers
    callers = [
        CallerInfo(
            element_id=c.get("element_id"),
            hash_id=c.get("hash_id"),
            name=c.get("name"),
            element_type=c.get("type", ""),
            file_path=c.get("file", ""),
            line=c.get("line", 0),
            summary=c.get("summary"),
        )
        for c in result.get("callers", [])
    ]

    # Convert callees
    callees = [
        CalleeInfo(
            name=c.get("name"),
            receiver=c.get("receiver"),
            line=c.get("line", 0),
            element_id=c.get("element_id"),
            hash_id=c.get("hash_id"),
            element_type=c.get("type"),
            file_path=c.get("file"),
            target_line=c.get("target_line"),
            summary=c.get("summary"),
        )
        for c in result.get("callees", [])
    ]

    # Convert imports
    imports = [
        ImportInfo(
            name=i.get("name", ""),
            module=i.get("module", ""),
            alias=i.get("alias"),
            line=i.get("line", 0),
        )
        for i in result.get("imports", [])
    ]

    # Convert similar code
    similar_code = [
        SimilarCodeItem(
            element_id=s.get("element_id"),
            hash_id=s.get("hash_id"),
            name=s.get("name"),
            element_type=s.get("type", ""),
            file_path=s.get("file", ""),
            line=s.get("line", 0),
            summary=s.get("summary"),
            similarity=s.get("similarity", 0),
        )
        for s in result.get("similar_code", [])
    ]

    # Check embedding status
    embedding_status = {
        "has_summary": source.get("summary_embedding") is not None,
        "has_code": source.get("code_embedding") is not None,
    }

    return ExplainElementResponse(
        element=result.get("element", {}),
        callers=callers,
        callees=callees,
        imports=imports,
        similar_code=similar_code,
        parent=result.get("parent"),
        embedding_status=embedding_status,
    )
