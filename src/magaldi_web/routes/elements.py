"""Elements API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from magaldi_web.dependencies import get_es_repository
from magaldi_web.models import (
    CallInfo,
    ChildInfo,
    ClassAttributeInfo,
    CliCommandInfo,
    CodeMetricsInfo,
    CommentInfo,
    ComplexityInfo,
    ConcurrencyInfo,
    DecoratorDetailInfo,
    DocstringQualityInfo,
    ElementContext,
    ElementDetailResponse,
    EnvVarInfo,
    FeatureInfo,
    FeatureMember,
    FileContext,
    GlossaryFeatureAssociation,
    GlossaryInfo,
    HttpRouteInfo,
    ImportInfo,
    MetricsSummaryInfo,
    ParameterInfo,
    ParentContext,
    ParentFeatureInfo,
    PurityInfo,
    RepoRef,
    SectionMarkerInfo,
    SecurityIssueInfo,
    SiblingInfo,
    SideEffectInfo,
    SubfeatureInfo,
    TodoInfo,
    TypeAnnotationInfo,
)
from shared.db.elasticsearch import ElasticsearchRepository, INDEX_NAME

router = APIRouter()


@router.get("/elements/similar/{identifier}")
async def get_similar_elements(
    identifier: str = Path(..., description="Element hash_id (64-char SHA256) or element_id"),
    limit: int = Query(default=10, ge=1, le=50),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> list[dict]:
    """Find similar elements using vector similarity."""
    # Try hash_id first (64 hex characters), then fall back to element_id
    source = None
    if len(identifier) == 64 and all(c in "0123456789abcdef" for c in identifier.lower()):
        source = es_repo.get_document_by_hash_id(identifier)

    if not source:
        source = es_repo.get_document(identifier)

    if not source:
        raise HTTPException(status_code=404, detail="Element not found")

    element_id = source["element_id"]
    embedding = source.get("summary_embedding")

    if not embedding:
        raise HTTPException(status_code=400, detail="Element has no embedding")

    # Find similar elements
    client = es_repo._get_client()
    similar_result = client.search(
        index=INDEX_NAME,
        body={
            "size": limit + 1,  # +1 to exclude self
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": source["scope"]}},
                        {"term": {"repository": source["repository"]}},
                        {"term": {"username": source["username"]}},
                        {"exists": {"field": "summary_embedding"}},
                    ],
                    "must": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.qv, 'summary_embedding') + 1.0",
                                "params": {"qv": embedding},
                            },
                        },
                    },
                },
            },
            "_source": [
                "element_id",
                "hash_id",
                "name",
                "element_type",
                "relative_path",
                "line_start",
                "summary",
            ],
        },
    )

    similar = []
    for hit in similar_result.get("hits", {}).get("hits", []):
        s = hit["_source"]
        if s["element_id"] == element_id:
            continue  # Skip self
        similar.append(
            {
                "element_id": s["element_id"],
                "hash_id": s.get("hash_id"),
                "name": s["name"],
                "element_type": s["element_type"],
                "file_path": s["relative_path"],
                "line": s["line_start"],
                "summary": s.get("summary"),
                "similarity": round((hit["_score"] - 1.0), 3),  # Convert back to cosine similarity
            }
        )
        if len(similar) >= limit:
            break

    return similar


@router.get("/elements/{identifier}", response_model=ElementDetailResponse)
async def get_element_detail(
    identifier: str = Path(..., description="Element hash_id (64-char SHA256) or element_id"),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> ElementDetailResponse:
    """Get detailed information about a code element."""
    # Try hash_id first (64 hex characters), then fall back to element_id
    source = None
    if len(identifier) == 64 and all(c in "0123456789abcdef" for c in identifier.lower()):
        source = es_repo.get_document_by_hash_id(identifier)

    if not source:
        # Try as element_id
        source = es_repo.get_document(identifier)

    if not source:
        raise HTTPException(status_code=404, detail="Element not found")

    element_id = source["element_id"]
    client = es_repo._get_client()

    # Get file context (only for code elements that have a relative_path)
    file_context = None
    if source["element_type"] not in ("file", "feature", "subfeature") and source.get("relative_path"):
        file_result = client.search(
            index=INDEX_NAME,
            body={
                "size": 1,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"scope": source["scope"]}},
                            {"term": {"repository": source["repository"]}},
                            {"term": {"username": source["username"]}},
                            {"term": {"relative_path": source["relative_path"]}},
                            {"term": {"element_type": "file"}},
                        ],
                    },
                },
                "_source": ["element_id", "hash_id", "name", "summary"],
            },
        )
        file_hits = file_result.get("hits", {}).get("hits", [])
        if file_hits:
            fs = file_hits[0]["_source"]
            file_context = FileContext(
                element_id=fs["element_id"],
                hash_id=fs.get("hash_id"),
                name=fs["name"],
                summary=fs.get("summary"),
            )

    # Get parent context
    parent_context = None
    parent_id = source.get("parent_id")
    if parent_id:
        parent_result = client.search(
            index=INDEX_NAME,
            body={
                "size": 1,
                "query": {"term": {"element_id": parent_id}},
                "_source": ["element_id", "hash_id", "name", "element_type", "summary"],
            },
        )
        parent_hits = parent_result.get("hits", {}).get("hits", [])
        if parent_hits:
            ps = parent_hits[0]["_source"]
            parent_context = ParentContext(
                element_id=ps["element_id"],
                hash_id=ps.get("hash_id"),
                name=ps["name"],
                element_type=ps["element_type"],
                summary=ps.get("summary"),
            )

    # Get children
    children = []
    children_result = client.search(
        index=INDEX_NAME,
        body={
            "size": 100,
            "query": {"term": {"parent_id": element_id}},
            "_source": ["element_id", "hash_id", "name", "element_type", "line_start", "summary", "signature"],
            "sort": [{"line_start": "asc"}],
        },
    )
    for hit in children_result.get("hits", {}).get("hits", []):
        cs = hit["_source"]
        children.append(
            ChildInfo(
                element_id=cs["element_id"],
                hash_id=cs.get("hash_id"),
                name=cs["name"],
                element_type=cs["element_type"],
                line=cs["line_start"],
                summary=cs.get("summary"),
                signature=cs.get("signature"),
            )
        )

    # Get siblings (if has parent)
    siblings = []
    if parent_id:
        siblings_result = client.search(
            index=INDEX_NAME,
            body={
                "size": 50,
                "query": {
                    "bool": {
                        "filter": [{"term": {"parent_id": parent_id}}],
                        "must_not": [{"term": {"element_id": element_id}}],
                    },
                },
                "_source": ["element_id", "hash_id", "name", "element_type", "line_start", "summary"],
                "sort": [{"line_start": "asc"}],
            },
        )
        for hit in siblings_result.get("hits", {}).get("hits", []):
            ss = hit["_source"]
            siblings.append(
                SiblingInfo(
                    element_id=ss["element_id"],
                    hash_id=ss.get("hash_id"),
                    name=ss["name"],
                    element_type=ss["element_type"],
                    line=ss["line_start"],
                    summary=ss.get("summary"),
                )
            )

    # Get decorators (already a list in ES, but handle string for backwards compat)
    decorators = []
    raw_decorators = source.get("decorators")
    if raw_decorators:
        if isinstance(raw_decorators, list):
            decorators = raw_decorators
        elif isinstance(raw_decorators, str):
            import json
            try:
                decorators = json.loads(raw_decorators)
            except (json.JSONDecodeError, TypeError):
                decorators = []

    # Get feature info for features/subfeatures
    feature_info = None
    if source["element_type"] in ("feature", "subfeature"):
        member_ids = source.get("member_ids", [])
        members = []

        # Fetch member element details (limit to first 50 for performance)
        if member_ids:
            members_result = client.mget(
                index=INDEX_NAME,
                ids=member_ids[:50],
                _source=["element_id", "hash_id", "name", "element_type", "relative_path", "line_start", "summary", "signature"],
            )
            for doc in members_result.get("docs", []):
                if doc.get("found") and doc.get("_source"):
                    ms = doc["_source"]
                    members.append(
                        FeatureMember(
                            element_id=ms["element_id"],
                            hash_id=ms.get("hash_id"),
                            name=ms["name"],
                            element_type=ms["element_type"],
                            file_path=ms.get("relative_path", ""),
                            line=ms.get("line_start", 0),
                            summary=ms.get("summary"),
                            signature=ms.get("signature"),
                        )
                    )

        # Get parent feature info for subfeatures
        parent_feature = None
        if source["element_type"] == "subfeature":
            parent_label = source.get("parent_feature_label")
            if parent_label:
                parent_feature = ParentFeatureInfo(
                    label=parent_label,
                    summary=source.get("parent_feature_summary"),
                )

        # Get subfeatures for features
        subfeatures = []
        if source["element_type"] == "feature":
            feature_label = source.get("cluster_label") or source.get("name")
            if feature_label:
                # Query subfeatures by parent_feature_label
                subfeatures_result = client.search(
                    index=INDEX_NAME,
                    body={
                        "size": 50,
                        "query": {
                            "bool": {
                                "filter": [
                                    {"term": {"scope": source["scope"]}},
                                    {"term": {"repository": source["repository"]}},
                                    {"term": {"username": source.get("username", "main")}},
                                    {"term": {"element_type": "subfeature"}},
                                    {"term": {"parent_feature_label": feature_label}},
                                ],
                            },
                        },
                        "_source": ["element_id", "hash_id", "cluster_label", "summary", "member_count"],
                        "sort": [{"member_count": "desc"}],
                    },
                )
                for hit in subfeatures_result.get("hits", {}).get("hits", []):
                    sf = hit["_source"]
                    subfeatures.append(
                        SubfeatureInfo(
                            element_id=sf["element_id"],
                            hash_id=sf.get("hash_id"),
                            label=sf.get("cluster_label", ""),
                            summary=sf.get("summary"),
                            member_count=sf.get("member_count", 0),
                        )
                    )

        feature_info = FeatureInfo(
            member_count=source.get("member_count", len(member_ids)),
            members=members,
            parent_feature=parent_feature,
            subfeatures=subfeatures,
        )

    # Get glossary info for glossary elements
    glossary_info = None
    if source["element_type"] == "glossary":
        raw_associations = source.get("feature_associations", [])
        feature_ids = [assoc.get("feature_id") for assoc in raw_associations if assoc.get("feature_id")]

        # Fetch full feature details
        feature_details: dict[str, dict] = {}
        if feature_ids:
            features_result = client.mget(
                index=INDEX_NAME,
                ids=feature_ids,
                _source=["element_id", "hash_id", "cluster_label", "summary", "member_count"],
            )
            for doc in features_result.get("docs", []):
                if doc.get("found") and doc.get("_source"):
                    feature_details[doc["_id"]] = doc["_source"]

        feature_associations = []
        for assoc in raw_associations:
            feature_id = assoc.get("feature_id", "")
            details = feature_details.get(feature_id, {})
            feature_associations.append(
                GlossaryFeatureAssociation(
                    feature_id=feature_id,
                    hash_id=details.get("hash_id"),
                    feature_label=details.get("cluster_label") or assoc.get("feature_label", ""),
                    summary=details.get("summary"),
                    member_count=details.get("member_count", 0),
                )
            )

        glossary_info = GlossaryInfo(
            description=source.get("description", ""),
            total_count=source.get("total_count", 0),
            feature_count=len(feature_associations),
            file_paths=source.get("file_paths", []),
            feature_associations=feature_associations,
        )

    # Parse class attributes
    class_attributes = []
    raw_class_attrs = source.get("class_attributes", [])
    if raw_class_attrs:
        for attr in raw_class_attrs:
            class_attributes.append(
                ClassAttributeInfo(
                    name=attr.get("name", ""),
                    type=attr.get("type"),
                    line=attr.get("line"),
                )
            )

    # Parse imports (for file elements)
    imports = []
    raw_imports = source.get("imports", [])
    if raw_imports:
        for imp in raw_imports:
            imports.append(
                ImportInfo(
                    name=imp.get("name", ""),
                    module=imp.get("module", ""),
                    alias=imp.get("alias"),
                    line=imp.get("line", 0),
                    is_internal=imp.get("is_internal", False),
                )
            )

    # Parse decorator details (rich decorator info with args)
    decorator_details = []
    raw_decorator_details = source.get("decorator_details", [])
    if raw_decorator_details:
        for dec in raw_decorator_details:
            decorator_details.append(
                DecoratorDetailInfo(
                    name=dec.get("name", ""),
                    args=dec.get("args"),
                    full=dec.get("full"),
                )
            )

    # Parse parameters (for functions/methods)
    parameters = []
    raw_parameters = source.get("parameters", [])
    if raw_parameters:
        for param in raw_parameters:
            parameters.append(
                ParameterInfo(
                    name=param.get("name", ""),
                    type=param.get("type"),
                    default=param.get("default"),
                )
            )

    # Parse calls (for functions/methods)
    calls = []
    raw_calls = source.get("calls", [])
    if raw_calls:
        for call in raw_calls:
            calls.append(
                CallInfo(
                    name=call.get("name", ""),
                    receiver=call.get("receiver"),
                    line=call.get("line", 0),
                    resolved_id=call.get("resolved_id"),
                )
            )

    # For glossary elements, use description as summary
    summary = source.get("summary")
    if source["element_type"] == "glossary":
        summary = source.get("description", "")

    # Parse code metrics for functions/methods
    complexity = None
    code_metrics_info = None
    docstring_quality = None
    concurrency = None
    security_issues = []
    env_vars = []

    if source["element_type"] in ("function", "method"):
        # Complexity
        raw_complexity = source.get("complexity")
        if raw_complexity:
            complexity = ComplexityInfo(
                cyclomatic=raw_complexity.get("cyclomatic", 0),
                nesting_depth=raw_complexity.get("nesting_depth", 0),
                branch_count=raw_complexity.get("branch_count", 0),
            )

        # Code metrics
        raw_code_metrics = source.get("code_metrics")
        if raw_code_metrics:
            code_metrics_info = CodeMetricsInfo(
                line_count=raw_code_metrics.get("line_count", 0),
                param_count=raw_code_metrics.get("param_count", 0),
                char_count=raw_code_metrics.get("char_count", 0),
            )

        # Docstring quality
        raw_docstring_quality = source.get("docstring_quality")
        if raw_docstring_quality:
            docstring_quality = DocstringQualityInfo(
                has_docstring=raw_docstring_quality.get("has_docstring", False),
                has_params=raw_docstring_quality.get("has_params", False),
                has_return=raw_docstring_quality.get("has_return", False),
                coverage=raw_docstring_quality.get("coverage", 0.0),
            )

        # Concurrency
        raw_concurrency = source.get("concurrency")
        if raw_concurrency:
            concurrency = ConcurrencyInfo(
                is_async=raw_concurrency.get("is_async", False),
                uses_locks=raw_concurrency.get("uses_locks", False),
                uses_threads=raw_concurrency.get("uses_threads", False),
                patterns=raw_concurrency.get("patterns", []),
            )

    # Security issues (for functions/methods/variables)
    if source["element_type"] in ("function", "method", "variable", "constant"):
        raw_security_issues = source.get("security_issues", [])
        for issue in raw_security_issues:
            security_issues.append(
                SecurityIssueInfo(
                    kind=issue.get("kind", ""),
                    pattern=issue.get("pattern", ""),
                    line=issue.get("line", 0),
                    severity=issue.get("severity", "info"),
                    message=issue.get("message"),
                )
            )

    # Environment variables (for functions/methods)
    if source["element_type"] in ("function", "method"):
        raw_env_vars = source.get("env_vars", [])
        for ev in raw_env_vars:
            env_vars.append(
                EnvVarInfo(
                    name=ev.get("name", ""),
                    line=ev.get("line", 0),
                    access_type=ev.get("access_type", ""),
                )
            )

    # Aggregated metrics summary (for files/classes)
    metrics_summary = None
    if source["element_type"] in ("file", "class"):
        raw_metrics_summary = source.get("metrics_summary")
        if raw_metrics_summary:
            metrics_summary = MetricsSummaryInfo(
                total_elements=raw_metrics_summary.get("total_elements") or 0,
                total_functions=raw_metrics_summary.get("total_functions") or 0,
                avg_complexity=raw_metrics_summary.get("avg_complexity") or 0.0,
                max_complexity=raw_metrics_summary.get("max_complexity") or 0,
                total_lines=raw_metrics_summary.get("total_lines") or 0,
                documented_pct=raw_metrics_summary.get("documented_pct") or 0.0,
                async_count=raw_metrics_summary.get("async_count") or 0,
                security_issue_count=raw_metrics_summary.get("security_issue_count") or 0,
                security_by_severity=raw_metrics_summary.get("security_by_severity") or {},
            )

    # Type annotations
    type_annotations = []
    raw_type_annotations = source.get("type_annotations", [])
    for ta in raw_type_annotations:
        type_annotations.append(
            TypeAnnotationInfo(
                name=ta.get("name", ""),
                kind=ta.get("kind", ""),
                location=ta.get("location", ""),
                line=ta.get("line", 0),
                generic_args=ta.get("generic_args"),
            )
        )

    # TODOs
    todos = []
    raw_todos = source.get("todos", [])
    for todo in raw_todos:
        todos.append(
            TodoInfo(
                kind=todo.get("kind", ""),
                text=todo.get("text", ""),
                line=todo.get("line", 0),
                assignee=todo.get("assignee"),
                priority=todo.get("priority"),
                issue_ref=todo.get("issue_ref"),
            )
        )

    # Section markers
    section_markers = []
    raw_section_markers = source.get("section_markers", [])
    for sm in raw_section_markers:
        section_markers.append(
            SectionMarkerInfo(
                label=sm.get("label", ""),
                line=sm.get("line", 0),
                style=sm.get("style", ""),
            )
        )

    # Associated comments
    associated_comments = []
    raw_comments = source.get("associated_comments", [])
    for comment in raw_comments:
        associated_comments.append(
            CommentInfo(
                text=comment.get("text", ""),
                line=comment.get("line", 0),
                kind=comment.get("kind", ""),
                position=comment.get("position", ""),
            )
        )

    # HTTP routes
    http_routes = []
    raw_http_routes = source.get("http_routes", [])
    for route in raw_http_routes:
        http_routes.append(
            HttpRouteInfo(
                method=route.get("method", ""),
                path=route.get("path", ""),
                path_params=route.get("path_params", []),
                framework=route.get("framework", ""),
                line=route.get("line"),
            )
        )

    # CLI commands
    cli_commands = []
    raw_cli_commands = source.get("cli_commands", [])
    for cmd in raw_cli_commands:
        cli_commands.append(
            CliCommandInfo(
                name=cmd.get("name", ""),
                options=cmd.get("options", []),
                framework=cmd.get("framework", ""),
            )
        )

    # Purity
    purity = None
    raw_purity = source.get("purity")
    if raw_purity:
        purity = PurityInfo(
            level=raw_purity.get("level", ""),
            confidence=raw_purity.get("confidence", ""),
            reasons=raw_purity.get("reasons", []),
        )

    # Side effects
    side_effects = []
    raw_side_effects = source.get("side_effects", [])
    for se in raw_side_effects:
        side_effects.append(
            SideEffectInfo(
                kind=se.get("kind", ""),
                target=se.get("target"),
                line=se.get("line", 0),
            )
        )

    return ElementDetailResponse(
        element_id=source["element_id"],
        hash_id=source.get("hash_id"),
        name=source["name"],
        element_type=source["element_type"],
        file_path=source.get("relative_path", ""),
        line_start=source.get("line_start", 0),
        line_end=source.get("line_end"),
        language=source.get("language", "unknown"),
        summary=summary,
        signature=source.get("signature"),
        docstring=source.get("docstring"),
        raw_code=source.get("raw_code"),
        decorators=decorators,
        decorator_details=decorator_details,
        visibility=source.get("visibility"),
        is_async=source.get("is_async", False),
        is_test=source.get("is_test", False),
        indexed_at=source.get("indexed_at"),
        # Enhanced context for classes
        base_classes=source.get("base_classes", []),
        class_attributes=class_attributes,
        # Enhanced context for functions/methods
        return_type=source.get("return_type"),
        parameters=parameters,
        exceptions_raised=source.get("exceptions_raised", []),
        attributes_modified=source.get("attributes_modified", []),
        calls=calls,
        # For file elements
        imports=imports,
        element_count=source.get("element_count"),
        # Type annotations
        type_annotations=type_annotations,
        # Pattern detection
        detected_patterns=source.get("detected_patterns", []),
        pattern_confidence=source.get("pattern_confidence", {}),
        # Documentation artifacts
        todos=todos,
        section_markers=section_markers,
        associated_comments=associated_comments,
        # API surface
        http_routes=http_routes,
        cli_commands=cli_commands,
        # Purity and side effects
        purity=purity,
        side_effects=side_effects,
        mutated_state=source.get("mutated_state", []),
        # API surface flag
        is_public_api=source.get("is_public_api", False),
        # Variable context (how variables are used)
        context_usages=source.get("context_usages", []),
        # Context and relationships
        context=ElementContext(
            file=file_context,
            parent=parent_context,
            children=children,
            siblings=siblings,
        ),
        repository=RepoRef(
            scope=source["scope"],
            name=source["repository"],
        ),
        feature_info=feature_info,
        glossary_info=glossary_info,
        # Code metrics
        complexity=complexity,
        code_metrics=code_metrics_info,
        docstring_quality=docstring_quality,
        security_issues=security_issues,
        env_vars=env_vars,
        concurrency=concurrency,
        metrics_summary=metrics_summary,
    )
