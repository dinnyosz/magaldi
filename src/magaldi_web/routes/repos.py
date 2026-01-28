"""Repository API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from magaldi_web.dependencies import get_es_repository
from magaldi_web.models import (
    ActiveUser,
    Contributor,
    ElementInfo,
    ElementStats,
    FileDetailResponse,
    FileInfo,
    FileTreeResponse,
    RepoDetailResponse,
    RepoListResponse,
    RepoSummary,
    TreeNode,
)
from shared.db.elasticsearch import ElasticsearchRepository, INDEX_NAME

router = APIRouter()


@router.get("/repos", response_model=RepoListResponse)
async def list_repos(
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> RepoListResponse:
    """List all indexed repositories."""
    client = es_repo._get_client()

    result = client.search(
        index=INDEX_NAME,
        body={
            "size": 0,
            "query": {"term": {"username": "main"}},
            "aggs": {
                "repos": {
                    "composite": {
                        "size": 1000,
                        "sources": [
                            {"scope": {"terms": {"field": "scope.keyword"}}},
                            {"repository": {"terms": {"field": "repository.keyword"}}},
                        ],
                    },
                    "aggs": {
                        "file_count": {"filter": {"term": {"element_type": "file"}}},
                        "languages": {"terms": {"field": "language.keyword", "size": 10}},
                    },
                },
            },
        },
    )

    repos = []
    for bucket in result.get("aggregations", {}).get("repos", {}).get("buckets", []):
        languages = [
            lang["key"]
            for lang in bucket.get("languages", {}).get("buckets", [])
        ]
        repos.append(
            RepoSummary(
                scope=bucket["key"]["scope"],
                name=bucket["key"]["repository"],
                file_count=bucket.get("file_count", {}).get("doc_count", 0),
                element_count=bucket["doc_count"],
                languages=languages,
            )
        )

    return RepoListResponse(repos=repos, total=len(repos))


@router.get("/repos/{scope}/{repository}", response_model=RepoDetailResponse)
async def get_repo_detail(
    scope: str = Path(..., description="Repository scope"),
    repository: str = Path(..., description="Repository name"),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> RepoDetailResponse:
    """Get detailed information about a repository."""
    client = es_repo._get_client()

    # Get repo stats
    result = client.search(
        index=INDEX_NAME,
        body={
            "size": 0,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": "main"}},
                    ],
                },
            },
            "aggs": {
                "file_count": {"filter": {"term": {"element_type": "file"}}},
                "languages": {"terms": {"field": "language.keyword", "size": 20}},
            },
        },
    )

    total = result.get("hits", {}).get("total", {}).get("value", 0)
    if total == 0:
        raise HTTPException(status_code=404, detail="Repository not found")

    aggs = result.get("aggregations", {})
    file_count = aggs.get("file_count", {}).get("doc_count", 0)
    languages = [b["key"] for b in aggs.get("languages", {}).get("buckets", [])]

    # Get active users (non-main users with indexed data)
    users_result = client.search(
        index=INDEX_NAME,
        body={
            "size": 0,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                    ],
                    "must_not": [{"term": {"username": "main"}}],
                },
            },
            "aggs": {
                "users": {
                    "terms": {"field": "username.keyword", "size": 50},
                    "aggs": {
                        "file_count": {"filter": {"term": {"element_type": "file"}}},
                    },
                },
            },
        },
    )

    active_users = []
    for bucket in users_result.get("aggregations", {}).get("users", {}).get("buckets", []):
        active_users.append(
            ActiveUser(
                username=bucket["key"],
                files_modified=bucket.get("file_count", {}).get("doc_count", 0),
            )
        )

    return RepoDetailResponse(
        scope=scope,
        name=repository,
        file_count=file_count,
        element_count=total,
        languages=languages,
        active_users=active_users,
    )


@router.get("/repos/{scope}/{repository}/tree", response_model=FileTreeResponse)
async def get_file_tree(
    scope: str = Path(..., description="Repository scope"),
    repository: str = Path(..., description="Repository name"),
    username: str = Query(default="main", description="Username/branch"),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> FileTreeResponse:
    """Get file tree for a repository."""
    client = es_repo._get_client()

    # Get all file elements
    result = client.search(
        index=INDEX_NAME,
        body={
            "size": 10000,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                        {"term": {"element_type": "file"}},
                    ],
                },
            },
            "_source": ["relative_path", "language", "summary"],
            "sort": [{"relative_path.keyword": "asc"}],
        },
    )

    # Build tree structure
    tree: dict[str, TreeNode] = {}

    for hit in result.get("hits", {}).get("hits", []):
        source = hit["_source"]
        path = source["relative_path"]
        parts = path.split("/")

        current_level = tree
        current_path = ""

        for i, part in enumerate(parts):
            current_path = f"{current_path}/{part}" if current_path else part
            is_file = i == len(parts) - 1

            if part not in current_level:
                if is_file:
                    node = TreeNode(
                        name=part,
                        type="file",
                        path=path,
                        language=source.get("language"),
                        children=[],
                    )
                else:
                    node = TreeNode(
                        name=part,
                        type="directory",
                        path=None,
                        children=[],
                    )
                current_level[part] = node
            else:
                node = current_level[part]

            if not is_file:
                # Create dict for children lookup
                if not hasattr(node, "_children_dict"):
                    node._children_dict = {c.name: c for c in node.children}
                current_level = node._children_dict

    # Convert dict to list
    def dict_to_list(d: dict[str, TreeNode]) -> list[TreeNode]:
        result = []
        for node in d.values():
            if node.type == "directory" and hasattr(node, "_children_dict"):
                node.children = dict_to_list(node._children_dict)
                delattr(node, "_children_dict")
            result.append(node)
        # Sort: directories first, then files, alphabetically
        result.sort(key=lambda n: (n.type == "file", n.name.lower()))
        return result

    tree_list = dict_to_list(tree)

    return FileTreeResponse(scope=scope, repository=repository, tree=tree_list)


@router.get("/repos/{scope}/{repository}/files/{file_path:path}", response_model=FileDetailResponse)
async def get_file_detail(
    scope: str = Path(..., description="Repository scope"),
    repository: str = Path(..., description="Repository name"),
    file_path: str = Path(..., description="File path"),
    username: str = Query(default="main", description="Username/branch"),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> FileDetailResponse:
    """Get detailed information about a file."""
    client = es_repo._get_client()

    # Get file element
    file_result = client.search(
        index=INDEX_NAME,
        body={
            "size": 1,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                        {"term": {"relative_path.keyword": file_path}},
                        {"term": {"element_type": "file"}},
                    ],
                },
            },
            "_source": ["relative_path", "language", "summary", "line_end"],
        },
    )

    file_hits = file_result.get("hits", {}).get("hits", [])
    if not file_hits:
        raise HTTPException(status_code=404, detail="File not found")

    file_source = file_hits[0]["_source"]

    # Get all elements in file
    elements_result = client.search(
        index=INDEX_NAME,
        body={
            "size": 1000,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                        {"term": {"relative_path.keyword": file_path}},
                    ],
                    "must_not": [{"term": {"element_type": "file"}}],
                },
            },
            "_source": [
                "element_id",
                "name",
                "element_type",
                "line_start",
                "line_end",
                "summary",
                "signature",
            ],
            "sort": [{"line_start": "asc"}],
        },
    )

    structure = []
    stats = {"classes": 0, "functions": 0, "methods": 0, "variables": 0}

    for hit in elements_result.get("hits", {}).get("hits", []):
        source = hit["_source"]
        element_type = source["element_type"]

        if element_type in stats:
            stats[element_type] += 1
        elif element_type == "function":
            stats["functions"] += 1
        elif element_type == "method":
            stats["methods"] += 1
        elif element_type == "class":
            stats["classes"] += 1
        elif element_type == "variable":
            stats["variables"] += 1

        structure.append(
            ElementInfo(
                element_id=source["element_id"],
                name=source["name"],
                element_type=source["element_type"],
                line_start=source["line_start"],
                line_end=source.get("line_end"),
                summary=source.get("summary"),
                signature=source.get("signature"),
            )
        )

    # Get contributors (users with versions of this file)
    contributors_result = client.search(
        index=INDEX_NAME,
        body={
            "size": 0,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"relative_path.keyword": file_path}},
                        {"term": {"element_type": "file"}},
                    ],
                    "must_not": [{"term": {"username": "main"}}],
                },
            },
            "aggs": {
                "users": {"terms": {"field": "username.keyword", "size": 50}},
            },
        },
    )

    contributors = [
        Contributor(username=bucket["key"])
        for bucket in contributors_result.get("aggregations", {}).get("users", {}).get("buckets", [])
    ]

    return FileDetailResponse(
        file=FileInfo(
            path=file_source["relative_path"],
            language=file_source.get("language", "unknown"),
            summary=file_source.get("summary"),
            line_count=file_source.get("line_end", 0),
        ),
        structure=structure,
        stats=ElementStats(**stats),
        contributors=contributors,
    )
