"""MCP Tool implementations for Magaldi.

Each tool function takes an ElasticsearchRepository and optional OllamaEmbedClient,
plus tool-specific parameters, and returns a dict or list result.
"""

from __future__ import annotations

from typing import Any

from shared.db.elasticsearch import ElasticsearchRepository
from shared.ai.embedding import OllamaEmbedClient


# =============================================================================
# SEARCH TOOLS
# =============================================================================


def search_code(
    es: ElasticsearchRepository,
    ollama: OllamaEmbedClient | None,
    query: str,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    element_types: list[str] | None = None,
    language: str | None = None,
    limit: int = 20,
    include_code: bool = False,
    brief: bool = False,
) -> list[dict[str, Any]]:
    """Semantic search for code elements.

    Tries vector search first, falls back to keyword search if Ollama unavailable.

    Args:
        es: Elasticsearch repository.
        ollama: Ollama client for query embedding (optional, falls back to keyword).
        query: Natural language search query.
        scope: Filter by scope.
        repository: Filter by repository.
        username: User branch to search.
        element_types: Filter by element types.
        language: Filter by programming language.
        limit: Maximum results.
        include_code: Include source code in results (for detailed inspection).
        brief: Minimal output - just name, type, file, line (for exploration).

    Returns:
        List of matching code elements.
    """
    # Validate limit
    limit = max(1, min(limit, 50))

    # Try vector search first, fall back to keyword search
    results = []
    if ollama is not None:
        try:
            query_embedding = ollama.embed_single(query)
            results = es.search_by_vector(
                embedding=query_embedding,
                scope=scope,
                repository=repository,
                username=username,
                element_types=element_types,
                size=limit,
            )
        except Exception:
            pass  # Fall through to keyword search

    # Fallback to keyword search if vector search failed or unavailable
    if not results:
        results = es.search_by_keyword(
            query=query,
            scope=scope,
            repository=repository,
            username=username,
            element_types=element_types,
            size=limit,
        )

    # Format results - keep only essential fields
    formatted = []
    for result in results:
        # Filter by language if specified
        if language and result.get("language") != language:
            continue

        # Build qualified name: ClassName.method_name for methods
        name = result.get("name")
        if result.get("element_type") == "method" and result.get("parent_id"):
            # Try to get parent class name from parent_id
            parent_doc = es.get_document(result["parent_id"])
            if parent_doc and parent_doc.get("element_type") == "class":
                name = f"{parent_doc.get('name')}.{name}"

        entry: dict[str, Any] = {
            "name": name,
            "type": result.get("element_type"),
            "file": result.get("relative_path"),
            "line": result.get("line_start"),
            "element_id": result.get("element_id"),
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

        formatted.append(entry)

    return formatted[:limit]


def search_features(
    es: ElasticsearchRepository,
    ollama: OllamaEmbedClient | None,
    query: str,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search for features/capabilities.

    Tries vector search first, falls back to keyword search if Ollama unavailable.

    Args:
        es: Elasticsearch repository.
        ollama: Ollama client for query embedding (optional, falls back to keyword).
        query: Search query for features.
        scope: Filter by scope.
        repository: Filter by repository.
        username: User branch.
        limit: Maximum results.

    Returns:
        List of matching features.
    """
    limit = max(1, min(limit, 50))

    # Try vector search first, fall back to keyword search
    results = []
    if ollama is not None:
        try:
            query_embedding = ollama.embed_single(query)
            results = es.search_by_vector(
                embedding=query_embedding,
                scope=scope,
                repository=repository,
                username=username,
                element_types=["feature"],
                size=limit,
            )
        except Exception:
            pass  # Fall through to keyword search

    # Fallback to keyword search if vector search failed or unavailable
    if not results:
        results = es.search_by_keyword(
            query=query,
            scope=scope,
            repository=repository,
            username=username,
            element_types=["feature"],
            size=limit,
        )

    formatted = []
    for result in results:
        formatted.append({
            "label": result.get("cluster_label", result.get("name")),
            "summary": result.get("summary", ""),
            "member_count": result.get("member_count", 0),
            "feature_id": result.get("element_id"),  # For follow-up queries
        })

    return formatted


def find_similar(
    es: ElasticsearchRepository,
    element_id: str,
    limit: int = 10,
    same_repo_only: bool = False,
) -> list[dict[str, Any]]:
    """Find code elements similar to a given element.

    Args:
        es: Elasticsearch repository.
        element_id: Source element ID.
        limit: Maximum results.
        same_repo_only: Only search within same repository.

    Returns:
        List of similar elements with similarity scores.
    """
    limit = max(1, min(limit, 50))

    # Get source element
    doc = es.get_document(element_id)
    if not doc:
        raise ValueError(f"Element not found: {element_id}")

    embedding = doc.get("embedding")
    if not embedding:
        raise ValueError(f"Element has no embedding: {element_id}")

    # Build search filters
    scope = doc.get("scope") if same_repo_only else None
    repository = doc.get("repository") if same_repo_only else None

    # Search excluding self
    results = es.search_by_vector(
        embedding=embedding,
        scope=scope,
        repository=repository,
        username=doc.get("username", "main"),
        size=limit + 1,  # Get one extra to filter out self
    )

    formatted = []
    for result in results:
        # Skip self
        if result.get("element_id") == element_id:
            continue

        entry: dict[str, Any] = {
            "name": result.get("name"),
            "type": result.get("element_type"),
            "file": result.get("relative_path"),
            "line": result.get("line_start"),
            "summary": result.get("summary", ""),
            "element_id": result.get("element_id"),
        }
        formatted.append(entry)

        if len(formatted) >= limit:
            break

    return formatted


# =============================================================================
# CONTEXT TOOLS
# =============================================================================


def get_element(
    es: ElasticsearchRepository,
    element_id: str,
    include_code: bool = False,
) -> dict[str, Any]:
    """Get full details of a code element.

    Args:
        es: Elasticsearch repository.
        element_id: Element ID.
        include_code: Include raw source code.

    Returns:
        Element details.
    """
    doc = es.get_document(element_id)
    if not doc:
        raise ValueError(f"Element not found: {element_id}")

    result: dict[str, Any] = {
        "name": doc.get("name"),
        "type": doc.get("element_type"),
        "file": doc.get("relative_path"),
        "line_start": doc.get("line_start"),
        "line_end": doc.get("line_end"),
        "summary": doc.get("summary", ""),
    }

    # Only include if present and meaningful
    if doc.get("signature"):
        result["signature"] = doc["signature"]
    if doc.get("docstring"):
        result["docstring"] = doc["docstring"]
    if doc.get("decorators"):
        result["decorators"] = doc["decorators"]
    if doc.get("is_async"):
        result["is_async"] = True
    if doc.get("parent_id"):
        result["parent_id"] = doc["parent_id"]

    if include_code:
        result["code"] = doc.get("raw_code", "")

    return result


def get_context(
    es: ElasticsearchRepository,
    element_id: str,
    include_siblings: bool = False,
    include_children: bool = True,
) -> dict[str, Any]:
    """Get hierarchical context for an element.

    Args:
        es: Elasticsearch repository.
        element_id: Element ID.
        include_siblings: Include sibling elements.
        include_children: Include child elements.

    Returns:
        Hierarchical context.
    """
    doc = es.get_document(element_id)
    if not doc:
        raise ValueError(f"Element not found: {element_id}")

    context: dict[str, Any] = {
        "element": {
            "id": doc.get("element_id"),
            "name": doc.get("name"),
            "type": doc.get("element_type"),
            "file": doc.get("relative_path"),
            "line_start": doc.get("line_start"),
            "line_end": doc.get("line_end"),
            "summary": doc.get("summary", ""),
            "signature": doc.get("signature", ""),
        },
        "file": None,
        "parent": None,
        "siblings": [],
        "children": [],
    }

    scope = doc.get("scope")
    repository = doc.get("repository")
    username = doc.get("username", "main")
    relative_path = doc.get("relative_path")
    parent_id = doc.get("parent_id")

    # Get file context
    file_doc = _find_file_element(es, scope, repository, username, relative_path)
    if file_doc:
        context["file"] = {
            "id": file_doc.get("element_id"),
            "name": file_doc.get("name"),
            "summary": file_doc.get("summary", ""),
        }

    # Get parent context
    if parent_id:
        parent_doc = es.get_document(parent_id)
        if parent_doc:
            context["parent"] = {
                "id": parent_doc.get("element_id"),
                "name": parent_doc.get("name"),
                "type": parent_doc.get("element_type"),
                "summary": parent_doc.get("summary", ""),
                "signature": parent_doc.get("signature", ""),
            }

    # Get siblings
    if include_siblings and parent_id:
        siblings = _find_children(es, parent_id)
        context["siblings"] = [
            {
                "id": s.get("element_id"),
                "name": s.get("name"),
                "type": s.get("element_type"),
                "line": s.get("line_start"),
                "summary": s.get("summary", ""),
            }
            for s in siblings
            if s.get("element_id") != element_id
        ]

    # Get children
    if include_children:
        children = _find_children(es, element_id)
        context["children"] = [
            {
                "id": c.get("element_id"),
                "name": c.get("name"),
                "type": c.get("element_type"),
                "line": c.get("line_start"),
                "summary": c.get("summary", ""),
                "signature": c.get("signature", ""),
            }
            for c in children
        ]

    return context


def get_file_structure(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    file_path: str,
    username: str = "main",
) -> dict[str, Any]:
    """Get full structure of a file.

    Args:
        es: Elasticsearch repository.
        scope: Repository scope.
        repository: Repository name.
        file_path: Relative file path.
        username: User branch.

    Returns:
        File structure with nested elements.
    """
    # Get file element
    file_doc = _find_file_element(es, scope, repository, username, file_path)
    if not file_doc:
        raise ValueError(f"File not found: {file_path}")

    # Get all elements in file
    elements = _find_elements_in_file(es, scope, repository, username, file_path)

    # Build tree structure
    def build_tree(parent_id: str | None) -> list[dict]:
        children = []
        for elem in elements:
            if elem.get("parent_id") == parent_id:
                node = {
                    "id": elem.get("element_id"),
                    "name": elem.get("name"),
                    "type": elem.get("element_type"),
                    "line_start": elem.get("line_start"),
                    "line_end": elem.get("line_end"),
                    "summary": elem.get("summary", ""),
                    "signature": elem.get("signature", ""),
                    "children": build_tree(elem.get("element_id")),
                }
                children.append(node)
        return sorted(children, key=lambda x: x.get("line_start", 0))

    file_id = file_doc.get("element_id")

    return {
        "file": {
            "path": file_path,
            "language": file_doc.get("language"),
            "summary": file_doc.get("summary", ""),
            "line_count": file_doc.get("line_end", 0),
        },
        "structure": build_tree(file_id),
        "stats": {
            "classes": sum(1 for e in elements if e.get("element_type") == "class"),
            "functions": sum(1 for e in elements if e.get("element_type") == "function"),
            "methods": sum(1 for e in elements if e.get("element_type") == "method"),
            "total": len(elements),
        },
    }


# =============================================================================
# DISCOVERY TOOLS
# =============================================================================


def list_repos(
    es: ElasticsearchRepository,
    scope: str | None = None,
) -> list[dict[str, Any]]:
    """List all indexed repositories.

    Args:
        es: Elasticsearch repository.
        scope: Filter by scope.

    Returns:
        List of repositories with statistics.
    """
    return es.get_indexed_repositories(scope=scope)


def list_features(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    username: str = "main",
) -> list[dict[str, Any]]:
    """List all features for a repository.

    Args:
        es: Elasticsearch repository.
        scope: Repository scope.
        repository: Repository name.
        username: User branch.

    Returns:
        List of features.
    """
    return es.get_features(scope, repository, username)


def get_repo_stats(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    username: str = "main",
) -> dict[str, Any]:
    """Get statistics for a repository.

    Args:
        es: Elasticsearch repository.
        scope: Repository scope.
        repository: Repository name.
        username: User branch.

    Returns:
        Repository statistics.
    """
    return es.get_repository_stats(scope, repository, username)


# =============================================================================
# NAVIGATION TOOLS
# =============================================================================


def get_children(
    es: ElasticsearchRepository,
    element_id: str,
) -> list[dict[str, Any]]:
    """Get child elements of a parent.

    Args:
        es: Elasticsearch repository.
        element_id: Parent element ID.

    Returns:
        List of child elements.
    """
    children = _find_children(es, element_id)
    return [
        {
            "element_id": c.get("element_id"),
            "name": c.get("name"),
            "type": c.get("element_type"),
            "line_start": c.get("line_start"),
            "line_end": c.get("line_end"),
            "summary": c.get("summary", ""),
            "signature": c.get("signature", ""),
        }
        for c in children
    ]


def get_feature_members(
    es: ElasticsearchRepository,
    feature_id: str,
) -> list[dict[str, Any]]:
    """Get all members of a feature cluster.

    Args:
        es: Elasticsearch repository.
        feature_id: Feature ID.

    Returns:
        List of member elements.
    """
    # Get feature document
    feature = es.get_document(feature_id)
    if not feature:
        raise ValueError(f"Feature not found: {feature_id}")

    member_ids = feature.get("member_ids", [])
    if not member_ids:
        return []

    # Fetch member documents
    members = []
    for member_id in member_ids:
        doc = es.get_document(member_id)
        if doc:
            members.append({
                "element_id": doc.get("element_id"),
                "name": doc.get("name"),
                "type": doc.get("element_type"),
                "file": doc.get("relative_path"),
                "line": doc.get("line_start"),
                "summary": doc.get("summary", ""),
                "signature": doc.get("signature", ""),
            })

    return members


# =============================================================================
# FILE TOOLS
# =============================================================================


def read_file(
    repo_root: str,
    file_path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict[str, Any]:
    """Read file contents from disk.

    Args:
        repo_root: Repository root path.
        file_path: Relative path to file.
        start_line: Start line (1-indexed, optional).
        end_line: End line (1-indexed, optional).

    Returns:
        File contents with metadata.
    """
    from pathlib import Path

    full_path = Path(repo_root) / file_path
    if not full_path.exists():
        raise ValueError(f"File not found: {file_path}")
    if not full_path.is_file():
        raise ValueError(f"Not a file: {file_path}")

    content = full_path.read_text()
    lines = content.splitlines()
    total_lines = len(lines)

    # Apply line range if specified
    if start_line is not None or end_line is not None:
        start_idx = (start_line - 1) if start_line else 0
        end_idx = end_line if end_line else total_lines
        lines = lines[start_idx:end_idx]
        content = "\n".join(lines)

    return {
        "path": file_path,
        "content": content,
        "total_lines": total_lines,
        "lines_returned": len(lines),
    }


def find_files(
    repo_root: str,
    pattern: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Find files by glob pattern.

    Args:
        repo_root: Repository root path.
        pattern: Glob pattern (e.g., '**/*.py', 'src/**/*.ts').
        limit: Maximum files to return.

    Returns:
        List of matching files with basic info.
    """
    from pathlib import Path

    root = Path(repo_root)
    matches = []

    for path in root.glob(pattern):
        if path.is_file() and not any(p.startswith('.') for p in path.parts):
            rel_path = path.relative_to(root)
            matches.append({
                "path": str(rel_path),
                "size": path.stat().st_size,
            })
            if len(matches) >= limit:
                break

    return sorted(matches, key=lambda x: x["path"])


def batch_get_elements(
    es: ElasticsearchRepository,
    element_ids: list[str],
    include_code: bool = False,
) -> list[dict[str, Any]]:
    """Get multiple elements by ID in one call.

    Args:
        es: Elasticsearch repository.
        element_ids: List of element IDs.
        include_code: Include source code.

    Returns:
        List of elements (preserves order, skips missing).
    """
    results = []
    for eid in element_ids:
        doc = es.get_document(eid)
        if doc:
            entry: dict[str, Any] = {
                "name": doc.get("name"),
                "type": doc.get("element_type"),
                "file": doc.get("relative_path"),
                "line": doc.get("line_start"),
                "summary": doc.get("summary", ""),
                "element_id": eid,
            }
            if doc.get("signature"):
                entry["signature"] = doc["signature"]
            if include_code and doc.get("raw_code"):
                entry["code"] = doc["raw_code"]
            results.append(entry)
    return results


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _find_file_element(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    username: str,
    relative_path: str,
) -> dict[str, Any] | None:
    """Find the file element for a given path."""
    client = es._get_client()
    result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                        {"term": {"relative_path": relative_path}},
                        {"term": {"element_type": "file"}},
                    ]
                }
            },
            "size": 1,
        },
    )

    hits = result.get("hits", {}).get("hits", [])
    return hits[0]["_source"] if hits else None


def _find_children(
    es: ElasticsearchRepository,
    parent_id: str,
) -> list[dict[str, Any]]:
    """Find all children of an element."""
    client = es._get_client()
    result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {
                "term": {"parent_id": parent_id}
            },
            "size": 100,
            "sort": [{"line_start": "asc"}],
        },
    )

    return [hit["_source"] for hit in result.get("hits", {}).get("hits", [])]


def _find_elements_in_file(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    username: str,
    relative_path: str,
) -> list[dict[str, Any]]:
    """Find all elements in a file."""
    client = es._get_client()
    result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                        {"term": {"relative_path": relative_path}},
                    ],
                    "must_not": [
                        {"term": {"element_type": "file"}},
                    ],
                }
            },
            "size": 500,
            "sort": [{"line_start": "asc"}],
        },
    )

    return [hit["_source"] for hit in result.get("hits", {}).get("hits", [])]


# =============================================================================
# CODE SEARCH TOOLS (regex/grep-based)
# =============================================================================


def grep_code(
    repo_root: str,
    pattern: str,
    glob: str | None = None,
    context_lines: int = 0,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Search code with regex pattern (like grep/ripgrep).

    Args:
        repo_root: Repository root path.
        pattern: Regex pattern to search.
        glob: File glob filter (e.g., '*.py', '*.ts').
        context_lines: Lines of context before/after match.
        limit: Maximum matches to return.

    Returns:
        List of matches with file, line, content, and context.
    """
    import re
    import subprocess
    from pathlib import Path

    root = Path(repo_root)
    results: list[dict[str, Any]] = []

    # Try ripgrep first (faster), fall back to manual search
    try:
        cmd = ["rg", "--json", "-n", "--max-count", str(limit * 2)]
        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])
        if glob:
            cmd.extend(["--glob", glob])
        cmd.extend([pattern, str(root)])

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        # Parse ripgrep JSON output
        for line in proc.stdout.splitlines():
            try:
                import json
                obj = json.loads(line)
                if obj.get("type") == "match":
                    data = obj["data"]
                    path = data["path"]["text"]
                    rel_path = str(Path(path).relative_to(root))
                    line_num = data["line_number"]
                    text = data["lines"]["text"].rstrip("\n")

                    results.append({
                        "file": rel_path,
                        "line": line_num,
                        "content": text,
                        "match": data.get("submatches", [{}])[0].get("match", {}).get("text", ""),
                    })

                    if len(results) >= limit:
                        break
            except (json.JSONDecodeError, KeyError):
                continue

        return results

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass  # Fall back to manual search

    # Manual fallback (slower but always works)
    compiled = re.compile(pattern)
    file_pattern = glob if glob else "**/*"

    for path in root.glob(file_pattern):
        if not path.is_file():
            continue
        if any(p.startswith('.') for p in path.parts):
            continue

        try:
            content = path.read_text(errors="ignore")
            lines = content.splitlines()

            for i, line in enumerate(lines):
                match = compiled.search(line)
                if match:
                    rel_path = str(path.relative_to(root))
                    entry: dict[str, Any] = {
                        "file": rel_path,
                        "line": i + 1,
                        "content": line,
                        "match": match.group(0),
                    }

                    # Add context if requested
                    if context_lines > 0:
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)
                        entry["context_before"] = lines[start:i]
                        entry["context_after"] = lines[i + 1:end]

                    results.append(entry)

                    if len(results) >= limit:
                        return results

        except (OSError, UnicodeDecodeError):
            continue

    return results


def find_usages(
    repo_root: str,
    es: ElasticsearchRepository,
    element_id: str,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Find where an element is used/called/referenced.

    Args:
        repo_root: Repository root path.
        es: Elasticsearch repository.
        element_id: Element to find usages of.
        limit: Maximum usages to return.

    Returns:
        List of usage locations with context.
    """
    import re

    # Get the element to find its name
    doc = es.get_document(element_id)
    if not doc:
        raise ValueError(f"Element not found: {element_id}")

    name = doc.get("name")
    element_type = doc.get("element_type")
    defining_file = doc.get("relative_path")
    defining_line = doc.get("line_start")

    # Build search pattern based on element type
    if element_type == "function":
        # Function calls: name(
        pattern = rf"\b{re.escape(name)}\s*\("
    elif element_type == "method":
        # Method calls: .name( or self.name(
        pattern = rf"\.{re.escape(name)}\s*\("
    elif element_type == "class":
        # Class references: inheritance, instantiation, type hints
        pattern = rf"\b{re.escape(name)}\b"
    else:
        # Generic: just the name as word boundary
        pattern = rf"\b{re.escape(name)}\b"

    # Search with grep_code
    matches = grep_code(
        repo_root=repo_root,
        pattern=pattern,
        glob="**/*.py",  # TODO: detect language from element
        context_lines=1,
        limit=limit + 10,  # Get extra to filter out definition
    )

    # Filter out the definition itself
    usages = []
    for match in matches:
        # Skip if it's the definition line
        if match["file"] == defining_file and match["line"] == defining_line:
            continue

        # Skip if it looks like a definition (def/class keyword)
        content = match["content"].strip()
        if element_type == "function" and content.startswith("def "):
            continue
        if element_type == "class" and content.startswith("class "):
            continue
        if element_type == "method" and content.startswith("def "):
            continue

        usages.append({
            "file": match["file"],
            "line": match["line"],
            "content": match["content"],
            "context_before": match.get("context_before", []),
            "context_after": match.get("context_after", []),
        })

        if len(usages) >= limit:
            break

    return usages


def find_implementations(
    repo_root: str,
    es: ElasticsearchRepository,
    element_id: str | None = None,
    class_name: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Find classes that implement/inherit from a protocol or base class.

    Args:
        repo_root: Repository root path.
        es: Elasticsearch repository.
        element_id: Element ID of the protocol/base class.
        class_name: Or just the class name to search for.
        limit: Maximum implementations to return.

    Returns:
        List of implementing classes with their info.
    """
    import re

    # Get the name to search for
    if element_id:
        doc = es.get_document(element_id)
        if not doc:
            raise ValueError(f"Element not found: {element_id}")
        name = doc.get("name")
    elif class_name:
        name = class_name
    else:
        raise ValueError("Either element_id or class_name required")

    # Search for class definitions that inherit from this name
    # Pattern: class SomeClass(Name) or class SomeClass(Name, Other)
    pattern = rf"class\s+\w+\s*\([^)]*\b{re.escape(name)}\b"

    matches = grep_code(
        repo_root=repo_root,
        pattern=pattern,
        glob="**/*.py",
        context_lines=2,
        limit=limit,
    )

    results = []
    for match in matches:
        # Extract class name from the match
        class_match = re.search(r"class\s+(\w+)", match["content"])
        impl_name = class_match.group(1) if class_match else "Unknown"

        results.append({
            "class_name": impl_name,
            "file": match["file"],
            "line": match["line"],
            "definition": match["content"].strip(),
            "context_after": match.get("context_after", []),
        })

    return results


def generate_skill(
    project_root: str | None = None,
    skill_name: str = "magaldi",
    scope: str = "project",
) -> dict[str, Any]:
    """Generate a SKILL.md file that teaches LLMs how to use this MCP effectively.

    Args:
        project_root: Project root directory (required for scope="project").
        skill_name: Name of the skill (default: "magaldi").
        scope: Where to install - "project" (.claude/skills in project) or "global" (~/.claude/skills).

    Returns:
        Dict with skill content and metadata.
    """
    from pathlib import Path

    skill_content = '''---
name: magaldi-code-discovery
description: Use when exploring, understanding, or searching codebases. Provides semantic search, usage tracking, and call graph analysis.
---

# Magaldi Code Discovery

Use this skill when the user asks to:
- Find code, functions, or classes
- Understand how something works
- Find where something is used
- Explore a codebase structure
- Refactor or modify existing code

## Core Principle: Semantic First, Details Later

**NEVER start by reading files or grepping.** Always use semantic search first.

```
WRONG: Read the file to understand it
RIGHT: search_code("what does this do") → then read specific parts
```

## Tool Priority (Use in This Order)

### 1. DISCOVER: Semantic Search (Start Here)
```
search_code(query="authentication logic", brief=true, limit=10)
```
- Use natural language: "function that validates tokens"
- Use `brief=true` for exploration (saves tokens)
- Returns summaries, not code

### 2. NARROW: Get Specific Elements
```
get_element(element_id="...", include_code=true)
```
- Only after you found relevant elements via search
- Use `include_code=true` only when you need implementation

### 3. TRACE: Find Relationships
```
find_usages(element_id="...")      # Where is this called?
find_implementations(class_name="Protocol")  # What implements this?
get_call_graph(element_id="...")   # Callers and callees
```
- Use for refactoring impact analysis
- Use before modifying shared code

### 4. PATTERN: Literal Search (Last Resort)
```
grep_code(pattern="\\.add_job\\(", context_lines=2)
```
- Only for exact patterns semantic search can't find
- Regex, specific strings, symbol occurrences

## Token Efficiency Rules

| DO | DON'T |
|----|-------|
| `search_code(brief=true)` | `search_code()` with full summaries |
| `get_element(one_id)` | `batch_get_elements(many_ids)` |
| `read_file(start_line=10, end_line=20)` | `read_file()` entire file |
| Search → narrow → read | Read everything then search |

## Workflow Examples

### "How does X work?"
```
1. search_code("X functionality", brief=true)
2. get_element(best_match, include_code=true)
3. get_context(element_id) if need surrounding code
```

### "Find all places that call X"
```
1. search_code("X", element_types=["function"])
2. find_usages(element_id)
```

### "What implements interface Y?"
```
1. find_implementations(class_name="Y")
2. get_element(each implementation) for details
```

### "Refactor function Z"
```
1. search_code("Z")
2. find_usages(element_id)  # Impact analysis
3. get_call_graph(element_id)  # Dependencies
4. THEN make changes
```

## Anti-Patterns (Never Do These)

1. **Don't grep first** - Wastes tokens, returns noise
2. **Don't read whole files** - Use line ranges or get_element
3. **Don't skip semantic search** - It's your best tool
4. **Don't ignore summaries** - They're pre-computed understanding
5. **Don't batch when you can iterate** - One element at a time

## Available Tools Reference

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `search_code` | Semantic search for code | First step, always |
| `search_features` | Find high-level capabilities | Understanding architecture |
| `get_element` | Get one element's details | After search found it |
| `get_context` | See element in its surroundings | Understanding hierarchy |
| `find_usages` | Where is this used? | Before refactoring |
| `find_implementations` | What implements this? | Finding concrete classes |
| `get_call_graph` | Callers and callees | Dependency analysis |
| `grep_code` | Regex pattern search | Literal matches only |
| `read_file` | Get file contents | Last resort, use line ranges |

## Remember

The index has already done the hard work:
- Code is parsed and structured
- Summaries explain what code does
- Embeddings enable semantic search

**Use the index. Don't re-read what's already understood.**
'''

    result = {
        "skill_name": skill_name,
        "content": skill_content,
        "version": "1.0.0",
        "scope": scope,
    }

    # Determine target path based on scope
    if scope == "global":
        skill_dir = Path.home() / ".claude" / "skills" / skill_name
        skill_path = skill_dir / "SKILL.md"
    elif scope == "project":
        if not project_root:
            result["error"] = "project_root is required for scope='project'"
            return result
        skill_dir = Path(project_root) / ".claude" / "skills" / skill_name
        skill_path = skill_dir / "SKILL.md"
    else:
        result["error"] = f"Invalid scope '{scope}'. Use 'project' or 'global'."
        return result

    # Check for existing skill in both locations to avoid duplication
    global_path = Path.home() / ".claude" / "skills" / skill_name / "SKILL.md"
    project_path = Path(project_root) / ".claude" / "skills" / skill_name / "SKILL.md" if project_root else None

    if skill_path.exists():
        result["skipped"] = True
        result["reason"] = f"Skill already exists at: {skill_path}"
        result["path"] = str(skill_path)
        return result

    # Warn if exists in the other location
    if scope == "project" and global_path.exists():
        result["warning"] = f"Note: Skill also exists globally at {global_path}"
    elif scope == "global" and project_path and project_path.exists():
        result["warning"] = f"Note: Skill also exists in project at {project_path}"

    # Write the skill file
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(skill_content)
    result["written_to"] = str(skill_path)

    return result


def get_call_graph(
    repo_root: str,
    es: ElasticsearchRepository,
    element_id: str,
    direction: str = "both",
) -> dict[str, Any]:
    """Get callers and/or callees of a function/method.

    Args:
        repo_root: Repository root path.
        es: Elasticsearch repository.
        element_id: Function/method element ID.
        direction: 'callers', 'callees', or 'both'.

    Returns:
        Call graph with callers and callees lists.
    """
    doc = es.get_document(element_id)
    if not doc:
        raise ValueError(f"Element not found: {element_id}")

    name = doc.get("name")
    element_type = doc.get("element_type")
    raw_code = doc.get("raw_code", "")
    defining_file = doc.get("relative_path")

    result: dict[str, Any] = {
        "element": {
            "name": name,
            "type": element_type,
            "file": defining_file,
        },
        "callers": [],
        "callees": [],
    }

    # Find callers (who calls this function)
    if direction in ("callers", "both"):
        usages = find_usages(repo_root, es, element_id, limit=20)
        for usage in usages:
            result["callers"].append({
                "file": usage["file"],
                "line": usage["line"],
                "content": usage["content"],
            })

    # Find callees (what this function calls)
    if direction in ("callees", "both") and raw_code:
        # Extract function/method calls from the code
        import re

        # Match function calls: name( but not def name(
        call_pattern = r"(?<!def\s)(?<!class\s)\b(\w+)\s*\("
        calls = re.findall(call_pattern, raw_code)

        # Deduplicate and filter builtins
        builtins = {"print", "len", "str", "int", "float", "list", "dict", "set",
                    "range", "enumerate", "zip", "map", "filter", "sorted", "type",
                    "isinstance", "hasattr", "getattr", "setattr", "super", "open"}

        seen = set()
        for call_name in calls:
            if call_name in seen or call_name in builtins:
                continue
            seen.add(call_name)

            result["callees"].append({
                "name": call_name,
                "type": "call",
            })

    return result
