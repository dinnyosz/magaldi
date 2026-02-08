"""Element navigation and inspection tools."""

from __future__ import annotations

import logging
from typing import Any

from shared.db.store import Repository

from ._utils import _resolve_scope_repo

logger = logging.getLogger(__name__)


def _find_file_element(
    es: Repository,
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
    es: Repository,
    parent_id: str,
) -> list[dict[str, Any]]:
    """Find all children of an element."""
    client = es._get_client()
    result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {"term": {"parent_id": parent_id}},
            "size": 100,
            "sort": [{"line_start": "asc"}],
        },
    )

    return [hit["_source"] for hit in result.get("hits", {}).get("hits", [])]


def _find_elements_in_file(
    es: Repository,
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


def get_element(
    es: Repository,
    element_id: str,
    include_code: bool = False,
    brief: bool = True,
) -> dict[str, Any]:
    """Get details of a code element.

    Args:
        es: Elasticsearch repository.
        element_id: Element ID.
        include_code: Include raw source code.
        brief: Return only core fields (default True). Set False for full details.

    Returns:
        Element details.
    """
    doc = es.get_document_by_id_or_hash(element_id)
    if not doc:
        raise ValueError(f"Element not found: {element_id}")

    # Core fields (always included)
    result: dict[str, Any] = {
        "hash_id": doc.get("hash_id"),
        "name": doc.get("name"),
        "type": doc.get("element_type"),
        "file": doc.get("relative_path"),
        "line_start": doc.get("line_start"),
        "line_end": doc.get("line_end"),
        "summary": doc.get("summary", ""),
    }

    # Brief mode: signature, visibility, and a few key fields
    if doc.get("signature"):
        result["signature"] = doc["signature"]
    if doc.get("visibility"):
        result["visibility"] = doc["visibility"]
    if doc.get("is_async"):
        result["is_async"] = True
    if doc.get("is_test"):
        result["is_test"] = True

    # Return early for brief mode
    if brief:
        if include_code:
            result["code"] = doc.get("raw_code", "")
        return result

    # Full mode: include all available fields
    if doc.get("docstring"):
        result["docstring"] = doc["docstring"]
    if doc.get("decorators"):
        result["decorators"] = doc["decorators"]
    if doc.get("decorator_details"):
        result["decorator_details"] = doc["decorator_details"]
    if doc.get("parent_id"):
        result["parent_id"] = doc["parent_id"]

    # Core metadata fields
    if doc.get("language"):
        result["language"] = doc["language"]
    if doc.get("scope"):
        result["scope"] = doc["scope"]
    if doc.get("repository"):
        result["repository"] = doc["repository"]
    if doc.get("level") is not None:
        result["level"] = doc["level"]

    # Enhanced context fields
    if doc.get("base_classes"):
        result["base_classes"] = doc["base_classes"]
    if doc.get("class_attributes"):
        result["class_attributes"] = doc["class_attributes"]
    if doc.get("exceptions_raised"):
        result["exceptions_raised"] = doc["exceptions_raised"]
    if doc.get("attributes_modified"):
        result["attributes_modified"] = doc["attributes_modified"]

    # Function/method return type and parameters
    if doc.get("return_type"):
        result["return_type"] = doc["return_type"]
    if doc.get("parameters"):
        result["parameters"] = doc["parameters"]

    # Imports (for file elements)
    if doc.get("imports"):
        result["imports"] = doc["imports"]

    # Calls (for function/method elements)
    if doc.get("calls"):
        result["calls"] = doc["calls"]

    # Type annotations
    if doc.get("type_annotations"):
        result["type_annotations"] = doc["type_annotations"]

    # Pattern detection
    if doc.get("detected_patterns"):
        result["detected_patterns"] = doc["detected_patterns"]
    if doc.get("pattern_confidence"):
        result["pattern_confidence"] = doc["pattern_confidence"]

    # Documentation artifacts
    if doc.get("todos"):
        result["todos"] = doc["todos"]
    if doc.get("section_markers"):
        result["section_markers"] = doc["section_markers"]
    if doc.get("associated_comments"):
        result["associated_comments"] = doc["associated_comments"]

    # API surface
    if doc.get("http_routes"):
        result["http_routes"] = doc["http_routes"]
    if doc.get("cli_commands"):
        result["cli_commands"] = doc["cli_commands"]

    # Purity and side effects
    if doc.get("purity"):
        result["purity"] = doc["purity"]
    if doc.get("side_effects"):
        result["side_effects"] = doc["side_effects"]
    if doc.get("mutated_state"):
        result["mutated_state"] = doc["mutated_state"]

    # Code metrics
    if doc.get("complexity"):
        result["complexity"] = doc["complexity"]
    if doc.get("code_metrics"):
        result["code_metrics"] = doc["code_metrics"]
    if doc.get("docstring_quality"):
        result["docstring_quality"] = doc["docstring_quality"]

    # Security and environment
    if doc.get("security_issues"):
        result["security_issues"] = doc["security_issues"]
    if doc.get("env_vars"):
        result["env_vars"] = doc["env_vars"]
    if doc.get("concurrency"):
        result["concurrency"] = doc["concurrency"]

    # API surface flag
    if doc.get("is_public_api"):
        result["is_public_api"] = True

    # Context usages (for variables - how they're used)
    if doc.get("context_usages"):
        result["context_usages"] = doc["context_usages"]

    # Aggregated metrics (for files/classes)
    if doc.get("metrics_summary"):
        result["metrics_summary"] = doc["metrics_summary"]

    # Document structure (for markdown/doc file elements)
    if doc.get("document_sections"):
        result["document_sections"] = doc["document_sections"]

    # Pre-computed semantic relationships (for functions/methods)
    if doc.get("semantic_related"):
        result["semantic_related"] = doc["semantic_related"]

    if include_code:
        result["code"] = doc.get("raw_code", "")

    return result


def get_context(
    es: Repository,
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
    doc = es.get_document_by_id_or_hash(element_id)
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


def get_children(
    es: Repository,
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
            "hash_id": c.get("hash_id"),
            "name": c.get("name"),
            "type": c.get("element_type"),
            "line_start": c.get("line_start"),
            "line_end": c.get("line_end"),
            "summary": c.get("summary", ""),
            "signature": c.get("signature", ""),
        }
        for c in children
    ]


def batch_get_elements(
    es: Repository,
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
        # Support both element_id and hash_id formats
        doc = es.get_document_by_id_or_hash(eid)
        if doc:
            entry: dict[str, Any] = {
                "name": doc.get("name"),
                "type": doc.get("element_type"),
                "file": doc.get("relative_path"),
                "line": doc.get("line_start"),
                "summary": doc.get("summary", ""),
                "element_id": doc.get("element_id", eid),  # Use canonical ID from doc
            }
            if doc.get("signature"):
                entry["signature"] = doc["signature"]
            if include_code and doc.get("raw_code"):
                entry["code"] = doc["raw_code"]
            results.append(entry)
    return results


def get_file_structure(
    es: Repository,
    scope: str | None = None,
    repository: str | None = None,
    file_path: str = "",
    username: str = "main",
) -> dict[str, Any]:
    """Get structure of a file (classes, functions, methods, imports).

    Args:
        es: Elasticsearch repository.
        scope: Repository scope (auto-detected from magaldi.yaml if not provided).
        repository: Repository name (auto-detected from magaldi.yaml if not provided).
        file_path: Relative file path.
        username: User branch.

    Returns:
        File structure with nested elements.
    """
    # Auto-detect scope/repository from magaldi.yaml if not provided
    scope, repository = _resolve_scope_repo(scope, repository)
    if not scope or not repository:
        raise ValueError(
            "scope and repository are required. Either provide them explicitly "
            "or create a magaldi.yaml file in your project root."
        )
    if not file_path:
        raise ValueError("file_path is required")

    # Get file element
    file_doc = _find_file_element(es, scope, repository, username, file_path)
    if not file_doc:
        raise ValueError(f"File not found: {file_path}")

    # Get all elements in file
    all_elements = _find_elements_in_file(es, scope, repository, username, file_path)

    # Filter to classes, functions, methods, imports (reduces output significantly)
    structure_types = {"class", "function", "method", "import"}
    elements = [e for e in all_elements if e.get("element_type") in structure_types]

    # Build tree structure (minimal - use get_element for details)
    def build_tree(parent_id: str | None) -> list[dict]:
        children = []
        for elem in elements:
            if elem.get("parent_id") == parent_id:
                node = {
                    "hash_id": elem.get("hash_id"),
                    "name": elem.get("name"),
                    "type": elem.get("element_type"),
                    "line": elem.get("line_start"),
                    "children": build_tree(elem.get("element_id")),
                }
                children.append(node)
        return sorted(children, key=lambda x: x.get("line", 0))

    file_id = file_doc.get("element_id")

    return {
        "file": file_path,
        "language": file_doc.get("language"),
        "structure": build_tree(file_id),
        "counts": {
            "classes": sum(1 for e in elements if e.get("element_type") == "class"),
            "functions": sum(1 for e in elements if e.get("element_type") == "function"),
            "methods": sum(1 for e in elements if e.get("element_type") == "method"),
            "imports": sum(1 for e in elements if e.get("element_type") == "import"),
        },
    }
