"""Element analysis tools."""

from __future__ import annotations

from typing import Any

from shared.db.store import Repository


def explain_element(
    es: Repository,
    element_id: str,
) -> dict[str, Any]:
    """Comprehensive overview of a code element.

    Provides a complete picture of an element including:
    - Basic info (name, type, file, signature, summary)
    - Who calls it (callers)
    - What it calls (callees)
    - Imports (if file element)
    - Similar code (top 3 similar)
    - Parent context (class or file)

    Args:
        es: Elasticsearch repository.
        element_id: Element ID to explain.

    Returns:
        Dict with element, callers, callees, imports, similar_code, parent.

    Raises:
        ValueError: If element not found.
    """
    doc = es.get_document_by_id_or_hash(element_id)
    if not doc:
        raise ValueError(f"Element not found: {element_id}")

    name = doc.get("name")
    element_type = doc.get("element_type")
    relative_path = doc.get("relative_path")
    line_start = doc.get("line_start")
    scope = doc.get("scope")
    repository = doc.get("repository")
    username = doc.get("username", "main")
    # Get actual element_id from doc (calls.resolved_id stores element_id format)
    target_element_id = doc.get("element_id")

    element_info: dict[str, Any] = {
        "element_id": target_element_id,
        "hash_id": doc.get("hash_id"),
        "name": name,
        "type": element_type,
        "file": relative_path,
        "line": line_start,
        "signature": doc.get("signature"),
        "summary": doc.get("summary", ""),
        "docstring": doc.get("docstring"),
        "decorators": doc.get("decorators", []),
        "is_test": doc.get("is_test", False),
    }

    # Add enhanced context fields if present
    if doc.get("decorator_details"):
        element_info["decorator_details"] = doc["decorator_details"]
    if doc.get("base_classes"):
        element_info["base_classes"] = doc["base_classes"]
    if doc.get("class_attributes"):
        element_info["class_attributes"] = doc["class_attributes"]
    if doc.get("exceptions_raised"):
        element_info["exceptions_raised"] = doc["exceptions_raised"]
    if doc.get("attributes_modified"):
        element_info["attributes_modified"] = doc["attributes_modified"]
    if doc.get("is_async"):
        element_info["is_async"] = True

    result: dict[str, Any] = {
        "element": element_info,
        "callers": [],
        "callees": [],
        "imports": [],
        "similar_code": [],
        "parent": None,
    }

    # Get callers (who calls this function/method)
    if element_type in ("function", "method"):
        callers = es.find_elements_calling(
            target_id=target_element_id,
            scope=scope,
            repository=repository,
            username=username,
            limit=5,  # Top 5 callers
        )
        for caller in callers:
            result["callers"].append(
                {
                    "element_id": caller.get("element_id"),
                    "hash_id": caller.get("hash_id"),
                    "name": caller.get("name"),
                    "type": caller.get("element_type"),
                    "file": caller.get("relative_path"),
                    "line": caller.get("line_start"),
                    "summary": caller.get("summary", ""),
                }
            )

        # Get callees (what this function/method calls)
        calls = es.get_calls(element_id)
        for call in calls:
            callee_entry: dict[str, Any] = {
                "name": call.get("name"),
                "receiver": call.get("receiver"),
                "line": call.get("line"),
            }
            resolved_id = call.get("resolved_id")
            if resolved_id:
                callee_entry["element_id"] = resolved_id
                resolved_doc = es.get_document(resolved_id)
                if resolved_doc:
                    callee_entry["hash_id"] = resolved_doc.get("hash_id")
                    callee_entry["type"] = resolved_doc.get("element_type")
                    callee_entry["file"] = resolved_doc.get("relative_path")
                    callee_entry["target_line"] = resolved_doc.get("line_start")
                    callee_entry["summary"] = resolved_doc.get("summary", "")
            result["callees"].append(callee_entry)

    # Get imports (if file element)
    if element_type == "file":
        imports = es.get_imports(element_id)
        result["imports"] = imports

    # Get similar code (top 3)
    summary_embedding = doc.get("summary_embedding")
    if summary_embedding:
        similar_results = es.search_by_vector(
            embedding=summary_embedding,
            embedding_type="summary",
            scope=scope,
            repository=repository,
            username=username,
            size=4,  # Get 4 to filter out self
        )
        for sim in similar_results:
            if sim.get("element_id") == element_id:
                continue  # Skip self
            result["similar_code"].append(
                {
                    "element_id": sim.get("element_id"),
                    "hash_id": sim.get("hash_id"),
                    "name": sim.get("name"),
                    "type": sim.get("element_type"),
                    "file": sim.get("relative_path"),
                    "line": sim.get("line_start"),
                    "summary": sim.get("summary", ""),
                    "similarity": sim.get("_score", 0),
                }
            )
            if len(result["similar_code"]) >= 3:
                break

    # Get parent context
    parent_id = doc.get("parent_id")
    if parent_id:
        parent_doc = es.get_document(parent_id)
        if parent_doc:
            result["parent"] = {
                "element_id": parent_id,
                "hash_id": parent_doc.get("hash_id"),
                "name": parent_doc.get("name"),
                "type": parent_doc.get("element_type"),
                "file": parent_doc.get("relative_path"),
                "line": parent_doc.get("line_start"),
                "summary": parent_doc.get("summary", ""),
            }

    return result
