"""Usage and implementation finding tools."""

from __future__ import annotations

import logging
import re
from typing import Any

from shared.db.store import Repository

from ._utils import _escape_for_lucene_regexp, _resolve_scope_repo

logger = logging.getLogger(__name__)


def find_usages(
    repo: Repository,
    element_id: str,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Find where an element is used/called/referenced.

    Searches indexed code in search backend using regexp search - no filesystem access needed.

    Args:
        repo: Search repository.
        element_id: Element to find usages of.
        limit: Maximum usages to return.

    Returns:
        List of usage locations with context.
    """
    limit = max(1, min(limit, 100))

    # Get the element to find its name (supports both element_id and hash_id)
    doc = repo.get_document_by_id_or_hash(element_id)
    if not doc:
        raise ValueError(f"Element not found: {element_id}")

    name = doc.get("name")
    element_type = doc.get("element_type")
    defining_file = doc.get("relative_path")
    defining_line = doc.get("line_start")
    scope = doc.get("scope")
    repository = doc.get("repository")
    username = doc.get("username", "main")

    # Escape name for Lucene regexp
    escaped_name = _escape_for_lucene_regexp(name)

    # Build Lucene regexp pattern based on element type
    # Note: .* in Lucene matches any string, \\( is literal paren
    if element_type == "function":
        # Function calls: name followed by optional whitespace then paren
        # Lucene: name.*\( (dots match any char, .* matches any string)
        pattern = f"{escaped_name}.*\\("
    elif element_type == "method":
        # Method calls: .name( (dot then name then paren)
        pattern = f"\\.{escaped_name}.*\\("
    elif element_type == "class":
        # Class references: just the name (word boundaries are tricky in Lucene)
        pattern = escaped_name
    else:
        # Generic: just the name
        pattern = escaped_name

    results = repo.search_by_regexp(
        pattern=pattern,
        scope=scope,
        repository=repository,
        username=username,
        glob="*.py",  # TODO: detect language from element
        size=limit + 10,  # Get extra to filter out definition
        include_tests=True,
    )

    # Filter out the definition itself and process results
    usages = []
    for result in results:
        result_file = result.get("relative_path")
        result_line = result.get("line_start")
        raw_code = result.get("raw_code", "")

        # Skip if it's the defining element itself (same file and line)
        if result_file == defining_file and result_line == defining_line:
            continue

        # Get the first line of raw_code as content
        lines = raw_code.splitlines() if raw_code else []
        content = lines[0] if lines else ""

        # For functions/methods, skip if this element IS a definition of the target
        # (i.e., if the first line defines a function/method with the same name)
        # This catches cases where a function has the same name in different files
        content_stripped = content.strip()
        if element_type == "function":
            # Skip only if this is defining the SAME function
            if content_stripped.startswith(f"def {name}(") or content_stripped.startswith(
                f"def {name} ("
            ):
                continue
        elif element_type == "class":
            if content_stripped.startswith(f"class {name}(") or content_stripped.startswith(
                f"class {name}:"
            ):
                continue
        elif element_type == "method":
            if content_stripped.startswith(f"def {name}(") or content_stripped.startswith(
                f"def {name} ("
            ):
                continue

        # Build context from raw_code lines
        context_before = []  # Empty for now (we only have the element, not surrounding code)
        context_after = lines[1:2] if len(lines) > 1 else []  # Second line if exists

        usages.append(
            {
                "file": result_file,
                "line": result_line,
                "content": content,
                "context_before": context_before,
                "context_after": context_after,
            }
        )

        if len(usages) >= limit:
            break

    return usages


def find_implementations(
    repo: Repository,
    element_id: str | None = None,
    class_name: str | None = None,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Find classes that implement/inherit from a protocol or base class.

    Searches indexed code in search backend using regexp search - no filesystem access needed.

    Args:
        repo: Search repository.
        element_id: Element ID of the protocol/base class.
        class_name: Or just the class name to search for.
        scope: Filter by scope.
        repository: Filter by repository.
        username: User branch to search.
        limit: Maximum implementations to return.

    Returns:
        List of implementing classes with their info.
    """
    scope, repository = _resolve_scope_repo(scope, repository)
    limit = max(1, min(limit, 100))

    # Get the name and scope/repo to search for
    if element_id:
        doc = repo.get_document_by_id_or_hash(element_id)
        if not doc:
            raise ValueError(f"Element not found: {element_id}")
        name = doc.get("name")
        scope = scope or doc.get("scope")
        repository = repository or doc.get("repository")
        username = doc.get("username", "main")
    elif class_name:
        name = class_name
    else:
        raise ValueError("Either element_id or class_name required")

    # Escape name for Lucene regexp
    escaped_name = _escape_for_lucene_regexp(name)

    # Build Lucene regexp pattern for class inheritance
    # Pattern: class.*\(.*{name}.*\) - matches class definitions with name in parents
    # Note: Lucene regexp uses .* for any string, \\( for literal paren
    pattern = f"class.*\\(.*{escaped_name}.*\\)"

    results = repo.search_by_regexp(
        pattern=pattern,
        scope=scope,
        repository=repository,
        username=username,
        glob="*.py",  # TODO: support other languages
        size=limit,
        include_tests=True,
    )

    # Process results and extract implementing class names
    implementations = []
    for result in results:
        # Skip the base class itself
        if result.get("name") == name:
            continue

        raw_code = result.get("raw_code", "")
        lines = raw_code.splitlines() if raw_code else []
        first_line = lines[0] if lines else ""

        # Extract class name from the first line
        class_match = re.search(r"class\s+(\w+)", first_line)
        impl_name = class_match.group(1) if class_match else "Unknown"

        # Get context (second line if exists)
        context_after = lines[1:3] if len(lines) > 1 else []

        implementations.append(
            {
                "class_name": impl_name,
                "file": result.get("relative_path"),
                "line": result.get("line_start"),
                "definition": first_line.strip(),
                "context_after": context_after,
            }
        )

    return implementations
