"""Cross-file call resolution.

This module resolves function/method calls that reference elements in other files,
using import information, type annotations, and embedding similarity.

Strategy 1-2 (at parse time, in parsers/base.py):
- Same-file bare function calls
- Self-method calls (self.method(), this.method(), $this->method())

Strategy 3-5 (this module, resolve_all_calls):
- Import-based calls (from utils import process; process())
- Module method calls (import utils; utils.process())
- Type-annotated calls (repo: Repository; repo.get_document())

Strategy 6 (this module, resolve_calls_by_embedding):
- Embedding similarity for remaining untyped calls with receiver

Semantic relationships (compute_semantic_relationships):
- Pre-compute top-K similar functions for each element via vector similarity
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.db.repositories import Repository

logger = logging.getLogger(__name__)


def resolve_all_calls(
    repo: Repository,
    scope: str,
    repository: str,
    username: str = "main",
) -> tuple[int, int, int]:
    """Full call resolution pass - re-resolves ALL calls in the repository.

    Used during partial parsing to ensure call graphs are complete even when
    only some files were re-parsed. Clears existing resolved_ids and re-resolves
    to handle renamed/moved functions.

    Args:
        repo: Repository instance.
        scope: Repository scope.
        repository: Repository name.
        username: Username branch.

    Returns:
        Tuple of (total_calls_processed, import_resolved, type_resolved).
    """
    total_processed = 0
    import_resolved = 0
    type_resolved = 0

    # Get ALL elements with calls (not just unresolved)
    elements = repo.find_all_elements_with_calls(scope, repository, username)
    logger.info(f"Full resolution: found {len(elements)} elements with calls")

    for elem in elements:
        element_id = elem.get("element_id", "")
        relative_path = elem.get("relative_path", "")
        parameters = elem.get("parameters", [])

        # Build param type map for type-based resolution
        param_types: dict[str, str] = {}
        if parameters:
            for p in parameters:
                if p.get("type"):
                    param_types[p["name"]] = p["type"]

        # Get file's imports
        file_imports = repo.get_file_imports(relative_path, scope, repository, username)
        import_map = _build_import_map(file_imports) if file_imports else {}

        calls = elem.get("calls", [])
        updated = False

        for call in calls:
            total_processed += 1
            receiver = call.get("receiver")
            name = call.get("name")
            category = call.get("category", "unknown")

            # Clear existing resolved_id to re-resolve
            old_resolved_id = call.get("resolved_id")
            resolved_id = None

            # Strategy 3: Bare call matching an import
            if receiver is None and name in import_map:
                import_info = import_map[name]
                resolved_id = _lookup_element_by_import(
                    repo, import_info, name, scope, repository, username,
                    caller_path=relative_path,
                )
                if resolved_id:
                    import_resolved += 1

            # Strategy 4: Method call on imported module
            elif receiver and receiver in import_map:
                import_info = import_map[receiver]
                resolved_id = _lookup_element_by_import(
                    repo, import_info, name, scope, repository, username,
                    caller_path=relative_path,
                )
                if resolved_id:
                    import_resolved += 1

            # Strategy 5: Type-annotated method call
            elif receiver and category == "type_resolvable" and receiver in param_types:
                type_name = param_types[receiver]
                resolved_id = _lookup_method_by_type(
                    repo, type_name, name, scope, repository, username
                )
                if resolved_id:
                    type_resolved += 1
                    call["category"] = "resolved"

            # Update if changed
            if resolved_id != old_resolved_id:
                call["resolved_id"] = resolved_id
                updated = True

        if updated:
            repo.store_calls(element_id, calls)

    # Strategy 5.5: Return-type propagation
    # Re-fetch elements to get updated calls after strategies 3-5
    elements = repo.find_all_elements_with_calls(scope, repository, username)
    return_type_count = _resolve_via_return_types(
        repo, elements, scope, repository, username
    )

    total_resolved = import_resolved + type_resolved + return_type_count
    logger.info(
        f"Full resolution: resolved {total_resolved}/{total_processed} calls "
        f"({import_resolved} via imports, {type_resolved} via type annotations, "
        f"{return_type_count} via return-type propagation)"
    )
    return total_processed, import_resolved, type_resolved


def resolve_cross_file_calls(
    repo: Repository,
    scope: str,
    repository: str,
    username: str = "main",
) -> tuple[int, int, int]:
    """Resolve calls using imports and indexed elements.

    This is Phase 2 of call resolution, run after all files are parsed and stored.
    It uses import information and type annotations to resolve calls.

    NOTE: For partial parsing (only changed files), use resolve_all_calls() instead
    to ensure call graphs are complete.

    Args:
        repo: Repository instance.
        scope: Repository scope.
        repository: Repository name.
        username: Username branch.

    Returns:
        Tuple of (total_unresolved_calls_processed, import_resolved, type_resolved).
    """
    total_processed = 0
    import_resolved = 0
    type_resolved = 0

    # Get all elements with unresolved calls
    elements = repo.find_elements_with_unresolved_calls(scope, repository, username)
    logger.info(f"Found {len(elements)} elements with unresolved calls")

    for elem in elements:
        element_id = elem.get("element_id", "")
        relative_path = elem.get("relative_path", "")
        parameters = elem.get("parameters", [])

        # Build param type map for type-based resolution
        param_types: dict[str, str] = {}
        if parameters:
            for p in parameters:
                if p.get("type"):
                    param_types[p["name"]] = p["type"]

        # Get file's imports
        file_imports = repo.get_file_imports(relative_path, scope, repository, username)
        import_map = _build_import_map(file_imports) if file_imports else {}

        calls = elem.get("calls", [])
        updated = False

        for call in calls:
            if call.get("resolved_id"):
                continue

            total_processed += 1
            receiver = call.get("receiver")
            name = call.get("name")
            category = call.get("category", "unknown")

            resolved_id = None

            # Strategy 3: Bare call matching an import
            # e.g., from utils import process; process()
            if receiver is None and name in import_map:
                import_info = import_map[name]
                resolved_id = _lookup_element_by_import(
                    repo, import_info, name, scope, repository, username,
                    caller_path=relative_path,
                )
                if resolved_id:
                    import_resolved += 1

            # Strategy 4: Method call on imported module
            # e.g., import utils; utils.process()
            elif receiver and receiver in import_map:
                import_info = import_map[receiver]
                resolved_id = _lookup_element_by_import(
                    repo, import_info, name, scope, repository, username,
                    caller_path=relative_path,
                )
                if resolved_id:
                    import_resolved += 1

            # Strategy 5: Type-annotated method call
            # e.g., def foo(repo: Repository): repo.get_document()
            elif receiver and category == "type_resolvable" and receiver in param_types:
                type_name = param_types[receiver]
                resolved_id = _lookup_method_by_type(
                    repo, type_name, name, scope, repository, username
                )
                if resolved_id:
                    type_resolved += 1
                    call["category"] = "resolved"

            if resolved_id:
                call["resolved_id"] = resolved_id
                updated = True

        if updated:
            repo.store_calls(element_id, calls)

    # Strategy 5.5: Return-type propagation
    # Re-fetch elements to get updated calls after strategies 3-5
    all_elements = repo.find_all_elements_with_calls(scope, repository, username)
    return_type_count = _resolve_via_return_types(
        repo, all_elements, scope, repository, username
    )

    total_resolved = import_resolved + type_resolved + return_type_count
    logger.info(
        f"Resolved {total_resolved}/{total_processed} cross-file calls "
        f"({import_resolved} via imports, {type_resolved} via type annotations, "
        f"{return_type_count} via return-type propagation)"
    )
    return total_processed, import_resolved, type_resolved


def _build_import_map(imports: list[dict]) -> dict[str, dict]:
    """Build a map from local name to import info.

    Args:
        imports: List of import dicts with keys: name, module, alias, line.

    Returns:
        Dict mapping local name (alias or name) to full import info.
    """
    result: dict[str, dict] = {}
    for imp in imports:
        # Use alias if available, otherwise use the imported name
        local_name = imp.get("alias") or imp.get("name")
        if local_name:
            result[local_name] = imp
    return result


def _lookup_element_by_import(
    repo: Repository,
    import_info: dict,
    element_name: str,
    scope: str,
    repository: str,
    username: str,
    caller_path: str | None = None,
) -> str | None:
    """Look up element ID for an imported name.

    Args:
        repo: Repository.
        import_info: Import dict with module info.
        element_name: Name of the element to find.
        scope: Repository scope.
        repository: Repository name.
        username: Username branch.
        caller_path: Relative path of the calling file, needed for
            resolving relative imports.

    Returns:
        Element ID if found, None otherwise.
    """
    module = import_info.get("module", "")
    if not module:
        return None

    # Convert module path to possible file paths
    # e.g., "magaldi_core.storage" -> "src/magaldi_core/storage.py"
    # e.g., ".utils" -> relative import (needs caller_path)
    # e.g., "os" -> external module (skip)

    # Skip external/stdlib modules (no dots or starts with common stdlib names)
    if _is_external_module(module):
        return None

    # Try to find the element
    possible_paths = _module_to_file_paths(module, caller_path)

    for file_path in possible_paths:
        # Look for function with this name in the file
        element_id = _find_element_in_file(
            repo, file_path, element_name, scope, repository, username
        )
        if element_id:
            return element_id

    return None


def _is_external_module(module: str) -> bool:
    """Check if module is external (stdlib or third-party).

    Returns True for modules that aren't part of the local codebase.
    """
    # Relative imports are always internal
    if module.startswith("."):
        return False

    # Common stdlib modules
    stdlib_prefixes = {
        "os", "sys", "re", "json", "math", "time", "datetime", "collections",
        "itertools", "functools", "pathlib", "typing", "logging", "unittest",
        "dataclasses", "abc", "asyncio", "hashlib", "base64", "uuid", "copy",
        "io", "tempfile", "shutil", "subprocess", "threading", "multiprocessing",
    }

    # Check if first component is stdlib
    first_component = module.split(".")[0]
    if first_component in stdlib_prefixes:
        return True

    # Common third-party modules
    third_party_prefixes = {
        "numpy", "pandas", "requests", "flask", "django", "fastapi", "pydantic",
        "pytest", "elasticsearch", "redis", "sqlalchemy", "boto3", "torch",
        "tensorflow", "sklearn", "scipy", "matplotlib", "pillow", "PIL",
        "litellm", "openai", "anthropic", "tiktoken", "httpx", "aiohttp",
        "tree_sitter", "mcp", "click", "typer", "rich", "tqdm",
    }

    return first_component in third_party_prefixes


def _module_to_file_paths(
    module: str, caller_path: str | None = None
) -> list[str]:
    """Convert module path to possible file paths.

    Args:
        module: Module path like "magaldi_core.storage" or ".utils".
        caller_path: Relative path of the file making the import,
            needed for resolving relative imports.

    Returns:
        List of possible file paths to try.
    """
    paths: list[str] = []

    if module.startswith("."):
        if not caller_path:
            return paths

        # Count leading dots to determine relative depth
        # 1 dot = current package, 2 dots = parent package, etc.
        dots = 0
        for ch in module:
            if ch == ".":
                dots += 1
            else:
                break

        # Get base directory from caller's path
        from pathlib import PurePosixPath

        caller_dir = str(PurePosixPath(caller_path).parent)

        # Go up (dots - 1) directories
        base_dir = caller_dir
        for _ in range(dots - 1):
            parent = str(PurePosixPath(base_dir).parent)
            if parent == base_dir:
                break
            base_dir = parent

        # Get the remaining module path after dots
        remaining = module[dots:]  # e.g., "utils" from ".utils"

        if remaining:
            sub_path = remaining.replace(".", "/")
            paths.append(f"{base_dir}/{sub_path}.py")
            paths.append(f"{base_dir}/{sub_path}/__init__.py")
        else:
            # from . import foo -> look in current package's __init__.py
            paths.append(f"{base_dir}/__init__.py")

        return paths

    # Convert dots to path separators
    # e.g., "magaldi_core.storage" -> "magaldi_core/storage"
    path_base = module.replace(".", "/")

    # Try common source directories
    for src_prefix in ["src/", ""]:
        # Try as Python file
        paths.append(f"{src_prefix}{path_base}.py")
        # Try as package __init__
        paths.append(f"{src_prefix}{path_base}/__init__.py")

    return paths


def _unwrap_type(type_name: str) -> str:
    """Unwrap wrapper types to get the base class name.

    Handles Optional, Union, generics, and qualified names.

    Examples:
        "Optional[Repository]" -> "Repository"
        "Union[Repository, None]" -> "Repository"
        "list[str]" -> "list"
        "db.Repository" -> "Repository"
        "Repository" -> "Repository"
    """
    stripped = type_name.strip()

    # Strip Optional[] wrapper
    if stripped.startswith("Optional[") and stripped.endswith("]"):
        stripped = stripped[9:-1].strip()

    # Strip Union[X, None] pattern (common for Optional in older Python)
    if stripped.startswith("Union[") and stripped.endswith("]"):
        inner = stripped[6:-1]
        parts = [p.strip() for p in inner.split(",")]
        non_none = [p for p in parts if p != "None"]
        if len(non_none) == 1:
            stripped = non_none[0]

    # Strip remaining generics: "list[str]" -> "list"
    base_type = stripped.split("[")[0].strip()

    # Handle qualified names: "db.Repository" -> "Repository"
    if "." in base_type:
        base_type = base_type.split(".")[-1]

    return base_type


def _lookup_method_by_type(
    repo: Repository,
    type_name: str,
    method_name: str,
    scope: str,
    repository: str,
    username: str,
) -> str | None:
    """Look up a method by the type of its receiver.

    Resolves calls like `es.get_document()` when `repo: Repository`.

    Args:
        repo: Repository.
        type_name: Type annotation of the receiver (e.g., "Repository").
        method_name: Name of the method to find.
        scope: Repository scope.
        repository: Repository name.
        username: Username branch.

    Returns:
        Element ID of the method if found, None otherwise.
    """
    base_type = _unwrap_type(type_name)

    # Look up the class by name (without path since we only have the type name)
    class_doc = repo.get_document_by_name_only(
        name=base_type,
        element_type="class",
        scope=scope,
        repository=repository,
        username=username,
    )

    if not class_doc:
        return None

    class_id = class_doc.get("element_id")
    if not class_id:
        return None

    # Look up the method on this class
    method_doc = repo.get_method_by_class(
        class_id=class_id,
        method_name=method_name,
        scope=scope,
        repository=repository,
        username=username,
    )

    # Fallback: try base classes (single-level inheritance)
    if not method_doc:
        base_classes = class_doc.get("base_classes") or []
        for base_name in base_classes:
            base_class_doc = repo.get_document_by_name_only(
                name=base_name,
                element_type="class",
                scope=scope,
                repository=repository,
                username=username,
            )
            if base_class_doc and base_class_doc.get("element_id"):
                method_doc = repo.get_method_by_class(
                    class_id=base_class_doc["element_id"],
                    method_name=method_name,
                    scope=scope,
                    repository=repository,
                    username=username,
                )
                if method_doc:
                    break

    if method_doc:
        return method_doc.get("element_id")  # type: ignore[no-any-return]

    return None


_ASSIGNMENT_PATTERN = re.compile(r"(\w+)\s*=\s*(?:await\s+)?(\w+)\s*\(", re.MULTILINE)


def _build_receiver_type_map(
    raw_code: str,
    resolved_calls: dict[str, str],
    repo: Repository,
) -> dict[str, str]:
    """Build map from variable name to inferred type via return_type.

    For patterns like `result = get_user()`, if get_user is resolved
    and its target has a return_type, maps "result" -> return_type.

    Args:
        raw_code: Source code of the function.
        resolved_calls: Map of call name -> resolved element_id.
        repo: Repository instance.

    Returns:
        Dict mapping variable name to inferred type name.
    """
    type_map: dict[str, str] = {}

    for match in _ASSIGNMENT_PATTERN.finditer(raw_code):
        var_name = match.group(1)
        func_name = match.group(2)

        if func_name in resolved_calls:
            resolved_id = resolved_calls[func_name]
            doc = repo.get_document(resolved_id)
            if doc and doc.get("return_type"):
                type_map[var_name] = doc["return_type"]

    return type_map


def _resolve_via_return_types(
    repo: Repository,
    elements: list[dict],
    scope: str,
    repository: str,
    username: str,
) -> int:
    """Strategy 5.5: Resolve calls using return-type propagation.

    For patterns like `result = get_user(); result.save()`, infers the
    type of `result` from `get_user()`'s return_type, then resolves
    `save()` on that type.

    Args:
        repo: Repository instance.
        elements: Elements with calls (from find_all_elements_with_calls).
        scope: Repository scope.
        repository: Repository name.
        username: Username branch.

    Returns:
        Number of calls resolved via return-type propagation.
    """
    return_type_resolved = 0

    # Find elements that could benefit from return-type propagation:
    # must have both unresolved receiver calls AND resolved bare calls
    candidates: list[str] = []
    for elem in elements:
        calls = elem.get("calls", [])
        has_unresolved_receiver = any(
            c.get("receiver")
            and not c.get("resolved_id")
            and c.get("category") in ("untyped", "unknown", "type_resolvable")
            for c in calls
        )
        has_resolved_bare = any(
            not c.get("receiver") and c.get("resolved_id") for c in calls
        )
        if has_unresolved_receiver and has_resolved_bare:
            candidates.append(elem.get("element_id", ""))

    if not candidates:
        return 0

    logger.info(f"Return-type propagation: {len(candidates)} candidate elements")

    for elem_id in candidates:
        if not elem_id:
            continue

        doc = repo.get_document(elem_id)
        if not doc:
            continue

        raw_code = doc.get("raw_code", "")
        calls = doc.get("calls", [])

        if not raw_code:
            continue

        # Build resolved bare call map: func_name -> resolved_id
        resolved_bare: dict[str, str] = {}
        for c in calls:
            if not c.get("receiver") and c.get("resolved_id"):
                resolved_bare[c["name"]] = c["resolved_id"]

        # Infer types from return types
        inferred_types = _build_receiver_type_map(raw_code, resolved_bare, repo)

        if not inferred_types:
            continue

        # Try to resolve unresolved receiver calls using inferred types
        updated = False
        for call in calls:
            if call.get("resolved_id") or not call.get("receiver"):
                continue
            receiver = call["receiver"]
            if receiver in inferred_types:
                resolved_id = _lookup_method_by_type(
                    repo,
                    inferred_types[receiver],
                    call["name"],
                    scope,
                    repository,
                    username,
                )
                if resolved_id:
                    call["resolved_id"] = resolved_id
                    call["category"] = "return_type_resolved"
                    return_type_resolved += 1
                    updated = True

        if updated:
            repo.store_calls(elem_id, calls)

    if return_type_resolved:
        logger.info(
            f"Return-type propagation: resolved {return_type_resolved} calls"
        )

    return return_type_resolved


def _find_element_in_file(
    repo: Repository,
    file_path: str,
    element_name: str,
    scope: str,
    repository: str,
    username: str,
) -> str | None:
    """Find an element by name in a specific file.

    Args:
        repo: Repository.
        file_path: Relative path to the file.
        element_name: Name of the element to find.
        scope: Repository scope.
        repository: Repository name.
        username: Username branch.

    Returns:
        Element ID if found, None otherwise.
    """
    # Build possible element IDs for function or class
    for elem_type in ["function", "class"]:
        # We don't know the exact line number, so we need to search
        # Try to find by querying
        doc = repo.get_document_by_name(
            name=element_name,
            element_type=elem_type,
            relative_path=file_path,
            scope=scope,
            repository=repository,
            username=username,
        )
        if doc:
            return doc.get("element_id")  # type: ignore[no-any-return]

    return None


# =============================================================================
# Strategy 6: Embedding-based call resolution
# =============================================================================


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two L2-normalized vectors (= dot product)."""
    return sum(x * y for x, y in zip(a, b))


def _merge_candidates(
    user_candidates: list[dict],
    main_candidates: list[dict],
) -> list[dict]:
    """Merge candidates from user and main, user version wins on duplicates."""
    seen: dict[str, dict] = {}
    # Add main first so user overwrites
    for c in main_candidates:
        key = f"{c.get('relative_path')}:{c.get('name')}:{c.get('element_type')}"
        seen[key] = c
    for c in user_candidates:
        key = f"{c.get('relative_path')}:{c.get('name')}:{c.get('element_type')}"
        seen[key] = c  # user wins
    return list(seen.values())


def resolve_calls_by_embedding(
    repo: Repository,
    scope: str,
    repository: str,
    username: str = "main",
    similarity_threshold: float = 0.7,
) -> tuple[int, int, int]:
    """Strategy 6: Resolve untyped calls using embedding similarity.

    For calls with a receiver but no type annotation, finds candidate
    functions/methods by name, then uses embedding cosine similarity
    to pick the best match when multiple candidates exist.

    Queries both the user's index and "main" to get a complete view,
    with user's elements taking priority.

    Args:
        repo: Repository instance.
        scope: Repository scope.
        repository: Repository name.
        username: Username branch.
        similarity_threshold: Minimum cosine similarity to accept a match.

    Returns:
        Tuple of (total_processed, single_match_resolved, embedding_resolved).
    """
    total_processed = 0
    single_resolved = 0
    embedding_resolved = 0

    # Cache: call name -> merged candidate list
    name_cache: dict[str, list[dict]] = {}
    # Cache: element_id -> caller_embedding
    embedding_cache: dict[str, list[float] | None] = {}

    # Get all elements with calls
    elements = repo.find_all_elements_with_calls(scope, repository, username)
    # Also get main's elements if user is not main
    if username != "main":
        main_elements = repo.find_all_elements_with_calls(scope, repository, "main")
        # Merge: user elements win on duplicates
        elem_map: dict[str, dict] = {}
        for e in main_elements:
            elem_map[e.get("element_id", "")] = e
        for e in elements:
            elem_map[e.get("element_id", "")] = e
        elements = list(elem_map.values())

    logger.info(f"Embedding resolution: found {len(elements)} elements with calls")

    for elem in elements:
        element_id = elem.get("element_id", "")
        calls = elem.get("calls", [])
        updated = False

        for call in calls:
            if call.get("resolved_id"):
                continue

            category = call.get("category", "unknown")
            if category not in ("untyped", "unknown"):
                continue

            receiver = call.get("receiver")
            if receiver is None:
                continue  # Bare calls too ambiguous

            name = call.get("name")
            if not name:
                continue

            total_processed += 1

            # Find candidates (cached by name)
            if name not in name_cache:
                candidates = repo.find_candidates_by_name(
                    name, scope, repository, username
                )
                if username != "main":
                    main_candidates = repo.find_candidates_by_name(
                        name, scope, repository, "main"
                    )
                    candidates = _merge_candidates(candidates, main_candidates)
                name_cache[name] = candidates

            candidates = name_cache[name]

            if not candidates:
                continue

            if len(candidates) == 1:
                # Single candidate — resolve directly
                call["resolved_id"] = candidates[0].get("element_id")
                call["category"] = "embedding_resolved"
                single_resolved += 1
                updated = True
                continue

            # Multiple candidates — use embedding similarity
            # Get caller's embedding (cached) — uses caller_embedding for asymmetric matching
            if element_id not in embedding_cache:
                caller_embedding = repo.get_embedding(element_id, "caller")
                embedding_cache[element_id] = caller_embedding
            caller_embedding = embedding_cache[element_id]

            if not caller_embedding:
                continue  # Caller has no embedding

            best_score = -1.0
            best_candidate_id = None

            for candidate in candidates:
                candidate_embedding = candidate.get("summary_embedding")
                if not candidate_embedding:
                    continue

                score = _cosine_similarity(caller_embedding, candidate_embedding)
                if score > best_score:
                    best_score = score
                    best_candidate_id = candidate.get("element_id")

            if best_score >= similarity_threshold and best_candidate_id:
                call["resolved_id"] = best_candidate_id
                call["category"] = "embedding_resolved"
                embedding_resolved += 1
                updated = True

        if updated:
            repo.store_calls(element_id, calls)

    total_resolved = single_resolved + embedding_resolved
    logger.info(
        f"Embedding resolution: resolved {total_resolved}/{total_processed} calls "
        f"({single_resolved} single match, {embedding_resolved} via similarity)"
    )
    return total_processed, single_resolved, embedding_resolved


# =============================================================================
# Semantic relationship computation
# =============================================================================


def compute_semantic_relationships(
    repo: Repository,
    scope: str,
    repository: str,
    username: str = "main",
    top_k: int = 10,
    min_score: float = 0.5,
) -> tuple[int, int]:
    """Pre-compute semantic relationships for all functions/methods.

    For each function/method with an embedding, finds the top-K most similar
    elements by vector similarity and stores them on the element.

    Queries both user and main indices for a complete view.

    Args:
        repo: Repository instance.
        scope: Repository scope.
        repository: Repository name.
        username: Username branch.
        top_k: Number of similar elements to store per element.
        min_score: Minimum similarity score to include.

    Returns:
        Tuple of (elements_processed, total_relationships_stored).
    """
    elements_processed = 0
    total_relationships = 0

    # Get all functions/methods with embeddings via direct ES query
    from shared.db.repositories.base import INDEX_NAME

    client = repo._get_client()

    # Build query for all functions/methods in the repo
    filter_clauses: list[dict] = [
        {"term": {"scope": scope}},
        {"term": {"repository": repository}},
        {"terms": {"element_type": ["function", "method"]}},
        {"exists": {"field": "summary_embedding"}},
    ]

    # Query both user and main
    if username and username != "main":
        filter_clauses.append({
            "bool": {
                "should": [
                    {"term": {"username": username}},
                    {"term": {"username": "main"}},
                ],
                "minimum_should_match": 1,
            }
        })
    else:
        filter_clauses.append({"term": {"username": "main"}})

    result = client.search(
        index=INDEX_NAME,
        body={
            "query": {"bool": {"filter": filter_clauses}},
            "size": 10000,
            "_source": [
                "element_id",
                "hash_id",
                "username",
                "name",
                "element_type",
                "relative_path",
                "summary_embedding",
            ],
        },
    )

    all_elements = [hit["_source"] for hit in result.get("hits", {}).get("hits", [])]

    # Deduplicate: user version wins over main
    if username and username != "main":
        seen: dict[str, dict] = {}
        for elem in all_elements:
            key = f"{elem.get('relative_path')}:{elem.get('name')}:{elem.get('element_type')}"
            existing = seen.get(key)
            if existing is None:
                seen[key] = elem
            elif elem.get("username") == username:
                seen[key] = elem  # user wins
        all_elements = list(seen.values())

    logger.info(f"Semantic relationships: processing {len(all_elements)} elements")

    for elem in all_elements:
        element_id = elem.get("element_id")
        embedding = elem.get("summary_embedding")

        if not element_id or not embedding:
            continue

        # Find similar elements via vector search (searches both user + main)
        similar = repo.search_by_vector(
            embedding=embedding,
            scope=scope,
            repository=repository,
            username=None,  # Search all users for complete coverage
            element_types=["function", "method"],
            size=top_k + 5,  # Extra to account for self + filtering
            min_score=min_score,
            embedding_type="summary",
        )

        # Filter out self and build relationship list
        related: list[dict] = []
        for s in similar:
            s_id = s.get("element_id")
            if s_id == element_id:
                continue
            score = s.get("_score", 0.0)
            related.append({
                "element_id": s_id,
                "hash_id": s.get("hash_id", ""),
                "score": round(score, 4),
            })
            if len(related) >= top_k:
                break

        if related:
            repo.store_semantic_related(element_id, related)
            total_relationships += len(related)

        elements_processed += 1

    logger.info(
        f"Semantic relationships: {total_relationships} relationships "
        f"stored across {elements_processed} elements"
    )
    return elements_processed, total_relationships
