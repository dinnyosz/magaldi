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
- Return-type propagation (result = get_user(); result.save())
- Constructor-based inference (repo = Repository(); repo.get())
- Scope-aware binding (conn = db.connect(); conn.cursor(); with/for/except)

Strategy 6 (this module, resolve_calls_by_embedding):
- RRF-scored embedding + name + receiver affinity for untyped calls

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
) -> tuple[int, int, int, int, int]:
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
        Tuple of (total_calls_processed, import_resolved, type_resolved,
        constructor_resolved, scope_resolved).
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
        language = elem.get("language", "python")
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

            # Reset resolved categories back to their base category so
            # strategies 5.5/5.6/5.7 can re-process them on subsequent runs
            if category in (
                "resolved", "return_type_resolved", "constructor_resolved",
                "scope_resolved", "embedding_resolved",
            ):
                if receiver is None:
                    category = "unknown"
                elif param_types and receiver in param_types:
                    category = "type_resolvable"
                else:
                    category = "untyped"
                call["category"] = category

            # Strategy 3: Bare call matching an import
            if receiver is None and name in import_map:
                import_info = import_map[name]
                resolved_id = _lookup_element_by_import(
                    repo, import_info, name, scope, repository, username,
                    caller_path=relative_path,
                    language=language,
                )
                if resolved_id:
                    import_resolved += 1
                    call["category"] = "resolved"

            # Strategy 4: Method call on imported module
            elif receiver and receiver in import_map:
                import_info = import_map[receiver]
                resolved_id = _lookup_element_by_import(
                    repo, import_info, name, scope, repository, username,
                    caller_path=relative_path,
                    language=language,
                )
                if resolved_id:
                    import_resolved += 1
                    call["category"] = "resolved"

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

    # Strategy 5.6: Constructor-based type inference
    # Re-fetch elements to get updated calls after 5.5
    elements = repo.find_all_elements_with_calls(scope, repository, username)
    constructor_count = _resolve_via_constructors(
        repo, elements, scope, repository, username
    )

    # Strategy 5.7: Scope-aware type binding (AST-based)
    # Re-fetch elements to get updated calls after 5.6
    elements = repo.find_all_elements_with_calls(scope, repository, username)
    scope_count = _resolve_via_scope_bindings(
        repo, elements, scope, repository, username
    )

    total_resolved = (
        import_resolved + type_resolved + return_type_count
        + constructor_count + scope_count
    )
    logger.info(
        f"Full resolution: resolved {total_resolved}/{total_processed} calls "
        f"({import_resolved} via imports, {type_resolved} via type annotations, "
        f"{return_type_count} via return-type, {constructor_count} via constructors, "
        f"{scope_count} via scope analysis)"
    )
    return total_processed, import_resolved, type_resolved, constructor_count, scope_count


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
        language = elem.get("language", "python")
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
                    language=language,
                )
                if resolved_id:
                    import_resolved += 1
                    call["category"] = "resolved"

            # Strategy 4: Method call on imported module
            # e.g., import utils; utils.process()
            elif receiver and receiver in import_map:
                import_info = import_map[receiver]
                resolved_id = _lookup_element_by_import(
                    repo, import_info, name, scope, repository, username,
                    caller_path=relative_path,
                    language=language,
                )
                if resolved_id:
                    import_resolved += 1
                    call["category"] = "resolved"

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

    # Strategy 5.6: Constructor-based type inference
    all_elements = repo.find_all_elements_with_calls(scope, repository, username)
    constructor_count = _resolve_via_constructors(
        repo, all_elements, scope, repository, username
    )

    # Strategy 5.7: Scope-aware type binding (AST-based)
    all_elements = repo.find_all_elements_with_calls(scope, repository, username)
    scope_count = _resolve_via_scope_bindings(
        repo, all_elements, scope, repository, username
    )

    total_resolved = (
        import_resolved + type_resolved + return_type_count
        + constructor_count + scope_count
    )
    logger.info(
        f"Resolved {total_resolved}/{total_processed} cross-file calls "
        f"({import_resolved} via imports, {type_resolved} via type annotations, "
        f"{return_type_count} via return-type, {constructor_count} via constructors, "
        f"{scope_count} via scope analysis)"
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
    language: str = "python",
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
        language: Programming language of the importing file.

    Returns:
        Element ID if found, None otherwise.
    """
    from magaldi_core.module_resolver import get_module_resolver

    module = import_info.get("module", "")
    if not module:
        return None

    # Use language-specific module resolver
    resolver = get_module_resolver(language)
    if resolver:
        if resolver.is_external_module(module):
            return None
        possible_paths = resolver.module_to_file_paths(module, caller_path)
    else:
        # Fallback: use Python resolver for unknown languages
        if _is_external_module(module):
            return None
        possible_paths = _module_to_file_paths(module, caller_path)

    for file_path in possible_paths:
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

    # Batch fetch full documents for all candidates
    docs = repo.get_documents_batch(candidates)

    for elem_id in candidates:
        if not elem_id:
            continue

        doc = docs.get(elem_id)
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


# Pattern: var = [await] [module.]ClassName(
# Captures: (var_name, optional_module, ClassName)
_CONSTRUCTOR_PATTERN = re.compile(
    r"(\w+)\s*=\s*(?:await\s+)?(?:\w+\.)?([A-Z]\w*)\s*\(", re.MULTILINE
)

# Names that look like classes but are built-in constructors (not user types)
_BUILTIN_CONSTRUCTORS: set[str] = {
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "RuntimeError", "StopIteration", "NotImplementedError",
    "OSError", "IOError", "FileNotFoundError", "PermissionError",
    "ConnectionError", "TimeoutError", "UnicodeError",
    # Built-in types that are PascalCase
    "True", "False", "None",
}


def _is_likely_class_name(name: str) -> bool:
    """Check if name looks like a class (PascalCase, not all-caps)."""
    if not name or len(name) < 2:
        return False
    return name[0].isupper() and not name.isupper()


def _build_constructor_type_map(raw_code: str) -> dict[str, str]:
    """Build map from variable name to class name via constructor calls.

    For patterns like `repo = Repository()`, maps "repo" -> "Repository".

    Args:
        raw_code: Source code of the function.

    Returns:
        Dict mapping variable name to class name.
    """
    type_map: dict[str, str] = {}

    for match in _CONSTRUCTOR_PATTERN.finditer(raw_code):
        var_name = match.group(1)
        class_name = match.group(2)

        if not _is_likely_class_name(class_name):
            continue

        if class_name in _BUILTIN_CONSTRUCTORS:
            continue

        type_map[var_name] = class_name

    return type_map


def _resolve_via_constructors(
    repo: Repository,
    elements: list[dict],
    scope: str,
    repository: str,
    username: str,
) -> int:
    """Strategy 5.6: Resolve calls using constructor-based type inference.

    For patterns like `repo = Repository(); repo.get()`, infers the type
    of `repo` from the constructor call `Repository()`, then resolves
    `get()` on that class.

    Unlike Strategy 5.5, this does NOT require the constructor call to be
    resolved — it uses PascalCase pattern matching to identify constructors.

    Args:
        repo: Repository instance.
        elements: Elements with calls (from find_all_elements_with_calls).
        scope: Repository scope.
        repository: Repository name.
        username: Username branch.

    Returns:
        Number of calls resolved via constructor type inference.
    """
    constructor_resolved = 0

    # Find elements with unresolved receiver calls
    candidates: list[str] = []
    for elem in elements:
        calls = elem.get("calls", [])
        has_unresolved_receiver = any(
            c.get("receiver")
            and not c.get("resolved_id")
            and c.get("category") in ("untyped", "unknown", "type_resolvable")
            for c in calls
        )
        if has_unresolved_receiver:
            candidates.append(elem.get("element_id", ""))

    if not candidates:
        return 0

    logger.info(f"Constructor resolution: {len(candidates)} candidate elements")

    # Batch fetch full documents for all candidates
    docs = repo.get_documents_batch(candidates)

    for elem_id in candidates:
        if not elem_id:
            continue

        doc = docs.get(elem_id)
        if not doc:
            continue

        raw_code = doc.get("raw_code", "")
        calls = doc.get("calls", [])

        if not raw_code:
            continue

        # Build constructor type map: var_name -> ClassName
        constructor_types = _build_constructor_type_map(raw_code)

        if not constructor_types:
            continue

        # Try to resolve unresolved receiver calls using constructor types
        updated = False
        for call in calls:
            if call.get("resolved_id") or not call.get("receiver"):
                continue
            receiver = call["receiver"]
            if receiver in constructor_types:
                resolved_id = _lookup_method_by_type(
                    repo,
                    constructor_types[receiver],
                    call["name"],
                    scope,
                    repository,
                    username,
                )
                if resolved_id:
                    call["resolved_id"] = resolved_id
                    call["category"] = "constructor_resolved"
                    constructor_resolved += 1
                    updated = True

        if updated:
            repo.store_calls(elem_id, calls)

    if constructor_resolved:
        logger.info(
            f"Constructor resolution: resolved {constructor_resolved} calls"
        )

    return constructor_resolved


def _resolve_via_scope_bindings(
    repo: Repository,
    elements: list[dict],
    scope: str,
    repository: str,
    username: str,
    max_passes: int = 3,
) -> int:
    """Strategy 5.7: Resolve calls using AST-based scope analysis.

    Re-parses each element's raw_code with tree-sitter to extract ALL
    variable binding patterns (assignments, with-as, for-in, except-as),
    then resolves unresolved receiver calls where the receiver matches
    a bound variable with a known type.

    Handles patterns that regex-based strategies 5.5/5.6 miss:
    - var = receiver.method()  (method return type propagation)
    - with expr() as var:      (context manager bindings)
    - for var in expr:         (loop variable bindings)
    - except ExcType as var:   (exception handler bindings)
    - Chained resolution via multi-pass (up to max_passes)

    Args:
        repo: Repository instance.
        elements: Elements with calls (from find_all_elements_with_calls).
        scope: Repository scope.
        repository: Repository name.
        username: Username branch.
        max_passes: Maximum resolution passes per element for chaining.

    Returns:
        Number of calls resolved via scope analysis.
    """
    from magaldi_core.scope_bindings import extract_variable_bindings

    scope_resolved = 0

    # Find elements with unresolved receiver calls
    candidates: list[str] = []
    for elem in elements:
        calls = elem.get("calls", [])
        has_unresolved_receiver = any(
            c.get("receiver")
            and not c.get("resolved_id")
            and c.get("category") in ("untyped", "unknown", "type_resolvable")
            for c in calls
        )
        if has_unresolved_receiver:
            candidates.append(elem.get("element_id", ""))

    if not candidates:
        return 0

    logger.info(f"Scope binding resolution: {len(candidates)} candidate elements")

    # Batch fetch full documents for all candidates
    docs = repo.get_documents_batch(candidates)

    for elem_id in candidates:
        if not elem_id:
            continue

        doc = docs.get(elem_id)
        if not doc:
            continue

        raw_code = doc.get("raw_code", "")
        language = doc.get("language", "")
        calls = doc.get("calls", [])
        parameters = doc.get("parameters", [])

        if not raw_code or language != "python":
            continue

        # Extract variable bindings from AST
        bindings = extract_variable_bindings(raw_code, language)
        if not bindings:
            continue

        # Build param type map from function parameters
        param_types: dict[str, str] = {}
        if parameters:
            for p in parameters:
                if p.get("type"):
                    param_types[p["name"]] = p["type"]

        # Multi-pass resolution: each pass may discover new types
        # that enable further resolution in the next pass
        updated = False
        for _pass_num in range(max_passes):
            pass_resolved = 0

            # Build resolved call maps from current state
            resolved_calls = _build_resolved_maps(calls)

            # Resolve binding types
            binding_types: dict[str, str] = {}
            for binding in bindings:
                var = binding.variable
                inferred_type = _resolve_binding_type(
                    binding, resolved_calls, param_types, binding_types, repo,
                    scope, repository, username,
                )
                if inferred_type:
                    binding_types[var] = inferred_type

            # Try to resolve unresolved receiver calls using binding types
            for call in calls:
                if call.get("resolved_id") or not call.get("receiver"):
                    continue
                receiver = call["receiver"]
                if receiver in binding_types:
                    resolved_id = _lookup_method_by_type(
                        repo,
                        binding_types[receiver],
                        call["name"],
                        scope,
                        repository,
                        username,
                    )
                    if resolved_id:
                        call["resolved_id"] = resolved_id
                        call["category"] = "scope_resolved"
                        scope_resolved += 1
                        pass_resolved += 1
                        updated = True

            # Stop if no progress in this pass
            if pass_resolved == 0:
                break

        if updated:
            repo.store_calls(elem_id, calls)

    if scope_resolved:
        logger.info(f"Scope binding resolution: resolved {scope_resolved} calls")

    return scope_resolved


def _build_resolved_maps(calls: list[dict]) -> dict[str, str]:
    """Build a map from call key to resolved_id for resolved calls.

    Maps both bare calls (name -> resolved_id) and receiver calls
    ((receiver, name) -> resolved_id).

    Returns:
        Dict with string keys: "name" for bare calls,
        "receiver.name" for receiver calls.
    """
    resolved: dict[str, str] = {}
    for c in calls:
        if c.get("resolved_id"):
            if c.get("receiver"):
                key = f"{c['receiver']}.{c['name']}"
                resolved[key] = c["resolved_id"]
            else:
                resolved[c["name"]] = c["resolved_id"]
    return resolved


def _resolve_binding_type(
    binding: object,  # BindingInfo
    resolved_calls: dict[str, str],
    param_types: dict[str, str],
    binding_types: dict[str, str],
    repo: Repository,
    scope: str,
    repository: str,
    username: str,
) -> str | None:
    """Determine the type of a variable from its binding pattern.

    Uses resolved call targets, parameter types, and previously-resolved
    binding types to infer the type of a variable.

    Args:
        binding: BindingInfo object.
        resolved_calls: Map of call keys to resolved element IDs.
        param_types: Map of parameter names to type annotations.
        binding_types: Map of already-resolved variable names to types.
        repo: Repository instance.
        scope: Repository scope.
        repository: Repository name.
        username: Username branch.

    Returns:
        Type name string if determined, None otherwise.
    """
    from magaldi_core.scope_bindings import (
        SOURCE_ASSIGNMENT_CALL,
        SOURCE_ASSIGNMENT_METHOD_CALL,
        SOURCE_CONSTRUCTOR,
        SOURCE_EXCEPT_AS,
        SOURCE_FOR_IN,
        SOURCE_WITH_AS,
        BindingInfo,
    )

    if not isinstance(binding, BindingInfo):
        return None

    source = binding.source

    # Direct type bindings — type is known immediately
    if source == SOURCE_EXCEPT_AS and binding.type_name:
        return binding.type_name

    if source == SOURCE_CONSTRUCTOR and binding.type_name:
        return binding.type_name

    # Call-based bindings — need to look up return_type
    if source == SOURCE_ASSIGNMENT_CALL and binding.call_name:
        # var = func() — look up func's return_type
        resolved_id = resolved_calls.get(binding.call_name)
        if resolved_id:
            return _get_return_type(repo, resolved_id)

    if source == SOURCE_ASSIGNMENT_METHOD_CALL and binding.call_name and binding.call_receiver:
        # var = receiver.method() — need receiver's type, then look up method's return_type
        return _resolve_method_call_type(
            binding.call_receiver,
            binding.call_name,
            resolved_calls,
            param_types,
            binding_types,
            repo,
            scope,
            repository,
            username,
        )

    if source == SOURCE_WITH_AS:
        # with expr() as var: — type comes from expression's return_type
        if binding.call_name and binding.call_receiver:
            return _resolve_method_call_type(
                binding.call_receiver,
                binding.call_name,
                resolved_calls,
                param_types,
                binding_types,
                repo,
                scope,
                repository,
                username,
            )
        if binding.call_name:
            resolved_id = resolved_calls.get(binding.call_name)
            if resolved_id:
                return _get_return_type(repo, resolved_id)

    if source == SOURCE_FOR_IN:
        # for var in expr(): — need to unwrap generic return type
        return_type = None
        if binding.call_name and binding.call_receiver:
            return_type = _resolve_method_call_type(
                binding.call_receiver,
                binding.call_name,
                resolved_calls,
                param_types,
                binding_types,
                repo,
                scope,
                repository,
                username,
            )
        elif binding.call_name:
            resolved_id = resolved_calls.get(binding.call_name)
            if resolved_id:
                return_type = _get_return_type(repo, resolved_id)
        if return_type:
            return _unwrap_iterable_type(return_type)

    return None


def _resolve_method_call_type(
    receiver: str,
    method_name: str,
    resolved_calls: dict[str, str],
    param_types: dict[str, str],
    binding_types: dict[str, str],
    repo: Repository,
    scope: str,
    repository: str,
    username: str,
) -> str | None:
    """Resolve the return type of a receiver.method() call.

    Checks three sources for the receiver's type:
    1. Already-resolved binding types (from earlier bindings or passes)
    2. Parameter type annotations
    3. Already-resolved receiver.method() calls

    Args:
        receiver: The receiver variable name.
        method_name: The method name being called.
        resolved_calls: Map of call keys to resolved IDs.
        param_types: Map of parameter names to types.
        binding_types: Map of already-resolved variable names to types.
        repo: Repository instance.
        scope: Repository scope.
        repository: Repository name.
        username: Username branch.

    Returns:
        Return type of the method if determinable, None otherwise.
    """
    # Check if receiver.method() is already resolved directly
    call_key = f"{receiver}.{method_name}"
    resolved_id = resolved_calls.get(call_key)
    if resolved_id:
        return _get_return_type(repo, resolved_id)

    # Check if receiver's type is known (from bindings, params, or prior resolution)
    receiver_type = binding_types.get(receiver) or param_types.get(receiver)
    if not receiver_type:
        return None

    # Look up method on the receiver's type to get its return_type
    base_type = _unwrap_type(receiver_type)
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

    method_doc = repo.get_method_by_class(
        class_id=class_id,
        method_name=method_name,
        scope=scope,
        repository=repository,
        username=username,
    )

    if method_doc:
        return method_doc.get("return_type")

    return None


def _get_return_type(repo: Repository, element_id: str) -> str | None:
    """Look up an element's return_type from the index.

    Args:
        repo: Repository instance.
        element_id: Element ID to look up.

    Returns:
        The return_type string if available, None otherwise.
    """
    doc = repo.get_document(element_id)
    if doc:
        return doc.get("return_type")
    return None


def _unwrap_iterable_type(type_name: str) -> str | None:
    """Extract element type from an iterable type annotation.

    Handles: list[Item] -> Item, List[Item] -> Item,
    Iterable[Item] -> Item, Iterator[Item] -> Item,
    Sequence[Item] -> Item, set[Item] -> Item.

    Args:
        type_name: Type annotation string.

    Returns:
        Element type if extractable, None otherwise.
    """
    if not type_name:
        return None

    # Match generic collection types: list[X], List[X], etc.
    import re

    match = re.match(
        r"(?:list|List|Iterable|Iterator|Sequence|set|Set|frozenset|"
        r"FrozenSet|tuple|Tuple|Generator|AsyncGenerator|AsyncIterator|"
        r"AsyncIterable|Collection|Deque|deque)\[(.+)\]$",
        type_name,
    )
    if match:
        inner = match.group(1)
        # Handle Union types in generics: list[Item | None] -> Item
        if "|" in inner:
            parts = [p.strip() for p in inner.split("|") if p.strip() != "None"]
            return parts[0] if parts else None
        # Handle nested generics: skip them
        if "[" in inner:
            return None
        return inner

    return None


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
    # Search for function, class, method, interface, trait, enum
    for elem_type in ["function", "class", "method", "interface", "trait", "enum"]:
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


def _receiver_class_affinity(receiver: str, candidate: dict) -> float:
    """Score how well a receiver variable name relates to a candidate's class context.

    Heuristic scoring:
    - "repo" → parent class "Repository" (prefix match) → 1.0
    - "es_repo" → parent class "ElasticsearchRepository" (substring) → 0.8
    - "client" → file "http_client.py" (substring in path) → 0.6
    - No match → 0.0

    Args:
        receiver: The receiver variable name (e.g., "repo", "client").
        candidate: Candidate element dict with parent_id and relative_path.

    Returns:
        Affinity score between 0.0 and 1.0.
    """
    receiver_lower = receiver.lower()

    # Extract class name from parent_id if available
    # parent_id format: {scope}:{repo}:{user}:{path}:class:{ClassName}:{line}
    parent_id = candidate.get("parent_id", "") or ""
    parent_class = ""
    if ":class:" in parent_id:
        parts = parent_id.split(":")
        try:
            class_idx = parts.index("class")
            if class_idx + 1 < len(parts):
                parent_class = parts[class_idx + 1]
        except ValueError:
            pass

    parent_lower = parent_class.lower()
    path_lower = (candidate.get("relative_path", "") or "").lower()

    # Extract filename stem from path (e.g., "repositories/search.py" → "search")
    path_stem = path_lower.rsplit("/", 1)[-1].removesuffix(".py").removesuffix(".js").removesuffix(".ts")

    best_score = 0.0

    if parent_lower:
        # Exact prefix: "repo" is prefix of "repository"
        if parent_lower.startswith(receiver_lower) and len(receiver_lower) >= 3:
            best_score = max(best_score, 1.0)
        # Substring: "search" in "searchrepository"
        elif receiver_lower in parent_lower and len(receiver_lower) >= 3:
            best_score = max(best_score, 0.8)
        # Abbreviation: "sr" doesn't match, but "search_repo" contains "repo"
        # Split by underscore and check parts
        elif any(
            parent_lower.startswith(part) or part in parent_lower
            for part in receiver_lower.split("_")
            if len(part) >= 3
        ):
            best_score = max(best_score, 0.7)

    # File path matching
    if path_stem:
        if path_stem.startswith(receiver_lower) and len(receiver_lower) >= 3:
            best_score = max(best_score, 0.6)
        elif receiver_lower in path_stem and len(receiver_lower) >= 3:
            best_score = max(best_score, 0.5)
        elif any(
            part in path_stem
            for part in receiver_lower.split("_")
            if len(part) >= 3
        ):
            best_score = max(best_score, 0.4)

    return best_score


def _score_candidates_rrf(
    call: dict,
    candidates: list[dict],
    caller_embedding: list[float] | None,
    k: int = 60,
) -> list[tuple[dict, float]]:
    """Score candidates using Reciprocal Rank Fusion across multiple signals.

    Fuses three ranking signals:
    1. Receiver-to-class-name affinity (name heuristics)
    2. Embedding cosine similarity (semantic match)
    3. Path context match (file proximity hints)

    RRF formula: score = sum(1 / (k + rank_i)) for each signal.

    Args:
        call: The unresolved call dict (with receiver, name).
        candidates: List of candidate element dicts.
        caller_embedding: The calling element's caller_embedding, or None.
        k: RRF smoothing constant (default 60, standard value).

    Returns:
        List of (candidate, rrf_score) tuples sorted descending by score.
    """
    receiver = call.get("receiver", "") or ""
    n = len(candidates)

    if n == 0:
        return []

    # Signal 1: Receiver-class affinity
    affinity_scores = [
        (i, _receiver_class_affinity(receiver, c)) for i, c in enumerate(candidates)
    ]
    affinity_ranked = sorted(affinity_scores, key=lambda x: -x[1])

    # Signal 2: Embedding similarity
    if caller_embedding:
        embed_scores = []
        for i, c in enumerate(candidates):
            c_emb = c.get("summary_embedding")
            score = _cosine_similarity(caller_embedding, c_emb) if c_emb else -1.0
            embed_scores.append((i, score))
        embed_ranked = sorted(embed_scores, key=lambda x: -x[1])
    else:
        # No embedding available — all get same rank
        embed_ranked = [(i, 0.0) for i in range(n)]

    # Signal 3: Path context — does receiver appear in candidate's path?
    # Lighter signal: just check if any receiver part appears in the path
    path_scores = []
    receiver_lower = receiver.lower()
    receiver_parts = [p for p in receiver_lower.split("_") if len(p) >= 3]
    for i, c in enumerate(candidates):
        path = (c.get("relative_path", "") or "").lower()
        score = 0.0
        if receiver_lower in path and len(receiver_lower) >= 3:
            score = 1.0
        elif any(part in path for part in receiver_parts):
            score = 0.5
        path_scores.append((i, score))
    path_ranked = sorted(path_scores, key=lambda x: -x[1])

    # Build rank maps (1-indexed)
    def _build_rank_map(ranked: list[tuple[int, float]]) -> dict[int, int]:
        rank_map: dict[int, int] = {}
        for rank, (idx, _score) in enumerate(ranked, 1):
            rank_map[idx] = rank
        return rank_map

    affinity_ranks = _build_rank_map(affinity_ranked)
    embed_ranks = _build_rank_map(embed_ranked)
    path_ranks = _build_rank_map(path_ranked)

    # Compute RRF scores
    results: list[tuple[dict, float]] = []
    for i, c in enumerate(candidates):
        rrf = (
            1.0 / (k + affinity_ranks[i])
            + 1.0 / (k + embed_ranks[i])
            + 1.0 / (k + path_ranks[i])
        )
        results.append((c, rrf))

    results.sort(key=lambda x: -x[1])
    return results


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


# Path segments that indicate test fixtures — candidates from these paths
# are excluded when the caller is production code, to avoid mis-resolution
# (e.g. dict.get() → tests/fixtures/teatro_production.ts:method:get).
_TEST_FIXTURE_SEGMENTS = frozenset({
    "tests/fixtures",
    "test/fixtures",
    "tests/fixture",
    "test/fixture",
    "__fixtures__",
    "__mocks__",
})


def _is_test_fixture_path(path: str) -> bool:
    """Check if a path is under a test fixture directory."""
    path_lower = path.lower()
    return any(seg in path_lower for seg in _TEST_FIXTURE_SEGMENTS)


def _is_test_path(path: str) -> bool:
    """Check if a path is a test file or under a test directory."""
    path_lower = path.lower()
    # Check directory-level test paths
    if "/tests/" in path_lower or "/test/" in path_lower:
        return True
    # Check file-level test patterns
    parts = path_lower.rsplit("/", 1)
    filename = parts[-1] if parts else path_lower
    return filename.startswith("test_") or filename.endswith("_test.py")


def _filter_candidates_for_caller(
    candidates: list[dict],
    caller_path: str,
) -> list[dict]:
    """Filter candidates based on caller context.

    When the caller is production code, exclude test fixture candidates
    to prevent mis-resolution (e.g., dict.get() resolving to a test fixture
    method named "get").

    Test code can resolve to anything (fixtures, production, etc.).
    """
    if _is_test_path(caller_path):
        # Test code can reference anything
        return candidates

    # Production code: exclude test fixture candidates
    return [
        c for c in candidates
        if not _is_test_fixture_path(c.get("relative_path", ""))
    ]


def resolve_calls_by_embedding(
    repo: Repository,
    scope: str,
    repository: str,
    username: str = "main",
    min_rrf_score: float = 0.048,
) -> tuple[int, int, int]:
    """Strategy 6: Resolve untyped calls using RRF-scored multi-signal matching.

    For calls with a receiver but no type annotation, finds candidate
    functions/methods by name, then uses Reciprocal Rank Fusion across
    receiver-class affinity, embedding similarity, and path context
    to pick the best match when multiple candidates exist.

    Queries both the user's index and "main" to get a complete view,
    with user's elements taking priority.

    Filters out test fixture candidates when the caller is production code
    to prevent mis-resolution (e.g., add() → test fixture function).

    Args:
        repo: Repository instance.
        scope: Repository scope.
        repository: Repository name.
        username: Username branch.
        min_rrf_score: Minimum RRF score to accept a match (default 0.048).

    Returns:
        Tuple of (total_processed, single_match_resolved, rrf_resolved).
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
        caller_path = elem.get("relative_path", "")
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

            # Filter out test fixture candidates when caller is production code
            candidates = _filter_candidates_for_caller(
                name_cache[name], caller_path
            )

            if not candidates:
                continue

            if len(candidates) == 1:
                # Single candidate — resolve directly
                call["resolved_id"] = candidates[0].get("element_id")
                call["category"] = "embedding_resolved"
                single_resolved += 1
                updated = True
                continue

            # Multiple candidates — use RRF-scored multi-signal matching
            # Get caller's embedding (cached) — used as one signal in RRF
            if element_id not in embedding_cache:
                caller_embedding = repo.get_embedding(element_id, "caller")
                embedding_cache[element_id] = caller_embedding
            caller_embedding = embedding_cache[element_id]

            # RRF works even without embeddings (falls back to name/path signals)
            scored = _score_candidates_rrf(call, candidates, caller_embedding)

            if scored and scored[0][1] >= min_rrf_score:
                best_candidate, best_score = scored[0]
                call["resolved_id"] = best_candidate.get("element_id")
                call["category"] = "embedding_resolved"
                embedding_resolved += 1
                updated = True

        if updated:
            repo.store_calls(element_id, calls)

    total_resolved = single_resolved + embedding_resolved
    logger.info(
        f"RRF resolution: resolved {total_resolved}/{total_processed} calls "
        f"({single_resolved} single match, {embedding_resolved} via RRF scoring)"
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
