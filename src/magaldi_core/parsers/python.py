"""Python parser using tree-sitter.

Extracts code elements from Python source files including:
- Classes with attributes and base classes
- Functions and methods with calls and exceptions
- Variables and constants with usage context
- Extended code intelligence (patterns, purity, metrics, etc.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from magaldi_core.analysis.concurrency import detect_concurrency, detect_env_vars
from magaldi_core.analysis.metrics import (
    analyze_docstring,
    compute_code_metrics,
    compute_complexity,
)
from magaldi_core.analysis.security import detect_security_issues
from magaldi_core.parsers.base import (
    Call,
    CodeElement,
    Import,
    TreeSitterParser,
    build_extracted_calls,
    determine_visibility,
    extract_docstring,
    extract_preceding_doc_comment,
    find_variable_usages,
    generate_element_id,
)
from magaldi_core.tree_sitter_manager import (
    DecoratorInfo,
    ExtractedElement,
    associate_comments,
    detect_cli_commands,
    detect_http_routes,
    detect_patterns,
    detect_public_api,
    extract_comments,
    extract_python_base_classes,
    extract_python_calls,
    extract_python_class_attributes,
    extract_python_class_members,
    extract_python_elements,
    extract_python_imports,
    extract_python_modified_attributes,
    extract_python_raised_exceptions,
    extract_section_markers,
    extract_side_effects,
    extract_todos,
    extract_top_level_python_calls,
    extract_type_annotations,
)

if TYPE_CHECKING:
    from magaldi_core.change_detection import FileInfo


class PythonParser(TreeSitterParser):
    """Parse Python files using tree-sitter."""

    def __init__(self) -> None:
        super().__init__("python")

    def parse(
        self,
        content: str,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
    ) -> list[CodeElement]:
        """Parse Python content and extract elements."""
        elements: list[CodeElement] = []
        lines = content.split("\n")

        # Create file-level element
        file_element = self._create_file_element(
            content, file_info, scope, repository, username, "python"
        )
        elements.append(file_element)

        # Parse with tree-sitter
        tree = self.manager.parse(content.encode("utf-8"), "python")
        extracted = extract_python_elements(tree, lines)

        # Extract imports and populate on file element
        extracted_imports = extract_python_imports(tree, lines)
        file_element.imports = [
            Import(
                name=imp.name,
                module=imp.module,
                alias=imp.alias,
                line=imp.line,
            )
            for imp in extracted_imports
        ]

        # Extract top-level calls and populate on file element
        top_level_calls = extract_top_level_python_calls(tree)
        file_element.calls = [
            Call(name=c.name, receiver=c.receiver, line=c.line)
            for c in top_level_calls
        ]

        # Convert ExtractedElements to CodeElements
        for ext in extracted:
            if ext.element_type == "class":
                class_elem = self._convert_class(ext, file_info, scope, repository, username, lines)
                elements.append(class_elem)

                # Extract class members
                if ext.node:
                    methods, class_vars = extract_python_class_members(ext.node, lines)

                    for method_ext in methods:
                        method_elem = self._convert_method(
                            method_ext, class_elem, file_info, scope, repository, username, lines
                        )
                        elements.append(method_elem)

                    for var_ext in class_vars:
                        var_elem = self._convert_variable(
                            var_ext, file_info, scope, repository, username, lines, parent=class_elem
                        )
                        elements.append(var_elem)

            elif ext.element_type == "function":
                func_elem = self._convert_function(ext, file_info, scope, repository, username, lines)
                elements.append(func_elem)

            elif ext.element_type in ("constant", "variable"):
                var_elem = self._convert_variable(ext, file_info, scope, repository, username, lines)
                elements.append(var_elem)

            elif ext.element_type == "import":
                import_elem = self._convert_import(ext, file_info, scope, repository, username)
                elements.append(import_elem)

        # Set parent IDs for elements without explicit parents
        self._set_hierarchy(elements, file_element)

        # Resolve same-file and self-method calls (Phase 1)
        self._resolve_calls_in_file(elements, self_keyword="self")

        # === EXTENDED CODE INTELLIGENCE EXTRACTION ===
        self._extract_extended_intelligence(elements, file_element, content, lines)

        # Compute roll-up statistics for file and class elements
        self._compute_rollup_stats(elements)

        return elements

    def _extract_extended_intelligence(
        self,
        elements: list[CodeElement],
        file_element: CodeElement,
        content: str,
        _lines: list[str],
    ) -> None:
        """Extract extended code intelligence for all elements."""
        # Extract file-level documentation
        todos = extract_todos(content)
        section_markers = extract_section_markers(content)
        all_comments = extract_comments(content)

        # Populate file element with documentation
        file_element.todos = [
            {"kind": t.kind, "text": t.text, "line": t.line,
             "assignee": t.assignee, "priority": t.priority, "issue_ref": t.issue_ref}
            for t in todos
        ]
        file_element.section_markers = [
            {"label": m.label, "line": m.line, "style": m.style}
            for m in section_markers
        ]

        # Process each element for extended intelligence
        for elem in elements:
            if elem.element_type == "file":
                continue

            # Associate comments
            assoc_comments = associate_comments(elem.line_start, all_comments)
            elem.associated_comments = [
                {"text": c.text, "line": c.line, "kind": c.kind, "position": c.position}
                for c in assoc_comments
            ]

            # Type annotations (for functions/methods)
            if elem.element_type in ("function", "method"):
                # Parse for type annotations if we have raw_code
                if elem.raw_code:
                    try:
                        elem_tree = self.manager.parse(elem.raw_code.encode("utf-8"), "python")
                        type_annots = extract_type_annotations(elem_tree.root_node, "python")
                        elem.type_annotations = [
                            {"name": a.name, "kind": a.kind, "location": a.location,
                             "line": a.line, "generic_args": a.generic_args}
                            for a in type_annots
                        ]
                    except Exception:
                        pass  # Skip type extraction if parsing fails

                # Side effects and mutated state (purity analysis disabled —
                # 97% false-positive rate, see quality reports)
                mutations = elem.attributes_modified or []
                effects = extract_side_effects(
                    build_extracted_calls(elem.calls), mutations, "python"
                )
                elem.side_effects = [
                    {"kind": e.kind, "target": e.target, "line": e.line}
                    for e in effects
                ]
                elem.mutated_state = mutations

                # Code metrics (Tier 1)
                if elem.raw_code:
                    try:
                        elem_tree = self.manager.parse(elem.raw_code.encode("utf-8"), "python")
                        elem.complexity = compute_complexity(elem_tree.root_node, "python")
                    except Exception:
                        pass

                elem.code_metrics = compute_code_metrics(elem.raw_code or "", elem.parameters)
                elem.docstring_quality = analyze_docstring(
                    elem.docstring, elem.parameters, elem.return_type
                )

                # Security analysis
                elem.security_issues = detect_security_issues(elem.raw_code or "", "python")

                # Environment and concurrency detection
                elem.env_vars = detect_env_vars(elem.calls, elem.raw_code or "", "python")
                elem.concurrency = detect_concurrency(
                    elem.calls, elem.raw_code or "", elem.decorators, elem.is_async, "python"
                )

            # Security analysis for variables/constants (check for hardcoded secrets)
            if elem.element_type in ("variable", "constant"):
                elem.security_issues = detect_security_issues(elem.raw_code or "", "python")

            # API surface detection
            if elem.decorator_details:
                dec_infos = [
                    DecoratorInfo(name=d.get("name", ""), args=d.get("args"), full=d.get("full"))
                    for d in elem.decorator_details
                ]
                routes = detect_http_routes(dec_infos, "python")
                elem.http_routes = [
                    {"method": r.method, "path": r.path,
                     "path_params": r.path_params, "framework": r.framework}
                    for r in routes
                ]
                commands = detect_cli_commands(dec_infos, elem.name, "python")
                elem.cli_commands = [
                    {"name": c.name, "options": c.options, "framework": c.framework}
                    for c in commands
                ]
                elem.is_public_api = detect_public_api(
                    elem.name, dec_infos, elem.visibility, "python"
                )
            else:
                elem.is_public_api = detect_public_api(
                    elem.name, [], elem.visibility, "python"
                )

            # Pattern detection (for classes)
            if elem.element_type == "class":
                # Collect method names from child elements
                class_methods = [
                    e.name for e in elements
                    if e.parent_id == elem.element_id and e.element_type == "method"
                ]

                # Collect class-level variables (defined at class scope, not in methods)
                class_variables = [
                    e.name for e in elements
                    if e.parent_id == elem.element_id and e.element_type == "variable"
                ]

                class_info = {
                    "name": elem.name,
                    "attributes": [a.get("name", "") for a in (elem.class_attributes or [])],
                    "methods": class_methods,
                    "class_variables": class_variables,
                    "decorators": elem.decorators or [],
                }
                patterns, confidence = detect_patterns(class_info, [], "python")
                elem.detected_patterns = patterns
                elem.pattern_confidence = confidence

    def _compute_rollup_stats(self, elements: list[CodeElement]) -> None:
        """Compute aggregated metrics for file and class elements from their children."""
        # Build parent-child map
        children_by_parent: dict[str, list[CodeElement]] = {}
        for elem in elements:
            if elem.parent_id:
                if elem.parent_id not in children_by_parent:
                    children_by_parent[elem.parent_id] = []
                children_by_parent[elem.parent_id].append(elem)

        # Compute stats for file and class elements
        for elem in elements:
            if elem.element_type not in ("file", "class"):
                continue

            children = children_by_parent.get(elem.element_id, [])
            if not children:
                continue

            # Collect metrics from descendant functions/methods
            all_descendants = self._get_all_descendants(elem.element_id, children_by_parent)

            security_issue_count = 0
            security_by_severity: dict[str, int] = {}
            complexities: list[int] = []
            undocumented_count = 0
            function_count = 0
            async_count = 0
            env_var_count = 0

            for child in all_descendants:
                if child.element_type in ("function", "method"):
                    function_count += 1

                    # Security issues
                    for issue in child.security_issues:
                        security_issue_count += 1
                        sev = issue.get("severity", "info")
                        security_by_severity[sev] = security_by_severity.get(sev, 0) + 1

                    # Complexity
                    if child.complexity:
                        complexities.append(child.complexity.get("cyclomatic", 1))

                    # Documentation
                    if child.docstring_quality and not child.docstring_quality.get("has_docstring", True):
                        undocumented_count += 1

                    # Async
                    if child.concurrency and child.concurrency.get("is_async"):
                        async_count += 1

                    # Env vars
                    env_var_count += len(child.env_vars)

                elif child.element_type in ("variable", "constant"):
                    # Security issues on variables
                    for issue in child.security_issues:
                        security_issue_count += 1
                        sev = issue.get("severity", "info")
                        security_by_severity[sev] = security_by_severity.get(sev, 0) + 1

            # Only set if there are meaningful stats
            if function_count > 0 or security_issue_count > 0:
                elem.metrics_summary = {
                    "security_issue_count": security_issue_count,
                    "security_by_severity": security_by_severity if security_by_severity else None,
                    "max_complexity": max(complexities) if complexities else None,
                    "avg_complexity": round(sum(complexities) / len(complexities), 1) if complexities else None,
                    "undocumented_count": undocumented_count,
                    "function_count": function_count,
                    "async_count": async_count,
                    "env_var_count": env_var_count,
                }

    def _get_all_descendants(
        self,
        parent_id: str,
        children_by_parent: dict[str, list[CodeElement]],
    ) -> list[CodeElement]:
        """Get all descendants of an element (recursive)."""
        result: list[CodeElement] = []
        direct_children = children_by_parent.get(parent_id, [])
        for child in direct_children:
            result.append(child)
            result.extend(self._get_all_descendants(child.element_id, children_by_parent))
        return result

    def _convert_class(
        self,
        ext: ExtractedElement,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        lines: list[str],
    ) -> CodeElement:
        """Convert extracted class to CodeElement."""
        docstring = extract_docstring(lines, ext.line_start - 1)

        # Extract class attributes and base classes from AST
        class_attributes = None
        base_classes = None
        if ext.node:
            class_attributes = extract_python_class_attributes(ext.node) or None
            base_classes = extract_python_base_classes(ext.node) or None

        # Convert decorator_details to dicts for storage
        decorator_details = None
        if ext.decorator_details:
            decorator_details = [
                {"name": d.name, "args": d.args, "full": d.full}
                for d in ext.decorator_details
            ]

        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="class",
            name=ext.name,
            language="python",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            docstring=docstring,
            decorators=ext.decorators or [],
            decorator_details=decorator_details,
            level=1,
            visibility=determine_visibility(ext.name),
            class_attributes=class_attributes,
            base_classes=base_classes,
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, "class", ext.name, ext.get_byte_offset()
        )
        return elem

    def _convert_function(
        self,
        ext: ExtractedElement,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        lines: list[str],
    ) -> CodeElement:
        """Convert extracted function to CodeElement."""
        docstring = extract_docstring(lines, ext.line_start - 1)

        # Extract calls and exceptions from function body
        calls: list[Call] = []
        exceptions_raised = None
        if ext.node:
            extracted_calls = extract_python_calls(ext.node)
            calls = [
                Call(name=c.name, receiver=c.receiver, line=c.line)
                for c in extracted_calls
            ]
            exceptions_raised = extract_python_raised_exceptions(ext.node) or None

        # Convert decorator_details to dicts for storage
        decorator_details = None
        if ext.decorator_details:
            decorator_details = [
                {"name": d.name, "args": d.args, "full": d.full}
                for d in ext.decorator_details
            ]

        # Convert parameters to dicts for storage
        parameters = None
        if ext.parameters:
            parameters = [
                {"name": p.name, "type": p.type, "default": p.default}
                for p in ext.parameters
            ]

        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="function",
            name=ext.name,
            language="python",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.signature,
            docstring=docstring,
            decorators=ext.decorators or [],
            decorator_details=decorator_details,
            is_async=ext.is_async,
            visibility=determine_visibility(ext.name),
            level=2,
            calls=calls,
            exceptions_raised=exceptions_raised,
            return_type=ext.return_type,
            parameters=parameters or [],
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, "function", ext.name, ext.get_byte_offset()
        )
        return elem

    def _convert_method(
        self,
        ext: ExtractedElement,
        parent_class: CodeElement,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        lines: list[str],
    ) -> CodeElement:
        """Convert extracted method to CodeElement."""
        docstring = extract_docstring(lines, ext.line_start - 1)

        # Extract calls, exceptions, and modified attributes from method body
        calls: list[Call] = []
        exceptions_raised = None
        attributes_modified = None
        if ext.node:
            extracted_calls = extract_python_calls(ext.node)
            calls = [
                Call(name=c.name, receiver=c.receiver, line=c.line)
                for c in extracted_calls
            ]
            exceptions_raised = extract_python_raised_exceptions(ext.node) or None
            attributes_modified = extract_python_modified_attributes(ext.node) or None

        # Convert decorator_details to dicts for storage
        decorator_details = None
        if ext.decorator_details:
            decorator_details = [
                {"name": d.name, "args": d.args, "full": d.full}
                for d in ext.decorator_details
            ]

        # Convert parameters to dicts for storage
        parameters = None
        if ext.parameters:
            parameters = [
                {"name": p.name, "type": p.type, "default": p.default}
                for p in ext.parameters
            ]

        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="method",
            name=ext.name,
            language="python",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.signature,
            docstring=docstring,
            decorators=ext.decorators or [],
            decorator_details=decorator_details,
            is_async=ext.is_async,
            visibility=determine_visibility(ext.name),
            level=2,
            parent_id=parent_class.element_id,
            calls=calls,
            exceptions_raised=exceptions_raised,
            attributes_modified=attributes_modified,
            return_type=ext.return_type,
            parameters=parameters or [],
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, "method", ext.name, ext.get_byte_offset()
        )
        return elem

    def _convert_variable(
        self,
        ext: ExtractedElement,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        lines: list[str],
        parent: CodeElement | None = None,
    ) -> CodeElement:
        """Convert extracted variable/constant to CodeElement."""
        # Find usages
        usages = find_variable_usages(ext.name, lines, ext.line_start)

        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type=ext.element_type,  # 'constant' or 'variable'
            name=ext.name,
            language="python",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            docstring=extract_preceding_doc_comment(lines, ext.line_start, "python"),
            level=3,
            visibility=determine_visibility(ext.name),
            context_usages=usages,
            parent_id=parent.element_id if parent else None,
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, ext.element_type, ext.name, ext.get_byte_offset()
        )
        return elem

    def _convert_import(
        self,
        ext: ExtractedElement,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
    ) -> CodeElement:
        """Convert import statement to CodeElement."""
        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="import",
            name=ext.name,  # Module name
            language="python",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.signature,
            level=0,  # Top level, same as file
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path,
            "import", ext.name, ext.get_byte_offset()
        )
        return elem
