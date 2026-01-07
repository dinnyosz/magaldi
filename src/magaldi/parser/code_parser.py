"""Phase 3: Parsing - Extract code elements using Tree-sitter.

This module parses source files and extracts structured code elements:
- Classes
- Functions/Methods
- Module-level variables

Each element includes position, content, and hierarchy information.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from magaldi.parser.change_detection import ChangeManifest, FileInfo


class ParsingError(Exception):
    """Raised when parsing fails."""

    pass


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class CodeElement:
    """A parsed code element (class, function, method, variable)."""

    # Identity
    element_id: str = ""

    # Location
    scope: str = ""
    repository: str = ""
    username: str = ""
    relative_path: str = ""

    # Element info
    element_type: str = ""  # 'file', 'class', 'function', 'method', 'constant', 'variable'
    name: str = ""
    language: str = ""

    # Position (1-indexed lines)
    line_start: int = 0
    line_end: int = 0

    # Content
    raw_code: str = ""
    signature: str | None = None
    docstring: str | None = None

    # Metadata
    decorators: list[str] = field(default_factory=list)
    is_async: bool = False
    visibility: str = "public"  # 'public', 'private', 'protected'

    # Hierarchy
    level: int = 0  # 0=file, 1=class, 2=function/method, 3=variable
    parent_id: str | None = None

    # Type info
    return_type: str | None = None
    parameters: list[dict[str, Any]] = field(default_factory=list)

    # Context (for variables - how they're used)
    context_usages: list[str] = field(default_factory=list)


@dataclass
class ParsedFile:
    """Result of parsing a single file."""

    file_info: FileInfo
    elements: list[CodeElement] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    line_count: int = 0


@dataclass
class ParsingResult:
    """Result of parsing all files in a manifest."""

    scope: str
    repository: str
    username: str

    parsed_files: list[ParsedFile] = field(default_factory=list)
    failed_files: list[tuple[FileInfo, str]] = field(default_factory=list)

    @property
    def total_elements(self) -> int:
        return sum(len(pf.elements) for pf in self.parsed_files)

    @property
    def elements_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for pf in self.parsed_files:
            for elem in pf.elements:
                counts[elem.element_type] = counts.get(elem.element_type, 0) + 1
        return counts


# =============================================================================
# ELEMENT ID GENERATION
# =============================================================================


def generate_element_id(
    scope: str,
    repository: str,
    username: str,
    relative_path: str,
    element_type: str,
    name: str,
    line_start: int,
) -> str:
    """Generate unique element ID.

    Format: {scope}:{repository}:{username}:{relative_path}:{type}:{name}:{line}
    """
    return ":".join([
        scope,
        repository,
        username,
        relative_path,
        element_type,
        name,
        str(line_start),
    ])


# =============================================================================
# PYTHON PARSER (regex-based fallback, can be replaced with Tree-sitter)
# =============================================================================


class PythonParser:
    """Parse Python files and extract code elements."""

    # Patterns for Python code elements
    CLASS_PATTERN = re.compile(
        r'^(?P<decorators>(?:@[\w.]+(?:\([^)]*\))?\s*\n)*)'
        r'^(?P<indent>[ \t]*)class\s+(?P<name>\w+)(?:\([^)]*\))?:',
        re.MULTILINE
    )

    FUNCTION_PATTERN = re.compile(
        r'^(?P<decorators>(?:@[\w.]+(?:\([^)]*\))?\s*\n)*)'
        r'^(?P<indent>[ \t]*)(?P<async>async\s+)?def\s+(?P<name>\w+)\s*'
        r'\((?P<params>[^)]*)\)(?:\s*->\s*(?P<return>[^:]+))?:',
        re.MULTILINE
    )

    VARIABLE_PATTERN = re.compile(
        r'^(?P<name>[A-Z][A-Z0-9_]*)\s*(?::\s*[^=]+)?\s*=\s*(?P<value>.+)$',
        re.MULTILINE
    )

    DOCSTRING_PATTERN = re.compile(
        r'^[ \t]*(?:"""(.+?)"""|\'\'\'(.+?)\'\'\')',
        re.MULTILINE | re.DOTALL
    )

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
        line_count = len(lines)

        # Create file-level element
        file_element = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="file",
            name=Path(file_info.relative_path).name,
            language="python",
            line_start=1,
            line_end=line_count,
            level=0,
            raw_code="",  # Don't store full file content
        )
        file_element.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path,
            "file", file_element.name, 1
        )
        elements.append(file_element)

        # Parse classes
        class_elements = self._parse_classes(content, lines, file_info, scope, repository, username)
        elements.extend(class_elements)

        # Parse standalone functions (not inside classes)
        func_elements = self._parse_functions(content, lines, file_info, scope, repository, username, class_elements)
        elements.extend(func_elements)

        # Parse module-level constants (UPPER_CASE)
        const_elements = self._parse_constants(content, lines, file_info, scope, repository, username)
        elements.extend(const_elements)

        # Parse class variables
        class_var_elements = self._parse_class_variables(content, lines, file_info, scope, repository, username, class_elements)
        elements.extend(class_var_elements)

        # Set parent IDs
        self._set_hierarchy(elements, file_element)

        return elements

    def _parse_classes(
        self,
        content: str,
        lines: list[str],
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
    ) -> list[CodeElement]:
        """Extract class definitions."""
        elements = []

        for match in self.CLASS_PATTERN.finditer(content):
            name = match.group("name")
            start_pos = match.start()
            line_start = content[:start_pos].count("\n") + 1

            # Find class end by indentation
            indent = len(match.group("indent"))
            line_end = self._find_block_end(lines, line_start - 1, indent)

            # Extract raw code
            raw_code = "\n".join(lines[line_start - 1:line_end])

            # Extract docstring
            docstring = self._extract_docstring(lines, line_start)

            # Extract decorators
            decorators = self._extract_decorators(match.group("decorators") or "")

            element = CodeElement(
                scope=scope,
                repository=repository,
                username=username,
                relative_path=file_info.relative_path,
                element_type="class",
                name=name,
                language="python",
                line_start=line_start,
                line_end=line_end,
                raw_code=raw_code,
                docstring=docstring,
                decorators=decorators,
                level=1,
                visibility="private" if name.startswith("_") else "public",
            )
            element.element_id = generate_element_id(
                scope, repository, username, file_info.relative_path,
                "class", name, line_start
            )

            elements.append(element)

            # Parse methods inside this class
            method_elements = self._parse_methods(
                raw_code, lines, line_start, element,
                file_info, scope, repository, username
            )
            elements.extend(method_elements)

        return elements

    def _parse_methods(
        self,
        class_content: str,
        all_lines: list[str],
        class_start: int,
        parent_class: CodeElement,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
    ) -> list[CodeElement]:
        """Extract methods from a class."""
        elements = []

        for match in self.FUNCTION_PATTERN.finditer(class_content):
            indent = len(match.group("indent"))
            if indent == 0:
                continue  # Skip class-level definition itself

            name = match.group("name")
            is_async = bool(match.group("async"))
            params = match.group("params") or ""
            return_type = match.group("return")

            # Calculate actual line number
            start_in_class = class_content[:match.start()].count("\n")
            line_start = class_start + start_in_class

            # Find method end
            line_end = self._find_block_end(all_lines, line_start - 1, indent)

            # Extract raw code
            raw_code = "\n".join(all_lines[line_start - 1:line_end])

            # Build signature
            signature = f"{'async ' if is_async else ''}def {name}({params})"
            if return_type:
                signature += f" -> {return_type.strip()}"

            # Extract docstring
            docstring = self._extract_docstring(all_lines, line_start)

            # Extract decorators - look at lines before the def
            decorators = self._extract_decorators_from_lines(all_lines, line_start - 1)

            # Determine visibility
            if name.startswith("__") and not name.endswith("__"):
                visibility = "private"
            elif name.startswith("_"):
                visibility = "protected"
            else:
                visibility = "public"

            element = CodeElement(
                scope=scope,
                repository=repository,
                username=username,
                relative_path=file_info.relative_path,
                element_type="method",
                name=name,
                language="python",
                line_start=line_start,
                line_end=line_end,
                raw_code=raw_code,
                signature=signature,
                docstring=docstring,
                decorators=decorators,
                is_async=is_async,
                visibility=visibility,
                level=2,
                parent_id=parent_class.element_id,
                return_type=return_type.strip() if return_type else None,
                parameters=self._parse_parameters(params),
            )
            element.element_id = generate_element_id(
                scope, repository, username, file_info.relative_path,
                "method", name, line_start
            )
            elements.append(element)

        return elements

    def _parse_functions(
        self,
        content: str,
        lines: list[str],
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        class_elements: list[CodeElement],
    ) -> list[CodeElement]:
        """Extract standalone function definitions."""
        elements = []

        # Get class line ranges to exclude
        class_ranges = [
            (c.line_start, c.line_end)
            for c in class_elements
            if c.element_type == "class"
        ]

        for match in self.FUNCTION_PATTERN.finditer(content):
            indent = len(match.group("indent"))
            if indent > 0:
                continue  # Skip indented (method) definitions

            name = match.group("name")
            start_pos = match.start()
            line_start = content[:start_pos].count("\n") + 1

            # Skip if inside a class
            if any(start <= line_start <= end for start, end in class_ranges):
                continue

            is_async = bool(match.group("async"))
            params = match.group("params") or ""
            return_type = match.group("return")

            # Find function end
            line_end = self._find_block_end(lines, line_start - 1, 0)

            # Extract raw code
            raw_code = "\n".join(lines[line_start - 1:line_end])

            # Build signature
            signature = f"{'async ' if is_async else ''}def {name}({params})"
            if return_type:
                signature += f" -> {return_type.strip()}"

            # Extract docstring
            docstring = self._extract_docstring(lines, line_start)

            # Extract decorators
            decorators = self._extract_decorators(match.group("decorators") or "")

            element = CodeElement(
                scope=scope,
                repository=repository,
                username=username,
                relative_path=file_info.relative_path,
                element_type="function",
                name=name,
                language="python",
                line_start=line_start,
                line_end=line_end,
                raw_code=raw_code,
                signature=signature,
                docstring=docstring,
                decorators=decorators,
                is_async=is_async,
                visibility="private" if name.startswith("_") else "public",
                level=2,
                return_type=return_type.strip() if return_type else None,
                parameters=self._parse_parameters(params),
            )
            element.element_id = generate_element_id(
                scope, repository, username, file_info.relative_path,
                "function", name, line_start
            )
            elements.append(element)

        return elements

    def _find_variable_usages(
        self, name: str, lines: list[str], declaration_line: int, max_usages: int = 5
    ) -> list[str]:
        """Find lines where a variable is used (excluding declaration).

        Args:
            name: Variable name to search for.
            lines: All lines in the file.
            declaration_line: Line number of the declaration (1-indexed).
            max_usages: Maximum number of usages to return.

        Returns:
            List of "line N: <code>" strings showing usages.
        """
        usages = []
        # Pattern to match the variable name as a whole word
        pattern = re.compile(rf'\b{re.escape(name)}\b')

        for i, line in enumerate(lines):
            line_num = i + 1
            if line_num == declaration_line:
                continue  # Skip the declaration itself

            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue  # Skip empty lines and comments

            if pattern.search(line):
                # Truncate long lines
                display_line = stripped[:80] + '...' if len(stripped) > 80 else stripped
                usages.append(f"line {line_num}: {display_line}")
                if len(usages) >= max_usages:
                    break

        return usages

    def _parse_constants(
        self,
        content: str,
        lines: list[str],
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
    ) -> list[CodeElement]:
        """Extract module-level constants (UPPER_CASE variables)."""
        elements = []

        for match in self.VARIABLE_PATTERN.finditer(content):
            # Only match at module level (no indentation)
            start_pos = match.start()
            line_start = content[:start_pos].count("\n") + 1

            # Check if at start of line or only whitespace before
            line = lines[line_start - 1]
            if line.lstrip() != line:
                continue  # Skip indented

            name = match.group("name")
            value = match.group("value").strip()

            # Find usages of this constant in the file
            usages = self._find_variable_usages(name, lines, line_start)

            element = CodeElement(
                scope=scope,
                repository=repository,
                username=username,
                relative_path=file_info.relative_path,
                element_type="constant",
                name=name,
                language="python",
                line_start=line_start,
                line_end=line_start,
                raw_code=line.strip(),
                level=3,
                visibility="public",
                context_usages=usages,
            )
            element.element_id = generate_element_id(
                scope, repository, username, file_info.relative_path,
                "constant", name, line_start
            )
            elements.append(element)

        return elements

    # Pattern for class variables: name = value or name: type = value (with indentation)
    CLASS_VAR_PATTERN = re.compile(
        r'^(?P<indent>\s+)(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*(?::\s*[^=]+)?\s*=\s*(?P<value>.+)$',
        re.MULTILINE
    )

    def _parse_class_variables(
        self,
        content: str,
        lines: list[str],
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        class_elements: list[CodeElement],
    ) -> list[CodeElement]:
        """Extract class-level variables (attributes defined in class body, not in methods)."""
        elements = []

        for class_elem in class_elements:
            if class_elem.element_type != "class":
                continue

            class_start = class_elem.line_start
            class_end = class_elem.line_end
            class_indent = len(lines[class_start - 1]) - len(lines[class_start - 1].lstrip())
            body_indent = class_indent + 4  # Expected indentation for class body

            # Track method ranges to exclude variables inside methods
            method_ranges = []
            for match in self.FUNCTION_PATTERN.finditer(class_elem.raw_code):
                method_start_rel = class_elem.raw_code[:match.start()].count("\n")
                method_start_abs = class_start + method_start_rel
                method_end_abs = self._find_block_end(lines, method_start_abs - 1, body_indent)
                method_ranges.append((method_start_abs, method_end_abs))

            # Scan class body for variable assignments
            for line_num in range(class_start, class_end):
                line = lines[line_num] if line_num < len(lines) else ""
                stripped = line.strip()

                # Skip empty, comments, decorators, def, class
                if not stripped or stripped.startswith('#') or stripped.startswith('@'):
                    continue
                if stripped.startswith('def ') or stripped.startswith('class '):
                    continue

                # Skip if inside a method
                abs_line = line_num + 1
                in_method = any(start <= abs_line <= end for start, end in method_ranges)
                if in_method:
                    continue

                # Check indentation - should be exactly class body level
                current_indent = len(line) - len(line.lstrip())
                if current_indent != body_indent:
                    continue

                # Match variable assignment pattern
                var_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*(?::\s*[^=]+)?\s*=\s*(.+)$', stripped)
                if not var_match:
                    continue

                name = var_match.group(1)
                # Skip dunder and private that look like methods
                if name.startswith('__') and name.endswith('__'):
                    continue

                # Find usages within the class
                class_lines = lines[class_start - 1:class_end]
                usages = self._find_variable_usages(name, class_lines, line_num - class_start + 2, max_usages=5)

                element = CodeElement(
                    scope=scope,
                    repository=repository,
                    username=username,
                    relative_path=file_info.relative_path,
                    element_type="variable",
                    name=name,
                    language="python",
                    line_start=abs_line,
                    line_end=abs_line,
                    raw_code=stripped,
                    level=3,
                    visibility="private" if name.startswith("_") else "public",
                    parent_id=class_elem.element_id,
                    context_usages=usages,
                )
                element.element_id = generate_element_id(
                    scope, repository, username, file_info.relative_path,
                    "variable", name, abs_line
                )
                elements.append(element)

        return elements

    def _find_block_end(self, lines: list[str], start_idx: int, base_indent: int) -> int:
        """Find the end line of an indented block."""
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if not line.strip():
                continue  # Skip empty lines
            if line.startswith("#"):
                continue  # Skip comments

            # Check indentation
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= base_indent and line.strip():
                return i  # End of block

        return len(lines)  # End of file

    def _extract_docstring(self, lines: list[str], block_start: int) -> str | None:
        """Extract docstring from the start of a block."""
        # Look at the line after the definition
        if block_start >= len(lines):
            return None

        for i in range(block_start, min(block_start + 3, len(lines))):
            line = lines[i].strip()
            if line.startswith('"""') or line.startswith("'''"):
                quote = line[:3]
                if line.count(quote) >= 2:
                    # Single line docstring
                    return line[3:-3].strip()
                else:
                    # Multi-line docstring
                    docstring_lines = [line[3:]]
                    for j in range(i + 1, len(lines)):
                        end_line = lines[j]
                        if quote in end_line:
                            docstring_lines.append(end_line[:end_line.index(quote)])
                            break
                        docstring_lines.append(end_line)
                    return "\n".join(docstring_lines).strip()

        return None

    def _extract_decorators(self, decorator_text: str) -> list[str]:
        """Extract decorator names from decorator text."""
        decorators = []
        for line in decorator_text.strip().split("\n"):
            line = line.strip()
            if line.startswith("@"):
                # Extract just the name (without arguments)
                name = line[1:].split("(")[0].strip()
                if name:
                    decorators.append(name)
        return decorators

    def _extract_decorators_from_lines(self, lines: list[str], def_line_idx: int) -> list[str]:
        """Extract decorators by looking at lines before a def.

        Args:
            lines: All lines in the file.
            def_line_idx: 0-indexed line number of the def statement.

        Returns:
            List of decorator names.
        """
        decorators = []
        # Look backwards from the def line
        idx = def_line_idx - 1
        while idx >= 0:
            line = lines[idx].strip()
            if line.startswith("@"):
                # Extract decorator name (without arguments)
                name = line[1:].split("(")[0].strip()
                if name:
                    decorators.insert(0, name)  # Insert at front to preserve order
                idx -= 1
            elif not line or line.startswith("#"):
                # Skip empty lines and comments between decorators
                idx -= 1
            else:
                # Hit non-decorator content, stop
                break
        return decorators

    def _parse_parameters(self, params_str: str) -> list[dict[str, Any]]:
        """Parse parameter string into structured list."""
        params = []
        if not params_str.strip():
            return params

        for param in params_str.split(","):
            param = param.strip()
            if not param or param == "self" or param == "cls":
                continue

            param_info: dict[str, Any] = {"name": param, "type": None, "default": None}

            # Check for type annotation
            if ":" in param:
                name_part, type_part = param.split(":", 1)
                param_info["name"] = name_part.strip()
                # Check for default value
                if "=" in type_part:
                    type_str, default = type_part.split("=", 1)
                    param_info["type"] = type_str.strip()
                    param_info["default"] = default.strip()
                else:
                    param_info["type"] = type_part.strip()
            elif "=" in param:
                name, default = param.split("=", 1)
                param_info["name"] = name.strip()
                param_info["default"] = default.strip()

            params.append(param_info)

        return params

    def _set_hierarchy(self, elements: list[CodeElement], file_element: CodeElement) -> None:
        """Set parent IDs for elements without explicit parents."""
        for elem in elements:
            if elem.element_type == "file":
                continue
            if not elem.parent_id:
                elem.parent_id = file_element.element_id


# =============================================================================
# JAVASCRIPT PARSER (simplified)
# =============================================================================


class JavaScriptParser:
    """Parse JavaScript/TypeScript files and extract code elements."""

    CLASS_PATTERN = re.compile(
        r'^(?P<export>export\s+)?(?P<default>default\s+)?class\s+(?P<name>\w+)',
        re.MULTILINE
    )

    FUNCTION_PATTERN = re.compile(
        r'^(?P<export>export\s+)?(?P<async>async\s+)?function\s+(?P<name>\w+)\s*\(',
        re.MULTILINE
    )

    ARROW_FUNCTION_PATTERN = re.compile(
        r'^(?P<export>export\s+)?(?:const|let|var)\s+(?P<name>\w+)\s*=\s*(?P<async>async\s+)?\([^)]*\)\s*=>',
        re.MULTILINE
    )

    def parse(
        self,
        content: str,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
    ) -> list[CodeElement]:
        """Parse JavaScript content and extract elements."""
        elements: list[CodeElement] = []
        lines = content.split("\n")
        line_count = len(lines)

        # Create file-level element
        file_element = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="file",
            name=Path(file_info.relative_path).name,
            language=file_info.language,
            line_start=1,
            line_end=line_count,
            level=0,
        )
        file_element.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path,
            "file", file_element.name, 1
        )
        elements.append(file_element)

        # Parse classes
        for match in self.CLASS_PATTERN.finditer(content):
            name = match.group("name")
            start_pos = match.start()
            line_start = content[:start_pos].count("\n") + 1
            line_end = self._find_brace_block_end(content, match.end(), lines)

            raw_code = "\n".join(lines[line_start - 1:line_end])

            element = CodeElement(
                scope=scope,
                repository=repository,
                username=username,
                relative_path=file_info.relative_path,
                element_type="class",
                name=name,
                language=file_info.language,
                line_start=line_start,
                line_end=line_end,
                raw_code=raw_code,
                level=1,
                parent_id=file_element.element_id,
            )
            element.element_id = generate_element_id(
                scope, repository, username, file_info.relative_path,
                "class", name, line_start
            )
            elements.append(element)

        # Parse functions
        for match in self.FUNCTION_PATTERN.finditer(content):
            name = match.group("name")
            is_async = bool(match.group("async"))
            start_pos = match.start()
            line_start = content[:start_pos].count("\n") + 1
            line_end = self._find_brace_block_end(content, match.end(), lines)

            raw_code = "\n".join(lines[line_start - 1:line_end])

            element = CodeElement(
                scope=scope,
                repository=repository,
                username=username,
                relative_path=file_info.relative_path,
                element_type="function",
                name=name,
                language=file_info.language,
                line_start=line_start,
                line_end=line_end,
                raw_code=raw_code,
                is_async=is_async,
                level=2,
                parent_id=file_element.element_id,
            )
            element.element_id = generate_element_id(
                scope, repository, username, file_info.relative_path,
                "function", name, line_start
            )
            elements.append(element)

        # Parse arrow functions
        for match in self.ARROW_FUNCTION_PATTERN.finditer(content):
            name = match.group("name")
            is_async = bool(match.group("async"))
            start_pos = match.start()
            line_start = content[:start_pos].count("\n") + 1

            element = CodeElement(
                scope=scope,
                repository=repository,
                username=username,
                relative_path=file_info.relative_path,
                element_type="function",
                name=name,
                language=file_info.language,
                line_start=line_start,
                line_end=line_start,  # Simplified
                is_async=is_async,
                level=2,
                parent_id=file_element.element_id,
            )
            element.element_id = generate_element_id(
                scope, repository, username, file_info.relative_path,
                "function", name, line_start
            )
            elements.append(element)

        return elements

    def _find_brace_block_end(self, content: str, start_pos: int, lines: list[str]) -> int:
        """Find end of a brace-delimited block."""
        brace_count = 0
        found_open = False

        for i, char in enumerate(content[start_pos:], start_pos):
            if char == "{":
                brace_count += 1
                found_open = True
            elif char == "}":
                brace_count -= 1
                if found_open and brace_count == 0:
                    return content[:i + 1].count("\n") + 1

        return len(lines)


# =============================================================================
# PARSER REGISTRY
# =============================================================================


PARSERS: dict[str, PythonParser | JavaScriptParser] = {
    "python": PythonParser(),
    "javascript": JavaScriptParser(),
    "typescript": JavaScriptParser(),  # Use same parser for TS
}


def get_parser(language: str) -> PythonParser | JavaScriptParser | None:
    """Get parser for language."""
    return PARSERS.get(language)


# =============================================================================
# MAIN PARSE FUNCTION
# =============================================================================


def parse_file(
    file_info: FileInfo,
    scope: str,
    repository: str,
    username: str,
) -> ParsedFile:
    """Parse a single file and extract code elements.

    Args:
        file_info: File information from change detection.
        scope: Repository scope.
        repository: Repository name.
        username: Username (branch).

    Returns:
        ParsedFile with extracted elements.
    """
    result = ParsedFile(file_info=file_info)

    try:
        # Read file content
        content = file_info.absolute_path.read_text(encoding="utf-8", errors="replace")
        result.line_count = content.count("\n") + 1

        # Get parser for language
        parser = get_parser(file_info.language)
        if parser is None:
            result.parse_errors.append(f"No parser for language: {file_info.language}")
            return result

        # Parse and extract elements
        result.elements = parser.parse(content, file_info, scope, repository, username)

    except Exception as e:
        result.parse_errors.append(str(e))

    return result


def parse_files(
    manifest: ChangeManifest,
    on_progress: Callable[[int, int], None] | None = None,
) -> ParsingResult:
    """Parse all files in a change manifest.

    Args:
        manifest: Change manifest from Phase 2.
        on_progress: Optional callback(completed, total) for progress updates.

    Returns:
        ParsingResult with all parsed files and elements.
    """
    result = ParsingResult(
        scope=manifest.scope,
        repository=manifest.repository,
        username=manifest.username,
    )

    # Parse new and modified files
    files_to_parse = manifest.new_files + manifest.modified_files
    total = len(files_to_parse)

    for i, file_info in enumerate(files_to_parse):
        try:
            parsed = parse_file(
                file_info,
                manifest.scope,
                manifest.repository,
                manifest.username,
            )
            result.parsed_files.append(parsed)
        except Exception as e:
            result.failed_files.append((file_info, str(e)))

        if on_progress:
            on_progress(i + 1, total)

    return result
