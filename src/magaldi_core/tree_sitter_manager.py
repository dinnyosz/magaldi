"""Tree-sitter parser manager for multiple languages.

This module provides:
- Lazy loading of tree-sitter language parsers
- Unified interface for parsing source code into ASTs
- Tree traversal utilities for extracting code elements

All extraction functions and data classes are re-exported from the
extractors and analysis subpackages for backward compatibility.
"""

from __future__ import annotations

from typing import Any

import tree_sitter_bash as ts_bash
import tree_sitter_dockerfile as ts_dockerfile
import tree_sitter_javascript as ts_javascript
import tree_sitter_markdown as ts_markdown
import tree_sitter_php as ts_php
import tree_sitter_python as ts_python
import tree_sitter_rust as ts_rust
import tree_sitter_toml as ts_toml
import tree_sitter_typescript as ts_typescript
import tree_sitter_yaml as ts_yaml
from tree_sitter import Language, Parser, Tree

from magaldi_core.analysis.api_detection import (
    detect_cli_commands,
    detect_http_routes,
    detect_patterns,
    detect_public_api,
)
from magaldi_core.analysis.semantic import (
    IMPURE_CALLS,
    analyze_purity,
    associate_comments,
    extract_comments,
    extract_section_markers,
    extract_side_effects,
    extract_todos,
    extract_type_annotations,
)
from magaldi_core.extractors.base import (
    find_nodes,
    get_child_by_field,
    get_children_by_type,
    get_node_text,
    walk_tree,
)
from magaldi_core.extractors.javascript import (
    JavaScriptExtractor,
    extract_javascript_base_class,
    extract_javascript_calls,
    extract_javascript_class_fields,
    extract_javascript_class_members,
    extract_javascript_elements,
    extract_javascript_imports,
    extract_javascript_modified_properties,
    extract_javascript_references,
    extract_javascript_thrown_exceptions,
    extract_top_level_javascript_calls,
)
from magaldi_core.extractors.php import (
    PHPExtractor,
    extract_php_base_class,
    extract_php_calls,
    extract_php_class_members,
    extract_php_class_properties,
    extract_php_elements,
    extract_php_imports,
    extract_php_modified_properties,
    extract_php_thrown_exceptions,
    extract_top_level_php_calls,
)
from magaldi_core.extractors.python import (
    PythonExtractor,
    extract_python_base_classes,
    extract_python_calls,
    extract_python_class_attributes,
    extract_python_class_members,
    extract_python_elements,
    extract_python_imports,
    extract_python_modified_attributes,
    extract_python_raised_exceptions,
    extract_python_references,
    extract_top_level_python_calls,
)
from magaldi_core.extractors.rust import (
    RustExtractor,
    extract_rust_calls,
    extract_rust_elements,
    extract_rust_impl_members,
    extract_rust_impl_traits,
    extract_rust_imports,
    extract_rust_modified_fields,
    extract_rust_panics,
    extract_rust_struct_fields,
    extract_top_level_rust_calls,
)
from magaldi_core.extractors.types import (
    CliCommand,
    Comment,
    DecoratorInfo,
    ExtractedCall,
    ExtractedElement,
    ExtractedImport,
    ExtractedReference,
    HttpRoute,
    ParameterInfo,
    PurityInfo,
    SectionMarker,
    SideEffect,
    TodoItem,
    TreeSitterError,
    TypeAnnotation,
)
from magaldi_core.query_runner import (
    QueryMatch,
    QueryResult,
    QueryRunner,
    QueryRunnerManager,
)

# =============================================================================
# TREE-SITTER MANAGER
# =============================================================================


class TreeSitterManager:
    """Manages tree-sitter parsers for multiple languages.

    This class provides:
    - Language registration with their tree-sitter bindings
    - Parser creation and caching
    - AST parsing for source code
    """

    # Language configuration: name -> (module, language_func)
    LANGUAGE_CONFIG: dict[str, tuple[Any, str]] = {
        "python": (ts_python, "language"),
        "javascript": (ts_javascript, "language"),
        "typescript": (ts_typescript, "language_typescript"),
        "tsx": (ts_typescript, "language_tsx"),
        "php": (ts_php, "language_php"),
        "rust": (ts_rust, "language"),
        "markdown": (ts_markdown, "language"),
        "yaml": (ts_yaml, "language"),
        "toml": (ts_toml, "language"),
        "dockerfile": (ts_dockerfile, "language"),
        "bash": (ts_bash, "language"),
    }

    def __init__(self) -> None:
        self._languages: dict[str, Language] = {}
        self._parsers: dict[str, Parser] = {}
        self._extractors: dict[str, Any] = {}
        self._query_manager = QueryRunnerManager()

    def _get_language(self, language: str) -> Language:
        """Get or create Language object for a language."""
        if language in self._languages:
            return self._languages[language]

        if language not in self.LANGUAGE_CONFIG:
            raise TreeSitterError(f"Unsupported language: {language}")

        module, func_name = self.LANGUAGE_CONFIG[language]
        lang_func = getattr(module, func_name)
        lang = Language(lang_func())

        self._languages[language] = lang
        return lang

    def get_parser(self, language: str) -> Parser:
        """Get or create parser for a language."""
        if language in self._parsers:
            return self._parsers[language]

        parser = Parser(self._get_language(language))
        self._parsers[language] = parser
        return parser

    def parse(self, content: bytes, language: str) -> Tree:
        """Parse source code into AST."""
        parser = self.get_parser(language)
        return parser.parse(content)

    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported."""
        return language in self.LANGUAGE_CONFIG

    def get_extractor(self, language: str) -> Any:
        """Get or create an extractor for a language.

        Args:
            language: Language name (python, javascript, typescript, php, rust)

        Returns:
            An extractor instance for the language, or None if unsupported.
        """
        if language in self._extractors:
            return self._extractors[language]

        extractor = None
        if language == "python":
            extractor = PythonExtractor()
        elif language in ("javascript", "typescript", "tsx"):
            extractor = JavaScriptExtractor(language=language)
        elif language == "php":
            extractor = PHPExtractor()
        elif language == "rust":
            extractor = RustExtractor()
        elif language == "markdown":
            from magaldi_core.extractors.markdown import MarkdownExtractor
            extractor = MarkdownExtractor()
        elif language == "yaml":
            from magaldi_core.extractors.yaml_lang import YamlExtractor
            extractor = YamlExtractor()
        elif language == "toml":
            from magaldi_core.extractors.toml_lang import TomlExtractor
            extractor = TomlExtractor()
        elif language == "dockerfile":
            from magaldi_core.extractors.dockerfile import DockerfileExtractor
            extractor = DockerfileExtractor()
        elif language == "bash":
            from magaldi_core.extractors.bash import BashExtractor
            extractor = BashExtractor()

        if extractor:
            self._extractors[language] = extractor

        return extractor

    def extract_elements(
        self, source: str, language: str
    ) -> list[ExtractedElement]:
        """Extract code elements from source code.

        This is a convenience method that parses the source and
        extracts elements in one step.

        Args:
            source: Source code as string.
            language: Language name.

        Returns:
            List of extracted elements.
        """
        extractor = self.get_extractor(language)
        if not extractor:
            return []

        tree = self.parse(source.encode("utf-8"), language)
        lines = source.split("\n")
        return extractor.extract_elements(tree, lines)

    def get_query_runner(self, language: str) -> QueryRunner:
        """Get a query runner for executing SCM queries on a language.

        Args:
            language: Language name (python, javascript, etc.)

        Returns:
            QueryRunner instance for the specified language.

        Raises:
            TreeSitterError: If the language is not supported.
        """
        lang = self._get_language(language)
        return self._query_manager.get_runner(lang, language)

    def run_query(
        self, source: str, language: str, query_name: str
    ) -> QueryResult:
        """Run a query on source code.

        This is a convenience method that parses the source and
        runs a query in one step.

        Args:
            source: Source code as string.
            language: Language name.
            query_name: Name of the query file (without .scm).

        Returns:
            QueryResult containing all matches.
        """
        tree = self.parse(source.encode("utf-8"), language)
        runner = self.get_query_runner(language)
        return runner.run(query_name, tree)

    def run_query_on_tree(
        self, tree: Tree, language: str, query_name: str
    ) -> QueryResult:
        """Run a query on an already-parsed tree.

        Use this when you already have a parsed tree to avoid re-parsing.

        Args:
            tree: Already-parsed tree-sitter Tree.
            language: Language name.
            query_name: Name of the query file (without .scm).

        Returns:
            QueryResult containing all matches.
        """
        runner = self.get_query_runner(language)
        return runner.run(query_name, tree)

    def list_queries(self, language: str) -> list[str]:
        """List available query files for a language.

        Args:
            language: Language name.

        Returns:
            List of query names (without .scm extension).
        """
        runner = self.get_query_runner(language)
        return runner.list_queries()


# =============================================================================
# GLOBAL SINGLETON
# =============================================================================

_manager: TreeSitterManager | None = None


def get_manager() -> TreeSitterManager:
    """Get or create global TreeSitterManager instance."""
    global _manager
    if _manager is None:
        _manager = TreeSitterManager()
    return _manager


# =============================================================================
# PUBLIC API EXPORTS
# =============================================================================

__all__ = [
    # Manager
    "TreeSitterManager",
    "TreeSitterError",
    "get_manager",
    # Query runner
    "QueryRunner",
    "QueryMatch",
    "QueryResult",
    # Tree utilities
    "walk_tree",
    "find_nodes",
    "get_node_text",
    "get_child_by_field",
    "get_children_by_type",
    # Data classes
    "ExtractedImport",
    "DecoratorInfo",
    "ParameterInfo",
    "ExtractedElement",
    "ExtractedReference",
    "ExtractedCall",
    "TypeAnnotation",
    "TodoItem",
    "SectionMarker",
    "Comment",
    "HttpRoute",
    "CliCommand",
    "PurityInfo",
    "SideEffect",
    # Python extraction
    "extract_python_elements",
    "extract_python_class_members",
    "extract_python_imports",
    "extract_python_references",
    "extract_python_calls",
    "extract_top_level_python_calls",
    "extract_python_class_attributes",
    "extract_python_base_classes",
    "extract_python_raised_exceptions",
    "extract_python_modified_attributes",
    # JavaScript extraction
    "extract_javascript_elements",
    "extract_javascript_class_members",
    "extract_javascript_imports",
    "extract_javascript_references",
    "extract_javascript_calls",
    "extract_top_level_javascript_calls",
    "extract_javascript_class_fields",
    "extract_javascript_base_class",
    "extract_javascript_thrown_exceptions",
    "extract_javascript_modified_properties",
    # PHP extraction
    "extract_php_elements",
    "extract_php_class_members",
    "extract_php_imports",
    "extract_php_calls",
    "extract_top_level_php_calls",
    "extract_php_class_properties",
    "extract_php_base_class",
    "extract_php_thrown_exceptions",
    "extract_php_modified_properties",
    # Rust extraction
    "extract_rust_elements",
    "extract_rust_impl_members",
    "extract_rust_imports",
    "extract_rust_calls",
    "extract_top_level_rust_calls",
    "extract_rust_struct_fields",
    "extract_rust_impl_traits",
    "extract_rust_panics",
    "extract_rust_modified_fields",
    # Analysis - semantic
    "extract_todos",
    "extract_section_markers",
    "extract_comments",
    "associate_comments",
    "analyze_purity",
    "extract_side_effects",
    "extract_type_annotations",
    "IMPURE_CALLS",
    # Analysis - API detection
    "detect_http_routes",
    "detect_cli_commands",
    "detect_public_api",
    "detect_patterns",
]
