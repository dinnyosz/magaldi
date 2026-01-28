"""Tree-sitter query runner for SCM-based code extraction.

This module provides infrastructure for loading and executing tree-sitter
queries from .scm files, enabling declarative pattern-based code extraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from tree_sitter import Language, Query, QueryCursor, Tree

if TYPE_CHECKING:
    from tree_sitter import Node

log = logging.getLogger(__name__)

# Path to query files
QUERIES_DIR = Path(__file__).parent / "queries"


@dataclass
class QueryMatch:
    """A single match from a tree-sitter query.

    Attributes:
        pattern_index: Index of the matched pattern in the query.
        captures: Dict mapping capture names to lists of matched nodes.
    """

    pattern_index: int
    captures: dict[str, list["Node"]]

    def get(self, capture_name: str) -> "Node | None":
        """Get the first node for a capture name, or None."""
        nodes = self.captures.get(capture_name, [])
        return nodes[0] if nodes else None

    def get_all(self, capture_name: str) -> list["Node"]:
        """Get all nodes for a capture name."""
        return self.captures.get(capture_name, [])

    def get_text(self, capture_name: str) -> str | None:
        """Get the text of the first node for a capture name."""
        node = self.get(capture_name)
        return node.text.decode("utf-8") if node else None

    def has(self, capture_name: str) -> bool:
        """Check if a capture name is present in this match."""
        return capture_name in self.captures and len(self.captures[capture_name]) > 0


@dataclass
class QueryResult:
    """Result of running a query on a tree.

    Attributes:
        query_name: Name of the query file (without .scm).
        language: Language the query was run for.
        matches: List of all matches found.
    """

    query_name: str
    language: str
    matches: list[QueryMatch] = field(default_factory=list)

    def filter_by_capture(self, capture_name: str) -> Iterator[QueryMatch]:
        """Yield only matches that have the specified capture."""
        for match in self.matches:
            if match.has(capture_name):
                yield match

    def __len__(self) -> int:
        return len(self.matches)


class QueryRunner:
    """Loads and executes tree-sitter queries from .scm files.

    This class provides:
    - Query file loading from the queries/ directory
    - Query compilation and caching
    - Query execution on parsed trees
    - Match filtering and iteration

    Usage:
        runner = QueryRunner(language)
        result = runner.run("elements", tree)
        for match in result.filter_by_capture("function.def"):
            name = match.get_text("function.name")
            print(f"Found function: {name}")
    """

    def __init__(self, language: Language, language_name: str) -> None:
        """Initialize the query runner.

        Args:
            language: The tree-sitter Language object.
            language_name: Name of the language (python, javascript, etc.)
        """
        self._language = language
        self._language_name = language_name
        self._queries: dict[str, Query] = {}
        self._query_dir = QUERIES_DIR / language_name

    def _load_query(self, query_name: str) -> Query:
        """Load and compile a query from a .scm file.

        Args:
            query_name: Name of the query file (without .scm extension).

        Returns:
            Compiled Query object.

        Raises:
            FileNotFoundError: If the query file doesn't exist.
            QueryError: If the query syntax is invalid.
        """
        if query_name in self._queries:
            return self._queries[query_name]

        query_path = self._query_dir / f"{query_name}.scm"
        if not query_path.exists():
            raise FileNotFoundError(
                f"Query file not found: {query_path}. "
                f"Available queries: {self.list_queries()}"
            )

        query_text = query_path.read_text(encoding="utf-8")
        query = Query(self._language, query_text)

        self._queries[query_name] = query
        log.debug(f"Loaded query '{query_name}' with {query.pattern_count} patterns")
        return query

    def run(self, query_name: str, tree: Tree) -> QueryResult:
        """Run a query on a parsed tree.

        Args:
            query_name: Name of the query file (without .scm).
            tree: Parsed tree-sitter Tree.

        Returns:
            QueryResult containing all matches.
        """
        query = self._load_query(query_name)
        cursor = QueryCursor(query)

        matches = []
        for pattern_idx, captures in cursor.matches(tree.root_node):
            matches.append(QueryMatch(pattern_index=pattern_idx, captures=captures))

        return QueryResult(
            query_name=query_name,
            language=self._language_name,
            matches=matches,
        )

    def run_on_node(self, query_name: str, node: "Node") -> QueryResult:
        """Run a query on a specific node (not the whole tree).

        Args:
            query_name: Name of the query file (without .scm).
            node: The node to query within.

        Returns:
            QueryResult containing matches within the node.
        """
        query = self._load_query(query_name)
        cursor = QueryCursor(query)

        matches = []
        for pattern_idx, captures in cursor.matches(node):
            matches.append(QueryMatch(pattern_index=pattern_idx, captures=captures))

        return QueryResult(
            query_name=query_name,
            language=self._language_name,
            matches=matches,
        )

    def list_queries(self) -> list[str]:
        """List available query files for this language.

        Returns:
            List of query names (without .scm extension).
        """
        if not self._query_dir.exists():
            return []
        return [p.stem for p in self._query_dir.glob("*.scm")]

    def reload_query(self, query_name: str) -> None:
        """Force reload a query from disk (for hot-reloading).

        Args:
            query_name: Name of the query to reload.
        """
        if query_name in self._queries:
            del self._queries[query_name]
        self._load_query(query_name)

    def get_query_source(self, query_name: str) -> str:
        """Get the source code of a query file.

        Args:
            query_name: Name of the query file.

        Returns:
            The query file contents as a string.
        """
        query_path = self._query_dir / f"{query_name}.scm"
        if not query_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_path}")
        return query_path.read_text(encoding="utf-8")


class QueryRunnerManager:
    """Manages QueryRunner instances for multiple languages.

    This is typically accessed via the TreeSitterManager.
    """

    def __init__(self) -> None:
        self._runners: dict[str, QueryRunner] = {}

    def get_runner(self, language: Language, language_name: str) -> QueryRunner:
        """Get or create a QueryRunner for a language.

        Args:
            language: The tree-sitter Language object.
            language_name: Name of the language.

        Returns:
            QueryRunner for the specified language.
        """
        if language_name not in self._runners:
            self._runners[language_name] = QueryRunner(language, language_name)
        return self._runners[language_name]

    def list_languages_with_queries(self) -> list[str]:
        """List languages that have query files defined.

        Returns:
            List of language names with queries/ subdirectories.
        """
        if not QUERIES_DIR.exists():
            return []
        return [
            d.name
            for d in QUERIES_DIR.iterdir()
            if d.is_dir() and list(d.glob("*.scm"))
        ]
