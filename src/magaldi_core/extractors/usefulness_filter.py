"""Variable/constant usefulness filter for all language extractors.

This module provides shared logic for determining whether a variable or constant
assignment is useful for code discovery, or should be skipped as transient/temporary.

The filter is applied AFTER extraction to ensure consistent behavior across all
language extractors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# USEFULNESS CRITERIA
# =============================================================================

# Factory functions that produce useful, indexable values (language-agnostic names)
USEFUL_FACTORIES = frozenset({
    # Enums (Python, TypeScript)
    "Enum", "IntEnum", "StrEnum", "Flag", "IntFlag", "auto",
    # Typing constructs (Python)
    "TypeVar", "ParamSpec", "TypeVarTuple", "NewType", "TypedDict",
    # Collections
    "namedtuple", "NamedTuple", "defaultdict", "OrderedDict", "Counter",
    # Regex
    "compile", "RegExp",
    # Logging
    "getLogger",
    # Threading primitives
    "Lock", "RLock", "Semaphore", "BoundedSemaphore", "Condition", "Event", "Barrier",
    "Mutex", "RwLock",  # Rust
    # Paths
    "Path", "PurePath", "PurePosixPath", "PureWindowsPath",
    # Dataclass
    "field",
    # Functools
    "partial", "lru_cache", "cache",
    # JavaScript/TypeScript specific
    "Symbol", "Map", "Set", "WeakMap", "WeakSet",
    "createContext", "createStore", "defineStore",  # React/Vue/Pinia
    "ref", "reactive", "computed",  # Vue
    "signal", "computed",  # Solid/Angular signals
})

# Attribute-style factory calls that produce useful values
USEFUL_ATTRIBUTE_FACTORIES = frozenset({
    # Python
    "re.compile",
    "logging.getLogger",
    "threading.Lock",
    "threading.RLock",
    "threading.Semaphore",
    "threading.Condition",
    "threading.Event",
    "pathlib.Path",
    "functools.partial",
    "functools.lru_cache",
    "collections.defaultdict",
    "collections.OrderedDict",
    "collections.Counter",
    "collections.namedtuple",
    # JavaScript
    "React.createContext",
    "React.createRef",
    "React.forwardRef",
    "React.memo",
    "React.lazy",
    # Rust
    "Mutex::new",
    "RwLock::new",
    "Arc::new",
    "Rc::new",
})

# Short/temporary variable names to always skip (language-agnostic)
SKIP_NAMES = frozenset({
    # Loop variables
    "i", "j", "k", "n", "m", "idx", "index",
    # Coordinates
    "x", "y", "z", "w",
    # Throwaway
    "_", "__", "___",
    # Self references
    "self", "cls", "mcs", "this",
    # Common temps
    "tmp", "temp", "ret", "res", "val", "var", "out", "result",
    # Exception handlers
    "e", "ex", "err", "exc", "error",
    # PHP specific
    "row", "item", "key", "value",
})

# Literal AST node types by language that indicate useful values
LITERAL_TYPES = {
    "python": {
        "string", "integer", "float", "true", "false", "none",
        "list", "dictionary", "tuple", "set",
        "concatenated_string", "binary_operator",
    },
    "javascript": {
        "string", "number", "true", "false", "null", "undefined",
        "array", "object", "regex", "template_string",
    },
    "typescript": {
        "string", "number", "true", "false", "null", "undefined",
        "array", "object", "regex", "template_string",
        "as_expression",  # const x = {...} as const
    },
    "php": {
        "string", "integer", "float", "boolean", "null",
        "array_creation_expression", "encapsed_string",
    },
    "rust": {
        "string_literal", "integer_literal", "float_literal", "boolean_literal",
        "char_literal", "array_expression", "tuple_expression",
    },
}

# Type reference node types by language (for type aliases)
TYPE_REFERENCE_TYPES = {
    "python": {"identifier", "subscript", "attribute"},
    "javascript": {"identifier", "member_expression", "type_identifier"},
    "typescript": {"identifier", "member_expression", "type_identifier", "generic_type"},
    "php": {"name", "qualified_name"},
    "rust": {"identifier", "scoped_identifier", "generic_type"},
}


# =============================================================================
# EXTRACTION STATS
# =============================================================================

@dataclass
class SkippedVariable:
    """Record of a variable that was skipped by the usefulness filter."""

    name: str
    line: int
    reason: str
    value_type: str
    value_preview: str


@dataclass
class ExtractionStats:
    """Statistics about variable extraction for a single file."""

    kept_count: int = 0
    skipped_count: int = 0
    skipped_variables: list[SkippedVariable] = field(default_factory=list)

    def record_skip(
        self, name: str, line: int, reason: str, value_type: str, value_preview: str
    ) -> None:
        """Record a skipped variable."""
        self.skipped_count += 1
        self.skipped_variables.append(
            SkippedVariable(
                name=name,
                line=line,
                reason=reason,
                value_type=value_type,
                value_preview=value_preview[:60] if value_preview else "",
            )
        )

    def record_keep(self) -> None:
        """Record a kept variable."""
        self.kept_count += 1

    def log_summary(self, file_path: str) -> None:
        """Log a summary of extraction stats."""
        if self.skipped_count > 0:
            logger.debug(
                "Variable filter for %s: kept=%d, skipped=%d",
                file_path,
                self.kept_count,
                self.skipped_count,
            )
            for sv in self.skipped_variables:
                logger.debug(
                    "  SKIP: %s (line %d) - %s [%s] %s",
                    sv.name,
                    sv.line,
                    sv.reason,
                    sv.value_type,
                    sv.value_preview,
                )


# Thread-local stats for current file being parsed
_current_stats: ExtractionStats | None = None

# Global accumulator for total filtered stats across all files
_global_kept: int = 0
_global_skipped: int = 0


def get_extraction_stats() -> ExtractionStats:
    """Get or create extraction stats for current parse."""
    global _current_stats
    if _current_stats is None:
        _current_stats = ExtractionStats()
    return _current_stats


def reset_extraction_stats() -> ExtractionStats | None:
    """Reset and return the extraction stats. Call after parsing a file."""
    global _current_stats, _global_kept, _global_skipped
    stats = _current_stats
    if stats:
        _global_kept += stats.kept_count
        _global_skipped += stats.skipped_count
    _current_stats = None
    return stats


def get_global_filter_stats() -> tuple[int, int]:
    """Get total kept and skipped counts across all files.

    Returns:
        Tuple of (kept_count, skipped_count).
    """
    return _global_kept, _global_skipped


def reset_global_filter_stats() -> tuple[int, int]:
    """Reset and return global filter stats. Call at start of parsing session.

    Returns:
        Tuple of (kept_count, skipped_count) before reset.
    """
    global _global_kept, _global_skipped
    kept, skipped = _global_kept, _global_skipped
    _global_kept = 0
    _global_skipped = 0
    return kept, skipped


# =============================================================================
# USEFULNESS CHECK
# =============================================================================

def is_useful_name(name: str) -> tuple[bool, str]:
    """Check if a variable name suggests it's useful.

    Returns:
        Tuple of (is_useful, skip_reason). If is_useful is True, skip_reason is empty.
    """
    # Skip short/temp names
    if name in SKIP_NAMES:
        return False, "temp_name"

    # Keep constants by naming convention (UPPER_CASE)
    if name.isupper() and len(name) > 1:
        return True, ""

    # Keep dunder names (module metadata like __all__, __version__)
    if name.startswith("__") and name.endswith("__"):
        return True, ""

    # Name alone doesn't determine usefulness
    return True, ""


def is_useful_value_type(
    value_type: str,
    language: str,
) -> tuple[bool, str]:
    """Check if a value type suggests the assignment is useful.

    Args:
        value_type: The AST node type of the value.
        language: The programming language.

    Returns:
        Tuple of (is_useful, skip_reason). If is_useful is True, skip_reason is empty.
    """
    # Literal values are useful (configuration, data)
    literal_types = LITERAL_TYPES.get(language, set())
    if value_type in literal_types:
        return True, ""

    # Type references are useful (type aliases)
    type_ref_types = TYPE_REFERENCE_TYPES.get(language, set())
    if value_type in type_ref_types:
        return True, ""

    # Arrow/lambda functions are useful
    if value_type in ("arrow_function", "lambda", "function_expression", "closure_expression"):
        return True, ""

    # new_expression needs to check the constructor name (handled by is_useful_factory_call)
    # So we don't auto-accept it here
    if value_type == "new_expression":
        return True, ""  # Let is_useful_factory_call decide

    # Value type alone doesn't determine usefulness
    return True, ""


def is_useful_factory_call(func_name: str) -> tuple[bool, str]:
    """Check if a function/constructor call produces a useful value.

    Args:
        func_name: The name of the function being called.

    Returns:
        Tuple of (is_useful, skip_reason). If is_useful is True, skip_reason is empty.
    """
    # Known useful factory functions
    if func_name in USEFUL_FACTORIES:
        return True, ""

    # Known useful attribute factories
    if func_name in USEFUL_ATTRIBUTE_FACTORIES:
        return True, ""

    # Check the last part of attribute for useful factories
    if "." in func_name or "::" in func_name:
        # Split on . or ::
        parts = func_name.replace("::", ".").split(".")
        last_part = parts[-1]
        if last_part in USEFUL_FACTORIES:
            return True, ""

    # Instance creation (PascalCase class name)
    if func_name and func_name[0].isupper() and not func_name.isupper():
        return False, "instance_creation"

    # Method calls (contains . or ::)
    if "." in func_name or "::" in func_name:
        return False, "method_call_result"

    # Function call results (lowercase function)
    if func_name and func_name[0].islower():
        return False, "function_call_result"

    return True, ""


def should_skip_variable(
    name: str,
    value_type: str,
    func_name: str | None,
    language: str,
    is_module_level: bool = True,
) -> tuple[bool, str]:
    """Determine if a variable should be skipped.

    This is the main entry point for the usefulness filter.

    Args:
        name: Variable name.
        value_type: AST node type of the value.
        func_name: Function name if value is a call, None otherwise.
        language: Programming language.
        is_module_level: Whether this is a module-level (not local) variable.

    Returns:
        Tuple of (should_skip, reason). If should_skip is False, reason is empty.
    """
    # Check name first
    is_useful, reason = is_useful_name(name)
    if not is_useful:
        return True, reason

    # Constants by naming convention are always useful
    if name.isupper() and len(name) > 1:
        return False, ""

    # Dunder names are always useful
    if name.startswith("__") and name.endswith("__"):
        return False, ""

    # Check value type
    is_useful, reason = is_useful_value_type(value_type, language)
    if not is_useful:
        return True, reason

    # Check if it's a call
    if func_name is not None:
        is_useful, reason = is_useful_factory_call(func_name)
        if not is_useful:
            return True, reason

    # Module-level variables with non-call values are generally useful
    if is_module_level and func_name is None:
        return False, ""

    # Local variables without clear usefulness indicators should be skipped
    if not is_module_level:
        return True, "local_variable"

    return False, ""
