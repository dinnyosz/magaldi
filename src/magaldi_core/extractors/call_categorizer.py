"""Call categorization logic for classifying function/method calls.

This module provides utilities to categorize calls as:
- builtin: Language built-ins — bare functions (len, str) and standard type
  methods (dict.get, list.append, Array.push, Vec.iter, etc.)
- stdlib: Standard library (os.path, json.dumps, etc.)
- external: Third-party libraries (logger.info, requests.get, etc.)
- type_resolvable: Could be resolved via type annotation
- untyped: Method call on object without type info
- dynamic: Dynamically determined calls
- resolved: Successfully resolved to an element
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from .types import CallCategory

if TYPE_CHECKING:
    pass


class CallLike(Protocol):
    """Protocol for call objects (works with both Call and ExtractedCall)."""

    name: str
    receiver: str | None
    resolved_id: str | None
    category: str


# Python built-in functions
BUILTINS: set[str] = {
    # Built-in functions
    "abs", "aiter", "all", "any", "anext", "ascii",
    "bin", "bool", "breakpoint", "bytearray", "bytes",
    "callable", "chr", "classmethod", "compile", "complex",
    "delattr", "dict", "dir", "divmod",
    "enumerate", "eval", "exec",
    "filter", "float", "format", "frozenset",
    "getattr", "globals",
    "hasattr", "hash", "help", "hex",
    "id", "input", "int", "isinstance", "issubclass", "iter",
    "len", "list", "locals",
    "map", "max", "memoryview", "min",
    "next",
    "object", "oct", "open", "ord",
    "pow", "print", "property",
    "range", "repr", "reversed", "round",
    "set", "setattr", "slice", "sorted", "staticmethod", "str", "sum", "super",
    "tuple", "type",
    "vars",
    "zip",
    # Common type constructors used as functions
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "RuntimeError", "StopIteration", "NotImplementedError",
}

# Language-specific built-in methods that should never go through embedding
# resolution. These are methods on standard/primitive types (dict, list, str,
# Array, Vec, etc.) that exist in every program and would match random codebase
# elements by name if sent to Strategy 6.
#
# NOTE: If the receiver is a typed parameter (e.g. repo: Repository), the call
# will still be categorized as TYPE_RESOLVABLE so Strategy 5 can resolve it.
BUILTIN_METHODS: dict[str, set[str] | None] = {
    "python": {
        # dict
        "get", "keys", "values", "items", "pop", "update", "setdefault",
        "clear", "copy", "fromkeys",
        # list
        "append", "extend", "insert", "remove", "sort", "reverse",
        "index", "count",
        # str
        "split", "join", "strip", "lstrip", "rstrip", "replace", "find",
        "rfind", "startswith", "endswith", "upper", "lower", "title",
        "format", "encode", "decode",
        # set
        "add", "discard", "union", "intersection", "difference",
        "symmetric_difference", "issubset", "issuperset",
        # file / io
        "close", "read", "write", "seek", "tell", "flush",
        # Future / Task
        "result", "cancel", "done", "exception",
        # Executor
        "submit", "shutdown",
        # Lock / Condition / Event
        "acquire", "release", "wait", "notify",
        # Queue
        "put", "empty", "full", "qsize",
        # logging (receiver = logger)
        "debug", "info", "warning", "error", "critical",
    },
    "javascript": {
        # Array
        "push", "pop", "shift", "unshift", "splice", "slice", "concat",
        "join", "reverse", "sort", "indexOf", "lastIndexOf", "find",
        "findIndex", "filter", "map", "reduce", "reduceRight", "forEach",
        "some", "every", "includes", "flat", "flatMap", "fill", "entries",
        "keys", "values", "at", "from",
        # String
        "split", "trim", "trimStart", "trimEnd", "replace", "replaceAll",
        "match", "matchAll", "search", "startsWith", "endsWith",
        "charAt", "charCodeAt", "substring", "toLowerCase", "toUpperCase",
        "padStart", "padEnd", "repeat", "normalize",
        # Object
        "hasOwnProperty", "toString", "valueOf", "toLocaleString",
        # Map / Set
        "get", "set", "has", "delete", "clear",
        # Promise
        "then", "catch", "finally",
        # JSON / console (receiver-based)
        "parse", "stringify", "log", "warn", "error", "info", "debug",
        # DOM
        "addEventListener", "removeEventListener", "querySelector",
        "querySelectorAll", "getElementById", "setAttribute",
        "getAttribute", "appendChild", "removeChild",
    },
    # TypeScript and TSX inherit from JavaScript
    "typescript": None,
    "tsx": None,
    "php": {
        # Array-like
        "push", "pop", "shift", "unshift", "count", "merge",
        "map", "filter", "reduce", "keys", "values", "slice",
        "splice", "sort", "reverse", "search", "unique", "flip",
        "combine", "chunk", "pad", "fill", "column",
        # String
        "trim", "replace", "split", "join", "substr", "substring",
        "indexOf", "contains", "startsWith", "endsWith",
        "toUpper", "toLower", "length", "format",
        # Collection (Laravel / common)
        "get", "set", "has", "all", "first", "last", "each",
        "where", "pluck", "groupBy", "sortBy", "toArray", "toJson",
        "isEmpty", "isNotEmpty", "flatten", "collapse",
        # Common object
        "save", "delete", "update", "create", "find", "findOrFail",
        "close", "read", "write", "flush",
        # Logging
        "debug", "info", "warning", "error", "critical", "log",
    },
    "rust": {
        # Option / Result
        "unwrap", "unwrap_or", "unwrap_or_else", "unwrap_or_default",
        "expect", "is_some", "is_none", "is_ok", "is_err",
        "ok", "err", "map", "map_err", "map_or", "map_or_else",
        "and_then", "or_else", "filter",
        # Iterator
        "iter", "into_iter", "iter_mut", "next",
        "fold", "collect", "enumerate", "zip",
        "skip", "take", "chain", "flatten", "flat_map",
        "any", "all", "find", "position", "count", "sum",
        "min", "max", "cloned", "copied", "rev", "peekable",
        "for_each",
        # Vec
        "push", "pop", "len", "is_empty", "contains", "insert",
        "remove", "retain", "clear", "extend", "drain", "truncate",
        "sort", "sort_by", "sort_by_key", "dedup", "split_at",
        # String / &str
        "to_string", "as_str", "to_owned", "to_lowercase", "to_uppercase",
        "trim", "starts_with", "ends_with", "replace",
        "split", "chars", "bytes", "parse",
        "push_str", "as_bytes", "format",
        # HashMap (insert already in Vec)
        "get", "contains_key", "keys", "values",
        "entry", "or_insert", "or_insert_with",
        # Common trait methods
        "clone", "fmt", "eq", "cmp", "partial_cmp", "hash",
        "into", "from", "try_from", "try_into", "as_ref", "as_mut",
        "borrow", "borrow_mut", "deref", "deref_mut",
        "default", "drop",
        # I/O
        "read", "write", "flush", "close", "seek",
        "read_to_string", "write_all",
        # Logging (log crate)
        "debug", "info", "warn", "error", "trace",
    },
    "bash": set(),  # Bash doesn't have method calls
}

# Resolved set for languages that inherit from another
_BUILTIN_METHODS_CACHE: dict[str, set[str]] = {}


def _get_builtin_methods(language: str) -> set[str]:
    """Get builtin methods for a language, resolving inheritance."""
    if language in _BUILTIN_METHODS_CACHE:
        return _BUILTIN_METHODS_CACHE[language]

    methods = BUILTIN_METHODS.get(language)
    if methods is None:
        # Inherit: tsx/typescript -> javascript
        if language in ("typescript", "tsx"):
            methods = BUILTIN_METHODS.get("javascript") or set()
        else:
            methods = set()

    _BUILTIN_METHODS_CACHE[language] = methods
    return methods


# Standard library module prefixes
STDLIB_MODULES: set[str] = {
    "abc", "aifc", "argparse", "array", "ast", "asyncio",
    "base64", "bdb", "binascii", "binhex", "bisect", "builtins",
    "calendar", "cgi", "cgitb", "chunk", "cmath", "cmd", "code",
    "codecs", "codeop", "collections", "colorsys", "compileall",
    "concurrent", "configparser", "contextlib", "contextvars", "copy",
    "copyreg", "cProfile", "crypt", "csv", "ctypes", "curses",
    "dataclasses", "datetime", "dbm", "decimal", "difflib", "dis",
    "distutils", "doctest",
    "email", "encodings", "enum", "errno",
    "faulthandler", "fcntl", "filecmp", "fileinput", "fnmatch",
    "fractions", "ftplib", "functools",
    "gc", "getopt", "getpass", "gettext", "glob", "graphlib", "grp", "gzip",
    "hashlib", "heapq", "hmac", "html", "http",
    "imaplib", "imghdr", "imp", "importlib", "inspect", "io", "ipaddress",
    "itertools",
    "json",
    "keyword",
    "lib2to3", "linecache", "locale", "logging", "lzma",
    "mailbox", "mailcap", "marshal", "math", "mimetypes", "mmap",
    "modulefinder", "multiprocessing",
    "netrc", "nis", "nntplib", "numbers",
    "operator", "optparse", "os",
    "pathlib", "pdb", "pickle", "pickletools", "pipes", "pkgutil",
    "platform", "plistlib", "poplib", "posix", "posixpath", "pprint",
    "profile", "pstats", "pty", "pwd", "py_compile", "pyclbr", "pydoc",
    "queue", "quopri",
    "random", "re", "readline", "reprlib", "resource", "rlcompleter", "runpy",
    "sched", "secrets", "select", "selectors", "shelve", "shlex", "shutil",
    "signal", "site", "smtpd", "smtplib", "sndhdr", "socket", "socketserver",
    "spwd", "sqlite3", "ssl", "stat", "statistics", "string", "stringprep",
    "struct", "subprocess", "sunau", "symtable", "sys", "sysconfig", "syslog",
    "tabnanny", "tarfile", "telnetlib", "tempfile", "termios", "test",
    "textwrap", "threading", "time", "timeit", "tkinter", "token", "tokenize",
    "trace", "traceback", "tracemalloc", "tty", "turtle", "turtledemo",
    "types", "typing",
    "unicodedata", "unittest", "urllib", "uu", "uuid",
    "venv",
    "warnings", "wave", "weakref", "webbrowser", "winreg", "winsound",
    "wsgiref",
    "xdrlib", "xml", "xmlrpc",
    "zipapp", "zipfile", "zipimport", "zlib",
}

# Common third-party library prefixes
EXTERNAL_MODULES: set[str] = {
    # Data science
    "numpy", "np", "pandas", "pd", "scipy", "sklearn", "scikit_learn",
    "matplotlib", "plt", "seaborn", "sns", "plotly",
    # ML/AI
    "torch", "tensorflow", "tf", "keras", "transformers", "openai",
    "anthropic", "litellm", "langchain", "tiktoken",
    # Web frameworks
    "flask", "django", "fastapi", "starlette", "uvicorn", "gunicorn",
    "aiohttp", "httpx", "requests", "urllib3",
    # Databases
    "sqlalchemy", "alembic", "psycopg2", "pymysql", "pymongo", "redis",
    "elasticsearch", "motor", "asyncpg",
    # Testing
    "pytest", "unittest", "mock", "hypothesis", "faker", "factory_boy",
    # CLI
    "click", "typer", "argparse", "fire",
    # Utils
    "pydantic", "attrs", "dataclasses_json", "marshmallow",
    "rich", "tqdm", "colorama", "termcolor",
    "yaml", "pyyaml", "toml", "tomli",
    "dotenv", "python_dotenv",
    "boto3", "botocore", "aws_cdk",
    "celery", "rq", "dramatiq",
    "pillow", "PIL", "cv2", "opencv",
    # Parsing
    "tree_sitter", "lark", "ply", "pyparsing",
    # Logging
    "loguru", "structlog",
    # Async
    "asyncio", "trio", "anyio", "gevent",
    # MCP
    "mcp",
}

# Self-reference patterns by language
SELF_REFERENCES: dict[str, set[str]] = {
    "python": {"self", "cls"},
    "javascript": {"this"},
    "typescript": {"this"},
    "php": {"this", "$this"},
    "rust": {"self", "Self"},
}


def categorize_call(
    call: CallLike,
    language: str,
    param_types: dict[str, str] | None = None,
) -> str:
    """Categorize a call based on its receiver and name.

    Args:
        call: The extracted call to categorize.
        language: Programming language (python, javascript, etc.)
        param_types: Optional dict mapping parameter names to their types.

    Returns:
        CallCategory constant.
    """
    # Already resolved
    if call.resolved_id:
        return CallCategory.RESOLVED

    receiver = call.receiver
    name = call.name

    # Check for built-in functions (bare calls)
    if receiver is None:
        if name in BUILTINS:
            return CallCategory.BUILTIN
        # Could be a local function - leave as unknown for now
        return CallCategory.UNKNOWN

    # Check for self/this references
    self_refs = SELF_REFERENCES.get(language, set())
    if receiver in self_refs:
        # Self-method call that wasn't resolved = missing method
        return CallCategory.UNTYPED

    # Check for dynamic calls
    if receiver in {"getattr", "globals", "locals"}:
        return CallCategory.DYNAMIC

    # Check for stdlib module calls
    if receiver in STDLIB_MODULES or receiver.split(".")[0] in STDLIB_MODULES:
        return CallCategory.STDLIB

    # Check for external library calls
    if receiver in EXTERNAL_MODULES or receiver.split(".")[0] in EXTERNAL_MODULES:
        return CallCategory.EXTERNAL

    # Check if receiver matches a typed parameter — takes priority over
    # builtin methods because typed receivers might have project-specific
    # methods (e.g. repo.get() on Repository class).
    if param_types and receiver in param_types:
        return CallCategory.TYPE_RESOLVABLE

    # Check for language-specific built-in methods (dict.get, list.append, etc.)
    # Only when the receiver is NOT a typed parameter (checked above).
    builtin_methods = _get_builtin_methods(language)
    if name in builtin_methods:
        return CallCategory.BUILTIN

    # Method call on object without type info
    return CallCategory.UNTYPED


def categorize_calls(
    calls: list[CallLike],
    language: str,
    parameters: list[dict[str, Any]] | None = None,
) -> None:
    """Categorize all calls in a list, modifying them in place.

    Args:
        calls: List of calls to categorize (Call or ExtractedCall objects).
        language: Programming language.
        parameters: Optional list of function parameters (as dicts with 'name' and 'type' keys).
    """
    # Build param_types dict from parameters
    param_types: dict[str, str] | None = None
    if parameters:
        param_types = {
            p["name"]: p["type"] for p in parameters if p.get("type")
        }

    for call in calls:
        if call.category == CallCategory.UNKNOWN or call.category == "unknown":
            call.category = categorize_call(call, language, param_types)
