"""Per-language module path resolution for call resolution.

Converts import module paths to possible file paths and detects
external (stdlib/third-party) modules for each supported language.

Each language has its own module format:
- Python: dotted paths (magaldi_core.storage, .utils)
- JavaScript/TypeScript: file paths (./utils, ../lib, react)
- PHP: backslash namespaces (App\\Models\\User)
- Rust: double-colon crate paths (crate::storage::Backend)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import PurePosixPath


def _normalize_path(path: str) -> str:
    """Normalize a POSIX path by collapsing '..' references.

    PurePosixPath doesn't resolve '..' segments, so we do it manually.
    Example: 'src/app/../lib/foo' -> 'src/lib/foo'
    """
    parts: list[str] = []
    for part in path.split("/"):
        if part == "..":
            if parts and parts[-1] != "..":
                parts.pop()
            else:
                parts.append(part)
        elif part != ".":
            parts.append(part)
    return "/".join(parts)


class ModuleResolver(ABC):
    """Base class for language-specific module resolution."""

    @abstractmethod
    def is_external_module(self, module: str) -> bool:
        """Check if module is external (stdlib/third-party).

        Returns True for modules that aren't part of the local codebase.
        """
        ...

    @abstractmethod
    def module_to_file_paths(
        self, module: str, caller_path: str | None = None
    ) -> list[str]:
        """Convert import module path to possible file paths in the repo.

        Args:
            module: Module path as extracted from import statement.
            caller_path: Relative path of the file making the import,
                needed for resolving relative imports.

        Returns:
            List of possible file paths to try.
        """
        ...


# =============================================================================
# PYTHON
# =============================================================================


class PythonModuleResolver(ModuleResolver):
    """Python: dotted paths, relative imports with dots."""

    # Common stdlib modules
    _STDLIB_PREFIXES: frozenset[str] = frozenset({
        "os", "sys", "re", "json", "math", "time", "datetime", "collections",
        "itertools", "functools", "pathlib", "typing", "logging", "unittest",
        "dataclasses", "abc", "asyncio", "hashlib", "base64", "uuid", "copy",
        "io", "tempfile", "shutil", "subprocess", "threading", "multiprocessing",
        "contextlib", "textwrap", "enum", "string", "struct", "secrets",
        "html", "xml", "csv", "configparser", "argparse", "gettext",
        "http", "urllib", "email", "mailbox", "mimetypes",
        "pickle", "shelve", "sqlite3", "socket", "ssl",
        "signal", "select", "selectors", "statistics", "random",
        "glob", "fnmatch", "zipfile", "tarfile", "gzip", "bz2", "lzma",
        "weakref", "types", "traceback", "warnings", "inspect",
        "importlib", "pkgutil", "pprint", "reprlib", "dis",
        "token", "tokenize", "ast", "compileall", "codeop",
        "operator", "decimal", "fractions", "numbers",
        "array", "bisect", "heapq", "queue",
        "concurrent", "sched",
    })

    # Common third-party modules
    _THIRD_PARTY_PREFIXES: frozenset[str] = frozenset({
        "numpy", "pandas", "requests", "flask", "django", "fastapi", "pydantic",
        "pytest", "elasticsearch", "redis", "sqlalchemy", "boto3", "torch",
        "tensorflow", "sklearn", "scipy", "matplotlib", "pillow", "PIL",
        "litellm", "openai", "anthropic", "tiktoken", "httpx", "aiohttp",
        "tree_sitter", "mcp", "click", "typer", "rich", "tqdm",
        "celery", "kombu", "starlette", "uvicorn", "gunicorn",
        "jinja2", "markupsafe", "werkzeug", "itsdangerous",
        "cryptography", "jwt", "passlib", "bcrypt",
        "paramiko", "fabric", "invoke",
        "yaml", "toml", "tomli", "tomllib",
        "attr", "attrs", "cattrs",
        "arrow", "pendulum", "dateutil",
        "loguru", "structlog",
        "sentry_sdk",
    })

    def is_external_module(self, module: str) -> bool:
        """Check if module is external (stdlib or third-party)."""
        # Relative imports are always internal
        if module.startswith("."):
            return False

        first_component = module.split(".")[0]
        return (
            first_component in self._STDLIB_PREFIXES
            or first_component in self._THIRD_PARTY_PREFIXES
        )

    def module_to_file_paths(
        self, module: str, caller_path: str | None = None
    ) -> list[str]:
        """Convert Python dotted module path to possible file paths."""
        paths: list[str] = []

        if module.startswith("."):
            if not caller_path:
                return paths

            # Count leading dots to determine relative depth
            dots = 0
            for ch in module:
                if ch == ".":
                    dots += 1
                else:
                    break

            caller_dir = str(PurePosixPath(caller_path).parent)

            # Go up (dots - 1) directories
            base_dir = caller_dir
            for _ in range(dots - 1):
                parent = str(PurePosixPath(base_dir).parent)
                if parent == base_dir:
                    break
                base_dir = parent

            remaining = module[dots:]

            if remaining:
                sub_path = remaining.replace(".", "/")
                paths.append(f"{base_dir}/{sub_path}.py")
                paths.append(f"{base_dir}/{sub_path}/__init__.py")
            else:
                paths.append(f"{base_dir}/__init__.py")

            return paths

        # Absolute import: convert dots to path separators
        path_base = module.replace(".", "/")

        for src_prefix in ["src/", ""]:
            paths.append(f"{src_prefix}{path_base}.py")
            paths.append(f"{src_prefix}{path_base}/__init__.py")

        return paths


# =============================================================================
# JAVASCRIPT / TYPESCRIPT
# =============================================================================


class JavaScriptModuleResolver(ModuleResolver):
    """JS/TS: file-relative paths, bare specifier detection."""

    # Node.js built-in modules
    _NODE_BUILTINS: frozenset[str] = frozenset({
        "assert", "async_hooks", "buffer", "child_process", "cluster",
        "console", "constants", "crypto", "dgram", "diagnostics_channel",
        "dns", "domain", "events", "fs", "http", "http2", "https",
        "inspector", "module", "net", "os", "path", "perf_hooks",
        "process", "punycode", "querystring", "readline", "repl",
        "stream", "string_decoder", "sys", "timers", "tls", "trace_events",
        "tty", "url", "util", "v8", "vm", "wasi", "worker_threads", "zlib",
    })

    # Common npm packages
    _NPM_PACKAGES: frozenset[str] = frozenset({
        "react", "react-dom", "next", "vue", "svelte", "angular",
        "express", "koa", "fastify", "hapi", "nestjs",
        "lodash", "underscore", "ramda",
        "axios", "node-fetch", "got", "superagent",
        "moment", "dayjs", "date-fns", "luxon",
        "chalk", "commander", "yargs", "inquirer", "ora",
        "winston", "pino", "bunyan", "debug",
        "jest", "mocha", "chai", "sinon", "vitest",
        "webpack", "rollup", "esbuild", "vite", "parcel",
        "typescript", "ts-node", "tsx",
        "eslint", "prettier",
        "mongoose", "sequelize", "typeorm", "prisma", "knex",
        "redis", "ioredis",
        "socket.io", "ws",
        "uuid", "nanoid",
        "zod", "joi", "yup", "class-validator",
        "rxjs", "immer", "zustand", "redux", "mobx",
        "tailwindcss", "styled-components", "emotion",
        "graphql", "apollo", "urql",
        "sharp", "jimp",
        "dotenv",
        "bcrypt", "jsonwebtoken", "passport",
    })

    # File extensions to try when resolving relative imports
    _EXTENSIONS: tuple[str, ...] = (
        ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    )

    def is_external_module(self, module: str) -> bool:
        """External if known builtin/npm package or likely external bare specifier.

        Bare specifiers (no ./ or ../ prefix) are npm packages by convention,
        but we check for common internal patterns first to avoid false positives
        with workspace packages and path aliases.
        """
        # Relative imports are always internal
        if module.startswith(".") or module.startswith("/"):
            return False

        # node: protocol (e.g., "node:fs")
        if module.startswith("node:"):
            return True

        # Strip scoped package prefix to get base name
        # @scope/package -> scope/package
        base = module.lstrip("@").split("/")[0]

        if base in self._NODE_BUILTINS:
            return True
        if base in self._NPM_PACKAGES:
            return True

        # Bare specifiers without ./ are npm packages by convention,
        # but common internal patterns should be treated as internal:
        # - Paths with extensions (e.g., "utils.js")
        # - Tilde paths (e.g., "~/utils")
        # - Hash paths (e.g., "#utils")
        if module.startswith("~") or module.startswith("#"):
            return False
        if any(module.endswith(ext) for ext in self._EXTENSIONS):
            return False

        # Unknown bare specifiers are likely npm packages
        return True

    def module_to_file_paths(
        self, module: str, caller_path: str | None = None
    ) -> list[str]:
        """Convert JS/TS module path to possible file paths."""
        paths: list[str] = []

        # Only resolve relative imports
        if not module.startswith("."):
            return paths

        if not caller_path:
            return paths

        caller_dir = str(PurePosixPath(caller_path).parent)

        # Resolve the relative path
        if module.startswith("./"):
            relative = module[2:]
        elif module.startswith("../"):
            relative = module
        else:
            relative = module[1:]  # bare "." import

        resolved = str(PurePosixPath(caller_dir) / relative)

        # Normalize parent references (PurePosixPath doesn't resolve ..)
        # Manually collapse "foo/../bar" -> "bar"
        resolved = _normalize_path(resolved)

        # Check if path already has an extension
        has_ext = any(resolved.endswith(ext) for ext in self._EXTENSIONS)

        if has_ext:
            # Direct path with extension
            paths.append(resolved)
            # Also try without extension + other extensions
            base = resolved
            for ext in self._EXTENSIONS:
                if resolved.endswith(ext):
                    base = resolved[: -len(ext)]
                    break
            for ext in self._EXTENSIONS:
                candidate = f"{base}{ext}"
                if candidate != resolved:
                    paths.append(candidate)
        else:
            # Try each extension
            for ext in self._EXTENSIONS:
                paths.append(f"{resolved}{ext}")
            # Try index files
            for ext in self._EXTENSIONS:
                paths.append(f"{resolved}/index{ext}")

        return paths


# =============================================================================
# PHP
# =============================================================================


class PHPModuleResolver(ModuleResolver):
    """PHP: backslash namespaces, PSR-4 conventions."""

    # Known third-party/framework namespace prefixes
    _EXTERNAL_PREFIXES: frozenset[str] = frozenset({
        "Illuminate", "Symfony", "Doctrine", "Laravel", "PHPUnit",
        "Psr", "GuzzleHttp", "Monolog", "Carbon", "League", "Ramsey",
        "Composer", "PhpParser", "Predis", "Aws", "Google", "Firebase",
        "Stripe", "Twilio", "SendGrid", "Mailgun",
        "Spatie", "Barryvdh", "Livewire", "Inertia",
        "Sentry", "Bugsnag",
        "PHPStan", "Rector", "Pest",
    })

    def is_external_module(self, module: str) -> bool:
        """External if namespace matches known third-party prefixes."""
        # Get first namespace component
        first_component = module.lstrip("\\").split("\\")[0]
        return first_component in self._EXTERNAL_PREFIXES

    def module_to_file_paths(
        self, module: str, caller_path: str | None = None  # noqa: ARG002
    ) -> list[str]:
        """Convert PHP namespace to possible file paths via PSR-4 convention."""
        paths: list[str] = []

        # Strip leading backslash
        clean = module.lstrip("\\")

        # Convert namespace separators to directory separators
        # App\Models\User -> App/Models/User
        file_path = clean.replace("\\", "/")

        # Try common root directories with PSR-4 conventions
        # PSR-4: namespace prefix maps to a base directory
        # Common patterns:
        #   App\* -> app/* or src/*
        #   Tests\* -> tests/*
        parts = file_path.split("/")

        if parts[0] in ("App", "Application"):
            # Laravel/common convention: App\Models\User -> app/Models/User.php
            remainder = "/".join(parts[1:])
            paths.append(f"app/{remainder}.php")
            paths.append(f"src/{remainder}.php")
        elif parts[0] in ("Tests", "Test"):
            remainder = "/".join(parts[1:])
            paths.append(f"tests/{remainder}.php")
        else:
            # Try with full namespace path
            paths.append(f"src/{file_path}.php")
            paths.append(f"app/{file_path}.php")
            paths.append(f"lib/{file_path}.php")

        # Always try the full path as-is
        paths.append(f"{file_path}.php")

        return paths


# =============================================================================
# RUST
# =============================================================================


class RustModuleResolver(ModuleResolver):
    """Rust: double-colon crate paths, std detection."""

    # Rust stdlib crates
    _STDLIB_CRATES: frozenset[str] = frozenset({
        "std", "core", "alloc", "test", "proc_macro",
    })

    # Known third-party crates
    _EXTERNAL_CRATES: frozenset[str] = frozenset({
        "serde", "serde_json", "serde_yaml",
        "tokio", "async_std", "futures", "async_trait",
        "clap", "structopt",
        "anyhow", "thiserror", "eyre",
        "tracing", "tracing_subscriber", "log", "env_logger",
        "regex",
        "reqwest", "hyper", "warp", "actix", "actix_web", "axum", "rocket",
        "diesel", "sqlx", "sea_orm", "rusqlite",
        "redis",
        "rand", "chrono", "uuid",
        "lazy_static", "once_cell",
        "itertools", "rayon",
        "sled", "rocksdb",
        "bytes", "crossbeam",
        "num", "num_traits",
        "tempfile",
        "pretty_assertions", "proptest", "criterion",
    })

    def is_external_module(self, module: str) -> bool:
        """External if stdlib crate, known third-party, or not a local path."""
        # Get first path component
        first = module.split("::")[0]

        # Stdlib
        if first in self._STDLIB_CRATES:
            return True

        # Known external crates
        if first in self._EXTERNAL_CRATES:
            return True

        # Local paths: crate::, super::, self:: are internal;
        # unknown bare crate names are assumed external
        return first not in ("crate", "super", "self")

    def module_to_file_paths(
        self, module: str, caller_path: str | None = None
    ) -> list[str]:
        """Convert Rust crate path to possible file paths."""
        paths: list[str] = []

        parts = module.split("::")

        if not parts:
            return paths

        if parts[0] == "crate":
            # crate::storage::Backend -> src/storage.rs or src/storage/mod.rs
            # Drop "crate" prefix, drop the final element (it's the item name)
            module_parts = parts[1:]

            if not module_parts:
                return paths

            # The last part might be a type/function name, not a module
            # Try both with and without the last part as a module
            for end in range(len(module_parts), 0, -1):
                sub = "/".join(module_parts[:end])
                paths.append(f"src/{sub}.rs")
                paths.append(f"src/{sub}/mod.rs")

        elif parts[0] == "super":
            # super::utils -> go up one module from caller
            if not caller_path:
                return paths

            caller_dir = str(PurePosixPath(caller_path).parent)
            parent_dir = str(PurePosixPath(caller_dir).parent)

            remaining_parts = parts[1:]
            if remaining_parts:
                for end in range(len(remaining_parts), 0, -1):
                    sub = "/".join(remaining_parts[:end])
                    paths.append(f"{parent_dir}/{sub}.rs")
                    paths.append(f"{parent_dir}/{sub}/mod.rs")
            else:
                paths.append(f"{parent_dir}/mod.rs")

        elif parts[0] == "self":
            # self::item -> same module directory
            if not caller_path:
                return paths

            caller_dir = str(PurePosixPath(caller_path).parent)

            remaining_parts = parts[1:]
            if remaining_parts:
                for end in range(len(remaining_parts), 0, -1):
                    sub = "/".join(remaining_parts[:end])
                    paths.append(f"{caller_dir}/{sub}.rs")
                    paths.append(f"{caller_dir}/{sub}/mod.rs")

        return paths


# =============================================================================
# REGISTRY
# =============================================================================


_PYTHON_RESOLVER = PythonModuleResolver()
_JAVASCRIPT_RESOLVER = JavaScriptModuleResolver()
_PHP_RESOLVER = PHPModuleResolver()
_RUST_RESOLVER = RustModuleResolver()

_RESOLVERS: dict[str, ModuleResolver] = {
    "python": _PYTHON_RESOLVER,
    "javascript": _JAVASCRIPT_RESOLVER,
    "typescript": _JAVASCRIPT_RESOLVER,
    "tsx": _JAVASCRIPT_RESOLVER,
    "php": _PHP_RESOLVER,
    "rust": _RUST_RESOLVER,
}


def get_module_resolver(language: str) -> ModuleResolver | None:
    """Get the module resolver for a language.

    Args:
        language: Programming language name.

    Returns:
        ModuleResolver instance, or None for unsupported languages.
    """
    return _RESOLVERS.get(language)
