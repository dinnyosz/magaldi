"""Code relationship types for the knowledge graph.

This module defines the data structures for representing relationships
between code elements, enabling queries like:
- "What's the full CLI command path?"
- "What routes are under /api/v1?"
- "What code uses this config key?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RelationshipType(str, Enum):
    """Types of relationships between code elements."""

    # =========================================================================
    # Registration & Hierarchy
    # =========================================================================

    # CLI command hierarchies (Click, Typer, argparse)
    BELONGS_TO_GROUP = "belongs_to_group"  # @group.command -> group
    DEFINES_GROUP = "defines_group"  # @main.group defines a CLI group

    # HTTP route hierarchies (FastAPI, Flask, Express)
    MOUNTS_ROUTER = "mounts_router"  # app.include_router(router, prefix="/api")
    REGISTERS_ROUTE = "registers_route"  # @router.get("/users")

    # Event handlers
    HANDLES_EVENT = "handles_event"  # @emitter.on("user.created")
    TRIGGERS_EVENT = "triggers_event"  # emit("user.created")

    # Plugin/middleware registration
    REGISTERS_PLUGIN = "registers_plugin"
    REGISTERS_MIDDLEWARE = "registers_middleware"

    # =========================================================================
    # Inheritance & Types
    # =========================================================================

    INHERITS = "inherits"  # class Foo(Bar)
    IMPLEMENTS = "implements"  # class Foo(Protocol)

    # =========================================================================
    # Composition & Dependencies
    # =========================================================================

    INJECTS = "injects"  # Constructor DI, @inject
    IMPORTS = "imports"  # from x import y
    INSTANTIATES = "instantiates"  # Foo()

    # =========================================================================
    # Data Flow
    # =========================================================================

    READS_CONFIG = "reads_config"  # config.get("KEY")
    READS_ENV = "reads_env"  # os.environ["KEY"]
    QUERIES_MODEL = "queries_model"  # Model.query.filter()
    MUTATES_MODEL = "mutates_model"  # model.save()

    # =========================================================================
    # API Contracts
    # =========================================================================

    ACCEPTS_TYPE = "accepts_type"  # Request body type
    RETURNS_TYPE = "returns_type"  # Response type
    RAISES = "raises"  # Exception types

    # =========================================================================
    # Control Flow
    # =========================================================================

    CALLS = "calls"  # Function calls (already tracked in elements)
    CALLED_BY = "called_by"  # Inverse of calls

    # =========================================================================
    # Testing
    # =========================================================================

    TESTS = "tests"  # test_foo tests foo
    MOCKS = "mocks"  # @patch("module.foo")
    USES_FIXTURE = "uses_fixture"  # pytest fixture dependency


class ExternalRefType(str, Enum):
    """Types of external references (things outside code elements)."""

    ENV_VAR = "env_var"  # Environment variables
    CONFIG_KEY = "config_key"  # Configuration keys
    EVENT = "event"  # Event names
    TABLE = "table"  # Database tables
    URL = "url"  # External URLs
    CLI_ENTRY_POINT = "cli_entry_point"  # Entry point from pyproject.toml


@dataclass
class CodeRelationship:
    """An edge in the code knowledge graph.

    Represents a relationship between two code elements or between
    a code element and an external reference.

    Uses hash_id (SHA256) for element references instead of full element_id
    for shorter, more stable identifiers.
    """

    # Unique identifier
    relationship_id: str

    # Deduplication key (same across users for same logical relationship)
    # Format: scope:repo:source_hash→target_hash:type
    relationship_key: str

    # Source and target (using hash_id, not full element_id)
    source_hash: str  # hash_id of the source element
    target_hash: str  # hash_id of target element, or external ref ID (env:KEY, etc.)

    # Relationship type
    relationship_type: RelationshipType

    # Scoping (for multi-user support)
    scope: str
    repository: str
    username: str
    priority: int = 0  # 0 for main, 1 for user branches

    # Metadata
    confidence: float = 1.0  # 0-1, how certain is this relationship
    line: int | None = None  # Where in source this relationship is defined

    # Type-specific details
    details: dict[str, Any] = field(default_factory=dict)
    # Examples:
    # - BELONGS_TO_GROUP: {"command_name": "serve", "group_path": ["main", "web"]}
    # - MOUNTS_ROUTER: {"prefix": "/api/v1", "tags": ["users"]}
    # - READS_ENV: {"key": "DATABASE_URL", "default": None}


@dataclass
class ExternalReference:
    """A reference to something outside the indexed code.

    Examples: environment variables, config keys, event names, database tables.
    """

    # Unique identifier (e.g., "env:DATABASE_URL", "event:user.created")
    ref_id: str

    # Type of reference
    ref_type: ExternalRefType

    # Human-readable name
    name: str

    # Optional description
    description: str | None = None

    # Scoping
    scope: str = ""
    repository: str = ""
    username: str = "main"

    # Usage count (denormalized for quick lookups)
    usages_count: int = 0


@dataclass
class CommandTreeNode:
    """A node in a CLI command tree."""

    name: str  # Command or group name
    hash_id: str | None  # hash_id if this is a real command/group
    full_path: str  # Full CLI path (e.g., "magaldi web serve")
    is_group: bool  # True if this is a group, False if command
    children: list[CommandTreeNode] = field(default_factory=list)

    # Additional metadata
    description: str | None = None
    options: list[dict[str, Any]] = field(default_factory=list)
    file_path: str | None = None  # Source file path
    line: int | None = None  # Line number


@dataclass
class RouteTreeNode:
    """A node in an HTTP route tree."""

    path_segment: str  # This segment (e.g., "users", "{id}")
    full_path: str  # Full URL path (e.g., "/api/v1/users/{id}")
    hash_id: str | None  # hash_id if this is a route handler
    methods: list[str] = field(default_factory=list)  # ["GET", "POST"]
    children: list[RouteTreeNode] = field(default_factory=list)

    # Additional metadata
    framework: str | None = None
    description: str | None = None
    file_path: str | None = None  # Source file path
    line: int | None = None  # Line number


# =============================================================================
# Helper Functions
# =============================================================================


def generate_relationship_key(
    source_hash: str,
    target_hash: str,
    relationship_type: RelationshipType,
    scope: str,
    repository: str,
) -> str:
    """Generate a unique key for deduplication across users.

    Uses hash_id which is already user-independent (based on file path,
    element type, name, and line - without username).

    Args:
        source_hash: hash_id of source element (64-char SHA256)
        target_hash: hash_id of target element or external ref ID
        relationship_type: Type of relationship
        scope: Repository scope
        repository: Repository name

    Returns:
        Deduplication key in format: scope:repo:source_hash→target_hash:type
    """
    return f"{scope}:{repository}:{source_hash}→{target_hash}:{relationship_type.value}"


def generate_relationship_id(
    source_hash: str,
    target_hash: str,
    relationship_type: RelationshipType,
    scope: str,
    repository: str,
    username: str,
) -> str:
    """Generate a unique relationship ID including username.

    Args:
        source_hash: hash_id of source element
        target_hash: hash_id of target element or external ref ID
        relationship_type: Type of relationship
        scope: Repository scope
        repository: Repository name
        username: Username who owns this relationship

    Returns:
        Unique relationship ID with 12-char hash suffix
    """
    import hashlib

    content = f"{scope}:{repository}:{username}:{source_hash}→{target_hash}:{relationship_type.value}"
    hash_suffix = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"rel:{scope}:{repository}:{username}:{hash_suffix}"


def compute_hash_id(
    scope: str,
    repository: str,
    relative_path: str,
    element_type: str,
    name: str,
    line: int,
) -> str:
    """Compute hash_id for an element (without username).

    This matches the hash_id computation in the indexing pipeline,
    ensuring consistent IDs across the system.
    """
    import hashlib

    # Hash is computed from user-independent parts
    content = f"{scope}:{repository}:{relative_path}:{element_type}:{name}:{line}"
    return hashlib.sha256(content.encode()).hexdigest()


def get_priority_for_username(username: str) -> int:
    """Get priority value for a username (user > main)."""
    return 0 if username == "main" else 1
