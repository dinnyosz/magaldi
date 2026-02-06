"""Base Elasticsearch repository with connection management.

Contains shared constants and the ElasticsearchBase class that provides
connection management and index creation for all sub-repositories.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from elasticsearch import Elasticsearch

if TYPE_CHECKING:
    from shared.config import MagaldiConfig


def generate_hash_id(element_id: str) -> str:
    """Generate a URL-safe hash ID from an element ID.

    Uses full SHA256 hex digest (64 characters, 256 bits).
    Guaranteed unique - no collision risk.

    Args:
        element_id: Full element ID string.

    Returns:
        64-character hex string suitable for URLs.
    """
    return hashlib.sha256(element_id.encode()).hexdigest()


# Index name for code elements
INDEX_NAME = "magaldi-code-elements"

# Index name for relationships (knowledge graph edges)
RELATIONSHIPS_INDEX_NAME = "magaldi-relationships"

# Index name for external references (env vars, config keys, events, etc.)
EXTERNAL_REFS_INDEX_NAME = "magaldi-external-refs"

# Index mapping with dense_vector for embeddings
INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "element_id": {"type": "keyword"},
            "hash_id": {"type": "keyword"},  # Short URL-safe ID for routing
            "scope": {"type": "keyword"},
            "repository": {"type": "keyword"},
            "username": {"type": "keyword"},
            "relative_path": {"type": "keyword"},
            "element_type": {"type": "keyword"},
            "name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "language": {"type": "keyword"},
            "line_start": {"type": "integer"},
            "line_end": {"type": "integer"},
            "raw_code": {
                "type": "text",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 32766,  # Max for keyword, allows regexp on full code
                    }
                }
            },
            "signature": {"type": "text"},
            "docstring": {"type": "text"},
            "summary": {"type": "text"},
            "level": {"type": "integer"},
            "parent_id": {"type": "keyword"},
            "decorators": {"type": "keyword"},
            "visibility": {"type": "keyword"},
            "is_async": {"type": "boolean"},
            "is_test": {"type": "boolean"},  # Whether element is test code
            "file_hash": {"type": "keyword"},  # For change detection (stored on all elements)
            "content_hash": {"type": "keyword"},  # SHA256 of raw_code for element-level change detection
            "element_count": {"type": "integer"},  # Total elements in file (only on file elements)
            "indexed_at": {"type": "date"},
            "cluster_id": {"type": "keyword"},  # Feature cluster ID (primary, for backward compat)
            "cluster_label": {"type": "keyword"},  # Human-readable cluster label (primary)
            "member_count": {"type": "integer"},  # Number of elements in feature (for feature docs)
            "member_ids": {"type": "keyword"},  # Element IDs of members (for feature docs)
            "parent_feature_label": {"type": "keyword"},  # Parent feature label (for subfeatures)
            "parent_feature_summary": {"type": "text"},  # Parent feature summary (for subfeatures)
            # === SOFT CLUSTERING FIELDS ===
            # Soft feature memberships on code elements (function, method, class)
            "feature_memberships": {
                "type": "nested",
                "properties": {
                    "feature_id": {"type": "keyword"},  # Feature element ID
                    "label": {"type": "keyword"},  # Human-readable label
                    "score": {"type": "float"},  # Membership probability (0-1)
                    "is_primary": {"type": "boolean"},  # True for highest-score feature
                },
            },
            # Soft subfeature memberships on code elements
            "subfeature_memberships": {
                "type": "nested",
                "properties": {
                    "subfeature_id": {"type": "keyword"},  # Subfeature element ID
                    "label": {"type": "keyword"},  # Human-readable label
                    "parent_feature_id": {"type": "keyword"},  # Parent feature ID
                    "parent_feature_label": {"type": "keyword"},  # Parent feature label
                    "score": {"type": "float"},  # Membership probability
                    "is_primary": {"type": "boolean"},  # True for highest-score in this parent
                },
            },
            # Connected features on feature documents (from feature affinity matrix)
            "connected_features": {
                "type": "nested",
                "properties": {
                    "feature_id": {"type": "keyword"},  # Connected feature ID
                    "label": {"type": "keyword"},  # Connected feature label
                    "affinity": {"type": "float"},  # Connection strength (0-1)
                },
            },
            # Cross-feature subfeature connections on feature/subfeature documents
            "connected_subfeatures_cross": {
                "type": "nested",
                "properties": {
                    "subfeature_id": {"type": "keyword"},  # Connected subfeature ID
                    "label": {"type": "keyword"},  # Subfeature label
                    "parent_feature_id": {"type": "keyword"},  # Parent feature of connected subfeature
                    "parent_feature_label": {"type": "keyword"},  # Parent feature label
                    "affinity": {"type": "float"},  # Connection strength
                },
            },
            # Connected subfeatures on subfeature documents (within same + cross-feature)
            "connected_subfeatures": {
                "type": "nested",
                "properties": {
                    "subfeature_id": {"type": "keyword"},
                    "label": {"type": "keyword"},
                    "parent_feature_id": {"type": "keyword"},
                    "parent_feature_label": {"type": "keyword"},
                    "affinity": {"type": "float"},
                },
            },
            "term": {"type": "keyword"},  # Glossary term
            "total_count": {"type": "integer"},  # Glossary term occurrence count
            "file_paths": {"type": "keyword"},  # Array of file paths
            "feature_associations": {"type": "object"},  # Feature linking data
            "updated_at": {"type": "date"},  # Update timestamp
            # Renamed: was "embedding", now explicit for summary-based embedding
            "summary_embedding": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine",
            },
            # NEW: embedding of raw_code for structural similarity
            "code_embedding": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine",
            },
            # On file elements - stores imports
            "imports": {
                "type": "nested",
                "properties": {
                    "name": {"type": "keyword"},        # Imported name
                    "module": {"type": "keyword"},      # Source module
                    "alias": {"type": "keyword"},       # Alias if any
                    "line": {"type": "integer"},        # Line number
                },
            },
            # On function/method elements - stores calls
            "calls": {
                "type": "nested",
                "properties": {
                    "name": {"type": "keyword"},        # Function name
                    "receiver": {"type": "keyword"},    # self, utils, null
                    "line": {"type": "integer"},        # Line number
                    "resolved_id": {"type": "keyword"}, # Resolved element ID (or null)
                },
            },
            # Pre-computed semantic relationships (top-K similar elements)
            "semantic_related": {
                "type": "nested",
                "properties": {
                    "element_id": {"type": "keyword"},
                    "hash_id": {"type": "keyword"},
                    "score": {"type": "float"},
                },
            },
            # Enhanced context fields (extracted during parsing)
            # For classes: instance attributes from __init__
            "class_attributes": {
                "type": "nested",
                "properties": {
                    "name": {"type": "keyword"},
                    "type": {"type": "keyword"},
                    "line": {"type": "integer"},
                },
            },
            # For classes: base class names
            "base_classes": {"type": "keyword"},
            # For functions/methods: exception types raised
            "exceptions_raised": {"type": "keyword"},
            # For methods: attributes modified (self.X assignments)
            "attributes_modified": {"type": "keyword"},
            # Rich decorator info (name, args, full text)
            "decorator_details": {
                "type": "nested",
                "properties": {
                    "name": {"type": "keyword"},
                    "args": {"type": "text"},
                    "full": {"type": "text"},
                },
            },
            # Function/method return type
            "return_type": {"type": "keyword"},
            # Function/method parameters (structured)
            "parameters": {
                "type": "nested",
                "properties": {
                    "name": {"type": "keyword"},
                    "type": {"type": "keyword"},
                    "default": {"type": "keyword"},
                },
            },
            # === EXTENDED CODE INTELLIGENCE MAPPINGS ===
            # Type Flow
            "type_annotations": {
                "type": "nested",
                "properties": {
                    "name": {"type": "keyword"},
                    "kind": {"type": "keyword"},
                    "location": {"type": "keyword"},
                    "line": {"type": "integer"},
                    "generic_args": {"type": "keyword"},
                },
            },
            # Pattern Detection
            "detected_patterns": {"type": "keyword"},
            "pattern_confidence": {"type": "object"},
            # Documentation
            "todos": {
                "type": "nested",
                "properties": {
                    "kind": {"type": "keyword"},
                    "text": {"type": "text"},
                    "line": {"type": "integer"},
                    "assignee": {"type": "keyword"},
                    "priority": {"type": "keyword"},
                    "issue_ref": {"type": "keyword"},
                },
            },
            "section_markers": {
                "type": "nested",
                "properties": {
                    "label": {"type": "keyword"},
                    "line": {"type": "integer"},
                    "style": {"type": "keyword"},
                },
            },
            "associated_comments": {
                "type": "nested",
                "properties": {
                    "text": {"type": "text"},
                    "line": {"type": "integer"},
                    "kind": {"type": "keyword"},
                    "position": {"type": "keyword"},
                },
            },
            # Document structure (for markdown/doc file elements)
            "document_sections": {
                "type": "nested",
                "properties": {
                    "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "level": {"type": "integer"},
                    "line_start": {"type": "integer"},
                    "line_end": {"type": "integer"},
                },
            },
            # API Surface
            "is_public_api": {"type": "boolean"},
            "http_routes": {
                "type": "nested",
                "properties": {
                    "method": {"type": "keyword"},
                    "path": {"type": "keyword"},
                    "path_params": {"type": "keyword"},
                    "framework": {"type": "keyword"},
                },
            },
            "cli_commands": {
                "type": "nested",
                "properties": {
                    "name": {"type": "keyword"},
                    "options": {"type": "nested"},
                    "framework": {"type": "keyword"},
                },
            },
            # Purity/Mutation
            "purity": {
                "type": "object",
                "properties": {
                    "level": {"type": "keyword"},
                    "confidence": {"type": "keyword"},
                    "reasons": {"type": "keyword"},
                },
            },
            "side_effects": {
                "type": "nested",
                "properties": {
                    "kind": {"type": "keyword"},
                    "target": {"type": "keyword"},
                    "line": {"type": "integer"},
                },
            },
            "mutated_state": {"type": "keyword"},
            # === CODE METRICS (Tier 1) ===
            "complexity": {
                "type": "object",
                "properties": {
                    "cyclomatic": {"type": "integer"},
                    "nesting_depth": {"type": "integer"},
                    "branch_count": {"type": "integer"},
                },
            },
            "code_metrics": {
                "type": "object",
                "properties": {
                    "line_count": {"type": "integer"},
                    "param_count": {"type": "integer"},
                    "char_count": {"type": "integer"},
                },
            },
            "docstring_quality": {
                "type": "object",
                "properties": {
                    "has_docstring": {"type": "boolean"},
                    "has_params": {"type": "boolean"},
                    "has_return": {"type": "boolean"},
                    "coverage": {"type": "float"},
                },
            },
            # Security
            "security_issues": {
                "type": "nested",
                "properties": {
                    "kind": {"type": "keyword"},
                    "pattern": {"type": "keyword"},
                    "line": {"type": "integer"},
                    "severity": {"type": "keyword"},
                    "message": {"type": "text"},
                },
            },
            # Environment & Concurrency
            "env_vars": {
                "type": "nested",
                "properties": {
                    "name": {"type": "keyword"},
                    "line": {"type": "integer"},
                    "access_type": {"type": "keyword"},
                },
            },
            "concurrency": {
                "type": "object",
                "properties": {
                    "is_async": {"type": "boolean"},
                    "uses_locks": {"type": "boolean"},
                    "uses_threads": {"type": "boolean"},
                    "patterns": {"type": "keyword"},
                },
            },
            # Roll-up statistics (for file and class elements)
            "metrics_summary": {
                "type": "object",
                "properties": {
                    "security_issue_count": {"type": "integer"},
                    "security_by_severity": {"type": "object"},
                    "max_complexity": {"type": "integer"},
                    "avg_complexity": {"type": "float"},
                    "undocumented_count": {"type": "integer"},
                    "function_count": {"type": "integer"},
                    "async_count": {"type": "integer"},
                    "env_var_count": {"type": "integer"},
                },
            },
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
}

# Mapping for relationships index (knowledge graph edges)
RELATIONSHIPS_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            # Unique identifier for this relationship
            "relationship_id": {"type": "keyword"},
            # Deduplication key (same across users for same logical relationship)
            "relationship_key": {"type": "keyword"},
            # Source and target (using hash_id for elements)
            "source_hash": {"type": "keyword"},
            "target_hash": {"type": "keyword"},
            # Relationship type
            "relationship_type": {"type": "keyword"},
            # Scoping for multi-user support
            "scope": {"type": "keyword"},
            "repository": {"type": "keyword"},
            "username": {"type": "keyword"},
            # Priority for user vs main (user=1 shadows main=0)
            "priority": {"type": "byte"},
            # Metadata
            "confidence": {"type": "float"},
            "line": {"type": "integer"},
            "indexed_at": {"type": "date"},
            # Type-specific details (flexible structure)
            # Examples:
            # - BELONGS_TO_GROUP: {"command_name": "serve", "group_path": ["main", "web"], "full_command": "magaldi web serve"}
            # - MOUNTS_ROUTER: {"prefix": "/api/v1", "tags": ["users"]}
            # - READS_ENV: {"key": "DATABASE_URL", "default": null}
            "details": {"type": "flattened"},
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
}

# Mapping for external references index
EXTERNAL_REFS_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            # Unique identifier (e.g., "env:DATABASE_URL", "event:user.created")
            "ref_id": {"type": "keyword"},
            # Type of reference
            "ref_type": {"type": "keyword"},
            # Human-readable name
            "name": {"type": "keyword"},
            # Searchable name
            "name_text": {"type": "text"},
            # Optional description
            "description": {"type": "text"},
            # Scoping
            "scope": {"type": "keyword"},
            "repository": {"type": "keyword"},
            "username": {"type": "keyword"},
            # Usage count (denormalized for quick lookups)
            "usages_count": {"type": "integer"},
            # Timestamp
            "indexed_at": {"type": "date"},
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
}


class ElasticsearchBase:
    """Base class providing Elasticsearch connection management.

    All sub-repositories receive an instance of this class and use it
    to get the ES client and ensure the index exists.
    """

    def __init__(self, config: MagaldiConfig | None = None):
        from shared.config import get_config
        self._config = config or get_config()
        self._client: Elasticsearch | None = None

    def _get_client(self) -> Elasticsearch:
        """Get or create Elasticsearch client."""
        if self._client is None:
            es_config = self._config.elasticsearch
            self._client = Elasticsearch(
                hosts=[{
                    "host": es_config.host,
                    "port": es_config.port,
                    "scheme": es_config.scheme,
                }],
                timeout=es_config.timeout,
                retry_on_timeout=es_config.retry_on_timeout,
                max_retries=es_config.max_retries,
            )
            # Ensure index exists
            self._ensure_index()
        return self._client

    def _ensure_index(self) -> None:
        """Create indices if they don't exist."""
        client = self._client
        if client:
            # Main elements index
            if not client.indices.exists(index=INDEX_NAME):
                client.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)
            # Relationships index (knowledge graph edges)
            if not client.indices.exists(index=RELATIONSHIPS_INDEX_NAME):
                client.indices.create(
                    index=RELATIONSHIPS_INDEX_NAME, body=RELATIONSHIPS_INDEX_MAPPING
                )
            # External references index (env vars, config keys, events, etc.)
            if not client.indices.exists(index=EXTERNAL_REFS_INDEX_NAME):
                client.indices.create(
                    index=EXTERNAL_REFS_INDEX_NAME, body=EXTERNAL_REFS_INDEX_MAPPING
                )

    def close(self) -> None:
        """Close Elasticsearch client."""
        if self._client:
            self._client.close()
            self._client = None

    @property
    def config(self) -> MagaldiConfig:
        """Get the configuration."""
        return self._config
