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
            "cluster_id": {"type": "keyword"},  # Feature cluster ID
            "cluster_label": {"type": "keyword"},  # Human-readable cluster label
            "member_count": {"type": "integer"},  # Number of elements in feature (for feature docs)
            "member_ids": {"type": "keyword"},  # Element IDs of members (for feature docs)
            "parent_feature_label": {"type": "keyword"},  # Parent feature label (for subfeatures)
            "parent_feature_summary": {"type": "text"},  # Parent feature summary (for subfeatures)
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
            )
            # Ensure index exists
            self._ensure_index()
        return self._client

    def _ensure_index(self) -> None:
        """Create index if it doesn't exist."""
        client = self._client
        if client and not client.indices.exists(index=INDEX_NAME):
            client.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)

    def close(self) -> None:
        """Close Elasticsearch client."""
        if self._client:
            self._client.close()
            self._client = None

    @property
    def config(self) -> MagaldiConfig:
        """Get the configuration."""
        return self._config
