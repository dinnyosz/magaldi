# Glossary Feature Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract domain concepts from element names and link them to features for enhanced codebase discovery.

**Architecture:** A two-phase worker job (extract → link) that runs after parsing, storing glossary terms in Elasticsearch with feature associations. Three new MCP tools expose glossary data, and two existing tools are enhanced.

**Tech Stack:** Python, Elasticsearch, pytest, Click (CLI)

---

## Task 1: Term Extractor Module

**Files:**
- Create: `src/shared/ai/glossary/__init__.py`
- Create: `src/shared/ai/glossary/extractor.py`
- Create: `tests/test_glossary_extractor.py`

### Step 1: Write failing tests for term splitting

```python
# tests/test_glossary_extractor.py
"""Tests for glossary term extraction."""

from __future__ import annotations

import pytest

from shared.ai.glossary.extractor import extract_terms, split_name, COMMON_TERMS


class TestSplitName:
    """Tests for name splitting logic."""

    def test_splits_camel_case(self):
        """Test CamelCase splitting."""
        assert split_name("UserService") == ["user", "service"]

    def test_splits_pascal_case(self):
        """Test PascalCase splitting."""
        assert split_name("GetUserById") == ["get", "user", "by", "id"]

    def test_splits_snake_case(self):
        """Test snake_case splitting."""
        assert split_name("user_service") == ["user", "service"]

    def test_splits_mixed_case(self):
        """Test mixed CamelCase and snake_case."""
        assert split_name("get_UserById") == ["get", "user", "by", "id"]

    def test_handles_acronyms(self):
        """Test handling of acronyms like HTTP, API."""
        assert split_name("HTTPClient") == ["http", "client"]
        assert split_name("parseAPIResponse") == ["parse", "api", "response"]

    def test_handles_single_word(self):
        """Test single word names."""
        assert split_name("user") == ["user"]
        assert split_name("User") == ["user"]

    def test_handles_empty_string(self):
        """Test empty string."""
        assert split_name("") == []


class TestExtractTerms:
    """Tests for term extraction with filtering."""

    def test_filters_common_terms(self):
        """Test that common programming terms are filtered out."""
        # "get" and "service" are common, "user" is domain-specific
        result = extract_terms("getUserService")
        assert "user" in result
        assert "get" not in result
        assert "service" not in result

    def test_filters_single_char_terms(self):
        """Test that single character terms are filtered."""
        result = extract_terms("getA")
        assert "a" not in result

    def test_returns_unique_terms(self):
        """Test that duplicate terms are deduplicated."""
        result = extract_terms("userUserData")
        assert result.count("user") == 1

    def test_extracts_domain_terms(self):
        """Test extraction of domain-specific terms."""
        result = extract_terms("EmailValidationService")
        assert "email" in result
        assert "validation" in result
        assert "service" not in result

    def test_preserves_term_order(self):
        """Test that terms maintain appearance order."""
        result = extract_terms("registration_email_sender")
        assert result == ["registration", "email", "sender"]


class TestCommonTerms:
    """Tests for the common terms set."""

    def test_common_terms_includes_verbs(self):
        """Test that common verbs are in the filter set."""
        verbs = ["get", "set", "add", "remove", "delete", "update", "create"]
        for verb in verbs:
            assert verb in COMMON_TERMS

    def test_common_terms_includes_suffixes(self):
        """Test that architectural suffixes are in the filter set."""
        suffixes = ["service", "controller", "handler", "manager", "factory"]
        for suffix in suffixes:
            assert suffix in COMMON_TERMS
```

### Step 2: Run tests to verify they fail

Run: `pytest tests/test_glossary_extractor.py -v`
Expected: FAIL with "No module named 'shared.ai.glossary'"

### Step 3: Create glossary module with extractor

```python
# src/shared/ai/glossary/__init__.py
"""Glossary extraction and linking for domain concept discovery."""
```

```python
# src/shared/ai/glossary/extractor.py
"""Term extraction from code element names."""

from __future__ import annotations

import re


COMMON_TERMS: set[str] = {
    # Verbs
    "get", "set", "add", "remove", "delete", "update", "create",
    "find", "fetch", "load", "save", "init", "handle", "process",
    "validate", "check", "is", "has", "can", "should", "do", "run",
    "start", "stop", "on", "before", "after", "pre", "post",

    # Architectural suffixes
    "service", "controller", "handler", "manager", "factory",
    "repository", "provider", "helper", "util", "utils",
    "impl", "interface", "abstract", "base", "default",
    "client", "server", "worker", "job", "task",

    # Common patterns
    "by", "for", "with", "from", "to", "and", "or", "the", "all",
    "id", "ids", "name", "type", "data", "info", "item", "items",
    "list", "array", "map", "dict", "set", "config", "options", "params",
    "request", "response", "result", "error", "exception", "status",
    "test", "spec", "mock", "stub", "fake",

    # Type-related
    "str", "string", "int", "integer", "bool", "boolean", "float",
    "none", "null", "void", "any", "object", "class", "func", "function",
    "method", "attr", "attribute", "prop", "property", "field", "key", "value",

    # Common single words
    "new", "old", "tmp", "temp", "async", "sync", "callback", "promise",
}


def split_name(name: str) -> list[str]:
    """Split an element name into component terms.

    Handles CamelCase, PascalCase, snake_case, and mixed formats.

    Args:
        name: Element name to split.

    Returns:
        List of lowercase terms.
    """
    if not name:
        return []

    # First, split on underscores and hyphens
    parts = re.split(r"[_\-]", name)

    terms: list[str] = []
    for part in parts:
        if not part:
            continue
        # Split CamelCase: insert space before uppercase letters
        # Handle sequences of uppercase (acronyms) by keeping them together
        camel_split = re.sub(r"([a-z])([A-Z])", r"\1 \2", part)
        camel_split = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", camel_split)

        for term in camel_split.split():
            lower_term = term.lower()
            if lower_term:
                terms.append(lower_term)

    return terms


def extract_terms(name: str, min_length: int = 2) -> list[str]:
    """Extract domain-specific terms from an element name.

    Splits the name, filters common programming terms, and returns
    unique domain-specific terms in order of appearance.

    Args:
        name: Element name to extract terms from.
        min_length: Minimum term length (default: 2).

    Returns:
        List of unique domain terms in order of appearance.
    """
    raw_terms = split_name(name)

    seen: set[str] = set()
    result: list[str] = []

    for term in raw_terms:
        # Skip if too short
        if len(term) < min_length:
            continue
        # Skip if common term
        if term in COMMON_TERMS:
            continue
        # Skip if already seen
        if term in seen:
            continue

        seen.add(term)
        result.append(term)

    return result
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/test_glossary_extractor.py -v`
Expected: PASS (all tests)

### Step 5: Commit

```bash
git add src/shared/ai/glossary/ tests/test_glossary_extractor.py
git commit -m "feat(glossary): add term extraction module"
```

---

## Task 2: Glossary Aggregator

**Files:**
- Modify: `src/shared/ai/glossary/extractor.py`
- Modify: `tests/test_glossary_extractor.py`

### Step 1: Write failing tests for aggregation

```python
# Add to tests/test_glossary_extractor.py

from shared.ai.glossary.extractor import aggregate_glossary_terms, GlossaryEntry


class TestAggregateGlossaryTerms:
    """Tests for glossary term aggregation."""

    def test_aggregates_from_multiple_elements(self):
        """Test aggregation across multiple elements."""
        elements = [
            {"element_id": "id1", "name": "UserService", "relative_path": "user.py"},
            {"element_id": "id2", "name": "UserController", "relative_path": "user.py"},
            {"element_id": "id3", "name": "EmailSender", "relative_path": "email.py"},
        ]

        result = aggregate_glossary_terms(elements)

        assert "user" in result
        assert result["user"].total_count == 2
        assert set(result["user"].element_ids) == {"id1", "id2"}
        assert result["user"].file_paths == ["user.py"]

    def test_tracks_file_paths(self):
        """Test that file paths are tracked correctly."""
        elements = [
            {"element_id": "id1", "name": "UserService", "relative_path": "services/user.py"},
            {"element_id": "id2", "name": "UserModel", "relative_path": "models/user.py"},
        ]

        result = aggregate_glossary_terms(elements)

        assert "user" in result
        assert set(result["user"].file_paths) == {"services/user.py", "models/user.py"}

    def test_returns_empty_for_no_domain_terms(self):
        """Test that elements with only common terms return empty."""
        elements = [
            {"element_id": "id1", "name": "GetService", "relative_path": "a.py"},
        ]

        result = aggregate_glossary_terms(elements)

        assert len(result) == 0

    def test_glossary_entry_dataclass(self):
        """Test GlossaryEntry dataclass structure."""
        entry = GlossaryEntry(
            term="user",
            total_count=5,
            element_ids=["id1", "id2"],
            file_paths=["a.py", "b.py"],
        )

        assert entry.term == "user"
        assert entry.total_count == 5
        assert len(entry.element_ids) == 2
```

### Step 2: Run tests to verify they fail

Run: `pytest tests/test_glossary_extractor.py::TestAggregateGlossaryTerms -v`
Expected: FAIL with "cannot import name 'aggregate_glossary_terms'"

### Step 3: Implement aggregation

```python
# Add to src/shared/ai/glossary/extractor.py

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GlossaryEntry:
    """A glossary term with its occurrences."""

    term: str
    total_count: int = 0
    element_ids: list[str] = field(default_factory=list)
    file_paths: list[str] = field(default_factory=list)


def aggregate_glossary_terms(
    elements: list[dict[str, Any]],
) -> dict[str, GlossaryEntry]:
    """Aggregate glossary terms from a list of elements.

    Args:
        elements: List of element dicts with 'element_id', 'name', 'relative_path'.

    Returns:
        Dict mapping term to GlossaryEntry with aggregated data.
    """
    entries: dict[str, GlossaryEntry] = {}

    for element in elements:
        element_id = element.get("element_id", "")
        name = element.get("name", "")
        file_path = element.get("relative_path", "")

        terms = extract_terms(name)

        for term in terms:
            if term not in entries:
                entries[term] = GlossaryEntry(term=term)

            entry = entries[term]
            entry.total_count += 1
            entry.element_ids.append(element_id)

            if file_path and file_path not in entry.file_paths:
                entry.file_paths.append(file_path)

    return entries
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/test_glossary_extractor.py -v`
Expected: PASS (all tests)

### Step 5: Commit

```bash
git add src/shared/ai/glossary/extractor.py tests/test_glossary_extractor.py
git commit -m "feat(glossary): add term aggregation"
```

---

## Task 3: Elasticsearch Glossary Storage

**Files:**
- Modify: `src/shared/db/elasticsearch.py`
- Create: `tests/test_glossary_elasticsearch.py`

### Step 1: Write failing tests for glossary storage

```python
# tests/test_glossary_elasticsearch.py
"""Tests for glossary Elasticsearch operations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from datetime import datetime

import pytest


class TestIndexGlossary:
    """Tests for index_glossary method."""

    def test_indexes_glossary_entry(self):
        """Test indexing a glossary entry."""
        from shared.db.elasticsearch import ElasticsearchRepository

        mock_client = MagicMock()

        with patch.object(ElasticsearchRepository, "_get_client", return_value=mock_client):
            repo = ElasticsearchRepository.__new__(ElasticsearchRepository)
            repo._client = None
            repo._get_client = lambda: mock_client

            result = repo.index_glossary(
                glossary_id="scope:repo:main:glossary:user",
                scope="scope",
                repository="repo",
                username="main",
                term="user",
                total_count=5,
                element_ids=["id1", "id2"],
                file_paths=["a.py", "b.py"],
            )

            assert result is True
            mock_client.index.assert_called_once()
            call_kwargs = mock_client.index.call_args[1]
            doc = call_kwargs["document"]

            assert doc["term"] == "user"
            assert doc["element_type"] == "glossary"
            assert doc["total_count"] == 5
            assert doc["element_ids"] == ["id1", "id2"]
            assert doc["file_paths"] == ["a.py", "b.py"]


class TestGetGlossaryTerms:
    """Tests for get_glossary_terms method."""

    def test_returns_all_glossary_terms(self):
        """Test getting all glossary terms for a repo."""
        from shared.db.elasticsearch import ElasticsearchRepository

        mock_client = MagicMock()
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "scope:repo:main:glossary:user",
                        "_source": {
                            "term": "user",
                            "total_count": 10,
                            "element_ids": ["id1"],
                            "file_paths": ["user.py"],
                        },
                    },
                    {
                        "_id": "scope:repo:main:glossary:email",
                        "_source": {
                            "term": "email",
                            "total_count": 5,
                            "element_ids": ["id2"],
                            "file_paths": ["email.py"],
                        },
                    },
                ]
            }
        }

        with patch.object(ElasticsearchRepository, "_get_client", return_value=mock_client):
            repo = ElasticsearchRepository.__new__(ElasticsearchRepository)
            repo._client = None
            repo._get_client = lambda: mock_client

            result = repo.get_glossary_terms(
                scope="scope",
                repository="repo",
                username="main",
            )

            assert len(result) == 2
            assert result[0]["term"] == "user"
            assert result[1]["term"] == "email"


class TestUpdateGlossaryFeatureAssociations:
    """Tests for update_glossary_feature_associations method."""

    def test_updates_feature_associations(self):
        """Test updating feature associations on glossary entry."""
        from shared.db.elasticsearch import ElasticsearchRepository

        mock_client = MagicMock()

        with patch.object(ElasticsearchRepository, "_get_client", return_value=mock_client):
            repo = ElasticsearchRepository.__new__(ElasticsearchRepository)
            repo._client = None
            repo._get_client = lambda: mock_client

            associations = [
                {
                    "feature_id": "feat1",
                    "feature_label": "user_auth",
                    "frequency": 3,
                    "total_members": 5,
                    "percentage": 60.0,
                }
            ]

            result = repo.update_glossary_feature_associations(
                glossary_id="scope:repo:main:glossary:user",
                feature_associations=associations,
            )

            assert result is True
            mock_client.update.assert_called_once()
```

### Step 2: Run tests to verify they fail

Run: `pytest tests/test_glossary_elasticsearch.py -v`
Expected: FAIL with "has no attribute 'index_glossary'"

### Step 3: Implement Elasticsearch methods

Add to `src/shared/db/elasticsearch.py` in the `ElasticsearchRepository` class (after `index_subfeature` method, around line 1138):

```python
    def index_glossary(
        self,
        glossary_id: str,
        scope: str,
        repository: str,
        username: str,
        term: str,
        total_count: int,
        element_ids: list[str],
        file_paths: list[str],
    ) -> bool:
        """Index a glossary entry document.

        Args:
            glossary_id: Unique glossary ID (scope:repo:user:glossary:term).
            scope: Repository scope.
            repository: Repository name.
            username: Username/branch.
            term: The glossary term.
            total_count: Total occurrences across elements.
            element_ids: List of element IDs containing this term.
            file_paths: List of file paths where term appears.

        Returns:
            True on success.
        """
        from datetime import datetime

        doc: dict[str, Any] = {
            "element_id": glossary_id,
            "hash_id": generate_hash_id(glossary_id),
            "scope": scope,
            "repository": repository,
            "username": username,
            "element_type": "glossary",
            "name": term,
            "term": term,
            "total_count": total_count,
            "element_ids": element_ids,
            "file_paths": file_paths,
            "feature_associations": [],
            "indexed_at": datetime.now().isoformat(),
            "level": -3,  # Glossary is conceptual, above features
        }

        client = self._get_client()
        client.index(index=INDEX_NAME, id=glossary_id, document=doc)
        return True

    def get_glossary_terms(
        self,
        scope: str,
        repository: str,
        username: str = "main",
        min_count: int = 1,
    ) -> list[dict[str, Any]]:
        """Get all glossary terms for a repository.

        Args:
            scope: Scope to filter by.
            repository: Repository to filter by.
            username: Username/branch (default: main).
            min_count: Minimum occurrence count to include (default: 1).

        Returns:
            List of glossary entry dicts sorted by total_count descending.
        """
        client = self._get_client()

        query: dict[str, Any] = {
            "bool": {
                "must": [
                    {"term": {"scope": scope}},
                    {"term": {"repository": repository}},
                    {"term": {"username": username}},
                    {"term": {"element_type": "glossary"}},
                    {"range": {"total_count": {"gte": min_count}}},
                ]
            }
        }

        response = client.search(
            index=INDEX_NAME,
            query=query,
            size=1000,
            sort=[{"total_count": "desc"}],
        )

        results: list[dict[str, Any]] = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            results.append({
                "glossary_id": hit.get("_id"),
                "term": source.get("term"),
                "total_count": source.get("total_count"),
                "element_ids": source.get("element_ids", []),
                "file_paths": source.get("file_paths", []),
                "feature_associations": source.get("feature_associations", []),
            })

        return results

    def get_glossary_term(
        self,
        scope: str,
        repository: str,
        term: str,
        username: str = "main",
    ) -> dict[str, Any] | None:
        """Get a specific glossary term.

        Args:
            scope: Scope to filter by.
            repository: Repository to filter by.
            term: The glossary term to retrieve.
            username: Username/branch (default: main).

        Returns:
            Glossary entry dict or None if not found.
        """
        glossary_id = f"{scope}:{repository}:{username}:glossary:{term}"
        client = self._get_client()

        try:
            response = client.get(index=INDEX_NAME, id=glossary_id)
            if response.get("found"):
                source = response.get("_source", {})
                return {
                    "glossary_id": glossary_id,
                    "term": source.get("term"),
                    "total_count": source.get("total_count"),
                    "element_ids": source.get("element_ids", []),
                    "file_paths": source.get("file_paths", []),
                    "feature_associations": source.get("feature_associations", []),
                }
        except Exception:
            pass

        return None

    def search_glossary(
        self,
        scope: str,
        repository: str,
        query: str,
        username: str = "main",
    ) -> list[dict[str, Any]]:
        """Search glossary terms by partial match.

        Args:
            scope: Scope to filter by.
            repository: Repository to filter by.
            query: Partial term to search for.
            username: Username/branch (default: main).

        Returns:
            List of matching glossary entries.
        """
        client = self._get_client()

        es_query: dict[str, Any] = {
            "bool": {
                "must": [
                    {"term": {"scope": scope}},
                    {"term": {"repository": repository}},
                    {"term": {"username": username}},
                    {"term": {"element_type": "glossary"}},
                    {"wildcard": {"term": f"*{query.lower()}*"}},
                ]
            }
        }

        response = client.search(
            index=INDEX_NAME,
            query=es_query,
            size=100,
            sort=[{"total_count": "desc"}],
        )

        results: list[dict[str, Any]] = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            results.append({
                "glossary_id": hit.get("_id"),
                "term": source.get("term"),
                "total_count": source.get("total_count"),
            })

        return results

    def update_glossary_feature_associations(
        self,
        glossary_id: str,
        feature_associations: list[dict[str, Any]],
    ) -> bool:
        """Update feature associations for a glossary entry.

        Args:
            glossary_id: Glossary entry ID.
            feature_associations: List of feature association dicts.

        Returns:
            True on success.
        """
        client = self._get_client()

        client.update(
            index=INDEX_NAME,
            id=glossary_id,
            doc={
                "feature_associations": feature_associations,
                "updated_at": datetime.now().isoformat(),
            },
        )

        return True

    def delete_glossary(
        self,
        scope: str,
        repository: str,
        username: str,
    ) -> int:
        """Delete all glossary entries for a repository.

        Args:
            scope: Scope to filter by.
            repository: Repository to filter by.
            username: Username/branch.

        Returns:
            Number of deleted documents.
        """
        client = self._get_client()

        response = client.delete_by_query(
            index=INDEX_NAME,
            query={
                "bool": {
                    "must": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                        {"term": {"element_type": "glossary"}},
                    ]
                }
            },
        )

        return response.get("deleted", 0)
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/test_glossary_elasticsearch.py -v`
Expected: PASS (all tests)

### Step 5: Commit

```bash
git add src/shared/db/elasticsearch.py tests/test_glossary_elasticsearch.py
git commit -m "feat(glossary): add Elasticsearch storage methods"
```

---

## Task 4: Glossary Linker Module

**Files:**
- Create: `src/shared/ai/glossary/linker.py`
- Create: `tests/test_glossary_linker.py`

### Step 1: Write failing tests for linker

```python
# tests/test_glossary_linker.py
"""Tests for glossary-feature linking."""

from __future__ import annotations

import pytest

from shared.ai.glossary.linker import (
    compute_feature_associations,
    FeatureAssociation,
)


class TestComputeFeatureAssociations:
    """Tests for computing feature-glossary associations."""

    def test_computes_frequency_and_percentage(self):
        """Test that frequency and percentage are computed correctly."""
        feature = {
            "feature_id": "feat1",
            "label": "user_auth",
            "member_ids": ["id1", "id2", "id3", "id4", "id5"],
        }

        # Mock element names - 3 out of 5 contain "user"
        element_names = {
            "id1": "UserService",
            "id2": "UserController",
            "id3": "AuthManager",
            "id4": "UserValidator",
            "id5": "SessionHandler",
        }

        glossary_terms = {"user", "auth", "session", "validator"}

        result = compute_feature_associations(feature, element_names, glossary_terms)

        user_assoc = next((a for a in result if a.term == "user"), None)
        assert user_assoc is not None
        assert user_assoc.frequency == 3
        assert user_assoc.total_members == 5
        assert user_assoc.percentage == 60.0

    def test_only_includes_existing_glossary_terms(self):
        """Test that only terms in the glossary are included."""
        feature = {
            "feature_id": "feat1",
            "label": "test",
            "member_ids": ["id1"],
        }

        element_names = {"id1": "FooBarBaz"}
        glossary_terms = {"foo"}  # Only "foo" is in glossary

        result = compute_feature_associations(feature, element_names, glossary_terms)

        terms = [a.term for a in result]
        assert "foo" in terms
        assert "bar" not in terms
        assert "baz" not in terms

    def test_handles_empty_feature(self):
        """Test handling of feature with no members."""
        feature = {
            "feature_id": "feat1",
            "label": "empty",
            "member_ids": [],
        }

        result = compute_feature_associations(feature, {}, {"user"})

        assert result == []

    def test_returns_sorted_by_frequency(self):
        """Test that results are sorted by frequency descending."""
        feature = {
            "feature_id": "feat1",
            "label": "test",
            "member_ids": ["id1", "id2", "id3"],
        }

        element_names = {
            "id1": "EmailUserNotification",
            "id2": "EmailSender",
            "id3": "UserProfile",
        }

        glossary_terms = {"email", "user", "notification", "sender", "profile"}

        result = compute_feature_associations(feature, element_names, glossary_terms)

        # email appears in 2 elements, user in 2, others in 1
        frequencies = [a.frequency for a in result]
        assert frequencies == sorted(frequencies, reverse=True)
```

### Step 2: Run tests to verify they fail

Run: `pytest tests/test_glossary_linker.py -v`
Expected: FAIL with "No module named 'shared.ai.glossary.linker'"

### Step 3: Implement linker

```python
# src/shared/ai/glossary/linker.py
"""Feature-glossary linking logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shared.ai.glossary.extractor import extract_terms


@dataclass
class FeatureAssociation:
    """Association between a glossary term and a feature."""

    term: str
    feature_id: str
    feature_label: str
    frequency: int
    total_members: int
    percentage: float


def compute_feature_associations(
    feature: dict[str, Any],
    element_names: dict[str, str],
    glossary_terms: set[str],
) -> list[FeatureAssociation]:
    """Compute glossary associations for a feature.

    Args:
        feature: Feature dict with 'feature_id', 'label', 'member_ids'.
        element_names: Dict mapping element_id to element name.
        glossary_terms: Set of valid glossary terms.

    Returns:
        List of FeatureAssociation sorted by frequency descending.
    """
    feature_id = feature.get("feature_id", "")
    feature_label = feature.get("label", "")
    member_ids = feature.get("member_ids", [])

    if not member_ids:
        return []

    total_members = len(member_ids)

    # Count occurrences of each glossary term across members
    term_counts: dict[str, int] = {}

    for member_id in member_ids:
        name = element_names.get(member_id, "")
        terms = extract_terms(name)

        for term in terms:
            # Only count if term is in the glossary
            if term in glossary_terms:
                term_counts[term] = term_counts.get(term, 0) + 1

    # Build associations
    associations: list[FeatureAssociation] = []

    for term, frequency in term_counts.items():
        percentage = (frequency / total_members) * 100

        associations.append(FeatureAssociation(
            term=term,
            feature_id=feature_id,
            feature_label=feature_label,
            frequency=frequency,
            total_members=total_members,
            percentage=round(percentage, 1),
        ))

    # Sort by frequency descending
    associations.sort(key=lambda a: a.frequency, reverse=True)

    return associations


def link_glossary_to_features(
    glossary_terms: set[str],
    features: list[dict[str, Any]],
    element_names: dict[str, str],
) -> dict[str, list[FeatureAssociation]]:
    """Link glossary terms to features.

    Args:
        glossary_terms: Set of glossary terms.
        features: List of feature dicts.
        element_names: Dict mapping element_id to element name.

    Returns:
        Dict mapping glossary term to list of feature associations.
    """
    term_to_features: dict[str, list[FeatureAssociation]] = {
        term: [] for term in glossary_terms
    }

    for feature in features:
        associations = compute_feature_associations(
            feature, element_names, glossary_terms
        )

        for assoc in associations:
            term_to_features[assoc.term].append(assoc)

    return term_to_features
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/test_glossary_linker.py -v`
Expected: PASS (all tests)

### Step 5: Commit

```bash
git add src/shared/ai/glossary/linker.py tests/test_glossary_linker.py
git commit -m "feat(glossary): add feature linking logic"
```

---

## Task 5: Glossary CLI Command

**Files:**
- Modify: `src/shared/cli.py`
- Modify: `tests/test_cli.py`

### Step 1: Write failing test for CLI command

```python
# Add to tests/test_cli.py

from click.testing import CliRunner
from shared.cli import main


class TestExtractGlossaryCommand:
    """Tests for extract-glossary CLI command."""

    def test_extract_glossary_command_exists(self):
        """Test that extract-glossary command is registered."""
        runner = CliRunner()
        result = runner.invoke(main, ["extract-glossary", "--help"])

        assert result.exit_code == 0
        assert "Extract glossary" in result.output
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_cli.py::TestExtractGlossaryCommand -v`
Expected: FAIL with "No such command 'extract-glossary'"

### Step 3: Implement CLI command

Add to `src/shared/cli.py` (after the extract-features command, around line 329):

```python
# =============================================================================
# EXTRACT-GLOSSARY COMMAND
# =============================================================================


@main.command("extract-glossary")
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--user", "-u", required=True, help="Username/branch to extract glossary from")
@click.option("--link-features", is_flag=True, help="Also link glossary to existing features")
def extract_glossary(
    repo_path: str,
    user: str,
    link_features: bool,
) -> None:
    """Extract glossary terms from indexed code elements.

    Scans element names to identify domain-specific terminology
    and stores them for enhanced code discovery.

    REPO_PATH is the path to the repository (used to load magaldi.yaml).
    """
    from pathlib import Path

    from magaldi_core.discovery import load_repo_config
    from shared.ai.glossary.extractor import aggregate_glossary_terms
    from shared.ai.glossary.linker import link_glossary_to_features
    from shared.db.elasticsearch import ElasticsearchRepository

    config = load_config(skip_validation=False)

    # Load repo config to get scope/repository
    repo_config_path = Path(repo_path) / "magaldi.yaml"
    if not repo_config_path.exists():
        console.print(f"[red]Error:[/] magaldi.yaml not found in {repo_path}")
        sys.exit(1)

    repo_config = load_repo_config(repo_config_path)
    scope = repo_config["scope"]
    repository = Path(repo_path).name

    console.print("[bold blue]Glossary Extraction[/]")
    console.print(f"  Repository: {scope}/{repository} @{user}")
    console.print()

    try:
        es_repo = ElasticsearchRepository(config)

        # Phase 1: Extract glossary terms
        with console.status("[bold blue]Fetching elements...[/]"):
            elements = es_repo.get_all_elements(
                scope=scope,
                repository=repository,
                username=user,
            )

        console.print(f"  Found {len(elements)} elements")

        with console.status("[bold blue]Extracting glossary terms...[/]"):
            glossary_entries = aggregate_glossary_terms(elements)

        console.print(f"  Extracted {len(glossary_entries)} unique terms")

        # Delete existing glossary entries
        with console.status("[bold blue]Clearing existing glossary...[/]"):
            deleted = es_repo.delete_glossary(scope, repository, user)
            if deleted > 0:
                console.print(f"  Deleted {deleted} existing entries")

        # Index new glossary entries
        with console.status("[bold blue]Indexing glossary entries...[/]"):
            for term, entry in glossary_entries.items():
                glossary_id = f"{scope}:{repository}:{user}:glossary:{term}"
                es_repo.index_glossary(
                    glossary_id=glossary_id,
                    scope=scope,
                    repository=repository,
                    username=user,
                    term=term,
                    total_count=entry.total_count,
                    element_ids=entry.element_ids,
                    file_paths=entry.file_paths,
                )

        console.print(f"  Indexed {len(glossary_entries)} glossary entries")

        # Phase 2: Link to features (optional)
        if link_features:
            console.print()
            console.print("[bold blue]Linking to features...[/]")

            with console.status("[bold blue]Fetching features...[/]"):
                features = es_repo.get_features(scope, repository, user)

            console.print(f"  Found {len(features)} features")

            if features:
                # Build element name lookup
                element_names = {e["element_id"]: e.get("name", "") for e in elements}
                glossary_terms = set(glossary_entries.keys())

                with console.status("[bold blue]Computing associations...[/]"):
                    term_to_features = link_glossary_to_features(
                        glossary_terms, features, element_names
                    )

                # Update glossary entries with associations
                updated = 0
                with console.status("[bold blue]Updating glossary entries...[/]"):
                    for term, associations in term_to_features.items():
                        if associations:
                            glossary_id = f"{scope}:{repository}:{user}:glossary:{term}"
                            es_repo.update_glossary_feature_associations(
                                glossary_id=glossary_id,
                                feature_associations=[
                                    {
                                        "feature_id": a.feature_id,
                                        "feature_label": a.feature_label,
                                        "frequency": a.frequency,
                                        "total_members": a.total_members,
                                        "percentage": a.percentage,
                                    }
                                    for a in associations
                                ],
                            )
                            updated += 1

                console.print(f"  Updated {updated} glossary entries with feature links")

        # Print summary
        console.print()
        console.print("[green]Glossary extraction complete.[/]")

        # Show top terms
        top_terms = sorted(
            glossary_entries.values(),
            key=lambda e: e.total_count,
            reverse=True,
        )[:10]

        if top_terms:
            console.print()
            console.print("[bold]Top terms:[/]")
            for entry in top_terms:
                console.print(f"  {entry.term}: {entry.total_count} occurrences")

    except Exception as e:
        console.print(f"\n[red]Error:[/] {e}")
        sys.exit(1)
```

Also add the `get_all_elements` method to `ElasticsearchRepository` if it doesn't exist (check first):

```python
    def get_all_elements(
        self,
        scope: str,
        repository: str,
        username: str = "main",
        element_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get all elements for a repository.

        Args:
            scope: Scope to filter by.
            repository: Repository to filter by.
            username: Username/branch (default: main).
            element_types: Optional filter by element types.

        Returns:
            List of element dicts.
        """
        client = self._get_client()

        must_clauses: list[dict[str, Any]] = [
            {"term": {"scope": scope}},
            {"term": {"repository": repository}},
            {"term": {"username": username}},
        ]

        # Exclude non-element types
        must_not_clauses: list[dict[str, Any]] = [
            {"term": {"element_type": "feature"}},
            {"term": {"element_type": "subfeature"}},
            {"term": {"element_type": "glossary"}},
        ]

        if element_types:
            must_clauses.append({"terms": {"element_type": element_types}})

        query: dict[str, Any] = {
            "bool": {
                "must": must_clauses,
                "must_not": must_not_clauses,
            }
        }

        # Use scroll for large result sets
        results: list[dict[str, Any]] = []

        response = client.search(
            index=INDEX_NAME,
            query=query,
            size=1000,
            scroll="2m",
        )

        scroll_id = response.get("_scroll_id")
        hits = response.get("hits", {}).get("hits", [])

        while hits:
            for hit in hits:
                source = hit.get("_source", {})
                source["element_id"] = hit.get("_id")
                results.append(source)

            response = client.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = response.get("_scroll_id")
            hits = response.get("hits", {}).get("hits", [])

        # Clear scroll
        if scroll_id:
            try:
                client.clear_scroll(scroll_id=scroll_id)
            except Exception:
                pass

        return results
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_cli.py::TestExtractGlossaryCommand -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/shared/cli.py src/shared/db/elasticsearch.py tests/test_cli.py
git commit -m "feat(glossary): add extract-glossary CLI command"
```

---

## Task 6: MCP Glossary Tools

**Files:**
- Modify: `src/magaldi_mcp/tools.py`
- Modify: `src/magaldi_mcp/server.py`
- Create: `tests/test_mcp_glossary_tools.py`

### Step 1: Write failing tests for MCP tools

```python
# tests/test_mcp_glossary_tools.py
"""Tests for glossary MCP tools."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from magaldi_mcp.tools import (
    list_glossary,
    get_glossary_term,
    search_glossary,
)


class TestListGlossary:
    """Tests for list_glossary tool."""

    def test_returns_glossary_terms(self):
        """Test listing glossary terms."""
        mock_es = MagicMock()
        mock_es.get_glossary_terms.return_value = [
            {"term": "user", "total_count": 10},
            {"term": "email", "total_count": 5},
        ]

        result = list_glossary(
            es=mock_es,
            scope="scope",
            repository="repo",
        )

        assert len(result) == 2
        assert result[0]["term"] == "user"
        mock_es.get_glossary_terms.assert_called_once()


class TestGetGlossaryTerm:
    """Tests for get_glossary_term tool."""

    def test_returns_term_details(self):
        """Test getting glossary term details."""
        mock_es = MagicMock()
        mock_es.get_glossary_term.return_value = {
            "term": "user",
            "total_count": 10,
            "element_ids": ["id1", "id2"],
            "file_paths": ["user.py"],
            "feature_associations": [
                {"feature_label": "auth", "frequency": 3, "percentage": 60.0}
            ],
        }

        result = get_glossary_term(
            es=mock_es,
            scope="scope",
            repository="repo",
            term="user",
        )

        assert result["term"] == "user"
        assert result["total_count"] == 10
        assert len(result["feature_associations"]) == 1


class TestSearchGlossary:
    """Tests for search_glossary tool."""

    def test_searches_by_partial_match(self):
        """Test searching glossary by partial match."""
        mock_es = MagicMock()
        mock_es.search_glossary.return_value = [
            {"term": "user", "total_count": 10},
            {"term": "username", "total_count": 3},
        ]

        result = search_glossary(
            es=mock_es,
            scope="scope",
            repository="repo",
            query="user",
        )

        assert len(result) == 2
        mock_es.search_glossary.assert_called_with(
            scope="scope",
            repository="repo",
            query="user",
            username="main",
        )
```

### Step 2: Run tests to verify they fail

Run: `pytest tests/test_mcp_glossary_tools.py -v`
Expected: FAIL with "cannot import name 'list_glossary'"

### Step 3: Implement MCP tools

Add to `src/magaldi_mcp/tools.py` (at the end, before any closing brackets):

```python
# =============================================================================
# GLOSSARY TOOLS
# =============================================================================


def list_glossary(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    username: str = "main",
    min_count: int = 1,
) -> list[dict[str, Any]]:
    """List all glossary terms for a repository.

    Args:
        es: Elasticsearch repository.
        scope: Repository scope.
        repository: Repository name.
        username: User branch (default: main).
        min_count: Minimum occurrence count to include.

    Returns:
        List of glossary terms sorted by count.
    """
    return es.get_glossary_terms(
        scope=scope,
        repository=repository,
        username=username,
        min_count=min_count,
    )


def get_glossary_term(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    term: str,
    username: str = "main",
) -> dict[str, Any] | None:
    """Get details for a specific glossary term.

    Args:
        es: Elasticsearch repository.
        scope: Repository scope.
        repository: Repository name.
        term: The glossary term.
        username: User branch (default: main).

    Returns:
        Glossary entry with element IDs, files, and feature associations.
    """
    return es.get_glossary_term(
        scope=scope,
        repository=repository,
        term=term,
        username=username,
    )


def search_glossary(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    query: str,
    username: str = "main",
) -> list[dict[str, Any]]:
    """Search glossary terms by partial match.

    Args:
        es: Elasticsearch repository.
        scope: Repository scope.
        repository: Repository name.
        query: Partial term to search for.
        username: User branch (default: main).

    Returns:
        List of matching glossary terms.
    """
    return es.search_glossary(
        scope=scope,
        repository=repository,
        query=query,
        username=username,
    )
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/test_mcp_glossary_tools.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/magaldi_mcp/tools.py tests/test_mcp_glossary_tools.py
git commit -m "feat(glossary): add MCP tool functions"
```

---

## Task 7: Register MCP Tools in Server

**Files:**
- Modify: `src/magaldi_mcp/server.py`
- Modify: `tests/test_mcp_server.py`

### Step 1: Write failing test for tool registration

```python
# Add to tests/test_mcp_server.py

class TestGlossaryToolsRegistration:
    """Tests for glossary tool registration."""

    async def test_list_glossary_tool_registered(self, server):
        """Test that list_glossary tool is registered."""
        tools = await server.server.list_tools()
        tool_names = [t.name for t in tools]

        assert "list_glossary" in tool_names

    async def test_get_glossary_term_tool_registered(self, server):
        """Test that get_glossary_term tool is registered."""
        tools = await server.server.list_tools()
        tool_names = [t.name for t in tools]

        assert "get_glossary_term" in tool_names

    async def test_search_glossary_tool_registered(self, server):
        """Test that search_glossary tool is registered."""
        tools = await server.server.list_tools()
        tool_names = [t.name for t in tools]

        assert "search_glossary" in tool_names
```

### Step 2: Run tests to verify they fail

Run: `pytest tests/test_mcp_server.py::TestGlossaryToolsRegistration -v`
Expected: FAIL with "AssertionError: assert 'list_glossary' in tool_names"

### Step 3: Register tools in server

Add to `src/magaldi_mcp/server.py` in the `list_tools()` function (after the existing tools, before the closing bracket):

```python
                # =============================================================
                # GLOSSARY - Domain concept discovery
                # =============================================================
                Tool(
                    name="list_glossary",
                    description="LIST GLOSSARY: See all domain concepts extracted from code. "
                    "Shows terms like 'user', 'email', 'order' that appear in element names. "
                    "Use to understand what domain concepts exist in the codebase.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scope": {"type": "string", "description": "Filter by scope"},
                            "repository": {"type": "string", "description": "Filter by repo"},
                            "min_count": {
                                "type": "integer",
                                "description": "Minimum occurrence count (default: 1)",
                                "default": 1,
                            },
                        },
                        "required": ["scope", "repository"],
                    },
                ),
                Tool(
                    name="get_glossary_term",
                    description="GLOSSARY DETAILS: Get full details for a domain term. "
                    "Shows which elements contain it, which files, and linked features. "
                    "Use to understand where a concept is used across the codebase.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scope": {"type": "string", "description": "Repository scope"},
                            "repository": {"type": "string", "description": "Repository name"},
                            "term": {"type": "string", "description": "The glossary term"},
                        },
                        "required": ["scope", "repository", "term"],
                    },
                ),
                Tool(
                    name="search_glossary",
                    description="SEARCH GLOSSARY: Find domain terms by partial match. "
                    "Search 'user' to find 'user', 'username', 'userid'. "
                    "Use to discover related domain concepts.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scope": {"type": "string", "description": "Repository scope"},
                            "repository": {"type": "string", "description": "Repository name"},
                            "query": {"type": "string", "description": "Partial term to search"},
                        },
                        "required": ["scope", "repository", "query"],
                    },
                ),
```

Add tool handlers in the `call_tool()` function:

```python
            elif name == "list_glossary":
                from magaldi_mcp.tools import list_glossary

                result = list_glossary(
                    es=self._get_es(),
                    scope=args.get("scope", ""),
                    repository=args.get("repository", ""),
                    username=args.get("username", self.default_username),
                    min_count=args.get("min_count", 1),
                )
                return self._format_result(result)

            elif name == "get_glossary_term":
                from magaldi_mcp.tools import get_glossary_term

                result = get_glossary_term(
                    es=self._get_es(),
                    scope=args.get("scope", ""),
                    repository=args.get("repository", ""),
                    term=args.get("term", ""),
                    username=args.get("username", self.default_username),
                )
                return self._format_result(result)

            elif name == "search_glossary":
                from magaldi_mcp.tools import search_glossary

                result = search_glossary(
                    es=self._get_es(),
                    scope=args.get("scope", ""),
                    repository=args.get("repository", ""),
                    query=args.get("query", ""),
                    username=args.get("username", self.default_username),
                )
                return self._format_result(result)
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/test_mcp_server.py::TestGlossaryToolsRegistration -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/magaldi_mcp/server.py tests/test_mcp_server.py
git commit -m "feat(glossary): register MCP tools in server"
```

---

## Task 8: Enhance search_features with Glossary Filter

**Files:**
- Modify: `src/magaldi_mcp/tools.py`
- Modify: `src/magaldi_mcp/server.py`
- Modify: `tests/test_mcp_tools.py`

### Step 1: Write failing test for glossary filter

```python
# Add to tests/test_mcp_tools.py in TestSearchFeatures class

def test_search_features_with_glossary_filter(self, mock_es_repo, mock_embed_client):
    """Test filtering features by glossary term."""
    # Mock features with glossary associations
    mock_es_repo.search_by_vector.return_value = [
        {
            "element_id": "feat1",
            "name": "user_auth",
            "element_type": "feature",
            "summary": "User authentication",
            "member_count": 5,
            "feature_associations": [
                {"term": "user", "frequency": 3, "percentage": 60.0},
            ],
        },
        {
            "element_id": "feat2",
            "name": "email_sender",
            "element_type": "feature",
            "summary": "Email sending",
            "member_count": 3,
            "feature_associations": [
                {"term": "email", "frequency": 2, "percentage": 66.7},
            ],
        },
    ]

    result = search_features(
        es=mock_es_repo,
        embed_client=mock_embed_client,
        query="",
        glossary_term="user",
        min_percentage=50.0,
    )

    # Should only return features with "user" term at >= 50%
    assert len(result) == 1
    assert result[0]["name"] == "user_auth"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_mcp_tools.py::TestSearchFeatures::test_search_features_with_glossary_filter -v`
Expected: FAIL with "unexpected keyword argument 'glossary_term'"

### Step 3: Enhance search_features

Modify `search_features` in `src/magaldi_mcp/tools.py`:

```python
def search_features(
    es: ElasticsearchRepository,
    embed_client: CodeEmbeddingClient | None,
    query: str,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    limit: int = 20,
    glossary_term: str | None = None,
    min_percentage: float = 0.0,
) -> list[dict[str, Any]]:
    """Search for features/capabilities.

    Args:
        es: Elasticsearch repository.
        embed_client: Embedding client for query (optional).
        query: Natural language search query.
        scope: Filter by scope.
        repository: Filter by repository.
        username: User branch to search.
        limit: Maximum results.
        glossary_term: Filter by glossary term (optional).
        min_percentage: Minimum percentage for glossary term filter (default: 0).

    Returns:
        List of feature dicts with name, summary, member_count.
    """
    # ... existing vector/keyword search logic ...

    # After getting results, filter by glossary term if specified
    if glossary_term:
        filtered_results = []
        for result in results:
            associations = result.get("feature_associations", [])
            for assoc in associations:
                if assoc.get("term") == glossary_term:
                    if assoc.get("percentage", 0) >= min_percentage:
                        filtered_results.append(result)
                        break
        results = filtered_results

    # ... rest of formatting logic ...
```

Also update the tool schema in `server.py`:

```python
                Tool(
                    name="search_features",
                    description="...",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            # ... existing properties ...
                            "glossary_term": {
                                "type": "string",
                                "description": "Filter features by glossary term",
                            },
                            "min_percentage": {
                                "type": "number",
                                "description": "Minimum percentage for glossary term (default: 0)",
                                "default": 0,
                            },
                        },
                        # ...
                    },
                ),
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_mcp_tools.py::TestSearchFeatures::test_search_features_with_glossary_filter -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/magaldi_mcp/tools.py src/magaldi_mcp/server.py tests/test_mcp_tools.py
git commit -m "feat(glossary): add glossary filter to search_features"
```

---

## Task 9: Enhance get_feature_members with Glossary Terms

**Files:**
- Modify: `src/magaldi_mcp/tools.py`
- Modify: `tests/test_mcp_tools.py`

### Step 1: Write failing test

```python
# Add to tests/test_mcp_tools.py in TestGetFeatureMembers class

def test_includes_glossary_terms(self, mock_es_repo):
    """Test that glossary terms are included in response."""
    mock_es_repo.get_document.return_value = {
        "element_id": "feat1",
        "name": "user_auth",
        "element_type": "feature",
        "member_ids": ["id1", "id2"],
        "feature_associations": [
            {"term": "user", "frequency": 2, "total_members": 2, "percentage": 100.0},
        ],
    }
    mock_es_repo.batch_get_documents.return_value = {
        "id1": {"name": "UserService", "element_type": "class"},
        "id2": {"name": "UserController", "element_type": "class"},
    }

    result = get_feature_members(
        es=mock_es_repo,
        feature_id="feat1",
    )

    assert "glossary_terms" in result
    assert len(result["glossary_terms"]) == 1
    assert result["glossary_terms"][0]["term"] == "user"
    assert result["glossary_terms"][0]["percentage"] == 100.0
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_mcp_tools.py::TestGetFeatureMembers::test_includes_glossary_terms -v`
Expected: FAIL with "KeyError: 'glossary_terms'"

### Step 3: Enhance get_feature_members

Modify `get_feature_members` in `src/magaldi_mcp/tools.py` to include glossary terms in the response:

```python
def get_feature_members(
    es: ElasticsearchRepository,
    feature_id: str,
) -> dict[str, Any]:
    """Get all members of a feature.

    ...existing docstring...
    """
    # ... existing logic to get feature and members ...

    # Add glossary terms to response
    feature_associations = feature_doc.get("feature_associations", [])
    glossary_terms = [
        {
            "term": assoc.get("term"),
            "frequency": assoc.get("frequency"),
            "total_members": assoc.get("total_members"),
            "percentage": assoc.get("percentage"),
        }
        for assoc in feature_associations
    ]

    return {
        "feature_id": feature_id,
        "name": feature_doc.get("name"),
        "summary": feature_doc.get("summary"),
        "members": members,
        "glossary_terms": glossary_terms,
    }
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_mcp_tools.py::TestGetFeatureMembers::test_includes_glossary_terms -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/magaldi_mcp/tools.py tests/test_mcp_tools.py
git commit -m "feat(glossary): add glossary_terms to get_feature_members"
```

---

## Task 10: Integration Test

**Files:**
- Create: `tests/integration/test_glossary_e2e.py`

### Step 1: Write integration test

```python
# tests/integration/test_glossary_e2e.py
"""End-to-end integration tests for glossary feature."""

from __future__ import annotations

import pytest

from shared.ai.glossary.extractor import aggregate_glossary_terms, extract_terms
from shared.ai.glossary.linker import compute_feature_associations


class TestGlossaryE2E:
    """End-to-end tests for glossary extraction and linking."""

    def test_full_glossary_workflow(self):
        """Test complete workflow: extract → aggregate → link."""
        # Simulate parsed elements
        elements = [
            {"element_id": "id1", "name": "UserService", "relative_path": "services/user.py"},
            {"element_id": "id2", "name": "UserController", "relative_path": "controllers/user.py"},
            {"element_id": "id3", "name": "EmailService", "relative_path": "services/email.py"},
            {"element_id": "id4", "name": "UserEmailNotifier", "relative_path": "notifiers/user.py"},
            {"element_id": "id5", "name": "OrderProcessor", "relative_path": "processors/order.py"},
        ]

        # Step 1: Extract and aggregate glossary
        glossary = aggregate_glossary_terms(elements)

        assert "user" in glossary
        assert "email" in glossary
        assert "order" in glossary
        assert "notifier" in glossary

        assert glossary["user"].total_count == 3  # UserService, UserController, UserEmailNotifier
        assert glossary["email"].total_count == 2  # EmailService, UserEmailNotifier

        # Step 2: Simulate features
        features = [
            {
                "feature_id": "feat1",
                "label": "user_management",
                "member_ids": ["id1", "id2", "id4"],
            },
            {
                "feature_id": "feat2",
                "label": "notifications",
                "member_ids": ["id3", "id4"],
            },
        ]

        element_names = {e["element_id"]: e["name"] for e in elements}
        glossary_terms = set(glossary.keys())

        # Step 3: Compute associations for feature 1
        assocs = compute_feature_associations(features[0], element_names, glossary_terms)

        user_assoc = next((a for a in assocs if a.term == "user"), None)
        assert user_assoc is not None
        assert user_assoc.frequency == 3  # All 3 members have "user"
        assert user_assoc.percentage == 100.0

        # Step 4: Verify email in feature 2
        assocs2 = compute_feature_associations(features[1], element_names, glossary_terms)

        email_assoc = next((a for a in assocs2 if a.term == "email"), None)
        assert email_assoc is not None
        assert email_assoc.frequency == 2  # Both members have "email"
        assert email_assoc.percentage == 100.0
```

### Step 2: Run integration test

Run: `pytest tests/integration/test_glossary_e2e.py -v`
Expected: PASS (all tests)

### Step 3: Commit

```bash
git add tests/integration/test_glossary_e2e.py
git commit -m "test(glossary): add integration tests"
```

---

## Task 11: Web UI Glossary Routes

**Files:**
- Create: `src/magaldi_web/routes/glossary.py`
- Modify: `src/magaldi_web/app.py`
- Create: `tests/test_web_routes_glossary.py`

### Step 1: Write failing tests for Web API

```python
# tests/test_web_routes_glossary.py
"""Tests for glossary Web API routes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    from magaldi_web.app import app
    return TestClient(app)


@pytest.fixture
def mock_es_repo():
    """Create mock ES repository."""
    return MagicMock()


class TestGlossaryRoutes:
    """Tests for glossary API routes."""

    def test_list_glossary_endpoint(self, client, mock_es_repo):
        """Test GET /api/glossary endpoint."""
        mock_es_repo.get_glossary_terms.return_value = [
            {"term": "user", "total_count": 10, "file_paths": ["user.py"]},
            {"term": "email", "total_count": 5, "file_paths": ["email.py"]},
        ]

        with patch("magaldi_web.routes.glossary.get_es_repository", return_value=mock_es_repo):
            response = client.get("/api/glossary?scope=test&repository=repo")

        assert response.status_code == 200
        data = response.json()
        assert len(data["terms"]) == 2
        assert data["terms"][0]["term"] == "user"

    def test_get_glossary_term_endpoint(self, client, mock_es_repo):
        """Test GET /api/glossary/{term} endpoint."""
        mock_es_repo.get_glossary_term.return_value = {
            "term": "user",
            "total_count": 10,
            "element_ids": ["id1", "id2"],
            "file_paths": ["user.py"],
            "feature_associations": [
                {"feature_label": "auth", "percentage": 60.0}
            ],
        }

        with patch("magaldi_web.routes.glossary.get_es_repository", return_value=mock_es_repo):
            response = client.get("/api/glossary/user?scope=test&repository=repo")

        assert response.status_code == 200
        data = response.json()
        assert data["term"] == "user"
        assert len(data["feature_associations"]) == 1

    def test_search_glossary_endpoint(self, client, mock_es_repo):
        """Test GET /api/glossary/search endpoint."""
        mock_es_repo.search_glossary.return_value = [
            {"term": "user", "total_count": 10},
            {"term": "username", "total_count": 3},
        ]

        with patch("magaldi_web.routes.glossary.get_es_repository", return_value=mock_es_repo):
            response = client.get("/api/glossary/search?scope=test&repository=repo&query=user")

        assert response.status_code == 200
        data = response.json()
        assert len(data["terms"]) == 2
```

### Step 2: Run tests to verify they fail

Run: `pytest tests/test_web_routes_glossary.py -v`
Expected: FAIL with "404 Not Found"

### Step 3: Create glossary routes

```python
# src/magaldi_web/routes/glossary.py
"""Glossary API routes for domain concept exploration."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, HTTPException

from magaldi_web.dependencies import get_es_repository
from shared.db.elasticsearch import ElasticsearchRepository

router = APIRouter()


@router.get("/glossary")
async def list_glossary(
    scope: str,
    repository: str,
    username: str = "main",
    min_count: int = Query(default=1, ge=1),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> dict:
    """List all glossary terms for a repository.

    Args:
        scope: Repository scope.
        repository: Repository name.
        username: User branch (default: main).
        min_count: Minimum occurrence count to include.

    Returns:
        List of glossary terms sorted by count.
    """
    terms = es_repo.get_glossary_terms(
        scope=scope,
        repository=repository,
        username=username,
        min_count=min_count,
    )

    return {
        "terms": terms,
        "total": len(terms),
        "scope": scope,
        "repository": repository,
    }


@router.get("/glossary/search")
async def search_glossary(
    scope: str,
    repository: str,
    query: str,
    username: str = "main",
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> dict:
    """Search glossary terms by partial match.

    Args:
        scope: Repository scope.
        repository: Repository name.
        query: Partial term to search for.
        username: User branch (default: main).

    Returns:
        List of matching glossary terms.
    """
    terms = es_repo.search_glossary(
        scope=scope,
        repository=repository,
        query=query,
        username=username,
    )

    return {
        "terms": terms,
        "total": len(terms),
        "query": query,
    }


@router.get("/glossary/{term}")
async def get_glossary_term(
    term: str,
    scope: str,
    repository: str,
    username: str = "main",
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> dict:
    """Get details for a specific glossary term.

    Args:
        term: The glossary term.
        scope: Repository scope.
        repository: Repository name.
        username: User branch (default: main).

    Returns:
        Full glossary entry with element IDs, files, and feature associations.
    """
    result = es_repo.get_glossary_term(
        scope=scope,
        repository=repository,
        term=term,
        username=username,
    )

    if result is None:
        raise HTTPException(status_code=404, detail=f"Glossary term '{term}' not found")

    return result
```

### Step 4: Register routes in app

Add to `src/magaldi_web/app.py`:

```python
from magaldi_web.routes import glossary

# In the router includes section:
app.include_router(glossary.router, prefix="/api", tags=["glossary"])
```

### Step 5: Run tests to verify they pass

Run: `pytest tests/test_web_routes_glossary.py -v`
Expected: PASS

### Step 6: Commit

```bash
git add src/magaldi_web/routes/glossary.py src/magaldi_web/app.py tests/test_web_routes_glossary.py
git commit -m "feat(glossary): add Web UI API routes"
```

---

## Task 12: Web UI Glossary Page

**Files:**
- Modify: `src/magaldi_web/templates/` (if using templates)
- Or frontend JS/React components (if SPA)

This task depends on your frontend architecture. The API endpoints from Task 11 provide:

- `GET /api/glossary?scope=X&repository=Y` - List all terms
- `GET /api/glossary/search?scope=X&repository=Y&query=Z` - Search terms
- `GET /api/glossary/{term}?scope=X&repository=Y` - Term details

**UI Components needed:**

1. **Glossary List View** - Table/cards showing terms sorted by count
   - Term name
   - Occurrence count
   - Number of files
   - Click to view details

2. **Glossary Search** - Search box with autocomplete/instant results

3. **Term Detail View** - Modal or page showing:
   - Term name and count
   - List of files where it appears
   - Linked features with percentages
   - Link to view elements containing this term

4. **Integration with Features View** - Show glossary terms badge/tags on feature cards

### Step 1: Commit placeholder

```bash
git commit --allow-empty -m "chore(glossary): placeholder for Web UI components"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Term extractor module | `src/shared/ai/glossary/extractor.py` |
| 2 | Glossary aggregator | `src/shared/ai/glossary/extractor.py` |
| 3 | Elasticsearch storage | `src/shared/db/elasticsearch.py` |
| 4 | Glossary linker | `src/shared/ai/glossary/linker.py` |
| 5 | CLI command | `src/shared/cli.py` |
| 6 | MCP tool functions | `src/magaldi_mcp/tools.py` |
| 7 | MCP tool registration | `src/magaldi_mcp/server.py` |
| 8 | Enhanced search_features | `src/magaldi_mcp/tools.py` |
| 9 | Enhanced get_feature_members | `src/magaldi_mcp/tools.py` |
| 10 | Integration tests | `tests/integration/test_glossary_e2e.py` |
| 11 | Web UI API routes | `src/magaldi_web/routes/glossary.py` |
| 12 | Web UI components | Frontend templates/components |

Total: 12 tasks with TDD approach (test first, then implement)
