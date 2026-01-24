# AI-Powered Glossary Extraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace mechanical name-splitting glossary extraction with AI-powered extraction from feature/subfeature summaries, producing glossary items with names and descriptions.

**Architecture:** Process each feature/subfeature individually through an LLM prompt to extract glossary items (actors and concepts), then merge duplicates deterministically. Each glossary item links back to source features. Replaces the existing `extractor.py` approach while keeping the same storage format.

**Tech Stack:** Python, LiteLLM (via existing `llm_client.py`), Elasticsearch, pytest

---

## Task 1: Create AI Glossary Extractor Module

**Files:**
- Create: `src/shared/ai/glossary/ai_extractor.py`
- Create: `tests/test_glossary_ai_extractor.py`

### Step 1: Write failing test for single feature extraction

```python
# tests/test_glossary_ai_extractor.py
"""Tests for AI-powered glossary extraction."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from shared.ai.glossary.ai_extractor import (
    GlossaryItem,
    extract_glossary_from_feature,
)


class TestExtractGlossaryFromFeature:
    """Tests for extracting glossary items from a single feature."""

    @pytest.mark.asyncio
    async def test_extracts_items_from_feature_summary(self):
        """Test that glossary items are extracted from feature summary."""
        feature = {
            "feature_id": "scope:repo:main:feature:auth",
            "label": "authentication",
            "summary": "Handles user login and registration workflows.",
        }

        mock_response = [
            {"name": "user", "description": "Person who authenticates with the system"},
            {"name": "login", "description": "Process of verifying user credentials"},
            {"name": "registration", "description": "Process of creating a new user account"},
        ]

        with patch(
            "shared.ai.glossary.ai_extractor.call_llm_for_glossary",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await extract_glossary_from_feature(feature)

        assert len(result) == 3
        assert all(isinstance(item, GlossaryItem) for item in result)
        assert result[0].name == "user"
        assert result[0].description == "Person who authenticates with the system"
        assert result[0].source_feature_id == "scope:repo:main:feature:auth"

    @pytest.mark.asyncio
    async def test_handles_empty_response(self):
        """Test handling when LLM returns no items."""
        feature = {
            "feature_id": "scope:repo:main:feature:utils",
            "label": "utilities",
            "summary": "Generic helper functions.",
        }

        with patch(
            "shared.ai.glossary.ai_extractor.call_llm_for_glossary",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await extract_glossary_from_feature(feature)

        assert result == []
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_glossary_ai_extractor.py::TestExtractGlossaryFromFeature::test_extracts_items_from_feature_summary -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

### Step 3: Write minimal implementation

```python
# src/shared/ai/glossary/ai_extractor.py
"""AI-powered glossary extraction from feature summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GlossaryItem:
    """A glossary item extracted from a feature."""

    name: str
    description: str
    source_feature_id: str
    source_feature_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.source_feature_id and self.source_feature_id not in self.source_feature_ids:
            self.source_feature_ids = [self.source_feature_id]


async def call_llm_for_glossary(summary: str, label: str) -> list[dict[str, str]]:
    """Call LLM to extract glossary items from a summary.

    This is a placeholder - will be implemented in Task 2.
    """
    return []


async def extract_glossary_from_feature(
    feature: dict[str, Any],
) -> list[GlossaryItem]:
    """Extract glossary items from a single feature.

    Args:
        feature: Feature dict with feature_id, label, summary.

    Returns:
        List of GlossaryItem extracted from the feature.
    """
    feature_id = feature.get("feature_id") or feature.get("subfeature_id", "")
    label = feature.get("label", "")
    summary = feature.get("summary", "")

    if not summary:
        return []

    raw_items = await call_llm_for_glossary(summary, label)

    items = []
    for raw in raw_items:
        name = raw.get("name", "").lower().strip()
        description = raw.get("description", "").strip()

        if name and description:
            items.append(GlossaryItem(
                name=name,
                description=description,
                source_feature_id=feature_id,
            ))

    return items
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/test_glossary_ai_extractor.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/shared/ai/glossary/ai_extractor.py tests/test_glossary_ai_extractor.py
git commit -m "feat(glossary): add AI extractor module skeleton"
```

---

## Task 2: Implement LLM Call for Glossary Extraction

**Files:**
- Modify: `src/shared/ai/glossary/ai_extractor.py`
- Modify: `tests/test_glossary_ai_extractor.py`

### Step 1: Write failing test for LLM prompt construction

```python
# Add to tests/test_glossary_ai_extractor.py

from shared.ai.glossary.ai_extractor import build_glossary_prompt


class TestBuildGlossaryPrompt:
    """Tests for prompt construction."""

    def test_includes_summary_in_prompt(self):
        """Test that the summary is included in the prompt."""
        prompt = build_glossary_prompt(
            summary="Handles user authentication and login.",
            label="authentication",
        )

        assert "Handles user authentication and login." in prompt
        assert "authentication" in prompt

    def test_instructs_for_actors_and_concepts(self):
        """Test that prompt asks for actors and concepts."""
        prompt = build_glossary_prompt(
            summary="Test summary",
            label="test",
        )

        assert "actor" in prompt.lower() or "concept" in prompt.lower()
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_glossary_ai_extractor.py::TestBuildGlossaryPrompt -v`
Expected: FAIL with "ImportError"

### Step 3: Implement prompt builder and LLM call

```python
# Add to src/shared/ai/glossary/ai_extractor.py

import json
import re

from shared.ai.llm_client import LLMClient
from shared.config import MagaldiConfig


GLOSSARY_EXTRACTION_PROMPT = """You are extracting glossary terms from a code feature description.

Feature: {label}
Description: {summary}

Extract domain-specific glossary items that represent:
- Actors: entities that perform actions (e.g., user, admin, worker, client)
- Concepts: domain objects or processes (e.g., email, registration, authentication, payment)

For each item, provide:
- name: a short lowercase term (1-2 words)
- description: one sentence explaining what it represents in this codebase

Rules:
- Only extract terms that are meaningful in the domain context
- Ignore generic programming terms (function, class, method, variable, etc.)
- Ignore technical implementation details (cache, queue, handler, etc.)
- Focus on business/domain concepts

Return a JSON array of objects with "name" and "description" fields.
Return an empty array [] if no domain-specific terms are found.

Example output:
[
  {{"name": "user", "description": "A person who interacts with the system"}},
  {{"name": "registration", "description": "The process of creating a new account"}}
]

JSON output:"""


def build_glossary_prompt(summary: str, label: str) -> str:
    """Build the prompt for glossary extraction.

    Args:
        summary: Feature summary text.
        label: Feature label/name.

    Returns:
        Formatted prompt string.
    """
    return GLOSSARY_EXTRACTION_PROMPT.format(label=label, summary=summary)


def parse_llm_response(response: str) -> list[dict[str, str]]:
    """Parse LLM response to extract glossary items.

    Args:
        response: Raw LLM response text.

    Returns:
        List of dicts with name and description.
    """
    # Try to find JSON array in response
    response = response.strip()

    # Handle markdown code blocks
    if "```" in response:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if match:
            response = match.group(1).strip()

    try:
        data = json.loads(response)
        if isinstance(data, list):
            return [
                item for item in data
                if isinstance(item, dict) and "name" in item and "description" in item
            ]
    except json.JSONDecodeError:
        pass

    return []


async def call_llm_for_glossary(
    summary: str,
    label: str,
    config: MagaldiConfig | None = None,
) -> list[dict[str, str]]:
    """Call LLM to extract glossary items from a summary.

    Args:
        summary: Feature summary text.
        label: Feature label.
        config: Optional config (uses default if not provided).

    Returns:
        List of dicts with name and description.
    """
    if config is None:
        config = MagaldiConfig()

    prompt = build_glossary_prompt(summary, label)

    client = LLMClient(config)
    response = await client.complete(prompt)

    return parse_llm_response(response)
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/test_glossary_ai_extractor.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/shared/ai/glossary/ai_extractor.py tests/test_glossary_ai_extractor.py
git commit -m "feat(glossary): implement LLM call for glossary extraction"
```

---

## Task 3: Implement Glossary Merging Logic

**Files:**
- Modify: `src/shared/ai/glossary/ai_extractor.py`
- Modify: `tests/test_glossary_ai_extractor.py`

### Step 1: Write failing tests for merge logic

```python
# Add to tests/test_glossary_ai_extractor.py

from shared.ai.glossary.ai_extractor import merge_glossary_items


class TestMergeGlossaryItems:
    """Tests for merging glossary items from multiple features."""

    def test_merges_identical_names(self):
        """Test that items with same name are merged."""
        items = [
            GlossaryItem(
                name="user",
                description="A person using the system",
                source_feature_id="feature:auth",
            ),
            GlossaryItem(
                name="user",
                description="An authenticated individual",
                source_feature_id="feature:profile",
            ),
        ]

        merged = merge_glossary_items(items)

        assert len(merged) == 1
        assert merged[0].name == "user"
        assert set(merged[0].source_feature_ids) == {"feature:auth", "feature:profile"}

    def test_normalizes_name_variations(self):
        """Test that similar names are normalized and merged."""
        items = [
            GlossaryItem(name="user", description="Desc 1", source_feature_id="f1"),
            GlossaryItem(name="users", description="Desc 2", source_feature_id="f2"),
        ]

        merged = merge_glossary_items(items)

        # "users" should normalize to "user"
        assert len(merged) == 1
        assert merged[0].name == "user"

    def test_keeps_distinct_items_separate(self):
        """Test that different terms stay separate."""
        items = [
            GlossaryItem(name="user", description="A person", source_feature_id="f1"),
            GlossaryItem(name="email", description="Electronic mail", source_feature_id="f2"),
        ]

        merged = merge_glossary_items(items)

        assert len(merged) == 2
        names = {item.name for item in merged}
        assert names == {"user", "email"}

    def test_picks_longest_description(self):
        """Test that the most detailed description is kept."""
        items = [
            GlossaryItem(name="user", description="A person", source_feature_id="f1"),
            GlossaryItem(
                name="user",
                description="A person who interacts with the system through the UI",
                source_feature_id="f2",
            ),
        ]

        merged = merge_glossary_items(items)

        assert "interacts with the system" in merged[0].description
```

### Step 2: Run tests to verify they fail

Run: `pytest tests/test_glossary_ai_extractor.py::TestMergeGlossaryItems -v`
Expected: FAIL with "ImportError"

### Step 3: Implement merge logic

```python
# Add to src/shared/ai/glossary/ai_extractor.py

def normalize_term(name: str) -> str:
    """Normalize a glossary term name.

    Handles pluralization and common variations.

    Args:
        name: Raw term name.

    Returns:
        Normalized term.
    """
    name = name.lower().strip()

    # Simple depluralization for common patterns
    if name.endswith("ies"):
        # entries -> entry, but not "series"
        singular = name[:-3] + "y"
        if len(singular) > 2:
            name = singular
    elif name.endswith("es") and not name.endswith("sse"):
        # processes -> process, but keep "classes" handling simple
        if name.endswith("sses"):
            name = name[:-2]  # classes -> class
        elif name.endswith("xes") or name.endswith("ches") or name.endswith("shes"):
            name = name[:-2]
    elif name.endswith("s") and len(name) > 3 and not name.endswith("ss"):
        # users -> user
        name = name[:-1]

    return name


def merge_glossary_items(items: list[GlossaryItem]) -> list[GlossaryItem]:
    """Merge glossary items with same/similar names.

    Items with the same normalized name are merged:
    - Feature IDs are combined
    - The longest description is kept

    Args:
        items: List of GlossaryItem to merge.

    Returns:
        List of merged GlossaryItem.
    """
    # Group by normalized name
    grouped: dict[str, list[GlossaryItem]] = {}

    for item in items:
        normalized = normalize_term(item.name)
        if normalized not in grouped:
            grouped[normalized] = []
        grouped[normalized].append(item)

    # Merge each group
    merged = []
    for normalized_name, group in grouped.items():
        # Collect all feature IDs
        all_feature_ids: list[str] = []
        for item in group:
            for fid in item.source_feature_ids:
                if fid not in all_feature_ids:
                    all_feature_ids.append(fid)

        # Pick the longest description
        best_description = max(
            (item.description for item in group),
            key=len,
        )

        merged.append(GlossaryItem(
            name=normalized_name,
            description=best_description,
            source_feature_id=all_feature_ids[0] if all_feature_ids else "",
            source_feature_ids=all_feature_ids,
        ))

    # Sort by name for consistent ordering
    merged.sort(key=lambda x: x.name)

    return merged
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/test_glossary_ai_extractor.py::TestMergeGlossaryItems -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/shared/ai/glossary/ai_extractor.py tests/test_glossary_ai_extractor.py
git commit -m "feat(glossary): add merge logic for duplicate terms"
```

---

## Task 4: Create Main Extraction Pipeline Function

**Files:**
- Modify: `src/shared/ai/glossary/ai_extractor.py`
- Modify: `tests/test_glossary_ai_extractor.py`

### Step 1: Write failing test for full pipeline

```python
# Add to tests/test_glossary_ai_extractor.py

from shared.ai.glossary.ai_extractor import extract_glossary_from_features


class TestExtractGlossaryFromFeatures:
    """Tests for the full extraction pipeline."""

    @pytest.mark.asyncio
    async def test_processes_multiple_features(self):
        """Test processing multiple features and merging results."""
        features = [
            {
                "feature_id": "f1",
                "label": "authentication",
                "summary": "Handles user login.",
            },
            {
                "feature_id": "f2",
                "label": "registration",
                "summary": "Manages user registration.",
            },
        ]

        mock_responses = [
            [{"name": "user", "description": "A system user"}],
            [
                {"name": "user", "description": "A person registering"},
                {"name": "registration", "description": "Account creation process"},
            ],
        ]

        call_count = 0

        async def mock_call(*args, **kwargs):
            nonlocal call_count
            result = mock_responses[call_count]
            call_count += 1
            return result

        with patch(
            "shared.ai.glossary.ai_extractor.call_llm_for_glossary",
            side_effect=mock_call,
        ):
            result = await extract_glossary_from_features(features)

        # "user" appears in both, should be merged
        assert len(result) == 2
        user_item = next(i for i in result if i.name == "user")
        assert set(user_item.source_feature_ids) == {"f1", "f2"}

    @pytest.mark.asyncio
    async def test_handles_empty_features_list(self):
        """Test handling empty input."""
        result = await extract_glossary_from_features([])
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_features_without_summary(self):
        """Test that features without summaries are skipped."""
        features = [
            {"feature_id": "f1", "label": "test", "summary": ""},
            {"feature_id": "f2", "label": "real", "summary": "Real feature."},
        ]

        with patch(
            "shared.ai.glossary.ai_extractor.call_llm_for_glossary",
            new_callable=AsyncMock,
            return_value=[{"name": "item", "description": "An item"}],
        ) as mock_call:
            await extract_glossary_from_features(features)

        # Should only be called once (for f2)
        assert mock_call.call_count == 1
```

### Step 2: Run tests to verify they fail

Run: `pytest tests/test_glossary_ai_extractor.py::TestExtractGlossaryFromFeatures -v`
Expected: FAIL with "ImportError"

### Step 3: Implement main pipeline function

```python
# Add to src/shared/ai/glossary/ai_extractor.py

async def extract_glossary_from_features(
    features: list[dict[str, Any]],
    config: MagaldiConfig | None = None,
    progress_callback: callable | None = None,
) -> list[GlossaryItem]:
    """Extract and merge glossary items from multiple features.

    Processes each feature through the LLM, then merges duplicates.

    Args:
        features: List of feature/subfeature dicts with feature_id, label, summary.
        config: Optional config for LLM client.
        progress_callback: Optional callback(current, total) for progress updates.

    Returns:
        List of merged GlossaryItem.
    """
    if not features:
        return []

    all_items: list[GlossaryItem] = []
    total = len(features)

    for i, feature in enumerate(features):
        if progress_callback:
            progress_callback(i + 1, total)

        items = await extract_glossary_from_feature(feature, config)
        all_items.extend(items)

    return merge_glossary_items(all_items)


# Update extract_glossary_from_feature to accept config
async def extract_glossary_from_feature(
    feature: dict[str, Any],
    config: MagaldiConfig | None = None,
) -> list[GlossaryItem]:
    """Extract glossary items from a single feature.

    Args:
        feature: Feature dict with feature_id, label, summary.
        config: Optional config for LLM client.

    Returns:
        List of GlossaryItem extracted from the feature.
    """
    feature_id = feature.get("feature_id") or feature.get("subfeature_id", "")
    label = feature.get("label", "")
    summary = feature.get("summary", "")

    if not summary:
        return []

    raw_items = await call_llm_for_glossary(summary, label, config)

    items = []
    for raw in raw_items:
        name = raw.get("name", "").lower().strip()
        description = raw.get("description", "").strip()

        if name and description:
            items.append(GlossaryItem(
                name=name,
                description=description,
                source_feature_id=feature_id,
            ))

    return items
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/test_glossary_ai_extractor.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/shared/ai/glossary/ai_extractor.py tests/test_glossary_ai_extractor.py
git commit -m "feat(glossary): add main extraction pipeline"
```

---

## Task 5: Update CLI to Use AI Extraction

**Files:**
- Modify: `src/shared/cli.py`
- Modify: `tests/test_cli_glossary.py` (create if needed)

### Step 1: Write failing test for CLI integration

```python
# tests/test_cli_glossary.py
"""Tests for glossary CLI commands."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.cli import run_glossary_extraction_ai


class TestRunGlossaryExtractionAI:
    """Tests for AI-powered glossary extraction CLI."""

    @pytest.mark.asyncio
    async def test_fetches_features_and_subfeatures(self):
        """Test that both features and subfeatures are fetched."""
        mock_es = MagicMock()
        mock_es.get_features.return_value = [
            {"feature_id": "f1", "label": "auth", "summary": "Auth feature"}
        ]
        mock_es.get_subfeatures.return_value = [
            {"subfeature_id": "sf1", "label": "login", "summary": "Login subfeature"}
        ]
        mock_es.delete_glossary.return_value = 0

        with patch(
            "shared.cli.extract_glossary_from_features",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_extract:
            from shared.config import MagaldiConfig
            config = MagaldiConfig()

            await run_glossary_extraction_ai(
                scope="test",
                repository="repo",
                username="main",
                config=config,
                es_repo=mock_es,
            )

        # Should be called with combined features + subfeatures
        call_args = mock_extract.call_args[0][0]
        assert len(call_args) == 2
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_cli_glossary.py -v`
Expected: FAIL

### Step 3: Implement new CLI function

Add to `src/shared/cli.py`:

```python
# Add import at top
from shared.ai.glossary.ai_extractor import extract_glossary_from_features, GlossaryItem

# Add new function
async def run_glossary_extraction_ai(
    scope: str,
    repository: str,
    username: str,
    config: MagaldiConfig,
    es_repo: ElasticsearchRepository | None = None,
) -> dict | None:
    """Run AI-powered Glossary Extraction.

    Args:
        scope: Repository scope.
        repository: Repository name.
        username: Username/branch.
        config: Magaldi configuration.
        es_repo: Optional ES repository (creates one if not provided).

    Returns:
        Dict with glossary extraction results or None if no features.
    """
    if es_repo is None:
        from shared.db.elasticsearch import ElasticsearchRepository
        es_repo = ElasticsearchRepository(config)

    # Fetch features and subfeatures
    with console.status("[bold blue]Fetching features...[/]"):
        features = es_repo.get_features(scope, repository, username)
        subfeatures = es_repo.get_subfeatures(scope, repository, username)

    all_features = features + subfeatures

    if not all_features:
        console.print("  [dim]No features found[/]")
        return None

    console.print(f"  Found {len(features)} features, {len(subfeatures)} subfeatures")

    # Extract glossary using AI
    def progress(current: int, total: int) -> None:
        pass  # Rich status handles this

    with console.status("[bold blue]Extracting glossary with AI...[/]"):
        import asyncio
        glossary_items = asyncio.get_event_loop().run_until_complete(
            extract_glossary_from_features(all_features, config)
        )

    if not glossary_items:
        console.print("  [dim]No glossary items extracted[/]")
        return {"terms_count": 0}

    console.print(f"  Extracted [green]{len(glossary_items)}[/] glossary items")

    # Delete existing glossary entries
    with console.status("[bold blue]Clearing existing glossary...[/]"):
        deleted = es_repo.delete_glossary(scope, repository, username)
        if deleted > 0:
            console.print(f"  Deleted {deleted} existing entries")

    # Index new glossary entries
    with console.status("[bold blue]Indexing glossary entries...[/]"):
        for item in glossary_items:
            glossary_id = f"{scope}:{repository}:{username}:glossary:{item.name}"

            # Convert feature IDs to element IDs for storage compatibility
            # Features link to their member elements
            element_ids = item.source_feature_ids
            file_paths: list[str] = []  # Could be computed from features if needed

            es_repo.index_glossary(
                glossary_id=glossary_id,
                scope=scope,
                repository=repository,
                username=username,
                term=item.name,
                total_count=len(item.source_feature_ids),
                element_ids=element_ids,
                file_paths=file_paths,
                description=item.description,  # New field
            )

    console.print(f"  Indexed [green]{len(glossary_items)}[/] glossary entries")

    return {
        "terms_count": len(glossary_items),
        "terms": [item.name for item in glossary_items],
    }
```

### Step 4: Run tests

Run: `pytest tests/test_cli_glossary.py -v`
Expected: PASS (or may need ES schema update - see Task 6)

### Step 5: Commit

```bash
git add src/shared/cli.py tests/test_cli_glossary.py
git commit -m "feat(glossary): add AI extraction CLI command"
```

---

## Task 6: Add Description Field to Glossary Storage

**Files:**
- Modify: `src/shared/db/elasticsearch.py`
- Modify: `tests/test_glossary_elasticsearch.py`

### Step 1: Write failing test for description field

```python
# Add to tests/test_glossary_elasticsearch.py

def test_indexes_glossary_with_description(self, mock_repo, mock_es_client):
    """Test that description field is indexed."""
    mock_repo.index_glossary(
        glossary_id="test:repo:main:glossary:user",
        scope="test",
        repository="repo",
        username="main",
        term="user",
        total_count=5,
        element_ids=["e1", "e2"],
        file_paths=["path/to/file.py"],
        description="A person who uses the system",
    )

    call_args = mock_es_client.index.call_args
    body = call_args[1]["body"]
    assert body["description"] == "A person who uses the system"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_glossary_elasticsearch.py::test_indexes_glossary_with_description -v`
Expected: FAIL (unexpected keyword argument 'description')

### Step 3: Update index_glossary method

Modify `src/shared/db/elasticsearch.py`:

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
    description: str = "",  # Add this parameter
) -> bool:
    """Index a glossary entry.

    Args:
        glossary_id: Unique glossary ID.
        scope: Repository scope.
        repository: Repository name.
        username: User branch.
        term: The glossary term.
        total_count: Number of occurrences.
        element_ids: List of element IDs where term appears.
        file_paths: List of file paths where term appears.
        description: Human-readable description of the term.

    Returns:
        True if successful.
    """
    client = self._get_client()

    doc = {
        "element_id": glossary_id,
        "element_type": "glossary",
        "scope": scope,
        "repository": repository,
        "username": username,
        "term": term,
        "total_count": total_count,
        "element_ids": element_ids,
        "file_paths": file_paths,
        "description": description,  # Add this field
        "level": -3,
    }

    client.index(index=INDEX_NAME, id=glossary_id, body=doc)
    return True
```

### Step 4: Run tests

Run: `pytest tests/test_glossary_elasticsearch.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/shared/db/elasticsearch.py tests/test_glossary_elasticsearch.py
git commit -m "feat(glossary): add description field to storage"
```

---

## Task 7: Update MCP Tools to Return Description

**Files:**
- Modify: `src/magaldi_mcp/server.py` (if needed)
- Modify: `tests/test_mcp_glossary_tools.py`

### Step 1: Verify description is returned

Check if `get_glossary_term` and `list_glossary` already return description field from ES. If not, update the `_source` fields in queries.

### Step 2: Write test

```python
# Add to tests/test_mcp_glossary_tools.py

def test_returns_description_field(self, mock_es_repo):
    """Test that glossary terms include description."""
    mock_es_repo.get_glossary_term.return_value = {
        "term": "user",
        "total_count": 5,
        "element_ids": ["e1"],
        "description": "A person using the system",
    }

    # Call the MCP tool
    result = get_glossary_term_tool(
        scope="test", repository="repo", term="user"
    )

    assert "description" in result
    assert result["description"] == "A person using the system"
```

### Step 3: Update if needed and run tests

Run: `pytest tests/test_mcp_glossary_tools.py -v`
Expected: PASS

### Step 4: Commit

```bash
git add src/magaldi_mcp/server.py tests/test_mcp_glossary_tools.py
git commit -m "feat(glossary): include description in MCP responses"
```

---

## Task 8: Integration Test

**Files:**
- Create: `tests/test_glossary_ai_integration.py`

### Step 1: Write integration test

```python
# tests/test_glossary_ai_integration.py
"""Integration tests for AI glossary extraction."""

from __future__ import annotations

import pytest

from shared.ai.glossary.ai_extractor import (
    GlossaryItem,
    extract_glossary_from_features,
    merge_glossary_items,
    normalize_term,
)


class TestGlossaryIntegration:
    """End-to-end tests for glossary extraction workflow."""

    def test_normalize_common_plurals(self):
        """Test normalization of common plural forms."""
        assert normalize_term("users") == "user"
        assert normalize_term("entries") == "entry"
        assert normalize_term("classes") == "class"
        assert normalize_term("processes") == "process"

    def test_merge_workflow(self):
        """Test full merge workflow with realistic data."""
        items = [
            GlossaryItem(
                name="user",
                description="A person",
                source_feature_id="auth",
            ),
            GlossaryItem(
                name="users",
                description="People who use the system",
                source_feature_id="profile",
            ),
            GlossaryItem(
                name="email",
                description="Electronic mail address",
                source_feature_id="notification",
            ),
            GlossaryItem(
                name="registration",
                description="Account creation",
                source_feature_id="auth",
            ),
        ]

        merged = merge_glossary_items(items)

        assert len(merged) == 3  # user/users merged
        user = next(i for i in merged if i.name == "user")
        assert len(user.source_feature_ids) == 2
        # Should have the longer description
        assert "system" in user.description


@pytest.mark.integration
class TestLLMIntegration:
    """Tests that actually call the LLM (skipped by default)."""

    @pytest.mark.asyncio
    async def test_real_extraction(self):
        """Test real LLM extraction (requires running LLM)."""
        features = [
            {
                "feature_id": "test:repo:main:feature:auth",
                "label": "authentication",
                "summary": "Handles user authentication including login, logout, and password reset workflows. Validates credentials against the database and manages session tokens.",
            }
        ]

        result = await extract_glossary_from_features(features)

        # Should extract meaningful terms
        names = {item.name for item in result}
        # At minimum should find these common domain terms
        assert len(result) > 0
        # The LLM should identify at least "user" or "authentication" or similar
        print(f"Extracted terms: {names}")
```

### Step 2: Run tests

Run: `pytest tests/test_glossary_ai_integration.py -v -m "not integration"`
Expected: PASS (skipping the actual LLM test)

Run with LLM (optional): `pytest tests/test_glossary_ai_integration.py -v`

### Step 3: Commit

```bash
git add tests/test_glossary_ai_integration.py
git commit -m "test(glossary): add integration tests for AI extraction"
```

---

## Task 9: Wire Up CLI Command

**Files:**
- Modify: `src/shared/cli.py`

### Step 1: Add CLI option to switch between old and AI extraction

```python
# Update the extract_glossary command in cli.py

@cli.command()
@click.argument("repo_path", type=click.Path(exists=True))
@click.option("--user", "-u", default="main", help="Username/branch")
@click.option("--link-features", is_flag=True, help="Link to existing features")
@click.option("--ai", is_flag=True, help="Use AI-powered extraction from feature summaries")
def extract_glossary(
    repo_path: str,
    user: str,
    link_features: bool,
    ai: bool,
) -> None:
    """Extract domain terms from code elements.

    REPO_PATH: Path to the repository.
    """
    config = load_config_from_repo(repo_path)
    if not config:
        raise click.ClickException("No magaldi.yaml found")

    scope = config.scope
    repository = config.repository

    console.print(f"\n[bold]Glossary Extraction[/] - {scope}/{repository}")
    console.print(f"  User: {user}")
    console.print(f"  Mode: {'AI-powered' if ai else 'Name-based'}")
    console.print()

    if ai:
        import asyncio
        result = asyncio.run(
            run_glossary_extraction_ai(scope, repository, user, config)
        )
    else:
        result = run_glossary_extraction(
            scope, repository, user, config, link_features
        )

    if result:
        console.print(f"\n[green]Done![/] Extracted {result.get('terms_count', 0)} terms")
```

### Step 2: Test manually

Run: `magaldi extract-glossary /path/to/repo --ai`

### Step 3: Commit

```bash
git add src/shared/cli.py
git commit -m "feat(glossary): add --ai flag for AI-powered extraction"
```

---

## Summary

This plan replaces the mechanical name-splitting approach with AI-powered extraction:

1. **Task 1-4**: Core AI extractor module with LLM calls, parsing, and merge logic
2. **Task 5-6**: CLI integration and storage updates for description field
3. **Task 7**: MCP tool updates to expose description
4. **Task 8-9**: Integration tests and CLI wiring

The key design decisions:
- Process per-feature for guaranteed accurate linking
- Merge duplicates deterministically (longest description wins)
- Normalize plurals (users → user)
- Add `--ai` flag to CLI for opt-in, keeping old approach available
