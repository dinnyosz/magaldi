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
    """Tests that actually call the LLM (skipped by default with -m 'not integration')."""

    @pytest.mark.asyncio
    async def test_real_extraction(self):
        """Test real LLM extraction (requires running Ollama or other LLM)."""
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
        assert len(result) > 0
        print(f"Extracted terms: {names}")
        for item in result:
            print(f"  - {item.name}: {item.description}")
