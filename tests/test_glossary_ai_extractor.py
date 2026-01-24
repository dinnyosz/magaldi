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

    @pytest.mark.asyncio
    async def test_skips_llm_when_summary_missing(self):
        """Test that features without summary skip LLM call and return empty list."""
        feature = {
            "feature_id": "scope:repo:main:feature:empty",
            "label": "empty_feature",
            # No summary key
        }

        mock_llm = AsyncMock()
        with patch(
            "shared.ai.glossary.ai_extractor.call_llm_for_glossary",
            mock_llm,
        ):
            result = await extract_glossary_from_feature(feature)

        assert result == []
        mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_llm_when_summary_empty(self):
        """Test that features with empty summary skip LLM call and return empty list."""
        feature = {
            "feature_id": "scope:repo:main:feature:empty",
            "label": "empty_feature",
            "summary": "",  # Empty summary
        }

        mock_llm = AsyncMock()
        with patch(
            "shared.ai.glossary.ai_extractor.call_llm_for_glossary",
            mock_llm,
        ):
            result = await extract_glossary_from_feature(feature)

        assert result == []
        mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_filters_malformed_items_missing_name(self):
        """Test that items missing 'name' are filtered out."""
        feature = {
            "feature_id": "scope:repo:main:feature:auth",
            "label": "authentication",
            "summary": "Handles user login.",
        }

        mock_response = [
            {"name": "user", "description": "A valid user"},
            {"description": "Missing name field"},  # No 'name' key
            {"name": "", "description": "Empty name"},  # Empty 'name'
        ]

        with patch(
            "shared.ai.glossary.ai_extractor.call_llm_for_glossary",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await extract_glossary_from_feature(feature)

        assert len(result) == 1
        assert result[0].name == "user"

    @pytest.mark.asyncio
    async def test_filters_malformed_items_missing_description(self):
        """Test that items missing 'description' are filtered out."""
        feature = {
            "feature_id": "scope:repo:main:feature:auth",
            "label": "authentication",
            "summary": "Handles user login.",
        }

        mock_response = [
            {"name": "user", "description": "A valid user"},
            {"name": "orphan"},  # No 'description' key
            {"name": "empty_desc", "description": ""},  # Empty 'description'
        ]

        with patch(
            "shared.ai.glossary.ai_extractor.call_llm_for_glossary",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await extract_glossary_from_feature(feature)

        assert len(result) == 1
        assert result[0].name == "user"


class TestGlossaryItem:
    """Tests for the GlossaryItem dataclass."""

    def test_post_init_populates_source_feature_ids(self):
        """Test that __post_init__ populates source_feature_ids from source_feature_id."""
        item = GlossaryItem(
            name="user",
            description="A person who uses the system",
            source_feature_id="scope:repo:main:feature:auth",
        )

        assert item.source_feature_ids == ["scope:repo:main:feature:auth"]

    def test_post_init_does_not_duplicate_existing_id(self):
        """Test that __post_init__ doesn't duplicate if id already in list."""
        item = GlossaryItem(
            name="user",
            description="A person who uses the system",
            source_feature_id="scope:repo:main:feature:auth",
            source_feature_ids=["scope:repo:main:feature:auth"],
        )

        assert item.source_feature_ids == ["scope:repo:main:feature:auth"]

    def test_post_init_handles_empty_source_feature_id(self):
        """Test that __post_init__ handles empty source_feature_id gracefully."""
        item = GlossaryItem(
            name="user",
            description="A person who uses the system",
            source_feature_id="",
        )

        assert item.source_feature_ids == []
