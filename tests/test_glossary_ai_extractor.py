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
