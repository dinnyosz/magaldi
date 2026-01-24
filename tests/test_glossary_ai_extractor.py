"""Tests for AI-powered glossary extraction."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.ai.glossary.ai_extractor import (
    GlossaryItem,
    build_glossary_prompt,
    call_llm_for_glossary,
    extract_glossary_from_feature,
    parse_llm_response,
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

    def test_includes_json_format_instruction(self):
        """Test that prompt includes JSON format instructions."""
        prompt = build_glossary_prompt(
            summary="Test summary",
            label="test",
        )

        assert "JSON" in prompt
        assert "name" in prompt.lower()
        assert "description" in prompt.lower()

    def test_includes_example_output(self):
        """Test that prompt includes example output for LLM guidance."""
        prompt = build_glossary_prompt(
            summary="Test summary",
            label="test",
        )

        # Should have example JSON to guide the model
        assert '{"name"' in prompt or '"name":' in prompt


class TestParseLLMResponse:
    """Tests for parsing LLM response."""

    def test_parses_valid_json_array(self):
        """Test parsing valid JSON array."""
        response = '[{"name": "user", "description": "A person"}]'
        result = parse_llm_response(response)

        assert len(result) == 1
        assert result[0]["name"] == "user"
        assert result[0]["description"] == "A person"

    def test_parses_json_in_markdown_code_block(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        response = """```json
[{"name": "user", "description": "A person"}]
```"""
        result = parse_llm_response(response)

        assert len(result) == 1
        assert result[0]["name"] == "user"

    def test_parses_json_in_markdown_code_block_without_lang(self):
        """Test parsing JSON in code block without language specifier."""
        response = """```
[{"name": "user", "description": "A person"}]
```"""
        result = parse_llm_response(response)

        assert len(result) == 1
        assert result[0]["name"] == "user"

    def test_returns_empty_list_for_invalid_json(self):
        """Test that invalid JSON returns empty list."""
        response = "This is not JSON at all"
        result = parse_llm_response(response)

        assert result == []

    def test_filters_items_missing_name(self):
        """Test that items without 'name' key are filtered out."""
        response = """[
            {"name": "user", "description": "Valid"},
            {"description": "Missing name"},
            {"other": "field"}
        ]"""
        result = parse_llm_response(response)

        assert len(result) == 1
        assert result[0]["name"] == "user"

    def test_filters_items_missing_description(self):
        """Test that items without 'description' key are filtered out."""
        response = """[
            {"name": "user", "description": "Valid"},
            {"name": "orphan"},
            {"name": "other", "other": "field"}
        ]"""
        result = parse_llm_response(response)

        assert len(result) == 1
        assert result[0]["name"] == "user"

    def test_handles_empty_array(self):
        """Test handling of empty JSON array."""
        response = "[]"
        result = parse_llm_response(response)

        assert result == []

    def test_handles_non_list_json(self):
        """Test that non-list JSON returns empty list."""
        response = '{"name": "user", "description": "Single object"}'
        result = parse_llm_response(response)

        assert result == []

    def test_handles_whitespace_around_json(self):
        """Test handling of whitespace around JSON."""
        response = """

        [{"name": "user", "description": "A person"}]

        """
        result = parse_llm_response(response)

        assert len(result) == 1
        assert result[0]["name"] == "user"

    def test_filters_non_dict_items(self):
        """Test that non-dict items in array are filtered out."""
        response = '[{"name": "user", "description": "Valid"}, "string", 123, null]'
        result = parse_llm_response(response)

        assert len(result) == 1
        assert result[0]["name"] == "user"


class TestCallLLMForGlossary:
    """Tests for the LLM call function."""

    @pytest.mark.asyncio
    async def test_calls_llm_with_constructed_prompt(self):
        """Test that LLM is called with properly constructed prompt."""
        mock_client = MagicMock()
        mock_client.generate.return_value = '[{"name": "user", "description": "A person"}]'

        with patch(
            "shared.ai.glossary.ai_extractor.LLMClient",
            return_value=mock_client,
        ):
            result = await call_llm_for_glossary(
                summary="Handles user authentication.",
                label="authentication",
            )

        assert len(result) == 1
        assert result[0]["name"] == "user"
        mock_client.generate.assert_called_once()

        # Verify the prompt contains expected content
        call_args = mock_client.generate.call_args
        prompt = call_args.kwargs.get("prompt") or call_args.args[0]
        assert "Handles user authentication." in prompt
        assert "authentication" in prompt

    @pytest.mark.asyncio
    async def test_uses_config_for_llm_client(self):
        """Test that config is used to create the LLM client."""
        from shared.config import MagaldiConfig

        config = MagaldiConfig()
        config.llm.provider = "ollama"
        config.llm.url = "http://test:11434"
        config.llm.summarize_model = "test-model"

        mock_client = MagicMock()
        mock_client.generate.return_value = "[]"

        with patch(
            "shared.ai.glossary.ai_extractor.LLMClient",
            return_value=mock_client,
        ) as mock_llm_class:
            await call_llm_for_glossary(
                summary="Test summary",
                label="test",
                config=config,
            )

        # Verify LLMClient was instantiated with config values
        mock_llm_class.assert_called_once()
        call_kwargs = mock_llm_class.call_args.kwargs
        assert call_kwargs["model"] == "ollama/test-model"
        assert call_kwargs["api_base"] == "http://test:11434"

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_llm_error(self):
        """Test that LLM errors result in empty list."""
        from shared.ai.llm_client import LLMError

        mock_client = MagicMock()
        mock_client.generate.side_effect = LLMError("Connection failed")

        with patch(
            "shared.ai.glossary.ai_extractor.LLMClient",
            return_value=mock_client,
        ):
            result = await call_llm_for_glossary(
                summary="Test summary",
                label="test",
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_parses_markdown_wrapped_response(self):
        """Test that markdown-wrapped JSON response is parsed correctly."""
        mock_client = MagicMock()
        mock_client.generate.return_value = """```json
[{"name": "user", "description": "A person"}]
```"""

        with patch(
            "shared.ai.glossary.ai_extractor.LLMClient",
            return_value=mock_client,
        ):
            result = await call_llm_for_glossary(
                summary="Test summary",
                label="test",
            )

        assert len(result) == 1
        assert result[0]["name"] == "user"
