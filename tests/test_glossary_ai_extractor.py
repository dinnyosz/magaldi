"""Tests for AI-powered glossary extraction."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.ai.glossary.ai_extractor import (
    GlossaryItem,
    build_glossary_prompt,
    call_llm_for_glossary,
    extract_glossary_from_feature,
    extract_glossary_from_features,
    merge_glossary_items,
    normalize_term,
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

        # New format: LLM returns list of term names only
        mock_response = ["user", "login", "registration"]

        with patch(
            "shared.ai.glossary.ai_extractor.call_llm_for_glossary",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await extract_glossary_from_feature(feature)

        assert len(result) == 3
        assert all(isinstance(item, GlossaryItem) for item in result)
        assert result[0].name == "user"
        # Description is empty in Phase 1; will be generated in Phase 2
        assert result[0].description == ""
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
        """Test that empty term names are filtered out."""
        feature = {
            "feature_id": "scope:repo:main:feature:auth",
            "label": "authentication",
            "summary": "Handles user login.",
        }

        # New format: list of strings with some empty/invalid entries
        mock_response = ["user", "", "  "]  # Empty strings should be filtered

        with patch(
            "shared.ai.glossary.ai_extractor.call_llm_for_glossary",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await extract_glossary_from_feature(feature)

        assert len(result) == 1
        assert result[0].name == "user"

    @pytest.mark.asyncio
    async def test_handles_valid_terms_with_no_descriptions(self):
        """Test that valid terms are extracted even without descriptions (Phase 1 behavior)."""
        feature = {
            "feature_id": "scope:repo:main:feature:auth",
            "label": "authentication",
            "summary": "Handles user login.",
        }

        # New format: terms are just strings, descriptions come in Phase 2
        mock_response = ["user", "orphan", "empty_desc"]

        with patch(
            "shared.ai.glossary.ai_extractor.call_llm_for_glossary",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await extract_glossary_from_feature(feature)

        # All 3 terms should be valid (descriptions are generated in Phase 2)
        assert len(result) == 3
        assert result[0].name == "user"
        assert result[0].description == ""  # Empty until Phase 2


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

        # Should have example JSON array to guide the model (new format: ["term1", "term2"])
        assert '["term1"' in prompt or '"term"' in prompt.lower()


class TestParseLLMResponse:
    """Tests for parsing LLM response.

    Note: The new format returns list[str] (term names), not list[dict].
    Legacy dict format with {"name": ..., "description": ...} is also supported
    for backward compatibility.
    """

    def test_parses_valid_json_array_of_strings(self):
        """Test parsing valid JSON array of strings (new format)."""
        response = '["user", "login", "registration"]'
        result = parse_llm_response(response)

        assert len(result) == 3
        assert result[0] == "user"
        assert result[1] == "login"
        assert result[2] == "registration"

    def test_parses_legacy_dict_format(self):
        """Test parsing legacy dict format for backward compatibility."""
        response = '[{"name": "user", "description": "A person"}]'
        result = parse_llm_response(response)

        assert len(result) == 1
        assert result[0] == "user"

    def test_parses_json_in_markdown_code_block(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        response = """```json
["user", "login"]
```"""
        result = parse_llm_response(response)

        assert len(result) == 2
        assert result[0] == "user"

    def test_parses_json_in_markdown_code_block_without_lang(self):
        """Test parsing JSON in code block without language specifier."""
        response = """```
["user", "login"]
```"""
        result = parse_llm_response(response)

        assert len(result) == 2
        assert result[0] == "user"

    def test_returns_empty_list_for_invalid_json(self):
        """Test that invalid JSON returns empty list."""
        response = "This is not JSON at all"
        result = parse_llm_response(response)

        assert result == []

    def test_filters_empty_strings(self):
        """Test that empty strings are returned but will be filtered in extract phase."""
        response = '["user", "", "login"]'
        result = parse_llm_response(response)

        # parse_llm_response returns all strings; filtering is done in extract phase
        assert len(result) == 3
        assert result[0] == "user"
        assert result[1] == ""
        assert result[2] == "login"

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

        ["user", "login"]

        """
        result = parse_llm_response(response)

        assert len(result) == 2
        assert result[0] == "user"

    def test_extracts_names_from_mixed_format(self):
        """Test that mixed formats are handled (strings and dicts with name)."""
        response = '["user", {"name": "login"}, "register"]'
        result = parse_llm_response(response)

        assert len(result) == 3
        assert result[0] == "user"
        assert result[1] == "login"
        assert result[2] == "register"


class TestCallLLMForGlossary:
    """Tests for the LLM call function."""

    @pytest.mark.asyncio
    async def test_calls_llm_with_constructed_prompt(self):
        """Test that LLM is called with properly constructed messages."""
        mock_client = MagicMock()
        mock_client.generate_from_messages.return_value = '["user", "login"]'

        with patch(
            "shared.ai.glossary.ai_extractor.LLMClient",
            return_value=mock_client,
        ):
            result = await call_llm_for_glossary(
                summary="Handles user authentication.",
                label="authentication",
            )

        assert len(result) == 2
        assert result[0] == "user"
        assert result[1] == "login"
        mock_client.generate_from_messages.assert_called_once()

        # Verify the user message contains expected content
        call_args = mock_client.generate_from_messages.call_args
        messages = call_args.kwargs.get("messages")
        user_content = messages[1]["content"]  # User message is second
        assert "Handles user authentication." in user_content
        assert "authentication" in user_content

    @pytest.mark.asyncio
    async def test_uses_config_for_llm_client(self):
        """Test that config is used to create the LLM client."""
        from shared.config import MagaldiConfig, ModelConfig

        config = MagaldiConfig()
        # Add a test model to the models dict
        config.llm.models["test-model"] = ModelConfig(
            name="test-model-name",
            provider="ollama",
            url="http://test:11434",
        )
        config.llm.summarize_model = "test-model"

        mock_client = MagicMock()
        mock_client.generate_from_messages.return_value = "[]"

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
        assert call_kwargs["model"] == "ollama/test-model-name"
        assert call_kwargs["api_base"] == "http://test:11434"

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_llm_error(self):
        """Test that LLM errors result in empty list."""
        from shared.ai.llm_client import LLMError

        mock_client = MagicMock()
        mock_client.generate_from_messages.side_effect = LLMError("Connection failed")

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
        mock_client.generate_from_messages.return_value = """```json
["user", "login"]
```"""

        with patch(
            "shared.ai.glossary.ai_extractor.LLMClient",
            return_value=mock_client,
        ):
            result = await call_llm_for_glossary(
                summary="Test summary",
                label="test",
            )

        assert len(result) == 2
        assert result[0] == "user"
        assert result[1] == "login"


class TestNormalizeTerm:
    """Tests for term normalization."""

    def test_normalizes_simple_plural(self):
        """Test that simple plurals (ending in 's') are normalized."""
        assert normalize_term("users") == "user"
        assert normalize_term("emails") == "email"
        assert normalize_term("accounts") == "account"

    def test_normalizes_ies_plural(self):
        """Test that -ies plurals are normalized to -y."""
        assert normalize_term("entries") == "entry"
        assert normalize_term("categories") == "category"
        assert normalize_term("queries") == "query"

    def test_normalizes_es_plural(self):
        """Test that -es plurals are normalized correctly."""
        assert normalize_term("classes") == "class"
        assert normalize_term("processes") == "process"
        assert normalize_term("boxes") == "box"
        assert normalize_term("matches") == "match"
        assert normalize_term("flashes") == "flash"

    def test_preserves_short_words(self):
        """Test that short words are preserved."""
        assert normalize_term("as") == "as"
        assert normalize_term("is") == "is"

    def test_handles_lowercase_and_whitespace(self):
        """Test that terms are lowercased and trimmed."""
        assert normalize_term("User") == "user"
        assert normalize_term("  USER  ") == "user"
        assert normalize_term("Users") == "user"

    def test_preserves_already_singular(self):
        """Test that singular terms are preserved."""
        assert normalize_term("user") == "user"
        assert normalize_term("email") == "email"
        assert normalize_term("authentication") == "authentication"

    def test_handles_double_s_words(self):
        """Test that words ending in 'ss' are not depluralized."""
        assert normalize_term("class") == "class"
        assert normalize_term("process") == "process"
        assert normalize_term("access") == "access"

    def test_preserves_latin_origin_words_ending_in_us(self):
        """Test that Latin words ending in 'us' are not depluralized."""
        assert normalize_term("status") == "status"
        assert normalize_term("bonus") == "bonus"
        assert normalize_term("cactus") == "cactus"
        assert normalize_term("radius") == "radius"
        assert normalize_term("corpus") == "corpus"

    def test_preserves_latin_origin_words_ending_in_is(self):
        """Test that Latin words ending in 'is' are not depluralized."""
        assert normalize_term("analysis") == "analysis"
        assert normalize_term("basis") == "basis"
        assert normalize_term("thesis") == "thesis"
        assert normalize_term("crisis") == "crisis"


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

    def test_combines_feature_ids_preserving_order(self):
        """Test that feature IDs are combined in order of appearance."""
        items = [
            GlossaryItem(name="user", description="Short", source_feature_id="f1"),
            GlossaryItem(name="user", description="Medium desc", source_feature_id="f2"),
            GlossaryItem(name="user", description="Another desc", source_feature_id="f3"),
        ]

        merged = merge_glossary_items(items)

        assert len(merged) == 1
        assert merged[0].source_feature_ids == ["f1", "f2", "f3"]

    def test_does_not_duplicate_feature_ids(self):
        """Test that duplicate feature IDs are not added twice."""
        items = [
            GlossaryItem(name="user", description="Desc 1", source_feature_id="f1"),
            GlossaryItem(name="user", description="Desc 2", source_feature_id="f1"),
        ]

        merged = merge_glossary_items(items)

        assert len(merged) == 1
        assert merged[0].source_feature_ids == ["f1"]

    def test_returns_sorted_by_name(self):
        """Test that merged items are sorted alphabetically by name."""
        items = [
            GlossaryItem(name="zebra", description="An animal", source_feature_id="f1"),
            GlossaryItem(name="apple", description="A fruit", source_feature_id="f2"),
            GlossaryItem(name="banana", description="Another fruit", source_feature_id="f3"),
        ]

        merged = merge_glossary_items(items)

        assert len(merged) == 3
        assert [item.name for item in merged] == ["apple", "banana", "zebra"]

    def test_handles_empty_list(self):
        """Test that empty input returns empty output."""
        merged = merge_glossary_items([])
        assert merged == []

    def test_merges_ies_plural_variations(self):
        """Test merging items with -ies/-y variations."""
        items = [
            GlossaryItem(name="entry", description="A single entry", source_feature_id="f1"),
            GlossaryItem(name="entries", description="Multiple entries in the system", source_feature_id="f2"),
        ]

        merged = merge_glossary_items(items)

        assert len(merged) == 1
        assert merged[0].name == "entry"
        assert set(merged[0].source_feature_ids) == {"f1", "f2"}
        # Should keep the longer description
        assert "Multiple entries in the system" in merged[0].description

    def test_handles_single_item(self):
        """Test that a single item is returned unchanged (but normalized)."""
        items = [
            GlossaryItem(
                name="Users",
                description="People who use the system",
                source_feature_id="f1",
            ),
        ]

        merged = merge_glossary_items(items)

        assert len(merged) == 1
        assert merged[0].name == "user"  # Normalized
        assert merged[0].description == "People who use the system"
        assert merged[0].source_feature_ids == ["f1"]


class TestExtractGlossaryFromFeatures:
    """Tests for the full extraction pipeline."""

    @pytest.mark.asyncio
    async def test_processes_multiple_features(self):
        """Test processing multiple features and merging results."""
        features = [
            {"feature_id": "f1", "label": "authentication", "summary": "Handles user login."},
            {"feature_id": "f2", "label": "registration", "summary": "Manages user registration."},
        ]

        # New format: lists of term names
        mock_responses = [
            ["user"],
            ["user", "registration"],
        ]

        call_count = 0

        async def mock_call(*_args, **_kwargs):
            nonlocal call_count
            result = mock_responses[call_count]
            call_count += 1
            return result

        with patch(
            "shared.ai.glossary.ai_extractor.call_llm_for_glossary", side_effect=mock_call
        ):
            result = await extract_glossary_from_features(features)

        # "user" appears in both, should be merged
        assert len(result) == 2
        user_item = next(i for i in result if i.name == "user")
        assert set(user_item.source_feature_ids) == {"f1", "f2"}

    @pytest.mark.asyncio
    async def test_handles_empty_features_list(self):
        """Test that empty features list returns empty result."""
        result = await extract_glossary_from_features([])
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_features_without_summary(self):
        """Test that features without summary are skipped (no LLM call)."""
        features = [
            {"feature_id": "f1", "label": "test", "summary": ""},
            {"feature_id": "f2", "label": "real", "summary": "Real feature."},
        ]

        with patch(
            "shared.ai.glossary.ai_extractor.call_llm_for_glossary",
            new_callable=AsyncMock,
            return_value=["item"],
        ) as mock_call:
            await extract_glossary_from_features(features)

        # Should only be called once (for f2)
        assert mock_call.call_count == 1

    @pytest.mark.asyncio
    async def test_calls_progress_callback(self):
        """Test that progress callback is called for each feature."""
        features = [
            {"feature_id": "f1", "label": "a", "summary": "Summary 1"},
            {"feature_id": "f2", "label": "b", "summary": "Summary 2"},
        ]

        progress_calls = []

        def callback(current, total):
            progress_calls.append((current, total))

        with patch(
            "shared.ai.glossary.ai_extractor.call_llm_for_glossary",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await extract_glossary_from_features(features, progress_callback=callback)

        assert progress_calls == [(1, 2), (2, 2)]

    @pytest.mark.asyncio
    async def test_merges_items_across_features(self):
        """Test that items from different features are merged correctly."""
        features = [
            {"feature_id": "f1", "label": "auth", "summary": "Auth feature"},
            {"feature_id": "f2", "label": "profile", "summary": "Profile feature"},
            {"feature_id": "f3", "label": "settings", "summary": "Settings feature"},
        ]

        # New format: lists of term names
        mock_responses = [
            ["user"],
            ["user"],
            ["settings"],
        ]

        call_count = 0

        async def mock_call(*_args, **_kwargs):
            nonlocal call_count
            result = mock_responses[call_count]
            call_count += 1
            return result

        with patch(
            "shared.ai.glossary.ai_extractor.call_llm_for_glossary", side_effect=mock_call
        ):
            result = await extract_glossary_from_features(features)

        assert len(result) == 2  # "user" merged, "settings" separate

        user_item = next(i for i in result if i.name == "user")
        assert set(user_item.source_feature_ids) == {"f1", "f2"}

        settings_item = next(i for i in result if i.name == "setting")  # normalized
        assert settings_item.source_feature_ids == ["f3"]

    @pytest.mark.asyncio
    async def test_handles_llm_returning_empty_for_some_features(self):
        """Test that features with no glossary items are handled gracefully."""
        features = [
            {"feature_id": "f1", "label": "auth", "summary": "Has terms"},
            {"feature_id": "f2", "label": "utils", "summary": "No domain terms"},
        ]

        # New format: lists of term names
        mock_responses = [
            ["user"],
            [],  # No items for utils
        ]

        call_count = 0

        async def mock_call(*_args, **_kwargs):
            nonlocal call_count
            result = mock_responses[call_count]
            call_count += 1
            return result

        with patch(
            "shared.ai.glossary.ai_extractor.call_llm_for_glossary", side_effect=mock_call
        ):
            result = await extract_glossary_from_features(features)

        assert len(result) == 1
        assert result[0].name == "user"

    @pytest.mark.asyncio
    async def test_passes_config_to_extract_function(self):
        """Test that config is passed through to the extraction function."""
        from shared.config import MagaldiConfig

        config = MagaldiConfig()
        features = [
            {"feature_id": "f1", "label": "test", "summary": "Test feature"},
        ]

        with patch(
            "shared.ai.glossary.ai_extractor.extract_glossary_from_feature",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_extract:
            await extract_glossary_from_features(features, config=config)

        mock_extract.assert_called_once_with(features[0], config)
