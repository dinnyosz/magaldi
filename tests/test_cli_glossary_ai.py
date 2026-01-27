"""Tests for glossary extraction CLI function."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shared.ai.glossary.ai_extractor import GlossaryItem


class TestRunGlossaryExtraction:
    """Tests for run_glossary_extraction function (AI-powered).

    Note: run_glossary_extraction is now synchronous (uses concurrent threading internally).
    """

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = MagicMock()
        config.llm.url = "http://localhost:11434"
        config.llm.summarize_model = "qwen3:4b-instruct"
        config.llm.summarize_temperature = 0.7
        config.llm.summarize_top_p = 0.9
        config.llm.summarize_max_tokens = 512
        config.llm.provider = "ollama"
        config.llm.api_key = None
        return config

    @pytest.fixture
    def mock_es_repo(self):
        """Create a mock Elasticsearch repository."""
        repo = MagicMock()
        repo.get_features.return_value = [
            {
                "feature_id": "test:repo:main:feature:1",
                "label": "User Authentication",
                "summary": "Handles user login and registration",
            },
            {
                "feature_id": "test:repo:main:feature:2",
                "label": "Email Notifications",
                "summary": "Sends email notifications to users",
            },
        ]
        repo.get_subfeatures.return_value = [
            {
                "subfeature_id": "test:repo:main:subfeature:1",
                "label": "Password Reset",
                "summary": "Allows users to reset their password",
            },
        ]
        repo.delete_glossary.return_value = 0
        repo.index_glossary.return_value = True
        return repo

    @pytest.fixture
    def sample_glossary_items(self):
        """Create sample glossary items returned by AI extractor."""
        return [
            GlossaryItem(
                name="user",
                description="A person who interacts with the system",
                source_feature_id="test:repo:main:feature:1",
                source_feature_ids=["test:repo:main:feature:1", "test:repo:main:feature:2"],
            ),
            GlossaryItem(
                name="email",
                description="Electronic message sent to users",
                source_feature_id="test:repo:main:feature:2",
            ),
            GlossaryItem(
                name="password",
                description="Secret credentials for authentication",
                source_feature_id="test:repo:main:subfeature:1",
            ),
        ]

    @patch("shared.ai.glossary.ai_extractor.extract_glossary_from_features_concurrent")
    def test_extracts_glossary_from_features_and_subfeatures(
        self,
        mock_extract,
        mock_config,
        mock_es_repo,
        sample_glossary_items,
    ):
        """Test that features and subfeatures are fetched and combined."""
        from shared.cli import run_glossary_extraction

        mock_extract.return_value = sample_glossary_items

        result = run_glossary_extraction(
            scope="test",
            repository="repo",
            username="main",
            config=mock_config,
            es_repo=mock_es_repo,
        )

        # Verify features and subfeatures were fetched
        mock_es_repo.get_features.assert_called_once_with("test", "repo", "main")
        mock_es_repo.get_subfeatures.assert_called_once_with("test", "repo", "main")

        # Verify extract was called with combined list (2 features + 1 subfeature)
        mock_extract.assert_called_once()

        # Verify result
        assert result is not None
        assert result["terms_count"] == 3
        assert "user" in result["terms"]
        assert "email" in result["terms"]
        assert "password" in result["terms"]

    @patch("shared.ai.glossary.ai_extractor.extract_glossary_from_features_concurrent")
    def test_passes_indexing_params_to_extractor(
        self,
        mock_extract,
        mock_config,
        mock_es_repo,
        sample_glossary_items,
    ):
        """Test that incremental indexing params are passed to the extractor."""
        from shared.cli import run_glossary_extraction

        mock_extract.return_value = sample_glossary_items

        run_glossary_extraction(
            scope="test",
            repository="repo",
            username="main",
            config=mock_config,
            es_repo=mock_es_repo,
        )

        # Verify extractor was called with incremental indexing params
        mock_extract.assert_called_once()
        call_kwargs = mock_extract.call_args.kwargs

        # Indexing now happens inside the extractor, so verify params are passed
        assert call_kwargs["es_repo"] == mock_es_repo
        assert call_kwargs["scope"] == "test"
        assert call_kwargs["repository"] == "repo"
        assert call_kwargs["username"] == "main"

    @patch("shared.ai.glossary.ai_extractor.extract_glossary_from_features_concurrent")
    def test_returns_none_when_no_features(
        self,
        mock_extract,
        mock_config,
        mock_es_repo,
    ):
        """Test that None is returned when no features exist."""
        from shared.cli import run_glossary_extraction

        mock_es_repo.get_features.return_value = []
        mock_es_repo.get_subfeatures.return_value = []

        result = run_glossary_extraction(
            scope="test",
            repository="repo",
            username="main",
            config=mock_config,
            es_repo=mock_es_repo,
        )

        assert result is None
        mock_extract.assert_not_called()
        mock_es_repo.index_glossary.assert_not_called()

    @patch("shared.ai.glossary.ai_extractor.extract_glossary_from_features_concurrent")
    def test_returns_zero_terms_when_ai_extracts_nothing(
        self,
        mock_extract,
        mock_config,
        mock_es_repo,
    ):
        """Test handling when AI extraction returns no glossary items."""
        from shared.cli import run_glossary_extraction

        mock_extract.return_value = []

        result = run_glossary_extraction(
            scope="test",
            repository="repo",
            username="main",
            config=mock_config,
            es_repo=mock_es_repo,
        )

        assert result["terms_count"] == 0
        mock_es_repo.index_glossary.assert_not_called()

    @patch("shared.ai.glossary.ai_extractor.extract_glossary_from_features_concurrent")
    def test_deletes_existing_glossary_before_indexing(
        self,
        mock_extract,
        mock_config,
        mock_es_repo,
        sample_glossary_items,
    ):
        """Test that existing glossary is deleted before new entries are indexed."""
        from shared.cli import run_glossary_extraction

        mock_extract.return_value = sample_glossary_items
        mock_es_repo.delete_glossary.return_value = 5  # Simulating 5 deleted entries

        run_glossary_extraction(
            scope="test",
            repository="repo",
            username="main",
            config=mock_config,
            es_repo=mock_es_repo,
        )

        mock_es_repo.delete_glossary.assert_called_once_with("test", "repo", "main")

    @patch("shared.ai.glossary.ai_extractor.extract_glossary_from_features_concurrent")
    @patch("shared.db.elasticsearch.ElasticsearchRepository")
    def test_creates_es_repo_if_not_provided(
        self,
        mock_es_repo_class,
        _mock_extract,
        mock_config,
    ):
        """Test that ElasticsearchRepository is created when not provided."""
        from shared.cli import run_glossary_extraction

        mock_es_instance = MagicMock()
        mock_es_instance.get_features.return_value = []
        mock_es_instance.get_subfeatures.return_value = []
        mock_es_repo_class.return_value = mock_es_instance

        run_glossary_extraction(
            scope="test",
            repository="repo",
            username="main",
            config=mock_config,
            es_repo=None,
        )

        mock_es_repo_class.assert_called_once_with(mock_config)
        mock_es_instance.close.assert_called_once()

    @patch("shared.ai.glossary.ai_extractor.extract_glossary_from_features_concurrent")
    def test_does_not_close_provided_es_repo(
        self,
        _mock_extract,
        mock_config,
        mock_es_repo,
    ):
        """Test that provided ES repo is not closed."""
        from shared.cli import run_glossary_extraction

        mock_es_repo.get_features.return_value = []
        mock_es_repo.get_subfeatures.return_value = []

        run_glossary_extraction(
            scope="test",
            repository="repo",
            username="main",
            config=mock_config,
            es_repo=mock_es_repo,
        )

        mock_es_repo.close.assert_not_called()
