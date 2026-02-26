"""Tests for the unified LLM client module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.ai.llm_client import (
    EmbeddingClient,
    EmbeddingConfig,
    LLMClient,
    LLMConfig,
    LLMError,
    create_embedding_client_from_config,
    create_llm_client_from_config,
)

# =============================================================================
# LLM ERROR TESTS
# =============================================================================


class TestLLMError:
    """Tests for LLMError exception."""

    def test_error_inherits_from_exception(self):
        """Test that LLMError inherits from Exception."""
        assert issubclass(LLMError, Exception)

    def test_error_with_message(self):
        """Test creating error with message."""
        error = LLMError("Test error message")
        assert str(error) == "Test error message"

    def test_can_raise_and_catch(self):
        """Test raising and catching LLMError."""
        with pytest.raises(LLMError, match="test"):
            raise LLMError("test error")


# =============================================================================
# LLM CONFIG TESTS
# =============================================================================


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LLMConfig()

        assert config.model == "ollama/qwen3:4b-instruct"
        assert config.api_base is None
        assert config.api_key is None
        assert config.temperature == 0.2
        assert config.max_tokens == 512
        assert config.timeout == 180  # 3 minutes to handle queue wait with many workers
        assert config.max_retries == 3

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LLMConfig(
            model="gpt-4o-mini",
            api_key="sk-test-key",
            temperature=0.7,
            max_tokens=1024,
        )

        assert config.model == "gpt-4o-mini"
        assert config.api_key == "sk-test-key"
        assert config.temperature == 0.7
        assert config.max_tokens == 1024

    def test_from_ollama_url(self):
        """Test creating config from Ollama URL."""
        config = LLMConfig.from_ollama_url(
            url="http://localhost:11434",
            model="qwen2.5-coder:3b",
        )

        assert config.model == "ollama/qwen2.5-coder:3b"
        assert config.api_base == "http://localhost:11434"


# =============================================================================
# EMBEDDING CONFIG TESTS
# =============================================================================


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig dataclass."""

    def test_default_values(self):
        """Test default embedding configuration values."""
        config = EmbeddingConfig()

        assert config.model == "ollama/qwen3-embedding:0.6b"
        assert config.api_base is None
        assert config.api_key is None
        assert config.dimensions == 1024
        assert config.timeout == 30
        assert config.max_retries == 3

    def test_custom_values(self):
        """Test custom embedding configuration values."""
        config = EmbeddingConfig(
            model="text-embedding-3-small",
            api_key="sk-test-key",
            dimensions=1536,
        )

        assert config.model == "text-embedding-3-small"
        assert config.api_key == "sk-test-key"
        assert config.dimensions == 1536

    def test_from_ollama_url(self):
        """Test creating config from Ollama URL."""
        config = EmbeddingConfig.from_ollama_url(
            url="http://localhost:11434",
            model="qwen3-embedding:0.6b",
            dimensions=1024,
        )

        assert config.model == "ollama/qwen3-embedding:0.6b"
        assert config.api_base == "http://localhost:11434"
        assert config.dimensions == 1024


# =============================================================================
# LLM CLIENT TESTS
# =============================================================================


class TestLLMClient:
    """Tests for LLMClient class."""

    def test_init_with_model_only(self):
        """Test initialization with model only."""
        client = LLMClient(model="gpt-4o-mini")

        assert client.model == "gpt-4o-mini"
        assert client.api_base is None
        assert client.api_key is None
        assert client.provider == "openai"

    def test_init_with_ollama_model(self):
        """Test initialization with Ollama model."""
        client = LLMClient(
            model="ollama/qwen2.5-coder:3b",
            api_base="http://localhost:11434",
        )

        assert client.model == "ollama/qwen2.5-coder:3b"
        assert client.api_base == "http://localhost:11434"
        assert client.provider == "ollama"

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        client = LLMClient(
            model="gpt-4o-mini",
            api_key="sk-test-key",
        )

        assert client.api_key == "sk-test-key"

    def test_from_config(self):
        """Test creating client from config."""
        config = LLMConfig(
            model="ollama/test-model",
            api_base="http://localhost:11434",
        )

        client = LLMClient.from_config(config)

        assert client.model == "ollama/test-model"
        assert client.api_base == "http://localhost:11434"

    def test_from_ollama(self):
        """Test creating client for Ollama provider."""
        client = LLMClient.from_ollama(
            url="http://localhost:11434",
            model="qwen2.5-coder:3b",
        )

        assert client.model == "ollama/qwen2.5-coder:3b"
        assert client.api_base == "http://localhost:11434"

    def test_generate_success(self):
        """Test successful text generation."""
        client = LLMClient(model="test-model")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated text response"

        with patch("shared.ai.llm_client.completion", return_value=mock_response):
            result = client.generate("Test prompt")

        assert result == "Generated text response"

    def test_generate_strips_whitespace(self):
        """Test that generated text is stripped."""
        client = LLMClient(model="test-model")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "  Generated text  "

        with patch("shared.ai.llm_client.completion", return_value=mock_response):
            result = client.generate("Test prompt")

        assert result == "Generated text"

    def test_generate_with_custom_params(self):
        """Test generation with custom parameters."""
        client = LLMClient(model="test-model")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"

        with patch("shared.ai.llm_client.completion", return_value=mock_response) as mock_comp:
            client.generate(
                prompt="Test",
                temperature=0.8,
                max_tokens=1000,
                timeout=120,
            )

        call_kwargs = mock_comp.call_args.kwargs
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["max_tokens"] == 1000
        assert call_kwargs["timeout"] == 120

    def test_generate_with_model_override(self):
        """Test generation with model override."""
        client = LLMClient(model="default-model")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"

        with patch("shared.ai.llm_client.completion", return_value=mock_response) as mock_comp:
            client.generate("Test", model="override-model")

        call_kwargs = mock_comp.call_args.kwargs
        assert call_kwargs["model"] == "override-model"

    def test_generate_includes_api_base(self):
        """Test that api_base is included in request."""
        client = LLMClient(
            model="ollama/test-model",
            api_base="http://localhost:11434",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"

        with patch("shared.ai.llm_client.completion", return_value=mock_response) as mock_comp:
            client.generate("Test")

        call_kwargs = mock_comp.call_args.kwargs
        assert call_kwargs["api_base"] == "http://localhost:11434"

    def test_generate_includes_api_key(self):
        """Test that api_key is included in request."""
        client = LLMClient(
            model="gpt-4o-mini",
            api_key="sk-test-key",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"

        with patch("shared.ai.llm_client.completion", return_value=mock_response) as mock_comp:
            client.generate("Test")

        call_kwargs = mock_comp.call_args.kwargs
        assert call_kwargs["api_key"] == "sk-test-key"

    def test_generate_raises_on_empty_response(self):
        """Test that LLMError is raised on empty response."""
        client = LLMClient(model="test-model")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        with patch("shared.ai.llm_client.completion", return_value=mock_response), pytest.raises(LLMError, match="Empty response"):
            client.generate("Test")

    def test_generate_raises_on_exception(self):
        """Test that LLMError is raised on exception."""
        client = LLMClient(model="test-model", max_retries=0)

        with patch("shared.ai.llm_client.completion", side_effect=Exception("Connection error")), pytest.raises(LLMError, match="LLM generation"):
            client.generate("Test")

    def test_verify_model_returns_true_on_success(self):
        """Test verify_model returns True when model is available."""
        client = LLMClient(model="test-model")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"

        with patch("shared.ai.llm_client.completion", return_value=mock_response):
            assert client.verify_model() is True

    def test_verify_model_returns_false_on_failure(self):
        """Test verify_model returns False when model is unavailable."""
        client = LLMClient(model="test-model")

        with patch("shared.ai.llm_client.completion", side_effect=Exception("Model not found")):
            assert client.verify_model() is False


# =============================================================================
# EMBEDDING CLIENT TESTS
# =============================================================================


class TestEmbeddingClient:
    """Tests for EmbeddingClient class."""

    def test_init_with_model_only(self):
        """Test initialization with model only."""
        client = EmbeddingClient(model="text-embedding-3-small")

        assert client.model == "text-embedding-3-small"
        assert client.api_base is None
        assert client.api_key is None
        assert client.dimensions == 1024
        assert client.provider == "openai"

    def test_init_with_ollama_model(self):
        """Test initialization with Ollama model."""
        client = EmbeddingClient(
            model="ollama/qwen3-embedding:0.6b",
            api_base="http://localhost:11434",
        )

        assert client.model == "ollama/qwen3-embedding:0.6b"
        assert client.api_base == "http://localhost:11434"
        assert client.provider == "ollama"

    def test_from_config(self):
        """Test creating client from config."""
        config = EmbeddingConfig(
            model="ollama/test-model",
            api_base="http://localhost:11434",
            dimensions=768,
        )

        client = EmbeddingClient.from_config(config)

        assert client.model == "ollama/test-model"
        assert client.api_base == "http://localhost:11434"
        assert client.dimensions == 768

    def test_from_ollama(self):
        """Test creating client for Ollama provider."""
        client = EmbeddingClient.from_ollama(
            url="http://localhost:11434",
            model="qwen3-embedding:0.6b",
            dimensions=1024,
        )

        assert client.model == "ollama/qwen3-embedding:0.6b"
        assert client.api_base == "http://localhost:11434"
        assert client.dimensions == 1024

    def test_embed_success(self):
        """Test successful embedding generation."""
        client = EmbeddingClient(model="test-model")

        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1, 0.2, 0.3]}]

        with patch("shared.ai.llm_client.embedding", return_value=mock_response):
            result = client.embed("Test text")

        assert result == [0.1, 0.2, 0.3]

    def test_embed_includes_api_base(self):
        """Test that api_base is included in request."""
        client = EmbeddingClient(
            model="ollama/test-model",
            api_base="http://localhost:11434",
        )

        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1]}]

        with patch("shared.ai.llm_client.embedding", return_value=mock_response) as mock_embed:
            client.embed("Test")

        call_kwargs = mock_embed.call_args.kwargs
        assert call_kwargs["api_base"] == "http://localhost:11434"

    def test_embed_includes_api_key(self):
        """Test that api_key is included in request."""
        client = EmbeddingClient(
            model="text-embedding-3-small",
            api_key="sk-test-key",
        )

        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1]}]

        with patch("shared.ai.llm_client.embedding", return_value=mock_response) as mock_embed:
            client.embed("Test")

        call_kwargs = mock_embed.call_args.kwargs
        assert call_kwargs["api_key"] == "sk-test-key"

    def test_embed_raises_on_empty_response(self):
        """Test that LLMError is raised on empty response."""
        client = EmbeddingClient(model="test-model")

        mock_response = MagicMock()
        mock_response.data = []

        with patch("shared.ai.llm_client.embedding", return_value=mock_response), pytest.raises(LLMError, match="No embedding returned"):
            client.embed("Test")

    def test_embed_raises_on_exception(self):
        """Test that LLMError is raised on exception."""
        client = EmbeddingClient(model="test-model", max_retries=0)

        with patch("shared.ai.llm_client.embedding", side_effect=Exception("Connection error")), pytest.raises(LLMError, match="Embedding generation"):
            client.embed("Test")

    @pytest.mark.asyncio
    async def test_embed_async_success(self):
        """Test successful async embedding generation."""
        client = EmbeddingClient(model="test-model")

        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1, 0.2, 0.3]}]

        with patch("shared.ai.llm_client.aembedding", new_callable=AsyncMock) as mock_aembed:
            mock_aembed.return_value = mock_response
            result = await client.embed_async("Test text")

        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_async_includes_api_base(self):
        """Test that api_base is included in async request."""
        client = EmbeddingClient(
            model="ollama/test-model",
            api_base="http://localhost:11434",
        )

        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1]}]

        with patch("shared.ai.llm_client.aembedding", new_callable=AsyncMock) as mock_aembed:
            mock_aembed.return_value = mock_response
            await client.embed_async("Test")

        call_kwargs = mock_aembed.call_args.kwargs
        assert call_kwargs["api_base"] == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_embed_async_raises_on_empty_response(self):
        """Test that LLMError is raised on empty async response."""
        client = EmbeddingClient(model="test-model")

        mock_response = MagicMock()
        mock_response.data = []

        with patch("shared.ai.llm_client.aembedding", new_callable=AsyncMock) as mock_aembed:
            mock_aembed.return_value = mock_response
            with pytest.raises(LLMError, match="No embedding returned"):
                await client.embed_async("Test")

    @pytest.mark.asyncio
    async def test_embed_async_raises_on_exception(self):
        """Test that LLMError is raised on async exception."""
        client = EmbeddingClient(model="test-model", max_retries=0)

        with patch("shared.ai.llm_client.aembedding", new_callable=AsyncMock) as mock_aembed:
            mock_aembed.side_effect = Exception("Connection error")
            with pytest.raises(LLMError, match="Embedding generation"):
                await client.embed_async("Test")

    def test_embed_batch_success(self):
        """Test successful batch embedding generation."""
        client = EmbeddingClient(model="test-model")

        mock_response = MagicMock()
        mock_response.data = [
            {"embedding": [0.1, 0.2]},
            {"embedding": [0.3, 0.4]},
        ]

        with patch("shared.ai.llm_client.embedding", return_value=mock_response):
            result = client.embed_batch(["Text 1", "Text 2"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_embed_batch_empty_list(self):
        """Test batch embedding with empty list."""
        client = EmbeddingClient(model="test-model")

        result = client.embed_batch([])

        assert result == []

    def test_embed_batch_includes_api_base(self):
        """Test that api_base is included in batch request."""
        client = EmbeddingClient(
            model="ollama/test-model",
            api_base="http://localhost:11434",
        )

        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1]}]

        with patch("shared.ai.llm_client.embedding", return_value=mock_response) as mock_embed:
            client.embed_batch(["Test"])

        call_kwargs = mock_embed.call_args.kwargs
        assert call_kwargs["api_base"] == "http://localhost:11434"

    def test_embed_batch_raises_on_exception(self):
        """Test that LLMError is raised on batch exception."""
        client = EmbeddingClient(model="test-model", max_retries=0)

        with patch("shared.ai.llm_client.embedding", side_effect=Exception("Connection error")), pytest.raises(LLMError, match="Batch embedding generation"):
            client.embed_batch(["Test"])

    def test_verify_model_returns_true_on_success(self):
        """Test verify_model returns True when model is available."""
        client = EmbeddingClient(model="test-model")

        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1]}]

        with patch("shared.ai.llm_client.embedding", return_value=mock_response):
            assert client.verify_model() is True

    def test_verify_model_returns_false_on_failure(self):
        """Test verify_model returns False when model is unavailable."""
        client = EmbeddingClient(model="test-model")

        with patch("shared.ai.llm_client.embedding", side_effect=Exception("Model not found")):
            assert client.verify_model() is False


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_llm_client_from_config(self):
        """Test creating LLM client from legacy config."""
        client = create_llm_client_from_config(
            ollama_url="http://localhost:11434",
            model="qwen2.5-coder:3b",
        )

        assert isinstance(client, LLMClient)
        assert client.model == "ollama/qwen2.5-coder:3b"
        assert client.api_base == "http://localhost:11434"

    def test_create_embedding_client_from_config(self):
        """Test creating embedding client from legacy config."""
        client = create_embedding_client_from_config(
            ollama_url="http://localhost:11434",
            model="qwen3-embedding:0.6b",
            dimensions=1024,
        )

        assert isinstance(client, EmbeddingClient)
        assert client.model == "ollama/qwen3-embedding:0.6b"
        assert client.api_base == "http://localhost:11434"
        assert client.dimensions == 1024

    def test_create_embedding_client_default_dimensions(self):
        """Test creating embedding client with default dimensions."""
        client = create_embedding_client_from_config(
            ollama_url="http://localhost:11434",
            model="qwen3-embedding:0.6b",
        )

        assert client.dimensions == 1024


# =============================================================================
# NUM_CTX PARAMETER TESTS
# =============================================================================


class TestNumCtxParameter:
    """Tests for num_ctx parameter support."""

    def test_generate_passes_num_ctx_for_ollama(self):
        """Should pass num_ctx to LiteLLM for Ollama provider."""
        client = LLMClient(model="ollama/qwen3:4b", api_base="http://localhost:11434")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"

        with patch("shared.ai.llm_client.completion", return_value=mock_response) as mock_comp:
            client.generate("test prompt", num_ctx=4096)

        call_kwargs = mock_comp.call_args.kwargs
        assert call_kwargs.get("num_ctx") == 4096

    def test_generate_passes_num_ctx_for_llamacpp(self):
        """Should pass n_ctx via extra_body for llama.cpp provider."""
        client = LLMClient(model="openai/qwen3:4b", api_base="http://localhost:8080/v1")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"

        with patch("shared.ai.llm_client.completion", return_value=mock_response) as mock_comp:
            client.generate("test prompt", num_ctx=4096)

        call_kwargs = mock_comp.call_args.kwargs
        assert call_kwargs.get("extra_body", {}).get("n_ctx") == 4096

    def test_generate_from_messages_passes_num_ctx(self):
        """Should pass num_ctx in generate_from_messages."""
        client = LLMClient(model="ollama/qwen3:4b", api_base="http://localhost:11434")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"

        with patch("shared.ai.llm_client.completion", return_value=mock_response) as mock_comp:
            client.generate_from_messages(
                [{"role": "user", "content": "test"}],
                num_ctx=8192
            )

        call_kwargs = mock_comp.call_args.kwargs
        assert call_kwargs.get("num_ctx") == 8192

    def test_generate_without_num_ctx_does_not_include_it(self):
        """Should not include num_ctx if not provided."""
        client = LLMClient(model="ollama/qwen3:4b", api_base="http://localhost:11434")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"

        with patch("shared.ai.llm_client.completion", return_value=mock_response) as mock_comp:
            client.generate("test prompt")

        call_kwargs = mock_comp.call_args.kwargs
        assert "num_ctx" not in call_kwargs
