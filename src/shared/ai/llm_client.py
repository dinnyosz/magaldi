"""Unified LLM client using LiteLLM for provider abstraction.

This module provides a unified interface for text generation and embeddings
that works with multiple LLM providers (Ollama, OpenAI, Anthropic, etc.).

Usage:
    from shared.ai.llm_client import LLMClient, EmbeddingClient

    # Text generation
    llm = LLMClient(model="ollama/qwen2.5-coder:3b")
    response = llm.generate("Explain this code...")

    # Embeddings
    embed = EmbeddingClient(model="ollama/snowflake-arctic-embed2")
    vector = embed.embed("Some text to embed")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import litellm
from litellm import completion, embedding

# Disable LiteLLM telemetry
litellm.telemetry = False


class LLMError(Exception):
    """Raised when LLM operations fail."""

    pass


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    # Model identifier (format: provider/model or just model for OpenAI)
    # Examples:
    #   - "ollama/qwen2.5-coder:3b"
    #   - "gpt-4o-mini"
    #   - "claude-3-haiku-20240307"
    model: str = "ollama/qwen2.5-coder:3b"

    # API configuration
    api_base: str | None = None  # For Ollama: "http://localhost:11434"
    api_key: str | None = None  # For cloud providers

    # Generation settings
    temperature: float = 0.3
    max_tokens: int = 256
    timeout: int = 60

    # Retry settings
    max_retries: int = 3

    @classmethod
    def from_ollama_url(cls, url: str, model: str) -> LLMConfig:
        """Create config for Ollama provider.

        Args:
            url: Ollama server URL (e.g., "http://localhost:11434")
            model: Model name (e.g., "qwen2.5-coder:3b")

        Returns:
            LLMConfig configured for Ollama.
        """
        return cls(
            model=f"ollama/{model}",
            api_base=url,
        )


@dataclass
class EmbeddingConfig:
    """Configuration for embedding client."""

    # Model identifier
    # Examples:
    #   - "ollama/snowflake-arctic-embed2"
    #   - "text-embedding-3-small"
    model: str = "ollama/snowflake-arctic-embed2"

    # API configuration
    api_base: str | None = None
    api_key: str | None = None

    # Vector settings
    dimensions: int = 1024

    # Request settings
    timeout: int = 30
    max_retries: int = 3

    @classmethod
    def from_ollama_url(cls, url: str, model: str, dimensions: int = 1024) -> EmbeddingConfig:
        """Create config for Ollama embedding provider.

        Args:
            url: Ollama server URL
            model: Embedding model name
            dimensions: Expected embedding dimensions

        Returns:
            EmbeddingConfig configured for Ollama.
        """
        return cls(
            model=f"ollama/{model}",
            api_base=url,
            dimensions=dimensions,
        )


# =============================================================================
# LLM CLIENT (TEXT GENERATION)
# =============================================================================


class LLMClient:
    """Unified LLM client for text generation using LiteLLM.

    Supports multiple providers through LiteLLM:
    - Ollama: model="ollama/qwen2.5-coder:3b", api_base="http://localhost:11434"
    - OpenAI: model="gpt-4o-mini", api_key="sk-..."
    - Anthropic: model="claude-3-haiku-20240307", api_key="sk-ant-..."
    - And many more (see LiteLLM docs)
    """

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize LLM client.

        Args:
            model: Model identifier (e.g., "ollama/qwen2.5-coder:3b", "gpt-4o-mini")
            api_base: API base URL (required for Ollama, optional for cloud)
            api_key: API key (required for cloud providers)
        """
        self.model = model
        self.api_base = api_base
        self.api_key = api_key

        # Extract provider from model string
        self.provider = model.split("/")[0] if "/" in model else "openai"

    @classmethod
    def from_config(cls, config: LLMConfig) -> LLMClient:
        """Create client from config."""
        return cls(
            model=config.model,
            api_base=config.api_base,
            api_key=config.api_key,
        )

    @classmethod
    def from_ollama(cls, url: str, model: str) -> LLMClient:
        """Create client for Ollama provider.

        Args:
            url: Ollama server URL
            model: Model name without provider prefix

        Returns:
            LLMClient configured for Ollama.
        """
        return cls(
            model=f"ollama/{model}",
            api_base=url,
        )

    def verify_model(self) -> bool:
        """Check if model is available.

        Returns:
            True if model is accessible.
        """
        try:
            # Try a minimal completion to verify
            self.generate("test", max_tokens=1, timeout=10)
            return True
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 256,
        timeout: int = 60,
        model: str | None = None,
    ) -> str:
        """Generate text completion.

        Args:
            prompt: The prompt to send to the model.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
            model: Optional model override.

        Returns:
            Generated text.

        Raises:
            LLMError: If generation fails.
        """
        use_model = model or self.model

        try:
            # Build kwargs for litellm
            kwargs: dict[str, Any] = {
                "model": use_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }

            # Add api_base for Ollama
            if self.api_base:
                kwargs["api_base"] = self.api_base

            # Add api_key if provided
            if self.api_key:
                kwargs["api_key"] = self.api_key

            response = completion(**kwargs)

            # Extract text from response
            content = response.choices[0].message.content
            if content is None:
                raise LLMError(f"Empty response from model '{use_model}'")

            return content.strip()

        except Exception as e:
            raise LLMError(f"LLM generation failed for model '{use_model}': {e}") from e


# =============================================================================
# EMBEDDING CLIENT
# =============================================================================


class EmbeddingClient:
    """Unified embedding client using LiteLLM.

    Supports multiple providers:
    - Ollama: model="ollama/snowflake-arctic-embed2", api_base="http://localhost:11434"
    - OpenAI: model="text-embedding-3-small", api_key="sk-..."
    - And many more (see LiteLLM docs)
    """

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        dimensions: int = 1024,
    ):
        """Initialize embedding client.

        Args:
            model: Model identifier
            api_base: API base URL (required for Ollama)
            api_key: API key (required for cloud providers)
            dimensions: Expected embedding dimensions
        """
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.dimensions = dimensions

        # Extract provider from model string
        self.provider = model.split("/")[0] if "/" in model else "openai"

    @classmethod
    def from_config(cls, config: EmbeddingConfig) -> EmbeddingClient:
        """Create client from config."""
        return cls(
            model=config.model,
            api_base=config.api_base,
            api_key=config.api_key,
            dimensions=config.dimensions,
        )

    @classmethod
    def from_ollama(cls, url: str, model: str, dimensions: int = 1024) -> EmbeddingClient:
        """Create client for Ollama provider.

        Args:
            url: Ollama server URL
            model: Model name without provider prefix
            dimensions: Expected embedding dimensions

        Returns:
            EmbeddingClient configured for Ollama.
        """
        return cls(
            model=f"ollama/{model}",
            api_base=url,
            dimensions=dimensions,
        )

    def verify_model(self) -> bool:
        """Check if embedding model is available.

        Returns:
            True if model is accessible.
        """
        try:
            self.embed("test", timeout=10)
            return True
        except Exception:
            return False

    def embed(self, text: str, timeout: int = 30) -> list[float]:
        """Generate embedding for single text.

        Args:
            text: Text to embed.
            timeout: Request timeout in seconds.

        Returns:
            Embedding vector as list of floats.

        Raises:
            LLMError: If embedding generation fails.
        """
        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "input": [text],
                "timeout": timeout,
            }

            if self.api_base:
                kwargs["api_base"] = self.api_base

            if self.api_key:
                kwargs["api_key"] = self.api_key

            response = embedding(**kwargs)

            if not response.data:
                raise LLMError("No embedding returned")

            return response.data[0]["embedding"]

        except Exception as e:
            raise LLMError(f"Embedding generation failed: {e}") from e

    def embed_batch(self, texts: list[str], timeout: int = 60) -> list[list[float]]:
        """Generate embeddings for batch of texts.

        Args:
            texts: List of texts to embed.
            timeout: Request timeout in seconds.

        Returns:
            List of embedding vectors.

        Raises:
            LLMError: If embedding generation fails.
        """
        if not texts:
            return []

        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "input": texts,
                "timeout": timeout,
            }

            if self.api_base:
                kwargs["api_base"] = self.api_base

            if self.api_key:
                kwargs["api_key"] = self.api_key

            response = embedding(**kwargs)

            return [item["embedding"] for item in response.data]

        except Exception as e:
            raise LLMError(f"Batch embedding generation failed: {e}") from e


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================


# These aliases maintain backward compatibility with existing code
OllamaClient = LLMClient
OllamaEmbedClient = EmbeddingClient


def create_llm_client_from_config(
    ollama_url: str,
    model: str,
) -> LLMClient:
    """Create LLM client from legacy Ollama config.

    This is a convenience function for migrating existing code.

    Args:
        ollama_url: Ollama server URL
        model: Model name

    Returns:
        LLMClient configured for Ollama.
    """
    return LLMClient.from_ollama(ollama_url, model)


def create_embedding_client_from_config(
    ollama_url: str,
    model: str,
    dimensions: int = 1024,
) -> EmbeddingClient:
    """Create embedding client from legacy Ollama config.

    Args:
        ollama_url: Ollama server URL
        model: Embedding model name
        dimensions: Expected embedding dimensions

    Returns:
        EmbeddingClient configured for Ollama.
    """
    return EmbeddingClient.from_ollama(ollama_url, model, dimensions)
