"""Shared CLI utilities, console setup, and common functions.

This module contains:
- Warning suppression and aiohttp patches
- Console setup
- Common formatting and utility functions
- Main CLI group definition
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import aiohttp.client
import aiohttp.connector
import click
from rich.console import Console
from rich.markup import escape as rich_escape

if TYPE_CHECKING:
    from shared.config import MagaldiConfig

# Suppress warnings from LiteLLM/aiohttp
warnings.filterwarnings("ignore", category=ResourceWarning, message=".*[Uu]nclosed.*")
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

# Patch aiohttp to not emit unclosed session warnings
# This is necessary because aiohttp uses both warnings.warn AND loop.call_exception_handler
_original_client_del = aiohttp.client.ClientSession.__del__
_original_connector_del = aiohttp.connector.BaseConnector.__del__


def _quiet_client_del(self: Any, _warnings: Any = None) -> None:
    """Suppress unclosed session warning."""
    pass  # Do nothing - session will be GC'd anyway


def _quiet_connector_del(self: Any, _warnings: Any = None) -> None:
    """Suppress unclosed connector warning."""
    pass  # Do nothing - connector will be GC'd anyway


aiohttp.client.ClientSession.__del__ = _quiet_client_del  # type: ignore[method-assign]
aiohttp.connector.BaseConnector.__del__ = _quiet_connector_del  # type: ignore[method-assign]

console = Console()


def format_duration(seconds: float) -> str:
    """Format duration as hh:mm:ss or mm:ss."""
    total_secs = int(seconds)
    hours, remainder = divmod(total_secs, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def validate_llm_url(url: str) -> str:
    """Validate and normalize LLM server URL.

    Ensures URL has valid http/https scheme and no suspicious characters.
    Returns the validated URL with trailing slash removed.

    Args:
        url: The URL to validate.

    Returns:
        Normalized URL.

    Raises:
        ValueError: If URL is invalid or has unsafe scheme.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}. Only http/https allowed.")

    if not parsed.netloc:
        raise ValueError(f"Invalid URL: missing host in {url}")

    # Check for suspicious characters that might indicate injection attempts
    suspicious_chars = ["\n", "\r", "\t", " ", "<", ">", '"', "'", ";", "&", "|"]
    for char in suspicious_chars:
        if char in url:
            raise ValueError(f"Invalid URL: contains suspicious character {repr(char)}")

    return url.rstrip("/")


def get_model_column_width(config: MagaldiConfig) -> int:
    """Calculate the optimal width for model name columns in CLI displays.

    Takes into account all configured models and their potential tiered names
    (for Ollama models with context tier suffixes like -2k, -4k, -32k).

    Args:
        config: Magaldi configuration with model settings.

    Returns:
        Width in characters for the model column.
    """
    model_names = []

    # Collect all model names
    for model_cfg in [
        config.llm.get_summarize_model(),
        config.llm.get_summarize_model_small(),
        config.llm.get_embed_model(),
    ]:
        name = model_cfg.name
        if model_cfg.provider == "ollama":
            # For Ollama, add longest tier suffix (-32k = 4 chars)
            model_names.append(name + "-32k")
        else:
            model_names.append(name)

    # Return max length + 2 for padding, minimum 20
    return max(20, max(len(n) for n in model_names) + 2)


def check_model_availability(config: MagaldiConfig, skip_ai: bool) -> list[str]:
    """Check if required models are available from their configured providers.

    For Ollama models, this also creates tiered model aliases (e.g., model-2k, model-4k)
    to avoid reload overhead when context size changes between requests.

    Returns:
        List of error messages (empty if all models are available).
    """
    if skip_ai:
        return []

    import requests

    errors = []

    # Group models by provider and URL
    models_to_check = [
        ("summarize", config.llm.get_summarize_model()),
        ("summarize_small", config.llm.get_summarize_model_small()),
        ("embed", config.llm.get_embed_model()),
    ]

    # Cache Ollama model lists per URL
    ollama_models_cache: dict[str, set[str]] = {}

    def get_ollama_models(url: str) -> set[str] | None:
        """Get available models from Ollama server."""
        if url in ollama_models_cache:
            return ollama_models_cache[url]
        try:
            validated_url = validate_llm_url(url)
            response = requests.get(f"{validated_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = {m.get("name") for m in response.json().get("models", [])}
            ollama_models_cache[url] = models
            return models
        except requests.exceptions.ConnectionError:
            return None
        except (ValueError, Exception):
            return None

    def model_in_ollama(model_name: str, available: set[str]) -> bool:
        """Check if model is available (handles :latest tag)."""
        if model_name in available:
            return True
        if f"{model_name}:latest" in available:
            return True
        if ":" in model_name:
            base = model_name.rsplit(":", 1)[0]
            if base in available or f"{base}:latest" in available:
                return True
        return False

    # Track Ollama models that need tier warmup
    ollama_models_to_warmup: list[tuple[str, str]] = []  # (model_name, url)

    for purpose, model_cfg in models_to_check:
        if model_cfg.provider == "ollama":
            available = get_ollama_models(model_cfg.url)
            if available is None:
                errors.append(f"Cannot connect to Ollama at {model_cfg.url} for {purpose} model")
            elif not model_in_ollama(model_cfg.name, available):
                errors.append(f"Model '{model_cfg.name}' not found. Run: ollama pull {model_cfg.name}")
            else:
                # Model is available, add to warmup list (only LLM models, not embeddings)
                if purpose != "embed":
                    ollama_models_to_warmup.append((model_cfg.name, model_cfg.url))
        elif model_cfg.provider in ("lmstudio", "llamacpp", "vllm-mlx"):
            # Check if OpenAI-compatible server is running
            try:
                requests.get(f"{model_cfg.url.rstrip('/')}/v1/models", timeout=5)
            except requests.exceptions.ConnectionError:
                errors.append(
                    f"Cannot connect to OpenAI-compatible server at {model_cfg.url} for {purpose} model. "
                    f"Start the server and load a model."
                )
            except Exception:
                pass  # Server is running but endpoint may not exist

    # Warmup Ollama tiered models (create aliases if they don't exist)
    if not errors and ollama_models_to_warmup:
        _warmup_ollama_tiers(ollama_models_to_warmup)

    return errors


def _warmup_ollama_tiers(models: list[tuple[str, str]]) -> None:
    """Create tiered model aliases for Ollama models.

    This creates aliases like qwen3:4b-instruct-2k, qwen3:4b-instruct-4k, etc.
    to avoid model reload overhead when context size changes.

    Args:
        models: List of (model_name, url) tuples.
    """

    from rich.console import Console

    from shared.ai.ollama_models import warmup_ollama_tiers_verbose

    console = Console()

    # Deduplicate models
    unique_models = list(set(models))

    for model_name, url in unique_models:
        litellm_model = f"ollama/{model_name}"
        console.print(f"  [dim]Preparing tiered aliases for {model_name}...[/]")

        try:
            result = warmup_ollama_tiers_verbose(litellm_model, url)
            if result is None:
                console.print("    [yellow]→ skipped (already tiered)[/]")
            elif result.created or result.existed:
                # Show what was created
                if result.created:
                    created_str = ", ".join(f"{t//1024}k" for t in sorted(result.created))
                    console.print(f"    [green]→ created:[/] {created_str}")
                # Show what already existed
                if result.existed:
                    existed_str = ", ".join(f"{t//1024}k" for t in sorted(result.existed))
                    console.print(f"    [dim]→ existed:[/] {existed_str}")
                # Show failures
                if result.failed:
                    failed_str = ", ".join(f"{t//1024}k" for t in sorted(result.failed))
                    console.print(f"    [red]→ failed:[/] {failed_str}")
            else:
                console.print("    [red]→ all tiers failed[/]")
        except Exception as e:
            console.print(f"    [red]→ error:[/] {rich_escape(str(e))}")


# =============================================================================
# MAIN CLI GROUP
# =============================================================================


@click.group()
@click.version_option(version="0.1.0", prog_name="magaldi")
def main() -> None:
    """Magaldi - Code discovery engine for AI agents and developers."""
    pass
