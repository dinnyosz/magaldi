"""Shared CLI utilities, console setup, and common functions.

This module contains:
- Warning suppression and aiohttp patches
- Console setup
- Common formatting and utility functions
- Main CLI group definition
"""

from __future__ import annotations

# Suppress warnings from LiteLLM/aiohttp before any imports
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning, message=".*[Uu]nclosed.*")
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

# Patch aiohttp to not emit unclosed session warnings
# This is necessary because aiohttp uses both warnings.warn AND loop.call_exception_handler
import aiohttp.client
import aiohttp.connector

_original_client_del = aiohttp.client.ClientSession.__del__
_original_connector_del = aiohttp.connector.BaseConnector.__del__


def _quiet_client_del(self, _warnings=None):
    """Suppress unclosed session warning."""
    pass  # Do nothing - session will be GC'd anyway


def _quiet_connector_del(self, _warnings=None):
    """Suppress unclosed connector warning."""
    pass  # Do nothing - connector will be GC'd anyway


aiohttp.client.ClientSession.__del__ = _quiet_client_del
aiohttp.connector.BaseConnector.__del__ = _quiet_connector_del

import sys
from typing import TYPE_CHECKING

import click
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from shared.config import MagaldiConfig

console = Console()


def format_duration(seconds: float) -> str:
    """Format duration as hh:mm:ss or mm:ss."""
    total_secs = int(seconds)
    hours, remainder = divmod(total_secs, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def check_model_availability(config: "MagaldiConfig", skip_ai: bool) -> list[str]:
    """Check if required models are available from their configured providers.

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
            response = requests.get(f"{url.rstrip('/')}/api/tags", timeout=5)
            response.raise_for_status()
            models = {m.get("name") for m in response.json().get("models", [])}
            ollama_models_cache[url] = models
            return models
        except requests.exceptions.ConnectionError:
            return None
        except Exception:
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

    for purpose, model_cfg in models_to_check:
        if model_cfg.provider == "ollama":
            available = get_ollama_models(model_cfg.url)
            if available is None:
                errors.append(f"Cannot connect to Ollama at {model_cfg.url} for {purpose} model")
            elif not model_in_ollama(model_cfg.name, available):
                errors.append(f"Model '{model_cfg.name}' not found. Run: ollama pull {model_cfg.name}")
        elif model_cfg.provider == "llamacpp":
            # Check if llama-server is running
            try:
                response = requests.get(f"{model_cfg.url.rstrip('/')}/v1/models", timeout=5)
                # llama-server returns 200 if running
            except requests.exceptions.ConnectionError:
                errors.append(
                    f"Cannot connect to llama-server at {model_cfg.url} for {purpose} model. "
                    f"Start with: ./tools/benchmark-llama-server.sh"
                )
            except Exception:
                pass  # Server is running but endpoint may not exist

    return errors


# =============================================================================
# MAIN CLI GROUP
# =============================================================================


@click.group()
@click.version_option(version="0.1.0", prog_name="magaldi")
def main() -> None:
    """Magaldi - Code discovery engine for AI agents and developers."""
    pass
