"""Helper functions for benchmark commands.

This module contains utility functions for model configuration, backend
connection checking, and model warmup.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.config import BenchmarkConfig, ModelConfig


def parse_model_spec_to_config(spec: str, ollama_url: str | None = None) -> "ModelConfig":
    """Parse CLI model specification into a ModelConfig.

    Formats:
        "qwen3:4b" -> Ollama model (default)
        "lmstudio:xxx" -> LM Studio model
        "openai:gpt-4" -> OpenAI model
    """
    from shared.config import ModelConfig

    # Check for explicit provider prefix
    for prefix in ["lmstudio:", "openai:", "anthropic:", "vllm:", "llamacpp:"]:
        if spec.startswith(prefix):
            provider = prefix[:-1]
            model_name = spec[len(prefix):]
            if provider == "lmstudio":
                return ModelConfig(
                    name=model_name,
                    provider="lmstudio",
                    url="http://localhost:1234",
                    api_key="lm-studio",
                )
            elif provider == "openai":
                return ModelConfig(name=model_name, provider="openai")
            else:
                return ModelConfig(name=model_name, provider=provider)

    # Default to Ollama with default URL
    return ModelConfig(
        name=spec,
        provider="ollama",
        url=ollama_url or "http://localhost:11434",
    )


def get_model_api_config(model_config: "ModelConfig") -> dict:
    """Get api_base and api_key for a ModelConfig.

    Returns dict with 'api_base' and 'api_key' keys for LiteLLM.
    """
    result = {}
    api_base = model_config.get_api_base()
    if api_base:
        result["api_base"] = api_base
    if model_config.api_key:
        result["api_key"] = model_config.api_key
    return result


def check_backend_connections(
    model_configs: list["ModelConfig"],
    console,
) -> tuple[dict[str, list[str]], list["ModelConfig"], list["ModelConfig"]]:
    """Check backend connections and determine which models are available.

    Args:
        model_configs: List of model configurations to check.
        console: Rich console for output.

    Returns:
        Tuple of (available_models_by_provider, models_to_test, missing_models).
    """
    from shared.ai.ollama_benchmark import BenchmarkClient

    # Group models by provider to check each backend once
    models_by_provider: dict[str, list["ModelConfig"]] = defaultdict(list)
    for mc in model_configs:
        models_by_provider[mc.provider].append(mc)

    # Check each provider's connection
    available_models_by_provider: dict[str, list[str]] = {}

    for provider, provider_models in models_by_provider.items():
        url = provider_models[0].url

        if provider == "ollama":
            client = BenchmarkClient(api_base=url)
            if client.check_connection():
                available = client.list_models()
                available_models_by_provider[provider] = available
                console.print(f"  [green]✓[/] Ollama ({url}) | {len(available)} models")
            else:
                console.print(f"[red]Cannot connect to Ollama at {url}[/]")
                available_models_by_provider[provider] = []

        elif provider in ("lmstudio", "llamacpp"):
            try:
                import requests
                api_url = f"{url.rstrip('/')}/v1/models"
                resp = requests.get(api_url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    available = [m["id"] for m in data.get("data", [])]
                    available_models_by_provider[provider] = available
                    console.print(f"  [green]✓[/] {provider} ({url}) | {len(available)} models")
                else:
                    console.print(f"  [yellow]![/] {provider} ({url}) | Cannot list models (status {resp.status_code})")
                    available_models_by_provider[provider] = []
            except Exception as e:
                console.print(f"  [yellow]![/] {provider} ({url}) | Connection failed: {e}")
                available_models_by_provider[provider] = []

        else:
            # OpenAI, Anthropic, etc. - trust they're available
            available_models_by_provider[provider] = ["*"]
            console.print(f"  [green]✓[/] {provider} (cloud API)")

    # Check availability of each configured model
    missing_models: list["ModelConfig"] = []
    models_to_test: list["ModelConfig"] = []

    for mc in model_configs:
        available = available_models_by_provider.get(mc.provider, [])

        # Cloud providers - trust availability
        if available == ["*"]:
            models_to_test.append(mc)
            continue

        # Check if model is in available list
        if mc.name in available or f"{mc.name}:latest" in available:
            models_to_test.append(mc)
        else:
            # Try without tag
            base = mc.name.rsplit(":", 1)[0] if ":" in mc.name else mc.name
            if base in available or f"{base}:latest" in available:
                models_to_test.append(mc)
            else:
                missing_models.append(mc)

    return available_models_by_provider, models_to_test, missing_models


def warmup_benchmark_models(
    models_to_test: list["ModelConfig"],
    benchmark_config: "BenchmarkConfig",
    console,
) -> list["ModelConfig"]:
    """Warm up models and return the list of models that succeeded.

    Args:
        models_to_test: List of model configurations to warm up.
        benchmark_config: Benchmark configuration.
        console: Rich console for output.

    Returns:
        List of models that successfully warmed up.
    """
    from shared.ai.ollama_benchmark import BenchmarkClient

    models_failed_warmup: list["ModelConfig"] = []
    for mc in models_to_test:
        display_name = f"{mc.provider}/{mc.name}"
        api_config = get_model_api_config(mc)
        api_model = mc.get_litellm_model()
        with console.status(f"[bold blue]Warming up {display_name}...[/]"):
            warmup_client = BenchmarkClient(api_base=mc.url)
            success, warmup_time, error = warmup_client.warmup(
                api_model,
                timeout=120,
                **api_config,
            )
        if success:
            console.print(f"  [green]✓[/] {display_name} ({warmup_time:.1f}s)")
        else:
            console.print(f"  [red]✗[/] {display_name}: {error}")
            models_failed_warmup.append(mc)

    # Remove failed models
    return [mc for mc in models_to_test if mc not in models_failed_warmup]
