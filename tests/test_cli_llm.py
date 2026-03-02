"""Tests for the magaldi llm CLI command."""

from __future__ import annotations

from shared.cli.llm import ServerPlan, _build_server_plans
from shared.config import LLMConfig, ModelConfig

# =============================================================================
# _build_server_plans
# =============================================================================


class TestBuildServerPlans:
    """Test server plan generation from config."""

    def test_no_vllm_models_returns_empty(self) -> None:
        """Config with only Ollama models returns no plans."""
        config = LLMConfig(
            models={
                "qwen3-4b": ModelConfig(
                    name="qwen3:4b-instruct",
                    provider="ollama",
                    url="http://localhost:11434",
                ),
            }
        )
        plans = _build_server_plans(config)
        assert plans == []

    def test_single_llm_model(self) -> None:
        """Single vllm-mlx LLM model creates one plan."""
        config = LLMConfig(
            models={
                "qwen3-4b": ModelConfig(
                    name="mlx-community/Qwen3-4B-Instruct-2507-4bit",
                    provider="vllm-mlx",
                    url="http://localhost:8000",
                ),
            }
        )
        plans = _build_server_plans(config)
        assert len(plans) == 1
        assert plans[0].port == 8000
        assert plans[0].llm_model == "mlx-community/Qwen3-4B-Instruct-2507-4bit"
        assert plans[0].embedding_model is None

    def test_llm_with_embedding_same_port(self) -> None:
        """LLM + embedding on same port creates one plan with embed pre-loaded."""
        config = LLMConfig(
            models={
                "qwen3-4b": ModelConfig(
                    name="mlx-community/Qwen3-4B-Instruct-2507-4bit",
                    provider="vllm-mlx",
                    url="http://localhost:8000",
                ),
                "qwen3-embed": ModelConfig(
                    name="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
                    provider="vllm-mlx",
                    url="http://localhost:8000",
                    dimensions=1024,
                ),
            }
        )
        plans = _build_server_plans(config)
        assert len(plans) == 1
        assert plans[0].port == 8000
        assert plans[0].llm_model == "mlx-community/Qwen3-4B-Instruct-2507-4bit"
        assert plans[0].embedding_model == "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"

    def test_different_ports_create_multiple_plans(self) -> None:
        """Models on different ports create separate plans."""
        config = LLMConfig(
            models={
                "qwen3-4b": ModelConfig(
                    name="mlx-community/Qwen3-4B-Instruct-2507-4bit",
                    provider="vllm-mlx",
                    url="http://localhost:8000",
                ),
                "qwen3-small": ModelConfig(
                    name="mlx-community/Qwen3-1.7B-4bit",
                    provider="vllm-mlx",
                    url="http://localhost:8001",
                ),
            }
        )
        plans = _build_server_plans(config)
        assert len(plans) == 2
        ports = {p.port for p in plans}
        assert ports == {8000, 8001}

    def test_mixed_providers_only_vllm(self) -> None:
        """Only vllm-mlx models are included in plans."""
        config = LLMConfig(
            models={
                "ollama-model": ModelConfig(
                    name="qwen3:4b-instruct",
                    provider="ollama",
                    url="http://localhost:11434",
                ),
                "vllm-model": ModelConfig(
                    name="mlx-community/Qwen3-4B-Instruct-2507-4bit",
                    provider="vllm-mlx",
                    url="http://localhost:8000",
                ),
                "cloud-model": ModelConfig(
                    name="gpt-4o-mini",
                    provider="openai",
                    url="",
                    api_key="sk-...",
                ),
            }
        )
        plans = _build_server_plans(config)
        assert len(plans) == 1
        assert plans[0].llm_model == "mlx-community/Qwen3-4B-Instruct-2507-4bit"

    def test_only_embedding_models_skipped(self) -> None:
        """Port with only embedding models (no LLM) is skipped."""
        config = LLMConfig(
            models={
                "embed-only": ModelConfig(
                    name="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
                    provider="vllm-mlx",
                    url="http://localhost:8000",
                    dimensions=1024,
                ),
            }
        )
        plans = _build_server_plans(config)
        assert plans == []

    def test_default_port_when_missing(self) -> None:
        """URL without port defaults to 8000."""
        config = LLMConfig(
            models={
                "model": ModelConfig(
                    name="mlx-community/Qwen3-4B-Instruct-2507-4bit",
                    provider="vllm-mlx",
                    url="http://localhost",
                ),
            }
        )
        plans = _build_server_plans(config)
        assert len(plans) == 1
        assert plans[0].port == 8000

    def test_multiple_llm_models_same_port_uses_first(self) -> None:
        """Multiple LLM models on same port uses first as primary."""
        config = LLMConfig(
            models={
                "primary": ModelConfig(
                    name="mlx-community/Qwen3-4B-Instruct-2507-4bit",
                    provider="vllm-mlx",
                    url="http://localhost:8000",
                ),
                "secondary": ModelConfig(
                    name="mlx-community/Qwen3-1.7B-4bit",
                    provider="vllm-mlx",
                    url="http://localhost:8000",
                ),
            }
        )
        plans = _build_server_plans(config)
        assert len(plans) == 1
        # First model becomes primary
        assert plans[0].llm_model == "mlx-community/Qwen3-4B-Instruct-2507-4bit"


# =============================================================================
# ServerPlan
# =============================================================================


class TestServerPlan:
    """Test ServerPlan methods."""

    def test_pidfile_path(self) -> None:
        plan = ServerPlan(port=8000, host="0.0.0.0", llm_model="test")
        assert plan.pidfile.name == "vllm-mlx-8000.pid"

    def test_logfile_path(self) -> None:
        plan = ServerPlan(port=8001, host="0.0.0.0", llm_model="test")
        assert plan.logfile.name == "vllm-mlx-8001.log"

    def test_health_url(self) -> None:
        plan = ServerPlan(port=8000, host="0.0.0.0", llm_model="test")
        assert plan.health_url() == "http://0.0.0.0:8000/health"

    def test_is_running_false_when_no_server(self) -> None:
        """is_running returns False when nothing is listening."""
        plan = ServerPlan(port=19999, host="127.0.0.1", llm_model="test")
        assert plan.is_running() is False

    def test_get_pid_none_when_no_pidfile(self) -> None:
        """get_pid returns None when no pidfile exists."""
        plan = ServerPlan(port=8000, host="0.0.0.0", llm_model="test")
        # Ensure pidfile doesn't exist
        plan.pidfile.unlink(missing_ok=True)
        assert plan.get_pid() is None

    def test_continuous_batching_default_true(self) -> None:
        plan = ServerPlan(port=8000, host="0.0.0.0", llm_model="test")
        assert plan.continuous_batching is True


# =============================================================================
# ModelConfig vllm-mlx integration
# =============================================================================


class TestModelConfigVllmMlx:
    """Test ModelConfig with vllm-mlx provider."""

    def test_get_litellm_model_always_default(self) -> None:
        """vllm-mlx always returns openai/default regardless of name."""
        cfg = ModelConfig(
            name="mlx-community/Qwen3-4B-Instruct-2507-4bit",
            provider="vllm-mlx",
            url="http://localhost:8000",
        )
        assert cfg.get_litellm_model() == "openai/default"

    def test_get_api_base(self) -> None:
        """vllm-mlx appends /v1 to URL."""
        cfg = ModelConfig(
            name="mlx-community/Qwen3-4B-Instruct-2507-4bit",
            provider="vllm-mlx",
            url="http://localhost:8000",
        )
        assert cfg.get_api_base() == "http://localhost:8000/v1"

    def test_get_api_base_strips_trailing_slash(self) -> None:
        """Trailing slash is stripped before appending /v1."""
        cfg = ModelConfig(
            name="mlx-community/Qwen3-4B-Instruct-2507-4bit",
            provider="vllm-mlx",
            url="http://localhost:8000/",
        )
        assert cfg.get_api_base() == "http://localhost:8000/v1"

    def test_ollama_provider_unchanged(self) -> None:
        """Ollama provider still works as before."""
        cfg = ModelConfig(name="qwen3:4b-instruct", provider="ollama")
        assert cfg.get_litellm_model() == "ollama/qwen3:4b-instruct"
        assert cfg.get_api_base() == "http://localhost:11434"

    def test_llamacpp_provider_unchanged(self) -> None:
        """llamacpp provider still works as before."""
        cfg = ModelConfig(
            name="qwen3:4b-instruct",
            provider="llamacpp",
            url="http://localhost:8080",
        )
        assert cfg.get_litellm_model() == "openai/qwen3:4b-instruct"
        assert cfg.get_api_base() == "http://localhost:8080/v1"
