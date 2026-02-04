"""Central configuration management for Magaldi.

This module provides a single source of truth for all configuration values.
Configuration is loaded with the following priority (highest to lowest):
1. Environment variables (MAGALDI_*)
2. Configuration file (magaldi.yaml)
3. Dataclass defaults

Usage:
    from shared.config import load_config, get_config

    # At application startup
    load_config()  # Or load_config("/path/to/config.yaml")

    # In any module
    config = get_config()
    print(config.elasticsearch.host)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when configuration is invalid or incomplete."""

    pass


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================


@dataclass
class ElasticsearchConfig:
    """Elasticsearch configuration."""

    host: str = "localhost"
    port: int = 9200
    scheme: str = "http"
    index: str = "magaldi_code_elements"
    timeout: int = 30
    bulk_timeout: int = 300  # 5 minutes for bulk operations (delete_by_query, update_by_query)
    retry_on_timeout: bool = True
    max_retries: int = 3

    @property
    def url(self) -> str:
        """Build URL from components."""
        return f"{self.scheme}://{self.host}:{self.port}"


@dataclass
class RedisConfig:
    """Redis configuration."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None


@dataclass
class ModelConfig:
    """Configuration for a single model.

    Encapsulates everything needed to connect to and use a model.
    """

    name: str  # Model name (e.g., "qwen3:4b-instruct", "gpt-4o-mini")
    provider: str = "ollama"  # ollama, llamacpp, openai, anthropic
    url: str = "http://localhost:11434"  # API endpoint
    api_key: str | None = None  # For cloud providers

    # Optional model-specific settings
    temperature: float | None = None
    max_tokens: int | None = None
    dimensions: int | None = None  # For embedding models
    num_ctx: int | None = None  # Context window size (Ollama num_ctx)

    def get_litellm_model(self) -> str:
        """Get the full LiteLLM model identifier."""
        if self.provider == "ollama":
            return f"ollama/{self.name}"
        elif self.provider == "llamacpp":
            return f"openai/{self.name}"
        elif self.provider == "openai":
            return self.name
        else:
            return f"{self.provider}/{self.name}"

    def get_api_base(self) -> str | None:
        """Get the API base URL."""
        if self.provider == "ollama":
            return self.url
        elif self.provider == "llamacpp":
            return f"{self.url.rstrip('/')}/v1"
        elif self.provider == "openai":
            return None
        else:
            return self.url


@dataclass
class LLMConfig:
    """LLM configuration with named model definitions.

    Models are defined once with all their connection details,
    then referenced by name for different purposes.

    Example config.yaml:
        llm:
          models:
            qwen3-4b:
              name: qwen3:4b-instruct
              provider: llamacpp
              url: http://localhost:8080
            qwen3-small:
              name: qwen3:1.7b
              provider: llamacpp
              url: http://localhost:8080
            arctic:
              name: snowflake-arctic-embed2
              provider: ollama
              url: http://localhost:11434

          summarize_model: qwen3-4b
          summarize_model_small: qwen3-small
          embed_model: arctic
    """

    # Named model configurations
    # All models run via Ollama for simplicity and optimized Apple Silicon performance
    models: dict[str, ModelConfig] = field(default_factory=lambda: {
        "qwen3-4b": ModelConfig(
            name="qwen3:4b-instruct",
            provider="ollama",
            url="http://localhost:11434",
        ),
        "qwen3-small": ModelConfig(
            name="qwen3:1.7b",
            provider="ollama",
            url="http://localhost:11434",
        ),
        "arctic-embed": ModelConfig(
            name="snowflake-arctic-embed2",
            provider="ollama",
            url="http://localhost:11434",
            dimensions=1024,
        ),
    })

    # Which models to use for each purpose (reference by name)
    summarize_model: str = "qwen3-4b"
    summarize_model_small: str = "qwen3-small"
    embed_model: str = "arctic-embed"

    # Generation settings (defaults, can be overridden per-model)
    summarize_temperature: float = 0.2
    summarize_top_p: float = 0.95
    summarize_max_tokens: int = 512
    summarize_context_window: int = 8192
    embed_context_window: int = 8192

    # Context size for aggregation tasks (features, glossary)
    # Uses fixed large context since input size depends on cluster size
    aggregation_context_size: int = 16384

    def get_model(self, model_ref: str) -> ModelConfig:
        """Get a model configuration by reference name."""
        if model_ref not in self.models:
            raise KeyError(f"Model '{model_ref}' not found. Available: {list(self.models.keys())}")
        return self.models[model_ref]

    def get_summarize_model(self) -> ModelConfig:
        """Get the main summarization model config."""
        return self.get_model(self.summarize_model)

    def get_summarize_model_small(self) -> ModelConfig:
        """Get the small summarization model config."""
        return self.get_model(self.summarize_model_small)

    def get_embed_model(self) -> ModelConfig:
        """Get the embedding model config."""
        return self.get_model(self.embed_model)

    def get_model_for_element_type(self, element_type: str) -> ModelConfig:
        """Get the appropriate model for an element type.

        Uses small model for functions, methods, variables, constants.
        Uses main model for files, classes.
        """
        if element_type in ("function", "method", "variable", "constant"):
            return self.get_summarize_model_small()
        return self.get_summarize_model()

    # Convenience properties for embedding dimensions
    @property
    def embed_dimensions(self) -> int:
        """Get embedding dimensions from the embed model config."""
        model = self.get_embed_model()
        return model.dimensions or 1024





@dataclass
class WorkerConfig:
    """Configuration for a worker pool."""

    count: int = 4
    batch_size: int = 10
    claim_timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: float = 5.0


@dataclass
class WorkersConfig:
    """Configuration for all worker pools."""

    summarization: WorkerConfig = field(default_factory=WorkerConfig)
    embedding: WorkerConfig = field(
        default_factory=lambda: WorkerConfig(batch_size=20, retry_delay_seconds=2.0)
    )


@dataclass
class ParserConfig:
    """Parser configuration."""

    parallel_workers: int = 4
    parse_timeout_seconds: int = 30
    hash_algorithm: str = "sha256"
    hash_workers: int = 4
    hash_chunk_size: int = 65536
    max_file_size_bytes: int = 10485760  # 10 MB
    encoding_fallback: str = "latin-1"


@dataclass
class SearchConfig:
    """Search configuration."""

    default_limit: int = 10
    max_limit: int = 50
    query_cache_size: int = 1000
    query_cache_ttl_seconds: int = 300


@dataclass
class ClusteringConfig:
    """Feature clustering configuration.

    Controls how functions/methods are grouped into features using
    soft clustering algorithms.
    """

    # HDBSCAN parameters (used by both pipelines)
    min_cluster_size: int = 5  # Minimum elements to form a cluster
    min_samples: int = 2  # Density threshold for core points

    # Pipeline selection
    # "scalable": use UMAP + HDBSCAN (recommended, works well for all sizes)
    # "standard": use NMF (slower, has convergence issues on small datasets)
    # "auto": use scalable for >= scalable_threshold elements (legacy)
    soft_clustering_mode: str = "scalable"
    scalable_threshold: int = 1000  # Element count for auto mode (only used if mode="auto")

    # Standard pipeline (NMF) settings
    n_projection_runs: int = 50  # Random projection runs for cooccurrence
    projection_dims: int = 50  # Dimensions after projection
    n_features: int = 500  # Target features for NMF (0 = auto)
    nmf_max_iter: int = 1000  # Max NMF iterations

    # Scalable pipeline (UMAP + HDBSCAN) settings
    pca_components: int = 0  # 0 = auto (min 100 or 90% variance)
    umap_components: int = 50
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.1

    # Thresholds
    membership_threshold: float = 0.01  # Min membership score to keep
    affinity_threshold: float = 0.001  # Min affinity for connected features


@dataclass
class WebConfig:
    """Web server configuration."""

    host: str = "0.0.0.0"
    port: int = 8080
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000"])
    stats_cache_ttl_seconds: int = 30
    file_tree_cache_ttl_seconds: int = 300


@dataclass
class MCPConfig:
    """MCP server configuration."""

    server_name: str = "magaldi"
    server_version: str = "1.0.0"


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    file: str | None = None


@dataclass
class UserDataConfig:
    """User data management configuration."""

    expiration_days: int = 30
    cleanup_batch_size: int = 1000


@dataclass
class ModelParams:
    """Per-model generation parameters.

    Sources for recommended values:
    - Qwen3: huggingface.co/Qwen/Qwen3-4B (instruct: temp=0.7, top_p=0.8)
    - Granite: huggingface.co/ibm-granite/granite-3.1-3b-a800m-instruct (temp=0.6)
    - LFM2.5: huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct (temp=0.1, top_p=0.1)
    """

    temperature: float = 0.2
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    max_tokens: int | None = None  # Override default max_tokens (useful for thinking models)


@dataclass
class BenchmarkConfig:
    """Benchmark configuration for model comparison.

    Model selection based on:
    - arxiv.org/html/2507.03160v2 (SLM code generation study)
    - bentoml.com/blog/the-best-open-source-small-language-models

    Key findings:
    - Qwen2.5-Coder family: best stability and performance/efficiency ratio
    - 3B models offer sweet spot (59% pass@1, ~11GB VRAM)
    - 10% performance gain typically requires 4x VRAM increase

    Example config.yaml:
        benchmark:
          models:
            qwen3-small:
              name: qwen3:1.7b
              provider: ollama
              url: http://localhost:11434
            qwen3-4b:
              name: qwen3:4b-instruct
              provider: llamacpp
              url: http://localhost:8080
          benchmark_models:
            - qwen3-small
            - qwen3-4b
          eval_models:
            - qwen3-4b
    """

    # Named model configurations for benchmarking
    # NOTE: qwen3 models after July 2025 split into "thinking" and "instruct" variants
    #       The default qwen3:Xb has thinking baked in, use instruct variants instead
    # NOTE: Falcon H1 models require manual GGUF import - see plans/ollama_model_research.md
    models: dict[str, ModelConfig] = field(default_factory=lambda: {
        "qwen3-small": ModelConfig(
            name="qwen3:1.7b",
            provider="ollama",
            url="http://localhost:11434",
            num_ctx=16384,
        ),
        "qwen3-4b": ModelConfig(
            name="qwen3:4b-instruct",
            provider="ollama",
            url="http://localhost:11434",
            num_ctx=16384,
        ),
        # Falcon H1 models (hybrid Transformer + Mamba-2 SSM architecture)
        # Requires manual import from GGUF - see plans/ollama_model_research.md
        "falcon-h1-tiny": ModelConfig(
            name="falcon-h1-tiny-90m",
            provider="ollama",
            url="http://localhost:11434",
            num_ctx=16384,
        ),
        "falcon-h1-0.5b": ModelConfig(
            name="falcon-h1-0.5b",
            provider="ollama",
            url="http://localhost:11434",
            num_ctx=16384,
        ),
        "falcon-h1-1.5b": ModelConfig(
            name="falcon-h1-1.5b-deep",
            provider="ollama",
            url="http://localhost:11434",
            num_ctx=16384,
        ),
        "falcon-h1-3b": ModelConfig(
            name="falcon-h1-3b",
            provider="ollama",
            url="http://localhost:11434",
            num_ctx=16384,
        ),
    })

    # Which models to benchmark (reference by name)
    benchmark_models: list[str] = field(default_factory=lambda: [
        "qwen3-small",   # Best balance (8.2 rating, 114 t/s)
        "qwen3-4b",      # Best quality (8.7 rating, 65 t/s)
        # Falcon H1 models - hybrid Transformer + Mamba-2 SSM (requires GGUF import)
        "falcon-h1-tiny",        # 90M params, general instruct
        "falcon-h1-0.5b",        # 0.5B params, instruct
        "falcon-h1-1.5b",        # 1.5B params, deep model
        "falcon-h1-3b",          # 3B params, largest in set
    ])

    # Models used for evaluating/rating summaries (LLM-as-judge)
    eval_models: list[str] = field(default_factory=lambda: [
        "qwen3-4b",  # Quality evaluator
    ])

    # Generation settings (based on paper's methodology)
    # These are defaults; per-model params override when specified
    temperature: float = 0.2      # Paper used 0.2, top_p=0.95
    max_tokens: int = 512
    timeout: int = 120

    # Per-model parameters from HuggingFace model cards
    # Model name patterns are matched with startswith() for flexibility
    model_params: dict[str, ModelParams] = field(default_factory=lambda: {
        # Qwen2.5-Coder: huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
        # temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05
        "qwen2.5-coder": ModelParams(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.05,
        ),
        # Qwen3-Coder: huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
        # temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05
        "qwen3-coder": ModelParams(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.05,
        ),
        # Qwen3 (non-thinking mode, we set think=False in benchmark)
        # huggingface.co/Qwen/Qwen3-4B#best-practices
        # huggingface.co/Qwen/Qwen3-1.7B#best-practices
        # temperature=0.7, top_p=0.8, top_k=20, min_p=0, presence_penalty=1.5
        "qwen3": ModelParams(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
        ),
        # IBM Granite Code: Similar to Granite 3.x params
        # github.com/ibm-granite/granite-code-models
        # temperature=0.6, top_k=50, top_p=0.9, min_p=0.01, repetition_penalty=1.0
        "granite-code": ModelParams(
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            min_p=0.01,
            repetition_penalty=1.0,
        ),
        # IBM Granite 3.1 MoE: Based on Granite 3.3 official Replicate params
        # huggingface.co/ibm-granite/granite-3.3-8b-instruct-GGUF/discussions/2
        # temperature=0.6, top_k=50, top_p=0.9, min_p=0.01, repetition_penalty=1.0
        "granite3.1-moe": ModelParams(
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            min_p=0.01,
            repetition_penalty=1.0,
        ),
        # LFM2.5 Instruct: huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct
        # temperature=0.1, top_k=50, top_p=0.1, repetition_penalty=1.05
        "sam860/lfm2.5": ModelParams(
            temperature=0.1,
            top_p=0.1,
            top_k=50,
            repetition_penalty=1.05,
        ),
        # Official LFM2.5 Instruct from HuggingFace (same params)
        "hf.co/LiquidAI/LFM2.5": ModelParams(
            temperature=0.1,
            top_p=0.1,
            top_k=50,
            repetition_penalty=1.05,
        ),
        # LFM2 (all sizes): huggingface.co/LiquidAI/LFM2-8B-A1B
        # temperature=0.3, min_p=0.15, repetition_penalty=1.05
        "sam860/lfm2": ModelParams(
            temperature=0.3,
            min_p=0.15,
            repetition_penalty=1.05,
        ),
        # LFM2.5 Thinking: huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking
        # Default temp=0.05, inference example uses 0.1
        # Higher max_tokens needed: model outputs <think>...</think> before summary
        "lfm2.5-thinking": ModelParams(
            temperature=0.1,
            top_p=0.1,
            top_k=50,
            repetition_penalty=1.05,
            max_tokens=1024,  # Thinking models need more tokens
        ),
        # Falcon H1 Instruct models: Hybrid Transformer + Mamba-2 SSM architecture
        # huggingface.co/spaces/tiiuae/Falcon-H1-playground/discussions/1
        # "For optimal performance, the recommended model temperature is 0.1"
        "falcon-h1": ModelParams(
            temperature=0.1,
        ),
    })

    def get_model(self, model_ref: str) -> ModelConfig:
        """Get a model configuration by reference name."""
        if model_ref not in self.models:
            raise KeyError(f"Benchmark model '{model_ref}' not found. Available: {list(self.models.keys())}")
        return self.models[model_ref]

    def get_benchmark_models(self) -> list[ModelConfig]:
        """Get all models to benchmark."""
        return [self.get_model(ref) for ref in self.benchmark_models]

    def get_eval_model_configs(self) -> list[ModelConfig]:
        """Get all evaluation model configs."""
        return [self.get_model(ref) for ref in self.eval_models]

    def get_model_params(self, model_name: str) -> ModelParams:
        """Get parameters for a specific model.

        Matches model_name against keys using startswith() for flexibility.
        Falls back to default ModelParams if no match.
        """
        for pattern, params in self.model_params.items():
            if model_name.startswith(pattern):
                return params
        return ModelParams(temperature=self.temperature)


@dataclass
class MagaldiConfig:
    """Root configuration object containing all sections."""

    elasticsearch: ElasticsearchConfig = field(default_factory=ElasticsearchConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    workers: WorkersConfig = field(default_factory=WorkersConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    web: WebConfig = field(default_factory=WebConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    user_data: UserDataConfig = field(default_factory=UserDataConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)


# =============================================================================
# GLOBAL STATE
# =============================================================================

_config: MagaldiConfig | None = None


# =============================================================================
# PUBLIC API
# =============================================================================


def load_config(
    config_path: str | Path | None = None,
    skip_validation: bool = False,
) -> MagaldiConfig:
    """Load configuration from file and environment variables.

    Configuration priority (highest to lowest):
    1. Environment variables (MAGALDI_*)
    2. .env file
    3. Configuration file
    4. Dataclass defaults

    Args:
        config_path: Path to configuration file. If None, searches standard locations.
        skip_validation: If True, skip validation (useful for dry-run mode).

    Returns:
        Loaded and validated MagaldiConfig instance.

    Raises:
        ConfigurationError: If configuration is invalid.
    """
    global _config

    if _config is not None:
        return _config

    # Load .env file if present (doesn't override existing env vars)
    # Skip in tests by setting MAGALDI_SKIP_DOTENV=1
    if not os.environ.get("MAGALDI_SKIP_DOTENV"):
        load_dotenv()

    # Start with defaults
    config = MagaldiConfig()

    # Find and load config file
    if config_path is not None:
        config_path = Path(config_path)
    else:
        config_path = _find_config_file()

    if config_path and config_path.exists():
        config = _load_from_file(config, config_path)

    # Apply environment variable overrides
    config = _apply_env_overrides(config)

    # Validate (unless skipped)
    if not skip_validation:
        _validate_config(config)

    _config = config
    return config


def get_config() -> MagaldiConfig:
    """Get the loaded configuration.

    Must call load_config() first.

    Returns:
        The cached MagaldiConfig instance.

    Raises:
        RuntimeError: If load_config() has not been called.
    """
    if _config is None:
        raise RuntimeError(
            "Configuration not loaded. Call load_config() first at application startup."
        )
    return _config


def reset_config() -> None:
    """Reset the cached configuration.

    Primarily used for testing to ensure a clean state between tests.
    """
    global _config
    _config = None


# =============================================================================
# PRIVATE HELPERS
# =============================================================================


def _find_config_file() -> Path | None:
    """Find configuration file in standard locations.

    Search order:
    1. MAGALDI_CONFIG environment variable
    2. ~/.magaldi/config.yaml
    3. /etc/magaldi/config.yaml
    4. ./config/magaldi.yaml
    5. ./magaldi.yaml

    Returns:
        Path to config file if found, None otherwise.
    """
    locations: list[Path | None] = [
        Path(os.environ["MAGALDI_CONFIG"]) if "MAGALDI_CONFIG" in os.environ else None,
        Path.home() / ".magaldi" / "config.yaml",
        Path("/etc/magaldi/config.yaml"),
        Path("./config/magaldi.yaml"),
        Path("./magaldi.yaml"),
    ]

    for loc in locations:
        if loc and loc.exists():
            return loc

    return None


def _load_from_file(config: MagaldiConfig, config_path: Path) -> MagaldiConfig:
    """Load configuration from YAML file.

    Args:
        config: Base config with defaults.
        config_path: Path to YAML file.

    Returns:
        Config with file values merged in.
    """
    with open(config_path, encoding="utf-8") as f:
        file_config = yaml.safe_load(f) or {}

    return _merge_config(config, file_config)


def _merge_config(config: MagaldiConfig, file_config: dict[str, Any]) -> MagaldiConfig:
    """Merge file configuration into dataclass config.

    Args:
        config: Base config with defaults.
        file_config: Dictionary from YAML file.

    Returns:
        Config with file values merged in.
    """
    section_mapping = {
        "elasticsearch": config.elasticsearch,
        "redis": config.redis,
        "parser": config.parser,
        "search": config.search,
        "web": config.web,
        "mcp": config.mcp,
        "logging": config.logging,
        "user_data": config.user_data,
    }

    for section_name, section_obj in section_mapping.items():
        if section_name in file_config:
            for key, value in file_config[section_name].items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)

    # Handle nested workers config
    if "workers" in file_config:
        workers_config = file_config["workers"]
        if "summarization" in workers_config:
            for key, value in workers_config["summarization"].items():
                if hasattr(config.workers.summarization, key):
                    setattr(config.workers.summarization, key, value)
        if "embedding" in workers_config:
            for key, value in workers_config["embedding"].items():
                if hasattr(config.workers.embedding, key):
                    setattr(config.workers.embedding, key, value)

    # Handle LLM config with nested models
    if "llm" in file_config:
        llm_config = file_config["llm"]

        # Handle models dict specially
        if "models" in llm_config:
            for model_name, model_data in llm_config["models"].items():
                if isinstance(model_data, dict):
                    config.llm.models[model_name] = ModelConfig(**model_data)

        # Handle other LLM fields
        for key, value in llm_config.items():
            if key != "models" and hasattr(config.llm, key):
                setattr(config.llm, key, value)

    # Handle benchmark config with nested models
    if "benchmark" in file_config:
        bench_config = file_config["benchmark"]

        # Handle models dict specially
        if "models" in bench_config:
            for model_name, model_data in bench_config["models"].items():
                if isinstance(model_data, dict):
                    config.benchmark.models[model_name] = ModelConfig(**model_data)

        # Handle other benchmark fields (excluding nested dicts that need special handling)
        skip_keys = {"models", "model_params"}
        for key, value in bench_config.items():
            if key not in skip_keys and hasattr(config.benchmark, key):
                setattr(config.benchmark, key, value)

        # Handle model_params specially (dict of ModelParams)
        if "model_params" in bench_config:
            for model_pattern, params_data in bench_config["model_params"].items():
                if isinstance(params_data, dict):
                    config.benchmark.model_params[model_pattern] = ModelParams(**params_data)

    return config


def _apply_env_overrides(config: MagaldiConfig) -> MagaldiConfig:
    """Apply environment variable overrides.

    Environment variables follow the pattern: MAGALDI_{SECTION}_{KEY}
    Type conversion is handled automatically.

    Args:
        config: Config to apply overrides to.

    Returns:
        Config with environment overrides applied.
    """
    # Define mappings: env_var -> (section_attr, key, converter)
    env_mappings: dict[str, tuple[str, str] | tuple[str, str, type]] = {
        # Elasticsearch
        "MAGALDI_ELASTICSEARCH_HOST": ("elasticsearch", "host"),
        "MAGALDI_ELASTICSEARCH_PORT": ("elasticsearch", "port", int),
        "MAGALDI_ELASTICSEARCH_SCHEME": ("elasticsearch", "scheme"),
        "MAGALDI_ELASTICSEARCH_INDEX": ("elasticsearch", "index"),
        "MAGALDI_ELASTICSEARCH_TIMEOUT": ("elasticsearch", "timeout", int),
        "MAGALDI_ELASTICSEARCH_BULK_TIMEOUT": ("elasticsearch", "bulk_timeout", int),
        # Redis
        "MAGALDI_REDIS_HOST": ("redis", "host"),
        "MAGALDI_REDIS_PORT": ("redis", "port", int),
        "MAGALDI_REDIS_DB": ("redis", "db", int),
        "MAGALDI_REDIS_PASSWORD": ("redis", "password"),
        # LLM configuration - Model references
        "MAGALDI_LLM_SUMMARIZE_MODEL": ("llm", "summarize_model"),
        "MAGALDI_LLM_SUMMARIZE_MODEL_SMALL": ("llm", "summarize_model_small"),
        "MAGALDI_LLM_EMBED_MODEL": ("llm", "embed_model"),
        # Logging
        "MAGALDI_LOG_LEVEL": ("logging", "level"),
        # Web
        "MAGALDI_WEB_HOST": ("web", "host"),
        "MAGALDI_WEB_PORT": ("web", "port", int),
    }

    for env_var, mapping in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            section_name = mapping[0]
            attr_name = mapping[1]
            converter = mapping[2] if len(mapping) > 2 else str

            section = getattr(config, section_name)
            setattr(section, attr_name, converter(value))

    return config


def _validate_config(config: MagaldiConfig) -> None:
    """Validate configuration values.

    Args:
        config: Config to validate.

    Raises:
        ConfigurationError: If validation fails.
    """
    errors: list[str] = []

    # Validate model references exist
    try:
        config.llm.get_summarize_model()
    except KeyError as e:
        errors.append(f"Invalid summarize_model reference: {e}")

    try:
        config.llm.get_summarize_model_small()
    except KeyError as e:
        errors.append(f"Invalid summarize_model_small reference: {e}")

    try:
        embed_model = config.llm.get_embed_model()
        # Consistency check for snowflake model
        if embed_model.name == "snowflake-arctic-embed2" and config.llm.embed_dimensions != 1024:
            errors.append(
                f"snowflake-arctic-embed2 requires embed_dimensions=1024, "
                f"got {config.llm.embed_dimensions}"
            )
    except KeyError as e:
        errors.append(f"Invalid embed_model reference: {e}")

    # Port range checks
    if not (1 <= config.elasticsearch.port <= 65535):
        errors.append(f"Elasticsearch port must be between 1 and 65535, got {config.elasticsearch.port}")

    if not (1 <= config.redis.port <= 65535):
        errors.append(f"Redis port must be between 1 and 65535, got {config.redis.port}")

    if not (1 <= config.web.port <= 65535):
        errors.append(f"Web port must be between 1 and 65535, got {config.web.port}")

    if errors:
        raise ConfigurationError("\n".join(errors))
