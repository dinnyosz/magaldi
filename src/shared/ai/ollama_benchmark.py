"""Direct Ollama API client for benchmarking with detailed timing.

This module bypasses LiteLLM to get detailed timing information from Ollama,
including prefill time and token generation time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    model: str
    success: bool
    response: str = ""
    error: str = ""

    # Timing in seconds
    load_time: float = 0.0  # Time to load model (usually 0 if warm)
    prefill_time: float = 0.0  # Time to process input prompt
    generate_time: float = 0.0  # Time to generate output
    total_time: float = 0.0  # Wall clock time (measured by us)
    ollama_total_time: float = 0.0  # Total time reported by Ollama

    # Token counts
    prompt_tokens: int = 0  # Input tokens
    output_tokens: int = 0  # Generated tokens

    @property
    def tokens_per_second(self) -> float:
        """Calculate generation speed in tokens/second."""
        if self.generate_time > 0:
            return self.output_tokens / self.generate_time
        return 0.0

    @property
    def prefill_tokens_per_second(self) -> float:
        """Calculate prefill speed in tokens/second."""
        if self.prefill_time > 0:
            return self.prompt_tokens / self.prefill_time
        return 0.0


class OllamaBenchmarkClient:
    """Direct Ollama API client for benchmarking.

    Calls Ollama's /api/generate endpoint directly to get detailed timing:
    - prompt_eval_duration: Time to process input (prefill)
    - eval_duration: Time to generate output tokens
    - prompt_eval_count: Number of input tokens
    - eval_count: Number of output tokens
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize the client.

        Args:
            base_url: Ollama server URL.
        """
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def check_connection(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Get list of available models."""
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def warmup(self, model: str, timeout: int = 120) -> tuple[bool, float, str]:
        """Warm up a model by sending a tiny request.

        This loads model weights into GPU memory.

        Args:
            model: Model name (e.g., "qwen2.5-coder:3b")
            timeout: Timeout in seconds.

        Returns:
            Tuple of (success, warmup_time_seconds, error_message)
        """
        start = time.perf_counter()
        try:
            resp = self._session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": "Hi",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
                timeout=timeout,
            )
            elapsed = time.perf_counter() - start

            if resp.status_code == 200:
                return True, elapsed, ""
            else:
                error = resp.json().get("error", f"HTTP {resp.status_code}")
                return False, elapsed, error

        except requests.Timeout:
            elapsed = time.perf_counter() - start
            return False, elapsed, f"Timeout after {timeout}s"
        except Exception as e:
            elapsed = time.perf_counter() - start
            return False, elapsed, str(e)

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 512,
        timeout: int = 120,
    ) -> BenchmarkResult:
        """Generate completion and return detailed timing.

        Args:
            model: Model name.
            prompt: The prompt to send.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.

        Returns:
            BenchmarkResult with timing and token information.
        """
        start = time.perf_counter()

        try:
            resp = self._session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_predict": max_tokens,
                    },
                },
                timeout=timeout,
            )
            total_time = time.perf_counter() - start

            if resp.status_code != 200:
                error = resp.json().get("error", f"HTTP {resp.status_code}")
                return BenchmarkResult(
                    model=model,
                    success=False,
                    error=error,
                    total_time=total_time,
                )

            data = resp.json()

            # Extract timing (Ollama returns nanoseconds)
            load_ns = data.get("load_duration", 0)
            prompt_eval_ns = data.get("prompt_eval_duration", 0)
            eval_ns = data.get("eval_duration", 0)
            ollama_total_ns = data.get("total_duration", 0)

            return BenchmarkResult(
                model=model,
                success=True,
                response=data.get("response", "").strip(),
                load_time=load_ns / 1e9,
                prefill_time=prompt_eval_ns / 1e9,
                generate_time=eval_ns / 1e9,
                total_time=total_time,
                ollama_total_time=ollama_total_ns / 1e9,
                prompt_tokens=data.get("prompt_eval_count", 0),
                output_tokens=data.get("eval_count", 0),
            )

        except requests.Timeout:
            total_time = time.perf_counter() - start
            return BenchmarkResult(
                model=model,
                success=False,
                error=f"Timeout after {timeout}s",
                total_time=total_time,
            )
        except Exception as e:
            total_time = time.perf_counter() - start
            return BenchmarkResult(
                model=model,
                success=False,
                error=str(e),
                total_time=total_time,
            )

    def close(self) -> None:
        """Close the session."""
        self._session.close()
