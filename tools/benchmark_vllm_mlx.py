#!/usr/bin/env python3
"""Benchmark: vllm-mlx vs Ollama for Magaldi's summarization workload.

Measures what matters for Magaldi:
  1. Single-request latency & throughput (tok/s)
  2. Parallel request throughput (N concurrent workers)
  3. Embedding throughput

Usage:
  # Start both servers first:
  #   Ollama:   ollama serve  (default :11434)
  #   vllm-mlx: vllm-mlx serve mlx-community/Qwen3-4B-Instruct-2507-4bit \
  #               --port 8000 --continuous-batching \
  #               --embedding-model mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ

  python tools/benchmark_vllm_mlx.py
  python tools/benchmark_vllm_mlx.py --workers 1 2 4 8
  python tools/benchmark_vllm_mlx.py --ollama-only
  python tools/benchmark_vllm_mlx.py --vllm-only
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434"
VLLM_URL = "http://localhost:8000"

OLLAMA_MODEL = "ollama/qwen3:4b-instruct"
VLLM_MODEL = "openai/default"  # vllm-mlx uses "default" for loaded model

# Realistic Magaldi summarization prompt (function-level)
SUMMARIZATION_PROMPT = """\
Summarize this Python function in 4-6 sentences. Start with an action verb. \
Focus on what it does, its inputs/outputs, and when to call it. \
Do NOT start with "This function...".

File: src/shared/parallel_processor.py
File context: Parallel processing engine with adaptive throttling for AI workloads.

```python
def process(
    self,
    items: list[T],
    timeout: float | None = None,
) -> list[R]:
    \"\"\"Process items in parallel with tier-based throttling.

    Items are grouped by tier and processed with appropriate concurrency
    limits. Progress callbacks fire on each completion.

    Args:
        items: Items to process.
        timeout: Optional timeout in seconds for the entire batch.

    Returns:
        List of results in input order.
    \"\"\"
    if not items:
        return []

    # Group by tier
    tier_groups: dict[int, list[tuple[int, T]]] = {}
    for idx, item in enumerate(items):
        tier = self.tier_fn(item)
        tier_groups.setdefault(tier, []).append((idx, item))

    results: list[R | None] = [None] * len(items)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        futures = {}
        for tier in sorted(tier_groups.keys()):
            max_w = TIER_MAX_WORKERS.get(tier, self.max_workers)
            for idx, item in tier_groups[tier]:
                worker_id = self._acquire_worker_id()
                future = executor.submit(
                    process_wrapper, self, item, worker_id, idx
                )
                futures[future] = (idx, worker_id)

        for future in as_completed(futures):
            idx, worker_id = futures[future]
            try:
                result = future.result()
                results[idx] = result
                success = self.is_success(result)
                if self.on_result:
                    self.on_result(items[idx], result, success)
            except Exception as e:
                pass
            finally:
                self._release_worker_id(worker_id)

    return [r for r in results if r is not None]
```
"""

# Short prompt for embedding benchmark
EMBEDDING_TEXTS = [
    "Process items in parallel with tier-based throttling and adaptive concurrency limits.",
    "Initialize the processor with max workers, tier function, and process function callbacks.",
    "Acquire a worker ID from the available pool using thread-safe locking.",
    "Release a worker ID back to the pool after processing completes.",
    "Calculate throughput statistics with concurrency metrics and timing data.",
    "Group items by tier and submit to thread pool executor with appropriate limits.",
    "Track worker status including active runtime and processing state.",
    "Handle progress callbacks with completion percentage and ETA estimates.",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SingleResult:
    """Result from a single LLM request."""

    backend: str
    success: bool
    latency: float = 0.0  # seconds
    prompt_tokens: int = 0
    output_tokens: int = 0
    output_text: str = ""
    error: str = ""

    @property
    def tok_per_sec(self) -> float:
        if self.latency > 0 and self.output_tokens > 0:
            return self.output_tokens / self.latency
        return 0.0


@dataclass
class ConcurrencyResult:
    """Aggregate result from N concurrent requests."""

    backend: str
    concurrency: int
    results: list[SingleResult] = field(default_factory=list)

    @property
    def total_wall_time(self) -> float:
        """Wall time for entire batch (max latency since parallel)."""
        if not self.results:
            return 0.0
        return max(r.latency for r in self.results)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.results if r.success)

    @property
    def aggregate_tok_per_sec(self) -> float:
        """Total tokens generated / wall time."""
        wt = self.total_wall_time
        if wt > 0:
            return self.total_output_tokens / wt
        return 0.0

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    @property
    def median_latency(self) -> float:
        lats = [r.latency for r in self.results if r.success]
        return statistics.median(lats) if lats else 0.0

    @property
    def p95_latency(self) -> float:
        lats = sorted(r.latency for r in self.results if r.success)
        if not lats:
            return 0.0
        idx = int(len(lats) * 0.95)
        return lats[min(idx, len(lats) - 1)]


# ---------------------------------------------------------------------------
# Server checks
# ---------------------------------------------------------------------------


def check_ollama(url: str = OLLAMA_URL) -> bool:
    try:
        r = requests.get(f"{url}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def check_vllm(url: str = VLLM_URL) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Single request
# ---------------------------------------------------------------------------


def run_single_request(
    backend: str,
    model: str,
    api_base: str,
    prompt: str = SUMMARIZATION_PROMPT,
    max_tokens: int = 512,
    temperature: float = 0.2,
    timeout: int = 120,
    api_key: str | None = None,
) -> SingleResult:
    """Fire a single LiteLLM completion and measure timing."""
    import litellm

    litellm.telemetry = False
    litellm.drop_params = True

    start = time.perf_counter()
    try:
        kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }

        # Disable thinking for qwen3
        if "qwen3" in model.lower():
            kwargs["think"] = False

        if api_base:
            kwargs["api_base"] = api_base

        # vllm-mlx uses openai/ prefix which requires an API key
        if api_key:
            kwargs["api_key"] = api_key

        response = litellm.completion(**kwargs)
        elapsed = time.perf_counter() - start

        content = response.choices[0].message.content or ""
        usage = response.usage
        return SingleResult(
            backend=backend,
            success=True,
            latency=elapsed,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            output_text=content.strip(),
        )
    except Exception as e:
        elapsed = time.perf_counter() - start
        return SingleResult(
            backend=backend,
            success=False,
            latency=elapsed,
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Concurrent requests
# ---------------------------------------------------------------------------


def run_concurrent(
    backend: str,
    model: str,
    api_base: str,
    concurrency: int,
    prompt: str = SUMMARIZATION_PROMPT,
    max_tokens: int = 512,
    api_key: str | None = None,
) -> ConcurrencyResult:
    """Fire N concurrent requests and measure aggregate throughput."""
    results: list[SingleResult] = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                run_single_request,
                backend=backend,
                model=model,
                api_base=api_base,
                prompt=prompt,
                max_tokens=max_tokens,
                api_key=api_key,
            )
            for _ in range(concurrency)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    return ConcurrencyResult(
        backend=backend,
        concurrency=concurrency,
        results=results,
    )


# ---------------------------------------------------------------------------
# Embedding benchmark
# ---------------------------------------------------------------------------


def run_embedding_benchmark(
    backend: str,
    model: str,
    api_base: str,
    texts: list[str],
    batch_size: int = 8,
    api_key: str | None = None,
) -> dict:
    """Benchmark embedding throughput."""
    import litellm

    litellm.telemetry = False
    litellm.drop_params = True

    start = time.perf_counter()
    try:
        kwargs: dict = {
            "model": model,
            "input": texts[:batch_size],
            "api_base": api_base,
            "timeout": 30,
        }
        if api_key:
            kwargs["api_key"] = api_key
        response = litellm.embedding(**kwargs)
        elapsed = time.perf_counter() - start
        dims = len(response.data[0]["embedding"]) if response.data else 0
        return {
            "backend": backend,
            "success": True,
            "texts": len(texts[:batch_size]),
            "dimensions": dims,
            "latency": elapsed,
            "texts_per_sec": len(texts[:batch_size]) / elapsed if elapsed > 0 else 0,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "backend": backend,
            "success": False,
            "error": str(e),
            "latency": elapsed,
        }


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def warmup(backend: str, model: str, api_base: str, api_key: str | None = None) -> bool:
    """Send a tiny request to warm up model loading."""
    print(f"  Warming up {backend}...", end=" ", flush=True)
    r = run_single_request(
        backend=backend,
        model=model,
        api_base=api_base,
        prompt="Say hi",
        max_tokens=5,
        timeout=120,
        api_key=api_key,
    )
    if r.success:
        print(f"OK ({r.latency:.1f}s)")
        return True
    else:
        print(f"FAILED: {r.error}")
        return False


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_single_results(results: list[SingleResult]) -> None:
    """Print single-request comparison table."""
    print()
    print(f"  {'Backend':<12} {'Latency':>9} {'Out Tok':>8} {'Tok/s':>8} {'Prompt Tok':>11}")
    print(f"  {'─' * 12} {'─' * 9} {'─' * 8} {'─' * 8} {'─' * 11}")
    for r in results:
        if r.success:
            print(
                f"  {r.backend:<12} {r.latency:>8.2f}s {r.output_tokens:>8} "
                f"{r.tok_per_sec:>7.1f} {r.prompt_tokens:>11}"
            )
        else:
            print(f"  {r.backend:<12} {'FAILED':>9}  {r.error[:40]}")


def print_concurrency_results(results: list[ConcurrencyResult]) -> None:
    """Print concurrency comparison table."""
    print()
    print(
        f"  {'Backend':<12} {'Workers':>8} {'Wall Time':>10} {'Tot Tok':>8} "
        f"{'Agg Tok/s':>10} {'Med Lat':>8} {'P95 Lat':>8} {'Success':>8}"
    )
    print(
        f"  {'─' * 12} {'─' * 8} {'─' * 10} {'─' * 8} "
        f"{'─' * 10} {'─' * 8} {'─' * 8} {'─' * 8}"
    )
    for r in results:
        print(
            f"  {r.backend:<12} {r.concurrency:>8} {r.total_wall_time:>9.2f}s "
            f"{r.total_output_tokens:>8} {r.aggregate_tok_per_sec:>9.1f} "
            f"{r.median_latency:>7.2f}s {r.p95_latency:>7.2f}s "
            f"{r.success_rate:>7.0%}"
        )


def print_embedding_results(results: list[dict]) -> None:
    """Print embedding comparison table."""
    print()
    print(f"  {'Backend':<12} {'Texts':>6} {'Dims':>6} {'Latency':>9} {'Texts/s':>8}")
    print(f"  {'─' * 12} {'─' * 6} {'─' * 6} {'─' * 9} {'─' * 8}")
    for r in results:
        if r.get("success"):
            print(
                f"  {r['backend']:<12} {r['texts']:>6} {r['dimensions']:>6} "
                f"{r['latency']:>8.3f}s {r['texts_per_sec']:>7.1f}"
            )
        else:
            print(f"  {r['backend']:<12} {'FAILED':>6}  {r.get('error', '')[:40]}")


def print_speedup(
    ollama_results: list[ConcurrencyResult],
    vllm_results: list[ConcurrencyResult],
) -> None:
    """Print speedup comparison."""
    print()
    print(f"  {'Workers':>8} {'Ollama Tok/s':>13} {'vllm-mlx Tok/s':>15} {'Speedup':>8}")
    print(f"  {'─' * 8} {'─' * 13} {'─' * 15} {'─' * 8}")

    ollama_by_c = {r.concurrency: r for r in ollama_results}
    vllm_by_c = {r.concurrency: r for r in vllm_results}

    for c in sorted(set(ollama_by_c.keys()) | set(vllm_by_c.keys())):
        o = ollama_by_c.get(c)
        v = vllm_by_c.get(c)
        o_tps = o.aggregate_tok_per_sec if o else 0
        v_tps = v.aggregate_tok_per_sec if v else 0
        speedup = v_tps / o_tps if o_tps > 0 else 0
        print(
            f"  {c:>8} {o_tps:>12.1f} {v_tps:>14.1f} {speedup:>7.1f}x"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark vllm-mlx vs Ollama")
    parser.add_argument(
        "--workers",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="Concurrency levels to test (default: 1 2 4 8)",
    )
    parser.add_argument("--ollama-only", action="store_true", help="Benchmark Ollama only")
    parser.add_argument("--vllm-only", action="store_true", help="Benchmark vllm-mlx only")
    parser.add_argument("--runs", type=int, default=3, help="Runs per single-request test (default: 3)")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding benchmark")
    parser.add_argument(
        "--ollama-url", default=OLLAMA_URL, help=f"Ollama URL (default: {OLLAMA_URL})"
    )
    parser.add_argument(
        "--vllm-url", default=VLLM_URL, help=f"vllm-mlx URL (default: {VLLM_URL})"
    )
    parser.add_argument(
        "--ollama-model", default=OLLAMA_MODEL, help=f"Ollama model (default: {OLLAMA_MODEL})"
    )
    parser.add_argument(
        "--vllm-model", default=VLLM_MODEL, help=f"vllm-mlx model (default: {VLLM_MODEL})"
    )
    args = parser.parse_args()

    # Determine which backends to test
    test_ollama = not args.vllm_only
    test_vllm = not args.ollama_only

    # Dummy API key for vllm-mlx (OpenAI-compatible endpoint needs one)
    VLLM_API_KEY = "not-needed"

    # (name, model, api_base, api_key)
    backends: list[tuple[str, str, str, str | None]] = []

    if test_ollama:
        if check_ollama(args.ollama_url):
            backends.append(("ollama", args.ollama_model, args.ollama_url, None))
            print(f"✓ Ollama available at {args.ollama_url}")
        else:
            print(f"✗ Ollama not available at {args.ollama_url}")
            if not test_vllm:
                sys.exit(1)

    if test_vllm:
        if check_vllm(args.vllm_url):
            backends.append(("vllm-mlx", args.vllm_model, f"{args.vllm_url}/v1", VLLM_API_KEY))
            print(f"✓ vllm-mlx available at {args.vllm_url}")
        else:
            print(f"✗ vllm-mlx not available at {args.vllm_url}")
            if not test_ollama:
                sys.exit(1)

    if not backends:
        print("\nNo backends available. Start at least one server.")
        sys.exit(1)

    # Warmup
    print()
    active_backends = []
    for name, model, api_base, api_key in backends:
        if warmup(name, model, api_base, api_key=api_key):
            active_backends.append((name, model, api_base, api_key))

    if not active_backends:
        print("\nAll warmups failed.")
        sys.exit(1)

    # ── Test 1: Single-request latency ────────────────────────────────────
    print_header("Test 1: Single-Request Latency (best of {})".format(args.runs))

    single_results: list[SingleResult] = []
    for name, model, api_base, api_key in active_backends:
        best: SingleResult | None = None
        for run in range(args.runs):
            r = run_single_request(
                backend=name, model=model, api_base=api_base, api_key=api_key,
            )
            if best is None or (r.success and r.latency < (best.latency if best.success else float("inf"))):
                best = r
        if best:
            single_results.append(best)

    print_single_results(single_results)

    # ── Test 2: Parallel throughput ────────────────────────────────────────
    print_header("Test 2: Parallel Throughput (concurrent workers)")

    ollama_concurrent: list[ConcurrencyResult] = []
    vllm_concurrent: list[ConcurrencyResult] = []

    for concurrency in args.workers:
        for name, model, api_base, api_key in active_backends:
            print(f"  Running {name} with {concurrency} workers...", end=" ", flush=True)
            cr = run_concurrent(
                backend=name,
                model=model,
                api_base=api_base,
                concurrency=concurrency,
                api_key=api_key,
            )
            print(f"{cr.aggregate_tok_per_sec:.1f} tok/s")
            if name == "ollama":
                ollama_concurrent.append(cr)
            else:
                vllm_concurrent.append(cr)

    all_concurrent = ollama_concurrent + vllm_concurrent
    print_concurrency_results(all_concurrent)

    # Speedup table (if both backends tested)
    if ollama_concurrent and vllm_concurrent:
        print_header("Speedup: vllm-mlx vs Ollama")
        print_speedup(ollama_concurrent, vllm_concurrent)

    # ── Test 3: Embeddings ─────────────────────────────────────────────────
    if not args.skip_embeddings:
        print_header("Test 3: Embedding Throughput")

        embed_results: list[dict] = []
        for name, _model, api_base, api_key in active_backends:
            if name == "ollama":
                embed_model = "ollama/qwen3-embedding:0.6b"
                embed_base = args.ollama_url
            else:
                # Use openai/ prefix; the model name after the prefix is what
                # gets sent in the API request body
                embed_model = "openai/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
                embed_base = f"{args.vllm_url}/v1"
                # Ensure api_key is set for openai/ prefix
                api_key = api_key or "not-needed"

            r = run_embedding_benchmark(
                backend=name,
                model=embed_model,
                api_base=embed_base,
                texts=EMBEDDING_TEXTS,
                api_key=api_key,
            )
            embed_results.append(r)

        print_embedding_results(embed_results)

    # ── Summary ────────────────────────────────────────────────────────────
    print_header("Summary")
    if len(active_backends) == 2 and ollama_concurrent and vllm_concurrent:
        # Find the highest concurrency tested
        max_c = max(args.workers)
        o = next((r for r in ollama_concurrent if r.concurrency == max_c), None)
        v = next((r for r in vllm_concurrent if r.concurrency == max_c), None)
        if o and v and o.aggregate_tok_per_sec > 0:
            speedup = v.aggregate_tok_per_sec / o.aggregate_tok_per_sec
            print(f"  At {max_c} concurrent workers:")
            print(f"    Ollama:   {o.aggregate_tok_per_sec:>8.1f} tok/s aggregate")
            print(f"    vllm-mlx: {v.aggregate_tok_per_sec:>8.1f} tok/s aggregate")
            print(f"    Speedup:  {speedup:.1f}x")
            print()
            if speedup > 1.5:
                print("  → vllm-mlx shows significant advantage for parallel workloads")
            elif speedup > 1.0:
                print("  → vllm-mlx shows modest advantage for parallel workloads")
            else:
                print("  → No clear advantage for vllm-mlx at this concurrency level")
    elif len(active_backends) == 1:
        name = active_backends[0][0]
        concurrent = ollama_concurrent or vllm_concurrent
        if concurrent:
            max_c = max(r.concurrency for r in concurrent)
            r = next((r for r in concurrent if r.concurrency == max_c), None)
            if r:
                print(f"  {name} at {max_c} workers: {r.aggregate_tok_per_sec:.1f} tok/s aggregate")
    print()


if __name__ == "__main__":
    main()
