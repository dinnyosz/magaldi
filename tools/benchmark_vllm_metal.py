#!/usr/bin/env python3
"""Benchmark: vLLM Metal vs Ollama on Apple Silicon.

Measures head-to-head performance for local LLM inference:
  1. Single-request latency & tok/s (TTFT + generation)
  2. Concurrent request throughput (batched inference)
  3. Time-to-first-token (streaming)

Prerequisites:
  # Install vLLM Metal:
  curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
  source ~/.venv-vllm-metal/bin/activate

  # Start vLLM Metal server (in one terminal):
  vllm serve Qwen/Qwen2.5-3B-Instruct --port 8100 --max-model-len 4096

  # Ensure Ollama is running (in another terminal):
  ollama serve

Usage:
  python tools/benchmark_vllm_metal.py
  python tools/benchmark_vllm_metal.py --model Qwen/Qwen2.5-3B-Instruct --ollama-model qwen3:4b-instruct
  python tools/benchmark_vllm_metal.py --workers 1 2 4 8 16
  python tools/benchmark_vllm_metal.py --ollama-only
  python tools/benchmark_vllm_metal.py --vllm-only
  python tools/benchmark_vllm_metal.py --prompts short medium long
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from urllib.request import Request, urlopen
from urllib.error import URLError
from urllib.parse import urljoin

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434"
VLLM_URL = "http://localhost:8100"

DEFAULT_VLLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"

# ---------------------------------------------------------------------------
# Prompts of varying complexity (Magaldi-realistic)
# ---------------------------------------------------------------------------

PROMPTS = {
    "short": (
        "Summarize what this function does in 1-2 sentences. Start with an action verb.\n\n"
        "```python\n"
        "def get_cache_key(prefix: str, identifier: str) -> str:\n"
        '    return f"{prefix}:{identifier}"\n'
        "```"
    ),
    "medium": (
        "Summarize this Python function in 3-5 sentences. Start with an action verb. "
        "Focus on what it does, its inputs/outputs, and when to call it.\n\n"
        "```python\n"
        "def process(\n"
        "    self,\n"
        "    items: list[T],\n"
        "    timeout: float | None = None,\n"
        ") -> list[R]:\n"
        '    """Process items in parallel with tier-based throttling."""\n'
        "    if not items:\n"
        "        return []\n"
        "    tier_groups: dict[int, list[tuple[int, T]]] = {}\n"
        "    for idx, item in enumerate(items):\n"
        "        tier = self.tier_fn(item)\n"
        "        tier_groups.setdefault(tier, []).append((idx, item))\n"
        "    results: list[R | None] = [None] * len(items)\n"
        "    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:\n"
        "        futures = {}\n"
        "        for tier in sorted(tier_groups.keys()):\n"
        "            max_w = TIER_MAX_WORKERS.get(tier, self.max_workers)\n"
        "            for idx, item in tier_groups[tier]:\n"
        "                future = executor.submit(process_wrapper, self, item, idx)\n"
        "                futures[future] = idx\n"
        "        for future in as_completed(futures):\n"
        "            idx = futures[future]\n"
        "            results[idx] = future.result()\n"
        "    return [r for r in results if r is not None]\n"
        "```"
    ),
    "long": (
        "Analyze this Python class. Describe its purpose, key methods, design patterns, "
        "and how the components interact. 5-8 sentences.\n\n"
        "```python\n"
        "class ParallelProcessor(Generic[T, R]):\n"
        '    """Parallel processing engine with adaptive throttling for AI workloads.\n\n'
        "    Groups items by tier (priority level) and processes them with different\n"
        "    concurrency limits. Supports progress callbacks, timeout, and retry.\n"
        '    """\n\n'
        "    def __init__(\n"
        "        self,\n"
        "        process_fn: Callable[[T], R],\n"
        "        tier_fn: Callable[[T], int] = lambda _: 0,\n"
        "        max_workers: int = 4,\n"
        "        on_result: Callable[[T, R, bool], None] | None = None,\n"
        "        is_success: Callable[[R], bool] = lambda _: True,\n"
        "        retry_count: int = 0,\n"
        "        retry_delay: float = 1.0,\n"
        "    ):\n"
        "        self.process_fn = process_fn\n"
        "        self.tier_fn = tier_fn\n"
        "        self.max_workers = max_workers\n"
        "        self.on_result = on_result\n"
        "        self.is_success = is_success\n"
        "        self.retry_count = retry_count\n"
        "        self.retry_delay = retry_delay\n"
        "        self._worker_ids: set[int] = set(range(max_workers))\n"
        "        self._lock = threading.Lock()\n\n"
        "    def _acquire_worker_id(self) -> int:\n"
        "        with self._lock:\n"
        "            return self._worker_ids.pop()\n\n"
        "    def _release_worker_id(self, wid: int) -> None:\n"
        "        with self._lock:\n"
        "            self._worker_ids.add(wid)\n\n"
        "    def process(self, items: list[T], timeout: float | None = None) -> list[R]:\n"
        "        if not items:\n"
        "            return []\n"
        "        tier_groups: dict[int, list[tuple[int, T]]] = {}\n"
        "        for idx, item in enumerate(items):\n"
        "            tier = self.tier_fn(item)\n"
        "            tier_groups.setdefault(tier, []).append((idx, item))\n"
        "        results: list[R | None] = [None] * len(items)\n"
        "        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:\n"
        "            futures = {}\n"
        "            for tier in sorted(tier_groups.keys()):\n"
        "                for idx, item in tier_groups[tier]:\n"
        "                    wid = self._acquire_worker_id()\n"
        "                    future = executor.submit(self._run_with_retry, item, wid)\n"
        "                    futures[future] = (idx, wid)\n"
        "            for future in as_completed(futures):\n"
        "                idx, wid = futures[future]\n"
        "                try:\n"
        "                    result = future.result()\n"
        "                    results[idx] = result\n"
        "                    if self.on_result:\n"
        "                        self.on_result(items[idx], result, self.is_success(result))\n"
        "                finally:\n"
        "                    self._release_worker_id(wid)\n"
        "        return [r for r in results if r is not None]\n\n"
        "    def _run_with_retry(self, item: T, wid: int) -> R:\n"
        "        last_err = None\n"
        "        for attempt in range(self.retry_count + 1):\n"
        "            try:\n"
        "                return self.process_fn(item)\n"
        "            except Exception as e:\n"
        "                last_err = e\n"
        "                if attempt < self.retry_count:\n"
        "                    time.sleep(self.retry_delay * (attempt + 1))\n"
        "        raise last_err\n"
        "```"
    ),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SingleResult:
    """Result from a single LLM request."""
    backend: str
    success: bool
    latency: float = 0.0
    ttft: float = 0.0  # time to first token
    prompt_tokens: int = 0
    output_tokens: int = 0
    output_text: str = ""
    error: str = ""

    @property
    def generation_tok_per_sec(self) -> float:
        """Output tokens / generation time (excludes prefill)."""
        gen_time = self.latency - self.ttft
        if gen_time > 0 and self.output_tokens > 0:
            return self.output_tokens / gen_time
        return 0.0

    @property
    def total_tok_per_sec(self) -> float:
        """Output tokens / total latency."""
        if self.latency > 0 and self.output_tokens > 0:
            return self.output_tokens / self.latency
        return 0.0


@dataclass
class ConcurrencyResult:
    """Aggregate result from N concurrent requests."""
    backend: str
    concurrency: int
    wall_time: float = 0.0
    results: list[SingleResult] = field(default_factory=list)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.results if r.success)

    @property
    def aggregate_tok_per_sec(self) -> float:
        if self.wall_time > 0:
            return self.total_output_tokens / self.wall_time
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
# HTTP helpers (zero dependencies outside stdlib)
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    """POST JSON and return parsed response."""
    data = json.dumps(payload).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _post_json_stream(url: str, payload: dict, timeout: int = 120):
    """POST JSON with streaming, yield parsed SSE data chunks."""
    data = json.dumps(payload).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        buffer = b""
        for chunk in iter(lambda: resp.read(1), b""):
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                if line.startswith(b"data: "):
                    text = line[6:].decode()
                    if text == "[DONE]":
                        return
                    yield json.loads(text)


# ---------------------------------------------------------------------------
# Server checks
# ---------------------------------------------------------------------------

def check_ollama(url: str = OLLAMA_URL) -> bool:
    try:
        req = Request(f"{url}/api/tags")
        with urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def check_vllm(url: str = VLLM_URL) -> bool:
    try:
        req = Request(f"{url}/health")
        with urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Single request (non-streaming, for latency + throughput)
# ---------------------------------------------------------------------------

def run_single_ollama(
    model: str,
    prompt: str,
    api_base: str = OLLAMA_URL,
    max_tokens: int = 512,
    temperature: float = 0.2,
    timeout: int = 120,
) -> SingleResult:
    """Single request to Ollama /api/chat."""
    url = f"{api_base}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    start = time.perf_counter()
    try:
        resp = _post_json(url, payload, timeout=timeout)
        elapsed = time.perf_counter() - start

        content = resp.get("message", {}).get("content", "")
        # Ollama returns timing in nanoseconds
        eval_count = resp.get("eval_count", 0)
        prompt_eval_count = resp.get("prompt_eval_count", 0)
        # prompt_eval_duration is prefill time in nanoseconds
        prompt_eval_ns = resp.get("prompt_eval_duration", 0)
        ttft = prompt_eval_ns / 1e9 if prompt_eval_ns else 0.0

        return SingleResult(
            backend="ollama",
            success=True,
            latency=elapsed,
            ttft=ttft,
            prompt_tokens=prompt_eval_count,
            output_tokens=eval_count,
            output_text=content.strip(),
        )
    except Exception as e:
        elapsed = time.perf_counter() - start
        return SingleResult(backend="ollama", success=False, latency=elapsed, error=str(e))


def run_single_vllm(
    model: str,
    prompt: str,
    api_base: str = VLLM_URL,
    max_tokens: int = 512,
    temperature: float = 0.2,
    timeout: int = 120,
) -> SingleResult:
    """Single request to vLLM Metal OpenAI-compatible /v1/chat/completions."""
    url = f"{api_base}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    start = time.perf_counter()
    try:
        resp = _post_json(url, payload, timeout=timeout)
        elapsed = time.perf_counter() - start

        choice = resp.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        usage = resp.get("usage", {})

        return SingleResult(
            backend="vllm-metal",
            success=True,
            latency=elapsed,
            ttft=0.0,  # non-streaming, no TTFT
            prompt_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            output_text=content.strip(),
        )
    except Exception as e:
        elapsed = time.perf_counter() - start
        return SingleResult(backend="vllm-metal", success=False, latency=elapsed, error=str(e))


# ---------------------------------------------------------------------------
# Streaming request (for TTFT measurement)
# ---------------------------------------------------------------------------

def run_streaming_vllm(
    model: str,
    prompt: str,
    api_base: str = VLLM_URL,
    max_tokens: int = 512,
    temperature: float = 0.2,
    timeout: int = 120,
) -> SingleResult:
    """Streaming request to vLLM to measure TTFT."""
    url = f"{api_base}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    start = time.perf_counter()
    ttft = 0.0
    chunks: list[str] = []
    try:
        for chunk_data in _post_json_stream(url, payload, timeout=timeout):
            if ttft == 0.0:
                ttft = time.perf_counter() - start
            delta = chunk_data.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                chunks.append(content)
            # Check for usage in final chunk
            usage = chunk_data.get("usage")

        elapsed = time.perf_counter() - start
        full_text = "".join(chunks)

        return SingleResult(
            backend="vllm-metal",
            success=True,
            latency=elapsed,
            ttft=ttft,
            prompt_tokens=usage.get("prompt_tokens", 0) if usage else 0,
            output_tokens=usage.get("completion_tokens", 0) if usage else len(full_text.split()),
            output_text=full_text.strip(),
        )
    except Exception as e:
        elapsed = time.perf_counter() - start
        return SingleResult(backend="vllm-metal", success=False, latency=elapsed, error=str(e))


def run_streaming_ollama(
    model: str,
    prompt: str,
    api_base: str = OLLAMA_URL,
    max_tokens: int = 512,
    temperature: float = 0.2,
    timeout: int = 120,
) -> SingleResult:
    """Streaming request to Ollama to measure TTFT."""
    url = f"{api_base}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    start = time.perf_counter()
    ttft = 0.0
    chunks: list[str] = []
    eval_count = 0
    prompt_eval_count = 0

    try:
        data = json.dumps(payload).encode()
        req = Request(url, data=data, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            buffer = b""
            for raw_chunk in iter(lambda: resp.read(4096), b""):
                buffer += raw_chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    msg = obj.get("message", {})
                    content = msg.get("content", "")
                    if content and ttft == 0.0:
                        ttft = time.perf_counter() - start
                    if content:
                        chunks.append(content)
                    if obj.get("done"):
                        eval_count = obj.get("eval_count", 0)
                        prompt_eval_count = obj.get("prompt_eval_count", 0)

        elapsed = time.perf_counter() - start
        full_text = "".join(chunks)

        return SingleResult(
            backend="ollama",
            success=True,
            latency=elapsed,
            ttft=ttft,
            prompt_tokens=prompt_eval_count,
            output_tokens=eval_count if eval_count else len(full_text.split()),
            output_text=full_text.strip(),
        )
    except Exception as e:
        elapsed = time.perf_counter() - start
        return SingleResult(backend="ollama", success=False, latency=elapsed, error=str(e))


# ---------------------------------------------------------------------------
# Concurrent requests
# ---------------------------------------------------------------------------

def run_concurrent(
    backend: str,
    run_fn,
    concurrency: int,
    prompt: str,
    **kwargs,
) -> ConcurrencyResult:
    """Fire N concurrent requests and measure aggregate throughput."""
    results: list[SingleResult] = []
    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_fn, prompt=prompt, **kwargs)
            for _ in range(concurrency)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    wall_time = time.perf_counter() - wall_start

    return ConcurrencyResult(
        backend=backend,
        concurrency=concurrency,
        wall_time=wall_time,
        results=results,
    )


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def warmup(name: str, run_fn, **kwargs) -> bool:
    """Send a tiny request to warm up model loading."""
    print(f"  Warming up {name}...", end=" ", flush=True)
    r = run_fn(prompt="Say hi", max_tokens=5, timeout=180, **kwargs)
    if r.success:
        print(f"OK ({r.latency:.1f}s)")
        return True
    else:
        print(f"FAILED: {r.error[:80]}")
        return False


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_header(title: str) -> None:
    print()
    print("=" * 74)
    print(f"  {title}")
    print("=" * 74)


def print_single_results(results: list[SingleResult], label: str = "") -> None:
    if label:
        print(f"\n  [{label}]")
    print(f"  {'Backend':<13} {'Latency':>8} {'TTFT':>7} {'Out Tok':>8} {'Gen Tok/s':>10} {'Tot Tok/s':>10}")
    print(f"  {'─' * 13} {'─' * 8} {'─' * 7} {'─' * 8} {'─' * 10} {'─' * 10}")
    for r in results:
        if r.success:
            ttft_str = f"{r.ttft:.3f}s" if r.ttft > 0 else "n/a"
            print(
                f"  {r.backend:<13} {r.latency:>7.2f}s {ttft_str:>7} {r.output_tokens:>8} "
                f"{r.generation_tok_per_sec:>9.1f} {r.total_tok_per_sec:>9.1f}"
            )
        else:
            print(f"  {r.backend:<13} {'FAILED':>8}  {r.error[:50]}")


def print_concurrency_table(results: list[ConcurrencyResult]) -> None:
    print()
    print(
        f"  {'Backend':<13} {'Workers':>8} {'Wall':>7} {'Tot Tok':>8} "
        f"{'Agg Tok/s':>10} {'Med Lat':>8} {'P95 Lat':>8} {'OK':>5}"
    )
    print(
        f"  {'─' * 13} {'─' * 8} {'─' * 7} {'─' * 8} "
        f"{'─' * 10} {'─' * 8} {'─' * 8} {'─' * 5}"
    )
    for r in results:
        print(
            f"  {r.backend:<13} {r.concurrency:>8} {r.wall_time:>6.1f}s "
            f"{r.total_output_tokens:>8} {r.aggregate_tok_per_sec:>9.1f} "
            f"{r.median_latency:>7.2f}s {r.p95_latency:>7.2f}s "
            f"{r.success_rate:>4.0%}"
        )


def print_comparison(
    ollama_results: list[ConcurrencyResult],
    vllm_results: list[ConcurrencyResult],
) -> None:
    print()
    print(f"  {'Workers':>8} {'Ollama Tok/s':>13} {'vLLM Metal Tok/s':>16} {'Speedup':>8}")
    print(f"  {'─' * 8} {'─' * 13} {'─' * 16} {'─' * 8}")

    o_map = {r.concurrency: r for r in ollama_results}
    v_map = {r.concurrency: r for r in vllm_results}

    for c in sorted(set(o_map.keys()) | set(v_map.keys())):
        o_tps = o_map[c].aggregate_tok_per_sec if c in o_map else 0
        v_tps = v_map[c].aggregate_tok_per_sec if c in v_map else 0
        speedup = v_tps / o_tps if o_tps > 0 else float("inf")
        indicator = "  <-" if speedup > 1.5 else ""
        print(
            f"  {c:>8} {o_tps:>12.1f} {v_tps:>15.1f} {speedup:>7.2f}x{indicator}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM Metal vs Ollama on Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--vllm-model", default=DEFAULT_VLLM_MODEL,
                        help=f"HuggingFace model for vLLM (default: {DEFAULT_VLLM_MODEL})")
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL,
                        help=f"Ollama model tag (default: {DEFAULT_OLLAMA_MODEL})")
    parser.add_argument("--vllm-url", default=VLLM_URL,
                        help=f"vLLM Metal server URL (default: {VLLM_URL})")
    parser.add_argument("--ollama-url", default=OLLAMA_URL,
                        help=f"Ollama server URL (default: {OLLAMA_URL})")
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 8],
                        help="Concurrency levels to test (default: 1 2 4 8)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Runs per single-request test (default: 3)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens per response (default: 256)")
    parser.add_argument("--prompts", nargs="+", choices=["short", "medium", "long"],
                        default=["medium"],
                        help="Prompt sizes to benchmark (default: medium)")
    parser.add_argument("--ollama-only", action="store_true")
    parser.add_argument("--vllm-only", action="store_true")
    parser.add_argument("--skip-streaming", action="store_true",
                        help="Skip streaming/TTFT test")
    parser.add_argument("--skip-concurrency", action="store_true",
                        help="Skip concurrency test")
    args = parser.parse_args()

    test_ollama = not args.vllm_only
    test_vllm = not args.ollama_only

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║              vLLM Metal vs Ollama — Apple Silicon Benchmark         ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    # --- Check servers ---
    backends = []  # (name, check_ok)

    if test_ollama:
        ok = check_ollama(args.ollama_url)
        print(f"  {'✓' if ok else '✗'} Ollama at {args.ollama_url}  (model: {args.ollama_model})")
        if ok:
            backends.append("ollama")
        elif not test_vllm:
            sys.exit(1)

    if test_vllm:
        ok = check_vllm(args.vllm_url)
        print(f"  {'✓' if ok else '✗'} vLLM Metal at {args.vllm_url}  (model: {args.vllm_model})")
        if ok:
            backends.append("vllm-metal")
        elif not test_ollama:
            sys.exit(1)

    if not backends:
        print("\n  No backends available. Start at least one server.")
        sys.exit(1)

    # --- Warmup ---
    print()
    active = []
    if "ollama" in backends:
        if warmup("ollama", run_single_ollama, model=args.ollama_model, api_base=args.ollama_url):
            active.append("ollama")
    if "vllm-metal" in backends:
        if warmup("vllm-metal", run_single_vllm, model=args.vllm_model, api_base=args.vllm_url):
            active.append("vllm-metal")

    if not active:
        print("\n  All warmups failed.")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════════════
    # Test 1: Single-request latency
    # ═══════════════════════════════════════════════════════════════════════
    print_header(f"Test 1: Single-Request Latency (best of {args.runs})")

    for prompt_key in args.prompts:
        prompt = PROMPTS[prompt_key]
        results: list[SingleResult] = []

        for backend in active:
            best: SingleResult | None = None
            for _ in range(args.runs):
                if backend == "ollama":
                    r = run_single_ollama(
                        model=args.ollama_model, prompt=prompt,
                        api_base=args.ollama_url, max_tokens=args.max_tokens,
                    )
                else:
                    r = run_single_vllm(
                        model=args.vllm_model, prompt=prompt,
                        api_base=args.vllm_url, max_tokens=args.max_tokens,
                    )
                if best is None or (r.success and r.latency < getattr(best, "latency", float("inf"))):
                    best = r
            if best:
                results.append(best)

        print_single_results(results, label=f"prompt: {prompt_key}")

    # ═══════════════════════════════════════════════════════════════════════
    # Test 2: Streaming / TTFT
    # ═══════════════════════════════════════════════════════════════════════
    if not args.skip_streaming:
        print_header("Test 2: Time-to-First-Token (streaming)")

        prompt = PROMPTS[args.prompts[0]]
        stream_results: list[SingleResult] = []

        for backend in active:
            best: SingleResult | None = None
            for _ in range(args.runs):
                if backend == "ollama":
                    r = run_streaming_ollama(
                        model=args.ollama_model, prompt=prompt,
                        api_base=args.ollama_url, max_tokens=args.max_tokens,
                    )
                else:
                    r = run_streaming_vllm(
                        model=args.vllm_model, prompt=prompt,
                        api_base=args.vllm_url, max_tokens=args.max_tokens,
                    )
                if best is None or (r.success and r.ttft < getattr(best, "ttft", float("inf"))):
                    best = r
            if best:
                stream_results.append(best)

        print_single_results(stream_results, label=f"prompt: {args.prompts[0]}")

    # ═══════════════════════════════════════════════════════════════════════
    # Test 3: Concurrent throughput
    # ═══════════════════════════════════════════════════════════════════════
    if not args.skip_concurrency:
        print_header("Test 3: Concurrent Throughput")

        prompt = PROMPTS[args.prompts[0]]
        ollama_concurrent: list[ConcurrencyResult] = []
        vllm_concurrent: list[ConcurrencyResult] = []

        for concurrency in args.workers:
            for backend in active:
                print(f"  {backend} × {concurrency} workers...", end=" ", flush=True)
                if backend == "ollama":
                    cr = run_concurrent(
                        backend=backend,
                        run_fn=run_single_ollama,
                        concurrency=concurrency,
                        prompt=prompt,
                        model=args.ollama_model,
                        api_base=args.ollama_url,
                        max_tokens=args.max_tokens,
                    )
                    ollama_concurrent.append(cr)
                else:
                    cr = run_concurrent(
                        backend=backend,
                        run_fn=run_single_vllm,
                        concurrency=concurrency,
                        prompt=prompt,
                        model=args.vllm_model,
                        api_base=args.vllm_url,
                        max_tokens=args.max_tokens,
                    )
                    vllm_concurrent.append(cr)
                print(f"{cr.aggregate_tok_per_sec:.1f} tok/s")

        all_concurrent = ollama_concurrent + vllm_concurrent
        print_concurrency_table(all_concurrent)

        if ollama_concurrent and vllm_concurrent:
            print_header("Speedup: vLLM Metal vs Ollama")
            print_comparison(ollama_concurrent, vllm_concurrent)

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    print_header("Summary")

    if len(active) == 2 and not args.skip_concurrency:
        max_c = max(args.workers)
        o = next((r for r in ollama_concurrent if r.concurrency == max_c), None)
        v = next((r for r in vllm_concurrent if r.concurrency == max_c), None)
        if o and v and o.aggregate_tok_per_sec > 0:
            speedup = v.aggregate_tok_per_sec / o.aggregate_tok_per_sec
            print(f"  At {max_c} concurrent workers:")
            print(f"    Ollama:     {o.aggregate_tok_per_sec:>8.1f} tok/s  (model: {args.ollama_model})")
            print(f"    vLLM Metal: {v.aggregate_tok_per_sec:>8.1f} tok/s  (model: {args.vllm_model})")
            print(f"    Speedup:    {speedup:.2f}x")
            print()
            if speedup > 1.5:
                print("  → vLLM Metal shows significant advantage for parallel workloads")
            elif speedup > 1.0:
                print("  → vLLM Metal shows modest advantage")
            elif speedup > 0.8:
                print("  → Roughly equivalent performance")
            else:
                print("  → Ollama faster at this concurrency level")
    elif len(active) == 1:
        name = active[0]
        model = args.ollama_model if name == "ollama" else args.vllm_model
        print(f"  Tested {name} only (model: {model})")
        if not args.skip_concurrency:
            concurrent = ollama_concurrent or vllm_concurrent
            if concurrent:
                max_r = max(concurrent, key=lambda r: r.concurrency)
                print(f"  Peak throughput at {max_r.concurrency} workers: {max_r.aggregate_tok_per_sec:.1f} tok/s")

    print()
    print("  Note: Models differ in size/architecture. This benchmarks the")
    print("  *serving infrastructure*, not the models themselves.")
    print()


if __name__ == "__main__":
    main()
