#!/usr/bin/env python3
"""Benchmark llama-server vs Ollama for qwen3.5:4b on macOS.

Tests single-request latency, throughput (tokens/sec), and concurrent
request handling. Uses the same prompts for both backends.

Usage:
    python tools/benchmark_llama_vs_ollama.py
    python tools/benchmark_llama_vs_ollama.py --rounds 5 --concurrent 4
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import requests

# --- Config ---
OLLAMA_URL = "http://localhost:11434"
LLAMA_URL = "http://localhost:8090"
OLLAMA_MODEL = "qwen3.5:4b"  # Ollama model name
LLAMA_MODEL = "Qwen3.5-4B-Q4_K_M"  # llama-server model name (router mode)

# Code summarization prompts (realistic for Magaldi's use case)
PROMPTS = [
    "Summarize this Python function in one sentence:\n\ndef calculate_cosine_similarity(vec_a, vec_b):\n    dot = sum(a*b for a, b in zip(vec_a, vec_b))\n    norm_a = sum(a**2 for a in vec_a) ** 0.5\n    norm_b = sum(b**2 for b in vec_b) ** 0.5\n    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0",
    "Summarize this JavaScript function in one sentence:\n\nasync function fetchUserProfile(userId) {\n  const response = await fetch(`/api/users/${userId}`);\n  if (!response.ok) throw new Error(`HTTP ${response.status}`);\n  const data = await response.json();\n  return { ...data, fetchedAt: Date.now() };\n}",
    "Summarize this Rust function in one sentence:\n\npub fn parse_element_id(id: &str) -> Option<ElementParts> {\n    let parts: Vec<&str> = id.splitn(7, ':').collect();\n    if parts.len() < 7 { return None; }\n    Some(ElementParts {\n        scope: parts[0].to_string(),\n        repository: parts[1].to_string(),\n        username: parts[2].to_string(),\n        path: parts[3].to_string(),\n        element_type: parts[4].to_string(),\n        name: parts[5].to_string(),\n        line: parts[6].parse().ok()?,\n    })\n}",
    "Summarize this Python class in one sentence:\n\nclass RetryableHTTPClient:\n    def __init__(self, base_url, max_retries=3, backoff=1.5):\n        self.base_url = base_url\n        self.max_retries = max_retries\n        self.backoff = backoff\n        self.session = requests.Session()\n\n    def get(self, path, **kwargs):\n        for attempt in range(self.max_retries):\n            try:\n                resp = self.session.get(f'{self.base_url}{path}', **kwargs)\n                resp.raise_for_status()\n                return resp\n            except requests.RequestException:\n                if attempt == self.max_retries - 1:\n                    raise\n                time.sleep(self.backoff ** attempt)",
]

SYSTEM_PROMPT = (
    "You are a code documentation assistant. Respond with a single concise sentence. "
    "No thinking, no explanation, just the summary."
)


@dataclass
class RequestResult:
    prompt_idx: int
    tokens_generated: int
    time_to_first_token: float  # seconds
    total_time: float  # seconds
    tokens_per_sec: float
    prompt_tokens: int = 0
    error: str | None = None


@dataclass
class BenchmarkResult:
    backend: str
    results: list[RequestResult] = field(default_factory=list)

    @property
    def avg_ttft(self) -> float:
        vals = [r.time_to_first_token for r in self.results if not r.error]
        return statistics.mean(vals) if vals else 0

    @property
    def avg_tps(self) -> float:
        vals = [r.tokens_per_sec for r in self.results if not r.error]
        return statistics.mean(vals) if vals else 0

    @property
    def median_tps(self) -> float:
        vals = [r.tokens_per_sec for r in self.results if not r.error]
        return statistics.median(vals) if vals else 0

    @property
    def avg_total(self) -> float:
        vals = [r.total_time for r in self.results if not r.error]
        return statistics.mean(vals) if vals else 0

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens_generated for r in self.results if not r.error)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.error)


def query_ollama(prompt: str, prompt_idx: int) -> RequestResult:
    """Send a single request to Ollama and measure performance."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "options": {"num_predict": 128, "temperature": 0},
    }

    try:
        t0 = time.monotonic()
        ttft = 0.0
        tokens = 0
        prompt_tokens = 0

        with requests.post(
            f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=60
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                msg = data.get("message", {})
                # Count both thinking and content tokens (Qwen3 uses thinking)
                has_token = msg.get("content") or msg.get("thinking")
                if has_token:
                    if tokens == 0:
                        ttft = time.monotonic() - t0
                    tokens += 1
                if data.get("prompt_eval_count"):
                    prompt_tokens = data["prompt_eval_count"]
                if data.get("eval_count"):
                    # Use Ollama's own token count if available (more accurate)
                    tokens = data["eval_count"]
                if data.get("done"):
                    break

        total = time.monotonic() - t0
        tps = tokens / total if total > 0 else 0

        return RequestResult(
            prompt_idx=prompt_idx,
            tokens_generated=tokens,
            time_to_first_token=ttft,
            total_time=total,
            tokens_per_sec=tps,
            prompt_tokens=prompt_tokens,
        )
    except Exception as e:
        return RequestResult(
            prompt_idx=prompt_idx,
            tokens_generated=0,
            time_to_first_token=0,
            total_time=0,
            tokens_per_sec=0,
            error=str(e),
        )


def query_llama_server(prompt: str, prompt_idx: int) -> RequestResult:
    """Send a single request to llama-server and measure performance."""
    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "max_tokens": 128,
        "temperature": 0,
    }

    try:
        t0 = time.monotonic()
        ttft = 0.0
        tokens = 0

        with requests.post(
            f"{LLAMA_URL}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=60,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8")
                if not text.startswith("data: "):
                    continue
                text = text[6:]
                if text.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    continue
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    # Count both content and reasoning_content tokens (Qwen3 thinking)
                    has_token = delta.get("content") or delta.get("reasoning_content")
                    if has_token:
                        if tokens == 0:
                            ttft = time.monotonic() - t0
                        tokens += 1

        total = time.monotonic() - t0
        tps = tokens / total if total > 0 else 0

        return RequestResult(
            prompt_idx=prompt_idx,
            tokens_generated=tokens,
            time_to_first_token=ttft,
            total_time=total,
            tokens_per_sec=tps,
        )
    except Exception as e:
        return RequestResult(
            prompt_idx=prompt_idx,
            tokens_generated=0,
            time_to_first_token=0,
            total_time=0,
            tokens_per_sec=0,
            error=str(e),
        )


def warmup(backend: str) -> None:
    """Send one throwaway request to warm up the model."""
    print(f"  Warming up {backend}...")
    prompt = "Say hello in one word."
    if backend == "ollama":
        query_ollama(prompt, 0)
    else:
        query_llama_server(prompt, 0)


def run_sequential(backend: str, rounds: int) -> BenchmarkResult:
    """Run sequential single-request benchmark."""
    result = BenchmarkResult(backend=backend)
    fn = query_ollama if backend == "ollama" else query_llama_server

    for _round in range(rounds):
        for i, prompt in enumerate(PROMPTS):
            r = fn(prompt, i)
            result.results.append(r)
            if r.error:
                print(f"    [ERROR] prompt {i}: {r.error}")

    return result


def run_concurrent(backend: str, concurrency: int, rounds: int) -> BenchmarkResult:
    """Run concurrent requests benchmark."""
    result = BenchmarkResult(backend=backend)
    fn = query_ollama if backend == "ollama" else query_llama_server

    for _round in range(rounds):
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = []
            for i, prompt in enumerate(PROMPTS):
                futures.append(pool.submit(fn, prompt, i))

            for future in as_completed(futures):
                r = future.result()
                result.results.append(r)
                if r.error:
                    print(f"    [ERROR] prompt {r.prompt_idx}: {r.error}")

    return result


def print_comparison(
    label: str,
    ollama_result: BenchmarkResult,
    llama_result: BenchmarkResult,
) -> None:
    """Print side-by-side comparison."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"{'Metric':<30} {'Ollama':>12} {'llama-server':>12} {'Δ':>10}")
    print(f"{'-' * 64}")

    def row(name: str, o: float, l: float, unit: str = "", lower_better: bool = False) -> None:
        if o == 0 and l == 0:
            print(f"{name:<30} {'N/A':>12} {'N/A':>12}")
            return
        diff = ((l - o) / o * 100) if o else 0
        sign = "+" if diff > 0 else ""
        # For lower-is-better metrics, positive diff is worse
        color_indicator = ""
        if lower_better:
            color_indicator = "⬆" if diff > 0 else "⬇" if diff < 0 else ""
        else:
            color_indicator = "⬆" if diff > 0 else "⬇" if diff < 0 else ""
        print(
            f"{name:<30} {o:>10.2f}{unit:>2} {l:>10.2f}{unit:>2} {sign}{diff:>6.1f}% {color_indicator}"
        )

    row("Avg TTFT", ollama_result.avg_ttft, llama_result.avg_ttft, "s", lower_better=True)
    row("Avg tokens/sec", ollama_result.avg_tps, llama_result.avg_tps, "")
    row("Median tokens/sec", ollama_result.median_tps, llama_result.median_tps, "")
    row("Avg total time", ollama_result.avg_total, llama_result.avg_total, "s", lower_better=True)
    print(
        f"{'Total tokens':<30} {ollama_result.total_tokens:>12} {llama_result.total_tokens:>12}"
    )
    print(
        f"{'Errors':<30} {ollama_result.error_count:>12} {llama_result.error_count:>12}"
    )


def check_backends() -> tuple[bool, bool]:
    """Check which backends are available."""
    ollama_ok = False
    llama_ok = False

    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
    except Exception:
        pass

    try:
        r = requests.get(f"{LLAMA_URL}/health", timeout=3)
        llama_ok = r.status_code == 200
    except Exception:
        pass

    return ollama_ok, llama_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark llama-server vs Ollama")
    parser.add_argument("--rounds", type=int, default=3, help="Rounds per test (default: 3)")
    parser.add_argument("--concurrent", type=int, default=4, help="Concurrent requests (default: 4)")
    args = parser.parse_args()

    print("🔍 Checking backends...")
    ollama_ok, llama_ok = check_backends()
    print(f"  Ollama ({OLLAMA_URL}): {'✅' if ollama_ok else '❌'}")
    print(f"  llama-server ({LLAMA_URL}): {'✅' if llama_ok else '❌'}")

    if not ollama_ok and not llama_ok:
        print("\n❌ Neither backend is available. Start at least one.")
        sys.exit(1)

    print(f"\n📋 Config: {len(PROMPTS)} prompts × {args.rounds} rounds, concurrency={args.concurrent}")
    print(f"   Ollama model: {OLLAMA_MODEL}")
    print(f"   llama-server model: {LLAMA_MODEL}")

    # --- Sequential benchmark ---
    print("\n" + "=" * 60)
    print("  TEST 1: Sequential requests (single-stream)")
    print("=" * 60)

    ollama_seq = llama_seq = None

    if ollama_ok:
        warmup("ollama")
        print("  Running Ollama sequential...")
        ollama_seq = run_sequential("ollama", args.rounds)
        print(f"    → {ollama_seq.avg_tps:.1f} tok/s avg, {ollama_seq.avg_ttft:.3f}s TTFT")

    if llama_ok:
        warmup("llama-server")
        print("  Running llama-server sequential...")
        llama_seq = run_sequential("llama-server", args.rounds)
        print(f"    → {llama_seq.avg_tps:.1f} tok/s avg, {llama_seq.avg_ttft:.3f}s TTFT")

    if ollama_seq and llama_seq:
        print_comparison("Sequential Results", ollama_seq, llama_seq)

    # --- Concurrent benchmark ---
    print(f"\n{'=' * 60}")
    print(f"  TEST 2: Concurrent requests ({args.concurrent} parallel)")
    print(f"{'=' * 60}")

    ollama_conc = llama_conc = None

    if ollama_ok:
        warmup("ollama")
        print(f"  Running Ollama concurrent ({args.concurrent})...")
        ollama_conc = run_concurrent("ollama", args.concurrent, args.rounds)
        print(f"    → {ollama_conc.avg_tps:.1f} tok/s avg, {ollama_conc.avg_ttft:.3f}s TTFT")

    if llama_ok:
        warmup("llama-server")
        print(f"  Running llama-server concurrent ({args.concurrent})...")
        llama_conc = run_concurrent("llama-server", args.concurrent, args.rounds)
        print(f"    → {llama_conc.avg_tps:.1f} tok/s avg, {llama_conc.avg_ttft:.3f}s TTFT")

    if ollama_conc and llama_conc:
        print_comparison(f"Concurrent Results ({args.concurrent} parallel)", ollama_conc, llama_conc)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    if ollama_seq and llama_seq and ollama_seq.avg_tps > 0 and llama_seq.avg_tps > 0:
        seq_winner = "llama-server" if llama_seq.avg_tps > ollama_seq.avg_tps else "Ollama"
        seq_pct = abs(llama_seq.avg_tps - ollama_seq.avg_tps) / min(ollama_seq.avg_tps, llama_seq.avg_tps) * 100
        print(f"  Sequential:  {seq_winner} wins by {seq_pct:.1f}%")
    if ollama_conc and llama_conc and ollama_conc.avg_tps > 0 and llama_conc.avg_tps > 0:
        conc_winner = "llama-server" if llama_conc.avg_tps > ollama_conc.avg_tps else "Ollama"
        conc_pct = abs(llama_conc.avg_tps - ollama_conc.avg_tps) / min(ollama_conc.avg_tps, llama_conc.avg_tps) * 100
        print(f"  Concurrent:  {conc_winner} wins by {conc_pct:.1f}%")


if __name__ == "__main__":
    main()
