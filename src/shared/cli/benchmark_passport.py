#!/usr/bin/env python3
"""Benchmark semantic passport embedding formats for call resolution.

Compares passport text formats for code element embeddings by measuring
how well each format's cosine similarity predicts actual caller→callee
relationships (ground truth from statically resolved calls).

Every variant starts with the current production baseline (from
build_summary_embedding_text) and adds structural metadata on top.

Usage:
    magaldi-tools passport-benchmark
    magaldi-tools passport-benchmark --scope magaldi --repository magaldi
    magaldi-tools passport-benchmark --n-callers 100 --n-negatives 500
    magaldi-tools passport-benchmark --variants plus_calls,plus_full
"""

from __future__ import annotations

import argparse
import gc
import logging
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Call categories that count as ground truth (NOT embedding_resolved)
GROUND_TRUTH_CATEGORIES = {"resolved", "import_resolved", "type_resolvable", "unknown"}

ALL_VARIANT_NAMES = [
    "baseline",
    "plus_calls",
    "plus_params",
    "plus_calls_params",
    "plus_full",
    "plus_imports",
    "plus_siblings",
]

# Index name
INDEX_NAME = "magaldi-code-elements"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GroundTruthPair:
    """A known caller→callee relationship."""

    caller_id: str
    callee_id: str
    call_name: str


@dataclass
class VariantResult:
    """Benchmark results for a single passport variant."""

    name: str
    mrr: float = 0.0
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    mean_pos_sim: float = 0.0
    mean_neg_sim: float = 0.0
    separation: float = 0.0
    embed_time_s: float = 0.0
    worst_pairs: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Auto-detect scope/repository from magaldi.yaml
# ---------------------------------------------------------------------------


def _auto_detect_repo() -> tuple[str, str]:
    """Read scope/repository from magaldi.yaml in cwd."""
    import yaml

    yaml_path = Path.cwd() / "magaldi.yaml"
    if not yaml_path.exists():
        return "", ""
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    scope = cfg.get("scope", "")
    repository = cfg.get("repository") or cfg.get("name") or ""
    return scope, repository


# ---------------------------------------------------------------------------
# Phase 1: Connect to search backend + validate embedding model
# ---------------------------------------------------------------------------


def _connect() -> Any:
    """Create repository and verify connectivity."""
    from shared.config import load_config
    from shared.db.repositories import Repository

    config = load_config(skip_validation=True)
    repo = Repository(config)
    # Quick connectivity check
    client = repo._get_client()
    if not client.ping():
        console.print("[red]Error:[/] Cannot connect to search backend")
        sys.exit(1)
    return repo


def _verify_embedding_model() -> Any:
    """Create embedding client and verify model is available."""
    from shared.ai.embedding import CodeEmbeddingClient

    # Disable litellm's internal caching to reduce memory usage.
    # Without this, litellm caches all response objects (including
    # full embedding vectors), causing memory to balloon to 30+ GB
    # across ~2000 embedding calls.
    try:
        import litellm

        litellm.cache = None
        litellm.suppress_debug_info = True
    except ImportError:
        pass

    client = CodeEmbeddingClient(
        url="http://localhost:11434",
        model="qwen3-embedding:0.6b",
        provider="ollama",
        dimensions=1024,
    )
    if not client.verify_model():
        console.print(
            "[red]Error:[/] Embedding model qwen3-embedding:0.6b not available. "
            "Run: ollama pull qwen3-embedding:0.6b"
        )
        sys.exit(1)
    return client


# ---------------------------------------------------------------------------
# Phase 2: Sample ground truth pairs
# ---------------------------------------------------------------------------


def _sample_ground_truth(
    repo: Any,
    scope: str,
    repository: str,
    n_callers: int,
    seed: int,
) -> tuple[list[GroundTruthPair], set[str], set[str]]:
    """Sample callers with statically resolved calls and build ground truth pairs.

    Returns:
        (pairs, caller_ids, callee_ids)
    """
    client = repo._get_client()

    # Find elements with resolved calls (not embedding_resolved)
    query = {
        "bool": {
            "must": [
                {"term": {"scope": scope}},
                {"term": {"repository": repository}},
                {"term": {"username": "main"}},
                {
                    "nested": {
                        "path": "calls",
                        "query": {"exists": {"field": "calls.resolved_id"}},
                    }
                },
            ],
        }
    }

    result = client.search(
        index=INDEX_NAME,
        body={
            "query": query,
            "size": 10000,
            "_source": ["element_id", "calls"],
        },
    )

    hits = result.get("hits", {}).get("hits", [])
    if not hits:
        console.print("[red]Error:[/] No elements with resolved calls found")
        sys.exit(1)

    # Collect valid callers (those with non-embedding-resolved calls)
    valid_callers: list[dict] = []
    for hit in hits:
        doc = hit["_source"]
        calls = doc.get("calls", [])
        static_calls = [
            c
            for c in calls
            if c.get("resolved_id") and c.get("category") != "embedding_resolved"
        ]
        if static_calls:
            valid_callers.append({"element_id": doc["element_id"], "calls": static_calls})

    console.print(f"  Found {len(valid_callers)} elements with statically resolved calls")

    # Sample callers
    rng = random.Random(seed)
    if len(valid_callers) > n_callers:
        sampled = rng.sample(valid_callers, n_callers)
    else:
        sampled = valid_callers
        console.print(
            f"  [yellow]Warning:[/] Only {len(sampled)} callers available "
            f"(requested {n_callers})"
        )

    # Build ground truth pairs, verify callees exist
    pairs: list[GroundTruthPair] = []
    caller_ids: set[str] = set()
    callee_ids: set[str] = set()

    # Collect all potential callee IDs for batch verification
    all_callee_ids: set[str] = set()
    for caller in sampled:
        for call in caller["calls"]:
            all_callee_ids.add(call["resolved_id"])

    # Batch verify existence
    existing_callees = _batch_check_existence(client, list(all_callee_ids))

    for caller in sampled:
        for call in caller["calls"]:
            callee_id = call["resolved_id"]
            if callee_id in existing_callees:
                pairs.append(
                    GroundTruthPair(
                        caller_id=caller["element_id"],
                        callee_id=callee_id,
                        call_name=call.get("name", "?"),
                    )
                )
                caller_ids.add(caller["element_id"])
                callee_ids.add(callee_id)

    console.print(
        f"  Built {len(pairs)} ground truth pairs "
        f"({len(caller_ids)} callers → {len(callee_ids)} callees)"
    )
    return pairs, caller_ids, callee_ids


def _batch_check_existence(client: Any, element_ids: list[str]) -> set[str]:
    """Check which element IDs exist via mget."""
    if not element_ids:
        return set()

    response = client.mget(
        index=INDEX_NAME,
        ids=element_ids,
        _source=False,
    )

    return {
        doc["_id"]
        for doc in response.get("docs", [])
        if doc.get("found")
    }


# ---------------------------------------------------------------------------
# Phase 3: Build negative distractor pool
# ---------------------------------------------------------------------------


def _sample_negatives(
    repo: Any,
    scope: str,
    repository: str,
    n_negatives: int,
    exclude_ids: set[str],
    seed: int,
) -> set[str]:
    """Sample random function/method elements as negative distractors."""
    client = repo._get_client()

    query = {
        "bool": {
            "must": [
                {"term": {"scope": scope}},
                {"term": {"repository": repository}},
                {"term": {"username": "main"}},
                {"terms": {"element_type": ["function", "method"]}},
            ],
            "must_not": [{"ids": {"values": list(exclude_ids)}}],
        }
    }

    result = client.search(
        index=INDEX_NAME,
        body={
            "query": {"function_score": {"query": query, "random_score": {"seed": seed, "field": "_seq_no"}}},
            "size": n_negatives,
            "_source": ["element_id"],
        },
    )

    neg_ids = {
        hit["_source"]["element_id"]
        for hit in result.get("hits", {}).get("hits", [])
    }

    console.print(f"  Sampled {len(neg_ids)} negative distractors")
    return neg_ids


# ---------------------------------------------------------------------------
# Phase 4: Fetch full documents via mget
# ---------------------------------------------------------------------------


def _fetch_documents(client: Any, element_ids: list[str]) -> dict[str, dict]:
    """Fetch full documents for all element IDs via mget.

    Returns dict mapping element_id to document.
    """
    if not element_ids:
        return {}

    # mget in chunks of 500 to avoid huge payloads
    docs: dict[str, dict] = {}
    chunk_size = 500
    for i in range(0, len(element_ids), chunk_size):
        chunk = element_ids[i : i + chunk_size]
        response = client.mget(
            index=INDEX_NAME,
            ids=chunk,
            _source=[
                "element_id",
                "name",
                "element_type",
                "relative_path",
                "signature",
                "docstring",
                "summary",
                "calls",
                "parameters",
                "return_type",
                "decorators",
                "detected_patterns",
                "parent_id",
                "level",
                "language",
            ],
        )
        for doc in response.get("docs", []):
            if doc.get("found"):
                source = doc["_source"]
                docs[source["element_id"]] = source

    return docs


# ---------------------------------------------------------------------------
# Phase 5: Build passport texts for all variants
# ---------------------------------------------------------------------------


def _build_breadcrumbs(doc: dict) -> str:
    """Build file path + parent class breadcrumb."""
    parts = [doc.get("relative_path", "")]
    parent_id = doc.get("parent_id", "")
    if parent_id:
        # Parse class name from parent_id: scope:repo:user:path:class:ClassName:line
        id_parts = parent_id.split(":")
        if len(id_parts) >= 6 and id_parts[4] == "class":
            parts.append(id_parts[5])
    return " > ".join(p for p in parts if p)


def _format_params(doc: dict) -> str:
    """Format parameters as 'name: type' list."""
    params = doc.get("parameters", [])
    if not params:
        return ""
    formatted = []
    for p in params:
        name = p.get("name", "?")
        ptype = p.get("type")
        if ptype:
            formatted.append(f"{name}: {ptype}")
        else:
            formatted.append(name)
    return ", ".join(formatted)


def _format_calls(doc: dict) -> str:
    """Extract outbound call names (deduplicated)."""
    calls = doc.get("calls", [])
    if not calls:
        return ""
    seen: set[str] = set()
    names: list[str] = []
    for c in calls:
        receiver = c.get("receiver")
        name = c.get("name", "")
        key = f"{receiver}.{name}" if receiver else name
        if key not in seen:
            seen.add(key)
            names.append(key)
    return ", ".join(names)


def build_passport_text(
    variant: str,
    doc: dict,
    file_summaries_cache: dict[str, str],
    class_summaries_cache: dict[str, str],
    file_imports_cache: dict[str, list[dict]],
    sibling_names_cache: dict[str, list[str]],
) -> str:
    """Build passport text for a given variant and document.

    Every variant starts with the baseline (replicating build_summary_embedding_text
    for function/method) then adds structural metadata on top.
    """
    name = doc.get("name", "?")
    sig = doc.get("signature", "")
    summary = doc.get("summary", "")
    docstring = doc.get("docstring", "")
    rel_path = doc.get("relative_path", "")
    parent_id = doc.get("parent_id", "")
    element_id = doc.get("element_id", "")

    # === Baseline: replicates build_summary_embedding_text for function/method ===
    parts: list[str] = [f"File: {rel_path}"]

    file_summary = file_summaries_cache.get(rel_path, "")
    if file_summary:
        parts.append(f"File context: {file_summary}")

    # Class context for methods (parent is a class)
    if parent_id:
        id_parts = parent_id.split(":")
        if len(id_parts) >= 6 and id_parts[4] == "class":
            class_summary = class_summaries_cache.get(parent_id, "")
            if class_summary:
                parts.append(f"Class context: {class_summary}")

    parts.append(f"Function: {name}")

    if summary:
        parts.append(f"Summary: {summary}")
    if sig:
        parts.append(f"Signature: {sig}")
    if docstring:
        ds = docstring[:500]
        if len(docstring) > 500:
            ds += "..."
        parts.append(f"Docstring: {ds}")

    if variant == "baseline":
        return "\n".join(parts)

    # === Additions on top of baseline ===
    params_str = _format_params(doc)
    calls_str = _format_calls(doc)
    return_type = doc.get("return_type", "")
    decorators = doc.get("decorators", [])
    patterns = doc.get("detected_patterns", [])
    breadcrumbs = _build_breadcrumbs(doc)

    if variant == "plus_calls":
        if calls_str:
            parts.append(f"Calls: {calls_str}")
        return "\n".join(parts)

    if variant == "plus_params":
        if params_str:
            parts.append(f"Params: {params_str}")
        if return_type:
            parts.append(f"Returns: {return_type}")
        return "\n".join(parts)

    if variant == "plus_calls_params":
        if calls_str:
            parts.append(f"Calls: {calls_str}")
        if params_str:
            parts.append(f"Params: {params_str}")
        if return_type:
            parts.append(f"Returns: {return_type}")
        return "\n".join(parts)

    # plus_full base (also used by plus_imports and plus_siblings)
    if calls_str:
        parts.append(f"Calls: {calls_str}")
    if params_str:
        parts.append(f"Params: {params_str}")
    if return_type:
        parts.append(f"Returns: {return_type}")
    if breadcrumbs:
        parts.append(f"Location: {breadcrumbs}")
    if decorators:
        parts.append(f"Decorators: {', '.join(decorators)}")
    if patterns:
        parts.append(f"Patterns: {', '.join(patterns)}")

    if variant == "plus_full":
        return "\n".join(parts)

    if variant == "plus_imports":
        imports = file_imports_cache.get(rel_path, [])
        if imports:
            modules = list({imp.get("module", "") for imp in imports if imp.get("module")})
            if modules:
                parts.append(f"Imports: {', '.join(sorted(modules))}")
        return "\n".join(parts)

    if variant == "plus_siblings":
        siblings = sibling_names_cache.get(element_id, [])
        if siblings:
            parts.append(f"Siblings: {', '.join(siblings)}")
        return "\n".join(parts)

    return "\n".join(parts)  # fallback


# ---------------------------------------------------------------------------
# Context caches: parent summaries, file imports & sibling names
# ---------------------------------------------------------------------------


def _build_parent_summaries_cache(
    client: Any,
    docs: dict[str, dict],
    scope: str,
    repository: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build caches for file and class summaries (needed for baseline).

    Returns:
        (file_summaries, class_summaries) where:
        - file_summaries maps relative_path -> file summary text
        - class_summaries maps class_element_id -> class summary text
    """
    # --- File summaries ---
    file_summaries: dict[str, str] = {}
    paths = {doc.get("relative_path", "") for doc in docs.values() if doc.get("relative_path")}

    file_ids: list[str] = []
    path_by_id: dict[str, str] = {}
    for path in paths:
        file_name = Path(path).name
        file_id = f"{scope}:{repository}:main:{path}:file:{file_name}:1"
        file_ids.append(file_id)
        path_by_id[file_id] = path

    if file_ids:
        for i in range(0, len(file_ids), 500):
            chunk = file_ids[i : i + 500]
            response = client.mget(
                index=INDEX_NAME,
                ids=chunk,
                _source=["summary"],
            )
            for doc in response.get("docs", []):
                if doc.get("found"):
                    eid = doc["_id"]
                    summary = doc["_source"].get("summary", "")
                    if eid in path_by_id and summary:
                        file_summaries[path_by_id[eid]] = summary

    # --- Class summaries ---
    class_summaries: dict[str, str] = {}
    class_ids: set[str] = set()
    for doc in docs.values():
        parent_id = doc.get("parent_id", "")
        if parent_id:
            id_parts = parent_id.split(":")
            if len(id_parts) >= 6 and id_parts[4] == "class":
                class_ids.add(parent_id)

    if class_ids:
        class_id_list = list(class_ids)
        for i in range(0, len(class_id_list), 500):
            chunk = class_id_list[i : i + 500]
            response = client.mget(
                index=INDEX_NAME,
                ids=chunk,
                _source=["summary"],
            )
            for doc in response.get("docs", []):
                if doc.get("found"):
                    eid = doc["_id"]
                    summary = doc["_source"].get("summary", "")
                    if summary:
                        class_summaries[eid] = summary

    return file_summaries, class_summaries


def _build_file_imports_cache(
    client: Any,
    docs: dict[str, dict],
    scope: str,
    repository: str,
) -> dict[str, list[dict]]:
    """Build cache of file imports keyed by relative_path."""
    # Collect unique relative paths
    paths = {doc.get("relative_path", "") for doc in docs.values() if doc.get("relative_path")}

    cache: dict[str, list[dict]] = {}
    for path in paths:
        # Build file element ID
        file_name = Path(path).name
        file_id = f"{scope}:{repository}:main:{path}:file:{file_name}:1"
        response = client.mget(
            index=INDEX_NAME,
            ids=[file_id],
            _source=["imports"],
        )
        for doc in response.get("docs", []):
            if doc.get("found"):
                cache[path] = doc["_source"].get("imports", []) or []
                break
        if path not in cache:
            cache[path] = []

    return cache


def _build_sibling_names_cache(
    client: Any,
    docs: dict[str, dict],
) -> dict[str, list[str]]:
    """Build cache of sibling function names for each element.

    Groups elements by parent_id and lists sibling names (excluding self).
    """
    # Group by parent_id
    by_parent: dict[str, list[dict]] = {}
    for doc in docs.values():
        parent_id = doc.get("parent_id", "")
        if parent_id:
            by_parent.setdefault(parent_id, []).append(doc)

    # For elements whose parent_id is NOT in our doc pool, query ES (capped)
    missing_parents: set[str] = set()
    known_element_ids = set(docs.keys())
    for doc in docs.values():
        parent_id = doc.get("parent_id", "")
        if parent_id and parent_id not in known_element_ids:
            missing_parents.add(parent_id)

    # Query children for missing parents (cap at 50 queries to limit cost)
    extra_siblings: dict[str, list[str]] = {}  # parent_id -> child names
    for parent_id in list(missing_parents)[:50]:
        result = client.search(
            index=INDEX_NAME,
            body={
                "query": {"term": {"parent_id": parent_id}},
                "size": 100,
                "_source": ["element_id", "name", "element_type"],
            },
        )
        child_names: list[str] = []
        for hit in result.get("hits", {}).get("hits", []):
            src = hit["_source"]
            if src.get("element_type") in ("function", "method"):
                child_names.append(src.get("name", ""))
        extra_siblings[parent_id] = child_names

    # Build the cache: element_id -> list of sibling names
    cache: dict[str, list[str]] = {}
    for doc in docs.values():
        element_id = doc.get("element_id", "")
        parent_id = doc.get("parent_id", "")
        name = doc.get("name", "")

        siblings: list[str] = []

        # From co-pooled elements
        if parent_id in by_parent:
            siblings = [
                d.get("name", "")
                for d in by_parent[parent_id]
                if d.get("element_id") != element_id
                and d.get("element_type") in ("function", "method")
            ]

        # From extra queries
        if parent_id in extra_siblings:
            for sn in extra_siblings[parent_id]:
                if sn != name and sn not in siblings:
                    siblings.append(sn)

        cache[element_id] = siblings

    return cache


# ---------------------------------------------------------------------------
# Phase 6: Embed + evaluate per variant
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two L2-normalized vectors (= dot product)."""
    return sum(x * y for x, y in zip(a, b))


def _embed_all(
    embed_client: Any,
    texts: list[str],
    batch_size: int,
) -> list[list[float]]:
    """Embed all texts in batches, returning vectors in order."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_vecs = embed_client.embed_batch(batch, timeout=120)
        vectors.extend(batch_vecs)
    return vectors


def _evaluate_variant(
    variant_name: str,
    pairs: list[GroundTruthPair],
    embeddings: dict[str, list[float]],
    neg_ids: list[str],
) -> VariantResult:
    """Evaluate a variant's embeddings against ground truth pairs."""
    reciprocal_ranks: list[float] = []
    hits_1: list[int] = []
    hits_3: list[int] = []
    hits_5: list[int] = []
    pos_sims: list[float] = []
    neg_sims_all: list[float] = []
    worst_pairs: list[dict] = []

    for pair in pairs:
        caller_emb = embeddings.get(pair.caller_id)
        callee_emb = embeddings.get(pair.callee_id)
        if caller_emb is None or callee_emb is None:
            continue

        pos_sim = _cosine_similarity(caller_emb, callee_emb)
        pos_sims.append(pos_sim)

        # Compare against negatives
        neg_scores: list[float] = []
        for neg_id in neg_ids:
            neg_emb = embeddings.get(neg_id)
            if neg_emb is not None:
                neg_scores.append(_cosine_similarity(caller_emb, neg_emb))

        neg_sims_all.extend(neg_scores)

        # Rank: how many negatives score >= true callee?
        rank = 1 + sum(1 for ns in neg_scores if ns >= pos_sim)
        reciprocal_ranks.append(1.0 / rank)
        hits_1.append(1 if rank <= 1 else 0)
        hits_3.append(1 if rank <= 3 else 0)
        hits_5.append(1 if rank <= 5 else 0)

        worst_pairs.append(
            {
                "caller_id": pair.caller_id,
                "callee_id": pair.callee_id,
                "call_name": pair.call_name,
                "pos_sim": round(pos_sim, 4),
                "rank": rank,
            }
        )

    if not reciprocal_ranks:
        return VariantResult(name=variant_name)

    # Sort worst_pairs by rank descending (worst first)
    worst_pairs.sort(key=lambda x: x["rank"], reverse=True)

    return VariantResult(
        name=variant_name,
        mrr=statistics.mean(reciprocal_ranks),
        hit_at_1=statistics.mean(hits_1),
        hit_at_3=statistics.mean(hits_3),
        hit_at_5=statistics.mean(hits_5),
        mean_pos_sim=statistics.mean(pos_sims),
        mean_neg_sim=statistics.mean(neg_sims_all) if neg_sims_all else 0.0,
        separation=(
            statistics.mean(pos_sims) - statistics.mean(neg_sims_all)
            if neg_sims_all
            else 0.0
        ),
        worst_pairs=worst_pairs[:10],
    )


# ---------------------------------------------------------------------------
# Phase 7: Report results
# ---------------------------------------------------------------------------


def _print_results(results: list[VariantResult], n_pairs: int, n_negatives: int) -> None:
    """Print Rich table of results."""
    # Sort by MRR descending
    results.sort(key=lambda r: r.mrr, reverse=True)
    best_name = results[0].name if results else ""

    table = Table(title="Semantic Passport Benchmark Results")
    table.add_column("Variant", style="bold", min_width=24, no_wrap=True)
    table.add_column("MRR", justify="right")
    table.add_column("Hit@1", justify="right")
    table.add_column("Hit@3", justify="right")
    table.add_column("Hit@5", justify="right")
    table.add_column("Pos Sim", justify="right")
    table.add_column("Neg Sim", justify="right")
    table.add_column("Sep", justify="right")
    table.add_column("Time", justify="right")

    for r in results:
        marker = " **" if r.name == best_name else ""
        table.add_row(
            f"{r.name}{marker}",
            f"{r.mrr:.4f}",
            f"{r.hit_at_1:.3f}",
            f"{r.hit_at_3:.3f}",
            f"{r.hit_at_5:.3f}",
            f"{r.mean_pos_sim:.4f}",
            f"{r.mean_neg_sim:.4f}",
            f"{r.separation:.4f}",
            f"{r.embed_time_s:.1f}s",
        )

    console.print()
    console.print(table)
    console.print(f"\n  Ground truth pairs: {n_pairs} | Negatives: {n_negatives}")


def _save_markdown(
    results: list[VariantResult],
    n_pairs: int,
    n_negatives: int,
    n_callers: int,
    seed: int,
    output_dir: str,
) -> str:
    """Save results to markdown file. Returns file path."""
    results.sort(key=lambda r: r.mrr, reverse=True)
    best_name = results[0].name if results else ""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / f"{timestamp}_passport_benchmark.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Semantic Passport Embedding Benchmark",
        "",
        "## Context",
        "",
        "This benchmark tests how well different text representations ('passports')",
        "of code elements predict caller-callee relationships via embedding similarity.",
        "Ground truth is from statically resolved calls (NOT embedding-resolved).",
        "The goal: find the passport format that best distinguishes true callees",
        "from random distractors when comparing embedding vectors.",
        "",
        "## Setup",
        "",
        f"- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Callers sampled:** {n_callers}",
        f"- **Ground truth pairs:** {n_pairs}",
        f"- **Negative distractors:** {n_negatives}",
        f"- **Seed:** {seed}",
        "- **Embedding model:** qwen3-embedding:0.6b (1024 dims, L2-normalized)",
        f"- **Best variant:** `{best_name}`",
        "",
        "## How to Read the Results",
        "",
        "| Metric | Meaning | Good Value |",
        "|--------|---------|------------|",
        "| **MRR** | Mean Reciprocal Rank: avg of 1/rank of true callee among negatives."
        " Primary metric. | > 0.5 |",
        "| **Hit@K** | Fraction of pairs where true callee ranks in top K. | > 0.5 |",
        "| **Pos Sim** | Mean cosine similarity between caller and true callee. | High |",
        "| **Neg Sim** | Mean cosine similarity between caller and random distractors. | Low |",
        "| **Separation** | Pos Sim - Neg Sim. Higher = better discrimination."
        " Most actionable metric. | > 0.15 |",
        "",
        "**Key questions when analyzing:**",
        "1. Which variant has the highest MRR? That's the best passport format.",
        "2. Is Separation > 0.15? Below that, the format can't reliably distinguish"
        " true calls from noise.",
        "3. Do any `plus_*` variants beat `baseline`? If so, that metadata improves"
        " embedding quality.",
        "4. Do `plus_imports` or `plus_siblings` improve over `plus_full`?"
        " If not, the extra context adds noise.",
        "5. Check error analysis: are worst pairs cross-domain calls (expected hard)"
        " or same-module calls (unexpected failure)?",
        "",
        "## Variant Descriptions",
        "",
        "All variants start with the **baseline** (current production format from"
        " `build_summary_embedding_text`): file path + file context summary +"
        " class context summary + function name + LLM summary + signature + docstring.",
        "",
        "| Variant | Content |",
        "|---------|---------|",
        "| `baseline` | Current production format (no additions) |",
        "| `plus_calls` | baseline + outbound call names |",
        "| `plus_params` | baseline + typed parameter list + return type |",
        "| `plus_calls_params` | baseline + calls + typed params + return type |",
        "| `plus_full` | baseline + calls + params + returns + location breadcrumbs"
        " + decorators + detected patterns |",
        "| `plus_imports` | plus_full + file-level import module names |",
        "| `plus_siblings` | plus_full + sibling function names (same parent) |",
        "",
        "## Results",
        "",
        "| Variant | MRR | Hit@1 | Hit@3 | Hit@5 | Pos Sim | Neg Sim | Separation | Time |",
        "|---------|-----|-------|-------|-------|---------|---------|------------|------|",
    ]

    for r in results:
        marker = " **" if r.name == best_name else ""
        lines.append(
            f"| `{r.name}`{marker} | {r.mrr:.4f} | {r.hit_at_1:.3f} | {r.hit_at_3:.3f} | "
            f"{r.hit_at_5:.3f} | {r.mean_pos_sim:.4f} | {r.mean_neg_sim:.4f} | "
            f"{r.separation:.4f} | {r.embed_time_s:.1f}s |"
        )

    # Error analysis for best variant
    best = results[0] if results else None
    if best and best.worst_pairs:
        lines.extend([
            "",
            f"## Error Analysis: `{best.name}`",
            "",
            "Top-10 worst pairs (highest rank = hardest for the model to identify).",
            "Look for patterns: are failures concentrated on specific call types,"
            " cross-module calls, or utility functions?",
            "",
            "| Rank | Call | Pos Sim | Caller | Callee |",
            "|------|------|---------|--------|--------|",
        ])
        for wp in best.worst_pairs[:10]:
            # Shorten IDs for readability
            caller_short = wp["caller_id"].rsplit(":", 3)[-3:]
            callee_short = wp["callee_id"].rsplit(":", 3)[-3:]
            lines.append(
                f"| {wp['rank']} | `{wp['call_name']}` | {wp['pos_sim']:.4f} | "
                f"`{'·'.join(caller_short)}` | `{'·'.join(callee_short)}` |"
            )

    lines.extend([
        "",
        "## Next Steps",
        "",
        "Based on these results, consider:",
        "- If a `plus_*` variant beats `baseline`, add that metadata to"
        " `src/shared/ai/embedding.py:build_summary_embedding_text()`",
        "- If `plus_calls` improves MRR, outbound call names help the embedding"
        " model understand function relationships",
        "- If `plus_params` helps, typed signatures provide useful structural signal",
        "- If adding imports/siblings doesn't help, the embedding model may not leverage"
        " that context effectively",
        "- Re-run with `--n-callers 200 --n-negatives 1000` for more statistically"
        " significant results",
        "",
    ])
    out_path.write_text("\n".join(lines))
    return str(out_path)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_benchmark(
    scope: str,
    repository: str,
    n_callers: int = 50,
    n_negatives: int = 200,
    seed: int = 42,
    batch_size: int = 20,
    variants: list[str] | None = None,
    output_dir: str = "plans/benchmarks/data",
) -> None:
    """Run the full passport embedding benchmark."""
    if not variants:
        variants = list(ALL_VARIANT_NAMES)

    # Validate variant names
    for v in variants:
        if v not in ALL_VARIANT_NAMES:
            console.print(f"[red]Error:[/] Unknown variant '{v}'")
            console.print(f"  Available: {', '.join(ALL_VARIANT_NAMES)}")
            sys.exit(1)

    console.print("[bold]Semantic Passport Embedding Benchmark[/bold]")
    console.print(f"  scope={scope} repository={repository}")
    console.print(f"  variants={', '.join(variants)}")
    console.print()

    # Phase 1: Connect
    console.print("[bold]Phase 1:[/] Connecting to search backend + validating embedding model...")
    repo = _connect()
    embed_client = _verify_embedding_model()
    console.print("  [green]OK[/]")

    # Phase 2: Sample ground truth
    console.print("\n[bold]Phase 2:[/] Sampling ground truth pairs...")
    pairs, caller_ids, callee_ids = _sample_ground_truth(repo, scope, repository, n_callers, seed)
    if not pairs:
        console.print("[red]Error:[/] No valid ground truth pairs found")
        sys.exit(1)

    # Phase 3: Negative distractors
    console.print("\n[bold]Phase 3:[/] Sampling negative distractors...")
    exclude = caller_ids | callee_ids
    neg_ids = _sample_negatives(repo, scope, repository, n_negatives, exclude, seed)

    # Phase 4: Fetch full documents
    console.print("\n[bold]Phase 4:[/] Fetching documents...")
    all_ids = list(caller_ids | callee_ids | neg_ids)
    client = repo._get_client()
    docs = _fetch_documents(client, all_ids)
    console.print(f"  Fetched {len(docs)} documents")

    # Build parent summaries cache (needed for baseline in all variants)
    console.print("  Building parent summaries cache...")
    file_summaries_cache, class_summaries_cache = _build_parent_summaries_cache(
        client, docs, scope, repository
    )
    console.print(
        f"    Cached {len(file_summaries_cache)} file summaries, "
        f"{len(class_summaries_cache)} class summaries"
    )

    # Check which variants need extra context
    needs_imports = "plus_imports" in variants
    needs_siblings = "plus_siblings" in variants

    # Build context caches
    file_imports_cache: dict[str, list[dict]] = {}
    sibling_names_cache: dict[str, list[str]] = {}

    if needs_imports:
        console.print("  Building file imports cache...")
        file_imports_cache = _build_file_imports_cache(client, docs, scope, repository)
        console.print(f"    Cached imports for {len(file_imports_cache)} files")

    if needs_siblings:
        console.print("  Building sibling names cache...")
        sibling_names_cache = _build_sibling_names_cache(client, docs)
        console.print(f"    Cached siblings for {len(sibling_names_cache)} elements")

    # Phase 5 + 6: Build texts, embed, evaluate per variant
    # Use a global text→vector cache to avoid re-embedding identical texts
    # across variants (e.g. elements with no calls/params produce same text)
    console.print("\n[bold]Phase 5+6:[/] Building passport texts + embedding + evaluating...")
    results: list[VariantResult] = []
    neg_id_list = list(neg_ids)
    text_vector_cache: dict[str, list[float]] = {}

    for variant in variants:
        console.print(f"\n  [bold]{variant}[/]...")

        # Build texts for all elements
        texts_by_id: dict[str, str] = {}
        for eid, doc in docs.items():
            texts_by_id[eid] = build_passport_text(
                variant,
                doc,
                file_summaries_cache,
                class_summaries_cache,
                file_imports_cache,
                sibling_names_cache,
            )

        # Split into cached (already embedded) and new (need embedding)
        new_texts: list[str] = []
        new_ids: list[str] = []
        cached_count = 0
        for eid in texts_by_id:
            text = texts_by_id[eid]
            if text in text_vector_cache:
                cached_count += 1
            else:
                new_texts.append(text)
                new_ids.append(eid)

        # Embed only new texts
        t0 = time.monotonic()
        if new_texts:
            console.print(
                f"    Embedding {len(new_texts)} texts "
                f"({cached_count} cached from prior variants)..."
            )
            new_vectors = _embed_all(embed_client, new_texts, batch_size)
            for text, vec in zip(new_texts, new_vectors):
                text_vector_cache[text] = vec
        else:
            console.print(f"    All {cached_count} texts cached from prior variants")
        embed_time = time.monotonic() - t0
        console.print(f"    Done in {embed_time:.1f}s")

        # Map embeddings from cache
        embeddings: dict[str, list[float]] = {}
        for eid, text in texts_by_id.items():
            embeddings[eid] = text_vector_cache[text]

        # Evaluate
        result = _evaluate_variant(variant, pairs, embeddings, neg_id_list)
        result.embed_time_s = embed_time
        results.append(result)
        console.print(
            f"    MRR={result.mrr:.4f} | Hit@1={result.hit_at_1:.3f} | "
            f"Sep={result.separation:.4f}"
        )

        # Free per-variant data to reduce memory pressure
        del texts_by_id, embeddings
        gc.collect()

    # Phase 7: Report
    console.print("\n[bold]Phase 7:[/] Results")
    _print_results(results, len(pairs), len(neg_ids))

    md_path = _save_markdown(results, len(pairs), len(neg_ids), n_callers, seed, output_dir)
    console.print(f"\n  [green]Saved to:[/] {md_path}")

    repo.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark semantic passport embedding formats"
    )
    parser.add_argument("--scope", default=None, help="Repository scope")
    parser.add_argument("--repository", default=None, help="Repository name")
    parser.add_argument("--n-callers", type=int, default=50, help="Number of callers to sample")
    parser.add_argument("--n-negatives", type=int, default=200, help="Number of negative distractors")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=20, help="Embedding batch size")
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated variant names (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="plans/benchmarks/data",
        help="Output directory for markdown report",
    )

    args = parser.parse_args()

    # Auto-detect scope/repository if not provided
    scope = args.scope
    repository = args.repository
    if not scope or not repository:
        auto_scope, auto_repo = _auto_detect_repo()
        scope = scope or auto_scope
        repository = repository or auto_repo

    if not scope or not repository:
        console.print(
            "[red]Error:[/] Could not determine scope/repository. "
            "Provide --scope and --repository, or run from a directory with magaldi.yaml"
        )
        sys.exit(1)

    variants = None
    if args.variants:
        variants = [v.strip() for v in args.variants.split(",")]

    run_benchmark(
        scope=scope,
        repository=repository,
        n_callers=args.n_callers,
        n_negatives=args.n_negatives,
        seed=args.seed,
        batch_size=args.batch_size,
        variants=variants,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
