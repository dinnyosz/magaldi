#!/usr/bin/env python3
"""Generate training data for the variable scorer model.

Parses repos directly (Phases 1-3) to extract variables, scores them
individually with a teacher model via Ollama, quality-filters, and
outputs ChatML JSONL for fine-tuning.

Usage:
    python tools/training/generate_scoring_data.py \
      --repos-dir test_repos \
      --sample-size 10000 \
      --teacher-model qwen3-coder:30b \
      --output-dir tools/training/data/variable_scorer \
      --cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

# Add project root to path so we can import magaldi modules
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root / "src"))

import litellm

from magaldi_core.change_detection import (
    ChangeManifest,
    FileInfo,
    enumerate_files,
)
from magaldi_core.code_parser import parse_files
from magaldi_core.discovery import discover
from magaldi_core.variable_scoring.heuristic_filter import should_drop_variable
from magaldi_core.variable_scoring.prompts import build_user_prompt

# Suppress litellm noise
litellm.suppress_debug_info = True
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")

logger = logging.getLogger(__name__)

# Number of scoring dimensions (expanded from production's 4 to 7)
NUM_SCORES = 7

# =============================================================================
# EXPANDED TEACHER PROMPT (7 dimensions — production uses 4)
# =============================================================================

TEACHER_SYSTEM_PROMPT = """\
You are scoring variables for a coding agent's search index. The agent uses this \
index to find relevant code when navigating a codebase. Score each variable on \
seven dimensions (1-10) as comma-separated integers:
config_value,architectural_role,data_definition,general_usefulness,value_complexity,naming_quality,scope_significance

Ask: "If a coding agent searched for how this system works, would finding this \
variable help?" If no, score 1,1,1,1,1,1,1.

Scoring dimensions:
- config_value: Configuration, feature flag, tuning parameter, URL, path, prompt \
template, SQL/query string, default value, __all__ export list
- architectural_role: Infrastructure (DB connection, router, logger, app instance, \
middleware, compiled regex, decorator, sentinel)
- data_definition: Data structure, schema, type alias, enum, named tuple, mapping, \
protocol, TypeVar
- general_usefulness: Would a coding agent benefit from finding this while working?
- value_complexity: How complex/interesting is the assigned value? Simple literal=1, \
function call=3, multi-element collection/dict/config=7, complex expression/template=9
- naming_quality: How descriptive/self-documenting is the name? Single letter=1, \
abbreviation=3, clear descriptive name=7, fully qualified domain name=9
- scope_significance: Module-level constant=9, class attribute=6, function local=2, \
loop/temp variable=1

Rules:
- Output ONLY the number and seven scores per line. Never echo the variable code.
- Format: N. score,score,score,score,score,score,score
- Most variables should score LOW. Only ~30% of variables are worth keeping.
- Score 1,1,1,1,1,1,1 for: loop counters, temp variables, function/method call results, \
short names (i, j, x, tmp, res, val, err, _), intermediate computations, \
local assignments inside functions
- Score 1,1,1,1,1,1,1 for: generic assignments like result = func(), data = obj.method(), \
response = requests.get(), items = process(), client = Client()
- Score HIGH only for: module-level constants, string templates/prompts, framework \
instances, DB connections, loggers, type aliases, enums, compiled patterns, \
configuration dicts/lists, query strings, export lists, sentinels

Example input:
1. [src/app.py] MAX_RETRIES = 3
2. [src/app.py] app = Flask(__name__)
3. [src/app.py] result = process_items(data)
4. [src/db.py] engine = create_engine(DATABASE_URL)
5. [src/db.py] logger = logging.getLogger(__name__)
6. [src/db.py] data = json.loads(payload)

Correct output (scores only, no code):
1. 9,1,1,8,2,8,9
2. 1,10,1,9,3,7,9
3. 1,1,1,1,2,2,2
4. 1,9,1,9,4,7,9
5. 1,9,1,7,3,7,9
6. 1,1,1,1,2,1,2\
"""


def _parse_teacher_scores(
    output: str, batch_size: int
) -> list[tuple[int, ...] | None]:
    """Parse teacher output into score tuples (7 dimensions).

    Expected format: N. s,s,s,s,s,s,s
    """

    # Strip thinking blocks
    output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()

    scores: list[tuple[int, ...] | None] = [None] * batch_size
    # Match N. followed by exactly 7 comma-separated numbers
    pattern = re.compile(
        r"(\d+)\.\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)"
        r"\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)"
    )

    for match in pattern.finditer(output):
        idx = int(match.group(1))
        if 1 <= idx <= batch_size:
            try:
                vals = tuple(
                    min(10, max(1, int(match.group(g))))
                    for g in range(2, 2 + NUM_SCORES)
                )
                scores[idx - 1] = vals
            except (ValueError, IndexError):
                pass

    return scores


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class RawScoringResult:
    """A single teacher-scored variable, saved to raw/ for resume."""

    element_id: str
    file_path: str
    name: str
    raw_code: str
    repo_name: str
    language: str
    scores: tuple[int, ...]  # 7 dimensions
    heuristic_drop: bool
    heuristic_reason: str
    teacher_model: str
    timestamp: str


# =============================================================================
# PHASE 1-3: PARSE REPOS DIRECTLY
# =============================================================================


def extract_variables_from_repo(repo_path: str) -> list[dict]:
    """Run Phases 1-3 on a repo and return all variable/constant elements.

    Args:
        repo_path: Path to repo root (must have magaldi.yaml).

    Returns:
        List of dicts with keys: element_id, file_path, name, raw_code,
        repo_name, language.
    """
    # Phase 1: Discovery
    discovery_result = discover(repo_path, username="main")
    repo_name = discovery_result.repository
    logger.info(
        "Discovered %s: %d files, %d lines",
        repo_name,
        discovery_result.total_files,
        discovery_result.total_lines,
    )

    # Enumerate all files directly (skip change detection — we want everything)
    files = enumerate_files(
        discovery_result.repo_path,
        discovery_result.exclude_directories,
        discovery_result.exclude_files,
    )
    manifest = ChangeManifest(
        scope=discovery_result.scope,
        repository=repo_name,
        username="main",
        timestamp=datetime.now(),
        total_files_scanned=len(files),
        new_files=files,
    )
    logger.info("Files to parse: %d", manifest.files_to_parse)

    # Phase 3: Parsing
    parsing_result = parse_files(manifest)
    logger.info(
        "Parsed %d files, %d total elements, by type: %s",
        len(parsing_result.parsed_files),
        parsing_result.total_elements,
        parsing_result.elements_by_type,
    )

    # Extract variables and constants
    variables = []
    for parsed_file in parsing_result.parsed_files:
        for element in parsed_file.elements:
            if element.element_type in ("variable", "constant"):
                variables.append({
                    "element_id": element.element_id,
                    "file_path": element.relative_path,
                    "name": element.name,
                    "raw_code": element.raw_code,
                    "repo_name": repo_name,
                    "language": parsed_file.file_info.language,
                })

    logger.info("Extracted %d variables/constants from %s", len(variables), repo_name)
    return variables


# =============================================================================
# TEACHER SCORING
# =============================================================================


def score_variable_with_teacher(
    variable: dict,
    teacher_model: str,
    ollama_url: str,
    temperature: float = 0.2,
    timeout: int = 120,
) -> tuple[int, ...] | None:
    """Score a single variable using the teacher model (7 dimensions).

    Returns tuple of 7 scores or None on failure.
    """
    # Build prompt using the same format the student will see
    prompt_vars = [(1, variable["file_path"], variable["name"], variable["raw_code"])]
    user_prompt = build_user_prompt(prompt_vars, token_budget=2048)

    messages = [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = litellm.completion(
            model=f"ollama_chat/{teacher_model}",
            messages=messages,
            temperature=temperature,
            max_tokens=80,
            api_base=ollama_url,
            timeout=timeout,
            think=False,
        )

        content = response.choices[0].message.content or ""
        if not content.strip():
            reasoning = getattr(response.choices[0].message, "reasoning_content", None)
            if reasoning:
                content = reasoning

        # Parse scores (7 dimensions)
        parsed = _parse_teacher_scores(content, 1)
        if parsed[0] is not None:
            return parsed[0]

        logger.warning(
            "Failed to parse teacher output for %s: %r",
            variable["name"],
            content,
        )
        return None

    except Exception:
        logger.warning(
            "Teacher scoring failed for %s",
            variable["name"],
            exc_info=True,
        )
        return None


def score_batch_with_teacher(
    variables: list[dict],
    teacher_model: str,
    ollama_url: str,
    temperature: float = 0.2,
    timeout: int = 120,
) -> list[tuple[int, ...] | None]:
    """Score a batch of variables using the teacher model (7 dimensions).

    Returns list of 7-score tuples, None for failures.
    """
    prompt_vars = [
        (i + 1, v["file_path"], v["name"], v["raw_code"])
        for i, v in enumerate(variables)
    ]
    user_prompt = build_user_prompt(prompt_vars, token_budget=2048)

    messages = [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        output_budget = len(variables) * 30 + 50
        response = litellm.completion(
            model=f"ollama_chat/{teacher_model}",
            messages=messages,
            temperature=temperature,
            max_tokens=output_budget,
            api_base=ollama_url,
            timeout=timeout,
            think=False,
        )

        content = response.choices[0].message.content or ""
        if not content.strip():
            reasoning = getattr(response.choices[0].message, "reasoning_content", None)
            if reasoning:
                content = reasoning

        return _parse_teacher_scores(content, len(variables))

    except Exception:
        logger.warning(
            "Teacher batch scoring failed for %d variables",
            len(variables),
            exc_info=True,
        )
        return [None] * len(variables)


# =============================================================================
# QUALITY FILTERING
# =============================================================================


def validate_scores(
    variable: dict,
    scores: tuple[int, ...],
) -> tuple[bool, str]:
    """Validate a single scored variable.

    Returns (is_valid, reason).
    """
    # Gate 1: Dimension count check
    if len(scores) != NUM_SCORES:
        return False, f"wrong_dim_count_{len(scores)}"

    # Gate 2: Range check
    if not all(1 <= s <= 10 for s in scores):
        return False, "out_of_range"

    # Gate 3: Heuristic agreement
    heuristic_drop, _reason = should_drop_variable(variable["name"], variable["raw_code"])
    if heuristic_drop:
        max_score = max(scores)
        if max_score >= 7:
            return False, f"heuristic_disagreement_{variable['name']}"

    # Gate 4: UPPER_CASE constants should generally score high
    name = variable["name"]
    if name.isupper() and len(name) > 3 and max(scores) < 4:
        return False, f"constant_scored_low_{name}"

    return True, "ok"


def validate_batch_scores(
    variables: list[dict],
    all_scores: list[tuple[int, ...]],
) -> tuple[bool, str]:
    """Validate a batch of scored variables.

    Returns (is_valid, reason).
    """
    # Gate: Variance — reject if all variables got identical scores
    if len(all_scores) > 3:
        unique = set(all_scores)
        if len(unique) == 1:
            return False, "uniform_scores"

    # Validate each individual variable
    for var, scores in zip(variables, all_scores):
        valid, reason = validate_scores(var, scores)
        if not valid:
            return False, reason

    return True, "ok"


# =============================================================================
# RESUME SUPPORT
# =============================================================================


def _result_hash(element_id: str) -> str:
    """Deterministic hash for an element_id, used as raw result filename."""
    return hashlib.sha256(element_id.encode()).hexdigest()[:16]


def load_existing_results(raw_dir: Path) -> dict[str, RawScoringResult]:
    """Load previously scored results from raw/ directory.

    Returns dict of element_id -> RawScoringResult.
    """
    results: dict[str, RawScoringResult] = {}
    if not raw_dir.exists():
        return results

    for path in raw_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            result = RawScoringResult(
                element_id=data["element_id"],
                file_path=data["file_path"],
                name=data["name"],
                raw_code=data["raw_code"],
                repo_name=data["repo_name"],
                language=data["language"],
                scores=tuple(data["scores"]),
                heuristic_drop=data["heuristic_drop"],
                heuristic_reason=data["heuristic_reason"],
                teacher_model=data["teacher_model"],
                timestamp=data["timestamp"],
            )
            results[result.element_id] = result
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Skipping corrupt raw result: %s", path)
    return results


def save_raw_result(result: RawScoringResult, raw_dir: Path) -> None:
    """Save a single scoring result to raw/ for resume."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = _result_hash(result.element_id) + ".json"
    path = raw_dir / filename
    data = {
        "element_id": result.element_id,
        "file_path": result.file_path,
        "name": result.name,
        "raw_code": result.raw_code,
        "repo_name": result.repo_name,
        "language": result.language,
        "scores": list(result.scores),
        "heuristic_drop": result.heuristic_drop,
        "heuristic_reason": result.heuristic_reason,
        "teacher_model": result.teacher_model,
        "timestamp": result.timestamp,
    }
    path.write_text(json.dumps(data, indent=2))


# =============================================================================
# CHATML FORMATTING
# =============================================================================


def format_training_example(
    variables: list[dict],
    scores_list: list[tuple[int, ...]],
) -> dict:
    """Format a batch of variables + scores as a ChatML conversation.

    Returns a dict with "conversations" key for JSONL output.
    """
    # Build user prompt (same format as production)
    prompt_vars = [
        (i + 1, v["file_path"], v["name"], v["raw_code"])
        for i, v in enumerate(variables)
    ]
    user_prompt = build_user_prompt(prompt_vars, token_budget=2048)

    # Remove /no_think from training data (that's a runtime workaround)
    if user_prompt.startswith("/no_think\n"):
        user_prompt = user_prompt[len("/no_think\n"):]

    # Build assistant response (scores only, 7 per line)
    response_lines = []
    for i, scores in enumerate(scores_list):
        scores_str = ",".join(str(s) for s in scores)
        response_lines.append(f"{i + 1}. {scores_str}")
    assistant_content = "\n".join(response_lines)

    return {
        "conversations": [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def _estimate_variable_tokens(result: RawScoringResult) -> int:
    """Estimate total token cost for one variable in a conversation.

    Includes both the user-prompt line and the assistant score line.
    Uses ~3 chars per token — conservative ratio because code (brackets,
    underscores, paths, operators) tokenizes less efficiently than prose.
    """
    # User line: "N. [file_path] raw_code\n"
    code = result.raw_code.replace("\n", " ").strip()
    user_chars = len(f"1. [{result.file_path}] {code}\n")

    # Assistant line: "N. s,s,s,s,s,s,s\n" — 7 scores, each 1-2 digits
    response_chars = len(f"1. {','.join(str(s) for s in result.scores)}\n")

    return (user_chars + response_chars) // 3 + 5  # +5 for tokenizer overhead


def _estimate_system_prompt_tokens() -> int:
    """Estimate token cost of the system prompt + ChatML framing."""
    # ~3 chars/token for code-heavy content + ChatML overhead
    # (<|im_start|>system, <|im_end|>, role tags per message ≈ 25 tokens)
    return len(TEACHER_SYSTEM_PROMPT) // 3 + 25


def _is_trivial(result: RawScoringResult) -> bool:
    """Check if a result is trivial (all scores are 1)."""
    return all(s == 1 for s in result.scores)


def rebalance_results(
    results: list[RawScoringResult],
    max_trivial_ratio: float = 0.3,
    seed: int = 42,
) -> list[RawScoringResult]:
    """Rebalance training data by undersampling trivial (all-1s) examples.

    The raw training data is typically ~49% all-1s variables (trivial).
    This creates a strong bias toward outputting all-1s. Undersampling
    trivial examples to ``max_trivial_ratio`` forces the model to learn
    non-trivial scoring patterns from a more balanced distribution.

    Non-trivial examples are never dropped. Only trivial examples are
    removed to reach the target ratio.

    Args:
        results: All scored results (will not be mutated).
        max_trivial_ratio: Maximum fraction of trivial examples in the
            output. Set to 1.0 to disable rebalancing.
        seed: Random seed for reproducible undersampling.

    Returns:
        Rebalanced list of results.
    """
    if max_trivial_ratio >= 1.0:
        return list(results)

    trivial = [r for r in results if _is_trivial(r)]
    non_trivial = [r for r in results if not _is_trivial(r)]

    if not non_trivial:
        return list(results)

    # Calculate how many trivial examples to keep
    # target: trivial / (trivial + non_trivial) <= max_trivial_ratio
    # trivial <= max_trivial_ratio * (trivial + non_trivial)
    # trivial <= max_trivial_ratio * non_trivial / (1 - max_trivial_ratio)
    max_trivial = int(
        max_trivial_ratio * len(non_trivial) / (1.0 - max_trivial_ratio)
    )
    max_trivial = max(1, max_trivial)  # keep at least 1

    if len(trivial) <= max_trivial:
        return list(results)

    # Undersample trivial
    rng = random.Random(seed)
    rng.shuffle(trivial)
    kept_trivial = trivial[:max_trivial]

    rebalanced = non_trivial + kept_trivial
    logger.info(
        "Rebalanced: %d trivial -> %d (%.0f%%), %d non-trivial kept, "
        "%d total (was %d)",
        len(trivial),
        len(kept_trivial),
        len(kept_trivial) / len(rebalanced) * 100,
        len(non_trivial),
        len(rebalanced),
        len(results),
    )
    return rebalanced


def build_training_batches(
    results: list[RawScoringResult],
    seed: int = 42,
    max_seq_len: int = 1024,
) -> list[dict]:
    """Build ChatML training examples using greedy token-budget packing.

    Packs variables into batches until adding the next variable would exceed
    the token budget (max_seq_len). This guarantees every training example
    fits within max_seq_len with no silent truncation or wasted padding.

    Batch sizes are dynamic: short variables produce bigger batches,
    long variables produce smaller batches.

    Args:
        results: Scored variables to assemble into training conversations.
        seed: Random seed for shuffling.
        max_seq_len: Maximum sequence length in tokens. Batches are packed
            to fit within this budget.

    Returns:
        List of ChatML conversation dicts ready for JSONL output.
    """
    rng = random.Random(seed)
    results = list(results)  # copy to avoid mutating caller's list
    rng.shuffle(results)

    system_tokens = _estimate_system_prompt_tokens()
    # Header line "Score these variables:\n" in user prompt
    header_tokens = 8
    available_budget = max_seq_len - system_tokens - header_tokens

    examples = []
    current_batch: list[RawScoringResult] = []
    current_tokens = 0

    for result in results:
        var_tokens = _estimate_variable_tokens(result)

        # If this single variable exceeds the entire budget, truncate its
        # code and add it as a solo example (don't skip it)
        if var_tokens > available_budget:
            # Flush current batch first
            if current_batch:
                examples.append(_batch_to_example(current_batch))
                current_batch = []
                current_tokens = 0
            # Add as solo example (format_training_example will truncate)
            examples.append(_batch_to_example([result]))
            continue

        # Would this variable overflow the current batch?
        if current_tokens + var_tokens > available_budget and current_batch:
            examples.append(_batch_to_example(current_batch))
            current_batch = []
            current_tokens = 0

        current_batch.append(result)
        current_tokens += var_tokens

    # Flush remaining
    if current_batch:
        examples.append(_batch_to_example(current_batch))

    return examples


def _batch_to_example(batch: list[RawScoringResult]) -> dict:
    """Convert a batch of RawScoringResults to a ChatML training example."""
    variables = [
        {
            "file_path": r.file_path,
            "name": r.name,
            "raw_code": r.raw_code,
        }
        for r in batch
    ]
    scores_list = [r.scores for r in batch]
    return format_training_example(variables, scores_list)


# =============================================================================
# PROGRESS OUTPUT
# =============================================================================


def _format_duration(seconds: float) -> str:
    """Format seconds as mm:ss or hh:mm:ss."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ANSI color codes
_GREEN_BG = "\033[42;30m"  # green background, black text
_RED_BG = "\033[41;37m"    # red background, white text
_YELLOW_BG = "\033[43;30m" # yellow background, black text
_RED_FG = "\033[31m"       # red text
_RESET = "\033[0m"

# Short labels for source/reason
_SOURCE_ABBREV = {
    "single_letter": "H:SL",
    "throwaway_name": "H:TN",
    "throwaway_call_result": "H:TCR",
    "short_call_result": "H:SCR",
    "out_of_range": "F:rng",
    "uniform_scores": "F:uni",
}


def _short_source(source: str) -> str:
    """Shorten source label for display."""
    if source in _SOURCE_ABBREV:
        return _SOURCE_ABBREV[source]
    if source.startswith("heuristic_disagreement"):
        return "F:hd"
    if source.startswith("constant_scored_low"):
        return "F:csl"
    if source.startswith("wrong_dim_count"):
        return "F:dim"
    return source[:5]


def _print_progress_line(
    current: int,
    total: int,
    variable: dict,
    scores: tuple[int, ...] | None,
    decision: str,
    source: str,
    start_time: float,
) -> None:
    """Print a single-line progress update for each variable."""
    elapsed = time.time() - start_time
    rate = current / elapsed if elapsed > 0 else 0
    eta = (total - current) / rate if rate > 0 else 0

    # Truncate variable name and value for display
    name = variable["name"][:20].ljust(20)
    raw_code = variable.get("raw_code", "")
    # Extract the value part (after the = sign), collapse to one line
    if "=" in raw_code:
        value = raw_code.split("=", 1)[1].strip()
    else:
        value = raw_code.strip()
    value = " ".join(value.split())  # collapse newlines/whitespace
    if len(value) > 30:
        value = value[:27] + "..."
    value = value.ljust(30)

    # Right-align path so filename is always visible (truncate directory prefix)
    fpath = variable["file_path"]
    if len(fpath) > 25:
        fpath = "…" + fpath[-(24):]
    fpath = fpath.rjust(25)

    scores_str = ",".join(str(s) for s in scores) if scores else "-" * (NUM_SCORES * 2 - 1)
    src = _short_source(source)

    # Color the decision tag
    if decision == "KEEP":
        tag = f"{_GREEN_BG} KEEP {_RESET}"
    elif decision == "DROP":
        tag = f"{_RED_BG} DROP {_RESET}"
    elif decision == "FAIL":
        tag = f"{_RED_FG}FAIL{_RESET} "
    elif decision == "FILT":
        tag = f"{_YELLOW_BG} FILT {_RESET}"
    else:
        tag = f"{decision:<6}"

    print(
        f"[{current:>5}/{total}] {tag} {scores_str:<20} "
        f"{name}= {value} "
        f"{fpath} "
        f"{src:<5} "
        f"{_format_duration(elapsed)} ~{_format_duration(eta)} "
        f"{rate:.1f}/s",
        flush=True,
    )


def _print_scoring_summary(
    elapsed: float,
    scored: int,
    heuristic: int,
    teacher: int,
    failed: int,
    filtered: int,
    kept: int,
    dropped: int,
    total: int,
) -> None:
    """Print a summary after scoring completes."""
    print(f"\n{'='*80}")
    print("Scoring Summary")
    print(f"{'='*80}")
    print(f"  Total variables:    {total}")
    print(f"  Scored:             {scored}  ({scored/total*100:.0f}%)" if total else "")
    print(f"    Heuristic (all-1s):{heuristic}")
    print(f"    Teacher model:    {teacher}")
    print(f"  Failed:             {failed}")
    print(f"  Filtered (invalid): {filtered}")
    print(f"  ---")
    print(f"  KEEP (max >= 5):    {kept}  ({kept/scored*100:.0f}%)" if scored else "")
    print(f"  DROP (max < 5):     {dropped}  ({dropped/scored*100:.0f}%)" if scored else "")
    print(f"  ---")
    print(f"  Time:               {_format_duration(elapsed)}")
    print(f"  Rate:               {scored/elapsed:.1f} vars/s" if elapsed else "")
    print(f"{'='*80}\n")


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def discover_repos(repos_dir: str) -> list[str]:
    """Find all subdirectories with magaldi.yaml in the given directory."""
    repos_path = Path(repos_dir)
    if not repos_path.is_dir():
        logger.error("Repos directory not found: %s", repos_dir)
        sys.exit(1)

    repos = []
    for child in sorted(repos_path.iterdir()):
        if child.is_dir() and (child / "magaldi.yaml").exists():
            repos.append(str(child))

    if not repos:
        logger.error("No repos with magaldi.yaml found in %s", repos_dir)
        sys.exit(1)

    return repos


def sample_variables_evenly(
    per_repo_vars: dict[str, list[dict]],
    sample_size: int,
    seed: int = 42,
) -> list[dict]:
    """Sample variables evenly across repos up to sample_size.

    Each repo contributes at most ceil(sample_size / num_repos) variables.
    If a repo has fewer, the remainder is redistributed to other repos.
    """
    rng = random.Random(seed)
    repos = sorted(per_repo_vars.keys())
    num_repos = len(repos)
    per_repo_budget = sample_size // num_repos

    # Shuffle each repo's variables
    for repo in repos:
        rng.shuffle(per_repo_vars[repo])

    # First pass: take up to per_repo_budget from each
    sampled: list[dict] = []
    leftover_repos: list[str] = []
    remaining = sample_size

    for repo in repos:
        available = per_repo_vars[repo]
        take = min(per_repo_budget, len(available), remaining)
        sampled.extend(available[:take])
        remaining -= take
        if len(available) > take:
            leftover_repos.append(repo)

    # Second pass: fill remaining budget from repos that have more
    if remaining > 0 and leftover_repos:
        rng.shuffle(leftover_repos)
        for repo in leftover_repos:
            already_taken = min(per_repo_budget, len(per_repo_vars[repo]))
            available = per_repo_vars[repo][already_taken:]
            take = min(len(available), remaining)
            sampled.extend(available[:take])
            remaining -= take
            if remaining <= 0:
                break

    rng.shuffle(sampled)
    return sampled


def run_generation(args: argparse.Namespace) -> None:
    """Main data generation pipeline."""
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Check existing raw data before expensive repo parsing
    existing: dict[str, RawScoringResult] = {}
    if not args.no_cache:
        existing = load_existing_results(raw_dir)
        if existing:
            logger.info("Loaded %d existing results from raw/", len(existing))

    # Early exit: if we already have enough raw data, skip parsing + scoring
    if existing and len(existing) >= args.sample_size and not args.force_rescore:
        logger.info(
            "Already have %d raw results (>= sample_size %d). "
            "Skipping repo parsing and scoring. "
            "Use --force-rescore to re-score anyway, or --no-cache to start fresh.",
            len(existing),
            args.sample_size,
        )
        print(
            f"\n{'='*80}\n"
            f"Raw data sufficient: {len(existing)} scored variables "
            f"(requested {args.sample_size})\n"
            f"Skipping repo parsing and teacher scoring — "
            f"proceeding to training data assembly.\n"
            f"{'='*80}\n"
        )
    else:
        # Step 2: Discover repos and parse variables
        repo_paths = discover_repos(args.repos_dir)
        logger.info("Found %d repos in %s", len(repo_paths), args.repos_dir)

        per_repo_vars: dict[str, list[dict]] = {}
        for repo_path in repo_paths:
            repo_name = Path(repo_path).name
            try:
                variables = extract_variables_from_repo(repo_path)
                per_repo_vars[repo_name] = variables
            except Exception:
                logger.error("Failed to parse repo: %s", repo_path, exc_info=True)

        total_available = sum(len(v) for v in per_repo_vars.values())
        logger.info(
            "Total variables available: %d from %d repos",
            total_available, len(per_repo_vars),
        )

        if not per_repo_vars:
            logger.error("No variables extracted from any repo.")
            sys.exit(1)

        # Step 3: Sample evenly across repos
        sample_size = min(args.sample_size, total_available)
        all_variables = sample_variables_evenly(per_repo_vars, sample_size, args.seed)

        # Print per-repo sample counts
        repo_counts: dict[str, int] = {}
        for v in all_variables:
            repo_counts[v["repo_name"]] = repo_counts.get(v["repo_name"], 0) + 1
        logger.info(
            "Sampled %d variables from %d repos: %s",
            len(all_variables), len(repo_counts),
            ", ".join(f"{k}={v}" for k, v in sorted(repo_counts.items())),
        )

        # Step 4: Determine what still needs scoring
        to_score = [v for v in all_variables if v["element_id"] not in existing]
        logger.info(
            "Variables to score: %d (skipping %d already scored)",
            len(to_score),
            len(all_variables) - len(to_score),
        )

        # Step 5: Score variables with teacher model
        scored_count = 0
        heuristic_count = 0
        teacher_count = 0
        failed_count = 0
        filtered_count = 0
        keep_count = 0
        drop_count = 0
        start_time = time.time()
        threshold = 5

        for i, variable in enumerate(to_score):
            # Check heuristic filter
            heuristic_drop, heuristic_reason = should_drop_variable(
                variable["name"], variable["raw_code"]
            )

            if heuristic_drop:
                # Known junk — assign all-1s without calling teacher
                scores = tuple(1 for _ in range(NUM_SCORES))
                heuristic_count += 1
                decision = "DROP"
                source = heuristic_reason
            else:
                # Score with teacher model
                scores = score_variable_with_teacher(
                    variable,
                    teacher_model=args.teacher_model,
                    ollama_url=args.ollama_url,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )

                if scores is None:
                    failed_count += 1
                    _print_progress_line(
                        i + 1, len(to_score), variable, None, "FAIL",
                        "T:err", start_time,
                    )
                    continue

                # Validate scores
                valid, reason = validate_scores(variable, scores)
                if not valid:
                    filtered_count += 1
                    _print_progress_line(
                        i + 1, len(to_score), variable, scores, "FILT",
                        reason, start_time,
                    )
                    continue

                teacher_count += 1
                max_s = max(scores)
                decision = "KEEP" if max_s >= threshold else "DROP"
                source = "T"

            # Track keep/drop
            if decision == "KEEP":
                keep_count += 1
            else:
                drop_count += 1

            # Save raw result
            result = RawScoringResult(
                element_id=variable["element_id"],
                file_path=variable["file_path"],
                name=variable["name"],
                raw_code=variable["raw_code"],
                repo_name=variable["repo_name"],
                language=variable["language"],
                scores=scores,
                heuristic_drop=heuristic_drop,
                heuristic_reason=heuristic_reason,
                teacher_model=args.teacher_model,
                timestamp=datetime.now().isoformat(),
            )
            save_raw_result(result, raw_dir)
            existing[result.element_id] = result
            scored_count += 1

            _print_progress_line(
                i + 1, len(to_score), variable, scores, decision,
                source, start_time,
            )

        elapsed = time.time() - start_time
        _print_scoring_summary(
            elapsed, scored_count, heuristic_count, teacher_count,
            failed_count, filtered_count, keep_count, drop_count,
            len(to_score),
        )

    # Step 4: Assemble all results and build training data
    all_results = list(existing.values())
    logger.info("Total scored results: %d", len(all_results))

    if not all_results:
        logger.error("No scored results to build training data from.")
        sys.exit(1)

    # Step 5: Split into train/validation
    rng = random.Random(args.seed)
    rng.shuffle(all_results)

    val_size = max(1, int(len(all_results) * args.validation_split))
    val_results = all_results[:val_size]
    train_results = all_results[val_size:]

    logger.info(
        "Split: %d train, %d validation (%.0f%% split)",
        len(train_results),
        len(val_results),
        args.validation_split * 100,
    )

    # Step 5b: Rebalance training data (undersample trivial all-1s examples)
    # Only rebalance training set — validation stays natural for honest eval
    train_results = rebalance_results(
        train_results,
        max_trivial_ratio=args.max_trivial_ratio,
        seed=args.seed,
    )

    # Step 6: Build ChatML examples with token-budget packing
    train_examples = build_training_batches(
        train_results, seed=args.seed, max_seq_len=args.max_seq_len,
    )
    val_examples = build_training_batches(
        val_results, seed=args.seed + 1, max_seq_len=args.max_seq_len,
    )

    # Step 7: Write JSONL files
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"

    with open(train_path, "w") as f:
        for example in train_examples:
            f.write(json.dumps(example) + "\n")

    with open(val_path, "w") as f:
        for example in val_examples:
            f.write(json.dumps(example) + "\n")

    # Step 8: Print final summary
    score_dist = {"drop": 0, "keep": 0, "borderline": 0}
    for r in all_results:
        max_s = max(r.scores)
        if max_s <= 2:
            score_dist["drop"] += 1
        elif max_s >= 7:
            score_dist["keep"] += 1
        else:
            score_dist["borderline"] += 1

    # Training data trivial ratio (after rebalancing)
    train_trivial = sum(1 for r in train_results if _is_trivial(r))
    train_trivial_pct = train_trivial / len(train_results) * 100 if train_results else 0

    lang_counts: dict[str, int] = {}
    for r in all_results:
        lang_counts[r.language] = lang_counts.get(r.language, 0) + 1

    # Batch size stats
    all_examples = train_examples + val_examples
    batch_sizes = []
    for ex in all_examples:
        convos = ex.get("conversations", [])
        for msg in convos:
            if msg.get("role") == "assistant":
                batch_sizes.append(msg["content"].count("\n") + 1)
    avg_batch = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0
    min_batch = min(batch_sizes) if batch_sizes else 0
    max_batch = max(batch_sizes) if batch_sizes else 0

    print(f"\n{'='*80}")
    print("Output")
    print(f"{'='*80}")
    print(f"  Train:      {len(train_examples)} examples -> {train_path}")
    print(f"  Validation: {len(val_examples)} examples -> {val_path}")
    print(f"  Max seq len:{args.max_seq_len} tokens")
    print(f"  Batch sizes: min={min_batch}, avg={avg_batch:.1f}, max={max_batch}"
          f" (packed to fit {args.max_seq_len} tokens)")
    print(f"  Train data: {len(train_results)} vars "
          f"({train_trivial} trivial = {train_trivial_pct:.0f}%, "
          f"max_trivial_ratio={args.max_trivial_ratio})")
    print(f"  Scores:     {score_dist['drop']} clear drops, "
          f"{score_dist['borderline']} borderline, "
          f"{score_dist['keep']} clear keeps")
    print(f"  Languages:  {lang_counts}")
    print(f"{'='*80}")


# =============================================================================
# CLI
# =============================================================================


def _load_config_defaults(config_path: str) -> dict:
    """Load data_generation + training defaults from a YAML config file.

    Returns a flat dict of param_name -> value that can be used as CLI defaults.
    Only includes keys that are present in the config.
    """
    import yaml

    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file not found: %s", config_path)
        return {}

    with open(path) as f:
        config = yaml.safe_load(f) or {}

    defaults: dict = {}
    dg = config.get("data_generation", {})
    training = config.get("training", {})

    # Map config keys -> CLI arg names
    _config_map = {
        # data_generation section
        "teacher_model": ("teacher_model", dg),
        "ollama_url": ("ollama_url", dg),
        "validation_split": ("validation_split", dg),
        "temperature": ("temperature", dg),
        "timeout": ("timeout", dg),
        "max_trivial_ratio": ("max_trivial_ratio", dg),
        "sample_size": ("sample_size", dg),
        # training section (for max_seq_len)
        "max_seq_len": ("max_seq_len", training),
    }

    for arg_name, (key, section) in _config_map.items():
        if key in section:
            defaults[arg_name] = section[key]

    return defaults


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate training data for the variable scorer model."
    )

    parser.add_argument(
        "--config",
        help="Path to training config YAML. Values from data_generation section "
             "are used as defaults (CLI flags override).",
    )
    parser.add_argument(
        "--repos-dir",
        required=True,
        help="Directory containing repos to parse (each subdir with magaldi.yaml)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Total variables to sample across all repos (default: 10000, or from config)",
    )
    parser.add_argument(
        "--teacher-model",
        default=None,
        help="Ollama teacher model name (default: qwen3-coder:30b, or from config)",
    )
    parser.add_argument(
        "--ollama-url",
        default=None,
        help="Ollama API URL (default: http://localhost:11434, or from config)",
    )
    parser.add_argument(
        "--output-dir",
        default="tools/training/data/variable_scorer",
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=None,
        help="Fraction of data to hold out for validation (default: 0.1, or from config)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Teacher model temperature (default: 0.2, or from config)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Teacher model timeout in seconds (default: 120, or from config)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore previously scored results and start fresh (default: cache is ON)",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="(Deprecated, now the default) Reuse previously scored results from raw/",
    )
    parser.add_argument(
        "--force-rescore",
        action="store_true",
        help="Force repo parsing and scoring even if raw/ already has enough data. "
             "Useful to add new repos or increase sample size.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Max sequence length in tokens for batch packing (default: 1024, or from config)",
    )
    parser.add_argument(
        "--max-trivial-ratio",
        type=float,
        default=None,
        help="Max fraction of trivial (all-1s) examples in training data (default: 0.3). "
             "Set to 1.0 to disable rebalancing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Load config defaults, then apply hardcoded defaults for anything still None
    config_defaults = {}
    if args.config:
        config_defaults = _load_config_defaults(args.config)

    _hardcoded_defaults = {
        "sample_size": 10000,
        "teacher_model": "qwen3-coder:30b",
        "ollama_url": "http://localhost:11434",
        "validation_split": 0.1,
        "temperature": 0.2,
        "timeout": 120,
        "max_seq_len": 1024,
        "max_trivial_ratio": 0.3,
    }

    # Priority: CLI flag > config file > hardcoded default
    for param, hardcoded in _hardcoded_defaults.items():
        if getattr(args, param) is None:
            setattr(args, param, config_defaults.get(param, hardcoded))

    return args


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_generation(args)


if __name__ == "__main__":
    main()
