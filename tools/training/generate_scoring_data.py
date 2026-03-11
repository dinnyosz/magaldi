#!/usr/bin/env python3
"""Generate training data for the variable scorer model.

Parses repos directly (Phases 1-3) to extract variables, scores them
individually with a teacher model via Ollama, quality-filters, and
outputs ChatML JSONL for fine-tuning.

Usage:
    python tools/training/generate_scoring_data.py \
      --repos /path/to/repo1 /path/to/repo2 \
      --teacher-model qwen3-coder:30b \
      --output-dir tools/training/data/variable_scorer \
      --validation-split 0.1 \
      --resume
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
    user_prompt = build_user_prompt(prompt_vars, token_budget=1200)

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
    user_prompt = build_user_prompt(prompt_vars, token_budget=1200)

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
    user_prompt = build_user_prompt(prompt_vars, token_budget=1200)

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


def build_training_batches(
    results: list[RawScoringResult],
    seed: int = 42,
) -> list[dict]:
    """Build ChatML training examples from scored results.

    Mixes batch sizes to teach the model flexibility:
    - 10% single variables (1-3)
    - 40% medium batches (4-10)
    - 40% large batches (11-20)
    - 10% max batches (21-30)
    """
    rng = random.Random(seed)
    results = list(results)  # copy to avoid mutating caller's list
    rng.shuffle(results)

    # Define batch size distribution
    batch_sizes = (
        [(1, 3)] * 10 +   # 10% small
        [(4, 10)] * 40 +   # 40% medium
        [(11, 20)] * 40 +  # 40% large
        [(21, 30)] * 10    # 10% max
    )

    examples = []
    idx = 0
    batch_num = 0

    while idx < len(results):
        # Pick batch size range from distribution
        min_size, max_size = batch_sizes[batch_num % len(batch_sizes)]
        batch_size = rng.randint(min_size, max_size)
        batch_size = min(batch_size, len(results) - idx)

        batch_results = results[idx : idx + batch_size]
        variables = [
            {
                "file_path": r.file_path,
                "name": r.name,
                "raw_code": r.raw_code,
            }
            for r in batch_results
        ]
        scores_list = [r.scores for r in batch_results]

        example = format_training_example(variables, scores_list)
        examples.append(example)

        idx += batch_size
        batch_num += 1

    return examples


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
        f"{rate:.1f}/s"
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


def run_generation(args: argparse.Namespace) -> None:
    """Main data generation pipeline."""
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Parse all repos and collect variables
    all_variables: list[dict] = []
    for repo_path in args.repos:
        try:
            variables = extract_variables_from_repo(repo_path)
            all_variables.extend(variables)
        except Exception:
            logger.error("Failed to parse repo: %s", repo_path, exc_info=True)

    logger.info("Total variables extracted: %d from %d repos", len(all_variables), len(args.repos))

    if not all_variables:
        logger.error("No variables extracted. Check repo paths and magaldi.yaml configs.")
        sys.exit(1)

    # Step 2: Load existing results for resume
    existing: dict[str, RawScoringResult] = {}
    if args.resume:
        existing = load_existing_results(raw_dir)
        logger.info("Loaded %d existing results for resume", len(existing))

    # Step 3: Score variables with teacher model
    to_score = [v for v in all_variables if v["element_id"] not in existing]
    logger.info(
        "Variables to score: %d (skipping %d already scored)",
        len(to_score),
        len(all_variables) - len(to_score),
    )

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

    # Step 6: Build ChatML examples with varied batch sizes
    train_examples = build_training_batches(train_results, seed=args.seed)
    val_examples = build_training_batches(val_results, seed=args.seed + 1)

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

    lang_counts: dict[str, int] = {}
    for r in all_results:
        lang_counts[r.language] = lang_counts.get(r.language, 0) + 1

    print(f"\n{'='*80}")
    print("Output")
    print(f"{'='*80}")
    print(f"  Train:      {len(train_examples)} examples -> {train_path}")
    print(f"  Validation: {len(val_examples)} examples -> {val_path}")
    print(f"  Scores:     {score_dist['drop']} clear drops, "
          f"{score_dist['borderline']} borderline, "
          f"{score_dist['keep']} clear keeps")
    print(f"  Languages:  {lang_counts}")
    print(f"{'='*80}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate training data for the variable scorer model."
    )

    parser.add_argument(
        "--repos",
        nargs="+",
        required=True,
        help="Paths to repos to parse (each must have magaldi.yaml)",
    )
    parser.add_argument(
        "--teacher-model",
        default="qwen3-coder:30b",
        help="Ollama teacher model name (default: qwen3-coder:30b)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--output-dir",
        default="tools/training/data/variable_scorer",
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of data to hold out for validation (default: 0.1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Teacher model temperature (default: 0.2)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Teacher model timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previously scored results in raw/",
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

    return parser.parse_args()


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
