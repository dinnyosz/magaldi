"""Benchmark client for LLM backends using LiteLLM.

This module provides a unified client for benchmarking LLMs with timing information.
Works with any LiteLLM-supported provider (Ollama, OpenAI, Anthropic, etc.).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import requests

# =============================================================================
# EVALUATION CRITERIA PER ELEMENT TYPE
# =============================================================================

EVALUATION_CRITERIA: dict[str, dict[str, str]] = {
    "file": {
        "accuracy": "Summary correctly describes what the code actually does, no factual errors",
        "purpose": "Clear about module's primary job and what capability it provides",
        "domain": "Explains what problem space this code addresses",
        "architecture": "Mentions design patterns or abstractions it uses",
        "discoverability": "Helps agent know when to look here and what questions lead to this file",
        "dependencies": "Describes external modules or systems it integrates with",
        "agent_utility": "An agent reading only this summary can decide if this file is relevant to its task",
        "searchability": "Uses specific, distinctive terms that would match relevant search queries",
        "conciseness": "Every sentence adds value, no filler or redundancy",
        "no_enumeration": "Does NOT just list classes/functions (those are separate)",
    },
    "class": {
        "accuracy": "Summary correctly describes what the class does, no factual errors",
        "representation": "Clear what the class represents, models, or encapsulates",
        "responsibility": "Explains core responsibility and problem it solves",
        "instantiation": "Describes how to create an instance and what parameters are required",
        "state": "Mentions key attributes and what makes an instance valid vs invalid",
        "collaboration": "Explains how it works with other classes/modules",
        "agent_utility": "An agent reading only this summary can decide if this class is relevant to its task",
        "searchability": "Uses specific, distinctive terms that would match relevant search queries",
        "conciseness": "Every sentence adds value, no filler or redundancy",
        "no_enumeration": "Does NOT just list methods (those are separate)",
        "no_context_repeat": "Starts directly with what it does, not with filler or repeated context",
    },
    "interface": {
        "accuracy": "Summary correctly describes what the interface defines, no factual errors",
        "contract": "Clear what the interface requires implementors to provide",
        "purpose": "Explains what capability or behavior the interface abstracts",
        "methods": "Describes the key methods and their expected semantics",
        "implementors": "Hints at what types of classes would implement this",
        "agent_utility": "An agent reading only this summary can decide if this interface is relevant to its task",
        "searchability": "Uses specific, distinctive terms that would match relevant search queries",
        "conciseness": "Every sentence adds value, no filler or redundancy",
        "no_context_repeat": "Starts directly with what it defines, not with filler or repeated context",
    },
    "trait": {
        "accuracy": "Summary correctly describes what the trait provides, no factual errors",
        "behavior": "Clear what reusable behavior or capability the trait provides",
        "requirements": "Describes any abstract methods or constraints on implementing types",
        "composition": "Explains how this trait composes with other traits or classes",
        "state": "Mentions any default implementations or state expectations",
        "agent_utility": "An agent reading only this summary can decide if this trait is relevant to its task",
        "searchability": "Uses specific, distinctive terms that would match relevant search queries",
        "conciseness": "Every sentence adds value, no filler or redundancy",
        "no_context_repeat": "Starts directly with what it provides, not with filler or repeated context",
    },
    "enum": {
        "accuracy": "Summary correctly describes what the enum represents, no factual errors",
        "meaning": "Clear what concept or domain value set this enum models",
        "variants": "Describes the key variants and what each represents",
        "usage": "Explains when and where this enum is used in the system",
        "agent_utility": "An agent reading only this summary can decide if this enum is relevant to its task",
        "searchability": "Uses specific, distinctive terms that would match relevant search queries",
        "conciseness": "Every sentence adds value, no filler or redundancy",
        "no_context_repeat": "Starts directly with what it represents, not with filler or repeated context",
    },
    "type_alias": {
        "accuracy": "Summary correctly describes the aliased type, no factual errors",
        "definition": "Clear what the underlying type structure is",
        "purpose": "Explains why this alias exists and what abstraction it provides",
        "usage": "Describes where in the system this type is used",
        "agent_utility": "An agent reading only this summary can decide if this type is relevant to its task",
        "searchability": "Uses specific, distinctive terms that would match relevant search queries",
        "conciseness": "Every sentence adds value, no filler or redundancy",
        "no_context_repeat": "Starts directly with what it defines, not with filler or repeated context",
    },
    "function": {
        "accuracy": "Summary correctly describes what the function does, no factual errors",
        "operation": "Clear what operation, transformation, or task it performs",
        "interface": "Describes parameters and return value with types",
        "usage_scenarios": "Explains in what scenario an agent should call this",
        "side_effects": "Mentions external state changes, I/O, or exceptions",
        "preconditions": "Notes required input constraints, assumptions, or error conditions",
        "agent_utility": "An agent reading only this summary can decide if this function is relevant to its task",
        "searchability": "Uses specific, distinctive terms that would match relevant search queries",
        "conciseness": "Every sentence adds value, no filler or redundancy",
        "no_context_repeat": "Starts with an action verb, not with filler or repeated context",
    },
    "method": {
        "accuracy": "Summary correctly describes what the method does, no factual errors",
        "operation": "Clear what operation this method performs on/for the object",
        "interface": "Describes parameters and return value",
        "state_interaction": "Explains which instance attributes it reads or modifies",
        "lifecycle": "Describes whether this is setup/init, cleanup, or called repeatedly",
        "errors": "Mentions exceptions it can raise and preconditions that must hold",
        "agent_utility": "An agent reading only this summary can decide if this method is relevant to its task",
        "searchability": "Uses specific, distinctive terms that would match relevant search queries",
        "conciseness": "Every sentence adds value, no filler or redundancy",
        "no_context_repeat": "Starts with an action verb, not with filler or repeated context",
    },
    "constant": {
        "accuracy": "Summary correctly describes the constant's value or purpose, no factual errors",
        "value_meaning": "Clear what this value represents or configures",
        "usage": "Explains where in the system it is used",
        "constraints": "Notes valid values, min/max bounds, or relationships with other values",
        "agent_utility": "An agent reading only this summary can decide if this constant is relevant to its task",
        "searchability": "Uses specific, distinctive terms that would match relevant search queries",
        "conciseness": "Every sentence adds value, no filler or redundancy",
        "no_context_repeat": "Starts directly with what it represents, not with filler or repeated context",
    },
    "variable": {
        "accuracy": "Summary correctly describes what the variable holds, no factual errors",
        "data": "Clear what information this variable holds",
        "lifecycle": "Explains how it is initialized and when it changes",
        "role": "Describes how this variable influences its containing scope",
        "agent_utility": "An agent reading only this summary can decide if this variable is relevant to its task",
        "searchability": "Uses specific, distinctive terms that would match relevant search queries",
        "conciseness": "Every sentence adds value, no filler or redundancy",
        "no_context_repeat": "Starts directly with what it holds, not with filler or repeated context",
    },
    "import": {
        "accuracy": "Summary correctly describes what is imported, no factual errors",
        "source": "Clear what module or package the import comes from",
        "purpose": "Explains what capability the import provides to this file",
        "agent_utility": "An agent reading only this summary can decide if this import is relevant to its task",
        "searchability": "Uses specific, distinctive terms that would match relevant search queries",
        "conciseness": "Every sentence adds value, no filler or redundancy",
    },
}

# Criterion weights for weighted average scoring.
# Content criteria (accuracy, core purpose) are weighted higher than
# formatting criteria (no_context_repeat, no_enumeration, conciseness).
CRITERIA_WEIGHTS: dict[str, float] = {
    # Core content — high weight (these define summary quality)
    "accuracy": 3.0,
    "purpose": 2.0,
    "operation": 2.0,
    "representation": 2.0,
    "responsibility": 2.0,
    "contract": 2.0,
    "behavior": 2.0,
    "meaning": 2.0,
    "definition": 2.0,
    "source": 2.0,
    "data": 2.0,
    "value_meaning": 2.0,
    # Informational content — standard weight
    "domain": 1.5,
    "interface": 1.5,
    "usage_scenarios": 1.5,
    "side_effects": 1.5,
    "state_interaction": 1.5,
    "state": 1.5,
    "instantiation": 1.5,
    "collaboration": 1.5,
    "methods": 1.5,
    "implementors": 1.5,
    "requirements": 1.5,
    "composition": 1.5,
    "variants": 1.5,
    "usage": 1.5,
    "errors": 1.5,
    # Downstream utility — medium weight (agent consumption + embedding quality)
    "agent_utility": 1.5,
    "searchability": 1.5,
    "preconditions": 1.0,
    "lifecycle": 1.0,
    "role": 1.0,
    "architecture": 1.0,
    "discoverability": 1.0,
    "dependencies": 1.0,
    "constraints": 1.0,
    # Formatting — low weight (important but shouldn't dominate)
    "conciseness": 0.5,
    "no_context_repeat": 0.5,
    "no_enumeration": 0.5,
}

# Default weight for any criterion not listed above
DEFAULT_CRITERION_WEIGHT = 1.0

# Expected sentence counts per element type
EXPECTED_SENTENCES: dict[str, tuple[int, int]] = {
    "file": (4, 6),
    "class": (4, 6),
    "interface": (3, 5),
    "trait": (3, 5),
    "enum": (2, 4),
    "type_alias": (2, 3),
    "function": (4, 6),
    "method": (4, 6),
    "constant": (2, 3),
    "variable": (2, 3),
    "import": (1, 2),
}


@dataclass
class CriteriaScores:
    """Scores for each criterion in an evaluation."""

    scores: dict[str, int] = field(default_factory=dict)  # criterion -> score (1-10)
    notes: str = ""

    @property
    def average(self) -> float:
        """Calculate weighted average score across all criteria (1-10 scale).

        Uses CRITERIA_WEIGHTS to weight content criteria higher than formatting
        criteria. Falls back to DEFAULT_CRITERION_WEIGHT for unknown criteria.
        """
        if not self.scores:
            return 0.0
        total_weight = 0.0
        weighted_sum = 0.0
        for criterion, score in self.scores.items():
            weight = CRITERIA_WEIGHTS.get(criterion, DEFAULT_CRITERION_WEIGHT)
            weighted_sum += score * weight
            total_weight += weight
        if total_weight == 0.0:
            return 0.0
        return weighted_sum / total_weight


@dataclass
class EvaluationResult:
    """Result of evaluating summaries for one element."""

    element_type: str
    element_name: str
    evaluations: dict[str, CriteriaScores] = field(default_factory=dict)  # model -> scores
    raw_response: str = ""
    parse_error: str = ""

    def get_model_score(self, model: str) -> float:
        """Get average score (1-10) for a model."""
        if model in self.evaluations:
            return self.evaluations[model].average
        return 0.0


def _make_anonymous_labels(count: int) -> list[str]:
    """Generate anonymous labels for blind evaluation: A, B, C, ..., Z, AA, AB, ..."""
    labels = []
    for i in range(count):
        if i < 26:
            labels.append(chr(65 + i))  # A-Z
        else:
            labels.append(chr(65 + i // 26 - 1) + chr(65 + i % 26))  # AA, AB, ...
    return labels


def build_evaluation_prompt(
    element_type: str,
    element_name: str,
    source_code: str,
    summaries: dict[str, str],  # model -> summary
) -> tuple[str, dict[str, str]]:
    """Build evaluation prompt with anonymous labels for blind evaluation.

    Model names are replaced with anonymous labels (A, B, C, ...) to prevent
    the eval model from biasing scores based on model identity.

    Args:
        element_type: Type of element (file, class, function, method, constant, variable).
        element_name: Name of the element.
        source_code: The source code being summarized.
        summaries: Dict mapping model name to its summary.

    Returns:
        Tuple of (prompt string, label_to_model mapping).
        The mapping maps anonymous labels back to real model names.
    """
    criteria = EVALUATION_CRITERIA.get(element_type, EVALUATION_CRITERIA["function"])
    min_sent, max_sent = EXPECTED_SENTENCES.get(element_type, (4, 6))

    # Truncate source code if too long
    if len(source_code) > 2000:
        source_code = source_code[:2000] + "\n... (truncated)"

    # Build criteria description
    criteria_lines = []
    for key, desc in criteria.items():
        criteria_lines.append(f'  - "{key}": {desc}')
    criteria_text = "\n".join(criteria_lines)

    # Assign anonymous labels to models (blind evaluation)
    model_list = list(summaries.keys())
    labels = _make_anonymous_labels(len(model_list))
    label_to_model: dict[str, str] = dict(zip(labels, model_list))

    # Build summaries section with anonymous labels
    summaries_section = []
    for label, model in zip(labels, model_list):
        summary = summaries[model]
        summaries_section.append(f'### Summary {label}\n{summary if summary else "(generation failed)"}')
    summaries_text = "\n\n".join(summaries_section)

    prompt = f"""You are evaluating code summaries for a {element_type} named "{element_name}".

## Source Code
```
{source_code}
```

## Summaries to Evaluate

{summaries_text}

## Evaluation Criteria for {element_type} (expected {min_sent}-{max_sent} sentences)

Score each criterion 1-10:
  1-2 = Missing or completely wrong
  3-4 = Barely addressed, major gaps
  5-6 = Partially addressed, some gaps
  7-8 = Well addressed, minor issues
  9-10 = Excellently addressed, comprehensive

Criteria:
{criteria_text}

## Required Output

Output ONLY valid JSON (no markdown, no explanation) in this exact format:
{{
  "evaluations": {{
    "{labels[0]}": {{
{chr(10).join(f'      "{k}": <score 1-10>,' for k in list(criteria.keys())[:-1])}
      "{list(criteria.keys())[-1]}": <score 1-10>,
      "notes": "<brief note on strengths/weaknesses>"
    }}{(',' + chr(10) + '    "' + '": {...},'.join(labels[1:-1]) + '"' + ': {...},' if len(labels) > 2 else '') if len(labels) > 1 else ''}
    {f'''"{labels[-1]}": {{
{chr(10).join(f'      "{k}": <score 1-10>,' for k in list(criteria.keys())[:-1])}
      "{list(criteria.keys())[-1]}": <score 1-10>,
      "notes": "<brief note>"
    }}''' if len(labels) > 1 else ''}
  }}
}}

Evaluate all {len(labels)} summaries: {", ".join(labels)}

JSON output:"""

    return prompt, label_to_model


def _fix_single_quotes(s: str) -> str:
    """Replace single-quoted JSON strings with double-quoted strings.

    Thinking models sometimes output Python-style dicts with single quotes
    instead of valid JSON double quotes. This does a character-by-character
    replacement that handles escaped quotes and apostrophes in text.
    """
    result = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "'":
            # Replace opening single quote with double quote
            result.append('"')
            i += 1
            # Read until closing single quote (handle escapes)
            while i < len(s):
                if s[i] == "\\" and i + 1 < len(s):
                    # Escaped character — keep as-is but convert \' to '
                    if s[i + 1] == "'":
                        result.append("'")
                    else:
                        result.append(s[i])
                        result.append(s[i + 1])
                    i += 2
                elif s[i] == "'":
                    result.append('"')
                    i += 1
                    break
                elif s[i] == '"':
                    # Escape double quotes inside single-quoted strings
                    result.append('\\"')
                    i += 1
                else:
                    result.append(s[i])
                    i += 1
        elif ch == '"':
            # Already a double-quoted string — pass through verbatim
            result.append(ch)
            i += 1
            while i < len(s):
                if s[i] == "\\" and i + 1 < len(s):
                    result.append(s[i])
                    result.append(s[i + 1])
                    i += 2
                elif s[i] == '"':
                    result.append(s[i])
                    i += 1
                    break
                else:
                    result.append(s[i])
                    i += 1
        else:
            result.append(ch)
            i += 1
    return "".join(result)


def _parse_json_lenient(json_str: str) -> dict | None:
    """Parse JSON with lenient handling of common LLM output issues.

    Handles:
    - Trailing commas before } or ]
    - Single-quoted strings (Python-style dicts from thinking models)

    Returns parsed dict or None if all attempts fail.
    """
    # Attempt 1: strict JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Attempt 2: fix trailing commas
    fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Attempt 3: fix single-quoted strings
    fixed2 = _fix_single_quotes(json_str)
    try:
        return json.loads(fixed2)
    except json.JSONDecodeError:
        pass

    # Attempt 4: both fixes combined
    fixed3 = re.sub(r',(\s*[}\]])', r'\1', fixed2)
    try:
        return json.loads(fixed3)
    except json.JSONDecodeError:
        return None


def parse_evaluation_response(
    response: str,
    element_type: str,
    labels: list[str],
    label_to_model: dict[str, str],
) -> tuple[dict[str, CriteriaScores], str]:
    """Parse JSON evaluation response, mapping anonymous labels back to model names.

    Args:
        response: Raw LLM response.
        element_type: Element type for expected criteria.
        labels: List of anonymous labels (A, B, C, ...) used in the prompt.
        label_to_model: Mapping from label to real model name.

    Returns:
        Tuple of (evaluations dict, error message).
        evaluations maps real model name to CriteriaScores.
    """
    criteria = EVALUATION_CRITERIA.get(element_type, EVALUATION_CRITERIA["function"])
    evaluations: dict[str, CriteriaScores] = {}

    # Clean response - remove thinking tags
    cleaned = response
    thinking_patterns = [
        r"<think>.*?</think>\s*",
        r"<thinking>.*?</thinking>\s*",
        r"<reasoning>.*?</reasoning>\s*",
        r"<reflection>.*?</reflection>\s*",
    ]
    for pattern in thinking_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Handle unclosed tags
    unclosed_patterns = [
        r"<think>.*$",
        r"<thinking>.*$",
        r"<reasoning>.*$",
        r"<reflection>.*$",
    ]
    for pattern in unclosed_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Try to extract JSON from response
    cleaned = cleaned.strip()

    # Remove markdown code blocks if present
    if cleaned.startswith("```"):
        # Find the end of the code block
        lines = cleaned.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```"):
                in_block = not in_block
                continue
            if in_block or not line.startswith("```"):
                json_lines.append(line)
        cleaned = "\n".join(json_lines)

    # Find JSON object in response
    json_match = re.search(r'\{[\s\S]*\}', cleaned)
    if not json_match:
        return evaluations, "No JSON object found in response"

    json_str = json_match.group(0)

    data = _parse_json_lenient(json_str)
    if data is None:
        return evaluations, "JSON parse error in evaluation response"

    # Extract evaluations
    evals_data = data.get("evaluations", data)  # Handle both wrapped and unwrapped

    # Parse by anonymous label and map back to real model names
    for label in labels:
        if label not in evals_data:
            continue
        model_data = evals_data[label]
        scores = {}
        for criterion in criteria:
            if criterion in model_data:
                try:
                    score = int(model_data[criterion])
                    if 1 <= score <= 10:
                        scores[criterion] = score
                except (ValueError, TypeError):
                    pass
        notes = model_data.get("notes", "")
        real_model = label_to_model[label]
        if isinstance(notes, str):
            evaluations[real_model] = CriteriaScores(scores=scores, notes=notes)
        else:
            evaluations[real_model] = CriteriaScores(scores=scores, notes="")

    # Check for missing models (report by real model name)
    all_models = list(label_to_model.values())
    missing = [m for m in all_models if m not in evaluations or not evaluations[m].scores]
    if missing:
        return evaluations, f"Missing evaluations for: {', '.join(missing)}"

    return evaluations, ""


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    model: str
    success: bool
    response: str = ""
    error: str = ""

    # Timing in seconds
    total_time: float = 0.0  # Wall clock time (measured by client)

    # Token counts
    prompt_tokens: int = 0  # Input tokens
    output_tokens: int = 0  # Generated tokens
    prompt_chars: int = 0  # Input character count
    output_chars: int = 0  # Output character count

    @property
    def tokens_per_second(self) -> float:
        """Calculate generation speed (output tokens / wall time)."""
        if self.total_time > 0:
            return self.output_tokens / self.total_time
        return 0.0


class BenchmarkClient:
    """Unified benchmark client using LiteLLM.

    Works with any LiteLLM-supported provider (Ollama, OpenAI, Anthropic, etc.).
    Provides unified metrics:
    - Wall time (measured by client)
    - Token counts (prompt/output)
    - Character counts (prompt/output)
    - Tokens per second (output_tokens / wall_time)

    Model format: "provider/model" (e.g., "ollama/qwen2.5-coder:3b", "openai/gpt-4o")
    """

    # Models that use thinking/reasoning tags by default.
    # All Qwen3 variants (qwen3, qwen3.5, etc.) think by default in Ollama.
    THINKING_MODELS = ("qwen3", "deepseek-r1", "deepseek-coder-v2", "nemotron", "lfm2.5-thinking", "sam860/lfm2.5")

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize the client.

        Args:
            api_base: API base URL (required for Ollama, optional for cloud).
            api_key: API key (required for cloud providers).
        """
        self.api_base = api_base
        self.api_key = api_key

        # Import litellm lazily to avoid import overhead if not used
        import litellm
        self._litellm = litellm

        # Disable telemetry
        litellm.telemetry = False
        # Drop unsupported parameters (e.g., presence_penalty for Ollama)
        litellm.drop_params = True

    def check_connection(self) -> bool:
        """Check if the API is reachable."""
        if self.api_base and "11434" in self.api_base:
            # Ollama - check via API
            try:
                resp = requests.get(f"{self.api_base}/api/tags", timeout=5)
                return resp.status_code == 200
            except Exception:
                return False
        # For cloud providers, assume connected (will fail on first request if not)
        return True

    def list_models(self) -> list[str]:
        """Get list of available models.

        Note: LiteLLM doesn't have a unified model listing API.
        For Ollama, we fall back to the Ollama API directly.
        """
        if self.api_base and "11434" in self.api_base:
            # Ollama - use direct API
            try:
                resp = requests.get(f"{self.api_base}/api/tags", timeout=10)
                if resp.status_code == 200:
                    return [m["name"] for m in resp.json().get("models", [])]
            except Exception:
                pass
        return []

    def warmup(
        self,
        model: str,
        timeout: int = 120,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> tuple[bool, float, str]:
        """Warm up a model by sending a tiny request.

        Args:
            model: Model name in LiteLLM format (e.g., "ollama/qwen2.5-coder:3b").
            timeout: Timeout in seconds.
            api_base: Override the client's api_base for this call.
            api_key: Override the client's api_key for this call.

        Returns:
            Tuple of (success, warmup_time_seconds, error_message)
        """
        start = time.perf_counter()
        try:
            result = self.generate(
                model=model,
                prompt="Hi",
                max_tokens=5,
                timeout=timeout,
                api_base=api_base,
                api_key=api_key,
            )
            elapsed = time.perf_counter() - start
            if result.success:
                return True, elapsed, ""
            return False, elapsed, result.error
        except Exception as e:
            elapsed = time.perf_counter() - start
            return False, elapsed, str(e)

    # Common stop sequences to prevent models from continuing past intended response
    STOP_SEQUENCES = [
        "<|im_start|>",      # ChatML format
        "<|im_end|>",        # ChatML end
        "<|assistant|>",     # Phi format
        "<|user|>",          # Phi format
        "<|eot_id|>",        # Llama 3 format
        "<|start_header_id|>",  # Llama 3 format
        "[INST]",            # Mistral format
        "### Human:",        # Alpaca format
        "### User:",         # Alpaca format
    ]

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.2,
        top_p: float | None = 0.95,
        top_k: int | None = None,
        min_p: float | None = None,
        repetition_penalty: float | None = None,
        presence_penalty: float | None = None,
        max_tokens: int = 512,
        timeout: int = 120,
        api_base: str | None = None,
        api_key: str | None = None,
        stop: list[str] | None = None,
    ) -> BenchmarkResult:
        """Generate completion and return timing metrics.

        Args:
            model: Model name in LiteLLM format (e.g., "ollama/qwen2.5-coder:3b").
            prompt: The prompt to send.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter (passed via extra params).
            min_p: Min-p sampling parameter (passed via extra params).
            repetition_penalty: Repetition penalty (maps to frequency_penalty).
            presence_penalty: Presence penalty.
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
            api_base: Override the client's api_base for this call.
            api_key: Override the client's api_key for this call.
            stop: Custom stop sequences. If None, uses STOP_SEQUENCES.

        Returns:
            BenchmarkResult with timing and token information.
        """
        start = time.perf_counter()

        # Build kwargs for litellm
        kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "stop": stop if stop is not None else self.STOP_SEQUENCES,
        }

        # Add optional parameters
        if top_p is not None:
            kwargs["top_p"] = top_p
        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty
        if repetition_penalty is not None:
            # Map to frequency_penalty (OpenAI-style)
            kwargs["frequency_penalty"] = min(2.0, (repetition_penalty - 1.0) * 2)

        # Pass provider-specific params via extra_body (Ollama native API)
        extra_body: dict = {}
        if top_k is not None:
            extra_body["top_k"] = top_k
        if min_p is not None:
            extra_body["min_p"] = min_p

        # Add api_base (call param overrides instance)
        effective_api_base = api_base or self.api_base
        if effective_api_base:
            kwargs["api_base"] = effective_api_base

        # Add api_key (call param overrides instance)
        effective_api_key = api_key or self.api_key
        if effective_api_key:
            kwargs["api_key"] = effective_api_key

        # Disable thinking/reasoning for Ollama models.
        # Uses ollama_chat/ prefix which routes to native /api/chat supporting "think".
        # Always pass think=False for benchmarks — we want pure summarization output.
        if model.startswith("ollama_chat/"):
            kwargs["think"] = False

        if extra_body:
            kwargs["extra_body"] = extra_body

        try:
            response = self._litellm.completion(**kwargs)
            total_time = time.perf_counter() - start

            # Extract response text
            content = response.choices[0].message.content
            if content is None:
                content = ""
            response_text = content.strip()

            # Extract token counts from usage
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0

            return BenchmarkResult(
                model=model,
                success=True,
                response=response_text,
                total_time=total_time,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                prompt_chars=len(prompt),
                output_chars=len(response_text),
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
        """Close the client (no-op for LiteLLM)."""
        pass
