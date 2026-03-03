"""RALPH Prompt Improver — Reinforcement And Learning from Prompt History.

Iteratively optimizes code summarization prompts through a generate → evaluate → improve
feedback loop. Given an indexed element, the system:

1. Builds a baseline prompt from production templates
2. Generates a summary with the configured LLM
3. Evaluates the summary using LLM-as-judge (EVALUATION_CRITERIA)
4. Uses evaluation feedback to improve the prompt via a meta-prompting step
5. Repeats until score converges or max iterations reached

Usage:
    from shared.ai.prompt_improver import PromptImprover
    from shared.config import load_config

    config = load_config()
    improver = PromptImprover(config)
    result = improver.run(element_id="abc123def")
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from shared.ai.ollama_benchmark import (
    EVALUATION_CRITERIA,
    BenchmarkClient,
    CriteriaScores,
    build_evaluation_prompt,
    parse_evaluation_response,
)
from shared.ai.prompts import (
    SYSTEM_PROMPTS,
    USER_PROMPTS,
    build_messages,
    clean_summary,
    format_sentence_range,
)

if TYPE_CHECKING:
    from magaldi_core.parsers.base import CodeElement
    from shared.config import MagaldiConfig, ModelConfig

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class PromptVersion:
    """A versioned prompt (system + user template pair)."""

    system_prompt: str
    user_template: str
    iteration: int
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for YAML/JSON export."""
        return {
            "iteration": self.iteration,
            "system_prompt": self.system_prompt,
            "user_template": self.user_template,
            "reasoning": self.reasoning,
        }


@dataclass
class IterationResult:
    """Result of a single RALPH iteration."""

    iteration: int
    prompt: PromptVersion
    summary: str
    scores: CriteriaScores
    summary_time: float = 0.0
    eval_time: float = 0.0

    @property
    def avg_score(self) -> float:
        """Average score across all criteria."""
        return self.scores.average


@dataclass
class RALPHResult:
    """Complete result of a RALPH prompt optimization run."""

    element_type: str
    element_name: str
    element_id: str
    iterations: list[IterationResult] = field(default_factory=list)
    converged: bool = False
    stop_reason: str = ""

    @property
    def best_iteration(self) -> int:
        """Index of the highest-scoring iteration."""
        if not self.iterations:
            return 0
        return max(range(len(self.iterations)), key=lambda i: self.iterations[i].avg_score)

    @property
    def best_result(self) -> IterationResult | None:
        """The highest-scoring iteration result."""
        if not self.iterations:
            return None
        return self.iterations[self.best_iteration]

    @property
    def best_score(self) -> float:
        """Score of the best iteration."""
        best = self.best_result
        return best.avg_score if best else 0.0


# =============================================================================
# PROMPT IMPROVEMENT META-PROMPT
# =============================================================================

IMPROVEMENT_PROMPT = """You are an expert prompt engineer optimizing code summarization prompts.

## Task
Improve the summarization prompt below so that the summary scores higher on ALL evaluation criteria.

## Source Code Being Summarized
```
{source_code}
```

## Current Prompt

### System Prompt:
{current_system_prompt}

### User Prompt (with placeholders):
{current_user_template}

## Generated Summary (from current prompt):
{summary}

## Evaluation Scores (1-10 scale):
{scores_text}

## Evaluation Notes:
{eval_notes}

## Criteria Descriptions:
{criteria_text}

## Instructions

Analyze which criteria scored low and why. Then produce an improved version of BOTH the system prompt and user template.

Rules:
1. The user template MUST preserve these exact placeholders: {required_placeholders}
2. Keep the system prompt concise — under 500 words
3. Focus improvements on low-scoring criteria (< 7)
4. Maintain the anti-verbose rules: never start with "This function...", "This class...", etc.
5. Keep the sentence range instruction ({sentence_range} sentences)
6. Do NOT add examples or few-shot demonstrations (they waste context)

Output ONLY valid JSON (no markdown, no explanation):
{{
  "reasoning": "<1-2 sentences explaining what you changed and why>",
  "system_prompt": "<the improved system prompt>",
  "user_template": "<the improved user template with {{placeholders}} preserved>"
}}

JSON output:"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _reconstruct_element(doc: dict[str, Any]) -> CodeElement:
    """Reconstruct a CodeElement from an OpenSearch document.

    Mirrors the pattern in EmbeddingStore.get_element().
    """
    from magaldi_core.parsers.base import Call, CodeElement, Import

    # Reconstruct imports list
    imports = []
    for imp_data in doc.get("imports", []):
        if isinstance(imp_data, dict):
            imports.append(Import(
                name=imp_data.get("name", ""),
                module=imp_data.get("module", ""),
                alias=imp_data.get("alias"),
                line=imp_data.get("line", 0),
            ))

    # Reconstruct calls list
    calls = []
    for call_data in doc.get("calls", []):
        if isinstance(call_data, dict):
            calls.append(Call(
                name=call_data.get("name", ""),
                receiver=call_data.get("receiver"),
                line=call_data.get("line", 0),
                resolved_id=call_data.get("resolved_id"),
                category=call_data.get("category", "unknown"),
            ))

    return CodeElement(
        element_id=doc.get("element_id", ""),
        scope=doc.get("scope", ""),
        repository=doc.get("repository", ""),
        username=doc.get("username", ""),
        relative_path=doc.get("relative_path", ""),
        element_type=doc.get("element_type", ""),
        name=doc.get("name", ""),
        language=doc.get("language", ""),
        line_start=doc.get("line_start", 0),
        line_end=doc.get("line_end"),
        raw_code=doc.get("raw_code", ""),
        signature=doc.get("signature"),
        docstring=doc.get("docstring"),
        level=doc.get("level", 0),
        parent_id=doc.get("parent_id"),
        decorators=doc.get("decorators") or [],
        visibility=doc.get("visibility", "public"),
        is_async=doc.get("is_async", False),
        is_test=doc.get("is_test", False),
        imports=imports,
        calls=calls,
        class_attributes=doc.get("class_attributes"),
        base_classes=doc.get("base_classes"),
        exceptions_raised=doc.get("exceptions_raised"),
        attributes_modified=doc.get("attributes_modified"),
        context_usages=doc.get("context_usages") or [],
    )


def _get_parent_summaries(
    doc: dict[str, Any],
    repo: Any,
) -> dict[str, str]:
    """Fetch parent summaries (file, class) for context.

    Args:
        doc: Element document from OpenSearch.
        repo: Repository instance.

    Returns:
        Dict with 'file' and/or 'class' summary strings.
    """
    summaries: dict[str, str] = {}
    element_type = doc.get("element_type", "")

    if element_type == "file":
        return summaries

    # Get file summary
    scope = doc.get("scope", "")
    repository = doc.get("repository", "")
    username = doc.get("username", "")
    relative_path = doc.get("relative_path", "")
    filename = relative_path.split("/")[-1] if relative_path else ""

    file_id = f"{scope}:{repository}:{username}:{relative_path}:file:{filename}:1"
    file_doc = repo.get_document(file_id)
    if file_doc and file_doc.get("summary"):
        summaries["file"] = file_doc["summary"]

    # Get class summary for methods
    parent_id = doc.get("parent_id")
    if parent_id and element_type in ("method", "variable", "constant"):
        parent_doc = repo.get_document(parent_id)
        if parent_doc and parent_doc.get("element_type") == "class" and parent_doc.get("summary"):
            summaries["class"] = parent_doc["summary"]

    # Get function summary for nested variables/constants
    if parent_id and element_type in ("variable", "constant"):
        parent_doc = repo.get_document(parent_id)
        if parent_doc and parent_doc.get("element_type") in ("function", "method") and parent_doc.get("summary"):
            summaries["function"] = parent_doc["summary"]

    return summaries


def _get_template_key(element_type: str) -> str:
    """Map element type to template key."""
    if element_type == "method":
        return "method"
    if element_type in SYSTEM_PROMPTS:
        return element_type
    return "function"


def _get_required_placeholders(template_key: str) -> list[str]:
    """Extract placeholder names from a user template."""
    template = USER_PROMPTS.get(template_key, "")
    # Find all {placeholder} patterns (not {{ escaped braces }})
    return re.findall(r"(?<!\{)\{(\w+)\}(?!\})", template)


def _parse_improvement_response(response: str) -> dict[str, str] | None:
    """Parse JSON improvement response from the improver LLM.

    Returns dict with 'reasoning', 'system_prompt', 'user_template' or None on failure.
    """
    # Strip thinking tags
    cleaned = response
    for pattern in [
        r"<think>.*?</think>\s*",
        r"<thinking>.*?</thinking>\s*",
        r"<reasoning>.*?</reasoning>\s*",
        r"<reflection>.*?</reflection>\s*",
    ]:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Handle unclosed tags
    for pattern in [
        r"<think>.*$",
        r"<thinking>.*$",
        r"<reasoning>.*$",
        r"<reflection>.*$",
    ]:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    cleaned = cleaned.strip()

    # Remove markdown code blocks
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```"):
                in_block = not in_block
                continue
            json_lines.append(line)
        cleaned = "\n".join(json_lines)

    # Find JSON object
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if not json_match:
        return None

    json_str = json_match.group(0)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Try fixing trailing commas
        fixed = re.sub(r",(\s*[}\]])", r"\1", json_str)
        try:
            data = json.loads(fixed)
        except json.JSONDecodeError:
            return None

    # Validate required keys
    if "system_prompt" not in data or "user_template" not in data:
        return None

    return {
        "reasoning": data.get("reasoning", ""),
        "system_prompt": data["system_prompt"],
        "user_template": data["user_template"],
    }


def _format_user_template(
    user_template: str,
    element: CodeElement,
    parent_summaries: dict[str, str],
) -> str:
    """Format a user template with element data.

    This handles the same variables that build_messages() would populate.
    """
    from shared.ai.prompts import (
        build_attributes_section,
        build_base_classes_section,
        build_collaborators_section,
        build_document_sections_section,
        build_exceptions_section,
        build_imports_section,
        build_instantiation_hints,
        build_state_section,
        build_usage_examples,
        truncate_code,
    )

    element_type = element.element_type

    file_summary = parent_summaries.get("file", "No file context available.")
    class_summary = parent_summaries.get("class", "")
    function_summary = parent_summaries.get("function", "")

    class_context = ""
    if class_summary and element_type in ("method", "function", "variable", "constant"):
        class_context = f"Class context: {class_summary}"

    function_context = ""
    if function_summary and element_type in ("variable", "constant"):
        function_context = f"Function/method context: {function_summary}"

    docstring_section = ""
    if element.docstring:
        docstring_section = f"Docstring: {element.docstring}"

    decorators = ""
    if element.decorators:
        decorators = f"Decorators: {', '.join(element.decorators)}"

    usages_section = ""
    if element.context_usages:
        usages_section = "\nUsed in:\n" + "\n".join(f"- {u}" for u in element.context_usages)

    imports_section = build_imports_section(element) if element_type == "file" else ""
    document_sections_section = build_document_sections_section(element) if element_type == "file" else ""
    attributes_section = build_attributes_section(element) if element_type == "class" else ""
    base_classes_section = build_base_classes_section(element) if element_type in ("class", "interface", "trait") else ""
    collaborators_section = build_collaborators_section(element) if element_type == "class" else ""
    instantiation_hints = build_instantiation_hints(element) if element_type == "class" else ""
    exceptions_section = build_exceptions_section(element) if element_type in ("function", "method") else ""
    state_section = build_state_section(element) if element_type == "method" else ""
    usage_examples = build_usage_examples(element) if element_type in ("function", "constant") else ""

    code = truncate_code(element.raw_code or "", 4000)

    try:
        formatted = user_template.format(
            language=element.language or "code",
            file_path=element.relative_path,
            file_summary=file_summary,
            class_summary=class_summary,
            class_context=class_context,
            function_context=function_context,
            class_name=element.name if element_type == "class" else "",
            function_name=element.name,
            method_name=element.name,
            name=element.name,
            signature=element.signature or "",
            docstring_section=docstring_section,
            decorators=decorators,
            code=code,
            usages_section=usages_section,
            imports_section=imports_section,
            document_sections_section=document_sections_section,
            attributes_section=attributes_section,
            base_classes_section=base_classes_section,
            collaborators_section=collaborators_section,
            instantiation_hints=instantiation_hints,
            exceptions_section=exceptions_section,
            state_section=state_section,
            usage_examples=usage_examples,
        )
    except (KeyError, IndexError) as e:
        logger.warning("Failed to format improved user template: %s. Falling back to original.", e)
        return ""

    # Clean up extra blank lines from empty optional sections
    formatted = "\n".join(line for line in formatted.split("\n") if line.strip() or line == "")
    return formatted.strip()


# =============================================================================
# PROMPT IMPROVER ENGINE
# =============================================================================


class PromptImprover:
    """RALPH engine for iterative prompt optimization.

    Args:
        config: Magaldi configuration.
        summary_model: Override summary model config.
        eval_model: Override evaluation model config.
        improver_model: Override prompt improver model config. Defaults to eval_model.
    """

    def __init__(
        self,
        config: MagaldiConfig,
        summary_model: ModelConfig | None = None,
        eval_model: ModelConfig | None = None,
        improver_model: ModelConfig | None = None,
    ) -> None:
        self.config = config

        # Resolve models
        self._summary_mc = summary_model or config.llm.get_summarize_model()
        self._eval_mc = eval_model or self._resolve_eval_model(config)
        self._improver_mc = improver_model or self._eval_mc

    @staticmethod
    def _resolve_eval_model(config: MagaldiConfig) -> ModelConfig:
        """Resolve the best available eval model from config."""
        # Try benchmark eval models first
        if hasattr(config, "benchmark") and config.benchmark.eval_models:
            for eval_ref in config.benchmark.eval_models:
                try:
                    return config.benchmark.get_model(eval_ref)
                except KeyError:
                    continue

        # Fall back to the main summarize model
        return config.llm.get_summarize_model()

    def run(
        self,
        element_id: str,
        max_iterations: int = 5,
        target_score: float = 8.0,
        on_iteration: Any | None = None,
    ) -> RALPHResult:
        """Run the RALPH prompt optimization loop.

        Args:
            element_id: Element ID or hash to fetch from OpenSearch.
            max_iterations: Maximum number of iterations (including baseline).
            target_score: Stop when average score >= this value.
            on_iteration: Optional callback(iteration_result) for progress updates.

        Returns:
            RALPHResult with all iterations and best prompt.
        """
        from shared.db.store import Repository

        repo = Repository(self.config)

        try:
            # Step 1: Fetch element
            doc = repo.get_document_by_id_or_hash(element_id)
            if not doc:
                raise ValueError(f"Element not found: {element_id}")

            element = _reconstruct_element(doc)
            parent_summaries = _get_parent_summaries(doc, repo)
            template_key = _get_template_key(element.element_type)

            result = RALPHResult(
                element_type=element.element_type,
                element_name=element.name,
                element_id=doc.get("element_id", element_id),
            )

            # Step 2: Build baseline prompt (iteration 0)
            baseline_messages = build_messages(element, parent_summaries)
            current_prompt = PromptVersion(
                system_prompt=baseline_messages[0]["content"],
                user_template=USER_PROMPTS[template_key],
                iteration=0,
                reasoning="Production baseline prompt",
            )

            # Step 3: RALPH loop
            for i in range(max_iterations):
                # 3a: Generate summary
                iter_result = self._run_iteration(
                    iteration=i,
                    prompt=current_prompt,
                    element=element,
                    parent_summaries=parent_summaries,
                    source_code=element.raw_code or "",
                )
                result.iterations.append(iter_result)

                if on_iteration:
                    on_iteration(iter_result)

                # 3b: Check convergence
                should_stop, reason = self._should_stop(
                    result.iterations, target_score, max_iterations,
                )
                if should_stop:
                    result.converged = iter_result.avg_score >= target_score
                    result.stop_reason = reason
                    break

                # 3c: Improve prompt for next iteration
                improved = self._improve_prompt(
                    element=element,
                    source_code=element.raw_code or "",
                    current_prompt=current_prompt,
                    summary=iter_result.summary,
                    scores=iter_result.scores,
                    template_key=template_key,
                )

                if improved:
                    current_prompt = PromptVersion(
                        system_prompt=improved["system_prompt"],
                        user_template=improved["user_template"],
                        iteration=i + 1,
                        reasoning=improved.get("reasoning", ""),
                    )
                else:
                    # Improvement failed — keep current prompt and note it
                    logger.warning("Prompt improvement failed at iteration %d", i)
                    result.stop_reason = f"Improvement failed at iteration {i}"
                    break

            return result
        finally:
            repo.close()

    def _run_iteration(
        self,
        iteration: int,
        prompt: PromptVersion,
        element: CodeElement,
        parent_summaries: dict[str, str],
        source_code: str,
    ) -> IterationResult:
        """Run one iteration: generate summary + evaluate it.

        Args:
            iteration: Current iteration number.
            prompt: The prompt version to use.
            element: CodeElement being summarized.
            parent_summaries: Parent context summaries.
            source_code: Raw source code of the element.

        Returns:
            IterationResult with summary and scores.
        """
        # Generate summary
        summary, summary_time = self._generate_summary(prompt, element, parent_summaries)

        if not summary.strip():
            # Generation failed — score 0 across all criteria instead of
            # sending an empty string to the evaluator (which would score
            # formatting criteria high for an empty response).
            logger.warning("Summary generation failed at iteration %d", iteration)
            element_type = element.element_type
            criteria = EVALUATION_CRITERIA.get(element_type, {})
            zero_scores = dict.fromkeys(criteria, 0)
            scores = CriteriaScores(
                scores=zero_scores,
                notes="Summary generation failed — all scores set to 0.",
            )
            return IterationResult(
                iteration=iteration,
                prompt=prompt,
                summary="(generation failed)",
                scores=scores,
                summary_time=summary_time,
                eval_time=0.0,
            )

        # Evaluate summary
        scores, eval_time = self._evaluate_summary(
            element_type=element.element_type,
            element_name=element.name,
            source_code=source_code,
            summary=summary,
        )

        return IterationResult(
            iteration=iteration,
            prompt=prompt,
            summary=summary,
            scores=scores,
            summary_time=summary_time,
            eval_time=eval_time,
        )

    def _generate_summary(
        self,
        prompt: PromptVersion,
        element: CodeElement,
        parent_summaries: dict[str, str],
    ) -> tuple[str, float]:
        """Generate a summary using the given prompt.

        Returns (cleaned_summary, elapsed_seconds).
        """
        # Format user content from template
        user_content = _format_user_template(
            prompt.user_template, element, parent_summaries,
        )

        if not user_content:
            # Fallback: use build_messages directly
            messages = build_messages(element, parent_summaries)
            user_content = messages[1]["content"]

        messages = [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": user_content},
        ]

        from shared.ai.llm_client import LLMClient

        client = LLMClient.from_model_config(self._summary_mc)

        start = time.perf_counter()
        try:
            # Use a higher max_tokens than production summarization to
            # accommodate thinking models (e.g., qwen3.5) where the thinking
            # tokens consume part of the budget before the actual response.
            max_tokens = max(self.config.llm.summarize_max_tokens, 2048)
            raw = client.generate_from_messages(
                messages=messages,
                temperature=self.config.llm.summarize_temperature,
                max_tokens=max_tokens,
                timeout=180,
            )
            elapsed = time.perf_counter() - start
            return clean_summary(raw), elapsed
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error("Summary generation failed: %s", e)
            return "", elapsed

    def _evaluate_summary(
        self,
        element_type: str,
        element_name: str,
        source_code: str,
        summary: str,
    ) -> tuple[CriteriaScores, float]:
        """Evaluate a single summary using LLM-as-judge.

        Returns (CriteriaScores, elapsed_seconds).
        """
        summaries = {"candidate": summary}
        eval_prompt, label_to_model = build_evaluation_prompt(
            element_type=element_type,
            element_name=element_name,
            source_code=source_code,
            summaries=summaries,
        )
        labels = list(label_to_model.keys())

        bench_client = BenchmarkClient(api_base=self._eval_mc.url)
        api_model = self._eval_mc.get_litellm_model()

        start = time.perf_counter()
        max_retries = 3

        for attempt in range(max_retries):
            result = bench_client.generate(
                model=api_model,
                prompt=eval_prompt,
                temperature=0.1 + (attempt * 0.1),
                max_tokens=1024,
                timeout=120,
                api_base=self._eval_mc.get_api_base(),
                api_key=self._eval_mc.api_key,
            )

            if result.success:
                evaluations, error = parse_evaluation_response(
                    result.response, element_type, labels, label_to_model,
                )
                if not error and "candidate" in evaluations:
                    elapsed = time.perf_counter() - start
                    return evaluations["candidate"], elapsed
                if attempt < max_retries - 1:
                    continue

        elapsed = time.perf_counter() - start
        logger.warning("Evaluation failed after %d attempts", max_retries)
        return CriteriaScores(), elapsed

    def _improve_prompt(
        self,
        element: CodeElement,
        source_code: str,
        current_prompt: PromptVersion,
        summary: str,
        scores: CriteriaScores,
        template_key: str,
    ) -> dict[str, str] | None:
        """Use the improver LLM to produce a better prompt.

        Args:
            element: The code element.
            source_code: Raw source code.
            current_prompt: Current prompt version.
            summary: Summary generated by current prompt.
            scores: Evaluation scores.
            template_key: Template key for this element type.

        Returns:
            Dict with 'reasoning', 'system_prompt', 'user_template' or None on failure.
        """
        element_type = element.element_type
        criteria = EVALUATION_CRITERIA.get(element_type, EVALUATION_CRITERIA.get("function", {}))

        # Build scores text
        scores_lines = []
        for criterion, score in scores.scores.items():
            desc = criteria.get(criterion, "")
            marker = " ← NEEDS IMPROVEMENT" if score < 7 else ""
            scores_lines.append(f"  {criterion}: {score}/10 ({desc}){marker}")
        scores_text = "\n".join(scores_lines) if scores_lines else "  (no scores available)"

        # Build criteria descriptions
        criteria_lines = [f"  - {k}: {v}" for k, v in criteria.items()]
        criteria_text = "\n".join(criteria_lines)

        # Get required placeholders
        required_placeholders = _get_required_placeholders(template_key)

        # Calculate sentence range
        line_count = 1
        if element.line_end and element.line_start:
            line_count = max(1, element.line_end - element.line_start + 1)
        elif element.raw_code:
            line_count = element.raw_code.count("\n") + 1
        sentence_range = format_sentence_range(template_key, line_count)

        # Truncate source code for the improvement prompt
        truncated_source = source_code[:3000] if len(source_code) > 3000 else source_code

        prompt_text = IMPROVEMENT_PROMPT.format(
            source_code=truncated_source,
            current_system_prompt=current_prompt.system_prompt,
            current_user_template=current_prompt.user_template,
            summary=summary,
            scores_text=scores_text,
            eval_notes=scores.notes or "(no notes)",
            criteria_text=criteria_text,
            required_placeholders=", ".join(f"{{{p}}}" for p in required_placeholders),
            sentence_range=sentence_range,
        )

        bench_client = BenchmarkClient(api_base=self._improver_mc.url)
        api_model = self._improver_mc.get_litellm_model()

        max_retries = 3
        for attempt in range(max_retries):
            result = bench_client.generate(
                model=api_model,
                prompt=prompt_text,
                temperature=0.3 + (attempt * 0.1),
                max_tokens=2048,
                timeout=180,
                api_base=self._improver_mc.get_api_base(),
                api_key=self._improver_mc.api_key,
            )

            if result.success:
                parsed = _parse_improvement_response(result.response)
                if parsed:
                    # Validate that required placeholders are preserved
                    missing = [
                        p for p in required_placeholders
                        if f"{{{p}}}" not in parsed["user_template"]
                    ]
                    if missing:
                        logger.warning(
                            "Improved template missing placeholders: %s (attempt %d)",
                            missing, attempt,
                        )
                        if attempt < max_retries - 1:
                            continue
                        return None
                    return parsed

        logger.warning("Prompt improvement failed after %d attempts", max_retries)
        return None

    @staticmethod
    def _should_stop(
        iterations: list[IterationResult],
        target_score: float,
        max_iterations: int,
    ) -> tuple[bool, str]:
        """Determine whether the RALPH loop should stop.

        Returns (should_stop, reason).
        """
        if not iterations:
            return False, ""

        latest = iterations[-1]

        # Hit max iterations
        if len(iterations) >= max_iterations:
            return True, f"Reached max iterations ({max_iterations})"

        # Target score achieved
        if latest.avg_score >= target_score:
            return True, f"Target score reached ({latest.avg_score:.1f} >= {target_score})"

        # Plateau detection: score hasn't improved for 2 consecutive iterations
        if len(iterations) >= 3:
            recent = [r.avg_score for r in iterations[-3:]]
            if recent[-1] <= recent[-2] <= recent[-3]:
                return True, f"Score plateaued ({recent[-1]:.1f})"

        return False, ""
