"""Phase 5: Summarization - Generate summaries using LLM.

This module handles:
1. LLM client for text generation (via LiteLLM)
2. Prompt building for different element types
3. Hierarchical job processing
4. Summary storage

Supports multiple LLM providers through LiteLLM:
- Ollama (local)
- OpenAI
- Anthropic
- And many more
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from magaldi_core.code_parser import CodeElement

# Import the new LLM client
from shared.ai.llm_client import LLMClient, LLMError


class SummarizationError(Exception):
    """Raised when summarization fails."""

    pass


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SummarizationConfig:
    """Configuration for summarization."""

    # LLM settings (supports any LiteLLM provider)
    ollama_url: str = "http://localhost:11434"  # For Ollama provider
    model: str = "qwen2.5-coder:3b"
    provider: str = "ollama"  # ollama, openai, anthropic, etc.
    api_key: str | None = None  # For cloud providers

    # Generation settings (based on arxiv.org/html/2507.03160v2)
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 512
    timeout: int = 60

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 5.0

    # Code handling
    max_code_tokens: int = 4000


@dataclass
class SummarizationResult:
    """Result of summarization phase."""

    scope: str
    repository: str
    username: str

    # Counts by level
    files_summarized: int = 0
    classes_summarized: int = 0
    functions_summarized: int = 0
    variables_summarized: int = 0

    # Status
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0

    # Errors
    errors: list[str] = field(default_factory=list)


# =============================================================================
# LLM CLIENT FOR SUMMARIZATION
# =============================================================================


class SummarizationLLMClient:
    """Client for LLM text generation used in summarization.

    Supports multiple providers through LiteLLM:
    - Ollama (local)
    - OpenAI
    - Anthropic
    - And many more
    """

    def __init__(
        self,
        url: str,
        model: str,
        provider: str = "ollama",
        api_key: str | None = None,
    ):
        """Initialize LLM client.

        Args:
            url: API base URL (for Ollama: "http://localhost:11434")
            model: Model name (e.g., "qwen2.5-coder:3b", "gpt-4o-mini")
            provider: LLM provider (ollama, openai, anthropic, etc.)
            api_key: API key for cloud providers
        """
        self.url = url.rstrip("/") if url else ""
        self.model = model
        self.provider = provider
        self.api_key = api_key

        # Build full model identifier for LiteLLM
        if provider == "ollama":
            full_model = f"ollama/{model}"
            api_base = url
        elif provider == "openai":
            full_model = model
            api_base = None
        else:
            full_model = f"{provider}/{model}"
            api_base = None

        self._client = LLMClient(
            model=full_model,
            api_base=api_base,
            api_key=api_key,
        )

    def verify_model(self) -> bool:
        """Check if model is available."""
        return self._client.verify_model()

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 512,
        timeout: int = 60,
        model: str | None = None,
    ) -> str:
        """Generate completion from LLM.

        Args:
            prompt: The prompt to send to the model.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
            model: Optional model override (uses default if not specified).

        Returns:
            Generated text.

        Raises:
            ValueError: If response is empty or contains an error.
        """
        # Build model identifier for override if provided
        use_model = None
        if model:
            if self.provider == "ollama":
                use_model = f"ollama/{model}"
            elif self.provider == "openai":
                use_model = model
            else:
                use_model = f"{self.provider}/{model}"

        try:
            return self._client.generate(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=timeout,
                model=use_model,
            )
        except LLMError as e:
            raise ValueError(str(e)) from e


# =============================================================================
# REPOSITORY PROTOCOLS
# =============================================================================


class JobRepository(Protocol):
    """Interface for summarization job storage.

    All methods require scope, repository, and username for isolated queues.
    """

    def add_job(
        self,
        element_id: str,
        scope: str,
        repository: str,
        username: str,
        level: int,
        parent_id: str | None,
        dependencies_met: bool,
        priority: int = 0,
    ) -> None:
        """Add a summarization job to user's queue."""
        ...

    def get_job(
        self, element_id: str, scope: str, repository: str, username: str
    ) -> dict[str, Any] | None:
        """Get job by element ID from user's queue."""
        ...

    def claim_pending_jobs(
        self, worker_id: str, scope: str, repository: str, username: str, batch_size: int
    ) -> list[dict[str, Any]]:
        """Claim pending jobs from user's queue."""
        ...

    def mark_completed(
        self, element_id: str, scope: str, repository: str, username: str
    ) -> None:
        """Mark job as completed in user's queue."""
        ...

    def mark_failed(
        self, element_id: str, scope: str, repository: str, username: str, error_message: str
    ) -> None:
        """Mark job as failed in user's queue."""
        ...

    def unlock_dependencies(
        self, parent_element_id: str, scope: str, repository: str, username: str
    ) -> int:
        """Unlock jobs that depend on this element. Returns count unlocked."""
        ...


class SummaryStore(Protocol):
    """Interface for element and summary storage."""

    def store_element(self, element: CodeElement) -> None:
        """Store a code element."""
        ...

    def get_element(self, element_id: str) -> CodeElement | None:
        """Get element by ID."""
        ...

    def store_summary(self, element_id: str, summary: str) -> None:
        """Store summary for an element."""
        ...

    def get_summary(self, element_id: str) -> str | None:
        """Get summary for an element."""
        ...

    def get_parent_summaries(self, element: CodeElement) -> dict[str, str]:
        """Get summaries from parent elements."""
        ...


# =============================================================================
# IN-MEMORY IMPLEMENTATIONS
# =============================================================================


class InMemoryJobRepository:
    """In-memory implementation of job repository for testing.

    Note: This simple implementation stores all jobs globally.
    The scope/repository/username parameters are accepted for interface compatibility.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}

    def add_job(
        self,
        element_id: str,
        scope: str,
        repository: str,
        username: str,
        level: int,
        parent_id: str | None,
        dependencies_met: bool,
        priority: int = 0,
    ) -> None:
        self._jobs[element_id] = {
            "element_id": element_id,
            "scope": scope,
            "repository": repository,
            "username": username,
            "level": level,
            "parent_id": parent_id,
            "dependencies_met": dependencies_met,
            "status": "pending",
            "worker_id": None,
            "claimed_at": None,
            "completed_at": None,
            "error_message": None,
            "priority": priority if priority else (100 - level),
        }

    def get_job(
        self, element_id: str, scope: str, repository: str, username: str
    ) -> dict[str, Any] | None:
        return self._jobs.get(element_id)

    def claim_pending_jobs(
        self, worker_id: str, scope: str, repository: str, username: str, batch_size: int
    ) -> list[dict[str, Any]]:
        # Find pending jobs with dependencies met, sorted by level then priority
        available = [
            job
            for job in self._jobs.values()
            if job["status"] == "pending" and job["dependencies_met"]
        ]
        available.sort(key=lambda j: (j["level"], -j["priority"]))

        claimed = available[:batch_size]
        for job in claimed:
            job["status"] = "running"
            job["worker_id"] = worker_id
            job["claimed_at"] = datetime.now()

        return claimed

    def mark_completed(
        self, element_id: str, scope: str, repository: str, username: str
    ) -> None:
        if element_id in self._jobs:
            self._jobs[element_id]["status"] = "completed"
            self._jobs[element_id]["completed_at"] = datetime.now()

    def mark_failed(
        self, element_id: str, scope: str, repository: str, username: str, error_message: str
    ) -> None:
        if element_id in self._jobs:
            self._jobs[element_id]["status"] = "failed"
            self._jobs[element_id]["error_message"] = error_message
            self._jobs[element_id]["completed_at"] = datetime.now()

    def unlock_dependencies(
        self, parent_element_id: str, scope: str, repository: str, username: str
    ) -> int:
        count = 0
        for job in self._jobs.values():
            if (
                job["parent_id"] == parent_element_id
                and job["status"] == "pending"
                and not job["dependencies_met"]
            ):
                job["dependencies_met"] = True
                count += 1
        return count


class InMemorySummaryStore:
    """In-memory implementation of summary store for testing."""

    def __init__(self) -> None:
        self._elements: dict[str, CodeElement] = {}
        self._summaries: dict[str, str] = {}

    def store_element(self, element: CodeElement) -> None:
        self._elements[element.element_id] = element

    def get_element(self, element_id: str) -> CodeElement | None:
        return self._elements.get(element_id)

    def store_summary(self, element_id: str, summary: str) -> None:
        self._summaries[element_id] = summary

    def get_summary(self, element_id: str) -> str | None:
        return self._summaries.get(element_id)

    def get_parent_summaries(self, element: CodeElement) -> dict[str, str]:
        summaries: dict[str, str] = {}

        # For non-file elements, find file summary
        if element.level > 0:
            # Find file element for this path
            for eid, elem in self._elements.items():
                if (
                    elem.scope == element.scope
                    and elem.repository == element.repository
                    and elem.username == element.username
                    and elem.relative_path == element.relative_path
                    and elem.element_type == "file"
                ):
                    file_summary = self.get_summary(eid)
                    if file_summary:
                        summaries["file"] = file_summary
                    break

        # For methods, find class summary
        if element.parent_id and element.element_type in ("method", "function"):
            parent = self.get_element(element.parent_id)
            if parent and parent.element_type == "class":
                class_summary = self.get_summary(element.parent_id)
                if class_summary:
                    summaries["class"] = class_summary

        return summaries


# =============================================================================
# CODE TRUNCATION
# =============================================================================


def truncate_code(code: str, max_tokens: int = 4000) -> str:
    """Truncate code to fit context window.

    Args:
        code: Source code to truncate.
        max_tokens: Maximum tokens (rough estimate: 1 token ~= 4 chars).

    Returns:
        Truncated code with marker if truncated.
    """
    max_chars = max_tokens * 4

    if len(code) <= max_chars:
        return code

    truncated = code[:max_chars]

    # Try to end at a complete line
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.8:
        truncated = truncated[:last_newline]

    return truncated + "\n\n# ... (truncated)"


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

PROMPTS = {
    "file": """Summarize this {language} file in 4-6 sentences for an AI agent navigating this codebase. Address:
- The primary purpose and responsibility of this module
- What problem domain or capability it provides to the system
- Key patterns, abstractions, or architectural decisions used
- When an agent should look in this file (what tasks or questions lead here)
- Important dependencies or integrations with other parts of the system

Do NOT enumerate individual classes or functions - those are documented separately.

File: {file_path}

Code:
{code}

IMPORTANT: Write ONLY the 4-6 sentence summary below. Do NOT include any reasoning, analysis, bullet points, or explanations. Start directly with the first sentence of the summary.

Summary:""",
    "class": """Summarize this {language} class in 4-6 sentences for an AI agent navigating this codebase.

FOCUS on the class itself. Use the file context only to understand how this class fits in - do not repeat or summarize the file context.

Address:
- What this class represents, models, or encapsulates
- Its core responsibility and the problem it solves
- How and when to instantiate or use this class
- Key state it manages and invariants it maintains
- How it collaborates with other classes or modules

Do NOT enumerate individual methods - those are documented separately.

File context (for understanding only): {file_summary}

Class: {class_name}
{decorators}

Code:
{code}

IMPORTANT: Write ONLY the 4-6 sentence summary below. Do NOT include any reasoning, analysis, bullet points, or explanations. Start directly with the first sentence of the summary.

Summary:""",
    "function": """Describe this function in 4-6 sentences for an AI agent navigating this codebase.

FOCUS on the function itself. Use file/class context only to understand the function's role - do not repeat or summarize the context.

Address:
- What operation, transformation, or task this function performs
- The inputs it accepts (with their purposes) and what it returns
- When to call this function - what scenarios or tasks require it
- Side effects: external state changes, I/O, exceptions raised
- Key edge cases or preconditions the caller should know

File context (for understanding only): {file_summary}
{class_context}

Function: {function_name}
Signature: {signature}
{docstring_section}

Code:
{code}

IMPORTANT: Write ONLY the 4-6 sentence summary below. Do NOT include any reasoning, analysis, bullet points, or explanations. Start directly with the first sentence of the summary.

Summary:""",
    "method": """Describe this method in 4-6 sentences for an AI agent navigating this codebase.

FOCUS on the method itself. Use file/class context only to understand the method's role - do not repeat or summarize the context.

Address:
- What operation this method performs on or for the object
- The inputs it accepts (with their purposes) and what it returns
- How it reads or modifies the object's state
- When to call this method in the object's lifecycle
- Side effects, exceptions, or preconditions the caller should know

File context (for understanding only): {file_summary}
Class context (for understanding only): {class_summary}

Method: {method_name}
Signature: {signature}
{docstring_section}

Code:
{code}

IMPORTANT: Write ONLY the 4-6 sentence summary below. Do NOT include any reasoning, analysis, bullet points, or explanations. Start directly with the first sentence of the summary.

Summary:""",
    "constant": """Describe this constant in 2-3 sentences for an AI agent navigating this codebase.

FOCUS on the constant itself. Use context only to understand its purpose - do not repeat or summarize the context.

Address:
- What configuration, value, or data this constant represents
- Where and why this constant is used in the system
- Any important constraints or relationships with other values

File context (for understanding only): {file_summary}
{function_context}

Name: {name}
Value:
{code}
{usages_section}

IMPORTANT: Write ONLY the 2-3 sentence description below. Do NOT include any reasoning, analysis, bullet points, or explanations. Start directly with the first sentence.

Description:""",
    "variable": """Describe this variable in 2-3 sentences for an AI agent navigating this codebase.

FOCUS on the variable itself. Use context only to understand its purpose - do not repeat or summarize the context.

Address:
- What data, state, or configuration this variable holds
- How it is initialized and when it changes
- Its role in the containing scope's behavior

File context (for understanding only): {file_summary}
{class_context}
{function_context}

Name: {name}
Value:
{code}
{usages_section}

IMPORTANT: Write ONLY the 2-3 sentence description below. Do NOT include any reasoning, analysis, bullet points, or explanations. Start directly with the first sentence.

Description:""",
}


def build_prompt(
    element: CodeElement,
    parent_summaries: dict[str, str],
    max_code_tokens: int = 4000,
) -> str:
    """Build prompt with parent context.

    Args:
        element: Code element to summarize.
        parent_summaries: Dict with 'file' and/or 'class' summaries.
        max_code_tokens: Max tokens for code in prompt.

    Returns:
        Formatted prompt string.
    """
    element_type = element.element_type

    # Get template
    if element_type == "method":
        template = PROMPTS["method"]
    elif element_type in PROMPTS:
        template = PROMPTS[element_type]
    else:
        template = PROMPTS["function"]

    # Build context sections
    file_summary = parent_summaries.get("file", "No file context available.")
    class_summary = parent_summaries.get("class", "")
    function_summary = parent_summaries.get("function", "")

    class_context = ""
    if class_summary and element_type in ("method", "function", "variable", "constant"):
        class_context = f"Class context (for understanding only): {class_summary}"

    function_context = ""
    if function_summary and element_type in ("variable", "constant"):
        function_context = f"Function/method context (for understanding only): {function_summary}"

    docstring_section = ""
    if element.docstring:
        docstring_section = f"Docstring: {element.docstring}"

    decorators = ""
    if element.decorators:
        decorators = f"Decorators: {', '.join(element.decorators)}"

    # Build usages section for variables/constants
    usages_section = ""
    if element.context_usages:
        usages_section = "\nUsed in:\n" + "\n".join(f"- {u}" for u in element.context_usages)

    # Truncate code
    code = truncate_code(element.raw_code or "", max_code_tokens)

    return template.format(
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
    )


# =============================================================================
# SUMMARY CLEANING
# =============================================================================


def clean_summary(summary: str) -> str:
    """Clean and normalize generated summary.

    Args:
        summary: Raw summary from LLM.

    Returns:
        Cleaned summary.
    """
    summary = summary.strip()

    if not summary:
        return summary

    # Truncate at chat template markers (model starting a new turn)
    # These indicate the model went past the intended response
    chat_markers = [
        "<|im_start|>",      # ChatML format (Qwen, etc.)
        "<|im_end|>",        # ChatML end
        "<|im_sep|>",        # ChatML separator
        "<|assistant|>",     # Phi format
        "<|user|>",          # Phi format
        "<|end|>",           # Generic end
        "<|eot_id|>",        # Llama 3 format
        "<|start_header_id|>",  # Llama 3 format
        "[INST]",            # Mistral/Llama 2 format
        "[/INST]",           # Mistral/Llama 2 format
        "### ",              # Alpaca format (### Response:, ### Human:, etc.)
    ]
    for marker in chat_markers:
        if marker in summary:
            summary = summary.split(marker)[0]

    # Remove reasoning/thinking tags from models like nemotron-3-nano, DeepSeek, etc.
    # These models output chain-of-thought in <think>...</think> or similar tags
    thinking_patterns = [
        r"<think>.*?</think>\s*",      # <think>...</think>
        r"<thinking>.*?</thinking>\s*", # <thinking>...</thinking>
        r"<reasoning>.*?</reasoning>\s*", # <reasoning>...</reasoning>
        r"<reflection>.*?</reflection>\s*", # <reflection>...</reflection>
    ]
    for pattern in thinking_patterns:
        summary = re.sub(pattern, "", summary, flags=re.DOTALL | re.IGNORECASE)

    # Handle unclosed thinking tags (content got cut off)
    # Remove everything from opening tag to end if no closing tag
    unclosed_patterns = [
        r"<think>.*$",
        r"<thinking>.*$",
        r"<reasoning>.*$",
        r"<reflection>.*$",
    ]
    for pattern in unclosed_patterns:
        summary = re.sub(pattern, "", summary, flags=re.DOTALL | re.IGNORECASE)

    summary = summary.strip()

    if not summary:
        return summary

    # Remove common prefixes
    prefixes_to_remove = [
        "Summary:",
        "This function ",
        "This method ",
        "This class ",
        "This file ",
    ]

    for prefix in prefixes_to_remove:
        if summary.lower().startswith(prefix.lower()):
            summary = summary[len(prefix) :].strip()
            break

    # Ensure ends with period
    if summary and not summary.endswith("."):
        summary += "."

    # Capitalize first letter
    if summary:
        summary = summary[0].upper() + summary[1:]

    return summary


# =============================================================================
# DEPENDENCY RESOLUTION
# =============================================================================


def update_dependencies_after_completion(
    element_id: str,
    scope: str,
    repository: str,
    username: str,
    job_repo: JobRepository,
) -> int:
    """Mark dependent jobs as ready when parent completes.

    Args:
        element_id: Completed element ID.
        scope: Repository scope.
        repository: Repository name.
        username: User who owns the jobs.
        job_repo: Job repository.

    Returns:
        Count of jobs unlocked.
    """
    return job_repo.unlock_dependencies(element_id, scope, repository, username)


# =============================================================================
# SUMMARY GENERATION
# =============================================================================


def generate_summary(
    element: CodeElement,
    summary_store: SummaryStore,
    llm_client: SummarizationLLMClient,
    config: SummarizationConfig,
) -> str:
    """Generate summary for a single element.

    Args:
        element: Code element to summarize.
        summary_store: Summary store for parent context.
        llm_client: LLM client for text generation.
        config: Summarization config.

    Returns:
        Generated summary.
    """
    # Get parent summaries for context
    parent_summaries = summary_store.get_parent_summaries(element)

    # Build prompt
    prompt = build_prompt(element, parent_summaries, config.max_code_tokens)

    # Generate summary
    raw_summary = llm_client.generate(
        prompt=prompt,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
    )

    # Clean and return
    return clean_summary(raw_summary)


def process_summarization_job(
    element_id: str,
    scope: str,
    repository: str,
    username: str,
    job_repo: JobRepository,
    summary_store: SummaryStore,
    llm_client: SummarizationLLMClient,
    config: SummarizationConfig,
) -> bool:
    """Process a single summarization job.

    Args:
        element_id: Element ID to summarize.
        scope: Repository scope.
        repository: Repository name.
        username: User who owns this job.
        job_repo: Job repository.
        summary_store: Summary store.
        llm_client: LLM client for text generation.
        config: Summarization config.

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Get element
        element = summary_store.get_element(element_id)
        if element is None:
            job_repo.mark_failed(
                element_id, scope, repository, username,
                f"Element not found: {element_id}"
            )
            return False

        # Generate summary
        summary = generate_summary(element, summary_store, llm_client, config)

        # Store summary
        summary_store.store_summary(element_id, summary)

        # Mark job completed
        job_repo.mark_completed(element_id, scope, repository, username)

        # Unlock dependent jobs
        update_dependencies_after_completion(
            element_id, scope, repository, username, job_repo
        )

        return True

    except Exception as e:
        job_repo.mark_failed(element_id, scope, repository, username, str(e))
        return False
