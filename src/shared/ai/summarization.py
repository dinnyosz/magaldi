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
    model: str = "qwen3:4b-instruct"
    provider: str = "ollama"  # ollama, openai, anthropic, etc.
    api_key: str | None = None  # For cloud providers

    # Generation settings (based on arxiv.org/html/2507.03160v2)
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 512
    timeout: int = 180  # 3 minutes to handle queue wait with many workers

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
# DYNAMIC SENTENCE RANGES
# =============================================================================
# Sentence count scales with element size (line count) to avoid over-describing
# simple elements and under-describing complex ones.

LINE_THRESHOLDS: dict[str, dict[str, int]] = {
    "file":       {"tiny": 20,  "small": 50,  "medium": 200},
    "class":      {"tiny": 10,  "small": 30,  "medium": 100},
    "interface":  {"tiny": 5,   "small": 15,  "medium": 50},
    "type_alias": {"tiny": 1,   "small": 3,   "medium": 10},
    "function":   {"tiny": 5,   "small": 15,  "medium": 50},
    "method":     {"tiny": 5,   "small": 15,  "medium": 50},
    "constant":   {"tiny": 1,   "small": 3,   "medium": 5},
    "variable":   {"tiny": 1,   "small": 3,   "medium": 5},
}

SENTENCE_RANGES: dict[str, dict[str, tuple[int, int]]] = {
    "file": {
        "tiny":   (1, 2),
        "small":  (2, 3),
        "medium": (3, 4),
        "large":  (5, 6),
    },
    "class": {
        "tiny":   (1, 2),
        "small":  (2, 3),
        "medium": (3, 4),
        "large":  (5, 6),
    },
    "interface": {
        "tiny":   (1, 2),
        "small":  (2, 3),
        "medium": (3, 4),
        "large":  (4, 5),
    },
    "type_alias": {
        "tiny":   (1, 1),
        "small":  (1, 2),
        "medium": (2, 2),
        "large":  (2, 3),
    },
    "function": {
        "tiny":   (1, 2),
        "small":  (2, 3),
        "medium": (3, 4),
        "large":  (5, 6),
    },
    "method": {
        "tiny":   (1, 2),
        "small":  (2, 3),
        "medium": (3, 4),
        "large":  (4, 5),
    },
    "constant": {
        "tiny":   (1, 1),
        "small":  (1, 2),
        "medium": (2, 2),
        "large":  (2, 3),
    },
    "variable": {
        "tiny":   (1, 1),
        "small":  (1, 2),
        "medium": (2, 2),
        "large":  (2, 3),
    },
}


def get_size_tier(element_type: str, line_count: int) -> str:
    """Determine size tier based on element type and line count.

    Args:
        element_type: Type of element (file, class, function, etc.)
        line_count: Number of lines in the element.

    Returns:
        Size tier: "tiny", "small", "medium", or "large".
    """
    thresholds = LINE_THRESHOLDS.get(element_type, LINE_THRESHOLDS["function"])
    if line_count <= thresholds["tiny"]:
        return "tiny"
    elif line_count <= thresholds["small"]:
        return "small"
    elif line_count <= thresholds["medium"]:
        return "medium"
    return "large"


def get_sentence_range(element_type: str, line_count: int) -> tuple[int, int]:
    """Get sentence range for an element based on its type and size.

    Args:
        element_type: Type of element (file, class, function, etc.)
        line_count: Number of lines in the element.

    Returns:
        Tuple of (min_sentences, max_sentences).
    """
    tier = get_size_tier(element_type, line_count)
    ranges = SENTENCE_RANGES.get(element_type, SENTENCE_RANGES["function"])
    return ranges.get(tier, (3, 4))


def format_sentence_range(element_type: str, line_count: int) -> str:
    """Format sentence range as a string for prompts.

    Args:
        element_type: Type of element (file, class, function, etc.)
        line_count: Number of lines in the element.

    Returns:
        Formatted string like "2-3" or "1" (if min==max).
    """
    min_s, max_s = get_sentence_range(element_type, line_count)
    if min_s == max_s:
        return str(min_s)
    return f"{min_s}-{max_s}"


# =============================================================================
# LLM CLIENT FOR SUMMARIZATION
# =============================================================================


class SummarizationLLMClient:
    """Client for LLM text generation used in summarization.

    Supports multiple providers through LiteLLM:
    - Ollama (local)
    - llamacpp (llama.cpp server with continuous batching)
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
            url: API base URL (for Ollama: "http://localhost:11434",
                 for llamacpp: "http://localhost:8080/v1")
            model: Model name (e.g., "qwen2.5-coder:3b", "gpt-4o-mini")
            provider: LLM provider (ollama, llamacpp, openai, anthropic, etc.)
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
        elif provider == "llamacpp":
            # llama.cpp server exposes OpenAI-compatible API
            full_model = f"openai/{model}"
            api_base = url  # Should include /v1 suffix
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
        num_ctx: int | None = None,
    ) -> str:
        """Generate completion from LLM.

        Args:
            prompt: The prompt to send to the model.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
            model: Optional model override (uses default if not specified).
            num_ctx: Context window size for Ollama models.

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
            elif self.provider == "llamacpp":
                # llama.cpp uses OpenAI-compatible API
                use_model = f"openai/{model}"
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
                num_ctx=num_ctx,
            )
        except LLMError as e:
            raise ValueError(str(e)) from e

    def generate_from_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 512,
        timeout: int = 60,
        model: str | None = None,
        num_ctx: int | None = None,
    ) -> str:
        """Generate completion from messages (optimized for prefix caching).

        This method uses system + user messages to maximize Ollama's KV cache
        reuse. The system message (static instructions) gets cached, while
        the user message contains variable content.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
            model: Optional model override.
            num_ctx: Context window size for Ollama models.

        Returns:
            Generated text.

        Raises:
            ValueError: If response is empty or contains an error.
        """
        use_model = None
        if model:
            if self.provider == "ollama":
                use_model = f"ollama/{model}"
            elif self.provider == "llamacpp":
                use_model = f"openai/{model}"
            elif self.provider == "openai":
                use_model = model
            else:
                use_model = f"{self.provider}/{model}"

        try:
            return self._client.generate_from_messages(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=timeout,
                model=use_model,
                num_ctx=num_ctx,
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
# PROMPT TEMPLATES (Optimized for Prefix Caching)
# =============================================================================
# System messages are STATIC and get cached by Ollama's KV cache.
# User messages contain VARIABLE content (code, paths, context).
# This structure maximizes cache reuse across requests of the same type.

SYSTEM_PROMPTS = {
    "file": """You summarize code files for AI agents navigating codebases.

For each file, provide a {sentence_range} sentence summary answering:
1. PURPOSE: What is this module's primary job? What capability does it provide?
2. DOMAIN: What problem space does this code address?
3. ARCHITECTURE: What design patterns or abstractions does it use?
4. DISCOVERY: When should an agent look here? What questions lead to this file?
5. DEPENDENCIES: What external modules or systems does this integrate with?

Do NOT list individual classes/functions - those are documented separately.
Write ONLY the {sentence_range} sentence summary. No reasoning, explanations, or bullet points.
Start directly with what it does - never start with "This module...", "This file...", or similar.""",

    "class": """You summarize classes for AI agents navigating codebases.

For each class, provide a {sentence_range} sentence summary answering:
1. IDENTITY: What real-world concept or data structure does this class model?
2. RESPONSIBILITY: What problem does using this class solve?
3. INSTANTIATION: How do you create an instance? What parameters are required?
4. STATE: What key attributes does it hold? What makes an instance valid vs invalid?
5. COLLABORATORS: What other classes or modules does it work with?

Do NOT list methods - those are documented separately.
Write ONLY the {sentence_range} sentence summary. No reasoning, explanations, or bullet points.
Start directly with what it models/does - never start with "This class...", "The X class...", or similar.""",

    "interface": """You summarize interfaces for AI agents navigating codebases.

For each interface, provide a {sentence_range} sentence summary answering:
1. CONTRACT: What behavior or capability does this interface define?
2. IMPLEMENTERS: What types of classes should implement this?
3. METHODS: What key methods must implementers provide?
4. USAGE: When should code depend on this interface vs a concrete class?

Write ONLY the {sentence_range} sentence summary. No reasoning, explanations, or bullet points.
Start directly with what contract it defines - never start with "This interface...", "The X interface...", or similar.""",

    "type_alias": """You describe type aliases for AI agents navigating codebases.

For each type alias, provide a {sentence_range} sentence description answering:
1. MEANING: What concept or data shape does this type represent?
2. STRUCTURE: What is the underlying type structure?
3. USAGE: Where and why is this alias used instead of the raw type?

Write ONLY the {sentence_range} sentence description. No reasoning, explanations, or bullet points.
Start directly with what it represents - never start with "This type...", "The X type...", or similar.""",

    "function": """You describe functions for AI agents navigating codebases.

For each function, provide a {sentence_range} sentence description answering:
1. OPERATION: What does calling this function accomplish?
2. INTERFACE: What are the parameters and return value? What types?
3. WHEN TO USE: In what scenario should an agent call this?
4. SIDE EFFECTS: Does it modify external state, perform I/O, or raise exceptions?
5. EDGE CASES: What happens with empty/None inputs? What preconditions must hold?

Write ONLY the {sentence_range} sentence description. No reasoning, explanations, or bullet points.
Start with an action verb - never start with "This function...", "The X function...", or "This function is used to...".""",

    "method": """You describe methods for AI agents navigating codebases.

For each method, provide a {sentence_range} sentence description answering:
1. OPERATION: What does this method do to/for the object?
2. INTERFACE: What parameters does it take? What does it return?
3. STATE: Which instance attributes does it read or modify?
4. LIFECYCLE: Is this setup/init, cleanup/teardown, or called repeatedly?
5. ERRORS: What exceptions can it raise? What preconditions must hold?

Write ONLY the {sentence_range} sentence description. No reasoning, explanations, or bullet points.
Start with an action verb - never start with "This method...", "The X method...", or "This method is used to...".""",

    "constant": """You describe constants for AI agents navigating codebases.

For each constant, provide a {sentence_range} sentence description answering:
1. MEANING: What does this value represent or configure?
2. USAGE: Where in the system is it used?
3. CONSTRAINTS: What are valid values? Any min/max bounds?

Write ONLY the {sentence_range} sentence description. No reasoning, explanations, or bullet points.
Start directly with what it represents - never start with "This constant...", "The X constant...", or similar.""",

    "variable": """You describe variables for AI agents navigating codebases.

For each variable, provide a {sentence_range} sentence description answering:
1. DATA: What information does this variable hold?
2. LIFECYCLE: How is it initialized? When does it change?
3. ROLE: How does this variable influence its containing scope?

Write ONLY the {sentence_range} sentence description. No reasoning, explanations, or bullet points.
Start directly with what it holds - never start with "This variable...", "The X variable...", or similar.""",
}

# User message templates - contain variable content
USER_PROMPTS = {
    "file": """Language: {language}
File: {file_path}
{imports_section}

Code:
{code}""",

    "class": """File context: {file_summary}

Class: {class_name}
{decorators}
{base_classes_section}
{attributes_section}
{collaborators_section}
{instantiation_hints}

Code:
{code}
{usages_section}""",

    "interface": """File context: {file_summary}

Interface: {name}
{base_classes_section}

Code:
{code}
{usages_section}""",

    "type_alias": """File context: {file_summary}

Type alias: {name}

Code:
{code}
{usages_section}""",

    "function": """File context: {file_summary}
{class_context}

Function: {function_name}
Signature: {signature}
{docstring_section}
{exceptions_section}
{usage_examples}

Code:
{code}""",

    "method": """File context: {file_summary}
Class context: {class_summary}

Method: {method_name}
Signature: {signature}
{docstring_section}
{state_section}
{exceptions_section}

Code:
{code}
{usages_section}""",

    "constant": """File context: {file_summary}
{class_context}
{function_context}

Name: {name}
{usage_examples}

Value:
{code}""",

    "variable": """File context: {file_summary}
{class_context}
{function_context}

Name: {name}

Value:
{code}
{usages_section}""",
}

# Legacy single-prompt templates (kept for backwards compatibility)
PROMPTS = {
    "file": """Summarize this {language} file in {sentence_range} sentences for an AI agent navigating this codebase.

Answer these questions:
1. PURPOSE: What is this module's primary job? What capability does it provide?
2. DOMAIN: What problem space does this code address?
3. ARCHITECTURE: What design patterns or abstractions does it use? (e.g., factory, repository, decorator, event-driven)
4. DISCOVERY: When should an agent look here? What questions or tasks lead to this file?
5. DEPENDENCIES: What external modules or systems does this integrate with?
{imports_section}

Do NOT list individual classes/functions - those are documented separately.

File: {file_path}

Code:
{code}

Write ONLY the {sentence_range} sentence summary. No reasoning or bullet points.
Start directly with what it does - never start with "This module...", "This file...", or similar.

Summary:""",
    "class": """Summarize this {language} class in {sentence_range} sentences for an AI agent.

FOCUS on the class itself. Use file context only to understand how it fits in.

Answer these questions:
1. IDENTITY: What real-world concept or data structure does this class model?
2. RESPONSIBILITY: What problem does using this class solve?
3. INSTANTIATION: How do you create an instance? What parameters are required? Any factory methods or singletons?{instantiation_hints}
4. STATE: What key attributes does it hold? What makes an instance valid vs invalid?{attributes_section}
5. COLLABORATORS: What other classes or modules does it work with?{collaborators_section}

Do NOT list methods - those are documented separately.
{base_classes_section}

File context: {file_summary}

Class: {class_name}
{decorators}

Code:
{code}
{usages_section}

Write ONLY the {sentence_range} sentence summary. No reasoning or bullet points.
Start directly with what it models/does - never start with "This class...", "The X class...", or similar.

Summary:""",
    "interface": """Summarize this interface in {sentence_range} sentences for an AI agent.

FOCUS on the interface itself. Use file context only to understand how it fits in.

Answer these questions:
1. CONTRACT: What behavior or capability does this interface define?
2. IMPLEMENTERS: What types of classes should implement this?
3. METHODS: What key methods must implementers provide?
4. USAGE: When should code depend on this interface vs a concrete class?
{base_classes_section}

File context: {file_summary}

Interface: {name}

Code:
{code}
{usages_section}

Write ONLY the {sentence_range} sentence summary. No reasoning or bullet points.
Start directly with what contract it defines - never start with "This interface...", "The X interface...", or similar.

Summary:""",
    "type_alias": """Describe this type alias in {sentence_range} sentences for an AI agent.

FOCUS on the type alias itself.

Answer these questions:
1. MEANING: What concept or data shape does this type represent?
2. STRUCTURE: What is the underlying type structure?
3. USAGE: Where and why is this alias used instead of the raw type?

File context: {file_summary}

Type alias: {name}

Code:
{code}
{usages_section}

Write ONLY the {sentence_range} sentence description. No reasoning or bullet points.
Start directly with what it represents - never start with "This type...", "The X type...", or similar.

Description:""",
    "function": """Describe this function in {sentence_range} sentences for an AI agent.

FOCUS on the function itself. Use context only to understand its role.

Answer these questions:
1. OPERATION: What does calling this function accomplish?
2. INTERFACE: What are the parameters and return value? What types?
3. WHEN TO USE: In what scenario should an agent call this? What task requires it?{usage_examples}
4. SIDE EFFECTS: Does it modify external state, perform I/O, or raise exceptions?{exceptions_section}
5. EDGE CASES: What happens with empty/None inputs? What preconditions must hold? What errors can occur?

File context: {file_summary}
{class_context}

Function: {function_name}
Signature: {signature}
{docstring_section}

Code:
{code}

Write ONLY the {sentence_range} sentence summary. No reasoning or bullet points.
Start with an action verb - never start with "This function...", "The X function...", or "This function is used to...".

Summary:""",
    "method": """Describe this method in {sentence_range} sentences for an AI agent.

FOCUS on the method itself. Use context only to understand its role.

Answer these questions:
1. OPERATION: What does this method do to/for the object?
2. INTERFACE: What parameters does it take? What does it return?
3. STATE: Which instance attributes does it read? Which does it modify?{state_section}
4. LIFECYCLE: Is this a setup/init method? Cleanup/teardown? Called once or repeatedly? Must it be called in a specific order relative to other methods?
5. ERRORS: What exceptions can it raise? What preconditions must hold?{exceptions_section}

File context: {file_summary}
Class context: {class_summary}

Method: {method_name}
Signature: {signature}
{docstring_section}

Code:
{code}
{usages_section}

Write ONLY the {sentence_range} sentence summary. No reasoning or bullet points.
Start with an action verb - never start with "This method...", "The X method...", or "This method is used to...".

Summary:""",
    "constant": """Describe this constant in {sentence_range} sentences for an AI agent.

FOCUS on the constant itself.

Answer these questions:
1. MEANING: What does this value represent or configure?
2. USAGE: Where in the system is it used?{usage_examples}
3. CONSTRAINTS: What are valid values? Any min/max bounds? Related constants that must stay consistent?

File context: {file_summary}
{function_context}

Name: {name}
Value:
{code}

Write ONLY the {sentence_range} sentence description. No reasoning or bullet points.
Start directly with what it represents - never start with "This constant...", "The X constant...", or similar.

Description:""",
    "variable": """Describe this variable in {sentence_range} sentences for an AI agent.

FOCUS on the variable itself.

Answer these questions:
1. DATA: What information does this variable hold?
2. LIFECYCLE: How is it initialized? When and why does it change?
3. ROLE: How does this variable influence the behavior of its containing scope?

File context: {file_summary}
{class_context}
{function_context}

Name: {name}
Value:
{code}
{usages_section}

Write ONLY the {sentence_range} sentence description. No reasoning or bullet points.
Start directly with what it holds - never start with "This variable...", "The X variable...", or similar.

Description:""",
}


# =============================================================================
# CONTEXT BUILDER HELPERS
# =============================================================================


def _build_imports_section(element: CodeElement) -> str:
    """Build imports section for file prompt.

    Shows key external imports to indicate dependencies.
    """
    if not element.imports:
        return ""

    # Filter to external imports (not relative)
    external = [
        imp.module for imp in element.imports
        if not imp.module.startswith(".")
    ][:5]  # Limit to 5

    if not external:
        return ""

    return f"\nKey imports: {', '.join(external)}"


def _build_attributes_section(element: CodeElement) -> str:
    """Build attributes section for class prompt.

    Shows instance attributes defined in __init__.
    """
    if not element.class_attributes:
        return ""

    attr_names = [a["name"] for a in element.class_attributes][:5]  # Limit to 5
    if not attr_names:
        return ""

    return f"\nInstance attributes: {', '.join(attr_names)}"


def _build_base_classes_section(element: CodeElement) -> str:
    """Build base classes section for class prompt."""
    if not element.base_classes:
        return ""

    return f"\nInherits from: {', '.join(element.base_classes)}"


def _build_collaborators_section(element: CodeElement) -> str:
    """Build collaborators section for class prompt.

    Shows types this class interacts with based on calls.
    """
    if not element.calls:
        return ""

    # Get unique receivers that aren't 'self'
    receivers = list({
        c.receiver for c in element.calls
        if c.receiver and c.receiver != "self"
    })[:3]  # Limit to 3

    if not receivers:
        return ""

    return f"\nUses: {', '.join(receivers)}"


def _build_instantiation_hints(element: CodeElement) -> str:
    """Build instantiation hints for class prompt.

    Filters context_usages for instantiation examples.
    """
    if not element.context_usages:
        return ""

    # Filter for instantiation usages
    insts = [
        u for u in element.context_usages
        if "instantiated" in u.lower()
    ][:2]  # Limit to 2

    if not insts:
        return ""

    return "\nInstantiation examples:\n" + "\n".join(f"- {u}" for u in insts)


def _build_exceptions_section(element: CodeElement) -> str:
    """Build exceptions section for function/method prompt."""
    if not element.exceptions_raised:
        return ""

    return f"\nRaises: {', '.join(element.exceptions_raised)}"


def _build_state_section(element: CodeElement) -> str:
    """Build state section for method prompt.

    Shows which attributes this method modifies.
    """
    if not element.attributes_modified:
        return ""

    return f"\nModifies attributes: {', '.join(element.attributes_modified)}"


def _build_usage_examples(element: CodeElement) -> str:
    """Build usage examples section for function/constant prompts."""
    if not element.context_usages:
        return ""

    examples = element.context_usages[:3]  # Limit to 3
    return "\nUsage examples:\n" + "\n".join(f"- {u}" for u in examples)


# =============================================================================
# PROMPT BUILDING
# =============================================================================


def build_prompt(
    element: CodeElement,
    parent_summaries: dict[str, str],
    max_code_tokens: int = 4000,
) -> str:
    """Build prompt with parent context and enhanced context sections.

    Args:
        element: Code element to summarize.
        parent_summaries: Dict with 'file' and/or 'class' summaries.
        max_code_tokens: Max tokens for code in prompt.

    Returns:
        Formatted prompt string.
    """
    element_type = element.element_type

    # Get template key
    if element_type == "method":
        template_key = "method"
    elif element_type in PROMPTS:
        template_key = element_type
    else:
        template_key = "function"

    template = PROMPTS[template_key]

    # Calculate line count for dynamic sentence range
    line_count = 1
    if element.line_end and element.line_start:
        line_count = max(1, element.line_end - element.line_start + 1)
    elif element.raw_code:
        line_count = element.raw_code.count("\n") + 1

    # Get sentence range based on element type and size
    sentence_range = format_sentence_range(template_key, line_count)

    # Build parent context sections
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

    # Build usages section for variables/constants
    usages_section = ""
    if element.context_usages:
        usages_section = "\nUsed in:\n" + "\n".join(f"- {u}" for u in element.context_usages)

    # Build enhanced context sections based on element type
    imports_section = _build_imports_section(element) if element_type == "file" else ""
    attributes_section = _build_attributes_section(element) if element_type == "class" else ""
    base_classes_section = _build_base_classes_section(element) if element_type in ("class", "interface") else ""
    collaborators_section = _build_collaborators_section(element) if element_type == "class" else ""
    instantiation_hints = _build_instantiation_hints(element) if element_type == "class" else ""
    exceptions_section = _build_exceptions_section(element) if element_type in ("function", "method") else ""
    state_section = _build_state_section(element) if element_type == "method" else ""
    usage_examples = _build_usage_examples(element) if element_type in ("function", "constant") else ""

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
        sentence_range=sentence_range,
        # Enhanced context sections
        imports_section=imports_section,
        attributes_section=attributes_section,
        base_classes_section=base_classes_section,
        collaborators_section=collaborators_section,
        instantiation_hints=instantiation_hints,
        exceptions_section=exceptions_section,
        state_section=state_section,
        usage_examples=usage_examples,
    )


def build_messages(
    element: CodeElement,
    parent_summaries: dict[str, str],
    max_code_tokens: int = 4000,
) -> list[dict[str, str]]:
    """Build messages optimized for Ollama KV cache prefix caching.

    Returns system + user messages where:
    - System message: Static instructions (cached after first request of each type)
    - User message: Variable content (code, paths, context)

    This structure maximizes cache reuse - the system message tokens are
    processed once and cached, then reused for all subsequent requests
    of the same element type.

    Args:
        element: Code element to summarize.
        parent_summaries: Dict with 'file' and/or 'class' summaries.
        max_code_tokens: Max tokens for code in prompt.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    element_type = element.element_type

    # Map element type to template key
    if element_type == "method":
        template_key = "method"
    elif element_type in SYSTEM_PROMPTS:
        template_key = element_type
    else:
        template_key = "function"

    # Calculate line count for dynamic sentence range
    line_count = 1
    if element.line_end and element.line_start:
        line_count = max(1, element.line_end - element.line_start + 1)
    elif element.raw_code:
        line_count = element.raw_code.count("\n") + 1

    # Get sentence range based on element type and size
    sentence_range = format_sentence_range(template_key, line_count)

    # Get system prompt and format with sentence range
    system_prompt = SYSTEM_PROMPTS[template_key].format(sentence_range=sentence_range)

    # Build variable content for user message
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

    # Build enhanced context sections
    imports_section = _build_imports_section(element) if element_type == "file" else ""
    attributes_section = _build_attributes_section(element) if element_type == "class" else ""
    base_classes_section = _build_base_classes_section(element) if element_type in ("class", "interface") else ""
    collaborators_section = _build_collaborators_section(element) if element_type == "class" else ""
    instantiation_hints = _build_instantiation_hints(element) if element_type == "class" else ""
    exceptions_section = _build_exceptions_section(element) if element_type in ("function", "method") else ""
    state_section = _build_state_section(element) if element_type == "method" else ""
    usage_examples = _build_usage_examples(element) if element_type in ("function", "constant") else ""

    # Truncate code
    code = truncate_code(element.raw_code or "", max_code_tokens)

    # Format user message with variable content
    user_template = USER_PROMPTS[template_key]
    user_content = user_template.format(
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
        attributes_section=attributes_section,
        base_classes_section=base_classes_section,
        collaborators_section=collaborators_section,
        instantiation_hints=instantiation_hints,
        exceptions_section=exceptions_section,
        state_section=state_section,
        usage_examples=usage_examples,
    )

    # Clean up extra blank lines from empty optional sections
    user_content = "\n".join(line for line in user_content.split("\n") if line.strip() or line == "")
    user_content = user_content.strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


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

    Uses message-based format (system + user) optimized for Ollama's KV cache
    prefix caching. The system message contains static instructions that get
    cached, while the user message has variable content with shared context
    (file_summary, class_summary) at the top for maximum cache reuse.

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

    # Build messages optimized for prefix caching:
    # - System message: static instructions (cached per element type)
    # - User message: shared context first (file/class), then element-specific
    messages = build_messages(element, parent_summaries, config.max_code_tokens)

    # Generate summary using message-based API
    raw_summary = llm_client.generate_from_messages(
        messages=messages,
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
