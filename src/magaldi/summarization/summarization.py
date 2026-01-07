"""Phase 5: Summarization - Generate summaries using Ollama.

This module handles:
1. Ollama client for LLM interactions
2. Prompt building for different element types
3. Hierarchical job processing
4. Summary storage
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import requests

from magaldi.parser.code_parser import CodeElement


class SummarizationError(Exception):
    """Raised when summarization fails."""

    pass


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SummarizationConfig:
    """Configuration for summarization."""

    # Ollama settings
    ollama_url: str = "http://localhost:11434"
    model: str = "qwen2.5-coder:7b"

    # Generation settings
    temperature: float = 0.3
    max_tokens: int = 256
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
# OLLAMA CLIENT
# =============================================================================


class OllamaClient:
    """Client for Ollama API interactions."""

    def __init__(self, url: str, model: str):
        self.url = url.rstrip("/")
        self.model = model
        self.session = requests.Session()
        self.session.headers["Content-Type"] = "application/json"

    def verify_model(self) -> bool:
        """Check if model is available."""
        try:
            response = self.session.get(f"{self.url}/api/tags")
            models = response.json().get("models", [])
            return any(m.get("name") == self.model for m in models)
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 256,
        timeout: int = 60,
    ) -> str:
        """Generate completion from Ollama."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        response = self.session.post(
            f"{self.url}/api/generate",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

        return response.json().get("response", "").strip()


# =============================================================================
# REPOSITORY PROTOCOLS
# =============================================================================


class JobRepository(Protocol):
    """Interface for summarization job storage."""

    def add_job(
        self,
        element_id: str,
        level: int,
        parent_id: str | None,
        dependencies_met: bool,
    ) -> None:
        """Add a summarization job."""
        ...

    def get_job(self, element_id: str) -> dict[str, Any] | None:
        """Get job by element ID."""
        ...

    def claim_pending_jobs(
        self, worker_id: str, batch_size: int
    ) -> list[dict[str, Any]]:
        """Claim pending jobs that have dependencies met."""
        ...

    def mark_completed(self, element_id: str) -> None:
        """Mark job as completed."""
        ...

    def mark_failed(self, element_id: str, error_message: str) -> None:
        """Mark job as failed."""
        ...

    def unlock_dependencies(self, parent_element_id: str) -> int:
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
    """In-memory implementation of job repository for testing."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}

    def add_job(
        self,
        element_id: str,
        level: int,
        parent_id: str | None,
        dependencies_met: bool,
    ) -> None:
        self._jobs[element_id] = {
            "element_id": element_id,
            "level": level,
            "parent_id": parent_id,
            "dependencies_met": dependencies_met,
            "status": "pending",
            "worker_id": None,
            "claimed_at": None,
            "completed_at": None,
            "error_message": None,
            "priority": 100 - level,
        }

    def get_job(self, element_id: str) -> dict[str, Any] | None:
        return self._jobs.get(element_id)

    def claim_pending_jobs(
        self, worker_id: str, batch_size: int
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

    def mark_completed(self, element_id: str) -> None:
        if element_id in self._jobs:
            self._jobs[element_id]["status"] = "completed"
            self._jobs[element_id]["completed_at"] = datetime.now()

    def mark_failed(self, element_id: str, error_message: str) -> None:
        if element_id in self._jobs:
            self._jobs[element_id]["status"] = "failed"
            self._jobs[element_id]["error_message"] = error_message
            self._jobs[element_id]["completed_at"] = datetime.now()

    def unlock_dependencies(self, parent_element_id: str) -> int:
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
    "file": """Summarize this {language} file in 2-3 sentences. Focus on:
- Primary purpose and responsibility
- Key classes/functions it provides
- Notable patterns or design choices

File: {file_path}

Code:
{code}

Summary:""",
    "class": """Summarize this {language} class in 2-3 sentences. Include its purpose,
key responsibilities, and notable patterns.

File context: {file_summary}

Class: {class_name}
{decorators}

Code:
{code}

Summary:""",
    "function": """Summarize what this function does in 1-2 concise sentences.

File context: {file_summary}
{class_context}

Function: {function_name}
Signature: {signature}
{docstring_section}

Code:
{code}

Summary:""",
    "method": """Summarize what this method does in 1-2 concise sentences.

File context: {file_summary}
Class context: {class_summary}

Method: {method_name}
Signature: {signature}
{docstring_section}

Code:
{code}

Summary:""",
    "constant": """Describe this {language} constant in one sentence. Focus on its purpose and how it's used.

File: {file_path}
Name: {name}
Value:
{code}
{usages_section}
Description:""",
    "variable": """Describe this {language} class variable in one sentence. Focus on its purpose and how it's used.

File context: {file_summary}
Class context: {class_summary}

Name: {name}
Value:
{code}
{usages_section}
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

    class_context = ""
    if class_summary and element_type in ("method", "function"):
        class_context = f"Class context: {class_summary}"

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
    element_id: str, job_repo: JobRepository
) -> int:
    """Mark dependent jobs as ready when parent completes.

    Args:
        element_id: Completed element ID.
        job_repo: Job repository.

    Returns:
        Count of jobs unlocked.
    """
    return job_repo.unlock_dependencies(element_id)


# =============================================================================
# SUMMARY GENERATION
# =============================================================================


def generate_summary(
    element: CodeElement,
    summary_store: SummaryStore,
    ollama: OllamaClient,
    config: SummarizationConfig,
) -> str:
    """Generate summary for a single element.

    Args:
        element: Code element to summarize.
        summary_store: Summary store for parent context.
        ollama: Ollama client.
        config: Summarization config.

    Returns:
        Generated summary.
    """
    # Get parent summaries for context
    parent_summaries = summary_store.get_parent_summaries(element)

    # Build prompt
    prompt = build_prompt(element, parent_summaries, config.max_code_tokens)

    # Generate with Ollama
    raw_summary = ollama.generate(
        prompt=prompt,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
    )

    # Clean and return
    return clean_summary(raw_summary)


def process_summarization_job(
    element_id: str,
    job_repo: JobRepository,
    summary_store: SummaryStore,
    ollama: OllamaClient,
    config: SummarizationConfig,
) -> bool:
    """Process a single summarization job.

    Args:
        element_id: Element ID to summarize.
        job_repo: Job repository.
        summary_store: Summary store.
        ollama: Ollama client.
        config: Summarization config.

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Get element
        element = summary_store.get_element(element_id)
        if element is None:
            job_repo.mark_failed(element_id, f"Element not found: {element_id}")
            return False

        # Generate summary
        summary = generate_summary(element, summary_store, ollama, config)

        # Store summary
        summary_store.store_summary(element_id, summary)

        # Mark job completed
        job_repo.mark_completed(element_id)

        # Unlock dependent jobs
        update_dependencies_after_completion(element_id, job_repo)

        return True

    except Exception as e:
        job_repo.mark_failed(element_id, str(e))
        return False
