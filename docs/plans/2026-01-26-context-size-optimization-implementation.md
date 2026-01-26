# Context Size Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Optimize KV cache efficiency by computing and using per-element-type context sizes for local LLM inference.

**Architecture:** During Phase 3, track max char count per element type. Compute optimal `num_ctx` per type. Pass to Phase 5 summarization and Phase 6 features/glossary. LLM client passes `num_ctx` to local providers (Ollama, llama.cpp, LM Studio).

**Tech Stack:** Python dataclasses, LiteLLM, Ollama/llama.cpp

---

## Task 1: Add Context Size Computation Utilities

**Files:**
- Create: `src/shared/ai/context_size.py`
- Test: `tests/test_context_size.py`

**Step 1: Write the failing test for compute_num_ctx**

```python
# tests/test_context_size.py
"""Tests for context size computation utilities."""

import pytest

from shared.ai.context_size import (
    CONTEXT_TIERS,
    PROMPT_OVERHEAD,
    compute_context_sizes,
    compute_num_ctx,
)


class TestComputeNumCtx:
    """Tests for compute_num_ctx function."""

    def test_small_variable_returns_smallest_tier(self):
        """Variable with 100 chars should use 2048 context."""
        result = compute_num_ctx("variable", 100)
        assert result == 2048

    def test_medium_function_returns_4096(self):
        """Function with 4000 chars should use 4096 context."""
        result = compute_num_ctx("function", 4000)
        assert result == 4096

    def test_large_class_returns_8192(self):
        """Class with 20000 chars should use 8192 context."""
        result = compute_num_ctx("class", 20000)
        assert result == 8192

    def test_xlarge_file_returns_16384(self):
        """File with 50000 chars should use 16384 context."""
        result = compute_num_ctx("file", 50000)
        assert result == 16384

    def test_huge_file_returns_largest_tier(self):
        """File with 200000 chars should use largest tier."""
        result = compute_num_ctx("file", 200000)
        assert result == CONTEXT_TIERS[-1]

    def test_unknown_type_uses_default_overhead(self):
        """Unknown element type should use default overhead."""
        result = compute_num_ctx("unknown", 1000)
        assert result in CONTEXT_TIERS


class TestComputeContextSizes:
    """Tests for compute_context_sizes function."""

    def test_computes_sizes_for_all_types(self):
        """Should compute context size for each element type in max_chars."""
        max_chars = {
            "file": 40000,
            "class": 15000,
            "function": 3000,
            "method": 2000,
            "variable": 200,
            "constant": 100,
        }
        result = compute_context_sizes(max_chars)

        assert "file" in result
        assert "class" in result
        assert "function" in result
        assert "method" in result
        assert "variable" in result
        assert "constant" in result
        # All values should be valid tiers
        for tier in result.values():
            assert tier in CONTEXT_TIERS

    def test_empty_max_chars_returns_empty(self):
        """Empty input should return empty dict."""
        result = compute_context_sizes({})
        assert result == {}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_context_size.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'shared.ai.context_size'"

**Step 3: Write minimal implementation**

```python
# src/shared/ai/context_size.py
"""Context size computation utilities for KV cache optimization.

This module computes optimal context sizes (num_ctx) per element type
based on observed maximum code sizes during parsing. Using type-specific
context sizes improves KV cache efficiency for local LLM inference.
"""

from __future__ import annotations

# Context size tiers (powers of 2 for memory alignment)
CONTEXT_TIERS = [2048, 4096, 8192, 16384, 32768]

# Estimated prompt overhead per element type (tokens)
# Accounts for system prompt, context, and formatting
PROMPT_OVERHEAD = {
    "file": 400,
    "class": 500,
    "function": 400,
    "method": 450,
    "variable": 200,
    "constant": 200,
}

DEFAULT_OVERHEAD = 300


def compute_num_ctx(element_type: str, max_chars: int) -> int:
    """Compute optimal context size for an element type.

    Args:
        element_type: Type of code element (file, class, function, etc.)
        max_chars: Maximum character count observed for this type.

    Returns:
        Optimal num_ctx value from CONTEXT_TIERS.
    """
    # Estimate tokens: ~4 chars per token for code
    estimated_tokens = max_chars // 4
    overhead = PROMPT_OVERHEAD.get(element_type, DEFAULT_OVERHEAD)
    total_tokens = estimated_tokens + overhead

    # Find smallest tier that fits
    for tier in CONTEXT_TIERS:
        if total_tokens < tier:
            return tier

    # Fallback to largest tier
    return CONTEXT_TIERS[-1]


def compute_context_sizes(max_chars_by_type: dict[str, int]) -> dict[str, int]:
    """Compute context sizes for all element types.

    Args:
        max_chars_by_type: Dict mapping element_type to max char count.

    Returns:
        Dict mapping element_type to optimal num_ctx.
    """
    return {
        element_type: compute_num_ctx(element_type, max_chars)
        for element_type, max_chars in max_chars_by_type.items()
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_context_size.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/shared/ai/context_size.py tests/test_context_size.py
git commit -m "feat: add context size computation utilities for KV cache optimization"
```

---

## Task 2: Add aggregation_context_size to LLMConfig

**Files:**
- Modify: `src/shared/config.py:110-170`
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_config.py in TestLLMConfigDefaults class

def test_aggregation_context_size_default(self):
    """Should have default aggregation context size."""
    config = LLMConfig()
    assert config.aggregation_context_size == 16384
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::TestLLMConfigDefaults::test_aggregation_context_size_default -v`
Expected: FAIL with "AttributeError: 'LLMConfig' object has no attribute 'aggregation_context_size'"

**Step 3: Add field to LLMConfig**

In `src/shared/config.py`, add after line 169 (`embed_context_window: int = 8192`):

```python
    # Context size for aggregation tasks (features, glossary)
    # Uses fixed large context since input size depends on cluster size
    aggregation_context_size: int = 16384
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::TestLLMConfigDefaults::test_aggregation_context_size_default -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/shared/config.py tests/test_config.py
git commit -m "feat(config): add aggregation_context_size for features/glossary"
```

---

## Task 3: Add num_ctx Parameter to LLM Client

**Files:**
- Modify: `src/shared/ai/llm_client.py:453-526` (generate method)
- Modify: `src/shared/ai/llm_client.py:527-597` (generate_from_messages method)
- Test: `tests/test_llm_client.py`

**Step 1: Write the failing test for generate method**

```python
# Add to tests/test_llm_client.py

class TestNumCtxParameter:
    """Tests for num_ctx parameter support."""

    def test_generate_passes_num_ctx_for_ollama(self, mocker):
        """Should pass num_ctx to LiteLLM for Ollama provider."""
        mock_completion = mocker.patch("shared.ai.llm_client.completion")
        mock_completion.return_value.choices = [
            mocker.MagicMock(message=mocker.MagicMock(content="test response"))
        ]

        client = LLMClient(model="ollama/qwen3:4b", api_base="http://localhost:11434")
        client.generate("test prompt", num_ctx=4096)

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs.get("num_ctx") == 4096

    def test_generate_passes_num_ctx_for_llamacpp(self, mocker):
        """Should pass n_ctx via extra_body for llama.cpp provider."""
        mock_completion = mocker.patch("shared.ai.llm_client.completion")
        mock_completion.return_value.choices = [
            mocker.MagicMock(message=mocker.MagicMock(content="test response"))
        ]

        client = LLMClient(model="openai/qwen3:4b", api_base="http://localhost:8080/v1")
        client.generate("test prompt", num_ctx=4096)

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs.get("extra_body", {}).get("n_ctx") == 4096

    def test_generate_from_messages_passes_num_ctx(self, mocker):
        """Should pass num_ctx in generate_from_messages."""
        mock_completion = mocker.patch("shared.ai.llm_client.completion")
        mock_completion.return_value.choices = [
            mocker.MagicMock(message=mocker.MagicMock(content="test response"))
        ]

        client = LLMClient(model="ollama/qwen3:4b", api_base="http://localhost:11434")
        client.generate_from_messages(
            [{"role": "user", "content": "test"}],
            num_ctx=8192
        )

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs.get("num_ctx") == 8192

    def test_generate_without_num_ctx_does_not_include_it(self, mocker):
        """Should not include num_ctx if not provided."""
        mock_completion = mocker.patch("shared.ai.llm_client.completion")
        mock_completion.return_value.choices = [
            mocker.MagicMock(message=mocker.MagicMock(content="test response"))
        ]

        client = LLMClient(model="ollama/qwen3:4b", api_base="http://localhost:11434")
        client.generate("test prompt")

        call_kwargs = mock_completion.call_args[1]
        assert "num_ctx" not in call_kwargs
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_client.py::TestNumCtxParameter -v`
Expected: FAIL with "TypeError: generate() got an unexpected keyword argument 'num_ctx'"

**Step 3: Update generate method**

In `src/shared/ai/llm_client.py`, update the `generate` method signature and body:

```python
def generate(
    self,
    prompt: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_tokens: int = 512,
    timeout: int = 60,
    model: str | None = None,
    num_ctx: int | None = None,  # New parameter
) -> str:
    """Generate text completion.

    Args:
        prompt: The prompt to send to the model.
        temperature: Sampling temperature (0.0 to 1.0).
        top_p: Nucleus sampling parameter (0.0 to 1.0).
        max_tokens: Maximum tokens to generate.
        timeout: Request timeout in seconds.
        model: Optional model override.
        num_ctx: Optional context size for local providers (Ollama, llama.cpp, etc.)

    Returns:
        Generated text.

    Raises:
        LLMError: If generation fails after retries.
    """
    use_model = model or self.model

    # Check if this is a thinking model that needs think=false
    model_name = use_model.split("/")[-1] if "/" in use_model else use_model
    is_thinking_model = any(model_name.startswith(tm) for tm in self.THINKING_MODELS)

    def _do_generate() -> str:
        # Build kwargs for litellm
        kwargs: dict[str, Any] = {
            "model": use_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }

        # Add api_base for custom endpoints (Ollama, llama.cpp, etc.)
        if self.api_base:
            kwargs["api_base"] = self.api_base

        # Add api_key if provided, or use dummy key for OpenAI-compatible local servers
        if self.api_key:
            kwargs["api_key"] = self.api_key
        elif use_model.startswith("openai/") and self.api_base:
            # Local OpenAI-compatible servers (llama.cpp) don't need auth
            # but LiteLLM requires an API key for openai/ prefix
            kwargs["api_key"] = "not-needed"

        # Disable thinking mode for models that support it
        # LiteLLM added think parameter support in PR #15465 (Sept 2025)
        if is_thinking_model and use_model.startswith("ollama/"):
            kwargs["think"] = False

        # Add num_ctx for local providers (KV cache optimization)
        if num_ctx:
            if use_model.startswith("ollama/"):
                kwargs["num_ctx"] = num_ctx
            elif use_model.startswith("openai/") and self.api_base:
                # OpenAI-compatible local servers (llama.cpp, LM Studio, LocalAI)
                kwargs["extra_body"] = kwargs.get("extra_body", {})
                kwargs["extra_body"]["n_ctx"] = num_ctx

        response = completion(**kwargs)

        # Extract text from response
        content = response.choices[0].message.content
        if content is None:
            raise LLMError(f"Empty response from model '{use_model}'")

        return content.strip()

    return _retry_with_backoff(
        _do_generate,
        max_retries=self.max_retries,
        operation=f"LLM generation ({use_model})",
    )
```

**Step 4: Update generate_from_messages method similarly**

```python
def generate_from_messages(
    self,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_tokens: int = 512,
    timeout: int = 60,
    model: str | None = None,
    num_ctx: int | None = None,  # New parameter
) -> str:
    """Generate text completion from messages (system + user).

    This method is optimized for Ollama's KV cache prefix caching.
    The system message is static and gets cached, while the user
    message contains variable content.

    Args:
        messages: List of message dicts with 'role' and 'content'.
                  Typically [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        temperature: Sampling temperature (0.0 to 1.0).
        top_p: Nucleus sampling parameter (0.0 to 1.0).
        max_tokens: Maximum tokens to generate.
        timeout: Request timeout in seconds.
        model: Optional model override.
        num_ctx: Optional context size for local providers (Ollama, llama.cpp, etc.)

    Returns:
        Generated text.

    Raises:
        LLMError: If generation fails after retries.
    """
    use_model = model or self.model

    # Check if this is a thinking model that needs think=false
    model_name = use_model.split("/")[-1] if "/" in use_model else use_model
    is_thinking_model = any(model_name.startswith(tm) for tm in self.THINKING_MODELS)

    def _do_generate() -> str:
        kwargs: dict[str, Any] = {
            "model": use_model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }

        if self.api_base:
            kwargs["api_base"] = self.api_base

        if self.api_key:
            kwargs["api_key"] = self.api_key
        elif use_model.startswith("openai/") and self.api_base:
            kwargs["api_key"] = "not-needed"

        if is_thinking_model and use_model.startswith("ollama/"):
            kwargs["think"] = False

        # Add num_ctx for local providers (KV cache optimization)
        if num_ctx:
            if use_model.startswith("ollama/"):
                kwargs["num_ctx"] = num_ctx
            elif use_model.startswith("openai/") and self.api_base:
                # OpenAI-compatible local servers (llama.cpp, LM Studio, LocalAI)
                kwargs["extra_body"] = kwargs.get("extra_body", {})
                kwargs["extra_body"]["n_ctx"] = num_ctx

        response = completion(**kwargs)

        content = response.choices[0].message.content
        if content is None:
            raise LLMError(f"Empty response from model '{use_model}'")

        return content.strip()

    return _retry_with_backoff(
        _do_generate,
        max_retries=self.max_retries,
        operation=f"LLM generation ({use_model})",
    )
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_llm_client.py::TestNumCtxParameter -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/shared/ai/llm_client.py tests/test_llm_client.py
git commit -m "feat(llm): add num_ctx parameter for local provider KV cache optimization"
```

---

## Task 4: Add num_ctx to SummarizationLLMClient

**Files:**
- Modify: `src/shared/ai/summarization.py:149-250`
- Test: `tests/test_summarization.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_summarization.py

class TestSummarizationLLMClientNumCtx:
    """Tests for num_ctx parameter in SummarizationLLMClient."""

    def test_generate_passes_num_ctx(self, mocker):
        """Should pass num_ctx to underlying client."""
        mock_client = mocker.MagicMock()
        mock_client.generate.return_value = "test summary"
        mocker.patch.object(
            SummarizationLLMClient, "_client", mock_client, create=True
        )

        client = SummarizationLLMClient(
            url="http://localhost:11434",
            model="qwen3:4b",
            provider="ollama",
        )
        client._client = mock_client
        client.generate("test prompt", num_ctx=4096)

        mock_client.generate.assert_called_once()
        call_kwargs = mock_client.generate.call_args[1]
        assert call_kwargs.get("num_ctx") == 4096

    def test_generate_from_messages_passes_num_ctx(self, mocker):
        """Should pass num_ctx in generate_from_messages."""
        mock_client = mocker.MagicMock()
        mock_client.generate_from_messages.return_value = "test summary"

        client = SummarizationLLMClient(
            url="http://localhost:11434",
            model="qwen3:4b",
            provider="ollama",
        )
        client._client = mock_client
        client.generate_from_messages(
            [{"role": "user", "content": "test"}],
            num_ctx=8192
        )

        mock_client.generate_from_messages.assert_called_once()
        call_kwargs = mock_client.generate_from_messages.call_args[1]
        assert call_kwargs.get("num_ctx") == 8192
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_summarization.py::TestSummarizationLLMClientNumCtx -v`
Expected: FAIL with "TypeError: generate() got an unexpected keyword argument 'num_ctx'"

**Step 3: Update SummarizationLLMClient methods**

In `src/shared/ai/summarization.py`, update both `generate` and `generate_from_messages`:

```python
def generate(
    self,
    prompt: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_tokens: int = 512,
    timeout: int = 60,
    model: str | None = None,
    num_ctx: int | None = None,  # New parameter
) -> str:
    """Generate completion from LLM.

    Args:
        prompt: The prompt to send to the model.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        max_tokens: Maximum tokens to generate.
        timeout: Request timeout in seconds.
        model: Optional model override (uses default if not specified).
        num_ctx: Optional context size for local providers.

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
    num_ctx: int | None = None,  # New parameter
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
        num_ctx: Optional context size for local providers.

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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_summarization.py::TestSummarizationLLMClientNumCtx -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/shared/ai/summarization.py tests/test_summarization.py
git commit -m "feat(summarization): add num_ctx parameter to LLM client wrapper"
```

---

## Task 5: Track Max Chars During Parsing and Add to ParsingResult

**Files:**
- Modify: `src/magaldi_core/code_parser.py:301-322` (ParsingResult class)
- Test: `tests/test_code_parser.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_code_parser.py in TestParsingResult class

def test_max_chars_by_type_property(self):
    """Should compute max chars per element type."""
    from magaldi_core.code_parser import CodeElement, ParsedFile, ParsingResult

    # Create elements with different code sizes
    elements = [
        CodeElement(element_id="1", element_type="function", raw_code="x" * 1000),
        CodeElement(element_id="2", element_type="function", raw_code="x" * 2000),
        CodeElement(element_id="3", element_type="class", raw_code="x" * 5000),
        CodeElement(element_id="4", element_type="file", raw_code="x" * 10000),
    ]
    parsed_file = ParsedFile(
        file_info=mocker.MagicMock(),
        elements=elements,
    )
    result = ParsingResult(
        scope="test",
        repository="repo",
        username="user",
        parsed_files=[parsed_file],
    )

    max_chars = result.max_chars_by_type

    assert max_chars["function"] == 2000  # Max of 1000, 2000
    assert max_chars["class"] == 5000
    assert max_chars["file"] == 10000


def test_context_sizes_property(self):
    """Should compute context sizes from max chars."""
    from magaldi_core.code_parser import CodeElement, ParsedFile, ParsingResult

    elements = [
        CodeElement(element_id="1", element_type="function", raw_code="x" * 4000),
        CodeElement(element_id="2", element_type="variable", raw_code="x" * 100),
    ]
    parsed_file = ParsedFile(
        file_info=mocker.MagicMock(),
        elements=elements,
    )
    result = ParsingResult(
        scope="test",
        repository="repo",
        username="user",
        parsed_files=[parsed_file],
    )

    context_sizes = result.context_sizes

    assert "function" in context_sizes
    assert "variable" in context_sizes
    # Should be valid context tiers
    from shared.ai.context_size import CONTEXT_TIERS
    assert context_sizes["function"] in CONTEXT_TIERS
    assert context_sizes["variable"] in CONTEXT_TIERS
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_code_parser.py::TestParsingResult::test_max_chars_by_type_property -v`
Expected: FAIL with "AttributeError: 'ParsingResult' object has no attribute 'max_chars_by_type'"

**Step 3: Add properties to ParsingResult**

In `src/magaldi_core/code_parser.py`, add after the `elements_by_type` property:

```python
@property
def max_chars_by_type(self) -> dict[str, int]:
    """Get max character count per element type.

    Used for computing optimal context sizes for KV cache optimization.
    """
    max_chars: dict[str, int] = {}
    for pf in self.parsed_files:
        for elem in pf.elements:
            code_len = len(elem.raw_code or "")
            current_max = max_chars.get(elem.element_type, 0)
            max_chars[elem.element_type] = max(current_max, code_len)
    return max_chars

@property
def context_sizes(self) -> dict[str, int]:
    """Get computed context sizes per element type.

    Returns optimal num_ctx values for each element type based on
    observed maximum code sizes. Used for KV cache optimization
    during summarization.
    """
    from shared.ai.context_size import compute_context_sizes
    return compute_context_sizes(self.max_chars_by_type)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_code_parser.py::TestParsingResult -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/magaldi_core/code_parser.py tests/test_code_parser.py
git commit -m "feat(parser): add max_chars_by_type and context_sizes to ParsingResult"
```

---

## Task 6: Display Context Size Analysis in CLI

**Files:**
- Modify: `src/shared/cli.py:1547-1552` (print_parsing_result function)
- Test: Manual verification (CLI output)

**Step 1: Update print_parsing_result function**

In `src/shared/cli.py`, replace the `print_parsing_result` function:

```python
def print_parsing_result(result: ParsingResult) -> None:
    """Print parsing results including context size analysis."""
    # Existing summary line
    types = ", ".join(f"{t}: [green]{c}[/]" for t, c in sorted(result.elements_by_type.items()))
    failed = f" | [red]{len(result.failed_files)} failed[/]" if result.failed_files else ""
    console.print(f"  [green]{len(result.parsed_files)}[/] files → [green]{result.total_elements}[/] elements ({types}){failed}")

    # Context size analysis table
    max_chars = result.max_chars_by_type
    context_sizes = result.context_sizes

    if max_chars:
        console.print()
        console.print("  [dim]Context size analysis (for KV cache optimization):[/]")
        console.print("  [dim]  Element Type   Max Chars   Est. Tokens   Context Size[/]")
        console.print("  [dim]  ─────────────────────────────────────────────────────────[/]")

        # Sort by context size descending for readability
        sorted_types = sorted(max_chars.keys(), key=lambda t: context_sizes.get(t, 0), reverse=True)

        for element_type in sorted_types:
            chars = max_chars[element_type]
            tokens = chars // 4 + 300  # Rough estimate with overhead
            ctx_size = context_sizes.get(element_type, 0)
            console.print(f"  [dim]  {element_type:<14} {chars:>9,}   {tokens:>11,}   {ctx_size:>12,}[/]")
```

**Step 2: Verify manually**

Run: `magaldi parse /path/to/small/repo --user test --skip-ai`
Expected: Should show context size analysis table after parsing

**Step 3: Commit**

```bash
git add src/shared/cli.py
git commit -m "feat(cli): display context size analysis after parsing"
```

---

## Task 7: Pass num_ctx in Processor During Summarization

**Files:**
- Modify: `src/magaldi_core/processor.py:669-703` (_summarize_element function)
- Modify: `src/magaldi_core/processor.py:95-130` (ProcessingConfig class)
- Test: `tests/test_processor.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_processor.py

class TestContextSizePassthrough:
    """Tests for passing context sizes to summarization."""

    def test_summarize_element_uses_context_size(self, mocker):
        """Should pass num_ctx from context_sizes dict."""
        from magaldi_core.processor import _summarize_element, ProcessingConfig, _SummaryCache

        mock_llm = mocker.MagicMock()
        mock_llm.generate.return_value = "test summary"

        element = CodeElement(
            element_id="test:repo:user:file.py:function:foo:1",
            element_type="function",
            name="foo",
            raw_code="def foo(): pass",
        )

        config = ProcessingConfig()
        config.context_sizes = {"function": 4096, "file": 16384}

        cache = _SummaryCache()
        cache.add_element(element)

        _summarize_element(element, cache, mock_llm, config)

        # Verify num_ctx was passed
        call_kwargs = mock_llm.generate.call_args[1]
        assert call_kwargs.get("num_ctx") == 4096
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_processor.py::TestContextSizePassthrough -v`
Expected: FAIL with "AttributeError: 'ProcessingConfig' object has no attribute 'context_sizes'"

**Step 3: Add context_sizes to ProcessingConfig**

In `src/magaldi_core/processor.py`, add to ProcessingConfig class:

```python
# After existing fields, add:
context_sizes: dict[str, int] = field(default_factory=dict)
```

**Step 4: Update _summarize_element to use context_sizes**

```python
def _summarize_element(
    element: CodeElement,
    summary_cache: _SummaryCache,
    llm_client: SummarizationLLMClient,
    config: ProcessingConfig,
) -> str:
    """Generate summary for an element.

    Args:
        element: Element to summarize.
        summary_cache: Cache with parent summaries.
        llm_client: LLM client for text generation.
        config: Processing configuration.

    Returns:
        Generated summary.
    """
    # Get parent summaries for context
    parent_summaries = summary_cache.get_parent_summaries(element)

    # Build prompt with context
    prompt = build_prompt(element, parent_summaries, config.max_code_tokens)

    # Get model and context size for this element type
    model_config = config.get_model_for_element_type(element.element_type)
    num_ctx = config.context_sizes.get(element.element_type)

    # Generate summary
    raw_summary = llm_client.generate(
        prompt=prompt,
        temperature=config.summarize_temperature,
        max_tokens=config.summarize_max_tokens,
        timeout=config.summarize_timeout,
        model=model_config.name,
        num_ctx=num_ctx,
    )

    # Clean and return
    return clean_summary(raw_summary)
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_processor.py::TestContextSizePassthrough -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/magaldi_core/processor.py tests/test_processor.py
git commit -m "feat(processor): pass context sizes to summarization"
```

---

## Task 8: Wire Context Sizes Through CLI Pipeline

**Files:**
- Modify: `src/shared/cli.py:168-350` (parse command and run_processing)
- Test: Manual integration test

**Step 1: Update CLI to pass context_sizes from parsing to processing**

Find where `run_processing` is called and pass `context_sizes` from `ParsingResult`:

```python
# In the parse command, after run_parsing:
parsing_result = run_parsing(manifest)
print_parsing_result(parsing_result)

# When calling run_processing, include context_sizes:
processing_config.context_sizes = parsing_result.context_sizes
```

**Step 2: Verify integration**

Run: `magaldi parse /path/to/repo --user test`
Expected: Should show context size table and use those sizes during summarization

**Step 3: Commit**

```bash
git add src/shared/cli.py
git commit -m "feat(cli): wire context sizes from parsing to processing"
```

---

## Task 9: Add num_ctx to Feature Processing

**Files:**
- Modify: `src/shared/ai/clustering/feature_processor.py:330-391`
- Test: `tests/test_feature_processor.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_feature_processor.py

def test_generate_feature_summary_uses_aggregation_context(self, mocker):
    """Should use aggregation_context_size for features."""
    mock_llm = mocker.MagicMock()
    mock_llm.generate_from_messages.return_value = "Feature summary."

    config = FeatureProcessingConfig()
    config.aggregation_context_size = 16384

    cluster = ClusterResult(
        cluster_id=1,
        label="test_feature",
        element_ids=["id1", "id2"],
        element_names=["func1", "func2"],
        size=2,
    )

    _generate_feature_summary(cluster, {"id1": "summary1"}, mock_llm, config)

    call_kwargs = mock_llm.generate_from_messages.call_args[1]
    assert call_kwargs.get("num_ctx") == 16384
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_feature_processor.py::test_generate_feature_summary_uses_aggregation_context -v`
Expected: FAIL

**Step 3: Update FeatureProcessingConfig and _generate_feature_summary**

Add to FeatureProcessingConfig:
```python
aggregation_context_size: int = 16384
```

Update _generate_feature_summary:
```python
raw_summary = llm_client.generate_from_messages(
    messages=messages,
    temperature=config.summarize_temperature,
    top_p=config.summarize_top_p,
    max_tokens=config.summarize_max_tokens,
    timeout=config.summarize_timeout,
    num_ctx=config.aggregation_context_size,  # Add this
)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_feature_processor.py -v`
Expected: PASS

**Step 5: Also update subfeature processing similarly**

Update `_generate_subfeature_summary` to use `num_ctx=config.aggregation_context_size`

**Step 6: Commit**

```bash
git add src/shared/ai/clustering/feature_processor.py tests/test_feature_processor.py
git commit -m "feat(features): use aggregation_context_size for feature/subfeature summaries"
```

---

## Task 10: Add num_ctx to Glossary Processing

**Files:**
- Modify: `src/shared/ai/glossary/ai_extractor.py:282-329` and `477-535`
- Test: `tests/test_glossary_ai_extractor.py`

**Step 1: Update call_llm_for_glossary_sync and generate_glossary_summary_sync**

Both functions should use `config.llm.aggregation_context_size`:

```python
# In call_llm_for_glossary_sync:
response = client.generate_from_messages(
    messages=messages,
    temperature=config.llm.summarize_temperature,
    top_p=config.llm.summarize_top_p,
    max_tokens=128,
    num_ctx=config.llm.aggregation_context_size,  # Add this
)

# In generate_glossary_summary_sync:
response = client.generate_from_messages(
    messages=messages,
    temperature=config.llm.summarize_temperature,
    top_p=config.llm.summarize_top_p,
    max_tokens=512,
    num_ctx=config.llm.aggregation_context_size,  # Add this
)
```

**Step 2: Run existing tests**

Run: `pytest tests/test_glossary_ai_extractor.py -v`
Expected: PASS (no breaking changes)

**Step 3: Commit**

```bash
git add src/shared/ai/glossary/ai_extractor.py
git commit -m "feat(glossary): use aggregation_context_size for glossary extraction"
```

---

## Task 11: Final Integration Test

**Step 1: Run full test suite**

Run: `make check`
Expected: All tests pass, no lint/type errors

**Step 2: Manual end-to-end test**

Run: `magaldi parse /path/to/repo --user test`

Expected:
1. Phase 3 shows context size analysis table
2. Phase 5 uses computed context sizes (check Ollama logs for num_ctx)
3. Features/glossary use aggregation_context_size

**Step 3: Final commit**

```bash
git commit --allow-empty -m "feat: complete context size optimization for KV cache efficiency

- Track max char count per element type during parsing
- Compute optimal num_ctx per type
- Display context size analysis in CLI
- Pass num_ctx to local LLM providers (Ollama, llama.cpp, LM Studio)
- Use fixed aggregation_context_size for features/glossary"
```

---

Plan complete and saved to `docs/plans/2026-01-26-context-size-optimization-implementation.md`.

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
