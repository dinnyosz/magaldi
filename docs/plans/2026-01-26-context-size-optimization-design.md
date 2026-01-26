# Context Size Optimization for KV Cache Efficiency

**Date:** 2026-01-26

## Problem

Summarization currently uses a fixed context size for all elements. Since element types have vastly different prompt sizes (methods are small, files are large), this:
- Prevents effective KV cache reuse across element types
- Wastes memory when processing smaller elements

## Solution

1. **Code elements**: Track max char count per element type during parsing, compute optimal `num_ctx` per type, use that throughout summarization
2. **Features/Glossary**: Use fixed large context (input size depends on cluster size, not predictable at parse time)

## Design

### Code Elements: Dynamic Context Size per Type

**During Phase 3 (Parsing)**: Track max character count per element type across the entire repository.

**After parsing completes**: Compute `num_ctx` per element type based on observed max.

```python
# During parsing - track max per type
max_chars_by_type: dict[str, int] = {}

for element in parsed_elements:
    code_len = len(element.raw_code or "")
    current_max = max_chars_by_type.get(element.element_type, 0)
    max_chars_by_type[element.element_type] = max(current_max, code_len)

# After parsing - compute context sizes
PROMPT_OVERHEAD = {
    "file": 400, "class": 500, "function": 400,
    "method": 450, "variable": 200, "constant": 200,
}
CONTEXT_TIERS = [2048, 4096, 8192, 16384, 32768]

def compute_num_ctx(element_type: str, max_chars: int) -> int:
    estimated_tokens = (max_chars // 4) + PROMPT_OVERHEAD.get(element_type, 300)
    # Find smallest tier that fits
    for tier in CONTEXT_TIERS:
        if estimated_tokens < tier:
            return tier
    return CONTEXT_TIERS[-1]  # Fallback to largest

# Result: one num_ctx per element type
num_ctx_by_type = {
    etype: compute_num_ctx(etype, max_chars)
    for etype, max_chars in max_chars_by_type.items()
}
```

### End of Phase 3 Display

Show computed context sizes for visibility:

```
Phase 3: Parsing complete

Context size analysis (for KV cache optimization):
  Element Type   Max Chars   Est. Tokens   Context Size
  ─────────────────────────────────────────────────────
  file              42,318        10,979         16384
  class             18,456         5,114          8192
  function           6,892         2,123          4096
  method             5,204         1,751          4096
  variable             312           278          2048
  constant             156           239          2048
```

### Data Flow

**Phase 3 (Parsing)** returns context sizes alongside parsed elements:

```python
@dataclass
class ParsingResult:
    # Existing fields
    elements: list[CodeElement]
    files_parsed: int
    errors: list[str]
    # ...

    # New field
    context_sizes: dict[str, int]  # element_type -> num_ctx
```

**Phase 5 (Summarization)** receives and uses context sizes:

```python
def process_summarization(
    elements: list[CodeElement],
    context_sizes: dict[str, int],
    config: SummarizationConfig,
    # ...
) -> SummarizationResult:

    for element_type, num_ctx in context_sizes.items():
        elements_of_type = [e for e in elements if e.element_type == element_type]

        for element in elements_of_type:
            summary = generate_summary(element, num_ctx=num_ctx)
```

### LLM Client: num_ctx Parameter

Pass `num_ctx` to all local inference providers:

| Provider | Parameter | Notes |
|----------|-----------|-------|
| Ollama | `num_ctx` | Native support via LiteLLM |
| llama.cpp | `n_ctx` | Via `extra_body` |
| LM Studio | `n_ctx` | OpenAI-compatible, same as llama.cpp |
| vLLM | `max_model_len` | Server-side config |
| LocalAI | `context_size` | Via `extra_body` |

```python
def generate_from_messages(
    self,
    messages: list[dict[str, str]],
    num_ctx: int | None = None,
    # ... existing params
) -> str:
    kwargs = { ... }

    if num_ctx:
        if self.provider == "ollama":
            kwargs["num_ctx"] = num_ctx
        elif self.provider in ("llamacpp", "lmstudio", "localai"):
            # OpenAI-compatible servers
            kwargs["extra_body"] = kwargs.get("extra_body", {})
            kwargs["extra_body"]["n_ctx"] = num_ctx
```

Cloud providers (OpenAI, Anthropic) ignore `num_ctx` - no KV cache benefit on our end.

### Features & Glossary: Fixed Large Context

For aggregation tasks, input size is unpredictable (depends on cluster size / number of connected features), so use a fixed large context.

**Configuration** - add to `LLMConfig`:

```python
@dataclass
class LLMConfig:
    # ... existing fields ...

    # Context sizes for aggregation tasks (features, glossary)
    aggregation_context_size: int = 16384  # Fixed large context
```

## Files to Modify

| File | Changes |
|------|---------|
| `src/magaldi_core/processor.py` | Track max chars, compute context sizes, display stats |
| `src/shared/config.py` | Add `aggregation_context_size` to `LLMConfig` |
| `src/shared/ai/llm_client.py` | Add `num_ctx` param to `generate` and `generate_from_messages` |
| `src/shared/ai/summarization.py` | Pass `num_ctx` based on element type |
| `src/shared/ai/clustering/feature_processor.py` | Use `aggregation_context_size` |
| `src/shared/ai/glossary/ai_extractor.py` | Use `aggregation_context_size` |
