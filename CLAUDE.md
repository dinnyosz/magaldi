# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Magaldi is an open-source code discovery engine that helps AI agents and developers navigate codebases through intelligent indexing and semantic search. Named after Agustín Magaldi (who helped launch Eva Perón's career).

## Development Commands

```bash
# Setup
make setup                  # Create venv and install dependencies
source .venv/bin/activate   # Activate virtual environment
pip install -e .            # Install magaldi CLI (required after code changes)

# Start services (OpenSearch, Redis)
make services               # Core services only
make services-full          # Include Ollama
make services-down          # Stop all

# Testing
make test                   # Run all tests
make test-fast              # Skip slow and integration tests
make test-integration       # Integration tests only (needs services)
pytest tests/test_foo.py::test_bar -v  # Single test

# Code quality
make lint                   # Run ruff linter
make format                 # Format with ruff
make typecheck              # Run mypy on src/magaldi
make check                  # All: lint + typecheck + test

# Pull Ollama models (for AI features)
make ollama-pull            # qwen3.5:2b, qwen3.5:4b, qwen3-embedding:0.6b

# Use llama.cpp server for summarization (better batching)
./tools/benchmark-llama-server.sh --model qwen3.5:4b
```

## Architecture

The system consists of two pipelines:

**Parser Pipeline (Phases 1-5):**
- Discovery → Change Detection → Parsing → Variable Scoring → Storage
- CLI: `magaldi parse /path/to/repo --user <username>`

**AI Processing Pipeline (Phases 6-9):**
- Summarization → Embedding → MCP Server → Web UI

### Source Layout

```
src/
├── magaldi_core/      # Parser pipeline (phases 1-5)
├── magaldi_mcp/       # MCP server for Claude Code integration
├── magaldi_web/       # FastAPI web UI
└── shared/            # CLI, config, AI client, common modules
```

### Core Components

| Component | Technology |
|-----------|------------|
| Parser | Tree-sitter with S-expression queries |
| Storage & Search | OpenSearch 2.19.0 (knn_vector + Faiss HNSW) |
| Job Queue | Redis |
| AI Models | LiteLLM (supports Ollama, OpenAI, Anthropic, etc.) |
| MCP | Python MCP SDK for Claude Code integration |

### Key Data Model

**Element ID format:** `{scope}:{repository}:{username}:{relative_path}:{type}:{name}:{line}`

**Multi-user model:**
- `main` branch: Full parse by CI/central system
- User branches: Diff from main only, auto-expires after 30 days

**Hierarchy levels:**
- Level 0: File
- Level 1: Class
- Level 2: Function/Method
- Level 3: Variable

### Test Markers

```python
@pytest.mark.slow          # Exclude with: -m "not slow"
@pytest.mark.integration   # Tests requiring external services
```

## Planning Documents

All design documents are in `plans/`:
- `architecture_overview.md` - High-level architecture and status
- `phase1_discovery.md` - Path validation, config loading
- `phase2_change_detection.md` - SHA256 hashing, diff logic
- `phase3_parsing.md` - Tree-sitter extraction
- `phase4_storage.md` - Search backend storage, job creation
- `ollama_model_research.md` - Model selection rationale
- `magaldi_project_plan.md` - Complete implementation plan

## Supported Languages

Python, JavaScript, TypeScript, PHP, Rust, Java, Bash (via Tree-sitter grammars)

## Configuration

Repository config file: `magaldi.yaml` in repo root with required `scope` field.

Example:
```yaml
scope: myorg
repository: myrepo
```

Use `mcp__magaldi__generate_config(repo_path="/path/to/repo")` to auto-generate this file.

## MCP Integration

**See `.claude/skills/magaldi/SKILL.md` for detailed MCP tool usage guidance.**

Magaldi MCP tools **auto-detect `scope` and `repository`** from `magaldi.yaml` in the current directory. No need to specify these parameters manually - just call the tools directly.

## CRITICAL: Implementation Checklist

**BEFORE and AFTER implementing any of these changes, you MUST run `/check-magaldi-integrity`:**

- Adding new element types
- Extracting new metadata from code
- Modifying summarization prompts
- Adding new fields to `CodeElement`

The skill ensures:
1. **No gaps**: Every extracted data point surfaces in Web UI, MCP tools, AND summarization
2. **Anti-verbose prompts**: All summaries start with action/content, never "This X is..."
3. **Token efficiency**: Context added to prompts is minimal and conditional
4. **Source references**: No hardcoded values, always reference source files

## CRITICAL: Data Safety

**NEVER delete or drop data without explicit user permission.** This includes:
- Search indices (`DELETE /index-name`)
- Database tables or records
- Redis keys
- Any persistent storage

Always ask the user first before performing destructive operations.

## TODOs

- [ ] **LiteLLM Pydantic warning**: Check if [issue #11759](https://github.com/BerriAI/litellm/issues/11759) (Pydantic serialization) is resolved, then remove warning suppression from `src/shared/ai/llm_client.py`
- [ ] **MCP find_files returns empty for elasticsearch**: `mcp__magaldi__find_files(pattern="**/elasticsearch*.py")` returns `[]` but files exist at `src/shared/db/repositories/*.py`. Investigate why pattern matching fails - possible indexing or glob pattern issue.
- [ ] **LiteLLM socket leak with ThreadPoolExecutor**: When processing many elements (~300+), orphaned sockets accumulate causing "Too many open files". Related to [LiteLLM issue #13220](https://github.com/BerriAI/litellm/issues/13220). Potential fix: use thread-local LLM clients instead of shared instance in `processor.py`.
- [ ] **Duplicated provider→LiteLLM mapping**: `ModelConfig.get_litellm_model()` / `get_api_base()` in `config.py` is the source of truth, but the same provider-to-prefix mapping is duplicated in `SummarizationLLMClient.__init__()`, `_resolve_model_override()`, `SummarizationEmbeddingClient.__init__()`, and `LLMClient._build_kwargs()`. Adding a new provider (e.g. `lmstudio`) requires updating all of them. Refactor these to delegate to `ModelConfig` methods, or have `from_model_config()` pass pre-resolved `litellm_model` + `api_base` instead of raw provider strings.
- [ ] **Task tool `model` parameter instability**: Works in Claude Code v2.1.72 but has regressed before (v2.1.69–v2.1.71 returned 404). Monitor [issue #18873](https://github.com/anthropics/claude-code/issues/18873) for regressions. All skills now specify `model:` on subagent launches — if it breaks again, removing the param falls back to parent model.
- [ ] **Per-subagent effort/reasoning not yet supported**: The Task tool has no `reasoning_effort` or `thinking_budget` parameter. Track [issue #14321](https://github.com/anthropics/claude-code/issues/14321) — when it ships, update all skills to set effort levels (haiku tasks → low, sonnet → medium, opus → high).
- [ ] **Explore subagent ignores `model` parameter**: Explore agents inherit the parent model instead of using the specified one (e.g. haiku). Track [issue #29768](https://github.com/anthropics/claude-code/issues/29768) — avoid specifying `model: "haiku"` on Explore agents until fixed.
- [ ] **Expand variable scoring dimensions (4 → 7)**: When swapping in the fine-tuned `magaldi-scorer` model, add three new scoring dimensions. The training data can already include all 7 dimensions so the fine-tuned model learns them from day one. New dimensions:
  - `value_complexity`: How complex/interesting the assigned value is (e.g. `x = 3` → 1, `CORS_ORIGINS = ["http://localhost:3000", "https://myapp.com"]` → 8). Differentiates variables with generic names but meaningful values.
  - `naming_quality`: How descriptive/self-documenting the name is (`db_connection_pool` → 9, `conn` → 3, `x` → 1). Helps borderline cases where code is a generic call but the name reveals intent.
  - `scope_significance`: Module-level vs function-local vs class-level significance (`MODULE_CONSTANT` → 9, class attribute → 6, function local → 2). The parser already knows hierarchy level — this teaches the model to weight it.
  - Requires updating: `SYSTEM_PROMPT` in `prompts.py`, `VariableScore` dataclass in `models.py`, `_parse_scores` regex (7 numbers instead of 4), `max_score`/`as_tuple()`, `compare_scoring_models.py`, and all tests.
