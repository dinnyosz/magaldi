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

# Start services (Elasticsearch, Redis, Kibana)
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
make ollama-pull            # qwen3:1.7b, qwen3:4b-instruct, snowflake-arctic-embed2

# Use llama.cpp server for summarization (better batching)
./tools/benchmark-llama-server.sh --model qwen3:4b-instruct
```

## Architecture

The system consists of two pipelines:

**Parser Pipeline (Phases 1-4):**
- Discovery → Change Detection → Parsing → Storage
- CLI: `magaldi parse /path/to/repo --user <username>`

**AI Processing Pipeline (Phases 5-8):**
- Summarization → Embedding → MCP Server → Web UI

### Source Layout

```
src/
├── magaldi_core/      # Parser pipeline (phases 1-4)
├── magaldi_mcp/       # MCP server for Claude Code integration
├── magaldi_web/       # FastAPI web UI
└── shared/            # CLI, config, AI client, common modules
```

### Core Components

| Component | Technology |
|-----------|------------|
| Parser | Tree-sitter with S-expression queries |
| Storage & Search | Elasticsearch 8.11.0 (dense_vector) |
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
- `phase4_storage.md` - Elasticsearch storage, job creation
- `ollama_model_research.md` - Model selection rationale
- `magaldi_project_plan.md` - Complete implementation plan

## Supported Languages

Python, JavaScript, TypeScript, PHP, Rust (via Tree-sitter grammars)

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
- Elasticsearch indices (`DELETE /index-name`)
- Database tables or records
- Redis keys
- Any persistent storage

Always ask the user first before performing destructive operations.

## TODOs

- [ ] **LiteLLM Pydantic warning**: Check if [issue #11759](https://github.com/BerriAI/litellm/issues/11759) (Pydantic serialization) is resolved, then remove warning suppression from `src/shared/ai/llm_client.py`
- [ ] **LiteLLM aiohttp session leak**: Check if [issue #11657](https://github.com/BerriAI/litellm/issues/11657) (Ollama embeddings leak aiohttp sessions causing "Too many open files") is resolved, then remove `DISABLE_AIOHTTP_TRANSPORT` workaround from `src/shared/ai/llm_client.py`
- [ ] **MCP find_files returns empty for elasticsearch**: `mcp__magaldi__find_files(pattern="**/elasticsearch*.py")` returns `[]` but files exist at `src/shared/db/repositories/*.py`. Investigate why pattern matching fails - possible indexing or glob pattern issue.
- [ ] **LiteLLM socket leak with ThreadPoolExecutor**: When processing many elements (~300+), orphaned sockets accumulate causing "Too many open files". Related to [LiteLLM issue #13220](https://github.com/BerriAI/litellm/issues/13220). Potential fix: use thread-local LLM clients instead of shared instance in `processor.py`.
- [ ] **Redis type errors**: Fix mypy type errors in `src/shared/db/redis.py` related to async Redis client return types (`Awaitable[...] | ...` incompatibilities with `json.loads`). The Redis client methods return union types that mypy doesn't handle well.
