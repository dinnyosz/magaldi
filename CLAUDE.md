# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Magaldi is an open-source code discovery engine that helps AI agents and developers navigate codebases through intelligent indexing and semantic search. Named after Agustín Magaldi (who helped launch Eva Perón's career).

## Architecture

The system consists of two pipelines:

**Parser Pipeline (Phases 1-4):**
- Discovery → Change Detection → Parsing → Storage
- CLI-based invocation: `magaldi parse /path/to/repo --user <username>`

**AI Processing Pipeline (Phases 5-8):**
- Summarization → Embedding → MCP Server → Web UI

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

## TODOs

- [ ] **LiteLLM warnings**: Check if [issue #11759](https://github.com/BerriAI/litellm/issues/11759) (Pydantic serialization) and unclosed aiohttp session issues are resolved, then remove warning suppression from `src/shared/ai/llm_client.py`
