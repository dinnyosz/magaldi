# Magaldi Project - TODO

## How to Continue

Say this to Claude Code:

> "Review the plans directory and begin implementing Phase 1 (Discovery) following TDD approach from testing_strategy.md. Start with the test file structure."

## Current Status

**Planning phase complete** - All design documents drafted. Ready for implementation.

## Completed Design Documents

### Core Architecture
- [x] `plans/magaldi_project_plan.md` - Master plan
- [x] `plans/architecture_overview.md` - Architecture overview
- [x] `plans/ollama_model_research.md` - Model selection research
- [x] `plans/phase0_configuration.md` - Central configuration schema

### Parser Pipeline (Phases 1-4)
- [x] `plans/phase1_discovery.md` - Path validation, config loading
- [x] `plans/phase2_change_detection.md` - SHA256 hashing, diff logic
- [x] `plans/phase3_parsing.md` - Tree-sitter extraction
- [x] `plans/phase4_storage.md` - MySQL/ES storage, job creation

### AI Processing Pipeline (Phases 5-8)
- [x] `plans/phase5_summarization.md` - Ollama summarization, hierarchical job workers
- [x] `plans/phase6_embedding.md` - Vector generation, ES updates
- [x] `plans/phase7_mcp_server.md` - MCP tools, Claude Code integration
- [x] `plans/phase8_web_ui.md` - Dashboard, visualization, search UI

### Infrastructure & Testing
- [x] `plans/docker_setup.md` - Docker Compose, Dockerfiles, production config
- [x] `plans/testing_strategy.md` - TDD approach, test fixtures, verification

## Next Actions

1. **Set up development environment**
   - Run `docker-compose up -d` for MySQL, Elasticsearch, Redis, Ollama
   - Pull required Ollama models
   - Create Python virtual environment

2. **Implement Phase 0: Configuration**
   - Create `src/magaldi/config.py` with central config
   - Write tests first (TDD)

3. **Implement Phase 1: Discovery**
   - Path validation, config loading
   - Tests first, then implementation

4. **Continue phase by phase** following testing_strategy.md

## Technology Decisions Made

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Summarization Model | qwen2.5-coder:7b | Best benchmarks, Apache 2.0 license |
| Embedding Model | qwen3-embedding:0.6b | 32K context, 1024 dims, MTEB #1 |
| Database | Percona MySQL 8.0 | Multi-writer, ACID, familiar tooling |
| Vector Store | Elasticsearch 8.11.0 | Hybrid search, dense_vector support |
| Parser | Tree-sitter | Multi-language, fault-tolerant, fast |
