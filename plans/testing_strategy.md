# Magaldi Testing Strategy - TDD Approach

## Overview

Magaldi follows Test-Driven Development (TDD): **tests are written before implementation**. Each phase has defined test cases that must pass before the phase is considered complete.

```
┌─────────────────────────────────────────────────────────────────┐
│                     TDD WORKFLOW                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Write Test          2. Run Test         3. Write Code       │
│  (Red)                  (Fails)             (Green)             │
│                                                                 │
│  ┌─────────┐           ┌─────────┐         ┌─────────┐         │
│  │  TEST   │     →     │  FAIL   │    →    │  PASS   │         │
│  │  FIRST  │           │   ✗     │         │   ✓     │         │
│  └─────────┘           └─────────┘         └─────────┘         │
│                                                   │             │
│                                                   ▼             │
│                                            4. Refactor          │
│                                            (Still Green)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Test Categories

| Category | Purpose | Location |
|----------|---------|----------|
| Unit Tests | Test individual functions/classes | `tests/unit/` |
| Integration Tests | Test phase handoffs | `tests/integration/` |
| End-to-End Tests | Test full workflows | `tests/e2e/` |
| Manual Verification | Human validation | Verification checklist |

---

## Testing Tools

```python
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
markers = [
    "unit: Unit tests (fast, no external deps)",
    "integration: Integration tests (needs DB)",
    "e2e: End-to-end tests (needs all services)",
    "slow: Slow tests",
]

[tool.coverage.run]
source = ["src/magaldi"]
branch = true

[tool.coverage.report]
fail_under = 80
```

---

## Test Fixtures

### Common Test Data

```python
# tests/conftest.py
import pytest
from pathlib import Path
from magaldi.config import MagaldiConfig

# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def test_config():
    """Test configuration with test database."""
    return MagaldiConfig(
        mysql={"host": "localhost", "port": 3307, "database": "magaldi_test"},
        elasticsearch={"url": "http://localhost:9201", "index": "magaldi_test"},
        ollama={"url": "http://localhost:11434"},
    )


@pytest.fixture(autouse=True)
def reset_global_config():
    """Reset global config between tests."""
    from magaldi import config
    config._config = None
    yield
    config._config = None


# =============================================================================
# FILE/REPO FIXTURES
# =============================================================================

@pytest.fixture
def sample_python_file(tmp_path):
    """Create a sample Python file for parsing."""
    content = '''
"""Sample module for testing."""

class AuthService:
    """Authentication service."""

    def authenticate(self, username: str, password: str) -> bool:
        """Validate credentials."""
        return username == "admin" and password == "secret"

    def logout(self) -> None:
        """End session."""
        pass


def hash_password(password: str) -> str:
    """Hash a password."""
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()


API_VERSION = "1.0.0"
'''
    file_path = tmp_path / "auth.py"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_repo(tmp_path):
    """Create a sample repository structure."""
    # Create magaldi.yaml
    (tmp_path / "magaldi.yaml").write_text("scope: test-scope\n")

    # Create source files
    src = tmp_path / "src"
    src.mkdir()

    (src / "main.py").write_text('''
def main():
    """Entry point."""
    print("Hello")
''')

    (src / "utils.py").write_text('''
def helper():
    """Helper function."""
    return 42
''')

    return tmp_path


@pytest.fixture
def sample_magaldi_config(tmp_path):
    """Create sample magaldi.yaml."""
    config_path = tmp_path / "magaldi.yaml"
    config_path.write_text("""
scope: backend-services
name: auth-service
exclude_directories:
  - node_modules
  - __pycache__
""")
    return config_path


# =============================================================================
# DATABASE FIXTURES
# =============================================================================

@pytest.fixture
def db_connection(test_config):
    """Provide test database connection."""
    from magaldi.db import MySQLConnection

    conn = MySQLConnection(test_config.mysql)
    conn.connect()
    yield conn
    conn.close()


@pytest.fixture
def clean_db(db_connection):
    """Clean database before test."""
    db_connection.execute("DELETE FROM code_elements")
    db_connection.execute("DELETE FROM file_states")
    db_connection.execute("DELETE FROM summarization_jobs")
    db_connection.execute("DELETE FROM embedding_jobs")
    yield db_connection


@pytest.fixture
def es_client(test_config):
    """Provide test Elasticsearch client."""
    from magaldi.search import ElasticsearchClient

    client = ElasticsearchClient(test_config.elasticsearch)
    yield client
    # Cleanup: delete test index
    client.indices.delete(index=test_config.elasticsearch.index, ignore=[404])


# =============================================================================
# MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_ollama(mocker):
    """Mock Ollama client for unit tests."""
    mock = mocker.MagicMock()
    mock.generate.return_value = "This is a test summary."
    mock.embed_single.return_value = [0.1] * 1024
    mock.embed_batch.return_value = [[0.1] * 1024]
    return mock


# =============================================================================
# ELEMENT FIXTURES
# =============================================================================

@pytest.fixture
def sample_raw_element():
    """Create sample RawElement for testing."""
    from magaldi.parser.elements import RawElement

    return RawElement(
        element_id="test:repo:main:src/auth.py:function:authenticate:10",
        scope="test",
        repository="repo",
        username="main",
        relative_path="src/auth.py",
        element_type="function",
        name="authenticate",
        language="python",
        line_start=10,
        line_end=15,
        raw_code="def authenticate(username, password):\n    return True",
        signature="def authenticate(username, password)",
        docstring="Validate user credentials.",
        level=2,
        parent_id="test:repo:main:src/auth.py:file:auth.py:1",
        decorators=[],
        visibility="public",
        is_async=False,
    )
```

---

## Phase-by-Phase Test Specifications

### Phase 1: Discovery - Test Cases

```python
# tests/unit/test_phase1_discovery.py

class TestPathValidation:
    """Test path validation logic."""

    def test_valid_directory_accepted(self, tmp_path):
        """Valid directory path should be accepted."""
        # GIVEN a valid directory
        # WHEN validating the path
        # THEN it should return True

    def test_file_path_rejected(self, tmp_path):
        """File path (not directory) should be rejected."""

    def test_nonexistent_path_rejected(self):
        """Non-existent path should be rejected."""

    def test_symlink_ignored(self, tmp_path):
        """Symbolic links should be ignored."""


class TestConfigLoading:
    """Test magaldi.yaml loading."""

    def test_load_valid_config(self, sample_magaldi_config):
        """Valid config file should be parsed."""

    def test_missing_scope_error(self, tmp_path):
        """Config without scope should raise error."""

    def test_missing_config_prompt(self, tmp_path, mocker):
        """Missing config should prompt user to create."""


class TestLanguageDetection:
    """Test file language detection."""

    @pytest.mark.parametrize("extension,expected", [
        (".py", "python"),
        (".js", "javascript"),
        (".ts", "typescript"),
        (".tsx", "typescript"),
        (".rs", "rust"),
        (".php", "php"),
    ])
    def test_language_from_extension(self, extension, expected):
        """File extension should map to correct language."""

    def test_unsupported_extension_skipped(self):
        """Unsupported extensions should be skipped."""


class TestFileEnumeration:
    """Test file enumeration."""

    def test_enumerate_all_files(self, sample_repo):
        """Should find all supported files."""

    def test_exclude_directories(self, sample_repo):
        """Should exclude configured directories."""

    def test_exclude_patterns(self, sample_repo):
        """Should exclude files matching patterns."""
```

### Phase 1: Discovery - Manual Verification

```markdown
## Phase 1 Manual Verification Checklist

Before proceeding to Phase 2, verify:

### Setup
- [ ] Test repository created with sample files
- [ ] magaldi.yaml present with scope defined

### Run
```bash
magaldi parse /path/to/test-repo --user main --dry-run -vv
```

### Verify Output
- [ ] Path validation: Shows "Valid repository path"
- [ ] Config loading: Shows "Loaded config: scope=<your-scope>"
- [ ] Language detection: Lists detected languages
- [ ] File enumeration: Lists all discovered files
- [ ] Exclusions working: node_modules etc. not listed
- [ ] No errors or warnings

### Expected Output Example
```
[Discovery]
Validating path...                    ✓ /path/to/test-repo
Loading configuration...              ✓ scope: backend-services
Detecting languages...                python, typescript
Enumerating files...                  45 files found
  Excluded directories:               3 (node_modules, __pycache__, .git)
  Excluded files:                     2 (*.min.js)

Files to process:
  src/auth/login.py                   python
  src/auth/session.py                 python
  ...
```
```

---

### Phase 2: Change Detection - Test Cases

```python
# tests/unit/test_phase2_change_detection.py

class TestFileHashing:
    """Test SHA256 file hashing."""

    def test_hash_small_file(self, tmp_path):
        """Small file should hash correctly."""

    def test_hash_large_file(self, tmp_path):
        """Large file should hash correctly (streaming)."""

    def test_hash_deterministic(self, tmp_path):
        """Same content should produce same hash."""

    def test_hash_binary_file(self, tmp_path):
        """Binary files should be hashable."""


class TestChangeDetection:
    """Test change detection logic."""

    def test_new_file_detected(self, clean_db, sample_repo):
        """Files not in database are new."""

    def test_modified_file_detected(self, clean_db, sample_repo):
        """Files with different hash are modified."""

    def test_unchanged_file_skipped(self, clean_db, sample_repo):
        """Files with same hash are skipped."""

    def test_deleted_file_detected(self, clean_db, sample_repo):
        """Files in DB but not on disk are deleted."""


class TestUserBranchDiff:
    """Test user branch diffing against main."""

    def test_user_diff_from_main(self, clean_db):
        """User branch should diff against main."""

    def test_main_required_first(self, clean_db):
        """User branch requires main to exist."""
```

### Phase 2: Manual Verification

```markdown
## Phase 2 Manual Verification Checklist

### Setup
- [ ] Phase 1 verified
- [ ] Database running (MySQL)
- [ ] Test files modified since Phase 1

### Run Initial Parse
```bash
magaldi parse /path/to/test-repo --user main -v
```

### Verify Change Detection
- [ ] All files show as "new" on first run
- [ ] Hash computed for each file
- [ ] File states stored in database

### Run Second Parse (no changes)
```bash
magaldi parse /path/to/test-repo --user main -v
```

- [ ] Files show as "unchanged"
- [ ] Parse completes quickly (no re-parsing)

### Modify a File and Re-run
```bash
echo "# comment" >> /path/to/test-repo/src/auth.py
magaldi parse /path/to/test-repo --user main -v
```

- [ ] Modified file detected
- [ ] Only modified file re-parsed

### Verify Database
```sql
SELECT relative_path, file_hash, parsed_at
FROM file_states
WHERE scope = 'your-scope' AND repository = 'your-repo';
```
```

---

### Phase 3: Parsing - Test Cases

```python
# tests/unit/test_phase3_parsing.py

class TestTreeSitterParsing:
    """Test Tree-sitter parsing."""

    def test_parse_python_file(self, sample_python_file):
        """Python file should parse without errors."""

    def test_parse_with_syntax_errors(self, tmp_path):
        """Files with syntax errors should still parse (fault-tolerant)."""

    def test_parse_empty_file(self, tmp_path):
        """Empty files should return empty element list."""


class TestElementExtraction:
    """Test code element extraction."""

    def test_extract_class(self, sample_python_file):
        """Classes should be extracted with metadata."""

    def test_extract_function(self, sample_python_file):
        """Functions should be extracted with signature."""

    def test_extract_method(self, sample_python_file):
        """Methods should be extracted with parent reference."""

    def test_extract_docstring(self, sample_python_file):
        """Docstrings should be captured."""

    def test_extract_decorators(self, tmp_path):
        """Decorators should be captured."""


class TestHierarchyBuilding:
    """Test element hierarchy."""

    def test_file_is_level_0(self, sample_python_file):
        """File element should be level 0."""

    def test_class_is_level_1(self, sample_python_file):
        """Class should be level 1 with file as parent."""

    def test_method_is_level_2(self, sample_python_file):
        """Method should be level 2 with class as parent."""

    def test_element_id_format(self, sample_python_file):
        """Element ID should follow format."""


class TestMultiLanguage:
    """Test multi-language support."""

    @pytest.mark.parametrize("language", ["python", "javascript", "typescript", "rust", "php"])
    def test_language_parsing(self, language, tmp_path):
        """Each supported language should parse."""
```

### Phase 3: Manual Verification

```markdown
## Phase 3 Manual Verification Checklist

### Run Parse
```bash
magaldi parse /path/to/test-repo --user main -vv
```

### Verify Element Extraction
- [ ] Files extracted (level 0)
- [ ] Classes extracted (level 1)
- [ ] Functions/methods extracted (level 2)
- [ ] Signatures captured correctly
- [ ] Docstrings captured
- [ ] Line numbers correct

### Verify Database
```sql
SELECT element_id, element_type, name, line_start, level, parent_id
FROM code_elements
WHERE scope = 'your-scope'
ORDER BY relative_path, line_start;
```

### Spot Check
- [ ] Pick a file, compare extracted elements to actual code
- [ ] Verify parent-child relationships are correct
- [ ] Verify element IDs are unique
```

---

### Phase 4: Storage - Test Cases

```python
# tests/unit/test_phase4_storage.py

class TestMySQLStorage:
    """Test MySQL storage operations."""

    def test_store_file_state(self, clean_db, sample_raw_element):
        """File state should be stored correctly."""

    def test_store_elements(self, clean_db, sample_raw_element):
        """Elements should be stored correctly."""

    def test_upsert_on_duplicate(self, clean_db, sample_raw_element):
        """Duplicate elements should update, not fail."""

    def test_transaction_rollback(self, clean_db, mocker):
        """Failed storage should rollback transaction."""


class TestElasticsearchIndexing:
    """Test Elasticsearch indexing."""

    def test_index_element(self, es_client, sample_raw_element):
        """Element should be indexed in ES."""

    def test_bulk_index(self, es_client):
        """Bulk indexing should work."""

    def test_index_mapping(self, es_client):
        """Index should have correct mapping."""


class TestJobCreation:
    """Test job creation for AI processing."""

    def test_create_summarization_jobs(self, clean_db, sample_raw_element):
        """Summarization jobs should be created."""

    def test_level_0_dependencies_met(self, clean_db):
        """Level 0 jobs should have dependencies_met=True."""

    def test_level_1_dependencies_not_met(self, clean_db):
        """Level 1+ jobs should have dependencies_met=False."""
```

### Phase 4: Manual Verification

```markdown
## Phase 4 Manual Verification Checklist

### Verify MySQL Storage
```sql
-- Check file states
SELECT COUNT(*) FROM file_states WHERE scope = 'your-scope';

-- Check elements
SELECT element_type, COUNT(*)
FROM code_elements
WHERE scope = 'your-scope'
GROUP BY element_type;

-- Check jobs created
SELECT level, status, COUNT(*)
FROM summarization_jobs
GROUP BY level, status;
```

### Verify Elasticsearch
```bash
curl -X GET "localhost:9200/magaldi_code_elements/_count"
curl -X GET "localhost:9200/magaldi_code_elements/_search?size=1"
```

### Verify Jobs
- [ ] Summarization jobs created for all elements
- [ ] Level 0 (file) jobs have dependencies_met=TRUE
- [ ] Level 1+ jobs have dependencies_met=FALSE
```

---

### Phase 5: Summarization - Test Cases

```python
# tests/unit/test_phase5_summarization.py

class TestPromptBuilding:
    """Test prompt construction."""

    def test_file_prompt(self, sample_raw_element):
        """File prompt should include code."""

    def test_function_prompt_with_context(self, sample_raw_element):
        """Function prompt should include parent summary."""

    def test_code_truncation(self):
        """Long code should be truncated."""


class TestSummaryGeneration:
    """Test summary generation (with mocked Ollama)."""

    def test_generate_summary(self, mock_ollama, sample_raw_element):
        """Summary should be generated."""

    def test_clean_summary(self):
        """Summary should be cleaned (remove prefixes, etc)."""


class TestDependencyResolution:
    """Test hierarchical processing."""

    def test_level_0_processed_first(self, clean_db):
        """Level 0 should be processed before level 1."""

    def test_dependencies_unlock(self, clean_db):
        """Completing parent should unlock children."""


class TestWorkerPool:
    """Test worker pool behavior."""

    def test_job_claiming(self, clean_db):
        """Workers should claim jobs atomically."""

    def test_stale_job_reclaim(self, clean_db):
        """Stale jobs should be reclaimed."""
```

### Phase 5: Manual Verification

```markdown
## Phase 5 Manual Verification Checklist

### Prerequisites
- [ ] Ollama running with qwen2.5-coder:7b
- [ ] Phase 4 completed (jobs exist)

### Start Summarization
```bash
magaldi summarize --scope your-scope --repo your-repo --user main -v
```

### Verify Progress
- [ ] Level 0 (files) processed first
- [ ] Level 1 (classes) processed after level 0
- [ ] Level 2 (functions) processed after level 1

### Verify Summaries
```sql
SELECT name, element_type, LEFT(summary, 100) as summary_preview
FROM code_elements
WHERE scope = 'your-scope' AND summary IS NOT NULL
LIMIT 10;
```

### Spot Check Quality
- [ ] Pick 3 summaries, verify they're accurate
- [ ] Summaries are 1-3 sentences
- [ ] No hallucinations or incorrect information
```

---

### Phase 6: Embedding - Test Cases

```python
# tests/unit/test_phase6_embedding.py

class TestContextBuilding:
    """Test embedding context construction."""

    def test_file_context(self, sample_raw_element):
        """File context should be minimal."""

    def test_function_with_class_context(self, sample_raw_element):
        """Function should include file + class context."""

    def test_context_token_estimation(self):
        """Token estimation should be reasonable."""


class TestEmbeddingGeneration:
    """Test embedding generation (with mocked Ollama)."""

    def test_generate_embedding(self, mock_ollama):
        """Embedding should be generated."""

    def test_embedding_dimensions(self, mock_ollama):
        """Embedding should have correct dimensions."""

    def test_batch_embedding(self, mock_ollama):
        """Batch embedding should work."""


class TestVectorStorage:
    """Test vector storage in Elasticsearch."""

    def test_store_vector(self, es_client):
        """Vector should be stored in ES."""

    def test_bulk_vector_update(self, es_client):
        """Bulk vector update should work."""
```

### Phase 6: Manual Verification

```markdown
## Phase 6 Manual Verification Checklist

### Prerequisites
- [ ] Ollama running with snowflake-arctic-embed2
- [ ] Phase 5 completed (summaries exist)

### Start Embedding
```bash
magaldi embed --scope your-scope --repo your-repo --user main -v
```

### Verify Vectors
```bash
# Check documents have embeddings
curl -X GET "localhost:9200/magaldi_code_elements/_search" -H 'Content-Type: application/json' -d'
{
  "query": { "exists": { "field": "embedding" } },
  "_source": ["element_id", "name"],
  "size": 5
}'
```

### Test Semantic Search
```bash
# This should return relevant results
curl -X POST "localhost:9200/magaldi_code_elements/_search" -H 'Content-Type: application/json' -d'
{
  "query": {
    "script_score": {
      "query": { "match_all": {} },
      "script": {
        "source": "cosineSimilarity(params.qv, '\''embedding'\'') + 1.0",
        "params": { "qv": [/* your test query embedding */] }
      }
    }
  }
}'
```

### Verify Coverage
```sql
SELECT
  COUNT(*) as total,
  SUM(CASE WHEN embedding_status = 'completed' THEN 1 ELSE 0 END) as embedded
FROM code_elements
WHERE scope = 'your-scope';
```
```

---

### Phase 7 & 8: Integration Tests

```python
# tests/integration/test_mcp_server.py

class TestMCPSearchTool:
    """Test MCP search_code tool."""

    def test_search_returns_results(self, populated_index):
        """Search should return relevant results."""

    def test_search_with_filters(self, populated_index):
        """Filters should narrow results."""

    def test_search_ranking(self, populated_index):
        """Results should be ranked by relevance."""


# tests/integration/test_web_api.py

class TestSearchAPI:
    """Test web search API."""

    def test_search_endpoint(self, client, populated_index):
        """POST /api/v1/search should work."""

    def test_search_with_filters(self, client, populated_index):
        """Filters should work."""


class TestBrowserAPI:
    """Test file browser API."""

    def test_file_tree(self, client, populated_db):
        """GET /api/v1/repos/{scope}/{repo}/tree should work."""

    def test_file_detail(self, client, populated_db):
        """GET /api/v1/repos/{scope}/{repo}/files/{path} should work."""
```

---

## Running Tests

```bash
# Run all unit tests (fast)
pytest tests/unit -v

# Run with coverage
pytest tests/unit --cov=src/magaldi --cov-report=html

# Run integration tests (needs services)
pytest tests/integration -v

# Run specific phase tests
pytest tests/unit/test_phase1*.py -v

# Run marked tests
pytest -m "not slow" -v

# Run single test
pytest tests/unit/test_phase1_discovery.py::TestPathValidation::test_valid_directory_accepted -v
```

---

## CI Pipeline

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: pytest tests/unit --cov=src/magaldi --cov-fail-under=80

  integration-tests:
    runs-on: ubuntu-latest
    services:
      mysql:
        image: percona:8.0
        env:
          MYSQL_ROOT_PASSWORD: test
          MYSQL_DATABASE: magaldi_test
        ports:
          - 3307:3306
      elasticsearch:
        image: elasticsearch:8.11.0
        env:
          discovery.type: single-node
          xpack.security.enabled: false
        ports:
          - 9201:9200
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -e ".[dev]"
      - run: pytest tests/integration -v
```

---

## Summary

| Phase | Unit Tests | Integration | Manual Verification |
|-------|-----------|-------------|---------------------|
| Phase 0 | Config loading | - | Config file works |
| Phase 1 | Path, config, enum | - | Files discovered |
| Phase 2 | Hashing, diffing | DB storage | Changes detected |
| Phase 3 | Parsing, hierarchy | - | Elements extracted |
| Phase 4 | MySQL, ES | DB + ES | Data persisted |
| Phase 5 | Prompts, workers | Ollama | Summaries quality |
| Phase 6 | Context, vectors | Ollama + ES | Search works |
| Phase 7 | Tool handlers | Full stack | MCP tools work |
| Phase 8 | API handlers | Full stack | UI works |
