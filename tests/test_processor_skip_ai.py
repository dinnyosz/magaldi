"""Tests for skip-ai fast path optimizations."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from magaldi_core.code_parser import CodeElement, ParsedFile
from magaldi_core.parsers.base import Call, Import
from magaldi_core.processor import (
    ProcessingConfig,
    ProcessingResult,
    ProgressState,
    _process_skip_ai_fast,
    process_elements,
)

# =============================================================================
# HELPERS
# =============================================================================


def _make_element(
    element_id: str,
    element_type: str = "function",
    name: str = "my_func",
    parent_id: str | None = None,
    level: int = 2,
    relative_path: str = "src/foo.py",
    content_hash: str | None = "abc123",
    raw_code: str = "def my_func(): pass",
    imports: list | None = None,
    calls: list | None = None,
) -> CodeElement:
    """Create a CodeElement for testing."""
    elem = CodeElement(
        element_id=element_id,
        element_type=element_type,
        name=name,
        scope="test",
        repository="repo",
        username="main",
        relative_path=relative_path,
        line_start=1,
        line_end=10,
        level=level,
        parent_id=parent_id,
        raw_code=raw_code,
        content_hash=content_hash,
    )
    if imports:
        elem.imports = imports
    if calls:
        elem.calls = calls
    return elem


def _make_file_element(
    path: str = "src/foo.py",
    content_hash: str = "file_hash_1",
    imports: list | None = None,
) -> CodeElement:
    """Create a file-level CodeElement."""
    return _make_element(
        element_id=f"test:repo:main:{path}:file:{path}:1",
        element_type="file",
        name=path,
        level=0,
        relative_path=path,
        content_hash=content_hash,
        raw_code="# file content",
        imports=imports,
    )


def _make_class_element(
    name: str = "MyClass",
    path: str = "src/foo.py",
    parent_file_id: str | None = None,
) -> CodeElement:
    """Create a class-level CodeElement."""
    return _make_element(
        element_id=f"test:repo:main:{path}:class:{name}:5",
        element_type="class",
        name=name,
        level=1,
        relative_path=path,
        parent_id=parent_file_id,
        content_hash=f"class_{name}_hash",
        raw_code=f"class {name}: pass",
    )


def _make_method_element(
    name: str = "my_method",
    path: str = "src/foo.py",
    parent_class_id: str | None = None,
    calls: list | None = None,
) -> CodeElement:
    """Create a method-level CodeElement."""
    return _make_element(
        element_id=f"test:repo:main:{path}:method:{name}:10",
        element_type="method",
        name=name,
        level=2,
        relative_path=path,
        parent_id=parent_class_id,
        content_hash=f"method_{name}_hash",
        raw_code=f"def {name}(self): pass",
        calls=calls,
    )


def _mock_repo() -> MagicMock:
    """Create a mock Repository with needed methods."""
    repo = MagicMock()
    # get_element_ids_by_file returns empty set (no existing elements)
    repo.get_element_ids_by_file.return_value = set()
    # get_element_processing_state_light returns empty (all new)
    repo.get_element_processing_state_light.return_value = {}
    repo.get_element_processing_state.return_value = {}
    # bulk_buffer context manager
    repo.bulk_buffer.return_value.__enter__ = MagicMock()
    repo.bulk_buffer.return_value.__exit__ = MagicMock(return_value=False)
    # index_element_complete returns True
    repo.index_element_complete.return_value = True
    return repo


def _make_parsed_file(elements: list[CodeElement], path: str = "src/foo.py") -> ParsedFile:
    """Create a ParsedFile from elements."""
    file_info = MagicMock()
    file_info.relative_path = path
    return ParsedFile(file_info=file_info, elements=elements)


# =============================================================================
# TESTS: _process_skip_ai_fast
# =============================================================================


class TestProcessSkipAiFast:
    """Tests for the skip-ai fast path function."""

    def test_basic_indexing(self):
        """Elements are indexed with correct summaries."""
        elem = _make_element(
            "test:repo:main:src/foo.py:function:my_func:1",
            element_type="function",
            name="my_func",
        )
        repo = _mock_repo()
        result = ProcessingResult(scope="test", repository="repo", username="main")

        result = _process_skip_ai_fast(
            elements_to_process=[elem],
            repo=repo,
            file_hashes={"src/foo.py": "fh1"},
            element_counts=None,
            result=result,
            total_found=1,
        )

        assert result.elements_processed == 1
        assert result.indexed == 1
        repo.index_element_complete.assert_called_once()
        call_kwargs = repo.index_element_complete.call_args
        assert call_kwargs[0][1] == "Function: my_func"  # summary

    def test_level_ordering(self):
        """Elements are processed in level order: files → classes → methods."""
        file_elem = _make_file_element()
        class_elem = _make_class_element(
            parent_file_id=file_elem.element_id,
        )
        method_elem = _make_method_element(
            parent_class_id=class_elem.element_id,
        )

        # Submit in reverse order
        repo = _mock_repo()
        result = ProcessingResult(scope="test", repository="repo", username="main")

        _process_skip_ai_fast(
            elements_to_process=[method_elem, class_elem, file_elem],
            repo=repo,
            file_hashes=None,
            element_counts=None,
            result=result,
            total_found=3,
        )

        # Verify order: file (level 0), class (level 1), method (level 2)
        calls = repo.index_element_complete.call_args_list
        assert len(calls) == 3
        assert calls[0][0][0].element_type == "file"
        assert calls[1][0][0].element_type == "class"
        assert calls[2][0][0].element_type == "method"

    def test_single_bulk_op_per_element(self):
        """Each element produces exactly one call to index_element_complete."""
        elements = [
            _make_file_element("a.py"),
            _make_file_element("b.py", content_hash="fh2"),
            _make_element(
                "test:repo:main:a.py:function:func1:5",
                name="func1",
                relative_path="a.py",
            ),
        ]
        repo = _mock_repo()
        result = ProcessingResult(scope="test", repository="repo", username="main")

        _process_skip_ai_fast(
            elements_to_process=elements,
            repo=repo,
            file_hashes=None,
            element_counts=None,
            result=result,
            total_found=3,
        )

        # Exactly 3 calls, one per element
        assert repo.index_element_complete.call_count == 3
        # No calls to the old separate methods
        repo.index_element.assert_not_called()

    def test_file_hash_and_element_count(self):
        """File hash and element count are passed correctly."""
        file_elem = _make_file_element("src/app.py")
        repo = _mock_repo()
        result = ProcessingResult(scope="test", repository="repo", username="main")

        _process_skip_ai_fast(
            elements_to_process=[file_elem],
            repo=repo,
            file_hashes={"src/app.py": "hash_xyz"},
            element_counts={"src/app.py": 5},
            result=result,
            total_found=1,
        )

        call_kwargs = repo.index_element_complete.call_args[1]
        assert call_kwargs["file_hash"] == "hash_xyz"
        assert call_kwargs["element_count"] == 5

    def test_imports_included(self):
        """File elements with imports have them included in the single write."""
        imports = [
            Import(name="os", module="os", alias=None, line=1),
            Import(name="json", module="json", alias=None, line=2),
        ]
        file_elem = _make_file_element("src/app.py", imports=imports)
        repo = _mock_repo()
        result = ProcessingResult(scope="test", repository="repo", username="main")

        _process_skip_ai_fast(
            elements_to_process=[file_elem],
            repo=repo,
            file_hashes=None,
            element_counts=None,
            result=result,
            total_found=1,
        )

        call_kwargs = repo.index_element_complete.call_args[1]
        assert call_kwargs["imports"] is not None
        assert len(call_kwargs["imports"]) == 2
        assert call_kwargs["imports"][0]["name"] == "os"

    def test_calls_included(self):
        """Method elements with calls have them included in the single write."""
        calls = [
            Call(name="process", receiver="self", line=15),
            Call(name="validate", receiver=None, line=16),
        ]
        method_elem = _make_method_element("do_stuff", calls=calls)
        repo = _mock_repo()
        result = ProcessingResult(scope="test", repository="repo", username="main")

        _process_skip_ai_fast(
            elements_to_process=[method_elem],
            repo=repo,
            file_hashes=None,
            element_counts=None,
            result=result,
            total_found=1,
        )

        call_kwargs = repo.index_element_complete.call_args[1]
        assert call_kwargs["calls"] is not None
        assert len(call_kwargs["calls"]) == 2
        assert call_kwargs["calls"][0]["name"] == "process"

    def test_progress_callback(self):
        """Progress callback is called during processing."""
        # Create 600 elements to trigger at least one progress callback (every 500)
        elements = [
            _make_element(
                f"test:repo:main:src/foo.py:function:func_{i}:1",
                name=f"func_{i}",
            )
            for i in range(600)
        ]
        repo = _mock_repo()
        result = ProcessingResult(scope="test", repository="repo", username="main")

        progress_states: list[ProgressState] = []

        def on_progress(state: ProgressState) -> None:
            progress_states.append(state)

        _process_skip_ai_fast(
            elements_to_process=elements,
            repo=repo,
            file_hashes=None,
            element_counts=None,
            result=result,
            total_found=600,
            on_progress=on_progress,
        )

        # Should have at least 2 callbacks: one at 500, one final
        assert len(progress_states) >= 2
        # Final callback should show all completed
        assert progress_states[-1].completed == 600

    def test_summary_format(self):
        """Summaries follow the expected format: 'Type: name'."""
        elements = [
            _make_file_element("app.py"),
            _make_class_element("Router"),
            _make_method_element("handle"),
        ]
        repo = _mock_repo()
        result = ProcessingResult(scope="test", repository="repo", username="main")

        _process_skip_ai_fast(
            elements_to_process=elements,
            repo=repo,
            file_hashes=None,
            element_counts=None,
            result=result,
            total_found=3,
        )

        calls = repo.index_element_complete.call_args_list
        summaries = [c[0][1] for c in calls]
        assert "File: app.py" in summaries
        assert "Class: Router" in summaries
        assert "Method: handle" in summaries

    def test_bulk_buffer_large_size(self):
        """Skip-ai fast path uses a large bulk buffer (2000 ops)."""
        repo = _mock_repo()
        result = ProcessingResult(scope="test", repository="repo", username="main")

        _process_skip_ai_fast(
            elements_to_process=[_make_file_element()],
            repo=repo,
            file_hashes=None,
            element_counts=None,
            result=result,
            total_found=1,
        )

        repo.bulk_buffer.assert_called_once_with(max_count=2000, max_interval=10.0)


# =============================================================================
# TESTS: Fully processed check for skip-ai
# =============================================================================


class TestSkipAiFullyProcessedCheck:
    """Tests for the is_fully_processed fix in skip-ai mode."""

    def test_skip_ai_element_with_summary_is_fully_processed(self):
        """In skip-ai mode, element with summary but no embeddings should be skipped."""
        file_elem = _make_file_element()
        parsed = _make_parsed_file([file_elem], "src/foo.py")

        repo = _mock_repo()
        # Return state with summary but NO embeddings
        repo.get_element_processing_state_light.return_value = {
            file_elem.element_id: {
                "content_hash": file_elem.content_hash,
                "has_summary": True,
                "craft_reason": None,
            }
        }

        config = ProcessingConfig(skip_ai=True)
        result = process_elements(
            parsed_files=[parsed],
            scope="test",
            repository="repo",
            username="main",
            repo=repo,
            config=config,
        )

        # Element should be skipped (not re-processed)
        assert result.elements_skipped == 1
        assert result.elements_processed == 0
        repo.index_element_complete.assert_not_called()

    def test_normal_mode_element_without_embeddings_not_fully_processed(self):
        """In normal mode, element with summary but no embeddings should NOT be skipped."""
        file_elem = _make_file_element()
        parsed = _make_parsed_file([file_elem], "src/foo.py")

        repo = _mock_repo()
        # Return state with summary but NO embeddings (via full mget)
        repo.get_element_processing_state.return_value = {
            file_elem.element_id: {
                "content_hash": file_elem.content_hash,
                "has_summary": True,
                "has_summary_embedding": False,
                "has_code_embedding": False,
                "has_caller_embedding": False,
                "craft_reason": None,
                "summary_embedding": None,
                "code_embedding": None,
                "caller_embedding": None,
            }
        }

        config = ProcessingConfig(skip_ai=False)
        # Normal mode needs LLM clients - mock them
        with patch("magaldi_core.processor.SummarizationLLMClient") as mock_llm, \
             patch("magaldi_core.processor.CodeEmbeddingClient") as mock_embed:
            mock_llm.from_model_config.return_value = MagicMock()
            mock_embed.from_model_config.return_value = MagicMock()
            mock_llm_instance = mock_llm.from_model_config.return_value
            mock_llm_instance.generate.return_value = "A test summary"

            mock_embed_instance = mock_embed.from_model_config.return_value
            mock_embed_instance.embed_single.return_value = [0.1] * 1024

            result = process_elements(
                parsed_files=[parsed],
                scope="test",
                repository="repo",
                username="main",
                repo=repo,
                config=config,
            )

        # Element should NOT be skipped (needs embeddings)
        assert result.elements_skipped == 0


# =============================================================================
# TESTS: Lightweight mget
# =============================================================================


class TestSkipAiLightMget:
    """Tests that skip-ai uses the lightweight mget."""

    def test_skip_ai_uses_light_mget(self):
        """Skip-ai mode should call get_element_processing_state_light."""
        file_elem = _make_file_element()
        parsed = _make_parsed_file([file_elem], "src/foo.py")

        repo = _mock_repo()
        config = ProcessingConfig(skip_ai=True)

        process_elements(
            parsed_files=[parsed],
            scope="test",
            repository="repo",
            username="main",
            repo=repo,
            config=config,
        )

        repo.get_element_processing_state_light.assert_called_once()
        repo.get_element_processing_state.assert_not_called()

    def test_normal_mode_uses_full_mget(self):
        """Normal mode should call get_element_processing_state (full)."""
        file_elem = _make_file_element()
        parsed = _make_parsed_file([file_elem], "src/foo.py")

        repo = _mock_repo()
        config = ProcessingConfig(skip_ai=False)

        with patch("magaldi_core.processor.SummarizationLLMClient") as mock_llm, \
             patch("magaldi_core.processor.CodeEmbeddingClient") as mock_embed:
            mock_llm.from_model_config.return_value = MagicMock()
            mock_embed.from_model_config.return_value = MagicMock()
            mock_llm_instance = mock_llm.from_model_config.return_value
            mock_llm_instance.generate.return_value = "A test summary"

            mock_embed_instance = mock_embed.from_model_config.return_value
            mock_embed_instance.embed_single.return_value = [0.1] * 1024

            process_elements(
                parsed_files=[parsed],
                scope="test",
                repository="repo",
                username="main",
                repo=repo,
                config=config,
            )

        repo.get_element_processing_state.assert_called_once()
        repo.get_element_processing_state_light.assert_not_called()


# =============================================================================
# TESTS: index_element_complete
# =============================================================================


class TestIndexElementComplete:
    """Tests for index_element_complete method."""

    def test_includes_all_fields(self):
        """Document includes summary, imports, calls, file_hash, element_count."""
        from shared.db.repositories.elements import ElementRepository

        base = MagicMock()
        elem_repo = ElementRepository(base)
        mock_buffer = MagicMock()
        elem_repo._bulk_buffer = mock_buffer

        imports = [{"name": "os", "module": "os", "alias": None, "line": 1}]
        calls = [
            {"name": "process", "receiver": None, "line": 10, "resolved_id": None, "category": "unknown"}
        ]

        elem = _make_file_element("src/app.py", imports=[])
        elem_repo.index_element_complete(
            elem,
            "File: src/app.py",
            indexed_at=datetime(2024, 1, 1),
            file_hash="fh_abc",
            element_count=5,
            craft_reason=None,
            imports=imports,
            calls=calls,
        )

        mock_buffer.add_index.assert_called_once()
        doc_id, doc = mock_buffer.add_index.call_args[0]

        # Core fields
        assert doc["element_id"] == elem.element_id
        assert doc["element_type"] == "file"
        assert doc["name"] == "src/app.py"
        assert doc["indexed_at"] == "2024-01-01T00:00:00"

        # Merged fields (would be separate update ops without index_element_complete)
        assert doc["summary"] == "File: src/app.py"
        assert doc["file_hash"] == "fh_abc"
        assert doc["element_count"] == 5
        assert doc["imports"] == imports
        assert doc["calls"] == calls

    def test_single_buffer_operation(self):
        """Only one add_index call, no add_update calls."""
        from shared.db.repositories.elements import ElementRepository

        base = MagicMock()
        elem_repo = ElementRepository(base)
        mock_buffer = MagicMock()
        elem_repo._bulk_buffer = mock_buffer

        elem = _make_element(
            "test:repo:main:foo.py:function:bar:1",
            name="bar",
        )
        elem_repo.index_element_complete(elem, "Function: bar")

        assert mock_buffer.add_index.call_count == 1
        assert mock_buffer.add_update.call_count == 0

    def test_optional_fields_omitted_when_none(self):
        """Optional fields (imports, calls, craft_reason) not in doc when None."""
        from shared.db.repositories.elements import ElementRepository

        base = MagicMock()
        elem_repo = ElementRepository(base)
        mock_buffer = MagicMock()
        elem_repo._bulk_buffer = mock_buffer

        elem = _make_element(
            "test:repo:main:foo.py:function:bar:1",
            name="bar",
        )
        elem_repo.index_element_complete(elem, "Function: bar")

        _, doc = mock_buffer.add_index.call_args[0]
        assert "imports" not in doc
        assert "calls" not in doc
        assert "craft_reason" not in doc
        assert "file_hash" not in doc
        assert "element_count" not in doc


# =============================================================================
# TESTS: Re-run skipping
# =============================================================================


class TestSkipAiRerun:
    """Tests that skip-ai re-runs correctly skip already-processed elements."""

    def test_rerun_all_skipped_when_unchanged(self):
        """On second run with same content, all elements should be skipped."""
        file_elem = _make_file_element()
        method_elem = _make_method_element(parent_class_id=file_elem.element_id)
        parsed = _make_parsed_file([file_elem, method_elem], "src/foo.py")

        repo = _mock_repo()
        # Simulate: both elements already exist with summaries
        repo.get_element_processing_state_light.return_value = {
            file_elem.element_id: {
                "content_hash": file_elem.content_hash,
                "has_summary": True,
                "craft_reason": None,
            },
            method_elem.element_id: {
                "content_hash": method_elem.content_hash,
                "has_summary": True,
                "craft_reason": None,
            },
        }

        config = ProcessingConfig(skip_ai=True)
        result = process_elements(
            parsed_files=[parsed],
            scope="test",
            repository="repo",
            username="main",
            repo=repo,
            config=config,
        )

        assert result.elements_skipped == 2
        assert result.elements_processed == 0
        repo.index_element_complete.assert_not_called()
