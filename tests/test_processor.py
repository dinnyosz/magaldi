"""Tests for processor module."""

import pytest
from unittest.mock import MagicMock

from magaldi_core.processor import (
    DependencyTracker,
    ProcessingConfig,
    ProcessingResult,
    TimingStats,
    WorkerStatus,
    ProgressState,
    ProcessedElement,
    _summarize_element,
    _SummaryCache,
)
from magaldi_core.code_parser import CodeElement


def make_element(
    element_id: str,
    element_type: str = "function",
    parent_id: str | None = None,
    level: int = 2,
) -> CodeElement:
    """Create a mock CodeElement for testing."""
    return CodeElement(
        element_id=element_id,
        element_type=element_type,
        name=element_id.split(":")[-1],
        relative_path="file.py",
        line_start=1,
        line_end=10,
        level=level,
        parent_id=parent_id,
        raw_code="def test(): pass",
    )


class TestDependencyTracker:
    """Tests for DependencyTracker class."""

    def test_level_0_elements_always_ready(self):
        """Level 0 elements (files) should always be ready."""
        file_elem = make_element("scope:repo:user:file.py:file:file.py:1", "file", None, 0)
        tracker = DependencyTracker([file_elem])

        ready = tracker.get_ready_elements()
        assert len(ready) == 1
        assert ready[0].element_id == file_elem.element_id

    def test_child_ready_when_parent_completed(self):
        """Child should be ready after parent is marked complete."""
        parent = make_element("scope:repo:user:file.py:class:MyClass:1", "class", None, 1)
        child = make_element(
            "scope:repo:user:file.py:method:my_method:5",
            "method",
            parent.element_id,
            2,
        )
        tracker = DependencyTracker([parent, child])

        # Initially only parent should be ready
        ready = tracker.get_ready_elements()
        assert len(ready) == 1
        assert ready[0].element_id == parent.element_id

        # Mark parent complete
        tracker.mark_complete(parent.element_id)

        # Now child should be ready
        ready = tracker.get_ready_elements()
        assert len(ready) == 1
        assert ready[0].element_id == child.element_id

    def test_child_ready_when_parent_not_in_tracker(self):
        """Child should be ready if parent was skipped (not in elements_to_process).

        This tests the fix for content_hash optimization where unchanged parents
        are not included in the processing list.
        """
        # Parent exists but is NOT in the tracker (was skipped due to unchanged content)
        parent_id = "scope:repo:user:file.py:class:UnchangedClass:1"

        # Child references the parent but parent is not in elements_to_process
        child = make_element(
            "scope:repo:user:file.py:method:changed_method:5",
            "method",
            parent_id,  # References parent that's not in tracker
            2,
        )

        # Only child is in the tracker (parent was unchanged/skipped)
        tracker = DependencyTracker([child])

        # Child should be ready immediately since parent is not in tracker
        ready = tracker.get_ready_elements()
        assert len(ready) == 1
        assert ready[0].element_id == child.element_id

    def test_multiple_children_of_skipped_parent(self):
        """Multiple children should all be ready when parent was skipped."""
        parent_id = "scope:repo:user:file.py:class:SkippedClass:1"

        child1 = make_element(
            "scope:repo:user:file.py:method:method1:5",
            "method",
            parent_id,
            2,
        )
        child2 = make_element(
            "scope:repo:user:file.py:method:method2:15",
            "method",
            parent_id,
            2,
        )

        tracker = DependencyTracker([child1, child2])

        ready = tracker.get_ready_elements(max_count=10)
        assert len(ready) == 2

    def test_mixed_ready_and_waiting(self):
        """Mix of elements with present and absent parents."""
        # Parent in tracker
        present_parent = make_element(
            "scope:repo:user:file.py:class:PresentClass:1",
            "class",
            None,
            1,
        )
        child_of_present = make_element(
            "scope:repo:user:file.py:method:child1:5",
            "method",
            present_parent.element_id,
            2,
        )

        # Parent NOT in tracker (skipped)
        absent_parent_id = "scope:repo:user:file.py:class:AbsentClass:20"
        child_of_absent = make_element(
            "scope:repo:user:file.py:method:child2:25",
            "method",
            absent_parent_id,
            2,
        )

        tracker = DependencyTracker([present_parent, child_of_present, child_of_absent])

        # Initially: present_parent ready, child_of_absent ready (parent skipped)
        ready = tracker.get_ready_elements(max_count=10)
        ready_ids = {e.element_id for e in ready}

        assert present_parent.element_id in ready_ids
        assert child_of_absent.element_id in ready_ids
        assert child_of_present.element_id not in ready_ids  # Must wait for parent

    def test_is_complete(self):
        """is_complete should return True when all elements processed."""
        elem1 = make_element("scope:repo:user:file.py:function:func1:1", "function", None, 2)
        elem2 = make_element("scope:repo:user:file.py:function:func2:10", "function", None, 2)

        tracker = DependencyTracker([elem1, elem2])

        assert not tracker.is_complete()

        tracker.get_ready_elements()
        tracker.mark_complete(elem1.element_id)
        assert not tracker.is_complete()

        tracker.mark_complete(elem2.element_id)
        assert tracker.is_complete()

    def test_mark_failed_unblocks_children(self):
        """Failed parent should still unblock children."""
        parent = make_element("scope:repo:user:file.py:class:MyClass:1", "class", None, 1)
        child = make_element(
            "scope:repo:user:file.py:method:my_method:5",
            "method",
            parent.element_id,
            2,
        )
        tracker = DependencyTracker([parent, child])

        # Get parent
        ready = tracker.get_ready_elements()
        assert len(ready) == 1

        # Mark parent as failed
        tracker.mark_failed(parent.element_id)

        # Child should now be ready
        ready = tracker.get_ready_elements()
        assert len(ready) == 1
        assert ready[0].element_id == child.element_id

    def test_pending_count(self):
        """pending_count should track remaining elements."""
        elements = [
            make_element(f"scope:repo:user:file.py:function:func{i}:1", "function", None, 2)
            for i in range(5)
        ]
        tracker = DependencyTracker(elements)

        assert tracker.pending_count() == 5

        tracker.get_ready_elements()
        tracker.mark_complete(elements[0].element_id)
        assert tracker.pending_count() == 4

        tracker.mark_complete(elements[1].element_id)
        assert tracker.pending_count() == 3


# =============================================================================
# PROCESSING CONFIG TESTS
# =============================================================================


class TestProcessingConfig:
    """Tests for ProcessingConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        from shared.config import ModelConfig

        config = ProcessingConfig()

        # Model configs are now ModelConfig objects
        assert config.summarize_model.name == "qwen3:4b-instruct"
        assert config.summarize_model.provider == "llamacpp"
        assert config.summarize_model_small.name == "qwen3:1.7b"
        assert config.embed_model.name == "snowflake-arctic-embed2"
        assert config.embed_model.provider == "ollama"
        assert config.skip_ai is False
        assert config.num_workers == 4

    def test_custom_values(self):
        """Test custom configuration values."""
        from shared.config import ModelConfig

        config = ProcessingConfig(
            summarize_model=ModelConfig(name="gpt-4", provider="openai", api_key="test-key"),
            num_workers=8,
        )

        assert config.summarize_model.name == "gpt-4"
        assert config.summarize_model.provider == "openai"
        assert config.summarize_model.api_key == "test-key"
        assert config.num_workers == 8

    def test_get_model_for_function(self):
        """Test model selection for function elements."""
        config = ProcessingConfig()

        model = config.get_model_for_element_type("function")
        assert model == config.summarize_model_small

    def test_get_model_for_method(self):
        """Test model selection for method elements."""
        config = ProcessingConfig()

        model = config.get_model_for_element_type("method")
        assert model == config.summarize_model_small

    def test_get_model_for_variable(self):
        """Test model selection for variable elements."""
        config = ProcessingConfig()

        model = config.get_model_for_element_type("variable")
        assert model == config.summarize_model_small

    def test_get_model_for_class(self):
        """Test model selection for class elements."""
        config = ProcessingConfig()

        model = config.get_model_for_element_type("class")
        assert model == config.summarize_model

    def test_get_model_for_file(self):
        """Test model selection for file elements."""
        config = ProcessingConfig()

        model = config.get_model_for_element_type("file")
        assert model == config.summarize_model


# =============================================================================
# PROCESSING RESULT TESTS
# =============================================================================


class TestProcessingResult:
    """Tests for ProcessingResult class."""

    def test_default_values(self):
        """Test default result values."""
        result = ProcessingResult(
            scope="test-scope",
            repository="test-repo",
            username="testuser",
        )

        assert result.scope == "test-scope"
        assert result.repository == "test-repo"
        assert result.username == "testuser"
        assert result.elements_processed == 0
        assert result.elements_skipped == 0
        assert result.elements_failed == 0
        assert result.summarized == 0
        assert result.embedded == 0
        assert result.indexed == 0
        assert result.errors == []
        assert result.failed_elements == []

    def test_with_counts(self):
        """Test result with custom counts."""
        result = ProcessingResult(
            scope="test-scope",
            repository="test-repo",
            username="testuser",
            elements_processed=10,
            elements_skipped=5,
            elements_failed=1,
            summarized=9,
            embedded=9,
            indexed=9,
            errors=["Test error"],
            failed_elements=[("elem1", "Error msg")],
        )

        assert result.elements_processed == 10
        assert result.elements_skipped == 5
        assert result.elements_failed == 1
        assert result.summarized == 9
        assert result.indexed == 9
        assert len(result.errors) == 1
        assert len(result.failed_elements) == 1


# =============================================================================
# TIMING STATS TESTS
# =============================================================================


class TestTimingStats:
    """Tests for TimingStats class."""

    def test_default_values(self):
        """Test default timing stats values."""
        stats = TimingStats()

        assert stats.phase_start == 0.0
        assert stats.total_summarize_by_type == {}
        assert stats.total_embed_by_type == {}

    def test_set_totals_by_type(self):
        """Test setting totals by type."""
        stats = TimingStats()
        stats.set_totals_by_type({"function": 10, "class": 5})

        assert stats.totals_by_type == {"function": 10, "class": 5}
        assert stats.summarize_counts_by_type["function"] == 0
        assert stats.summarize_counts_by_type["class"] == 0

    def test_record_timing(self):
        """Test recording timing for elements."""
        stats = TimingStats()
        stats.set_totals_by_type({"function": 2})

        stats.record(
            wall_time=1.0,
            summarize_time=0.5,
            embed_time=0.3,
            element_type="function",
            was_embedded=True,
        )

        assert stats.total_summarize_by_type["function"] == 0.5
        assert stats.total_embed_by_type["function"] == 0.3
        assert stats.summarize_counts_by_type["function"] == 1
        assert stats.embed_counts_by_type["function"] == 1

    def test_record_without_embedding(self):
        """Test recording timing when element not embedded."""
        stats = TimingStats()
        stats.set_totals_by_type({"function": 2})

        stats.record(
            wall_time=1.0,
            summarize_time=0.5,
            embed_time=0.0,
            element_type="function",
            was_embedded=False,
        )

        assert stats.total_summarize_by_type["function"] == 0.5
        assert stats.total_embed_by_type["function"] == 0.0
        assert stats.embed_counts_by_type["function"] == 0

    def test_total_summarize_count(self):
        """Test total summarize count property."""
        stats = TimingStats()
        stats.set_totals_by_type({"function": 3, "class": 2})

        stats.record(1.0, 0.5, 0.3, "function", True)
        stats.record(1.0, 0.5, 0.3, "function", True)
        stats.record(1.0, 0.5, 0.3, "class", True)

        assert stats.total_summarize_count == 3

    def test_total_embed_count(self):
        """Test total embed count property."""
        stats = TimingStats()
        stats.set_totals_by_type({"function": 3})

        stats.record(1.0, 0.5, 0.3, "function", True)
        stats.record(1.0, 0.5, 0.0, "function", False)
        stats.record(1.0, 0.5, 0.3, "function", True)

        assert stats.total_embed_count == 2

    def test_avg_summarize_time(self):
        """Test average summarize time calculation."""
        stats = TimingStats()
        stats.set_totals_by_type({"function": 2})

        stats.record(1.0, 0.4, 0.2, "function", True)
        stats.record(1.0, 0.6, 0.2, "function", True)

        # Average of 0.4 and 0.6 = 0.5
        assert abs(stats.avg_summarize_time - 0.5) < 0.001

    def test_avg_summarize_time_empty(self):
        """Test average summarize time when no data."""
        stats = TimingStats()

        assert stats.avg_summarize_time == 0.0

    def test_avg_embed_time(self):
        """Test average embed time calculation."""
        stats = TimingStats()
        stats.set_totals_by_type({"function": 2})

        stats.record(1.0, 0.5, 0.3, "function", True)
        stats.record(1.0, 0.5, 0.5, "function", True)

        # Average of 0.3 and 0.5 = 0.4
        assert abs(stats.avg_embed_time - 0.4) < 0.001

    def test_avg_summary_embed_time(self):
        """Test average summary embedding time calculation."""
        stats = TimingStats()
        stats.set_totals_by_type({"function": 2})

        stats.record(1.0, 0.5, 0.5, "function", True, summary_embed_time=0.2, code_embed_time=0.3)
        stats.record(1.0, 0.5, 0.5, "function", True, summary_embed_time=0.4, code_embed_time=0.1)

        # Average of 0.2 and 0.4 = 0.3
        assert abs(stats.avg_summary_embed_time - 0.3) < 0.001

    def test_avg_code_embed_time(self):
        """Test average code embedding time calculation."""
        stats = TimingStats()
        stats.set_totals_by_type({"function": 2})

        stats.record(1.0, 0.5, 0.5, "function", True, summary_embed_time=0.2, code_embed_time=0.3)
        stats.record(1.0, 0.5, 0.5, "function", True, summary_embed_time=0.4, code_embed_time=0.1)

        # Average of 0.3 and 0.1 = 0.2
        assert abs(stats.avg_code_embed_time - 0.2) < 0.001

    def test_avg_embed_times_empty(self):
        """Test average embed times when no data."""
        stats = TimingStats()

        assert stats.avg_summary_embed_time == 0.0
        assert stats.avg_code_embed_time == 0.0

    def test_get_type_stats(self):
        """Test getting per-type statistics."""
        stats = TimingStats()
        stats.set_totals_by_type({"function": 3, "class": 2})

        stats.record(1.0, 0.5, 0.3, "function", True)
        stats.record(1.0, 0.5, 0.3, "function", True)

        type_stats = stats.get_type_stats()

        # function: completed=2, total=3
        assert "function" in type_stats
        completed, total, avg_api, avg_summ, avg_embed = type_stats["function"]
        assert completed == 2
        assert total == 3
        assert avg_summ == 0.5

    def test_eta_seconds_no_data(self):
        """Test ETA calculation with no data."""
        stats = TimingStats()

        assert stats.eta_seconds(0, 10) is None

    def test_eta_seconds_with_data(self):
        """Test ETA calculation with data."""
        stats = TimingStats()
        stats.set_totals_by_type({"function": 10})

        # Process 2 elements, 8 remaining
        stats.record(1.0, 0.5, 0.5, "function", True)
        stats.record(1.0, 0.5, 0.5, "function", True)

        eta = stats.eta_seconds(2, 10, num_workers=1)
        # Remaining: 8 elements * 1.0s avg = 8s
        assert eta is not None
        assert abs(eta - 8.0) < 0.1

    def test_eta_seconds_with_multiple_workers(self):
        """Test ETA calculation with multiple workers."""
        stats = TimingStats()
        stats.set_totals_by_type({"function": 10})

        stats.record(1.0, 0.5, 0.5, "function", True)
        stats.record(1.0, 0.5, 0.5, "function", True)

        eta = stats.eta_seconds(2, 10, num_workers=2)
        # Remaining: 8 elements * 1.0s avg / 2 workers = 4s
        assert eta is not None
        assert abs(eta - 4.0) < 0.1


# =============================================================================
# WORKER STATUS TESTS
# =============================================================================


class TestWorkerStatus:
    """Tests for WorkerStatus class."""

    def test_default_empty(self):
        """Test default worker status is empty."""
        status = WorkerStatus()
        assert status.get_all() == {}

    def test_set_and_get(self):
        """Test setting and getting worker status."""
        status = WorkerStatus()

        status.set(0, "element1", "summarizing", "gpt-4")

        all_status = status.get_all()
        assert 0 in all_status
        # 5-tuple: (element_name, stage, model, ctx_size, start_time)
        elem, stage, model, ctx_size, start_time = all_status[0]
        assert elem == "element1"
        assert stage == "summarizing"
        assert model == "gpt-4"

    def test_set_multiple_workers(self):
        """Test setting multiple worker statuses."""
        status = WorkerStatus()

        status.set(0, "elem1", "summarizing", "model1")
        status.set(1, "elem2", "embedding", "model2")
        status.set(2, "elem3", "indexing", "")

        all_status = status.get_all()
        assert len(all_status) == 3

    def test_clear(self):
        """Test clearing worker status."""
        status = WorkerStatus()

        status.set(0, "element1", "summarizing", "model")
        status.clear(0)

        assert 0 not in status.get_all()

    def test_clear_nonexistent(self):
        """Test clearing nonexistent worker doesn't fail."""
        status = WorkerStatus()

        # Should not raise
        status.clear(999)

    def test_update_existing(self):
        """Test updating existing worker status."""
        status = WorkerStatus()

        status.set(0, "elem1", "summarizing", "model")
        status.set(0, "elem1", "embedding", "model")

        all_status = status.get_all()
        # 5-tuple: (element_name, stage, model, ctx_size, start_time)
        elem, stage, model, ctx_size, start_time = all_status[0]
        assert elem == "elem1"
        assert stage == "embedding"
        assert model == "model"


# =============================================================================
# PROGRESS STATE TESTS
# =============================================================================


class TestProgressState:
    """Tests for ProgressState class."""

    def test_creation(self):
        """Test creating progress state."""
        timing = TimingStats()
        workers = WorkerStatus()

        state = ProgressState(
            total=100,
            completed=50,
            skipped=10,
            failed=2,
            timing=timing,
            workers=workers,
            num_workers=4,
        )

        assert state.total == 100
        assert state.completed == 50
        assert state.skipped == 10
        assert state.failed == 2
        assert state.num_workers == 4
        assert state.recent_errors == []

    def test_with_errors(self):
        """Test progress state with errors."""
        timing = TimingStats()
        workers = WorkerStatus()

        state = ProgressState(
            total=100,
            completed=50,
            skipped=0,
            failed=1,
            timing=timing,
            workers=workers,
            recent_errors=[("elem1", "Error message")],
        )

        assert len(state.recent_errors) == 1
        assert state.recent_errors[0][0] == "elem1"


# =============================================================================
# PROCESSED ELEMENT TESTS
# =============================================================================


class TestProcessedElement:
    """Tests for ProcessedElement class."""

    def test_successful_element(self):
        """Test creating successful processed element."""
        elem = ProcessedElement(
            element_id="scope:repo:user:file.py:function:test:1",
            success=True,
            wall_time=1.5,
            summarize_time=0.8,
            embed_time=0.5,
        )

        assert elem.success is True
        assert elem.wall_time == 1.5
        assert elem.summarize_time == 0.8
        assert elem.embed_time == 0.5
        assert elem.error is None

    def test_failed_element(self):
        """Test creating failed processed element."""
        elem = ProcessedElement(
            element_id="scope:repo:user:file.py:function:test:1",
            success=False,
            wall_time=0.5,
            summarize_time=0.5,
            embed_time=0.0,
            error="API timeout",
        )

        assert elem.success is False
        assert elem.error == "API timeout"


# =============================================================================
# INDEX ELEMENT TESTS (IMPORTS AND CALLS)
# =============================================================================


class TestIndexElementImportsAndCalls:
    """Tests for _index_element storing imports and calls."""

    def test_file_element_with_imports_stores_imports(self):
        """File elements with imports should have imports stored in ES."""
        from magaldi_core.processor import _index_element
        from magaldi_core.code_parser import Import

        # Create file element with imports
        file_elem = CodeElement(
            element_id="scope:repo:user:file.py:file:file.py:1",
            element_type="file",
            name="file.py",
            relative_path="file.py",
            line_start=1,
            line_end=100,
            level=0,
            raw_code="",
            imports=[
                Import(name="os", module="os", alias=None, line=1),
                Import(name="pd", module="pandas", alias="pd", line=2),
            ],
        )

        # Mock ES repository
        mock_es = MagicMock()

        # Call _index_element
        result = _index_element(
            element=file_elem,
            summary="File summary",
            summary_embedding=None,
            code_embedding=None,
            es_repo=mock_es,
        )

        assert result is True
        mock_es.index_element.assert_called_once()
        mock_es.store_summary.assert_called_once_with(file_elem.element_id, "File summary")

        # Verify store_imports was called with correct data
        mock_es.store_imports.assert_called_once()
        call_args = mock_es.store_imports.call_args
        assert call_args[0][0] == file_elem.element_id
        imports_data = call_args[0][1]
        assert len(imports_data) == 2
        assert imports_data[0] == {"name": "os", "module": "os", "alias": None, "line": 1}
        assert imports_data[1] == {"name": "pd", "module": "pandas", "alias": "pd", "line": 2}

    def test_function_element_with_calls_stores_calls(self):
        """Function elements with calls should have calls stored in ES."""
        from magaldi_core.processor import _index_element
        from magaldi_core.code_parser import Call

        # Create function element with calls
        func_elem = CodeElement(
            element_id="scope:repo:user:file.py:function:my_func:10",
            element_type="function",
            name="my_func",
            relative_path="file.py",
            line_start=10,
            line_end=20,
            level=2,
            raw_code="def my_func(): pass",
            calls=[
                Call(name="helper", receiver=None, line=11, resolved_id=None),
                Call(name="process", receiver="utils", line=12, resolved_id="scope:repo:user:utils.py:function:process:5"),
            ],
        )

        # Mock ES repository
        mock_es = MagicMock()

        # Call _index_element
        result = _index_element(
            element=func_elem,
            summary="Function summary",
            summary_embedding=None,
            code_embedding=None,
            es_repo=mock_es,
        )

        assert result is True

        # Verify store_calls was called with correct data
        mock_es.store_calls.assert_called_once()
        call_args = mock_es.store_calls.call_args
        assert call_args[0][0] == func_elem.element_id
        calls_data = call_args[0][1]
        assert len(calls_data) == 2
        assert calls_data[0] == {"name": "helper", "receiver": None, "line": 11, "resolved_id": None}
        assert calls_data[1] == {
            "name": "process",
            "receiver": "utils",
            "line": 12,
            "resolved_id": "scope:repo:user:utils.py:function:process:5",
        }

    def test_method_element_with_calls_stores_calls(self):
        """Method elements with calls should have calls stored in ES."""
        from magaldi_core.processor import _index_element
        from magaldi_core.code_parser import Call

        # Create method element with calls
        method_elem = CodeElement(
            element_id="scope:repo:user:file.py:method:my_method:15",
            element_type="method",
            name="my_method",
            relative_path="file.py",
            line_start=15,
            line_end=25,
            level=2,
            raw_code="def my_method(self): pass",
            calls=[
                Call(name="_internal", receiver="self", line=16, resolved_id=None),
            ],
        )

        # Mock ES repository
        mock_es = MagicMock()

        # Call _index_element
        result = _index_element(
            element=method_elem,
            summary="Method summary",
            summary_embedding=None,
            code_embedding=None,
            es_repo=mock_es,
        )

        assert result is True

        # Verify store_calls was called
        mock_es.store_calls.assert_called_once()
        calls_data = mock_es.store_calls.call_args[0][1]
        assert len(calls_data) == 1
        assert calls_data[0]["name"] == "_internal"
        assert calls_data[0]["receiver"] == "self"

    def test_element_without_imports_does_not_call_store_imports(self):
        """Elements without imports should not call store_imports."""
        from magaldi_core.processor import _index_element

        # Create file element without imports
        file_elem = CodeElement(
            element_id="scope:repo:user:empty.py:file:empty.py:1",
            element_type="file",
            name="empty.py",
            relative_path="empty.py",
            line_start=1,
            line_end=10,
            level=0,
            raw_code="",
            imports=[],  # Empty imports
        )

        mock_es = MagicMock()

        _index_element(
            element=file_elem,
            summary="Empty file",
            summary_embedding=None,
            code_embedding=None,
            es_repo=mock_es,
        )

        # store_imports should NOT be called when imports is empty
        mock_es.store_imports.assert_not_called()

    def test_element_without_calls_does_not_call_store_calls(self):
        """Elements without calls should not call store_calls."""
        from magaldi_core.processor import _index_element

        # Create function element without calls
        func_elem = CodeElement(
            element_id="scope:repo:user:file.py:function:empty_func:10",
            element_type="function",
            name="empty_func",
            relative_path="file.py",
            line_start=10,
            line_end=15,
            level=2,
            raw_code="def empty_func(): pass",
            calls=[],  # Empty calls
        )

        mock_es = MagicMock()

        _index_element(
            element=func_elem,
            summary="Empty function",
            summary_embedding=None,
            code_embedding=None,
            es_repo=mock_es,
        )

        # store_calls should NOT be called when calls is empty
        mock_es.store_calls.assert_not_called()

    def test_class_element_does_not_store_calls(self):
        """Class elements should not store calls (only function/method do)."""
        from magaldi_core.processor import _index_element
        from magaldi_core.code_parser import Call

        # Create class element (classes shouldn't have calls stored directly)
        class_elem = CodeElement(
            element_id="scope:repo:user:file.py:class:MyClass:5",
            element_type="class",
            name="MyClass",
            relative_path="file.py",
            line_start=5,
            line_end=50,
            level=1,
            raw_code="class MyClass: pass",
            calls=[Call(name="ignored", receiver=None, line=6)],  # Even if present, shouldn't be stored
        )

        mock_es = MagicMock()

        _index_element(
            element=class_elem,
            summary="Class summary",
            summary_embedding=None,
            code_embedding=None,
            es_repo=mock_es,
        )

        # store_calls should NOT be called for class elements
        mock_es.store_calls.assert_not_called()

    def test_non_file_element_does_not_store_imports(self):
        """Non-file elements should not store imports (only file does)."""
        from magaldi_core.processor import _index_element
        from magaldi_core.code_parser import Import

        # Create function element with imports (shouldn't happen normally, but test the guard)
        func_elem = CodeElement(
            element_id="scope:repo:user:file.py:function:my_func:10",
            element_type="function",
            name="my_func",
            relative_path="file.py",
            line_start=10,
            line_end=20,
            level=2,
            raw_code="def my_func(): pass",
            imports=[Import(name="os", module="os", alias=None, line=1)],  # Shouldn't be on function
        )

        mock_es = MagicMock()

        _index_element(
            element=func_elem,
            summary="Function summary",
            summary_embedding=None,
            code_embedding=None,
            es_repo=mock_es,
        )

        # store_imports should NOT be called for non-file elements
        mock_es.store_imports.assert_not_called()


# =============================================================================
# CONTEXT SIZE TESTS
# =============================================================================


class TestPerElementContextSize:
    """Tests for per-element context size computation."""

    def test_small_element_uses_smallest_tier(self):
        """Small elements should use 2048 context tier."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "test summary"

        # Small function: 15 chars = ~4 tokens + 700 overhead = 704 → 2048 tier
        element = CodeElement(
            element_id="test:repo:user:file.py:function:foo:1",
            element_type="function",
            name="foo",
            raw_code="def foo(): pass",
        )

        config = ProcessingConfig()
        cache = _SummaryCache()
        cache.add_element(element)

        _summarize_element(element, cache, mock_llm, config)

        call_kwargs = mock_llm.generate.call_args[1]
        assert call_kwargs.get("num_ctx") == 2048

    def test_medium_element_uses_appropriate_tier(self):
        """Medium elements should use appropriate tier based on size."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "test summary"

        # 8000 chars = 2000 tokens + 700 overhead = 2700 → 4096 tier
        element = CodeElement(
            element_id="test:repo:user:file.py:function:big:1",
            element_type="function",
            name="big",
            raw_code="x" * 8000,
        )

        config = ProcessingConfig()
        cache = _SummaryCache()
        cache.add_element(element)

        _summarize_element(element, cache, mock_llm, config)

        call_kwargs = mock_llm.generate.call_args[1]
        assert call_kwargs.get("num_ctx") == 4096

    def test_large_file_uses_large_tier(self):
        """Large files should use appropriately large context tier."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "test summary"

        # 50000 chars = 12500 tokens + 300 overhead = 12800 → 16384 tier
        element = CodeElement(
            element_id="test:repo:user:file.py:file:file.py:1",
            element_type="file",
            name="file.py",
            raw_code="x" * 50000,
        )

        config = ProcessingConfig()
        cache = _SummaryCache()
        cache.add_element(element)

        _summarize_element(element, cache, mock_llm, config)

        call_kwargs = mock_llm.generate.call_args[1]
        assert call_kwargs.get("num_ctx") == 16384

    def test_element_type_affects_overhead(self):
        """Different element types have different prompt overheads."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "test summary"

        # 6000 chars with different types:
        # - function: 6000/4=1500 + 700 overhead = 2200 → 4096 tier
        # - file: 6000/4=1500 + 300 overhead = 1800 → 2048 tier
        func_element = CodeElement(
            element_id="test:repo:user:file.py:function:func:1",
            element_type="function",
            name="func",
            raw_code="x" * 6000,
        )
        file_element = CodeElement(
            element_id="test:repo:user:file.py:file:file.py:1",
            element_type="file",
            name="file.py",
            raw_code="x" * 6000,
        )

        config = ProcessingConfig()
        cache = _SummaryCache()
        cache.add_element(func_element)
        cache.add_element(file_element)

        _summarize_element(func_element, cache, mock_llm, config)
        func_num_ctx = mock_llm.generate.call_args[1].get("num_ctx")

        _summarize_element(file_element, cache, mock_llm, config)
        file_num_ctx = mock_llm.generate.call_args[1].get("num_ctx")

        assert func_num_ctx == 4096
        assert file_num_ctx == 2048

    def test_status_shows_context_tier(self):
        """Status should display context tier (e.g., 'summ@2K')."""
        from magaldi_core.processor import (
            _process_single_element,
            WorkerStatus,
            _SummaryCache,
        )

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "test summary"
        mock_embed = MagicMock()
        mock_es = MagicMock()

        # Small function should show 2K tier
        small_element = CodeElement(
            element_id="test:repo:user:file.py:function:small:1",
            element_type="function",
            name="small",
            raw_code="def small(): pass",
            relative_path="file.py",
        )

        config = ProcessingConfig()
        config.skip_ai = True  # Skip actual LLM calls
        cache = _SummaryCache()
        cache.add_element(small_element)
        worker_status = WorkerStatus()
        status_updates = []

        def on_status_change():
            status_updates.append(worker_status.get_all().copy())

        _process_single_element(
            element=small_element,
            summary_cache=cache,
            llm_client=mock_llm,
            embed_client=mock_embed,
            config=config,
            file_hashes={},
            element_counts={},
            es_repo=mock_es,
            worker_id=0,
            worker_status=worker_status,
            on_status_change=on_status_change,
        )

        # Check that at least one status update had the context tier
        # Status is now 5-tuple: (element_name, stage, model, ctx_size, start_time)
        summ_updates = [
            s[0] for s in status_updates
            if 0 in s and s[0][1] == "summarizing"
        ]
        assert len(summ_updates) > 0
        # Small function should be 2K tier (shown in ctx_size field, index 3)
        ctx_sizes = [u[3] for u in summ_updates]
        assert "2K" in ctx_sizes
