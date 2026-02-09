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
)
from magaldi_core.job_tracker import SummaryCache
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

    def test_three_level_hierarchy_file_class_method(self):
        """Method must wait for class, class must wait for file.

        Hierarchy: file → class → method
        Method cannot be ready until class is done.
        Class cannot be ready until file is done.
        """
        file_elem = make_element(
            "scope:repo:user:file.py:file:file.py:1",
            "file",
            None,
            0,
        )
        class_elem = make_element(
            "scope:repo:user:file.py:class:MyClass:10",
            "class",
            file_elem.element_id,  # parent is file
            1,
        )
        method_elem = make_element(
            "scope:repo:user:file.py:method:my_method:20",
            "method",
            class_elem.element_id,  # parent is class
            2,
        )

        tracker = DependencyTracker([file_elem, class_elem, method_elem])

        # Step 1: Only file should be ready (no parent)
        ready = tracker.get_ready_elements(max_count=10)
        assert len(ready) == 1
        assert ready[0].element_id == file_elem.element_id
        tracker.mark_complete(file_elem.element_id)

        # Step 2: Now class should be ready (file done), but not method
        ready = tracker.get_ready_elements(max_count=10)
        assert len(ready) == 1
        assert ready[0].element_id == class_elem.element_id
        tracker.mark_complete(class_elem.element_id)

        # Step 3: Now method should be ready (class done)
        ready = tracker.get_ready_elements(max_count=10)
        assert len(ready) == 1
        assert ready[0].element_id == method_elem.element_id

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
        """Multiple children should all be ready when parent was skipped.

        Note: Due to tier-based batching with warmup, the first call returns 1 element
        (warmup task). After warmup completes, subsequent calls return all ready elements.
        """
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

        # First call during tier warmup returns 1 element
        ready = tracker.get_ready_elements(max_count=10)
        assert len(ready) == 1

        # Mark warmup task complete
        tracker.mark_complete(ready[0].element_id)

        # Second call returns the remaining element
        ready = tracker.get_ready_elements(max_count=10)
        assert len(ready) == 1

    def test_mixed_ready_and_waiting(self):
        """Mix of elements with present and absent parents.

        Due to model+tier batching:
        - Classes use 'large' model, methods use 'small' model
        - Different models require drain and warmup
        """
        # Parent in tracker (class = large model)
        present_parent = make_element(
            "scope:repo:user:file.py:class:PresentClass:1",
            "class",
            None,
            1,
        )
        # Methods = small model
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

        # First call: class uses large model, warmup returns 1
        ready = tracker.get_ready_elements(max_count=10)
        assert len(ready) == 1
        assert ready[0].element_id == present_parent.element_id

        # Mark parent complete
        tracker.mark_complete(present_parent.element_id)

        # Second call: switching to small model (methods), warmup returns 1
        ready = tracker.get_ready_elements(max_count=10)
        assert len(ready) == 1  # Warmup for model switch
        first_method = ready[0]
        tracker.mark_complete(first_method.element_id)

        # Third call: same model, returns remaining method
        ready = tracker.get_ready_elements(max_count=10)
        assert len(ready) == 1
        # The other method should be returned
        assert ready[0].element_id != first_method.element_id

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
        assert config.embed_model.name == "qwen3-embedding:0.6b"
        assert config.embed_model.provider == "ollama"
        assert config.skip_ai is False
        assert config.num_workers == 0  # 0 = auto (use tier-based defaults)

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
        # Use tier data for throughput-normalized ETA
        stats.set_totals_by_type_tier({("function", 2048): 10})

        # Simulate 2 workers: each element takes 1.0s wall time with 2 workers active
        # base_time = 1.0 / 2 = 0.5s per element (throughput-normalized)
        stats.record(1.0, 0.5, 0.5, "function", True, tier=2048, avg_workers=2.0)
        stats.record(1.0, 0.5, 0.5, "function", True, tier=2048, avg_workers=2.0)

        eta = stats.eta_seconds(2, 10, num_workers=2)
        # Remaining: 8 elements * 0.5s avg (already throughput-normalized) = 4s
        assert eta is not None
        assert abs(eta - 4.0) < 0.1

    def test_eta_seconds_with_tier_data(self):
        """Test tier-aware ETA calculation."""
        stats = TimingStats()

        # Set up totals by (type, tier)
        stats.set_totals_by_type_tier({
            ("function", 2048): 5,
            ("function", 8192): 5,
        })

        # Record timing for 2k tier (fast) and 8k tier (slow)
        # wall_time is what's used for ETA, not summarize_time
        stats.record(0.5, 0.3, 0.2, "function", True, tier=2048)
        stats.record(0.5, 0.3, 0.2, "function", True, tier=2048)
        stats.record(2.0, 1.5, 0.5, "function", True, tier=8192)

        eta = stats.eta_seconds(3, 10, num_workers=1)
        # avg_2k = 0.5s (wall_time), avg_8k = 2.0s (wall_time)
        # Remaining: 3 @ 2k + 4 @ 8k
        # = 3 * 0.5 + 4 * 2.0 = 1.5 + 8.0 = 9.5s
        assert eta is not None
        assert abs(eta - 9.5) < 0.2

    def test_eta_seconds_tier_fallback_to_same_type(self):
        """Test ETA fallback to same type when tier not found."""
        stats = TimingStats()

        # Set up totals with a tier we don't have data for
        stats.set_totals_by_type_tier({
            ("function", 2048): 3,
            ("function", 4096): 3,  # No data for this tier
        })

        # Only record data for 2k tier (wall_time=1.0)
        stats.record(1.0, 0.6, 0.4, "function", True, tier=2048)
        stats.record(1.0, 0.6, 0.4, "function", True, tier=2048)

        eta = stats.eta_seconds(2, 6, num_workers=1)
        # avg_2k = 1.0s (wall_time)
        # 4k tier fallback: 1.0 * (4096/2048)^0.65 = 1.0 * 2^0.65 ≈ 1.569s
        # Remaining: 1 @ 2k + 3 @ 4k
        # = 1 * 1.0 + 3 * 1.569 ≈ 5.71s
        assert eta is not None
        assert abs(eta - 5.71) < 0.1

    def test_eta_seconds_tier_fallback_to_same_model(self):
        """Test ETA fallback to same model group when type not found."""
        stats = TimingStats()

        # Set up totals: method and function (both "small" model)
        stats.set_totals_by_type_tier({
            ("method", 2048): 3,
            ("function", 2048): 3,
        })

        # Only record data for function (wall_time=1.0)
        stats.record(1.0, 0.6, 0.4, "function", True, tier=2048)
        stats.record(1.0, 0.6, 0.4, "function", True, tier=2048)

        eta = stats.eta_seconds(2, 6, num_workers=1)
        # avg_func_2k = 1.0s (wall_time)
        # method/2k fallback to function's avg (same model group, same tier)
        # Remaining: 1 @ function/2k + 3 @ method/2k
        # = 1 * 1.0 + 3 * 1.0 = 4.0s
        assert eta is not None
        assert abs(eta - 4.0) < 0.2

    def test_get_eta_breakdown(self):
        """Test ETA breakdown returns per-(type, tier) estimates."""
        stats = TimingStats()

        # Set up totals by (type, tier)
        stats.set_totals_by_type_tier({
            ("function", 2048): 10,
            ("function", 8192): 5,
            ("class", 4096): 3,
        })

        # Record some completions
        stats.record(0.5, 0.3, 0.2, "function", True, tier=2048)
        stats.record(0.5, 0.3, 0.2, "function", True, tier=2048)
        stats.record(2.0, 1.5, 0.5, "function", True, tier=8192)
        stats.record(1.0, 0.7, 0.3, "class", True, tier=4096)

        breakdown = stats.get_eta_breakdown(num_workers=1)

        # Should have 3 entries (one for each type/tier combo with remaining > 0)
        assert len(breakdown) == 3

        # Check structure: (type, tier, remaining, total, eta_seconds)
        for elem_type, tier, remaining, total, eta in breakdown:
            assert elem_type in ("function", "class")
            assert tier in (2048, 4096, 8192)
            assert remaining > 0
            assert remaining <= total
            assert eta > 0

        # Should be sorted by ETA descending
        etas = [x[4] for x in breakdown]
        assert etas == sorted(etas, reverse=True)

    def test_get_eta_breakdown_empty_when_no_tier_data(self):
        """Test ETA breakdown returns empty list when no tier data."""
        stats = TimingStats()
        stats.set_totals_by_type({"function": 10})  # Only type-level totals

        breakdown = stats.get_eta_breakdown()
        assert breakdown == []

    def test_tier_overflow_detection(self):
        """Test that overflows are detected when prompt_tokens > assigned_tier."""
        stats = TimingStats()
        stats.set_totals_by_type({"function": 5})

        # Element within tier (1500 tokens < 2048 tier)
        stats.record(1.0, 0.5, 0.3, "function", True,
                     prompt_tokens=1500, response_tokens=100, assigned_tier=2048)

        # Element overflowing tier (2200 tokens > 2048 tier)
        stats.record(1.0, 0.5, 0.3, "function", True,
                     prompt_tokens=2200, response_tokens=100, assigned_tier=2048)

        # Another overflow
        stats.record(1.0, 0.5, 0.3, "function", True,
                     prompt_tokens=2500, response_tokens=100, assigned_tier=2048)

        assert stats.tier_overflow_counts[("function", 2048)] == 2
        assert stats.tier_sample_counts[("function", 2048)] == 3

    def test_tier_headroom_tracking(self):
        """Test headroom calculation: 1.0 - (prompt_tokens / assigned_tier)."""
        stats = TimingStats()
        stats.set_totals_by_type({"function": 2})

        # 50% headroom: 1024 / 2048 = 0.5 → headroom = 0.5
        stats.record(1.0, 0.5, 0.3, "function", True,
                     prompt_tokens=1024, response_tokens=100, assigned_tier=2048)

        # 25% headroom: 1536 / 2048 = 0.75 → headroom = 0.25
        stats.record(1.0, 0.5, 0.3, "function", True,
                     prompt_tokens=1536, response_tokens=100, assigned_tier=2048)

        key = ("function", 2048)
        avg_headroom = stats.tier_headroom_sum[key] / stats.tier_sample_counts[key]
        assert abs(avg_headroom - 0.375) < 0.001  # (0.5 + 0.25) / 2

        # Worst headroom should be 0.25 (the tighter one)
        assert abs(stats.tier_headroom_min[key] - 0.25) < 0.001

    def test_tier_headroom_negative_on_overflow(self):
        """Headroom should be negative when prompt exceeds tier."""
        stats = TimingStats()
        stats.set_totals_by_type({"function": 1})

        # 2560 / 2048 = 1.25 → headroom = -0.25
        stats.record(1.0, 0.5, 0.3, "function", True,
                     prompt_tokens=2560, response_tokens=100, assigned_tier=2048)

        key = ("function", 2048)
        assert abs(stats.tier_headroom_min[key] - (-0.25)) < 0.001

    def test_output_token_tracking(self):
        """Test per-type output token aggregation."""
        stats = TimingStats()
        stats.set_totals_by_type({"function": 3})

        stats.record(1.0, 0.5, 0.3, "function", True,
                     prompt_tokens=1000, response_tokens=80, assigned_tier=2048)
        stats.record(1.0, 0.5, 0.3, "function", True,
                     prompt_tokens=1000, response_tokens=120, assigned_tier=2048)
        stats.record(1.0, 0.5, 0.3, "function", True,
                     prompt_tokens=1000, response_tokens=60, assigned_tier=2048)

        assert stats.output_sample_counts["function"] == 3
        assert stats.output_tokens_sum["function"] == 260  # 80 + 120 + 60
        assert stats.output_tokens_max["function"] == 120

    def test_tier_accuracy_summary_no_issues(self):
        """Summary should show has_issues=False when everything is within bounds."""
        stats = TimingStats()
        stats.set_totals_by_type({"function": 2})

        # Comfortable headroom, low output tokens
        stats.record(1.0, 0.5, 0.3, "function", True,
                     prompt_tokens=1000, response_tokens=50, assigned_tier=2048)
        stats.record(1.0, 0.5, 0.3, "function", True,
                     prompt_tokens=1200, response_tokens=60, assigned_tier=2048)

        summary = stats.get_tier_accuracy_summary()
        assert summary["has_issues"] is False
        assert summary["input"] == []
        assert summary["output"] == []

    def test_tier_accuracy_summary_with_overflow(self):
        """Summary should report input overflows."""
        stats = TimingStats()
        stats.set_totals_by_type({"function": 3})

        stats.record(1.0, 0.5, 0.3, "function", True,
                     prompt_tokens=1000, response_tokens=50, assigned_tier=2048)
        # Overflow
        stats.record(1.0, 0.5, 0.3, "function", True,
                     prompt_tokens=2200, response_tokens=50, assigned_tier=2048)
        stats.record(1.0, 0.5, 0.3, "function", True,
                     prompt_tokens=1500, response_tokens=50, assigned_tier=2048)

        summary = stats.get_tier_accuracy_summary()
        assert summary["has_issues"] is True
        assert len(summary["input"]) == 1

        row = summary["input"][0]
        elem_type, tier, count, overflows, avg_pct, worst_pct = row
        assert elem_type == "function"
        assert tier == 2048
        assert count == 3
        assert overflows == 1
        assert worst_pct < 0  # The overflow case has negative headroom

    def test_tier_accuracy_summary_with_output_exceeded(self):
        """Summary should report output budget exceedances."""
        stats = TimingStats()
        stats.set_totals_by_type({"function": 2})

        # Output budget for function is 200 tokens
        stats.record(1.0, 0.5, 0.3, "function", True,
                     prompt_tokens=1000, response_tokens=150, assigned_tier=2048)
        stats.record(1.0, 0.5, 0.3, "function", True,
                     prompt_tokens=1000, response_tokens=250, assigned_tier=2048)

        summary = stats.get_tier_accuracy_summary()
        assert summary["has_issues"] is True
        assert len(summary["output"]) == 1

        row = summary["output"][0]
        elem_type, avg_tokens, max_tokens, budget = row
        assert elem_type == "function"
        assert max_tokens == 250
        assert budget == 200

    def test_tier_accuracy_summary_tight_headroom(self):
        """Summary should flag when worst headroom < 10% even without overflows."""
        stats = TimingStats()
        stats.set_totals_by_type({"function": 2})

        # 5% headroom: 1946 / 2048 ≈ 0.95 → headroom ≈ 0.05
        stats.record(1.0, 0.5, 0.3, "function", True,
                     prompt_tokens=1946, response_tokens=50, assigned_tier=2048)
        stats.record(1.0, 0.5, 0.3, "function", True,
                     prompt_tokens=1000, response_tokens=50, assigned_tier=2048)

        summary = stats.get_tier_accuracy_summary()
        assert summary["has_issues"] is True
        assert len(summary["input"]) == 1

        row = summary["input"][0]
        _, _, _, overflows, _, worst_pct = row
        assert overflows == 0  # No overflows, just tight headroom
        assert worst_pct < 10

    def test_tier_accuracy_summary_multiple_types_and_tiers(self):
        """Summary should track separately per (type, tier) combination."""
        stats = TimingStats()
        stats.set_totals_by_type({"function": 2, "class": 2})

        # Function at 2048 - overflow
        stats.record(1.0, 0.5, 0.3, "function", True,
                     prompt_tokens=2200, response_tokens=50, assigned_tier=2048)
        # Function at 4096 - fine
        stats.record(1.0, 0.5, 0.3, "function", True,
                     prompt_tokens=2000, response_tokens=50, assigned_tier=4096)
        # Class at 4096 - fine
        stats.record(1.0, 0.5, 0.3, "class", True,
                     prompt_tokens=2000, response_tokens=50, assigned_tier=4096)

        summary = stats.get_tier_accuracy_summary()
        assert summary["has_issues"] is True

        # Only function@2048 should appear (the overflow)
        assert len(summary["input"]) == 1
        assert summary["input"][0][0] == "function"
        assert summary["input"][0][1] == 2048

    def test_tier_metrics_skipped_when_no_prompt_tokens(self):
        """Tier accuracy should not track elements with 0 prompt tokens (skip_ai)."""
        stats = TimingStats()
        stats.set_totals_by_type({"function": 2})

        # No prompt tokens (skip_ai mode)
        stats.record(1.0, 0.0, 0.0, "function", True,
                     prompt_tokens=0, response_tokens=0, assigned_tier=2048)

        assert stats.tier_sample_counts == {}
        assert stats.output_sample_counts == {}


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

    def test_start_time_preserved_on_stage_update(self):
        """Test that start_time is preserved when updating stage.

        This is critical for throttling: we want total element runtime,
        not time since current stage started.
        """
        status = WorkerStatus()
        original_start = 1000.0  # Arbitrary start time

        # Set initial status with specific start time
        status.set(0, "elem1", "summarizing", "model", "4K", original_start)

        # Update to new stage with different start time (simulating stage_start_time = time.time())
        new_start = 2000.0  # Would be passed if stage resets timer
        status.set(0, "elem1", "embedding", "model", "4K", new_start)

        all_status = status.get_all()
        _, _, _, _, preserved_start = all_status[0]

        # Original start time should be preserved, not replaced
        assert preserved_start == original_start, (
            f"Start time should be preserved on stage update. "
            f"Expected {original_start}, got {preserved_start}"
        )


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
            caller_embedding=None,
            repo=mock_es,
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
            caller_embedding=None,
            repo=mock_es,
        )

        assert result is True

        # Verify store_calls was called with correct data
        mock_es.store_calls.assert_called_once()
        call_args = mock_es.store_calls.call_args
        assert call_args[0][0] == func_elem.element_id
        calls_data = call_args[0][1]
        assert len(calls_data) == 2
        assert calls_data[0] == {"name": "helper", "receiver": None, "line": 11, "resolved_id": None, "category": "unknown"}
        assert calls_data[1] == {
            "name": "process",
            "receiver": "utils",
            "line": 12,
            "resolved_id": "scope:repo:user:utils.py:function:process:5",
            "category": "unknown",
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
            caller_embedding=None,
            repo=mock_es,
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
            caller_embedding=None,
            repo=mock_es,
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
            caller_embedding=None,
            repo=mock_es,
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
            caller_embedding=None,
            repo=mock_es,
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
            caller_embedding=None,
            repo=mock_es,
        )

        # store_imports should NOT be called for non-file elements
        mock_es.store_imports.assert_not_called()


# =============================================================================
# CONTEXT SIZE TESTS
# =============================================================================


class TestPerElementContextSize:
    """Tests for per-element context size computation."""

    def test_small_element_uses_smallest_tier(self):
        """Small elements should use 1024 context tier."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "test summary"

        # Small function: 15 chars = ~4 tokens + 800 overhead = 804 → 1024 tier
        element = CodeElement(
            element_id="test:repo:user:file.py:function:foo:1",
            element_type="function",
            name="foo",
            raw_code="def foo(): pass",
        )

        config = ProcessingConfig()
        cache = SummaryCache()
        cache.add_element(element)

        _summarize_element(element, cache, mock_llm, config)

        call_kwargs = mock_llm.generate.call_args[1]
        assert call_kwargs.get("num_ctx") == 1024

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
        cache = SummaryCache()
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
        cache = SummaryCache()
        cache.add_element(element)

        _summarize_element(element, cache, mock_llm, config)

        call_kwargs = mock_llm.generate.call_args[1]
        assert call_kwargs.get("num_ctx") == 16384

    def test_summarize_element_returns_token_counts(self):
        """_summarize_element should return (summary, prompt_tokens, response_tokens)."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Handles user authentication"

        element = CodeElement(
            element_id="test:repo:user:file.py:function:auth:1",
            element_type="function",
            name="auth",
            raw_code="def auth(): pass",
        )

        config = ProcessingConfig()
        cache = SummaryCache()
        cache.add_element(element)

        result = _summarize_element(element, cache, mock_llm, config)

        assert isinstance(result, tuple)
        assert len(result) == 3
        summary, prompt_tokens, response_tokens = result
        assert isinstance(summary, str)
        assert prompt_tokens > 0
        assert response_tokens > 0
        # response_tokens should be len("Handles user authentication") // 4
        assert response_tokens == len("Handles user authentication") // 4

    def test_element_type_affects_overhead(self):
        """Different element types have different prompt overheads."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "test summary"

        # 5000 chars with different types:
        # - function: 5000/4=1250 + 800 overhead = 2050 → 4096 tier
        # - import: 5000/4=1250 + 350 overhead = 1600 → 2048 tier
        func_element = CodeElement(
            element_id="test:repo:user:file.py:function:func:1",
            element_type="function",
            name="func",
            raw_code="x" * 5000,
        )
        import_element = CodeElement(
            element_id="test:repo:user:file.py:import:imp:1",
            element_type="import",
            name="imp",
            raw_code="x" * 5000,
        )

        config = ProcessingConfig()
        cache = SummaryCache()
        cache.add_element(func_element)
        cache.add_element(import_element)

        _summarize_element(func_element, cache, mock_llm, config)
        func_num_ctx = mock_llm.generate.call_args[1].get("num_ctx")

        _summarize_element(import_element, cache, mock_llm, config)
        import_num_ctx = mock_llm.generate.call_args[1].get("num_ctx")

        assert func_num_ctx == 4096
        assert import_num_ctx == 2048

    def test_status_shows_context_tier(self):
        """Status should display context tier (e.g., '1K' for small elements)."""
        from magaldi_core.processor import (
            _process_single_element,
            WorkerStatus,
            SummaryCache,
        )

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "test summary"
        mock_embed = MagicMock()
        mock_es = MagicMock()

        # Small function should show 1K tier
        small_element = CodeElement(
            element_id="test:repo:user:file.py:function:small:1",
            element_type="function",
            name="small",
            raw_code="def small(): pass",
            relative_path="file.py",
        )

        config = ProcessingConfig()
        config.skip_ai = True  # Skip actual LLM calls
        cache = SummaryCache()
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
            repo=mock_es,
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
        # Small function should be 1K tier (shown in ctx_size field, index 3)
        ctx_sizes = [u[3] for u in summ_updates]
        assert "1K" in ctx_sizes


class TestDynamicWorkerScaling:
    """Tests for dynamic worker scaling with tier-based batching and warmup."""

    def test_warmup_returns_one_element(self):
        """First call during tier warmup should return only 1 element."""
        elements = [
            CodeElement(
                element_id=f"test:repo:user:file.py:function:f{i}:1",
                element_type="function",
                name=f"f{i}",
                raw_code="def f(): pass",
            )
            for i in range(15)
        ]

        # All small elements = same tier
        context_sizes = {e.element_id: 500 for e in elements}
        tracker = DependencyTracker(elements, context_sizes, max_num_workers=8)

        # First call: warmup, returns 1 element
        ready = tracker.get_ready_elements(max_count=20)
        assert len(ready) == 1
        assert tracker.is_tier_changing()  # Still in warmup

    def test_after_warmup_allows_max_workers(self):
        """After warmup completes, max_workers elements can be returned."""
        elements = [
            CodeElement(
                element_id=f"test:repo:user:file.py:function:f{i}:1",
                element_type="function",
                name=f"f{i}",
                raw_code="def f(): pass",
            )
            for i in range(15)
        ]

        context_sizes = {e.element_id: 500 for e in elements}
        tracker = DependencyTracker(elements, context_sizes, max_num_workers=8)

        # First call: warmup
        ready1 = tracker.get_ready_elements(max_count=20)
        assert len(ready1) == 1

        # Mark warmup task complete
        tracker.mark_complete(ready1[0].element_id)

        # Now should return up to max_workers (8) elements
        ready2 = tracker.get_ready_elements(max_count=20)
        assert len(ready2) == 8

    def test_tier_change_triggers_warmup(self):
        """Changing tiers should trigger new warmup."""
        # Mix of elements at different tiers but same level
        small_elements = [
            CodeElement(
                element_id=f"test:repo:user:file.py:function:small{i}:1",
                element_type="function",
                name=f"small{i}",
                raw_code="def f(): pass",
            )
            for i in range(3)
        ]
        large_elements = [
            CodeElement(
                element_id=f"test:repo:user:file.py:function:large{i}:10",
                element_type="function",
                name=f"large{i}",
                raw_code="x" * 100000,
            )
            for i in range(3)
        ]

        all_elements = small_elements + large_elements
        context_sizes = {e.element_id: 500 for e in small_elements}
        context_sizes.update({e.element_id: 30000 for e in large_elements})

        tracker = DependencyTracker(all_elements, context_sizes, max_num_workers=8)

        # First tier (small) - warmup
        ready1 = tracker.get_ready_elements(max_count=20)
        assert len(ready1) == 1
        tracker.mark_complete(ready1[0].element_id)

        # Small tier - can now get more
        ready2 = tracker.get_ready_elements(max_count=20)
        for e in ready2:
            tracker.mark_complete(e.element_id)

        # Now switching to large tier - should warmup again
        ready3 = tracker.get_ready_elements(max_count=20)
        assert len(ready3) == 1  # Warmup for new tier
        assert tracker.is_tier_changing()

    def test_get_current_max_workers(self):
        """get_current_max_workers should return configured max."""
        element = CodeElement(
            element_id="test:repo:user:file.py:function:f:1",
            element_type="function",
            name="f",
            raw_code="def f(): pass",
        )

        # Custom max workers
        tracker = DependencyTracker([element], {element.element_id: 500}, max_num_workers=16)
        assert tracker.get_current_max_workers() == 16

        # Default max workers
        tracker_default = DependencyTracker([element], {element.element_id: 500})
        assert tracker_default.get_current_max_workers() == 8  # DEFAULT_WORKERS

    def test_processed_element_carries_tier_metrics(self):
        """ProcessedElement should carry prompt_tokens, response_tokens, assigned_tier."""
        elem = ProcessedElement(
            element_id="scope:repo:user:file.py:function:test:1",
            success=True,
            wall_time=1.0,
            summarize_time=0.5,
            embed_time=0.3,
            prompt_tokens=500,
            response_tokens=120,
            assigned_tier=2048,
        )

        assert elem.prompt_tokens == 500
        assert elem.response_tokens == 120
        assert elem.assigned_tier == 2048

    def test_processed_element_tier_metrics_default_zero(self):
        """Tier metric fields should default to 0."""
        elem = ProcessedElement(
            element_id="scope:repo:user:file.py:function:test:1",
            success=True,
            wall_time=1.0,
            summarize_time=0.5,
            embed_time=0.3,
        )

        assert elem.prompt_tokens == 0
        assert elem.response_tokens == 0
        assert elem.assigned_tier == 0

    def test_post_warmup_gradual_ramp(self):
        """After warmup completes, throttle should recommend gradual ramp from 1."""
        from shared.throttling import compute_throttle_decision

        elements = [
            CodeElement(
                element_id=f"test:repo:user:file.py:function:f{i}:1",
                element_type="function",
                name=f"f{i}",
                raw_code="def f(): pass",
            )
            for i in range(5)
        ]
        context_sizes = {e.element_id: 500 for e in elements}
        tracker = DependencyTracker(elements, context_sizes, max_num_workers=8)

        # Warmup phase - get 1 element
        ready1 = tracker.get_ready_elements(max_count=20)
        assert len(ready1) == 1
        assert tracker.is_tier_changing()

        # Complete warmup task
        tracker.mark_complete(ready1[0].element_id)
        assert not tracker.is_tier_changing()

        # Compute throttle decision - should have post_warmup=True internally
        # and recommend starting at 1 worker (gradual ramp)
        decision = tracker.compute_throttle_decision(
            current_max_runtime=0.0,
            active_workers=0,
            throughput=0.0,
            avg_runtime=0.0,
            completion_count=0,
            avg_concurrency=0.0,
            avg_base_time=5.0,  # Historical data that would suggest more workers
        )

        # Post-warmup should force starting at 1 worker, not jumping based on historical
        assert decision.recommended_workers == 1
        assert "post-warmup" in decision.reason.lower()

        # Flag is NOT auto-consumed - must call clear_post_warmup() explicitly
        # (This matches processor behavior where clear is called after main throttle calc)
        tracker.clear_post_warmup()

        # Subsequent calls should NOT have post_warmup (flag was cleared)
        decision2 = tracker.compute_throttle_decision(
            current_max_runtime=0.0,
            active_workers=1,  # Now we have 1 active
            throughput=0.0,
            avg_runtime=0.0,
            completion_count=0,
            avg_concurrency=0.0,
            avg_base_time=5.0,
        )
        # Should be able to ramp up now (no longer post-warmup)
        assert "post-warmup" not in decision2.reason.lower()
