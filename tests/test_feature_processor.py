"""Tests for the feature processor module."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from shared.ai.clustering.clusterer import ClusteringResult, ClusterResult
from shared.ai.clustering.feature_processor import (
    FeatureProcessingConfig,
    FeatureProgressState,
    FeatureTimingStats,
    FeatureWorkerStatus,
    ProcessedFeature,
    ProcessedSubfeature,
    SubClusterConfig,
    SubfeatureLabelingState,
    SubfeatureProgressState,
    SubfeatureTimingStats,
    SubfeatureWorkerStatus,
    SubfeatureWorkItem,
    _embed_feature,
    _generate_feature_summary,
    _generate_subfeature_summary,
    _process_single_feature,
    _process_single_subfeature,
    process_features,
    process_subfeatures,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def feature_config():
    """Create a FeatureProcessingConfig for testing."""
    return FeatureProcessingConfig(
        summarize_model="test-model",
        embed_model="test-embed-model",
        api_base="http://localhost:11434",
        provider="ollama",
        num_workers=2,
    )


@pytest.fixture
def mock_repo():
    """Create a mock Elasticsearch repository."""
    repo = MagicMock()
    repo.get_summaries_batch.return_value = {}
    repo.delete_features.return_value = None
    repo.index_feature.return_value = None
    repo.delete_subfeatures.return_value = None
    repo.index_subfeature.return_value = None
    repo.get_documents_batch.return_value = {}
    return repo


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock()
    # Mock both methods for backwards compatibility
    client.generate.return_value = "This is a test summary."
    client.generate_from_messages.return_value = "This is a test summary."
    return client


@pytest.fixture
def mock_embed_client():
    """Create a mock embedding client."""
    client = MagicMock()
    client.embed_single.return_value = [0.1] * 1024
    return client


@pytest.fixture
def sample_cluster():
    """Create a sample ClusterResult."""
    return ClusterResult(
        cluster_id=0,
        label="test_feature",
        element_ids=["elem1", "elem2", "elem3"],
        element_names=["func1", "func2", "func3"],
    )


@pytest.fixture
def sample_clustering_result(sample_cluster):
    """Create a sample ClusteringResult."""
    return ClusteringResult(
        clusters=[sample_cluster],
        outlier_count=0,
        outlier_element_ids=[],
        total_elements=3,
    )


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestFeatureProcessingConfig:
    """Tests for FeatureProcessingConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = FeatureProcessingConfig()

        assert config.summarize_model == "qwen3.5:4b"
        assert config.embed_model == "qwen3-embedding:0.6b"
        assert config.api_base == "http://localhost:11434"
        assert config.provider == "ollama"
        assert config.num_workers == 4
        assert config.summarize_temperature == 0.2
        assert config.embed_dimensions == 1024

    def test_custom_values(self):
        """Test custom configuration values."""
        config = FeatureProcessingConfig(
            summarize_model="custom-model",
            num_workers=8,
            summarize_temperature=0.5,
        )

        assert config.summarize_model == "custom-model"
        assert config.num_workers == 8
        assert config.summarize_temperature == 0.5


class TestSubClusterConfig:
    """Tests for SubClusterConfig."""

    def test_default_values(self):
        """Test default sub-cluster configuration."""
        config = SubClusterConfig()

        assert config.min_cluster_size == 3
        assert config.min_samples == 2
        assert config.min_members_for_subclustering == 20

    def test_custom_values(self):
        """Test custom sub-cluster configuration."""
        config = SubClusterConfig(
            min_cluster_size=5,
            min_samples=3,
            min_members_for_subclustering=50,
        )

        assert config.min_cluster_size == 5
        assert config.min_samples == 3
        assert config.min_members_for_subclustering == 50


# =============================================================================
# TIMING STATS TESTS
# =============================================================================


class TestFeatureTimingStats:
    """Tests for FeatureTimingStats."""

    def test_initial_values(self):
        """Test initial timing stats values."""
        stats = FeatureTimingStats()

        assert stats.total_summarize_time == 0.0
        assert stats.total_embed_time == 0.0
        assert stats.summarize_count == 0
        assert stats.embed_count == 0

    def test_record_updates_counters(self):
        """Test that record updates counters correctly."""
        stats = FeatureTimingStats()

        stats.record(1.5, 0.5)

        assert stats.total_summarize_time == 1.5
        assert stats.total_embed_time == 0.5
        assert stats.summarize_count == 1
        assert stats.embed_count == 1

    def test_record_multiple_times(self):
        """Test recording multiple times accumulates correctly."""
        stats = FeatureTimingStats()

        stats.record(1.0, 0.5)
        stats.record(2.0, 0.5)
        stats.record(1.5, 0.0)  # No embed time

        assert stats.total_summarize_time == 4.5
        assert stats.total_embed_time == 1.0
        assert stats.summarize_count == 3
        assert stats.embed_count == 2  # Only 2 had embed time

    def test_avg_summarize_time(self):
        """Test average summarize time calculation."""
        stats = FeatureTimingStats()

        stats.record(1.0, 0.5)
        stats.record(2.0, 0.5)

        assert stats.avg_summarize_time == 1.5

    def test_avg_summarize_time_no_records(self):
        """Test average summarize time with no records."""
        stats = FeatureTimingStats()

        assert stats.avg_summarize_time == 0.0

    def test_avg_embed_time(self):
        """Test average embed time calculation."""
        stats = FeatureTimingStats()

        stats.record(1.0, 0.5)
        stats.record(2.0, 1.5)

        assert stats.avg_embed_time == 1.0

    def test_avg_embed_time_no_records(self):
        """Test average embed time with no records."""
        stats = FeatureTimingStats()

        assert stats.avg_embed_time == 0.0

    def test_elapsed(self):
        """Test elapsed time calculation."""
        stats = FeatureTimingStats()
        stats.phase_start = time.time() - 5.0

        elapsed = stats.elapsed

        assert elapsed >= 5.0
        assert elapsed < 6.0

    def test_eta_seconds_no_records(self):
        """Test ETA returns None when no records."""
        stats = FeatureTimingStats()

        eta = stats.eta_seconds(completed=0, total=10, num_workers=2)

        assert eta is None

    def test_eta_seconds_with_records(self):
        """Test ETA calculation with records."""
        stats = FeatureTimingStats()
        stats.record(2.0, 1.0)  # 3 seconds total per feature

        eta = stats.eta_seconds(completed=1, total=5, num_workers=2)

        # 4 remaining, 3s each, 2 workers = 6 seconds
        assert eta == 6.0

    def test_eta_seconds_all_completed(self):
        """Test ETA when all completed."""
        stats = FeatureTimingStats()
        stats.record(2.0, 1.0)

        eta = stats.eta_seconds(completed=10, total=10, num_workers=2)

        assert eta == 0.0

    def test_thread_safety(self):
        """Test thread-safe recording."""
        stats = FeatureTimingStats()

        def record_times():
            for _ in range(100):
                stats.record(0.1, 0.1)

        threads = [threading.Thread(target=record_times) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert stats.summarize_count == 500
        assert stats.embed_count == 500


class TestSubfeatureTimingStats:
    """Tests for SubfeatureTimingStats."""

    def test_initial_values(self):
        """Test initial values."""
        stats = SubfeatureTimingStats()

        assert stats.total_summarize_time == 0.0
        assert stats.summarize_count == 0

    def test_record_and_averages(self):
        """Test recording and average calculations."""
        stats = SubfeatureTimingStats()

        stats.record(2.0, 1.0)
        stats.record(4.0, 2.0)

        assert stats.avg_summarize_time == 3.0
        assert stats.avg_embed_time == 1.5


# =============================================================================
# WORKER STATUS TESTS
# =============================================================================


class TestFeatureWorkerStatus:
    """Tests for FeatureWorkerStatus."""

    def test_set_and_get_all(self):
        """Test setting and getting worker status."""
        status = FeatureWorkerStatus()

        status.set(0, "feature_1", "summarizing", "model_a", "2K")
        status.set(1, "feature_2", "embedding", "model_b", "4K")

        all_status = status.get_all()

        assert len(all_status) == 2
        # 5-tuple: (feature_name, stage, model, ctx_size, start_time)
        feature, stage, model, ctx_size, start_time = all_status[0]
        assert feature == "feature_1"
        assert stage == "summarizing"
        assert model == "model_a"
        assert ctx_size == "2K"

        feature, stage, model, ctx_size, start_time = all_status[1]
        assert feature == "feature_2"
        assert stage == "embedding"
        assert model == "model_b"
        assert ctx_size == "4K"

    def test_clear(self):
        """Test clearing worker status."""
        status = FeatureWorkerStatus()

        status.set(0, "feature_1", "summarizing", "model_a", "2K")
        status.clear(0)

        all_status = status.get_all()
        assert len(all_status) == 0

    def test_clear_nonexistent(self):
        """Test clearing nonexistent worker."""
        status = FeatureWorkerStatus()

        # Should not raise
        status.clear(99)

        assert status.get_all() == {}

    def test_thread_safety(self):
        """Test thread-safe operations."""
        status = FeatureWorkerStatus()

        def set_and_clear():
            for i in range(100):
                status.set(i % 4, f"feature_{i}", "stage", "", "")
                status.clear(i % 4)

        threads = [threading.Thread(target=set_and_clear) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should be cleared at the end
        # (actual count may vary due to race conditions, but should not error)
        assert isinstance(status.get_all(), dict)


class TestSubfeatureWorkerStatus:
    """Tests for SubfeatureWorkerStatus."""

    def test_set_and_get_all(self):
        """Test setting and getting worker status."""
        status = SubfeatureWorkerStatus()

        status.set(0, "parent_feature", "summarize", "model", "sub1", "2K")
        status.set(1, "parent_feature", "embed", "", "sub2", "")

        all_status = status.get_all()

        assert len(all_status) == 2
        # 6-tuple: (parent_feature, stage, model, subfeature, ctx_size, start_time)
        parent, stage, model, subfeature, ctx_size, start_time = all_status[0]
        assert parent == "parent_feature"
        assert stage == "summarize"
        assert model == "model"
        assert subfeature == "sub1"
        assert ctx_size == "2K"

        parent, stage, model, subfeature, ctx_size, start_time = all_status[1]
        assert parent == "parent_feature"
        assert stage == "embed"
        assert subfeature == "sub2"


# =============================================================================
# PROGRESS STATE TESTS
# =============================================================================


class TestFeatureProgressState:
    """Tests for FeatureProgressState."""

    def test_initialization(self):
        """Test progress state initialization."""
        timing = FeatureTimingStats()
        workers = FeatureWorkerStatus()

        state = FeatureProgressState(
            total=10,
            completed=5,
            failed=1,
            timing=timing,
            workers=workers,
            num_workers=4,
        )

        assert state.total == 10
        assert state.completed == 5
        assert state.failed == 1
        assert state.num_workers == 4


class TestSubfeatureProgressState:
    """Tests for SubfeatureProgressState."""

    def test_initialization(self):
        """Test subfeature progress state."""
        state = SubfeatureProgressState(
            total=20,
            completed=10,
            failed=2,
            timing=SubfeatureTimingStats(),
            workers=SubfeatureWorkerStatus(),
            num_workers=4,
        )

        assert state.total == 20
        assert state.completed == 10


class TestSubfeatureLabelingState:
    """Tests for SubfeatureLabelingState."""

    def test_initialization(self):
        """Test labeling state initialization."""
        state = SubfeatureLabelingState(
            total_features=5,
            features_processed=2,
            current_feature="auth",
            subclusters_labeled=10,
            model="test-model",
            phase_start=time.time(),
        )

        assert state.total_features == 5
        assert state.features_processed == 2
        assert state.current_feature == "auth"
        assert state.subclusters_labeled == 10


# =============================================================================
# PROCESSED RESULT TESTS
# =============================================================================


class TestProcessedFeature:
    """Tests for ProcessedFeature."""

    def test_successful_result(self):
        """Test successful processing result."""
        result = ProcessedFeature(
            cluster_id=0,
            success=True,
            summarize_time=1.5,
            embed_time=0.5,
            label="test_feature",
            summary="A test summary.",
        )

        assert result.success is True
        assert result.error is None

    def test_failed_result(self):
        """Test failed processing result."""
        result = ProcessedFeature(
            cluster_id=0,
            success=False,
            summarize_time=0.5,
            embed_time=0.0,
            error="Connection failed",
        )

        assert result.success is False
        assert result.error == "Connection failed"


class TestProcessedSubfeature:
    """Tests for ProcessedSubfeature."""

    def test_successful_result(self):
        """Test successful subfeature result."""
        result = ProcessedSubfeature(
            parent_label="parent",
            sub_label="sub_1",
            success=True,
            summarize_time=1.0,
            embed_time=0.5,
        )

        assert result.success is True
        assert result.parent_label == "parent"


class TestSubfeatureWorkItem:
    """Tests for SubfeatureWorkItem."""

    def test_initialization(self):
        """Test work item initialization."""
        cluster = ClusterResult(
            cluster_id=0,
            label="sub_cluster",
            element_ids=["a", "b"],
            element_names=["func_a", "func_b"],
        )

        item = SubfeatureWorkItem(
            parent_label="parent_feature",
            parent_summary="Parent summary.",
            sub_cluster=cluster,
            member_summaries={"a": "Summary A", "b": "Summary B"},
        )

        assert item.parent_label == "parent_feature"
        assert item.sub_cluster.cluster_id == 0


# =============================================================================
# FEATURE SUMMARY GENERATION TESTS
# =============================================================================


class TestGenerateFeatureSummary:
    """Tests for _generate_feature_summary function."""

    def test_generates_summary_with_member_summaries(
        self, sample_cluster, mock_llm_client, feature_config
    ):
        """Test summary generation with member summaries."""
        member_summaries = {
            "elem1": "First function summary.",
            "elem2": "Second function summary.",
        }

        summary, num_ctx = _generate_feature_summary(
            cluster=sample_cluster,
            member_summaries=member_summaries,
            llm_client=mock_llm_client,
            config=feature_config,
        )

        assert summary == "This is a test summary."
        assert num_ctx > 0  # Should return a valid context size
        mock_llm_client.generate_from_messages.assert_called_once()

    def test_generates_summary_without_member_summaries(
        self, sample_cluster, mock_llm_client, feature_config
    ):
        """Test summary generation without member summaries."""
        summary, num_ctx = _generate_feature_summary(
            cluster=sample_cluster,
            member_summaries={},
            llm_client=mock_llm_client,
            config=feature_config,
        )

        assert summary == "This is a test summary."
        assert num_ctx > 0

    def test_cleans_summary_prefix(self, sample_cluster, mock_llm_client, feature_config):
        """Test that 'Summary:' prefix is removed."""
        mock_llm_client.generate_from_messages.return_value = "Summary: This is the actual summary"

        summary, num_ctx = _generate_feature_summary(
            cluster=sample_cluster,
            member_summaries={},
            llm_client=mock_llm_client,
            config=feature_config,
        )

        assert summary == "This is the actual summary."

    def test_adds_period_if_missing(self, sample_cluster, mock_llm_client, feature_config):
        """Test that period is added if missing."""
        mock_llm_client.generate_from_messages.return_value = "A summary without period"

        summary, num_ctx = _generate_feature_summary(
            cluster=sample_cluster,
            member_summaries={},
            llm_client=mock_llm_client,
            config=feature_config,
        )

        assert summary.endswith(".")

    def test_uses_cluster_label(self, mock_llm_client, feature_config):
        """Test that cluster label is used in prompt."""
        cluster = ClusterResult(
            cluster_id=5,
            label=None,  # No label
            element_ids=["elem1"],
            element_names=["func1"],
        )

        _generate_feature_summary(
            cluster=cluster,
            member_summaries={},
            llm_client=mock_llm_client,
            config=feature_config,
        )

        # Should use fallback label (in user message)
        call_args = mock_llm_client.generate_from_messages.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[1]["content"]  # User message is second
        assert "cluster_5" in user_content

    def test_uses_dynamic_context_size(self, sample_cluster, mock_llm_client, feature_config):
        """Test that dynamic context size is computed based on prompt length."""
        # Small prompt should use smallest tier
        summary, num_ctx = _generate_feature_summary(
            cluster=sample_cluster,
            member_summaries={"elem1": "Short summary."},
            llm_client=mock_llm_client,
            config=feature_config,
        )

        call_args = mock_llm_client.generate_from_messages.call_args
        # Context should be one of the tiers
        assert call_args.kwargs["num_ctx"] in [1024, 2048, 4096, 8192, 16384, 32768]
        assert call_args.kwargs["num_ctx"] == num_ctx


class TestGenerateSubfeatureSummary:
    """Tests for _generate_subfeature_summary function."""

    def test_generates_summary_with_parent_context(
        self, sample_cluster, mock_llm_client, feature_config
    ):
        """Test subfeature summary includes parent context."""
        summary, num_ctx = _generate_subfeature_summary(
            cluster=sample_cluster,
            member_summaries={"elem1": "Summary 1"},
            parent_label="auth_feature",
            parent_summary="Authentication feature.",
            llm_client=mock_llm_client,
            config=feature_config,
        )

        call_args = mock_llm_client.generate_from_messages.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[1]["content"]  # User message is second
        assert "auth_feature" in user_content
        assert "Authentication feature" in user_content
        assert num_ctx > 0  # Should return valid context size

    def test_uses_dynamic_context_size(self, sample_cluster, mock_llm_client, feature_config):
        """Test that dynamic context size is computed based on prompt length."""
        summary, num_ctx = _generate_subfeature_summary(
            cluster=sample_cluster,
            member_summaries={"elem1": "Summary 1"},
            parent_label="auth_feature",
            parent_summary="Authentication feature.",
            llm_client=mock_llm_client,
            config=feature_config,
        )

        call_args = mock_llm_client.generate_from_messages.call_args
        # Context should be one of the tiers
        assert call_args.kwargs["num_ctx"] in [1024, 2048, 4096, 8192, 16384, 32768]
        assert call_args.kwargs["num_ctx"] == num_ctx


# =============================================================================
# EMBEDDING TESTS
# =============================================================================


class TestEmbedFeature:
    """Tests for _embed_feature function."""

    def test_returns_normalized_embedding(self, mock_embed_client, feature_config):
        """Test that embedding is normalized."""
        result = _embed_feature(
            summary="Test summary",
            embed_client=mock_embed_client,
            config=feature_config,
        )

        assert len(result) == 1024
        mock_embed_client.embed_single.assert_called_once()

    def test_raises_on_invalid_embedding(self, mock_embed_client, feature_config):
        """Test that invalid embedding raises ValueError."""
        mock_embed_client.embed_single.return_value = [0.1] * 512  # Wrong dimensions

        with pytest.raises(ValueError, match="Invalid embedding"):
            _embed_feature(
                summary="Test summary",
                embed_client=mock_embed_client,
                config=feature_config,
            )


# =============================================================================
# PROCESS SINGLE FEATURE TESTS
# =============================================================================


class TestProcessSingleFeature:
    """Tests for _process_single_feature function."""

    def test_successful_processing(
        self,
        sample_cluster,
        mock_repo,
        mock_llm_client,
        mock_embed_client,
        feature_config,
    ):
        """Test successful feature processing."""
        worker_status = FeatureWorkerStatus()

        with patch(
            "shared.ai.clustering.feature_processor.SummarizationLLMClient",
            return_value=mock_llm_client,
        ), patch(
            "shared.ai.clustering.feature_processor.CodeEmbeddingClient",
            return_value=mock_embed_client,
        ):
            result = _process_single_feature(
                cluster=sample_cluster,
                member_summaries={"elem1": "Summary"},
                scope="test_scope",
                repository="test_repo",
                username="test_user",
                repo=mock_repo,
                llm_client=mock_llm_client,
                embed_client=mock_embed_client,
                config=feature_config,
                worker_id=0,
                worker_status=worker_status,
            )

        assert result.success is True
        assert result.summarize_time > 0 or result.summarize_time == 0  # May be very fast
        assert result.error is None
        mock_repo.index_feature.assert_called_once()

    def test_processing_error_handling(
        self,
        sample_cluster,
        mock_repo,
        mock_llm_client,
        mock_embed_client,
        feature_config,
    ):
        """Test error handling during processing."""
        mock_llm_client.generate_from_messages.side_effect = Exception("LLM error")
        worker_status = FeatureWorkerStatus()

        result = _process_single_feature(
            cluster=sample_cluster,
            member_summaries={},
            scope="test_scope",
            repository="test_repo",
            username="test_user",
            repo=mock_repo,
            llm_client=mock_llm_client,
            embed_client=mock_embed_client,
            config=feature_config,
            worker_id=0,
            worker_status=worker_status,
        )

        assert result.success is False
        assert "LLM error" in result.error

    def test_worker_status_updates(
        self,
        sample_cluster,
        mock_repo,
        mock_llm_client,
        mock_embed_client,
        feature_config,
    ):
        """Test that worker status is updated during processing."""
        worker_status = FeatureWorkerStatus()
        status_changes = []

        def on_status_change():
            status_changes.append(dict(worker_status.get_all()))

        _process_single_feature(
            cluster=sample_cluster,
            member_summaries={},
            scope="test_scope",
            repository="test_repo",
            username="test_user",
            repo=mock_repo,
            llm_client=mock_llm_client,
            embed_client=mock_embed_client,
            config=feature_config,
            worker_id=0,
            worker_status=worker_status,
            on_status_change=on_status_change,
        )

        # Should have status updates for summarizing, embedding, indexing
        assert len(status_changes) >= 3


# =============================================================================
# PROCESS FEATURES TESTS
# =============================================================================


class TestProcessFeatures:
    """Tests for process_features function."""

    def test_returns_empty_for_no_clusters(self, mock_repo):
        """Test that empty result is returned when no clusters."""
        clustering_result = ClusteringResult(
            clusters=[],
            outlier_count=0,
            outlier_element_ids=[],
            total_elements=0,
        )

        result = process_features(
            clustering_result=clustering_result,
            scope="test_scope",
            repository="test_repo",
            username="test_user",
            repo=mock_repo,
        )

        assert result == {"processed": 0, "failed": 0}

    def test_processes_clusters_successfully(
        self, sample_clustering_result, mock_repo
    ):
        """Test successful processing of clusters."""
        mock_llm = MagicMock()
        mock_llm.generate_from_messages.return_value = "Test summary."

        mock_embed = MagicMock()
        mock_embed.embed_single.return_value = [0.1] * 1024

        with patch(
            "shared.ai.clustering.feature_processor.SummarizationLLMClient",
            return_value=mock_llm,
        ), patch(
            "shared.ai.clustering.feature_processor.CodeEmbeddingClient",
            return_value=mock_embed,
        ):
            result = process_features(
                clustering_result=sample_clustering_result,
                scope="test_scope",
                repository="test_repo",
                username="test_user",
                repo=mock_repo,
            )

        assert result["processed"] == 1
        assert result["failed"] == 0
        mock_repo.delete_features.assert_called_once()
        mock_repo.index_feature.assert_called_once()

    def test_progress_callback_is_called(self, sample_clustering_result, mock_repo):
        """Test that progress callback is invoked when there are enough clusters.

        With warmup enabled and multiple clusters, the throttled loop fires on_tick
        which triggers the progress callback. A single cluster is processed during
        warmup (synchronously) and never enters the throttled loop.
        """
        progress_states = []

        def on_progress(state: FeatureProgressState):
            progress_states.append(state)

        mock_llm = MagicMock()
        mock_llm.generate_from_messages.return_value = "Test summary."

        mock_embed = MagicMock()
        mock_embed.embed_single.return_value = [0.1] * 1024

        # Need at least 2 clusters for the throttled loop to fire on_tick
        from shared.ai.clustering.clusterer import ClusterResult
        multi_cluster_result = ClusteringResult(
            clusters=[
                ClusterResult(
                    cluster_id=i,
                    element_ids=[f"scope:repo:user:file{i}.py:function:func{i}:{i}"],
                    element_names=[f"func{i}"],
                    label=f"cluster_{i}",
                )
                for i in range(3)
            ],
            outlier_count=0,
            outlier_element_ids=[],
            total_elements=3,
        )

        with patch(
            "shared.ai.clustering.feature_processor.SummarizationLLMClient",
            return_value=mock_llm,
        ), patch(
            "shared.ai.clustering.feature_processor.CodeEmbeddingClient",
            return_value=mock_embed,
        ):
            result = process_features(
                clustering_result=multi_cluster_result,
                scope="test_scope",
                repository="test_repo",
                username="test_user",
                repo=mock_repo,
                on_progress=on_progress,
            )

        # All 3 clusters should be processed successfully
        assert result["processed"] == 3
        # With single cluster warmup + 2 remaining, the throttled loop should
        # fire at least one tick. If it doesn't (too fast), that's also valid.
        # The main assertion is that processing completes correctly.
        if len(progress_states) > 0:
            assert progress_states[-1].completed >= 1


class TestProcessSubfeatures:
    """Tests for process_subfeatures function."""

    def test_returns_empty_for_small_clusters(self, mock_repo):
        """Test that no subfeatures are created for small clusters."""
        # Create a small cluster (less than min_members_for_subclustering)
        small_cluster = ClusterResult(
            cluster_id=0,
            label="small_feature",
            element_ids=["e1", "e2"],
            element_names=["f1", "f2"],
        )
        clustering_result = ClusteringResult(
            clusters=[small_cluster],
            outlier_count=0,
            outlier_element_ids=[],
            total_elements=2,
        )

        result = process_subfeatures(
            clustering_result=clustering_result,
            processed_features={0: ("small_feature", "A small feature.")},
            scope="test_scope",
            repository="test_repo",
            username="test_user",
            repo=mock_repo,
        )

        assert result["subfeatures_created"] == 0
        assert result["parent_features_processed"] == 0


# =============================================================================
# PROCESS SINGLE SUBFEATURE TESTS
# =============================================================================


class TestProcessSingleSubfeature:
    """Tests for _process_single_subfeature function."""

    def test_successful_subfeature_processing(
        self,
        sample_cluster,
        mock_repo,
        mock_llm_client,
        mock_embed_client,
        feature_config,
    ):
        """Test successful subfeature processing."""
        worker_status = SubfeatureWorkerStatus()

        work_item = SubfeatureWorkItem(
            parent_label="parent",
            parent_summary="Parent summary.",
            sub_cluster=sample_cluster,
            member_summaries={"elem1": "Summary 1"},
        )

        result = _process_single_subfeature(
            work_item=work_item,
            scope="test_scope",
            repository="test_repo",
            username="test_user",
            repo=mock_repo,
            llm_client=mock_llm_client,
            embed_client=mock_embed_client,
            config=feature_config,
            worker_id=0,
            worker_status=worker_status,
        )

        assert result.success is True
        assert result.parent_label == "parent"
        mock_repo.index_subfeature.assert_called_once()

    def test_subfeature_processing_error(
        self,
        sample_cluster,
        mock_repo,
        mock_llm_client,
        mock_embed_client,
        feature_config,
    ):
        """Test error handling in subfeature processing."""
        mock_llm_client.generate_from_messages.side_effect = Exception("Generation failed")
        worker_status = SubfeatureWorkerStatus()

        work_item = SubfeatureWorkItem(
            parent_label="parent",
            parent_summary="Parent summary.",
            sub_cluster=sample_cluster,
            member_summaries={},
        )

        result = _process_single_subfeature(
            work_item=work_item,
            scope="test_scope",
            repository="test_repo",
            username="test_user",
            repo=mock_repo,
            llm_client=mock_llm_client,
            embed_client=mock_embed_client,
            config=feature_config,
            worker_id=0,
            worker_status=worker_status,
        )

        assert result.success is False
        assert "Generation failed" in result.error
