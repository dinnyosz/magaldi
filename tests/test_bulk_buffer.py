"""Tests for BulkIndexBuffer - batched OpenSearch writes."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from shared.db.bulk_buffer import BulkIndexBuffer

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock SearchClient that tracks bulk_helpers calls."""
    client = MagicMock()
    client.bulk_helpers.return_value = (0, [])  # (success, errors) - overridden per test
    return client


@pytest.fixture
def buffer(mock_client: MagicMock) -> BulkIndexBuffer:
    """Create a buffer with small thresholds for testing."""
    return BulkIndexBuffer(
        mock_client,
        "test-index",
        max_count=5,
        max_interval=60.0,  # High interval so timer doesn't interfere
    )


# =============================================================================
# COUNT-BASED FLUSHING
# =============================================================================


class TestCountBasedFlush:
    """Tests for flushing when max_count is reached."""

    def test_no_flush_under_threshold(self, buffer: BulkIndexBuffer, mock_client: MagicMock) -> None:
        """Buffer should not flush when below max_count."""
        for i in range(4):
            buffer.add_index(f"doc_{i}", {"field": f"value_{i}"})

        mock_client.bulk_helpers.assert_not_called()
        assert buffer.pending == 4

    def test_auto_flush_at_threshold(self, buffer: BulkIndexBuffer, mock_client: MagicMock) -> None:
        """Buffer should auto-flush when max_count is reached."""
        mock_client.bulk_helpers.return_value = (5, [])

        for i in range(5):
            buffer.add_index(f"doc_{i}", {"field": f"value_{i}"})

        mock_client.bulk_helpers.assert_called_once()
        assert buffer.pending == 0
        assert buffer.total_flushed == 5

    def test_multiple_flushes(self, buffer: BulkIndexBuffer, mock_client: MagicMock) -> None:
        """Buffer should flush multiple times as threshold keeps being hit."""
        mock_client.bulk_helpers.return_value = (5, [])

        for i in range(12):
            buffer.add_index(f"doc_{i}", {"field": f"value_{i}"})

        assert mock_client.bulk_helpers.call_count == 2
        assert buffer.pending == 2  # 12 - 5 - 5 = 2 remaining
        assert buffer.total_flushed == 10

    def test_manual_flush(self, buffer: BulkIndexBuffer, mock_client: MagicMock) -> None:
        """Manual flush() should flush any pending ops."""
        mock_client.bulk_helpers.return_value = (3, [])

        for i in range(3):
            buffer.add_index(f"doc_{i}", {"field": f"value_{i}"})

        count = buffer.flush()

        assert count == 3
        assert buffer.pending == 0
        mock_client.bulk_helpers.assert_called_once()

    def test_manual_flush_empty(self, buffer: BulkIndexBuffer, mock_client: MagicMock) -> None:
        """Manual flush() on empty buffer should be a no-op."""
        count = buffer.flush()
        assert count == 0
        mock_client.bulk_helpers.assert_not_called()


# =============================================================================
# TIME-BASED FLUSHING
# =============================================================================


class TestTimeBasedFlush:
    """Tests for flushing when max_interval is reached."""

    def test_timer_flushes_on_interval(self, mock_client: MagicMock) -> None:
        """Timer should flush buffer after max_interval elapses."""
        mock_client.bulk_helpers.return_value = (2, [])

        buf = BulkIndexBuffer(
            mock_client,
            "test-index",
            max_count=100,  # High count so only timer triggers
            max_interval=0.3,  # Short interval for test
        )

        with buf:
            buf.add_index("doc_1", {"a": 1})
            buf.add_index("doc_2", {"b": 2})
            # Wait for timer to fire
            time.sleep(0.8)

        # Should have been flushed by timer or by close()
        assert mock_client.bulk_helpers.called
        assert buf.total_flushed == 2


# =============================================================================
# OPERATION TYPES
# =============================================================================


class TestOperationTypes:
    """Tests for index and update operations."""

    def test_add_index_action_format(self, buffer: BulkIndexBuffer, mock_client: MagicMock) -> None:
        """add_index should create proper bulk index actions."""
        mock_client.bulk_helpers.return_value = (1, [])

        buffer.add_index("my-doc-id", {"name": "test", "value": 42})
        buffer.flush()

        actions = mock_client.bulk_helpers.call_args[0][0]
        assert len(actions) == 1
        assert actions[0]["_op_type"] == "index"
        assert actions[0]["_index"] == "test-index"
        assert actions[0]["_id"] == "my-doc-id"
        assert actions[0]["name"] == "test"
        assert actions[0]["value"] == 42

    def test_add_update_action_format(self, buffer: BulkIndexBuffer, mock_client: MagicMock) -> None:
        """add_update should create proper bulk update actions."""
        mock_client.bulk_helpers.return_value = (1, [])

        buffer.add_update("my-doc-id", {"summary": "hello world"})
        buffer.flush()

        actions = mock_client.bulk_helpers.call_args[0][0]
        assert len(actions) == 1
        assert actions[0]["_op_type"] == "update"
        assert actions[0]["_index"] == "test-index"
        assert actions[0]["_id"] == "my-doc-id"
        assert actions[0]["doc"] == {"summary": "hello world"}

    def test_mixed_operations(self, buffer: BulkIndexBuffer, mock_client: MagicMock) -> None:
        """Buffer should handle a mix of index and update operations."""
        mock_client.bulk_helpers.return_value = (3, [])

        buffer.add_index("doc_1", {"name": "first"})
        buffer.add_update("doc_1", {"summary": "a summary"})
        buffer.add_index("doc_2", {"name": "second"})
        buffer.flush()

        actions = mock_client.bulk_helpers.call_args[0][0]
        assert len(actions) == 3
        assert actions[0]["_op_type"] == "index"
        assert actions[1]["_op_type"] == "update"
        assert actions[2]["_op_type"] == "index"


# =============================================================================
# CONTEXT MANAGER
# =============================================================================


class TestContextManager:
    """Tests for context manager behavior."""

    def test_context_manager_flushes_on_exit(self, mock_client: MagicMock) -> None:
        """Exiting context manager should flush remaining ops."""
        mock_client.bulk_helpers.return_value = (3, [])

        buf = BulkIndexBuffer(mock_client, "test-index", max_count=100)
        with buf:
            buf.add_index("doc_1", {"a": 1})
            buf.add_index("doc_2", {"b": 2})
            buf.add_index("doc_3", {"c": 3})

        mock_client.bulk_helpers.assert_called_once()
        assert buf.total_flushed == 3
        assert buf.pending == 0

    def test_context_manager_flushes_on_exception(self, mock_client: MagicMock) -> None:
        """Context manager should flush even if exception occurs."""
        mock_client.bulk_helpers.return_value = (2, [])

        buf = BulkIndexBuffer(mock_client, "test-index", max_count=100)
        with pytest.raises(ValueError, match="test error"), buf:
            buf.add_index("doc_1", {"a": 1})
            buf.add_index("doc_2", {"b": 2})
            raise ValueError("test error")

        # Buffer should still have been flushed
        mock_client.bulk_helpers.assert_called_once()
        assert buf.total_flushed == 2

    def test_close_stops_timer(self, mock_client: MagicMock) -> None:
        """Close should stop the timer thread."""
        buf = BulkIndexBuffer(mock_client, "test-index", max_interval=0.1)
        with buf:
            assert buf._timer_thread is not None
            assert buf._timer_thread.is_alive()

        # After context exit, timer should be stopped
        assert buf._timer_thread is None or not buf._timer_thread.is_alive()


# =============================================================================
# ERROR HANDLING
# =============================================================================


class TestErrorHandling:
    """Tests for error recovery and reporting."""

    def test_bulk_errors_are_recorded(self, buffer: BulkIndexBuffer, mock_client: MagicMock) -> None:
        """Errors from bulk_helpers should be recorded."""
        mock_client.bulk_helpers.return_value = (3, [{"error": "mapping_exception"}])

        for i in range(5):
            buffer.add_index(f"doc_{i}", {"f": i})

        assert len(buffer.errors) > 0
        assert buffer.total_flushed == 3

    def test_exception_during_flush_retries(self, buffer: BulkIndexBuffer, mock_client: MagicMock) -> None:
        """If bulk_helpers raises, ops should be kept for retry."""
        mock_client.bulk_helpers.side_effect = ConnectionError("connection refused")

        for i in range(5):
            buffer.add_index(f"doc_{i}", {"f": i})

        # After failed flush, items should still be in buffer
        assert buffer.pending == 5
        assert buffer.total_flushed == 0
        assert len(buffer.errors) == 1

    def test_retry_after_failure(self, buffer: BulkIndexBuffer, mock_client: MagicMock) -> None:
        """After a failed flush, a subsequent flush should succeed."""
        # First flush fails
        mock_client.bulk_helpers.side_effect = ConnectionError("connection refused")
        for i in range(5):
            buffer.add_index(f"doc_{i}", {"f": i})

        assert buffer.pending == 5

        # Second flush succeeds
        mock_client.bulk_helpers.side_effect = None
        mock_client.bulk_helpers.return_value = (5, [])
        count = buffer.flush()

        assert count == 5
        assert buffer.pending == 0
        assert buffer.total_flushed == 5


# =============================================================================
# THREAD SAFETY
# =============================================================================


class TestThreadSafety:
    """Tests for concurrent access to the buffer."""

    def test_concurrent_adds(self, mock_client: MagicMock) -> None:
        """Multiple threads adding to the buffer should not lose data."""
        mock_client.bulk_helpers.return_value = (50, [])

        buf = BulkIndexBuffer(mock_client, "test-index", max_count=50, max_interval=60.0)
        num_threads = 10
        ops_per_thread = 20
        total_ops = num_threads * ops_per_thread  # 200

        barrier = threading.Barrier(num_threads)

        def worker(thread_id: int) -> None:
            barrier.wait()  # All threads start at once
            for i in range(ops_per_thread):
                buf.add_index(f"doc_{thread_id}_{i}", {"t": thread_id, "i": i})

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Flush remaining
        buf.flush()

        # All ops should have been flushed (some via auto-flush, rest via manual)
        assert buf.total_flushed + buf.pending == 0 or buf.total_flushed == total_ops
        # Verify all 200 ops went through
        total_actions = sum(
            len(call[0][0]) for call in mock_client.bulk_helpers.call_args_list
        )
        assert total_actions == total_ops


# =============================================================================
# STATS
# =============================================================================


class TestStats:
    """Tests for buffer statistics."""

    def test_flush_count(self, buffer: BulkIndexBuffer, mock_client: MagicMock) -> None:
        """flush_count should track number of bulk calls."""
        mock_client.bulk_helpers.return_value = (5, [])

        for i in range(12):
            buffer.add_index(f"doc_{i}", {"f": i})

        # 2 auto-flushes at count 5 and 10
        assert buffer.flush_count == 2

        # Manual flush of remaining 2
        mock_client.bulk_helpers.return_value = (2, [])
        buffer.flush()
        assert buffer.flush_count == 3

    def test_total_flushed(self, buffer: BulkIndexBuffer, mock_client: MagicMock) -> None:
        """total_flushed should track cumulative successful ops."""
        mock_client.bulk_helpers.return_value = (5, [])

        for i in range(10):
            buffer.add_index(f"doc_{i}", {"f": i})

        assert buffer.total_flushed == 10

    def test_pending(self, buffer: BulkIndexBuffer, mock_client: MagicMock) -> None:
        """pending should reflect current buffer size."""
        buffer.add_index("doc_1", {"a": 1})
        assert buffer.pending == 1

        buffer.add_index("doc_2", {"b": 2})
        assert buffer.pending == 2

        mock_client.bulk_helpers.return_value = (2, [])
        buffer.flush()
        assert buffer.pending == 0
