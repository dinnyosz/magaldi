"""Thread-safe bulk buffer for batching OpenSearch writes.

Collects index and update operations and flushes them in bulk when either:
- The buffer reaches a count threshold (default: 50 operations)
- A time interval has elapsed since the last flush (default: 5 seconds)

Usage:
    with BulkIndexBuffer(client, index_name) as buf:
        buf.add_index(doc_id, document)   # queues an index op
        buf.add_update(doc_id, partial)   # queues an update op
    # auto-flushes remaining ops on exit
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from shared.db.backends.base import SearchClient

logger = logging.getLogger(__name__)


class BulkIndexBuffer:
    """Thread-safe write-behind buffer for OpenSearch bulk operations.

    Supports both index (full document) and update (partial document) operations.
    Flushes automatically on count threshold or time interval.

    Args:
        client: Search backend client with bulk_helpers().
        index_name: Target index name.
        max_count: Flush after this many buffered operations.
        max_interval: Flush after this many seconds since last flush.
    """

    DEFAULT_MAX_COUNT = 50
    DEFAULT_MAX_INTERVAL = 5.0  # seconds

    def __init__(
        self,
        client: SearchClient,
        index_name: str,
        max_count: int = DEFAULT_MAX_COUNT,
        max_interval: float = DEFAULT_MAX_INTERVAL,
    ):
        self._client = client
        self._index_name = index_name
        self._max_count = max_count
        self._max_interval = max_interval

        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._last_flush_time = time.monotonic()

        # Stats
        self._total_flushed = 0
        self._flush_count = 0
        self._errors: list[str] = []

        # Timer thread for interval-based flushing
        self._stop_event = threading.Event()
        self._timer_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> BulkIndexBuffer:
        self._start_timer()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_index(self, doc_id: str, document: dict[str, Any]) -> None:
        """Queue an index (upsert full document) operation.

        Args:
            doc_id: Document ID.
            document: Full document body.
        """
        action = {
            "_op_type": "index",
            "_index": self._index_name,
            "_id": doc_id,
            **document,
        }
        self._add(action)

    def add_update(self, doc_id: str, partial: dict[str, Any]) -> None:
        """Queue a partial update operation.

        Args:
            doc_id: Document ID to update.
            partial: Fields to update (wrapped in {"doc": ...} automatically).
        """
        action = {
            "_op_type": "update",
            "_index": self._index_name,
            "_id": doc_id,
            "doc": partial,
        }
        self._add(action)

    def flush(self) -> int:
        """Flush the buffer immediately.

        Returns:
            Number of operations flushed (0 if buffer was empty).
        """
        with self._lock:
            return self._flush_locked()

    def close(self) -> None:
        """Stop the timer and flush remaining operations."""
        self._stop_timer()
        self.flush()

    @property
    def pending(self) -> int:
        """Number of operations waiting in the buffer."""
        with self._lock:
            return len(self._buffer)

    @property
    def total_flushed(self) -> int:
        """Total number of operations flushed so far."""
        return self._total_flushed

    @property
    def flush_count(self) -> int:
        """Number of bulk flush calls made."""
        return self._flush_count

    @property
    def errors(self) -> list[str]:
        """List of error messages from failed flush attempts."""
        return list(self._errors)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add(self, action: dict[str, Any]) -> None:
        """Add an action to the buffer, flushing if count threshold reached."""
        with self._lock:
            self._buffer.append(action)
            if len(self._buffer) >= self._max_count:
                self._flush_locked()

    def _flush_locked(self) -> int:
        """Flush while holding the lock. Returns count of ops flushed."""
        if not self._buffer:
            return 0

        actions = self._buffer
        self._buffer = []
        count = len(actions)
        self._last_flush_time = time.monotonic()

        # Release lock during the actual network call would be ideal,
        # but bulk_helpers is fast enough and we want simple correctness.
        # The lock is held briefly — new adds will just wait.
        try:
            success, errors = self._client.bulk_helpers(
                actions, raise_on_error=False, refresh=False
            )
            if errors:
                for err in errors[:5]:  # Cap stored errors
                    self._errors.append(str(err))
                logger.warning(
                    "Bulk flush: %d/%d succeeded, %d errors",
                    success, count, len(errors) if isinstance(errors, list) else errors,
                )
            self._total_flushed += success
        except Exception:
            logger.exception("Bulk flush failed for %d operations", count)
            self._errors.append(f"Flush of {count} ops failed")
            # Re-add to buffer so they can be retried on next flush
            self._buffer = actions + self._buffer
            return 0

        self._flush_count += 1
        return count

    # ------------------------------------------------------------------
    # Timer thread
    # ------------------------------------------------------------------

    def _start_timer(self) -> None:
        """Start the background timer for interval-based flushing."""
        if self._timer_thread is not None:
            return
        self._stop_event.clear()
        self._timer_thread = threading.Thread(
            target=self._timer_loop,
            daemon=True,
            name="bulk-flush-timer",
        )
        self._timer_thread.start()

    def _stop_timer(self) -> None:
        """Signal the timer thread to stop and wait for it."""
        self._stop_event.set()
        if self._timer_thread is not None:
            self._timer_thread.join(timeout=2.0)
            self._timer_thread = None

    def _timer_loop(self) -> None:
        """Background loop that checks for time-based flush."""
        while not self._stop_event.is_set():
            # Sleep in small increments so we can respond to stop quickly
            self._stop_event.wait(timeout=min(1.0, self._max_interval / 2))
            if self._stop_event.is_set():
                break

            with self._lock:
                elapsed = time.monotonic() - self._last_flush_time
                if elapsed >= self._max_interval and self._buffer:
                    self._flush_locked()
