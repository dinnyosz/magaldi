"""Tests for the _retry_on_overload decorator in search backends."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shared.db.backends.base import _retry_on_overload


class FakeTransportError(Exception):
    """Mimics opensearchpy.TransportError with status_code attribute."""

    def __init__(self, status_code: int, message: str = "") -> None:
        self.status_code = status_code
        super().__init__(f"TransportError({status_code}, '{message}')")


class TestRetryOnOverload:
    """Tests for _retry_on_overload decorator."""

    def test_no_error_returns_immediately(self) -> None:
        """Calls with no errors should return the result without retrying."""
        call_count = 0

        @_retry_on_overload(max_retries=3)
        def fn() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        assert fn() == "ok"
        assert call_count == 1

    @patch("shared.db.backends.base.time.sleep")
    def test_retries_on_429(self, mock_sleep: MagicMock) -> None:
        """Should retry on TransportError(429) and succeed when error clears."""
        call_count = 0

        @_retry_on_overload(max_retries=3, base_delay=1.0)
        def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise FakeTransportError(429, "circuit_breaking_exception")
            return "ok"

        assert fn() == "ok"
        assert call_count == 3
        assert mock_sleep.call_count == 2

    @patch("shared.db.backends.base.time.sleep")
    def test_gives_up_after_max_retries(self, mock_sleep: MagicMock) -> None:
        """Should raise after exhausting all retries."""
        call_count = 0

        @_retry_on_overload(max_retries=2, base_delay=1.0)
        def fn() -> str:
            nonlocal call_count
            call_count += 1
            raise FakeTransportError(429, "")

        with pytest.raises(FakeTransportError, match="429"):
            fn()

        # 1 initial + 2 retries = 3 total attempts
        assert call_count == 3
        assert mock_sleep.call_count == 2

    def test_non_429_errors_raise_immediately(self) -> None:
        """Non-429 errors should not be retried."""
        call_count = 0

        @_retry_on_overload(max_retries=3)
        def fn() -> str:
            nonlocal call_count
            call_count += 1
            raise FakeTransportError(400, "mapper_parsing_exception")

        with pytest.raises(FakeTransportError, match="400"):
            fn()

        assert call_count == 1

    def test_non_transport_errors_raise_immediately(self) -> None:
        """Exceptions without status_code should not be retried."""
        call_count = 0

        @_retry_on_overload(max_retries=3)
        def fn() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("something else")

        with pytest.raises(ValueError, match="something else"):
            fn()

        assert call_count == 1

    @patch("shared.db.backends.base.time.sleep")
    def test_exponential_backoff_delays(self, mock_sleep: MagicMock) -> None:
        """Backoff delays should increase exponentially."""
        call_count = 0

        @_retry_on_overload(max_retries=3, base_delay=2.0, max_delay=60.0)
        def fn() -> str:
            nonlocal call_count
            call_count += 1
            raise FakeTransportError(429, "")

        with pytest.raises(FakeTransportError):
            fn()

        # 3 retries = 3 sleep calls
        assert mock_sleep.call_count == 3
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        # delay = base_delay * 2^attempt + jitter(0,1)
        # attempt 0: 2.0 * 1 + [0,1) = [2.0, 3.0)
        # attempt 1: 2.0 * 2 + [0,1) = [4.0, 5.0)
        # attempt 2: 2.0 * 4 + [0,1) = [8.0, 9.0)
        assert 2.0 <= delays[0] < 3.0
        assert 4.0 <= delays[1] < 5.0
        assert 8.0 <= delays[2] < 9.0

    @patch("shared.db.backends.base.time.sleep")
    def test_max_delay_cap(self, mock_sleep: MagicMock) -> None:
        """Delay should be capped at max_delay."""
        call_count = 0

        @_retry_on_overload(max_retries=2, base_delay=10.0, max_delay=5.0)
        def fn() -> str:
            nonlocal call_count
            call_count += 1
            raise FakeTransportError(429, "")

        with pytest.raises(FakeTransportError):
            fn()

        delays = [call.args[0] for call in mock_sleep.call_args_list]
        for d in delays:
            assert d <= 5.0

    @patch("shared.db.backends.base.time.sleep")
    def test_preserves_method_args(self, mock_sleep: MagicMock) -> None:
        """Decorator should correctly pass through args and kwargs."""
        call_count = 0
        received_args: list[tuple] = []

        @_retry_on_overload(max_retries=1)
        def fn(a: int, b: str, *, c: bool = False) -> str:
            nonlocal call_count
            call_count += 1
            received_args.append((a, b, c))
            if call_count == 1:
                raise FakeTransportError(429, "")
            return f"{a}-{b}-{c}"

        result = fn(1, "x", c=True)
        assert result == "1-x-True"
        assert len(received_args) == 2
        assert received_args[0] == (1, "x", True)
        assert received_args[1] == (1, "x", True)

    @patch("shared.db.backends.base.time.sleep")
    def test_logs_warning_on_retry(self, mock_sleep: MagicMock) -> None:
        """Should log a warning on each retry attempt."""
        call_count = 0

        @_retry_on_overload(max_retries=2, base_delay=1.0)
        def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise FakeTransportError(429, "circuit_breaking_exception")
            return "ok"

        with patch("shared.db.backends.base.logger") as mock_logger:
            fn()
            assert mock_logger.warning.call_count == 2
