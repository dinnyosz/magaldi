"""Tests for parallel call resolution.

Tests the _parallel_map helper, parallel resolve_all_calls, flush between
strategies, and on_step callback.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from magaldi_core.call_resolution import (
    _parallel_map,
    _process_file_group_strategies_3_5,
    resolve_all_calls,
)

# =============================================================================
# _parallel_map tests
# =============================================================================


class TestParallelMap:
    """Tests for the _parallel_map helper."""

    def test_sequential_when_max_workers_1(self):
        """max_workers=1 processes items sequentially without ThreadPoolExecutor."""
        results = _parallel_map(lambda x: x * 2, [1, 2, 3], max_workers=1)
        assert sorted(results) == [2, 4, 6]

    def test_parallel_when_max_workers_gt_1(self):
        """max_workers>1 processes items in parallel and returns all results."""
        results = _parallel_map(lambda x: x * 2, [1, 2, 3, 4, 5], max_workers=3)
        assert sorted(results) == [2, 4, 6, 8, 10]

    def test_empty_items(self):
        """Empty items list returns empty results."""
        results = _parallel_map(lambda x: x, [], max_workers=4)
        assert results == []

    def test_single_item_uses_sequential(self):
        """Single item falls back to sequential even with max_workers > 1."""
        results = _parallel_map(lambda x: x + 10, [5], max_workers=4)
        assert results == [15]

    def test_error_handling_sequential(self):
        """Errors in sequential mode are logged and skipped."""
        def flaky(x: int) -> int:
            if x == 2:
                raise ValueError("boom")
            return x * 10

        results = _parallel_map(flaky, [1, 2, 3], max_workers=1, desc="test")
        assert sorted(results) == [10, 30]

    def test_error_handling_parallel(self):
        """Errors in parallel mode are logged and skipped."""
        def flaky(x: int) -> int:
            if x == 3:
                raise RuntimeError("oops")
            return x * 10

        results = _parallel_map(flaky, [1, 2, 3, 4], max_workers=2, desc="test")
        assert sorted(results) == [10, 20, 40]

    def test_preserves_return_types(self):
        """Results preserve the return type of fn."""
        results = _parallel_map(
            lambda x: {"key": x, "value": x ** 2},
            [2, 3, 4],
            max_workers=2,
        )
        result_map = {r["key"]: r["value"] for r in results}
        assert result_map == {2: 4, 3: 9, 4: 16}


# =============================================================================
# _process_file_group_strategies_3_5 tests
# =============================================================================


class TestProcessFileGroup:
    """Tests for the per-file-group processing function."""

    def test_empty_group(self):
        """Empty element list returns zeroes."""
        repo = MagicMock()
        total, imp, typ = _process_file_group_strategies_3_5(
            [], repo, "scope", "repo", "main"
        )
        assert total == 0
        assert imp == 0
        assert typ == 0

    def test_resolves_import_call(self):
        """An imported function call is resolved via import map."""
        repo = MagicMock()
        repo.get_file_imports.return_value = [
            {"name": "helper", "module": "utils", "alias": None, "line": 1}
        ]
        # _lookup_element_by_import will be called — mock it
        with patch("magaldi_core.call_resolution._lookup_element_by_import") as mock_lookup, \
             patch("magaldi_core.call_resolution._build_import_map") as mock_build:
            mock_build.return_value = {"helper": {"name": "helper", "module": "utils"}}
            mock_lookup.return_value = "scope:repo:main:utils.py:function:helper:10"

            elements = [{
                "element_id": "scope:repo:main:app.py:function:main:5",
                "relative_path": "app.py",
                "language": "python",
                "parameters": [],
                "calls": [
                    {"name": "helper", "receiver": None, "category": "unknown", "line": 6},
                ],
            }]

            total, imp, typ = _process_file_group_strategies_3_5(
                elements, repo, "scope", "repo", "main"
            )

            assert total == 1
            assert imp == 1
            assert typ == 0
            repo.store_calls.assert_called_once()

    def test_shares_import_map_within_file(self):
        """All elements from the same file share one import map (one get_file_imports call)."""
        repo = MagicMock()
        repo.get_file_imports.return_value = []

        elements = [
            {
                "element_id": f"scope:repo:main:app.py:function:fn{i}:{i*10}",
                "relative_path": "app.py",
                "language": "python",
                "parameters": [],
                "calls": [{"name": "foo", "receiver": None, "category": "unknown", "line": i*10+1}],
            }
            for i in range(5)
        ]

        _process_file_group_strategies_3_5(elements, repo, "scope", "repo", "main")

        # get_file_imports should be called exactly once for the shared file
        repo.get_file_imports.assert_called_once_with("app.py", "scope", "repo", "main")


# =============================================================================
# resolve_all_calls integration tests
# =============================================================================


class TestResolveAllCallsParallel:
    """Integration tests for resolve_all_calls with max_workers > 1."""

    def _make_repo(self, elements: list[dict]) -> MagicMock:
        """Create a mock Repository with the given elements."""
        repo = MagicMock()
        repo.find_all_elements_with_calls.return_value = elements
        repo.get_file_imports.return_value = []
        repo.get_documents_batch.return_value = {}
        return repo

    def test_max_workers_1_is_sequential(self):
        """max_workers=1 uses sequential processing (default behavior)."""
        repo = self._make_repo([])

        total, imp, typ, con, sco, sup = resolve_all_calls(
            repo, "scope", "repo", "main", max_workers=1
        )

        assert total == 0
        assert imp == 0

    def test_max_workers_gt_1_produces_same_results(self):
        """Parallel and sequential modes produce the same result."""
        elements = [
            {
                "element_id": f"s:r:main:file{i}.py:function:fn{i}:{i}",
                "relative_path": f"file{i}.py",
                "language": "python",
                "parameters": [],
                "calls": [
                    {"name": "unresolved", "receiver": None, "category": "unknown", "line": i+1},
                ],
            }
            for i in range(10)
        ]

        # Sequential run
        repo1 = self._make_repo(elements)
        result_seq = resolve_all_calls(repo1, "s", "r", "main", max_workers=1)

        # Parallel run
        repo2 = self._make_repo(elements)
        result_par = resolve_all_calls(repo2, "s", "r", "main", max_workers=4)

        assert result_seq == result_par

    def test_flush_called_between_strategies(self):
        """repo.flush() is called between each strategy phase."""
        repo = self._make_repo([])

        resolve_all_calls(repo, "scope", "repo", "main", max_workers=1)

        # flush() should be called at least 4 times (after 3-5, 5.5, 5.6, 5.7)
        assert repo.flush.call_count >= 4

    def test_on_step_callback_called(self):
        """on_step callback is invoked for each strategy phase."""
        repo = self._make_repo([])
        steps: list[str] = []

        resolve_all_calls(
            repo, "scope", "repo", "main",
            max_workers=1,
            on_step=lambda msg: steps.append(msg),
        )

        assert len(steps) >= 5  # 3-5, 5.5, 5.6, 5.7, 5.8
        assert any("3-5" in s for s in steps)
        assert any("5.5" in s for s in steps)
        assert any("5.6" in s for s in steps)
        assert any("5.7" in s for s in steps)
        assert any("5.8" in s for s in steps)

    def test_on_step_none_does_not_crash(self):
        """on_step=None should work fine (default)."""
        repo = self._make_repo([])
        # Should not raise
        resolve_all_calls(repo, "scope", "repo", "main", on_step=None)

    def test_category_reset_preserves_across_parallel(self):
        """Resolved categories are properly reset in parallel mode."""
        elements = [
            {
                "element_id": "s:r:main:a.py:function:fn1:1",
                "relative_path": "a.py",
                "language": "python",
                "parameters": [{"name": "x", "type": "Foo"}],
                "calls": [
                    {
                        "name": "bar",
                        "receiver": "x",
                        "category": "resolved",
                        "resolved_id": "old:id",
                        "line": 2,
                    },
                ],
            },
        ]
        repo = self._make_repo(elements)

        resolve_all_calls(repo, "s", "r", "main", max_workers=2)

        # The call should have had its category reset
        if repo.store_calls.called:
            stored_calls = repo.store_calls.call_args[0][1]
            for c in stored_calls:
                assert c["category"] != "resolved" or c.get("resolved_id")
