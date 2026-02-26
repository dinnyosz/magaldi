"""Tests for Phase 4 per-type-per-tier ETA breakdown in build_display."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from magaldi_core.processor.timing import TimingStats


def render_to_str(renderable) -> str:
    """Render a Rich renderable to plain text string (no ANSI escapes)."""
    console = Console(width=200, no_color=True, highlight=False)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def _make_timing_stats_with_tier_data() -> TimingStats:
    """Create TimingStats with per-(type, tier) data populated."""
    stats = TimingStats()
    import time

    stats.phase_start = time.time() - 30  # 30 seconds ago

    # Set totals by type
    stats.set_totals_by_type({"file": 5, "function": 10, "method": 8})

    # Set totals by (type, tier)
    stats.set_totals_by_type_tier({
        ("file", 4096): 3,
        ("file", 2048): 2,
        ("function", 2048): 6,
        ("function", 1024): 4,
        ("method", 1024): 8,
    })

    # Record some completions
    for _ in range(2):
        stats.record(
            wall_time=2.0,
            summarize_time=1.5,
            embed_time=0.5,
            element_type="file",
            tier=4096,
            avg_workers=2.0,
        )
    stats.record(
        wall_time=1.0,
        summarize_time=0.8,
        embed_time=0.2,
        element_type="file",
        tier=2048,
        avg_workers=2.0,
    )
    for _ in range(3):
        stats.record(
            wall_time=0.5,
            summarize_time=0.4,
            embed_time=0.1,
            element_type="function",
            tier=2048,
            avg_workers=2.0,
        )
    for _ in range(2):
        stats.record(
            wall_time=0.3,
            summarize_time=0.2,
            embed_time=0.1,
            element_type="function",
            tier=1024,
            avg_workers=2.0,
        )
    for _ in range(4):
        stats.record(
            wall_time=0.3,
            summarize_time=0.2,
            embed_time=0.1,
            element_type="method",
            tier=1024,
            avg_workers=2.0,
        )

    return stats


def _make_timing_stats_no_tier_data() -> TimingStats:
    """Create TimingStats without tier data (only type data)."""
    stats = TimingStats()
    import time

    stats.phase_start = time.time() - 10

    stats.set_totals_by_type({"file": 3, "function": 5})

    # Record completions without tier info
    stats.record(
        wall_time=2.0,
        summarize_time=1.5,
        embed_time=0.5,
        element_type="file",
        tier=0,  # No tier
        avg_workers=1.0,
    )
    stats.record(
        wall_time=0.5,
        summarize_time=0.4,
        embed_time=0.1,
        element_type="function",
        tier=0,
        avg_workers=1.0,
    )

    return stats


class TestPhase4EtaDisplay:
    """Test that Phase 4 build_display shows per-type-per-tier ETA table."""

    def test_eta_table_shown_when_tier_data_available(self):
        """When tier data is available, build_display should show the ETA table."""
        from shared.cli._runners import run_processing  # noqa: F401 — triggers module load
        from shared.cli.extract import build_eta_table

        timing = _make_timing_stats_with_tier_data()
        breakdown = timing.get_eta_breakdown_with_avg(_num_workers=2)

        # Should have breakdown data
        assert len(breakdown) > 0, "Expected per-(type, tier) breakdown data"

        type_order = ["file", "class", "interface", "type_alias", "function", "method", "constant", "variable"]
        type_colors = {
            "file": "cyan",
            "class": "magenta",
            "function": "blue",
            "method": "green",
        }
        eta_table = build_eta_table(breakdown, type_order, type_colors)

        # Should produce a table
        assert eta_table is not None, "Expected build_eta_table to return a Table"
        assert isinstance(eta_table, Table)

        # Render the table and check it contains type names and tier info
        output = render_to_str(eta_table)
        assert "file" in output
        assert "function" in output
        assert "method" in output

    def test_no_eta_table_without_tier_data(self):
        """When no tier data is available, build_eta_table should return None."""
        from shared.cli.extract import build_eta_table

        timing = _make_timing_stats_no_tier_data()
        breakdown = timing.get_eta_breakdown_with_avg(_num_workers=1)

        # Without tier data, breakdown should be empty
        assert len(breakdown) == 0

        type_order = ["file", "function"]
        eta_table = build_eta_table(breakdown, type_order)
        assert eta_table is None

    def test_eta_breakdown_has_correct_types(self):
        """ETA breakdown should include all registered (type, tier) combos."""
        timing = _make_timing_stats_with_tier_data()
        breakdown = timing.get_eta_breakdown_with_avg(_num_workers=2)

        types_in_breakdown = {item[0] for item in breakdown}
        assert "file" in types_in_breakdown
        assert "function" in types_in_breakdown
        assert "method" in types_in_breakdown

    def test_eta_breakdown_has_correct_tiers(self):
        """ETA breakdown should include all registered tiers per type."""
        timing = _make_timing_stats_with_tier_data()
        breakdown = timing.get_eta_breakdown_with_avg(_num_workers=2)

        # Build a lookup
        data = {(item[0], item[1]) for item in breakdown}
        assert ("file", 4096) in data
        assert ("file", 2048) in data
        assert ("function", 2048) in data
        assert ("function", 1024) in data
        assert ("method", 1024) in data

    def test_eta_breakdown_tracks_completion(self):
        """Done/total counts should match what was recorded."""
        timing = _make_timing_stats_with_tier_data()
        breakdown = timing.get_eta_breakdown_with_avg(_num_workers=2)

        # Build lookup: (type, tier) -> (avg, is_fallback, done, total)
        data = {(item[0], item[1]): (item[2], item[3], item[4], item[5]) for item in breakdown}

        # file@4096: recorded 2 completions out of 3 total
        avg, is_fallback, done, total = data[("file", 4096)]
        assert done == 2
        assert total == 3
        assert avg > 0

        # function@1024: recorded 2 completions out of 4 total
        avg, is_fallback, done, total = data[("function", 1024)]
        assert done == 2
        assert total == 4
        assert avg > 0

        # method@1024: recorded 4 completions out of 8 total
        avg, is_fallback, done, total = data[("method", 1024)]
        assert done == 4
        assert total == 8
        assert avg > 0

    def test_eta_table_shows_tier_abbreviations(self):
        """ETA table should display tier abbreviations like 1k, 2k, 4k."""
        from shared.cli.extract import build_eta_table

        timing = _make_timing_stats_with_tier_data()
        breakdown = timing.get_eta_breakdown_with_avg(_num_workers=2)
        type_order = ["file", "function", "method"]
        eta_table = build_eta_table(breakdown, type_order)

        assert eta_table is not None
        output = render_to_str(eta_table)

        # Should contain tier abbreviations
        assert "4k" in output
        assert "2k" in output
        assert "1k" in output

    def test_eta_table_shows_completion_counts(self):
        """ETA table cells should show done/total counts."""
        from shared.cli.extract import build_eta_table

        timing = _make_timing_stats_with_tier_data()
        breakdown = timing.get_eta_breakdown_with_avg(_num_workers=2)
        type_order = ["file", "function", "method"]
        eta_table = build_eta_table(breakdown, type_order)

        assert eta_table is not None
        output = render_to_str(eta_table)

        # file@4096: 2/3
        assert "2/3" in output
        # function@2048: 3/6
        assert "3/6" in output
        # method@1024: 4/8
        assert "4/8" in output

    def test_fallback_to_type_line_when_no_tier_data(self):
        """When no tier data, build_display should fall back to the single-line type progress."""
        timing = _make_timing_stats_no_tier_data()
        breakdown = timing.get_eta_breakdown_with_avg(_num_workers=1)

        from shared.cli.extract import build_eta_table

        type_order = ["file", "function"]
        eta_table = build_eta_table(breakdown, type_order)

        # No tier data → no ETA table → would use type_line fallback
        assert eta_table is None

        # Verify type_stats still returns data for fallback
        type_stats = timing.get_type_stats()
        assert "file" in type_stats
        assert "function" in type_stats

    def test_eta_table_with_all_completed(self):
        """ETA table should show green counts when all items are done for a type/tier."""
        from shared.cli.extract import build_eta_table

        timing = TimingStats()
        import time

        timing.phase_start = time.time() - 5

        timing.set_totals_by_type({"function": 3})
        timing.set_totals_by_type_tier({("function", 1024): 3})

        # Complete all 3
        for _ in range(3):
            timing.record(
                wall_time=0.5,
                summarize_time=0.3,
                embed_time=0.2,
                element_type="function",
                tier=1024,
                avg_workers=1.0,
            )

        breakdown = timing.get_eta_breakdown_with_avg(_num_workers=1)
        type_order = ["function"]
        eta_table = build_eta_table(breakdown, type_order)

        assert eta_table is not None
        output = render_to_str(eta_table)
        assert "3/3" in output
