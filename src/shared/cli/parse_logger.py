"""Parse run logging for the Magaldi CLI.

Writes detailed, structured log files during parse/watch runs.
Logs include:
- Phase timing breakdown
- Token budget exceeded details (element IDs, types, values)
- All errors with full context
- Processing statistics

Log files go to `logs/` in the repository root (gitignored).
Each run gets a timestamped file: `parse_<scope>_<repo>_<YYYYMMDD_HHMMSS>.log`
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from magaldi_core.change_detection import ChangeManifest
    from magaldi_core.discovery import DiscoveryResult
    from magaldi_core.variable_scoring.models import ScoringResult


class ParseRunLogger:
    """Structured logger for a single parse/watch run.

    Accumulates data throughout the run and writes a detailed log file
    that can be used by agents or humans to diagnose issues.
    """

    def __init__(self, repo_path: str, scope: str, repository: str, username: str, mode: str = "parse") -> None:
        self.repo_path = repo_path
        self.scope = scope
        self.repository = repository
        self.username = username
        self.mode = mode
        self.run_start = time.time()
        self.run_start_dt = datetime.now()

        # Phase timings: list of (phase_name, start_time, end_time, extra_info)
        self._phase_timings: list[tuple[str, float, float, dict[str, Any]]] = []
        self._current_phase: tuple[str, float] | None = None

        # Error log: list of (timestamp, phase, error_message, context)
        self._errors: list[tuple[float, str, str, dict[str, Any]]] = []

        # Token budget exceeded entries
        self._budget_exceeded: list[dict[str, Any]] = []

        # Processing stats
        self._processing_stats: dict[str, Any] = {}

        # Discovery/manifest info
        self._discovery_info: dict[str, Any] = {}
        self._manifest_info: dict[str, Any] = {}

        # Set up log file path
        ts = self.run_start_dt.strftime("%Y%m%d_%H%M%S")
        safe_scope = scope.replace("/", "_").replace("\\", "_")
        safe_repo = repository.replace("/", "_").replace("\\", "_")
        log_dir = Path(repo_path) / "logs"
        log_dir.mkdir(exist_ok=True)
        self.log_path = log_dir / f"{mode}_{safe_scope}_{safe_repo}_{ts}.log"

    # =========================================================================
    # Phase Timing
    # =========================================================================

    def start_phase(self, name: str) -> None:
        """Mark the start of a phase. Ends any currently running phase."""
        if self._current_phase is not None:
            self.end_phase()
        self._current_phase = (name, time.time())

    def end_phase(self, extra: dict[str, Any] | None = None) -> float:
        """Mark the end of the current phase. Returns duration in seconds."""
        if self._current_phase is None:
            return 0.0
        name, start = self._current_phase
        end = time.time()
        self._phase_timings.append((name, start, end, extra or {}))
        self._current_phase = None
        return end - start

    def get_phase_timings(self) -> list[tuple[str, float]]:
        """Return list of (phase_name, duration_seconds) for all completed phases."""
        return [(name, end - start) for name, start, end, _ in self._phase_timings]

    def get_total_elapsed(self) -> float:
        """Return total elapsed time since run start."""
        return time.time() - self.run_start

    # =========================================================================
    # Error Logging
    # =========================================================================

    def log_error(self, phase: str, message: str, context: dict[str, Any] | None = None) -> None:
        """Log an error with context."""
        self._errors.append((time.time(), phase, message, context or {}))

    # =========================================================================
    # Token Budget Exceeded
    # =========================================================================

    def log_budget_exceeded(
        self,
        input_rows: list[tuple[str, int, int, int, float, float]],
        output_rows: list[tuple[str, int, int, int]],
    ) -> None:
        """Log token budget exceeded details.

        Args:
            input_rows: (type, tier, count, overflows, avg_headroom_pct, worst_headroom_pct)
            output_rows: (type, avg_tokens, max_tokens, budget)
        """
        entry: dict[str, Any] = {"timestamp": time.time()}

        if input_rows:
            entry["input_overflows"] = [
                {
                    "element_type": t,
                    "tier": tier,
                    "count": count,
                    "overflows": overflows,
                    "avg_headroom_pct": round(avg, 1),
                    "worst_headroom_pct": round(worst, 1),
                }
                for t, tier, count, overflows, avg, worst in input_rows
            ]

        if output_rows:
            entry["output_exceeded"] = [
                {
                    "element_type": t,
                    "avg_tokens": avg,
                    "max_tokens": max_tok,
                    "budget": budget,
                    "exceeded_by": max_tok - budget,
                    "exceeded_pct": round((max_tok - budget) / budget * 100, 1),
                }
                for t, avg, max_tok, budget in output_rows
            ]

        self._budget_exceeded.append(entry)

    # =========================================================================
    # Processing Stats
    # =========================================================================

    def log_discovery(self, result: DiscoveryResult) -> None:
        """Log discovery phase results."""
        self._discovery_info = {
            "scope": result.scope,
            "repository": result.repository,
            "username": result.username,
            "total_files": result.total_files,
            "total_lines": result.total_lines,
            "languages": {
                lang: {"files": s.files, "lines": s.lines}
                for lang, s in result.languages.items()
            },
        }

    def log_manifest(self, manifest: ChangeManifest) -> None:
        """Log change manifest results."""
        self._manifest_info = {
            "total_scanned": manifest.total_files_scanned,
            "new_files": len(manifest.new_files),
            "modified_files": len(manifest.modified_files),
            "deleted_files": len(manifest.deleted_files),
            "unchanged": manifest.unchanged_count,
            "files_to_parse": manifest.files_to_parse,
        }

    def log_processing_stats(
        self,
        processed: int,
        skipped: int,
        indexed: int,
        deleted: int,
        failed_elements: list[tuple[str, str]],
        avg_wall: float = 0.0,
        avg_summ: float = 0.0,
        avg_embed: float = 0.0,
        elapsed: float = 0.0,
    ) -> None:
        """Log processing phase statistics."""
        self._processing_stats = {
            "processed": processed,
            "skipped": skipped,
            "indexed": indexed,
            "deleted": deleted,
            "failed_count": len(failed_elements),
            "avg_wall_time": round(avg_wall, 3),
            "avg_summarize_time": round(avg_summ, 3),
            "avg_embed_time": round(avg_embed, 3),
            "elapsed": round(elapsed, 2),
        }

        # Log each failed element as an error
        for element_id, error in failed_elements:
            self.log_error("processing", error, {"element_id": element_id})

    def log_scoring_stats(self, result: ScoringResult) -> None:
        """Log variable scoring statistics."""
        self._processing_stats["scoring"] = {
            "total_variables": result.total_variables,
            "kept": result.kept,
            "dropped": result.dropped,
            "errors": result.errors,
            "elapsed": round(result.elapsed, 2),
            "batch_count": result.batch_count,
        }

    # =========================================================================
    # Write Log File
    # =========================================================================

    def write(self) -> Path:
        """Write the accumulated log to disk. Returns the log file path."""
        # End any open phase
        if self._current_phase is not None:
            self.end_phase()

        total_elapsed = self.get_total_elapsed()
        lines: list[str] = []

        # Header
        lines.append("=" * 70)
        lines.append(f"Magaldi {self.mode.upper()} Run Log")
        lines.append("=" * 70)
        lines.append(f"Timestamp:  {self.run_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Repository: {self.scope}/{self.repository}")
        lines.append(f"User:       {self.username}")
        lines.append(f"Repo Path:  {self.repo_path}")
        lines.append(f"Mode:       {self.mode}")
        lines.append("")

        # Phase Timing Summary
        lines.append("-" * 70)
        lines.append("PHASE TIMING")
        lines.append("-" * 70)
        for name, start, end, extra in self._phase_timings:
            duration = end - start
            pct = (duration / total_elapsed * 100) if total_elapsed > 0 else 0
            lines.append(f"  {name:<30s} {_format_duration_precise(duration):>10s}  ({pct:5.1f}%)")
            if extra:
                for k, v in extra.items():
                    lines.append(f"    {k}: {v}")
        lines.append(f"  {'─' * 45}")
        lines.append(f"  {'TOTAL':<30s} {_format_duration_precise(total_elapsed):>10s}")
        lines.append("")

        # Discovery & Manifest
        if self._discovery_info:
            lines.append("-" * 70)
            lines.append("DISCOVERY")
            lines.append("-" * 70)
            for k, v in self._discovery_info.items():
                if k == "languages":
                    lines.append("  languages:")
                    for lang, stats in v.items():
                        lines.append(f"    {lang}: {stats['files']} files, {stats['lines']} lines")
                else:
                    lines.append(f"  {k}: {v}")
            lines.append("")

        if self._manifest_info:
            lines.append("-" * 70)
            lines.append("CHANGE DETECTION")
            lines.append("-" * 70)
            for k, v in self._manifest_info.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

        # Processing Stats
        if self._processing_stats:
            lines.append("-" * 70)
            lines.append("PROCESSING")
            lines.append("-" * 70)
            for k, v in self._processing_stats.items():
                if isinstance(v, dict):
                    lines.append(f"  {k}:")
                    for sk, sv in v.items():
                        lines.append(f"    {sk}: {sv}")
                else:
                    lines.append(f"  {k}: {v}")
            lines.append("")

        # Token Budget Exceeded (key section for debugging)
        if self._budget_exceeded:
            lines.append("-" * 70)
            lines.append("TOKEN BUDGET EXCEEDED")
            lines.append("-" * 70)
            for entry in self._budget_exceeded:
                ts = datetime.fromtimestamp(entry["timestamp"]).strftime("%H:%M:%S")
                lines.append(f"  [{ts}]")

                if "input_overflows" in entry:
                    lines.append("  Input Context Overflows:")
                    for row in entry["input_overflows"]:
                        lines.append(
                            f"    type={row['element_type']}, tier={row['tier']}, "
                            f"count={row['count']}, overflows={row['overflows']}, "
                            f"avg_headroom={row['avg_headroom_pct']}%, "
                            f"worst_headroom={row['worst_headroom_pct']}%"
                        )

                if "output_exceeded" in entry:
                    lines.append("  Output Token Budget Exceeded:")
                    for row in entry["output_exceeded"]:
                        lines.append(
                            f"    type={row['element_type']}, avg={row['avg_tokens']}, "
                            f"max={row['max_tokens']}, budget={row['budget']}, "
                            f"exceeded_by={row['exceeded_by']} (+{row['exceeded_pct']}%)"
                        )
                lines.append("")

        # Errors (the most useful section for agents)
        if self._errors:
            lines.append("-" * 70)
            lines.append(f"ERRORS ({len(self._errors)} total)")
            lines.append("-" * 70)
            for ts, phase, message, context in self._errors:
                dt = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                lines.append(f"  [{dt}] [{phase}] {message}")
                if context:
                    for k, v in context.items():
                        lines.append(f"    {k}: {v}")
            lines.append("")

        # Machine-readable JSON summary at the end
        lines.append("-" * 70)
        lines.append("JSON SUMMARY (machine-readable)")
        lines.append("-" * 70)
        summary = {
            "run_start": self.run_start_dt.isoformat(),
            "mode": self.mode,
            "scope": self.scope,
            "repository": self.repository,
            "username": self.username,
            "total_elapsed_seconds": round(total_elapsed, 2),
            "phase_timings": {
                name: round(end - start, 3)
                for name, start, end, _ in self._phase_timings
            },
            "error_count": len(self._errors),
            "budget_exceeded": len(self._budget_exceeded) > 0,
            "processing": self._processing_stats,
        }
        lines.append(json.dumps(summary, indent=2))
        lines.append("")

        # Write
        self.log_path.write_text("\n".join(lines), encoding="utf-8")
        return self.log_path

    @property
    def has_errors(self) -> bool:
        """True if any errors were logged."""
        return len(self._errors) > 0

    @property
    def has_budget_exceeded(self) -> bool:
        """True if any token budget exceeded entries were logged."""
        return len(self._budget_exceeded) > 0


def _format_duration_precise(seconds: float) -> str:
    """Format duration with sub-second precision for log files.

    Examples: "1.23s", "1:05.2", "1:02:03"
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    total_secs = int(seconds)
    hours, remainder = divmod(total_secs, 3600)
    minutes, secs = divmod(remainder, 60)
    frac = seconds - total_secs
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}.{int(frac * 10)}"
