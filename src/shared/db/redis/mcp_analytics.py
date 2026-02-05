"""MCP tool usage analytics repository.

This module tracks MCP tool calls, transitions between tools, and causal
relationships to help improve the MCP server and identify usage patterns.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, cast

from .base import RedisRepository


# MCP Analytics Redis keys
MCP_TOOL_CALLS = "magaldi:mcp:tool_calls"
MCP_TOOL_TRANSITIONS = "magaldi:mcp:tool_transitions"
MCP_CONFIRMED_TRANSITIONS = "magaldi:mcp:confirmed_transitions"  # Transitions with causal links
MCP_SESSION_PREFIX = "magaldi:mcp:session:"
MCP_DAILY_CALLS_PREFIX = "magaldi:mcp:daily:"
MCP_RECENT_CALLS = "magaldi:mcp:recent_calls"  # List of recent tool calls with details

# Known MCP tool names for causal detection (tool mention in output)
MCP_TOOL_NAMES = {
    "search_code", "search_features", "find_similar", "get_element",
    "batch_get_elements", "get_context", "get_children", "find_files",
    "get_file_structure", "list_features", "get_feature_members",
    "pattern_search", "find_usages", "find_implementations", "get_call_graph",
    "find_callers", "find_call_chain", "find_dead_code", "find_entry_points",
    "list_glossary", "get_glossary_term", "search_glossary",
    "find_dependencies", "find_dependents", "dependency_graph",
    "list_patterns", "find_by_pattern", "find_complex_functions",
    "find_security_issues", "find_undocumented", "find_env_usage",
    "find_async_code", "get_command_tree", "get_route_tree",
    "list_repos", "get_repo_stats", "explain_element", "generate_config",
    "generate_skill", "parser_lab_analyze", "parser_lab_create_test",
    "parser_lab_run_tests", "parser_lab_suggest_fix",
    # Context7 tools
    "resolve-library-id", "query-docs",
}


class RedisMCPAnalyticsRepository(RedisRepository):
    """Redis-based MCP tool usage analytics.

    Tracks:
    - Tool call counts (total and daily)
    - Tool transitions (which tool follows which)
    - Session-based tracking for transition computation
    """

    SESSION_TTL = 3600  # 1 hour session timeout
    DAILY_TTL = 30 * 24 * 3600  # 30 days retention for daily data
    TRANSITION_MAX_GAP_SECONDS = 10  # Max gap between calls to count as a transition

    def record_tool_call(
        self,
        tool_name: str,
        session_id: str | None = None,
    ) -> None:
        """Record a tool call and compute transitions.

        Transitions are only recorded if the gap between the previous tool's
        end time and this tool's start time is <= TRANSITION_MAX_GAP_SECONDS.
        This ensures only related tool calls are counted as transitions.

        Args:
            tool_name: Name of the tool being called.
            session_id: Optional session identifier for transition tracking.
        """
        client = self._get_client()
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")

        # Increment total call count
        client.hincrby(MCP_TOOL_CALLS, tool_name, 1)

        # Increment daily call count
        daily_key = f"{MCP_DAILY_CALLS_PREFIX}{today}:calls"
        client.hincrby(daily_key, tool_name, 1)
        client.expire(daily_key, self.DAILY_TTL)

        # Track transitions if we have a session
        if session_id:
            session_key = f"{MCP_SESSION_PREFIX}{session_id}"
            last_data = self._redis_get(session_key)

            if last_data:
                try:
                    # Parse previous tool data (JSON: {tool, end_time})
                    data = json.loads(last_data)
                    last_tool = data.get("tool")
                    last_end_time = data.get("end_time")

                    # Only record transition if previous call ended recently
                    if last_tool and last_end_time:
                        end_dt = datetime.fromisoformat(last_end_time)
                        gap_seconds = (now - end_dt).total_seconds()

                        if gap_seconds <= self.TRANSITION_MAX_GAP_SECONDS:
                            # Record transition from last_tool to tool_name
                            transition_key = f"{last_tool}:{tool_name}"
                            client.hincrby(MCP_TOOL_TRANSITIONS, transition_key, 1)

                            # Also track daily transitions
                            daily_trans_key = f"{MCP_DAILY_CALLS_PREFIX}{today}:transitions"
                            client.hincrby(daily_trans_key, transition_key, 1)
                            client.expire(daily_trans_key, self.DAILY_TTL)
                except (json.JSONDecodeError, ValueError):
                    # Invalid data format, skip transition tracking
                    pass

            # Update session with current tool (start_time for duration, end_time for transitions)
            session_data = json.dumps({
                "tool": tool_name,
                "start_time": now.isoformat(),
                "end_time": None,
            })
            client.setex(session_key, self.SESSION_TTL, session_data)

    def record_tool_end(self, session_id: str | None = None) -> None:
        """Record that the current tool call has ended.

        This updates the session with the end time, which is used to determine
        if subsequent tool calls should be counted as transitions. Also records
        the tool execution duration for runtime analytics.

        Args:
            session_id: Session identifier for transition tracking.
        """
        if not session_id:
            return

        client = self._get_client()
        session_key = f"{MCP_SESSION_PREFIX}{session_id}"
        current_data = self._redis_get(session_key)

        if current_data:
            try:
                data = json.loads(current_data)
                now = datetime.now()
                data["end_time"] = now.isoformat()

                # Calculate and record duration if we have start_time
                start_time = data.get("start_time")
                tool_name = data.get("tool")
                if start_time and tool_name:
                    start_dt = datetime.fromisoformat(start_time)
                    duration_ms = int((now - start_dt).total_seconds() * 1000)

                    # Record duration stats: total_ms and call_count for averaging
                    duration_key = f"{MCP_DAILY_CALLS_PREFIX}durations"
                    client.hincrby(duration_key, f"{tool_name}:total_ms", duration_ms)
                    client.hincrby(duration_key, f"{tool_name}:count", 1)

                client.setex(session_key, self.SESSION_TTL, json.dumps(data))
            except json.JSONDecodeError:
                pass

    def get_tool_durations(self) -> dict[str, dict[str, int | float]]:
        """Get tool execution duration statistics.

        Returns:
            Dict mapping tool name to {total_ms, count, avg_ms}.
        """
        duration_key = f"{MCP_DAILY_CALLS_PREFIX}durations"
        raw_data = self._redis_hgetall(duration_key)

        # Parse and aggregate
        tools: dict[str, dict[str, int]] = {}
        for key, value in raw_data.items():
            # key format: "tool_name:total_ms" or "tool_name:count"
            parts = key.rsplit(":", 1)
            if len(parts) == 2:
                tool_name, metric = parts
                if tool_name not in tools:
                    tools[tool_name] = {"total_ms": 0, "count": 0}
                tools[tool_name][metric] = int(value)

        # Calculate averages
        result: dict[str, dict[str, int | float]] = {}
        for tool_name, stats in tools.items():
            total_ms = stats.get("total_ms", 0)
            count = stats.get("count", 0)
            avg_ms = total_ms / count if count > 0 else 0
            result[tool_name] = {
                "total_ms": total_ms,
                "count": count,
                "avg_ms": round(avg_ms, 1),
            }

        return result

    def get_tool_counts(self) -> dict[str, int]:
        """Get all tool call counts.

        Returns:
            Dict mapping tool name to call count.
        """
        counts = self._redis_hgetall(MCP_TOOL_CALLS)
        return {k: int(v) for k, v in counts.items()}

    def get_tool_transitions(self) -> dict[str, dict[str, int]]:
        """Get tool transition matrix.

        Returns:
            Nested dict: from_tool -> to_tool -> count
        """
        raw_transitions = self._redis_hgetall(MCP_TOOL_TRANSITIONS)

        matrix: dict[str, dict[str, int]] = {}
        for transition_key, count in raw_transitions.items():
            parts = transition_key.split(":", 1)
            if len(parts) == 2:
                from_tool, to_tool = parts
                if from_tool not in matrix:
                    matrix[from_tool] = {}
                matrix[from_tool][to_tool] = int(count)

        return matrix

    def get_top_tools(self, limit: int = 10) -> list[tuple[str, int]]:
        """Get most frequently called tools.

        Args:
            limit: Maximum number of tools to return.

        Returns:
            List of (tool_name, count) tuples, sorted by count descending.
        """
        counts = self.get_tool_counts()
        sorted_tools = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_tools[:limit]

    def get_top_transitions(
        self, limit: int = 10
    ) -> list[tuple[str, str, int]]:
        """Get most common tool transitions.

        Args:
            limit: Maximum number of transitions to return.

        Returns:
            List of (from_tool, to_tool, count) tuples, sorted by count descending.
        """
        raw_transitions = self._redis_hgetall(MCP_TOOL_TRANSITIONS)

        transitions = []
        for transition_key, count in raw_transitions.items():
            parts = transition_key.split(":", 1)
            if len(parts) == 2:
                from_tool, to_tool = parts
                transitions.append((from_tool, to_tool, int(count)))

        transitions.sort(key=lambda x: x[2], reverse=True)
        return transitions[:limit]

    def get_daily_counts(self, date: str | None = None) -> dict[str, int]:
        """Get tool counts for a specific day.

        Args:
            date: Date string (YYYY-MM-DD). Defaults to today.

        Returns:
            Dict mapping tool name to call count for that day.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        daily_key = f"{MCP_DAILY_CALLS_PREFIX}{date}:calls"
        counts = self._redis_hgetall(daily_key)
        return {k: int(v) for k, v in counts.items()}

    def get_daily_transitions(
        self, date: str | None = None
    ) -> dict[str, dict[str, int]]:
        """Get transitions for a specific day.

        Args:
            date: Date string (YYYY-MM-DD). Defaults to today.

        Returns:
            Nested dict: from_tool -> to_tool -> count
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        daily_key = f"{MCP_DAILY_CALLS_PREFIX}{date}:transitions"
        raw_transitions = self._redis_hgetall(daily_key)

        matrix: dict[str, dict[str, int]] = {}
        for transition_key, count in raw_transitions.items():
            parts = transition_key.split(":", 1)
            if len(parts) == 2:
                from_tool, to_tool = parts
                if from_tool not in matrix:
                    matrix[from_tool] = {}
                matrix[from_tool][to_tool] = int(count)

        return matrix

    def get_confirmed_transitions(self) -> dict[str, dict[str, int]]:
        """Get confirmed transition matrix (transitions with causal links).

        Returns:
            Nested dict: from_tool -> to_tool -> count
        """
        raw_transitions = self._redis_hgetall(MCP_CONFIRMED_TRANSITIONS)

        matrix: dict[str, dict[str, int]] = {}
        for transition_key, count in raw_transitions.items():
            parts = transition_key.split(":", 1)
            if len(parts) == 2:
                from_tool, to_tool = parts
                if from_tool not in matrix:
                    matrix[from_tool] = {}
                matrix[from_tool][to_tool] = int(count)

        return matrix

    def get_transition_comparison(self) -> list[dict[str, Any]]:
        """Compare all transitions with confirmed transitions.

        Returns:
            List of transitions with confirmation rates.
        """
        all_transitions = self.get_tool_transitions()
        confirmed = self.get_confirmed_transitions()

        result = []
        for from_tool, targets in all_transitions.items():
            for to_tool, total_count in targets.items():
                confirmed_count = confirmed.get(from_tool, {}).get(to_tool, 0)
                confirmation_rate = (
                    round(confirmed_count / total_count * 100, 1)
                    if total_count > 0
                    else 0.0
                )
                result.append({
                    "from_tool": from_tool,
                    "to_tool": to_tool,
                    "total_count": total_count,
                    "confirmed_count": confirmed_count,
                    "confirmation_rate": confirmation_rate,
                })

        # Sort by total count descending
        result.sort(key=lambda x: cast(int, x["total_count"]), reverse=True)
        return result

    def clear_analytics(self) -> None:
        """Clear all analytics data.

        Warning: This permanently deletes all tracking data.
        """
        client = self._get_client()

        # Delete main keys
        client.delete(MCP_TOOL_CALLS)
        client.delete(MCP_TOOL_TRANSITIONS)
        client.delete(MCP_CONFIRMED_TRANSITIONS)
        client.delete(MCP_RECENT_CALLS)

        # Delete all session keys
        for key in client.scan_iter(f"{MCP_SESSION_PREFIX}*"):
            client.delete(key)

        # Delete all daily keys
        for key in client.scan_iter(f"{MCP_DAILY_CALLS_PREFIX}*"):
            client.delete(key)

    # Maximum size for stored inputs/outputs (in characters)
    MAX_IO_SIZE = 10000
    # Maximum number of recent calls to keep
    MAX_RECENT_CALLS = 1000
    # Time window for causal link detection (seconds)
    # 2 minutes to allow multiple follow-up calls from one search
    CAUSAL_LINK_WINDOW = 120
    # Number of previous calls to check for causal links
    CAUSAL_HISTORY_LOOKBACK = 15

    def _extract_trackable_values(self, output: str) -> set[str]:
        """Extract values from output that could be tracked as causal links.

        Looks for:
        - hash_id values (64 char hex strings)
        - library IDs (format: /org/project)
        - file paths (containing / and common extensions)
        - element IDs with colons (scope:repo:user:path:type:name:line)

        Args:
            output: Tool output string.

        Returns:
            Set of trackable values found.
        """
        import re

        values: set[str] = set()

        # Hash IDs (64 character hex strings)
        hash_pattern = r"\b([a-f0-9]{64})\b"
        values.update(re.findall(hash_pattern, output))

        # Library IDs (Context7 format: /org/project or /org/project/version)
        lib_pattern = r"(/[a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+(?:/[a-zA-Z0-9._-]+)?)"
        values.update(re.findall(lib_pattern, output))

        # Element IDs with colons (at least 3 colon-separated parts)
        element_pattern = r"\b([a-zA-Z0-9_-]+(?::[a-zA-Z0-9_./+-]+){3,})\b"
        values.update(re.findall(element_pattern, output))

        # File paths with extensions
        path_pattern = r"([a-zA-Z0-9_./+-]+\.(?:py|ts|tsx|js|jsx|rs|php|go|java|rb))\b"
        values.update(re.findall(path_pattern, output))

        # Filter out very short values (likely false positives)
        return {v for v in values if len(v) >= 8}

    def _detect_tool_mentions(self, output: str) -> set[str]:
        """Detect MCP tool names mentioned in output.

        Args:
            output: Tool output string.

        Returns:
            Set of tool names mentioned.
        """
        mentioned: set[str] = set()
        output_lower = output.lower()

        for tool_name in MCP_TOOL_NAMES:
            # Check for exact tool name or with underscores/hyphens
            if tool_name.lower() in output_lower:
                mentioned.add(tool_name)
            # Also check snake_case vs kebab-case variants
            alt_name = tool_name.replace("-", "_")
            if alt_name != tool_name and alt_name.lower() in output_lower:
                mentioned.add(tool_name)

        return mentioned

    def _find_matching_value(
        self,
        input_args: dict[str, Any],
        trackable_values: set[str],
    ) -> str | None:
        """Find if any input argument contains a trackable value.

        Args:
            input_args: Current tool's input arguments.
            trackable_values: Values extracted from previous output.

        Returns:
            The matched value, or None if no match.
        """
        input_str = json.dumps(input_args, default=str)

        for value in trackable_values:
            if value in input_str:
                return value

        return None

    def _detect_causal_link(
        self,
        tool_name: str,
        session_id: str,
        input_args: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Detect if current tool call was triggered by a previous call.

        Uses hybrid approach:
        1. Parameter matching: Check if input contains values from recent output
        2. Tool suggestion: Check if previous output mentioned this tool

        Args:
            tool_name: Name of the current tool being called.
            session_id: Session identifier.
            input_args: Current tool's input arguments.

        Returns:
            Causal link info dict, or None if no link detected.
        """
        # Get the most recent completed calls from this session
        # Check up to CAUSAL_HISTORY_LOOKBACK calls to find the original source
        # This handles cases where one search leads to multiple follow-up calls
        session_history_key = f"{MCP_SESSION_PREFIX}{session_id}:history"
        recent_calls_raw = self._redis_lrange(
            session_history_key, 0, self.CAUSAL_HISTORY_LOOKBACK - 1
        )

        if not recent_calls_raw:
            return None

        now = datetime.now()

        for call_raw in recent_calls_raw:
            try:
                prev_call = json.loads(call_raw)
            except json.JSONDecodeError:
                continue

            prev_output = prev_call.get("output_result", "")
            prev_tool = prev_call.get("tool_name", "")
            prev_end_time = prev_call.get("end_time")

            if not prev_output or not prev_tool:
                continue

            # Check time window
            if prev_end_time:
                try:
                    end_dt = datetime.fromisoformat(prev_end_time)
                    if (now - end_dt).total_seconds() > self.CAUSAL_LINK_WINDOW:
                        continue
                except ValueError:
                    pass

            # Strategy 1: Parameter matching
            trackable_values = self._extract_trackable_values(prev_output)
            matched_value = self._find_matching_value(input_args, trackable_values)

            if matched_value:
                return {
                    "call_id": prev_call.get("call_id"),
                    "tool_name": prev_tool,
                    "matched_value": matched_value,
                    "match_type": "parameter",
                }

            # Strategy 2: Tool suggestion (previous output mentioned this tool)
            mentioned_tools = self._detect_tool_mentions(prev_output)
            if tool_name in mentioned_tools or tool_name.replace("_", "-") in mentioned_tools:
                return {
                    "call_id": prev_call.get("call_id"),
                    "tool_name": prev_tool,
                    "matched_value": None,
                    "match_type": "tool_suggestion",
                }

        return None

    def record_tool_call_details(
        self,
        tool_name: str,
        session_id: str,
        input_args: dict[str, Any],
        timestamp: str | None = None,
    ) -> str:
        """Record a tool call with its input arguments.

        Also detects causal relationships with previous calls using:
        1. Parameter matching (value from previous output used in current input)
        2. Tool suggestion (previous output mentioned this tool)

        Args:
            tool_name: Name of the tool being called.
            session_id: Session identifier.
            input_args: Tool input arguments.
            timestamp: Optional ISO timestamp (defaults to now).

        Returns:
            Call ID for later updating with output.
        """
        import uuid

        client = self._get_client()
        call_id = f"{session_id}:{uuid.uuid4().hex[:8]}"
        now = timestamp or datetime.now().isoformat()

        # Truncate input args to prevent huge storage
        input_str = json.dumps(input_args, default=str)
        if len(input_str) > self.MAX_IO_SIZE:
            input_str = input_str[: self.MAX_IO_SIZE - 3] + "..."

        # Detect causal link with previous calls
        triggered_by = self._detect_causal_link(tool_name, session_id, input_args)

        call_data = {
            "call_id": call_id,
            "tool_name": tool_name,
            "session_id": session_id,
            "input_args": input_str,
            "output_result": None,
            "start_time": now,
            "end_time": None,
            "duration_ms": None,
            "triggered_by": triggered_by,  # Causal link info
            "suggested_tools": None,  # Will be populated in record_tool_output
        }

        # Store in session for later update
        session_call_key = f"{MCP_SESSION_PREFIX}{session_id}:current_call"
        client.setex(session_call_key, self.SESSION_TTL, json.dumps(call_data))

        return call_id

    def record_tool_output(
        self,
        session_id: str,
        output_result: str,
    ) -> None:
        """Record tool output and finalize the call record.

        Also extracts suggested tools from output and stores the call
        in session history for causal link detection in subsequent calls.

        Args:
            session_id: Session identifier.
            output_result: Tool output result (will be truncated if too large).
        """
        client = self._get_client()
        session_call_key = f"{MCP_SESSION_PREFIX}{session_id}:current_call"
        call_data_str = self._redis_get(session_call_key)

        if not call_data_str:
            return

        try:
            call_data = json.loads(call_data_str)
            now = datetime.now()
            call_data["end_time"] = now.isoformat()

            # Calculate duration
            if call_data.get("start_time"):
                start_dt = datetime.fromisoformat(call_data["start_time"])
                call_data["duration_ms"] = int((now - start_dt).total_seconds() * 1000)

            # Truncate output
            if len(output_result) > self.MAX_IO_SIZE:
                output_result = output_result[: self.MAX_IO_SIZE - 3] + "..."
            call_data["output_result"] = output_result

            # Extract suggested tools from output for causal tracking
            suggested_tools = list(self._detect_tool_mentions(output_result))
            call_data["suggested_tools"] = suggested_tools if suggested_tools else None

            # Push to recent calls list (LPUSH for newest first)
            client.lpush(MCP_RECENT_CALLS, json.dumps(call_data))
            # Trim to keep only recent calls
            client.ltrim(MCP_RECENT_CALLS, 0, self.MAX_RECENT_CALLS - 1)

            # Track confirmed transitions (with causal links)
            triggered_by = call_data.get("triggered_by")
            if triggered_by:
                source_tool = triggered_by.get("tool_name")
                target_tool = call_data.get("tool_name")
                if source_tool and target_tool:
                    # Increment confirmed transition counter
                    transition_key = f"{source_tool}:{target_tool}"
                    client.hincrby(MCP_CONFIRMED_TRANSITIONS, transition_key, 1)

            # Add to session history for causal link detection
            # This allows subsequent calls to check this call's output
            session_history_key = f"{MCP_SESSION_PREFIX}{session_id}:history"
            client.lpush(session_history_key, json.dumps(call_data))
            # Keep enough history for causal detection (CAUSAL_HISTORY_LOOKBACK + buffer)
            client.ltrim(session_history_key, 0, 29)
            client.expire(session_history_key, self.SESSION_TTL)

            # Clean up session key
            client.delete(session_call_key)

        except json.JSONDecodeError:
            pass

    def get_recent_calls(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent tool calls with details.

        Args:
            limit: Maximum number of calls to return.

        Returns:
            List of call records with input/output details.
        """
        raw_calls = self._redis_lrange(MCP_RECENT_CALLS, 0, limit - 1)

        calls = []
        for raw_call in raw_calls:
            try:
                calls.append(json.loads(raw_call))
            except json.JSONDecodeError:
                continue

        return calls

    def get_transition_details(
        self,
        from_tool: str | None = None,
        to_tool: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get detailed transition information.

        Finds tool call transitions and returns details about the caller's
        output and callee's input. Includes both consecutive (sequential)
        transitions and non-consecutive causal transitions.

        Args:
            from_tool: Filter by source tool (optional).
            to_tool: Filter by destination tool (optional).
            limit: Maximum number of transitions to return.

        Returns:
            List of transition records with caller output and callee input.
        """
        # Get recent calls and find transitions
        calls = self.get_recent_calls(limit=500)

        # Group by session
        sessions: dict[str, list[dict[str, Any]]] = {}
        for call in calls:
            session_id = call.get("session_id", "")
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(call)

        transitions = []
        seen_pairs: set[tuple[str, str]] = set()  # (caller_call_id, callee_call_id)

        for session_id, session_calls in sessions.items():
            # Sort by start_time (oldest first)
            session_calls.sort(key=lambda x: x.get("start_time", ""))

            # Build lookup by call_id for finding causal sources
            calls_by_id: dict[str, dict[str, Any]] = {}
            for call in session_calls:
                call_id = call.get("call_id")
                if call_id:
                    calls_by_id[call_id] = call

            # Find consecutive pairs (sequential transitions)
            for i in range(len(session_calls) - 1):
                caller = session_calls[i]
                callee = session_calls[i + 1]

                caller_name = caller.get("tool_name", "")
                callee_name = callee.get("tool_name", "")

                # Apply filters
                if from_tool and caller_name != from_tool:
                    continue
                if to_tool and callee_name != to_tool:
                    continue

                pair_key = (caller.get("call_id", str(i)), callee.get("call_id", str(i + 1)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # Calculate elapsed time between calls
                elapsed_ms = None
                if caller.get("end_time") and callee.get("start_time"):
                    try:
                        caller_end = datetime.fromisoformat(caller["end_time"])
                        callee_start = datetime.fromisoformat(callee["start_time"])
                        elapsed_ms = int(
                            (callee_start - caller_end).total_seconds() * 1000
                        )
                    except ValueError:
                        pass

                # Check if callee has causal link to caller
                triggered_by = callee.get("triggered_by")
                is_causal = False
                causal_match_type = None
                causal_matched_value = None

                if triggered_by and triggered_by.get("tool_name") == caller_name:
                    is_causal = True
                    causal_match_type = triggered_by.get("match_type")
                    causal_matched_value = triggered_by.get("matched_value")

                transitions.append(
                    {
                        "caller": caller_name,
                        "caller_input": caller.get("input_args"),
                        "caller_output": caller.get("output_result"),
                        "callee": callee_name,
                        "callee_input": callee.get("input_args"),
                        "callee_output": callee.get("output_result"),
                        "elapsed_ms": elapsed_ms,
                        "caller_duration_ms": caller.get("duration_ms"),
                        "callee_duration_ms": callee.get("duration_ms"),
                        "timestamp": callee.get("start_time"),
                        "is_consecutive": True,  # This is a consecutive (temporal) transition
                        "is_causal": is_causal,
                        "causal_match_type": causal_match_type,
                        "causal_matched_value": causal_matched_value,
                    }
                )

            # Find non-consecutive causal transitions (when filtering by tools)
            if from_tool and to_tool:
                for callee in session_calls:
                    callee_name = callee.get("tool_name", "")
                    if callee_name != to_tool:
                        continue

                    triggered_by = callee.get("triggered_by")
                    if not triggered_by:
                        continue

                    source_tool = triggered_by.get("tool_name", "")
                    if source_tool != from_tool:
                        continue

                    # Find the source caller in this session
                    source_call_id = triggered_by.get("call_id")
                    source_caller = calls_by_id.get(source_call_id) if source_call_id else None

                    # Skip if we already have this pair from consecutive logic
                    pair_key = (source_call_id or "", callee.get("call_id", ""))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    # Calculate elapsed time if we found the caller
                    elapsed_ms = None
                    caller_input = None
                    caller_output = None
                    caller_duration_ms = None

                    if source_caller:
                        caller_input = source_caller.get("input_args")
                        caller_output = source_caller.get("output_result")
                        caller_duration_ms = source_caller.get("duration_ms")

                        if source_caller.get("end_time") and callee.get("start_time"):
                            try:
                                caller_end = datetime.fromisoformat(source_caller["end_time"])
                                callee_start = datetime.fromisoformat(callee["start_time"])
                                elapsed_ms = int(
                                    (callee_start - caller_end).total_seconds() * 1000
                                )
                            except ValueError:
                                pass

                    transitions.append(
                        {
                            "caller": from_tool,
                            "caller_input": caller_input,
                            "caller_output": caller_output,
                            "callee": callee_name,
                            "callee_input": callee.get("input_args"),
                            "callee_output": callee.get("output_result"),
                            "elapsed_ms": elapsed_ms,
                            "caller_duration_ms": caller_duration_ms,
                            "callee_duration_ms": callee.get("duration_ms"),
                            "timestamp": callee.get("start_time"),
                            "is_consecutive": False,  # Non-consecutive causal transition
                            "is_causal": True,
                            "causal_match_type": triggered_by.get("match_type"),
                            "causal_matched_value": triggered_by.get("matched_value"),
                        }
                    )

            if len(transitions) >= limit:
                break

        # Sort: causal first, then by timestamp
        transitions.sort(key=lambda t: (not t.get("is_causal", False), t.get("timestamp", "")))

        return transitions[:limit]

    def get_causal_statistics(self) -> dict[str, Any]:
        """Get statistics about causal relationships between tool calls.

        Returns:
            Dict with causal chain statistics:
            - total_causal_links: Number of calls with detected causal links
            - by_match_type: Counts by match type (parameter vs tool_suggestion)
            - top_causal_pairs: Most common source->target causal pairs
            - causal_chains: Example causal chains
        """
        calls = self.get_recent_calls(limit=500)

        total_links = 0
        by_match_type: dict[str, int] = {"parameter": 0, "tool_suggestion": 0}
        causal_pairs: dict[str, int] = {}  # "source:target" -> count

        for call in calls:
            triggered_by = call.get("triggered_by")
            if triggered_by:
                total_links += 1
                match_type = triggered_by.get("match_type", "unknown")
                by_match_type[match_type] = by_match_type.get(match_type, 0) + 1

                # Track causal pairs
                source_tool = triggered_by.get("tool_name", "unknown")
                target_tool = call.get("tool_name", "unknown")
                pair_key = f"{source_tool}:{target_tool}"
                causal_pairs[pair_key] = causal_pairs.get(pair_key, 0) + 1

        # Get top causal pairs
        sorted_pairs = sorted(causal_pairs.items(), key=lambda x: x[1], reverse=True)
        top_pairs = [
            {"source": pair.split(":")[0], "target": pair.split(":")[1], "count": count}
            for pair, count in sorted_pairs[:10]
        ]

        return {
            "total_causal_links": total_links,
            "total_calls": len(calls),
            "causal_rate": round(total_links / len(calls) * 100, 1) if calls else 0,
            "by_match_type": by_match_type,
            "top_causal_pairs": top_pairs,
        }

    def get_calls_with_causal_info(
        self,
        limit: int = 100,
        only_with_links: bool = False,
    ) -> list[dict[str, Any]]:
        """Get recent calls with causal relationship information.

        Args:
            limit: Maximum number of calls to return.
            only_with_links: If True, only return calls that have causal links.

        Returns:
            List of call records with triggered_by and suggested_tools fields.
        """
        calls = self.get_recent_calls(limit=limit * 2 if only_with_links else limit)

        if only_with_links:
            calls = [c for c in calls if c.get("triggered_by")][:limit]

        return calls
