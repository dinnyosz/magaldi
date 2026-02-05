"""MCP self-review and analytics tools."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    """Represents a single tool call with its context."""

    tool_name: str
    params: dict[str, Any]
    result_snippet: str
    position: int
    is_magaldi: bool


def mcp_self_review(
    analytics_repo: Any,
    context: str,
    include_analytics: bool = True,
    focus_tools: list[str] | None = None,
) -> dict[str, Any]:
    """Analyze magaldi tool usage to identify improvement opportunities.

    Performs deep analysis of tool call sequences to find:
    1. Deviation patterns - when magaldi results led to fallback to other tools
    2. Missing information - what was searched for elsewhere after magaldi calls
    3. Specific, actionable suggestions for improving magaldi tool responses

    Args:
        analytics_repo: MCP analytics repository instance.
        context: The recent conversation context to analyze.
        include_analytics: Whether to include MCP analytics data.
        focus_tools: Optional list of specific tool names to focus on.

    Returns:
        dict with sequence analysis, deviation patterns, and improvement suggestions.
    """
    # ==========================================================================
    # STEP 1: Extract all tool calls in sequence (magaldi AND builtin)
    # ==========================================================================

    tool_sequence: list[ToolCall] = []

    # Pattern for magaldi tool invocations with parameters
    # Match: <invoke name="mcp__magaldi__tool_name">..params..</invoke>
    magaldi_invoke_pattern = r'<invoke name="mcp__magaldi__(\w+)">(.*?)</invoke>'

    # Pattern for builtin tool invocations
    builtin_invoke_pattern = r'<invoke name="(Read|Grep|Glob|Bash)">(.*?)</invoke>'

    # Pattern for parameter extraction
    param_pattern = r'<parameter name="(\w+)">([^<]*)</parameter>'

    # Pattern for tool results
    result_pattern = r'<result>\s*<name>([^<]+)</name>\s*<output>(.*?)</output>\s*</result>'

    # Find all tool invocations in order
    all_invocations: list[tuple[int, str, str, bool]] = []  # (position, tool_name, params_str, is_magaldi)

    for match in re.finditer(magaldi_invoke_pattern, context, re.DOTALL):
        all_invocations.append((match.start(), match.group(1), match.group(2), True))

    for match in re.finditer(builtin_invoke_pattern, context, re.DOTALL):
        all_invocations.append((match.start(), match.group(1), match.group(2), False))

    # Sort by position to get chronological order
    all_invocations.sort(key=lambda x: x[0])

    # Extract results
    tool_results: dict[str, list[str]] = {}
    for match in re.finditer(result_pattern, context, re.DOTALL):
        tool_name = match.group(1).replace("mcp__magaldi__", "")
        output = match.group(2)[:2000]  # Keep more context
        if tool_name not in tool_results:
            tool_results[tool_name] = []
        tool_results[tool_name].append(output)

    # Build tool sequence with parsed parameters
    for idx, (pos, tool_name, params_str, is_magaldi) in enumerate(all_invocations):
        params = {}
        for param_match in re.finditer(param_pattern, params_str):
            params[param_match.group(1)] = param_match.group(2)

        result_snippet = ""
        if tool_name in tool_results and len(tool_results[tool_name]) > 0:
            # Pop first result for this tool
            result_snippet = tool_results[tool_name].pop(0)

        tool_sequence.append(ToolCall(
            tool_name=tool_name,
            params=params,
            result_snippet=result_snippet,
            position=idx,
            is_magaldi=is_magaldi,
        ))

    # Filter if focus_tools specified
    if focus_tools:
        tool_sequence = [t for t in tool_sequence if t.tool_name in focus_tools or not t.is_magaldi]

    # ==========================================================================
    # STEP 2: Analyze transitions and detect deviation patterns
    # ==========================================================================

    deviation_patterns: list[dict[str, Any]] = []
    improvement_suggestions: list[dict[str, Any]] = []

    for i in range(len(tool_sequence) - 1):
        current = tool_sequence[i]
        next_tool = tool_sequence[i + 1]

        # Pattern 1: Magaldi search → Builtin Read/Grep (fallback pattern)
        if current.is_magaldi and not next_tool.is_magaldi:
            if next_tool.tool_name in ("Read", "Grep", "Glob"):
                # What file/pattern was accessed after magaldi?
                target = next_tool.params.get("file_path") or next_tool.params.get("pattern") or next_tool.params.get("path", "")

                # Was this target in magaldi results?
                target_in_results = target and target in current.result_snippet

                # Extract just the filename for readability
                target_short = target.split("/")[-1] if "/" in target else target

                if not target_in_results and target:
                    deviation_patterns.append({
                        "type": "fallback_to_builtin",
                        "severity": "medium",
                        "what_happened": (
                            f"Called {current.tool_name}('{current.params.get('query', '')[:50]}'), "
                            f"then immediately used {next_tool.tool_name} to access '{target_short}'"
                        ),
                        "why_this_matters": (
                            f"The file '{target_short}' was NOT in {current.tool_name} results. "
                            f"User had to fall back to builtin {next_tool.tool_name} to find what they needed. "
                            "This suggests the search results were missing relevant files."
                        ),
                        "magaldi_tool": current.tool_name,
                        "magaldi_query": current.params.get("query") or current.params.get("pattern") or current.params.get("hash_id", ""),
                        "builtin_tool": next_tool.tool_name,
                        "builtin_target": target,
                        "potential_cause": (
                            "Possible reasons: 1) The target file doesn't contain the search terms directly, "
                            "2) The target CALLS/WRAPS functions that match, but isn't indexed that way, "
                            "3) Search relevance ranking placed it too low"
                        ),
                    })

                    # Generate specific improvement suggestion
                    if current.tool_name == "search_code":
                        improvement_suggestions.append({
                            "tool": "search_code",
                            "priority": "high",
                            "issue": f"Search for '{current.params.get('query', '')[:50]}' didn't include '{target_short}'",
                            "observed_behavior": (
                                f"User searched for '{current.params.get('query', '')[:50]}', got results, "
                                f"but then had to manually Read '{target_short}' which wasn't in results."
                            ),
                            "suggested_improvement": (
                                "search_code could automatically include 'related files' that call/import/wrap "
                                "the found functions. When a function definition is found, also show files that USE it."
                            ),
                            "implementation_hint": (
                                "Use find_callers() on search results to discover files that reference them. "
                                "Add these as a 'related_files' section in the response."
                            ),
                        })

        # Pattern 2: Magaldi search → Magaldi search (refinement pattern)
        if current.is_magaldi and next_tool.is_magaldi:
            if current.tool_name == "search_code" and next_tool.tool_name == "search_code":
                current_query = current.params.get("query", "")
                next_query = next_tool.params.get("query", "")

                if current_query and next_query and current_query != next_query:
                    # Queries are different - analyze why
                    # Check if second query terms appear in first results
                    next_terms = set(next_query.lower().split())
                    first_result_lower = current.result_snippet.lower()
                    missing_terms = [t for t in next_terms if t not in first_result_lower and len(t) > 3]

                    deviation_patterns.append({
                        "type": "query_refinement",
                        "severity": "low",
                        "what_happened": (
                            f"Searched for '{current_query[:40]}', then searched again for '{next_query[:40]}'"
                        ),
                        "why_this_matters": (
                            "User changed their search query, suggesting the first results didn't "
                            "lead them to what they needed. This is normal during exploration, "
                            "but frequent refinements may indicate relevance issues."
                        ),
                        "first_query": current_query,
                        "second_query": next_query,
                        "terms_missing_from_first_results": missing_terms[:5] if missing_terms else [],
                        "analysis": (
                            f"The second query contained terms {missing_terms[:3]} that weren't in first results."
                            if missing_terms else
                            "Query changed direction - user may have been exploring different aspects."
                        ),
                    })

                    if missing_terms:
                        improvement_suggestions.append({
                            "tool": "search_code",
                            "priority": "medium",
                            "issue": f"Search for '{current_query[:40]}' didn't help find '{next_query[:40]}'",
                            "observed_behavior": (
                                f"First search returned results, but user immediately searched for "
                                f"different terms: {missing_terms[:3]}. These terms weren't in first results."
                            ),
                            "suggested_improvement": (
                                "search_code could show 'Related searches' or 'See also' suggestions "
                                "based on semantic similarity to help users find related concepts."
                            ),
                            "implementation_hint": (
                                "After search, use embedding similarity to find related queries/concepts "
                                "from the glossary or feature labels. Show as 'You might also search for: ...'"
                            ),
                        })

        # Pattern 3: search_code → get_element (expected but frequent = needs more detail)
        if current.is_magaldi and next_tool.is_magaldi:
            if current.tool_name == "search_code" and next_tool.tool_name == "get_element":
                # This is expected flow, but if search didn't have include_code=true, note it
                if current.params.get("include_code") != "true":
                    deviation_patterns.append({
                        "type": "detail_needed",
                        "severity": "info",
                        "what_happened": (
                            f"search_code('{current.params.get('query', '')[:40]}') was followed by "
                            f"get_element('{next_tool.params.get('hash_id', '')[:20]}...')"
                        ),
                        "why_this_matters": (
                            "This is NORMAL workflow - search found something, user wanted more detail. "
                            "However, if this happens frequently, search results might benefit from "
                            "including more context by default (summaries, signatures, or code snippets)."
                        ),
                        "from_tool": "search_code",
                        "to_tool": "get_element",
                        "note": "This is expected behavior, not necessarily a problem.",
                    })

    # Pattern 4: Multiple sequential searches without get_element (exploration without finding)
    search_streak = 0
    for tool in tool_sequence:
        if tool.is_magaldi and tool.tool_name == "search_code":
            search_streak += 1
        elif tool.is_magaldi and tool.tool_name in ("get_element", "find_usages", "get_context"):
            search_streak = 0

    if search_streak >= 3:
        improvement_suggestions.append({
            "tool": "search_code",
            "priority": "medium",
            "issue": f"{search_streak} consecutive searches without drilling into any result",
            "observed_behavior": (
                f"User performed {search_streak} search_code calls in a row without using "
                "get_element, find_usages, or get_context on any result. This suggests "
                "either exploratory browsing OR difficulty finding relevant code."
            ),
            "suggested_improvement": (
                "Consider these enhancements:\n"
                "1. Suggest search_features for high-level codebase exploration\n"
                "2. Improve result summaries to help users identify relevant matches faster\n"
                "3. Add 'Related searches' suggestions based on query patterns\n"
                "4. Show confidence scores to indicate result relevance"
            ),
            "implementation_hint": (
                "Track search patterns in MCP analytics. When user searches repeatedly "
                "without drilling in, proactively suggest: 'Try search_features for broader exploration'"
            ),
        })

    # ==========================================================================
    # STEP 3: Analyze what information was actually used from results
    # ==========================================================================

    usage_analysis: list[dict[str, Any]] = []

    for tool in tool_sequence:
        if not tool.is_magaldi or not tool.result_snippet:
            continue

        # Find context AFTER this tool's result
        result_pos = context.find(tool.result_snippet[:100])
        context_after = context[result_pos + 100:] if result_pos >= 0 else ""

        # Extract key identifiers from result
        hash_ids = re.findall(r'id:([a-f0-9]{20,})', tool.result_snippet)
        file_paths = re.findall(r'[\w/]+\.(?:py|ts|js|tsx|jsx|rs|php)', tool.result_snippet)
        element_names = re.findall(r'\[(?:function|method|class)\]\s+(\w+)', tool.result_snippet)

        used_ids = [h for h in hash_ids[:5] if h in context_after]
        used_paths = [p for p in file_paths[:5] if p in context_after]
        used_names = [n for n in element_names[:5] if n in context_after]

        usage_analysis.append({
            "tool": tool.tool_name,
            "query": tool.params.get("query") or tool.params.get("hash_id") or tool.params.get("pattern", ""),
            "results_returned": len(hash_ids) + len(element_names),
            "results_used": len(used_ids) + len(used_paths) + len(used_names),
            "used_items": (used_ids + used_paths + used_names)[:5],
            "utilization": f"{len(used_ids) + len(used_names)}/{len(hash_ids) + len(element_names)}" if hash_ids or element_names else "N/A",
        })

    # ==========================================================================
    # STEP 4: Include analytics if requested
    # ==========================================================================

    analytics_summary = None
    if include_analytics and analytics_repo:
        try:
            tool_counts = analytics_repo.get_tool_counts()
            top_transitions = analytics_repo.get_top_transitions(limit=20)
            causal_stats = analytics_repo.get_causal_statistics()

            # Analyze transition patterns from analytics
            # get_top_transitions returns list[tuple[from_tool, to_tool, count]]
            transition_insights: list[str] = []
            for from_tool, to_tool, count in top_transitions[:10]:
                if from_tool.startswith(("search_", "find_", "get_")) and to_tool in ("Read", "Grep", "Glob"):
                    transition_insights.append(
                        f"{from_tool} → {to_tool}: {count}x (potential gap in {from_tool} results)"
                    )

            analytics_summary = {
                "total_magaldi_calls": sum(v for k, v in tool_counts.items()
                                          if k.startswith(("search_", "find_", "get_", "list_", "explain_"))),
                "total_builtin_fallbacks": sum(v for k, v in tool_counts.items()
                                               if k in ("Read", "Grep", "Glob")),
                "causal_link_rate": causal_stats.get("causal_rate", 0),
                "top_transitions": transition_insights[:5],
                "top_tools": dict(sorted(tool_counts.items(), key=lambda x: -x[1])[:10]),
            }

            # Add analytics-based suggestions
            fallback_rate = analytics_summary["total_builtin_fallbacks"] / max(analytics_summary["total_magaldi_calls"], 1)
            if fallback_rate > 0.3:
                improvement_suggestions.append({
                    "tool": "general",
                    "issue": f"High fallback rate to builtin tools ({fallback_rate:.0%})",
                    "suggestion": "Magaldi tools may not be returning sufficient information",
                    "action": "Consider: 1) Including file paths in search results, 2) Showing code snippets by default, 3) Better relevance ranking",
                })

        except Exception as e:
            analytics_summary = {"error": str(e)}

    # ==========================================================================
    # STEP 5: Build final results
    # ==========================================================================

    # Build verbose summary
    magaldi_count = sum(1 for t in tool_sequence if t.is_magaldi)
    builtin_count = sum(1 for t in tool_sequence if not t.is_magaldi)
    high_priority = sum(1 for s in improvement_suggestions if s.get("priority") == "high")
    medium_priority = sum(1 for s in improvement_suggestions if s.get("priority") == "medium")

    fallback_rate = builtin_count / max(magaldi_count, 1)

    return {
        "note": (
            "=== MCP SELF-REVIEW RESULTS ===\n\n"
            "These are SUGGESTIONS based on observed tool usage patterns. "
            "Review each suggestion and decide whether it's worth implementing.\n\n"
            "IMPORTANT: Not all deviations indicate problems!\n"
            "- 'fallback_to_builtin' MAY indicate missing results, OR normal workflow\n"
            "- 'query_refinement' MAY indicate relevance issues, OR natural exploration\n"
            "- 'detail_needed' is usually EXPECTED behavior (search → get details)\n\n"
            "Focus on HIGH priority suggestions first. Look for patterns that repeat."
        ),
        "executive_summary": {
            "verdict": (
                "NEEDS ATTENTION: High fallback rate to builtin tools"
                if fallback_rate > 0.4 else
                "MODERATE: Some fallbacks detected, review suggestions"
                if fallback_rate > 0.2 else
                "GOOD: Magaldi tools handled most requests effectively"
            ),
            "total_tool_calls": len(tool_sequence),
            "magaldi_calls": magaldi_count,
            "builtin_fallbacks": builtin_count,
            "fallback_rate": f"{fallback_rate:.0%}",
            "fallback_interpretation": (
                f"{builtin_count} times a builtin tool (Read/Grep/Glob) was used. "
                f"This is {fallback_rate:.0%} of total calls. "
                + (
                    "High fallback rate suggests magaldi results may be missing relevant information."
                    if fallback_rate > 0.3 else
                    "Some fallbacks are normal - users often need full file context."
                )
            ),
            "suggestions_by_priority": {
                "high": high_priority,
                "medium": medium_priority,
                "total": len(improvement_suggestions),
            },
            "patterns_detected": len(deviation_patterns),
        },
        "tool_sequence": [
            {
                "step": i + 1,
                "tool": t.tool_name,
                "is_magaldi": t.is_magaldi,
                "params": {k: v[:50] + "..." if isinstance(v, str) and len(v) > 50 else v for k, v in t.params.items()},
            }
            for i, t in enumerate(tool_sequence[:20])
        ],
        "deviation_patterns": deviation_patterns,
        "usage_analysis": usage_analysis,
        "improvement_suggestions": improvement_suggestions,
        "analytics_summary": analytics_summary,
    }
