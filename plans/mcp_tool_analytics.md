# MCP Tool Usage Analytics

## Overview

Track MCP tool call patterns to understand:
1. **Tool call counts** - How often each tool is used
2. **Tool transitions** - Which tool is called after which (to understand workflows)
3. **Usage insights** - Identify which response data is actually needed to optimize token usage

## Data Model

### Redis Keys

```
magaldi:mcp:tool_calls              # Hash: tool_name -> call_count
magaldi:mcp:tool_transitions        # Hash: "from_tool:to_tool" -> count
magaldi:mcp:session:{session_id}    # String: last_tool_called (TTL: 1 hour)
magaldi:mcp:daily:{date}:calls      # Hash: tool_name -> count (TTL: 30 days)
magaldi:mcp:daily:{date}:transitions # Hash: transition -> count (TTL: 30 days)
```

### Session Tracking

The MCP protocol doesn't have explicit sessions, but we can infer them from:
- Use a sliding window (e.g., 60 seconds) - if no call within window, it's a new session
- Track `last_tool` per "session" to compute transitions

## Implementation Plan

### Phase 1: Backend - Redis Analytics Repository

**File: `src/shared/db/redis.py`**

Add `RedisMCPAnalyticsRepository` class with methods:
- `record_tool_call(tool_name: str, session_id: str | None)` - Increment counters, track transitions
- `get_tool_counts() -> dict[str, int]` - Get all tool call counts
- `get_tool_transitions() -> dict[str, dict[str, int]]` - Get transition matrix
- `get_top_tools(limit: int) -> list[tuple[str, int]]` - Most used tools
- `get_top_transitions(limit: int) -> list[tuple[str, str, int]]` - Most common sequences
- `clear_analytics()` - Reset all counters (admin only)

### Phase 2: MCP Server Integration

**File: `src/magaldi_mcp/server.py`**

Modify `_register_tools()` and `call_tool()`:
1. Add `RedisMCPAnalyticsRepository` instance to server
2. Generate a session ID based on process/connection
3. Before executing tool, call `record_tool_call()`
4. Track last tool called to compute transitions

```python
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    # Record analytics
    if self.analytics_repo:
        self.analytics_repo.record_tool_call(name, self._session_id)
    # ... existing code
```

### Phase 3: Backend API Endpoint

**File: `src/magaldi_web/routes/admin.py`**

Add endpoint:
```python
@router.get("/admin/mcp-analytics", response_model=MCPAnalyticsResponse)
async def get_mcp_analytics() -> MCPAnalyticsResponse:
    """Get MCP tool usage analytics."""
```

### Phase 4: Pydantic Models

**File: `src/magaldi_web/models.py`**

```python
class ToolUsageInfo(BaseModel):
    """Usage info for a single tool."""
    tool_name: str
    call_count: int
    percentage: float  # % of total calls

class ToolTransitionInfo(BaseModel):
    """Transition from one tool to another."""
    from_tool: str
    to_tool: str
    count: int
    percentage: float  # % of transitions from from_tool

class MCPAnalyticsResponse(BaseModel):
    """Response for MCP analytics endpoint."""
    total_calls: int
    tool_usage: list[ToolUsageInfo]
    top_transitions: list[ToolTransitionInfo]
    transition_matrix: dict[str, dict[str, int]]  # from_tool -> to_tool -> count
```

### Phase 5: Frontend API

**File: `src/magaldi_web/frontend/src/api.ts`**

```typescript
export interface ToolUsageInfo {
  tool_name: string
  call_count: number
  percentage: number
}

export interface ToolTransitionInfo {
  from_tool: string
  to_tool: string
  count: number
  percentage: number
}

export interface MCPAnalyticsResponse {
  total_calls: number
  tool_usage: ToolUsageInfo[]
  top_transitions: ToolTransitionInfo[]
  transition_matrix: Record<string, Record<string, number>>
}

export async function getMCPAnalytics(): Promise<MCPAnalyticsResponse> {
  const response = await fetch(`${API_BASE}/admin/mcp-analytics`)
  return response.json()
}
```

### Phase 6: Frontend UI

**File: `src/magaldi_web/frontend/src/pages/Admin.tsx`**

Add new section "MCP Tool Analytics" with:

1. **Summary Stats Card**
   - Total tool calls
   - Most used tool
   - Active sessions (if tracked)

2. **Tool Usage Table/Chart**
   - Bar chart showing call counts per tool
   - Sortable table with tool name, count, percentage

3. **Tool Transitions Visualization**
   - Option A: Heatmap/Matrix showing from→to transitions
   - Option B: Simple table of top N transitions
   - Shows patterns like "search_code → get_element → get_context"

4. **Insights Panel**
   - "Tools rarely followed by other tools" (terminal tools)
   - "Common workflows" (frequent sequences)

## UI Mockup

```
┌─────────────────────────────────────────────────────────────────┐
│ MCP Tool Analytics                                    [Refresh] │
├─────────────────────────────────────────────────────────────────┤
│ Total Calls: 1,234    |    Unique Tools: 25    |    Today: 156  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Tool Usage                          Top Transitions            │
│  ─────────────────                   ────────────────────────   │
│  search_code      ████████ 234       search_code → get_element  │
│  get_element      ██████   156       get_element → get_context  │
│  find_usages      █████    123       search_code → find_usages  │
│  get_context      ████     98        find_files → get_file_...  │
│  pattern_search   ███      67        get_context → get_element  │
│  find_files       ██       45                                   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ Transition Matrix (click to expand)                             │
│ ┌──────────────┬────────┬─────────┬───────────┬──────────────┐ │
│ │              │search_ │get_elem │get_context│find_usages   │ │
│ ├──────────────┼────────┼─────────┼───────────┼──────────────┤ │
│ │ search_code  │   -    │   45    │    12     │     34       │ │
│ │ get_element  │   8    │    -    │    28     │     15       │ │
│ │ get_context  │   5    │   22    │     -     │      8       │ │
│ └──────────────┴────────┴─────────┴───────────┴──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Token Optimization Insights

Based on transition data, we can identify:

1. **High-frequency sequences**: If `search_code → get_element` is common, consider:
   - Adding element details to `search_code` results (with `include_details` flag)
   - Pre-computing commonly needed follow-up data

2. **Terminal tools**: Tools that are rarely followed by others might return more data than needed

3. **Redundant sequences**: If `get_element → get_context` always happens, merge the responses

## Files to Modify

1. `src/shared/db/redis.py` - Add `RedisMCPAnalyticsRepository`
2. `src/magaldi_mcp/server.py` - Add analytics tracking
3. `src/magaldi_web/routes/admin.py` - Add API endpoint
4. `src/magaldi_web/models.py` - Add response models
5. `src/magaldi_web/frontend/src/api.ts` - Add API function + types
6. `src/magaldi_web/frontend/src/pages/Admin.tsx` - Add UI section

## Testing

1. Unit tests for `RedisMCPAnalyticsRepository`
2. Integration test for MCP server tracking
3. API endpoint tests
4. Manual UI testing

## Future Enhancements

- Time-series data for trends (hourly/daily aggregates)
- Per-repository analytics
- Export to CSV/JSON
- Alerts for unusual patterns
