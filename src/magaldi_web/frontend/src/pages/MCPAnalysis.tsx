import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Row,
  Col,
  Card,
  Spinner,
  Alert,
  Table,
  Button,
  Form,
  InputGroup,
  Collapse,
} from 'react-bootstrap'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts'
import {
  getMCPAnalytics,
  getMCPActivityHistory,
  clearMCPAnalytics,
  getMCPTransitionDetails,
  getMCPRecentCalls,
  getMCPCausalStatistics,
  ToolTransitionInfo,
  TransitionDetailInfo,
  ToolUsageInfo,
  RecentToolCallInfo,
  CausalLinkInfo,
} from '../api'

// Chart colors
const COLORS = ['#0d6efd', '#6610f2', '#6f42c1', '#d63384', '#dc3545', '#fd7e14', '#ffc107', '#198754', '#20c997', '#0dcaf0']

// Helper component for displaying causal link badge
function CausalBadge({ link }: { link: CausalLinkInfo }) {
  const isParameter = link.match_type === 'parameter'
  return (
    <span
      className={`badge ${isParameter ? 'bg-success' : 'bg-info'} me-1`}
      title={isParameter
        ? `Triggered by ${link.tool_name} via parameter: ${link.matched_value}`
        : `Triggered by ${link.tool_name} (tool suggestion)`
      }
    >
      <i className={`bi ${isParameter ? 'bi-link-45deg' : 'bi-chat-dots'} me-1`}></i>
      {link.tool_name}
      {isParameter && link.matched_value && (
        <code className="ms-1 text-white" style={{ fontSize: '0.7em' }}>
          {link.matched_value.length > 20 ? link.matched_value.slice(0, 20) + '...' : link.matched_value}
        </code>
      )}
    </span>
  )
}

function MCPAnalysis() {
  const [historyDays, setHistoryDays] = useState(7)
  const [expandedTransition, setExpandedTransition] = useState<string | null>(null)
  const [transitionFilter, setTransitionFilter] = useState<{ from?: string; to?: string }>({})
  const [expandedTool, setExpandedTool] = useState<string | null>(null)

  const { data: mcpAnalytics, isLoading: analyticsLoading, refetch: refetchAnalytics } = useQuery({
    queryKey: ['mcpAnalytics'],
    queryFn: getMCPAnalytics,
    refetchInterval: 30000,
  })

  const { data: activityHistory, isLoading: historyLoading, refetch: refetchHistory } = useQuery({
    queryKey: ['mcpActivityHistory', historyDays],
    queryFn: () => getMCPActivityHistory(historyDays),
    refetchInterval: 60000,
  })

  // Query for transition details - triggered when a transition is clicked
  const { data: transitionDetails, isLoading: detailsLoading } = useQuery({
    queryKey: ['mcpTransitionDetails', transitionFilter.from, transitionFilter.to],
    queryFn: () => getMCPTransitionDetails({
      from_tool: transitionFilter.from,
      to_tool: transitionFilter.to,
      limit: 100,
    }),
    enabled: !!transitionFilter.from && !!transitionFilter.to,
  })

  // Query for tool details - triggered when a tool is clicked
  const { data: toolDetails, isLoading: toolDetailsLoading } = useQuery({
    queryKey: ['mcpToolDetails', expandedTool],
    queryFn: () => getMCPRecentCalls({
      tool_name: expandedTool!,
      limit: 100,
    }),
    enabled: !!expandedTool,
  })

  // Query for causal statistics
  const { data: causalStats } = useQuery({
    queryKey: ['mcpCausalStats'],
    queryFn: getMCPCausalStatistics,
    refetchInterval: 30000,
  })

  const handleDaysChange = (newDays: number) => {
    const maxDays = activityHistory?.max_days || 30
    const validDays = Math.max(1, Math.min(newDays, maxDays))
    setHistoryDays(validDays)
  }

  const handleClearAnalytics = async () => {
    if (window.confirm('Are you sure you want to clear all MCP analytics data?')) {
      await clearMCPAnalytics()
      refetchAnalytics()
    }
  }

  const handleTransitionClick = (transition: ToolTransitionInfo) => {
    const key = `${transition.from_tool}->${transition.to_tool}`
    if (expandedTransition === key) {
      // Collapse if clicking same transition
      setExpandedTransition(null)
      setTransitionFilter({})
    } else {
      // Expand and fetch details (collapse tool details if open)
      setExpandedTransition(key)
      setTransitionFilter({ from: transition.from_tool, to: transition.to_tool })
      setExpandedTool(null)
    }
  }

  const handleToolClick = (tool: ToolUsageInfo) => {
    if (expandedTool === tool.tool_name) {
      // Collapse if clicking same tool
      setExpandedTool(null)
    } else {
      // Expand and fetch details (collapse transition details if open)
      setExpandedTool(tool.tool_name)
      setExpandedTransition(null)
      setTransitionFilter({})
    }
  }

  return (
    <div>
      <h1 className="mb-4">
        <i className="bi bi-graph-up me-2"></i>
        MCP Tool Analysis
      </h1>

      {/* MCP Tool Analytics */}
      <Card>
        <Card.Header className="d-flex justify-content-between align-items-center">
          <span>
            <i className="bi bi-activity me-2"></i>
            Tool Usage Analytics
          </span>
          <div className="d-flex align-items-center gap-2">
            <InputGroup size="sm" style={{ width: '140px' }}>
              <InputGroup.Text>Days</InputGroup.Text>
              <Form.Control
                type="number"
                min={1}
                max={activityHistory?.max_days || 30}
                value={historyDays}
                onChange={(e) => handleDaysChange(parseInt(e.target.value) || 7)}
              />
            </InputGroup>
            <Button
              variant="outline-secondary"
              size="sm"
              onClick={() => { refetchAnalytics(); refetchHistory(); }}
            >
              <i className="bi bi-arrow-clockwise"></i>
            </Button>
            <Button
              variant="outline-danger"
              size="sm"
              onClick={handleClearAnalytics}
            >
              Clear
            </Button>
          </div>
        </Card.Header>
        <Card.Body>
          {(analyticsLoading || historyLoading) ? (
            <div className="text-center py-3">
              <Spinner animation="border" size="sm" />
            </div>
          ) : !mcpAnalytics || mcpAnalytics.total_calls === 0 ? (
            <p className="text-muted mb-0 text-center">
              <i className="bi bi-info-circle me-2"></i>
              No MCP tool usage data yet. Data is collected when tools are called via the MCP server.
            </p>
          ) : (
            <>
              {/* Row 1: Charts */}
              <Row className="mb-4">
                {/* Tool Distribution Pie Chart */}
                <Col md={6}>
                  <h6 className="text-muted mb-3">
                    <i className="bi bi-pie-chart me-2"></i>
                    Tool Distribution
                  </h6>
                  <div style={{ height: 300 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={mcpAnalytics.tool_usage.slice(0, 8).map((tool, idx) => ({
                            name: tool.tool_name,
                            value: tool.call_count,
                            fill: COLORS[idx % COLORS.length],
                          }))}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={100}
                          paddingAngle={2}
                          dataKey="value"
                          label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                          labelLine={false}
                        >
                          {mcpAnalytics.tool_usage.slice(0, 8).map((_, idx) => (
                            <Cell key={`cell-${idx}`} fill={COLORS[idx % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip formatter={(value: number) => value.toLocaleString()} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </Col>

                {/* Daily Activity Line Chart */}
                <Col md={6}>
                  <h6 className="text-muted mb-3">
                    <i className="bi bi-graph-up-arrow me-2"></i>
                    Daily Activity ({historyDays} days)
                  </h6>
                  <div style={{ height: 300 }}>
                    {activityHistory && activityHistory.daily_activity.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart
                          data={activityHistory.daily_activity.map(day => ({
                            date: new Date(day.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
                            calls: day.total_calls,
                          }))}
                          margin={{ top: 5, right: 20, left: 0, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                          <XAxis
                            dataKey="date"
                            tick={{ fontSize: 11 }}
                            interval="preserveStartEnd"
                          />
                          <YAxis tick={{ fontSize: 11 }} />
                          <Tooltip
                            formatter={(value: number) => [value.toLocaleString(), 'Calls']}
                          />
                          <Line
                            type="monotone"
                            dataKey="calls"
                            stroke="#0d6efd"
                            strokeWidth={2}
                            dot={{ fill: '#0d6efd', strokeWidth: 2 }}
                            activeDot={{ r: 6 }}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="d-flex align-items-center justify-content-center h-100 text-muted">
                        No activity data for this period
                      </div>
                    )}
                  </div>
                </Col>
              </Row>

              {/* Causal Statistics Summary */}
              {causalStats && causalStats.total_causal_links > 0 && (
                <Row className="mb-4">
                  <Col>
                    <Card className="border-success">
                      <Card.Header className="bg-success text-white py-2">
                        <i className="bi bi-diagram-3 me-2"></i>
                        Causal Relationships
                        <small className="ms-2 opacity-75">
                          ({causalStats.causal_rate}% of calls triggered by previous calls)
                        </small>
                      </Card.Header>
                      <Card.Body className="py-2">
                        <Row>
                          <Col md={3}>
                            <div className="text-center">
                              <div className="fs-4 fw-bold text-success">{causalStats.total_causal_links}</div>
                              <small className="text-muted">Causal Links</small>
                            </div>
                          </Col>
                          <Col md={3}>
                            <div className="text-center">
                              <div className="fs-4 fw-bold text-primary">{causalStats.by_match_type.parameter || 0}</div>
                              <small className="text-muted">
                                <i className="bi bi-link-45deg me-1"></i>
                                Parameter Match
                              </small>
                            </div>
                          </Col>
                          <Col md={3}>
                            <div className="text-center">
                              <div className="fs-4 fw-bold text-info">{causalStats.by_match_type.tool_suggestion || 0}</div>
                              <small className="text-muted">
                                <i className="bi bi-chat-dots me-1"></i>
                                Tool Suggestion
                              </small>
                            </div>
                          </Col>
                          <Col md={3}>
                            <div className="text-muted small">
                              <strong>Top Causal Pairs:</strong>
                              <div style={{ maxHeight: '60px', overflow: 'auto' }}>
                                {causalStats.top_causal_pairs.slice(0, 3).map((pair, idx) => (
                                  <div key={idx}>
                                    <code className="small">{pair.source}</code>
                                    <i className="bi bi-arrow-right mx-1"></i>
                                    <code className="small">{pair.target}</code>
                                    <span className="text-muted ms-1">({pair.count})</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </Col>
                        </Row>
                      </Card.Body>
                    </Card>
                  </Col>
                </Row>
              )}

              {/* Row 2: Tables */}
              <Row>
                {/* Tool Usage Table (Clickable) */}
                <Col md={6}>
                  <h6 className="text-muted mb-3">
                    <i className="bi bi-list-ol me-2"></i>
                    Tool Usage (Total: {mcpAnalytics.total_calls.toLocaleString()})
                    <small className="text-info ms-2">(click row for details)</small>
                  </h6>
                  {mcpAnalytics.tool_usage.length > 0 ? (
                    <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                      <Table size="sm" className="mb-0" hover>
                        <thead className="sticky-top bg-body">
                          <tr>
                            <th>#</th>
                            <th>Tool</th>
                            <th className="text-end">Calls</th>
                            <th className="text-end">%</th>
                            <th className="text-end">Avg (ms)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {mcpAnalytics.tool_usage.map((tool, idx) => {
                            const duration = mcpAnalytics.tool_durations?.find(d => d.tool_name === tool.tool_name)
                            const isExpanded = expandedTool === tool.tool_name
                            return (
                              <tr
                                key={tool.tool_name}
                                onClick={() => handleToolClick(tool)}
                                style={{ cursor: 'pointer' }}
                              >
                                <td
                                  className="text-muted"
                                  style={{ borderLeft: isExpanded ? '3px solid var(--bs-primary)' : 'none' }}
                                >
                                  {idx + 1}
                                </td>
                                <td>
                                  <code className="small">{tool.tool_name}</code>
                                  {isExpanded && <i className="bi bi-chevron-down ms-2 text-primary"></i>}
                                </td>
                                <td className="text-end">{tool.call_count.toLocaleString()}</td>
                                <td className="text-end text-muted">{tool.percentage}%</td>
                                <td className="text-end text-muted">
                                  {duration ? duration.avg_ms.toLocaleString(undefined, { maximumFractionDigits: 0 }) : '-'}
                                </td>
                              </tr>
                            )
                          })}
                        </tbody>
                      </Table>
                    </div>
                  ) : (
                    <p className="text-muted small">No tool usage data</p>
                  )}
                </Col>

                {/* Transitions Table (Clickable) */}
                <Col md={6}>
                  <h6 className="text-muted mb-3">
                    <i className="bi bi-arrow-right-circle me-2"></i>
                    Tool Transitions
                    <small className="text-info ms-2">(click row for details)</small>
                  </h6>
                  {mcpAnalytics.top_transitions.length > 0 ? (
                    <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                      <Table size="sm" className="mb-0" hover>
                        <thead className="sticky-top bg-body">
                          <tr>
                            <th>#</th>
                            <th>From -&gt; To</th>
                            <th className="text-end" title="Temporal transitions (within 10s)">Sequential</th>
                            <th className="text-end" title="Confirmed causal connections (data passed between tools)">Causal</th>
                          </tr>
                        </thead>
                        <tbody>
                          {mcpAnalytics.top_transitions.map((transition, idx) => {
                            const key = `${transition.from_tool}->${transition.to_tool}`
                            const isExpanded = expandedTransition === key
                            const hasConfirmed = transition.confirmed_count > 0
                            return (
                              <tr
                                key={idx}
                                onClick={() => handleTransitionClick(transition)}
                                style={{ cursor: 'pointer' }}
                              >
                                <td
                                  className="text-muted"
                                  style={{ borderLeft: isExpanded ? '3px solid var(--bs-primary)' : 'none' }}
                                >
                                  {idx + 1}
                                </td>
                                <td>
                                  <code className="small">{transition.from_tool}</code>
                                  <i className={`bi bi-arrow-right mx-2 ${isExpanded ? 'text-primary' : 'text-muted'}`}></i>
                                  <code className="small">{transition.to_tool}</code>
                                  {isExpanded && <i className="bi bi-chevron-down ms-2 text-primary"></i>}
                                </td>
                                <td className="text-end">{transition.count.toLocaleString()}</td>
                                <td className="text-end">
                                  {hasConfirmed ? (
                                    <span
                                      className="badge bg-success"
                                      title="Confirmed causal connections (parameter or tool suggestion match)"
                                    >
                                      {transition.confirmed_count}
                                    </span>
                                  ) : (
                                    <span className="text-muted small">-</span>
                                  )}
                                </td>
                              </tr>
                            )
                          })}
                        </tbody>
                      </Table>
                    </div>
                  ) : (
                    <p className="text-muted small">No transition data yet (need at least 2 consecutive tool calls)</p>
                  )}
                </Col>
              </Row>

              {/* Transition Details (shown when a transition is clicked) */}
              <Collapse in={!!expandedTransition}>
                <div>
                  <Card className="mt-4 border-primary">
                    <Card.Header className="bg-primary text-white d-flex justify-content-between align-items-center">
                      <span>
                        <i className="bi bi-list-columns me-2"></i>
                        Transition Details: {expandedTransition}
                      </span>
                      <Button
                        variant="light"
                        size="sm"
                        onClick={() => { setExpandedTransition(null); setTransitionFilter({}); }}
                      >
                        <i className="bi bi-x-lg"></i>
                      </Button>
                    </Card.Header>
                    <Card.Body>
                      {detailsLoading ? (
                        <div className="text-center py-3">
                          <Spinner animation="border" size="sm" />
                        </div>
                      ) : !transitionDetails || transitionDetails.transitions.length === 0 ? (
                        <Alert variant="info">
                          No detailed transition data available. Transition details are recorded for new tool calls only.
                        </Alert>
                      ) : (
                        <div style={{ overflowX: 'auto' }}>
                          <Table bordered hover className="mb-0">
                            <thead className="table-dark">
                              <tr>
                                <th style={{ minWidth: '250px' }}>Caller Input</th>
                                <th style={{ minWidth: '120px' }}>Caller</th>
                                <th style={{ minWidth: '250px' }}>Caller Output</th>
                                <th style={{ minWidth: '250px' }}>Callee Input</th>
                                <th style={{ minWidth: '120px' }}>Callee</th>
                                <th style={{ minWidth: '250px' }}>Callee Output</th>
                                <th style={{ minWidth: '100px' }} className="text-center">Link / Gap</th>
                              </tr>
                            </thead>
                            <tbody>
                              {transitionDetails.transitions.map((detail: TransitionDetailInfo, idx: number) => (
                                <tr key={idx}>
                                  <td>
                                    <pre className="mb-0" style={{
                                      maxHeight: '200px',
                                      overflow: 'auto',
                                      whiteSpace: 'pre-wrap',
                                      wordBreak: 'break-word',
                                      backgroundColor: 'rgba(13, 110, 253, 0.1)',
                                      border: '1px solid rgba(13, 110, 253, 0.3)',
                                      color: 'inherit',
                                      padding: '6px',
                                      borderRadius: '4px',
                                      fontSize: '0.7rem',
                                    }}>
                                      {detail.caller_input || <em className="text-muted">No input</em>}
                                    </pre>
                                  </td>
                                  <td className="text-center align-top">
                                    <code className="small fw-bold">{detail.caller}</code>
                                    {detail.caller_duration_ms != null && (
                                      <small className="text-muted d-block">
                                        {detail.caller_duration_ms}ms
                                      </small>
                                    )}
                                  </td>
                                  <td>
                                    <pre className="mb-0" style={{
                                      maxHeight: '200px',
                                      overflow: 'auto',
                                      whiteSpace: 'pre-wrap',
                                      wordBreak: 'break-word',
                                      backgroundColor: 'rgba(25, 135, 84, 0.1)',
                                      border: '1px solid rgba(25, 135, 84, 0.3)',
                                      color: 'inherit',
                                      padding: '6px',
                                      borderRadius: '4px',
                                      fontSize: '0.7rem',
                                    }}>
                                      {detail.caller_output || <em className="text-muted">No output</em>}
                                    </pre>
                                  </td>
                                  <td>
                                    <pre className="mb-0" style={{
                                      maxHeight: '200px',
                                      overflow: 'auto',
                                      whiteSpace: 'pre-wrap',
                                      wordBreak: 'break-word',
                                      backgroundColor: 'rgba(255, 193, 7, 0.1)',
                                      border: '1px solid rgba(255, 193, 7, 0.3)',
                                      color: 'inherit',
                                      padding: '6px',
                                      borderRadius: '4px',
                                      fontSize: '0.7rem',
                                    }}>
                                      {detail.callee_input || <em className="text-muted">No input</em>}
                                    </pre>
                                  </td>
                                  <td className="text-center align-top">
                                    <code className="small fw-bold">{detail.callee}</code>
                                    {detail.callee_duration_ms != null && (
                                      <small className="text-muted d-block">
                                        {detail.callee_duration_ms}ms
                                      </small>
                                    )}
                                  </td>
                                  <td>
                                    <pre className="mb-0" style={{
                                      maxHeight: '200px',
                                      overflow: 'auto',
                                      whiteSpace: 'pre-wrap',
                                      wordBreak: 'break-word',
                                      backgroundColor: 'rgba(220, 53, 69, 0.1)',
                                      border: '1px solid rgba(220, 53, 69, 0.3)',
                                      color: 'inherit',
                                      padding: '6px',
                                      borderRadius: '4px',
                                      fontSize: '0.7rem',
                                    }}>
                                      {detail.callee_output || <em className="text-muted">No output</em>}
                                    </pre>
                                  </td>
                                  <td className="text-center align-top">
                                    <div>
                                      {detail.is_causal && (
                                        <span
                                          className="badge bg-success mb-1 me-1"
                                          title={detail.causal_match_type === 'parameter'
                                            ? `Causal: parameter match${detail.causal_matched_value ? ` (${detail.causal_matched_value.substring(0, 20)}...)` : ''}`
                                            : 'Causal: tool suggestion'}
                                        >
                                          <i className="bi bi-link-45deg me-1"></i>
                                          Causal
                                        </span>
                                      )}
                                      {detail.is_consecutive && (
                                        <span className="badge bg-secondary mb-1" title="Consecutive (temporal) transition">
                                          Temporal
                                        </span>
                                      )}
                                      {!detail.is_causal && !detail.is_consecutive && (
                                        <span className="badge bg-light text-dark mb-1" title="Unknown transition type">
                                          -
                                        </span>
                                      )}
                                    </div>
                                    {detail.elapsed_ms != null ? (
                                      <span className={`small ${detail.elapsed_ms > 1000 ? 'text-warning fw-bold' : 'text-muted'}`}>
                                        {detail.elapsed_ms}ms
                                      </span>
                                    ) : <span className="text-muted">-</span>}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </Table>
                        </div>
                      )}
                    </Card.Body>
                  </Card>
                </div>
              </Collapse>

              {/* Tool Details (shown when a tool is clicked) */}
              <Collapse in={!!expandedTool}>
                <div>
                  <Card className="mt-4 border-info">
                    <Card.Header className="bg-info text-white d-flex justify-content-between align-items-center">
                      <span>
                        <i className="bi bi-terminal me-2"></i>
                        Tool Details: {expandedTool}
                      </span>
                      <Button
                        variant="light"
                        size="sm"
                        onClick={() => setExpandedTool(null)}
                      >
                        <i className="bi bi-x-lg"></i>
                      </Button>
                    </Card.Header>
                    <Card.Body>
                      {toolDetailsLoading ? (
                        <div className="text-center py-3">
                          <Spinner animation="border" size="sm" />
                        </div>
                      ) : !toolDetails || toolDetails.calls.length === 0 ? (
                        <Alert variant="info">
                          No detailed call data available for this tool.
                        </Alert>
                      ) : (
                        <div style={{ overflowX: 'auto' }}>
                          <Table bordered hover className="mb-0">
                            <thead className="table-dark">
                              <tr>
                                <th style={{ minWidth: '300px' }}>Input</th>
                                <th style={{ minWidth: '150px' }}>Tool</th>
                                <th style={{ minWidth: '400px' }}>Output</th>
                              </tr>
                            </thead>
                            <tbody>
                              {toolDetails.calls.map((call: RecentToolCallInfo, idx: number) => (
                                <tr key={idx}>
                                  <td>
                                    <pre className="mb-0" style={{
                                      maxHeight: '200px',
                                      overflow: 'auto',
                                      whiteSpace: 'pre-wrap',
                                      wordBreak: 'break-word',
                                      backgroundColor: 'rgba(13, 110, 253, 0.1)',
                                      border: '1px solid rgba(13, 110, 253, 0.3)',
                                      color: 'inherit',
                                      padding: '6px',
                                      borderRadius: '4px',
                                      fontSize: '0.7rem',
                                    }}>
                                      {call.input_args || <em className="text-muted">No input</em>}
                                    </pre>
                                  </td>
                                  <td className="text-center align-top">
                                    <code className="small fw-bold">{call.tool_name}</code>
                                    {call.duration_ms != null && (
                                      <small className="text-muted d-block">
                                        {call.duration_ms}ms
                                      </small>
                                    )}
                                    {call.triggered_by && (
                                      <div className="mt-1">
                                        <CausalBadge link={call.triggered_by} />
                                      </div>
                                    )}
                                    {call.suggested_tools && call.suggested_tools.length > 0 && (
                                      <div className="mt-1">
                                        <small className="text-muted d-block">Suggests:</small>
                                        {call.suggested_tools.slice(0, 3).map((tool, i) => (
                                          <span key={i} className="badge bg-secondary me-1" style={{ fontSize: '0.65em' }}>
                                            {tool}
                                          </span>
                                        ))}
                                        {call.suggested_tools.length > 3 && (
                                          <span className="text-muted" style={{ fontSize: '0.65em' }}>
                                            +{call.suggested_tools.length - 3}
                                          </span>
                                        )}
                                      </div>
                                    )}
                                  </td>
                                  <td>
                                    <pre className="mb-0" style={{
                                      maxHeight: '200px',
                                      overflow: 'auto',
                                      whiteSpace: 'pre-wrap',
                                      wordBreak: 'break-word',
                                      backgroundColor: 'rgba(25, 135, 84, 0.1)',
                                      border: '1px solid rgba(25, 135, 84, 0.3)',
                                      color: 'inherit',
                                      padding: '6px',
                                      borderRadius: '4px',
                                      fontSize: '0.7rem',
                                    }}>
                                      {call.output_result || <em className="text-muted">No output</em>}
                                    </pre>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </Table>
                        </div>
                      )}
                    </Card.Body>
                  </Card>
                </div>
              </Collapse>
            </>
          )}
        </Card.Body>
      </Card>
    </div>
  )
}

export default MCPAnalysis
