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
  ToolTransitionInfo,
  TransitionDetailInfo,
} from '../api'

// Chart colors
const COLORS = ['#0d6efd', '#6610f2', '#6f42c1', '#d63384', '#dc3545', '#fd7e14', '#ffc107', '#198754', '#20c997', '#0dcaf0']

function MCPAnalysis() {
  const [historyDays, setHistoryDays] = useState(7)
  const [expandedTransition, setExpandedTransition] = useState<string | null>(null)
  const [transitionFilter, setTransitionFilter] = useState<{ from?: string; to?: string }>({})

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
      // Expand and fetch details
      setExpandedTransition(key)
      setTransitionFilter({ from: transition.from_tool, to: transition.to_tool })
    }
  }

  const formatTimestamp = (ts: string | null) => {
    if (!ts) return '-'
    try {
      return new Date(ts).toLocaleString()
    } catch {
      return ts
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

              {/* Row 2: Tables */}
              <Row>
                {/* Tool Usage Table */}
                <Col md={6}>
                  <h6 className="text-muted mb-3">
                    <i className="bi bi-list-ol me-2"></i>
                    Tool Usage (Total: {mcpAnalytics.total_calls.toLocaleString()})
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
                            return (
                              <tr key={tool.tool_name}>
                                <td className="text-muted">{idx + 1}</td>
                                <td>
                                  <code className="small">{tool.tool_name}</code>
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
                            <th className="text-end">Count</th>
                          </tr>
                        </thead>
                        <tbody>
                          {mcpAnalytics.top_transitions.map((transition, idx) => {
                            const key = `${transition.from_tool}->${transition.to_tool}`
                            const isExpanded = expandedTransition === key
                            return (
                              <tr
                                key={idx}
                                onClick={() => handleTransitionClick(transition)}
                                style={{ cursor: 'pointer' }}
                                className={isExpanded ? 'table-primary' : ''}
                              >
                                <td className="text-muted">{idx + 1}</td>
                                <td>
                                  <code className="small">{transition.from_tool}</code>
                                  <i className={`bi bi-arrow-right mx-2 ${isExpanded ? 'text-primary' : 'text-muted'}`}></i>
                                  <code className="small">{transition.to_tool}</code>
                                  {isExpanded && <i className="bi bi-chevron-down ms-2"></i>}
                                </td>
                                <td className="text-end">{transition.count.toLocaleString()}</td>
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
                                <th style={{ minWidth: '80px' }}>Gap</th>
                                <th style={{ minWidth: '130px' }}>Time</th>
                              </tr>
                            </thead>
                            <tbody>
                              {transitionDetails.transitions.map((detail: TransitionDetailInfo, idx: number) => (
                                <tr key={idx}>
                                  <td>
                                    <pre className="mb-0 small" style={{
                                      maxHeight: '200px',
                                      overflowY: 'auto',
                                      whiteSpace: 'pre-wrap',
                                      wordBreak: 'break-word',
                                      backgroundColor: '#e7f5ff',
                                      padding: '8px',
                                      borderRadius: '4px',
                                    }}>
                                      {detail.caller_input || <em className="text-muted">No input</em>}
                                    </pre>
                                  </td>
                                  <td className="text-center align-middle">
                                    <code className="small fw-bold">{detail.caller}</code>
                                    {detail.caller_duration_ms != null && (
                                      <small className="text-muted d-block">
                                        {detail.caller_duration_ms}ms
                                      </small>
                                    )}
                                  </td>
                                  <td>
                                    <pre className="mb-0 small" style={{
                                      maxHeight: '200px',
                                      overflowY: 'auto',
                                      whiteSpace: 'pre-wrap',
                                      wordBreak: 'break-word',
                                      backgroundColor: '#e6fcf5',
                                      padding: '8px',
                                      borderRadius: '4px',
                                    }}>
                                      {detail.caller_output || <em className="text-muted">No output</em>}
                                    </pre>
                                  </td>
                                  <td>
                                    <pre className="mb-0 small" style={{
                                      maxHeight: '200px',
                                      overflowY: 'auto',
                                      whiteSpace: 'pre-wrap',
                                      wordBreak: 'break-word',
                                      backgroundColor: '#fff9db',
                                      padding: '8px',
                                      borderRadius: '4px',
                                    }}>
                                      {detail.callee_input || <em className="text-muted">No input</em>}
                                    </pre>
                                  </td>
                                  <td className="text-center align-middle">
                                    <code className="small fw-bold">{detail.callee}</code>
                                    {detail.callee_duration_ms != null && (
                                      <small className="text-muted d-block">
                                        {detail.callee_duration_ms}ms
                                      </small>
                                    )}
                                  </td>
                                  <td>
                                    <pre className="mb-0 small" style={{
                                      maxHeight: '200px',
                                      overflowY: 'auto',
                                      whiteSpace: 'pre-wrap',
                                      wordBreak: 'break-word',
                                      backgroundColor: '#ffe3e3',
                                      padding: '8px',
                                      borderRadius: '4px',
                                    }}>
                                      {detail.callee_output || <em className="text-muted">No output</em>}
                                    </pre>
                                  </td>
                                  <td className="text-end align-middle">
                                    {detail.elapsed_ms != null ? (
                                      <span className={detail.elapsed_ms > 1000 ? 'text-warning fw-bold' : ''}>
                                        {detail.elapsed_ms}ms
                                      </span>
                                    ) : '-'}
                                  </td>
                                  <td className="small text-muted align-middle">
                                    {formatTimestamp(detail.timestamp)}
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
