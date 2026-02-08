import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Row,
  Col,
  Card,
  Badge,
  Spinner,
  Alert,
  Table,
  Button,
  Form,
  InputGroup,
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
import { getHealth, getIndexStats, getDashboard, getMCPAnalytics, getMCPActivityHistory, clearMCPAnalytics } from '../api'

// Chart colors
const COLORS = ['#0d6efd', '#6610f2', '#6f42c1', '#d63384', '#dc3545', '#fd7e14', '#ffc107', '#198754', '#20c997', '#0dcaf0']

function Admin() {
  const [historyDays, setHistoryDays] = useState(7)
  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: 10000,
  })

  const { data: dashboard, isLoading: dashboardLoading } = useQuery({
    queryKey: ['dashboard'],
    queryFn: getDashboard,
    refetchInterval: 5000,
  })

  const { data: indexStats, isLoading: statsLoading } = useQuery({
    queryKey: ['indexStats'],
    queryFn: getIndexStats,
    refetchInterval: 30000,
  })

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

  const getStatusBadge = (status: string) => {
    switch (status.toLowerCase()) {
      case 'healthy':
        return <Badge bg="success">Healthy</Badge>
      case 'degraded':
        return <Badge bg="warning">Degraded</Badge>
      case 'unhealthy':
        return <Badge bg="danger">Unhealthy</Badge>
      default:
        return <Badge bg="secondary">{status}</Badge>
    }
  }

  const getOverallStatus = (): string => {
    if (!health) return 'unknown'
    const statuses = [
      health.search.status,
      health.llm.status,
      health.redis.status,
    ]
    if (statuses.every((s) => s === 'healthy')) return 'healthy'
    if (statuses.some((s) => s === 'unhealthy')) return 'unhealthy'
    return 'degraded'
  }

  return (
    <div>
      <h1 className="mb-4">Admin Panel</h1>

      <Row className="mb-4">
        {/* Overall Status */}
        <Col md={4}>
          <Card className="h-100">
            <Card.Header>
              <i className="bi bi-heart-pulse me-2"></i>
              System Status
            </Card.Header>
            <Card.Body>
              {healthLoading ? (
                <div className="text-center py-3">
                  <Spinner animation="border" size="sm" />
                </div>
              ) : health ? (
                <>
                  <div className="d-flex justify-content-between align-items-center mb-3">
                    <span className="fs-5">Overall</span>
                    {getStatusBadge(getOverallStatus())}
                  </div>
                  <hr />
                  <Table borderless size="sm" className="mb-0">
                    <tbody>
                      <tr>
                        <td>
                          <i className="bi bi-database me-2"></i>
                          Search
                        </td>
                        <td className="text-end">
                          {getStatusBadge(health.search.status)}
                        </td>
                      </tr>
                      <tr>
                        <td>
                          <i className="bi bi-robot me-2"></i>
                          LLM
                        </td>
                        <td className="text-end">
                          {getStatusBadge(health.llm.status)}
                        </td>
                      </tr>
                      <tr>
                        <td>
                          <i className="bi bi-hdd-stack me-2"></i>
                          Redis
                        </td>
                        <td className="text-end">
                          {getStatusBadge(health.redis.status)}
                        </td>
                      </tr>
                    </tbody>
                  </Table>
                </>
              ) : (
                <Alert variant="warning">Unable to load health status</Alert>
              )}
            </Card.Body>
          </Card>
        </Col>

        {/* Index Stats */}
        <Col md={8}>
          <Card className="h-100">
            <Card.Header>
              <i className="bi bi-bar-chart me-2"></i>
              Search Index
            </Card.Header>
            <Card.Body>
              {statsLoading ? (
                <div className="text-center py-3">
                  <Spinner animation="border" size="sm" />
                </div>
              ) : indexStats ? (
                <Row>
                  <Col md={6} className="text-center">
                    <h3 className="text-primary">
                      {indexStats.document_count.toLocaleString()}
                    </h3>
                    <p className="text-muted mb-0">Documents</p>
                  </Col>
                  <Col md={6} className="text-center">
                    <h3 className="text-info">{indexStats.size_human}</h3>
                    <p className="text-muted mb-0">Index Size</p>
                  </Col>
                </Row>
              ) : (
                <Alert variant="warning">Unable to load index stats</Alert>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Processing Queues */}
      <Card>
        <Card.Header className="d-flex justify-content-between align-items-center">
          <span>
            <i className="bi bi-list-task me-2"></i>
            Processing Queues
          </span>
          {(dashboard?.queue_status?.total_pending ?? 0) > 0 && (
            <Badge bg="warning">
              {dashboard?.queue_status?.total_pending} pending
            </Badge>
          )}
        </Card.Header>
        <Card.Body>
          {dashboardLoading ? (
            <div className="text-center py-3">
              <Spinner animation="border" size="sm" />
            </div>
          ) : (dashboard?.queue_status?.total_pending ?? 0) === 0 &&
             (dashboard?.queue_status?.total_running ?? 0) === 0 ? (
            <p className="text-muted mb-0 text-center">
              <i className="bi bi-check-circle text-success me-2"></i>
              All queues empty - no pending jobs
            </p>
          ) : (
            <>
              <Row className="mb-3">
                <Col md={6}>
                  <h6 className="text-muted mb-2">Summarization</h6>
                  {Object.keys(dashboard?.queue_status?.summarization || {}).length > 0 ? (
                    <Table size="sm" className="mb-0">
                      <thead>
                        <tr>
                          <th>Scope</th>
                          <th>Repository</th>
                          <th>User</th>
                          <th className="text-end">Pending</th>
                          <th className="text-end">Running</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(dashboard?.queue_status?.summarization || {}).map(
                          ([queueId, info]) => {
                            const [scope, repo, user] = queueId.split('/')
                            return (
                              <tr key={queueId}>
                                <td><code>{scope}</code></td>
                                <td><code>{repo}</code></td>
                                <td><code>{user}</code></td>
                                <td className="text-end">
                                  {info.pending > 0 ? (
                                    <Badge bg="warning">{info.pending}</Badge>
                                  ) : (
                                    <span className="text-muted">0</span>
                                  )}
                                </td>
                                <td className="text-end">
                                  {info.running > 0 ? (
                                    <Badge bg="primary">{info.running}</Badge>
                                  ) : (
                                    <span className="text-muted">0</span>
                                  )}
                                </td>
                              </tr>
                            )
                          }
                        )}
                      </tbody>
                    </Table>
                  ) : (
                    <p className="text-muted small mb-0">No pending jobs</p>
                  )}
                </Col>
                <Col md={6}>
                  <h6 className="text-muted mb-2">Embedding</h6>
                  {Object.keys(dashboard?.queue_status?.embedding || {}).length > 0 ? (
                    <Table size="sm" className="mb-0">
                      <thead>
                        <tr>
                          <th>Scope</th>
                          <th>Repository</th>
                          <th>User</th>
                          <th className="text-end">Pending</th>
                          <th className="text-end">Running</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(dashboard?.queue_status?.embedding || {}).map(
                          ([queueId, info]) => {
                            const [scope, repo, user] = queueId.split('/')
                            return (
                              <tr key={queueId}>
                                <td><code>{scope}</code></td>
                                <td><code>{repo}</code></td>
                                <td><code>{user}</code></td>
                                <td className="text-end">
                                  {info.pending > 0 ? (
                                    <Badge bg="warning">{info.pending}</Badge>
                                  ) : (
                                    <span className="text-muted">0</span>
                                  )}
                                </td>
                                <td className="text-end">
                                  {info.running > 0 ? (
                                    <Badge bg="primary">{info.running}</Badge>
                                  ) : (
                                    <span className="text-muted">0</span>
                                  )}
                                </td>
                              </tr>
                            )
                          }
                        )}
                      </tbody>
                    </Table>
                  ) : (
                    <p className="text-muted small mb-0">No pending jobs</p>
                  )}
                </Col>
              </Row>
              <Row className="mb-3">
                <Col md={6}>
                  <h6 className="text-muted mb-2">Labeling</h6>
                  {Object.keys(dashboard?.queue_status?.labeling || {}).length > 0 ? (
                    <Table size="sm" className="mb-0">
                      <thead>
                        <tr>
                          <th>Scope</th>
                          <th>Repository</th>
                          <th>User</th>
                          <th className="text-end">Pending</th>
                          <th className="text-end">Running</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(dashboard?.queue_status?.labeling || {}).map(
                          ([queueId, info]) => {
                            const [scope, repo, user] = queueId.split('/')
                            return (
                              <tr key={queueId}>
                                <td><code>{scope}</code></td>
                                <td><code>{repo}</code></td>
                                <td><code>{user}</code></td>
                                <td className="text-end">
                                  {info.pending > 0 ? (
                                    <Badge bg="warning">{info.pending}</Badge>
                                  ) : (
                                    <span className="text-muted">0</span>
                                  )}
                                </td>
                                <td className="text-end">
                                  {info.running > 0 ? (
                                    <Badge bg="primary">{info.running}</Badge>
                                  ) : (
                                    <span className="text-muted">0</span>
                                  )}
                                </td>
                              </tr>
                            )
                          }
                        )}
                      </tbody>
                    </Table>
                  ) : (
                    <p className="text-muted small mb-0">No pending jobs</p>
                  )}
                </Col>
                <Col md={6}>
                  <h6 className="text-muted mb-2">Feature Processing</h6>
                  {Object.keys(dashboard?.queue_status?.feature || {}).length > 0 ? (
                    <Table size="sm" className="mb-0">
                      <thead>
                        <tr>
                          <th>Scope</th>
                          <th>Repository</th>
                          <th>User</th>
                          <th className="text-end">Pending</th>
                          <th className="text-end">Running</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(dashboard?.queue_status?.feature || {}).map(
                          ([queueId, info]) => {
                            const [scope, repo, user] = queueId.split('/')
                            return (
                              <tr key={queueId}>
                                <td><code>{scope}</code></td>
                                <td><code>{repo}</code></td>
                                <td><code>{user}</code></td>
                                <td className="text-end">
                                  {info.pending > 0 ? (
                                    <Badge bg="warning">{info.pending}</Badge>
                                  ) : (
                                    <span className="text-muted">0</span>
                                  )}
                                </td>
                                <td className="text-end">
                                  {info.running > 0 ? (
                                    <Badge bg="primary">{info.running}</Badge>
                                  ) : (
                                    <span className="text-muted">0</span>
                                  )}
                                </td>
                              </tr>
                            )
                          }
                        )}
                      </tbody>
                    </Table>
                  ) : (
                    <p className="text-muted small mb-0">No pending jobs</p>
                  )}
                </Col>
              </Row>
              <Row>
                <Col md={6}>
                  <h6 className="text-muted mb-2">Subfeature Processing</h6>
                  {Object.keys(dashboard?.queue_status?.subfeature || {}).length > 0 ? (
                    <Table size="sm" className="mb-0">
                      <thead>
                        <tr>
                          <th>Scope</th>
                          <th>Repository</th>
                          <th>User</th>
                          <th className="text-end">Pending</th>
                          <th className="text-end">Running</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(dashboard?.queue_status?.subfeature || {}).map(
                          ([queueId, info]) => {
                            const [scope, repo, user] = queueId.split('/')
                            return (
                              <tr key={queueId}>
                                <td><code>{scope}</code></td>
                                <td><code>{repo}</code></td>
                                <td><code>{user}</code></td>
                                <td className="text-end">
                                  {info.pending > 0 ? (
                                    <Badge bg="warning">{info.pending}</Badge>
                                  ) : (
                                    <span className="text-muted">0</span>
                                  )}
                                </td>
                                <td className="text-end">
                                  {info.running > 0 ? (
                                    <Badge bg="primary">{info.running}</Badge>
                                  ) : (
                                    <span className="text-muted">0</span>
                                  )}
                                </td>
                              </tr>
                            )
                          }
                        )}
                      </tbody>
                    </Table>
                  ) : (
                    <p className="text-muted small mb-0">No pending jobs</p>
                  )}
                </Col>
                <Col md={6}>
                  <h6 className="text-muted mb-2">Subfeature Labeling</h6>
                  {Object.keys(dashboard?.queue_status?.subfeature_labeling || {}).length > 0 ? (
                    <Table size="sm" className="mb-0">
                      <thead>
                        <tr>
                          <th>Scope</th>
                          <th>Repository</th>
                          <th>User</th>
                          <th className="text-end">Pending</th>
                          <th className="text-end">Running</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(dashboard?.queue_status?.subfeature_labeling || {}).map(
                          ([queueId, info]) => {
                            const [scope, repo, user] = queueId.split('/')
                            return (
                              <tr key={queueId}>
                                <td><code>{scope}</code></td>
                                <td><code>{repo}</code></td>
                                <td><code>{user}</code></td>
                                <td className="text-end">
                                  {info.pending > 0 ? (
                                    <Badge bg="warning">{info.pending}</Badge>
                                  ) : (
                                    <span className="text-muted">0</span>
                                  )}
                                </td>
                                <td className="text-end">
                                  {info.running > 0 ? (
                                    <Badge bg="primary">{info.running}</Badge>
                                  ) : (
                                    <span className="text-muted">0</span>
                                  )}
                                </td>
                              </tr>
                            )
                          }
                        )}
                      </tbody>
                    </Table>
                  ) : (
                    <p className="text-muted small mb-0">No pending jobs</p>
                  )}
                </Col>
              </Row>
            </>
          )}
        </Card.Body>
      </Card>

      {/* MCP Tool Analytics */}
      <Card className="mt-4">
        <Card.Header className="d-flex justify-content-between align-items-center">
          <span>
            <i className="bi bi-graph-up me-2"></i>
            MCP Tool Analytics
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

                {/* Transitions Table */}
                <Col md={6}>
                  <h6 className="text-muted mb-3">
                    <i className="bi bi-arrow-right-circle me-2"></i>
                    Tool Transitions
                  </h6>
                  {mcpAnalytics.top_transitions.length > 0 ? (
                    <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                      <Table size="sm" className="mb-0" hover>
                        <thead className="sticky-top bg-body">
                          <tr>
                            <th>#</th>
                            <th>From → To</th>
                            <th className="text-end">Count</th>
                          </tr>
                        </thead>
                        <tbody>
                          {mcpAnalytics.top_transitions.map((transition, idx) => (
                            <tr key={idx}>
                              <td className="text-muted">{idx + 1}</td>
                              <td>
                                <code className="small">{transition.from_tool}</code>
                                <i className="bi bi-arrow-right mx-2 text-muted"></i>
                                <code className="small">{transition.to_tool}</code>
                              </td>
                              <td className="text-end">{transition.count.toLocaleString()}</td>
                            </tr>
                          ))}
                        </tbody>
                      </Table>
                    </div>
                  ) : (
                    <p className="text-muted small">No transition data yet (need at least 2 consecutive tool calls)</p>
                  )}
                </Col>
              </Row>
            </>
          )}
        </Card.Body>
      </Card>
    </div>
  )
}

export default Admin
