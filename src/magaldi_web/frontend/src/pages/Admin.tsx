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
  ProgressBar,
  Form,
  InputGroup,
} from 'react-bootstrap'
import { getHealth, getIndexStats, getDashboard, getMCPAnalytics, getMCPActivityHistory, clearMCPAnalytics } from '../api'

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
      health.elasticsearch.status,
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
                          Elasticsearch
                        </td>
                        <td className="text-end">
                          {getStatusBadge(health.elasticsearch.status)}
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
              Elasticsearch Index
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
          <div>
            <Button
              variant="outline-secondary"
              size="sm"
              className="me-2"
              onClick={() => refetchAnalytics()}
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
          {analyticsLoading ? (
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
              {/* Summary Stats */}
              <Row className="mb-4">
                <Col md={4} className="text-center">
                  <h3 className="text-primary">{mcpAnalytics.total_calls.toLocaleString()}</h3>
                  <p className="text-muted mb-0">Total Calls</p>
                </Col>
                <Col md={4} className="text-center">
                  <h3 className="text-info">{mcpAnalytics.unique_tools}</h3>
                  <p className="text-muted mb-0">Unique Tools</p>
                </Col>
                <Col md={4} className="text-center">
                  <h3 className="text-success">{mcpAnalytics.today_calls.toLocaleString()}</h3>
                  <p className="text-muted mb-0">Today</p>
                </Col>
              </Row>

              <Row>
                {/* Tool Usage */}
                <Col md={6}>
                  <h6 className="text-muted mb-3">
                    <i className="bi bi-bar-chart me-2"></i>
                    Tool Usage
                  </h6>
                  {mcpAnalytics.tool_usage.length > 0 ? (
                    <Table size="sm" className="mb-0">
                      <thead>
                        <tr>
                          <th>Tool</th>
                          <th className="text-end" style={{ width: '80px' }}>Calls</th>
                          <th style={{ width: '150px' }}>Usage</th>
                        </tr>
                      </thead>
                      <tbody>
                        {mcpAnalytics.tool_usage.slice(0, 15).map((tool) => (
                          <tr key={tool.tool_name}>
                            <td>
                              <code className="small">{tool.tool_name}</code>
                            </td>
                            <td className="text-end">{tool.call_count.toLocaleString()}</td>
                            <td>
                              <div className="d-flex align-items-center">
                                <ProgressBar
                                  now={tool.percentage}
                                  style={{ height: '8px', flex: 1 }}
                                  variant="primary"
                                />
                                <span className="ms-2 small text-muted" style={{ width: '40px' }}>
                                  {tool.percentage}%
                                </span>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  ) : (
                    <p className="text-muted small">No tool usage data</p>
                  )}
                </Col>

                {/* Tool Transitions */}
                <Col md={6}>
                  <h6 className="text-muted mb-3">
                    <i className="bi bi-arrow-right-circle me-2"></i>
                    Top Tool Transitions
                  </h6>
                  {mcpAnalytics.top_transitions.length > 0 ? (
                    <Table size="sm" className="mb-0">
                      <thead>
                        <tr>
                          <th>From</th>
                          <th></th>
                          <th>To</th>
                          <th className="text-end">Count</th>
                        </tr>
                      </thead>
                      <tbody>
                        {mcpAnalytics.top_transitions.slice(0, 15).map((transition, idx) => (
                          <tr key={idx}>
                            <td>
                              <code className="small">{transition.from_tool}</code>
                            </td>
                            <td className="text-center text-muted">
                              <i className="bi bi-arrow-right"></i>
                            </td>
                            <td>
                              <code className="small">{transition.to_tool}</code>
                            </td>
                            <td className="text-end">
                              {transition.count.toLocaleString()}
                              <span className="text-muted small ms-1">
                                ({transition.percentage}%)
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  ) : (
                    <p className="text-muted small">No transition data yet (need at least 2 consecutive tool calls)</p>
                  )}
                </Col>
              </Row>
            </>
          )}
        </Card.Body>
      </Card>

      {/* Activity History */}
      <Card className="mt-4">
        <Card.Header className="d-flex justify-content-between align-items-center">
          <span>
            <i className="bi bi-calendar3 me-2"></i>
            Activity History
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
              onClick={() => refetchHistory()}
            >
              <i className="bi bi-arrow-clockwise"></i>
            </Button>
          </div>
        </Card.Header>
        <Card.Body>
          {historyLoading ? (
            <div className="text-center py-3">
              <Spinner animation="border" size="sm" />
            </div>
          ) : !activityHistory || activityHistory.daily_activity.length === 0 ? (
            <p className="text-muted mb-0 text-center">
              <i className="bi bi-info-circle me-2"></i>
              No activity data for the selected period.
            </p>
          ) : (
            <>
              {/* Summary */}
              <Row className="mb-4">
                <Col md={4} className="text-center">
                  <h3 className="text-primary">{activityHistory.total_calls.toLocaleString()}</h3>
                  <p className="text-muted mb-0">Total Calls ({historyDays} days)</p>
                </Col>
                <Col md={4} className="text-center">
                  <h3 className="text-info">
                    {activityHistory.daily_activity.length > 0
                      ? Math.round(activityHistory.total_calls / activityHistory.daily_activity.length)
                      : 0}
                  </h3>
                  <p className="text-muted mb-0">Avg. per Day</p>
                </Col>
                <Col md={4} className="text-center">
                  <h3 className="text-success">
                    {Math.max(...activityHistory.daily_activity.map(d => d.total_calls))}
                  </h3>
                  <p className="text-muted mb-0">Peak Day</p>
                </Col>
              </Row>

              {/* Daily Activity Chart (Simple Bar) */}
              <h6 className="text-muted mb-3">
                <i className="bi bi-bar-chart-fill me-2"></i>
                Daily Activity
              </h6>
              <div className="mb-4">
                {activityHistory.daily_activity.map((day) => {
                  const maxCalls = Math.max(...activityHistory.daily_activity.map(d => d.total_calls))
                  const percentage = maxCalls > 0 ? (day.total_calls / maxCalls) * 100 : 0
                  const date = new Date(day.date)
                  const dayLabel = date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })

                  return (
                    <div key={day.date} className="d-flex align-items-center mb-2">
                      <span className="text-muted small" style={{ width: '100px' }}>
                        {dayLabel}
                      </span>
                      <div className="flex-grow-1 mx-2">
                        <ProgressBar
                          now={percentage}
                          style={{ height: '20px' }}
                          variant={day.total_calls > 0 ? 'primary' : 'secondary'}
                        />
                      </div>
                      <span className="text-end" style={{ width: '60px' }}>
                        {day.total_calls.toLocaleString()}
                      </span>
                    </div>
                  )
                })}
              </div>

              {/* Top Tools per Day */}
              <h6 className="text-muted mb-3">
                <i className="bi bi-trophy me-2"></i>
                Top Tools by Day
              </h6>
              <Table size="sm" className="mb-0">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Total</th>
                    <th>Top Tool</th>
                    <th>Top 3 Tools</th>
                  </tr>
                </thead>
                <tbody>
                  {activityHistory.daily_activity
                    .slice()
                    .reverse()
                    .filter(day => day.total_calls > 0)
                    .slice(0, 10)
                    .map((day) => {
                      const sortedTools = Object.entries(day.tool_counts)
                        .sort((a, b) => b[1] - a[1])
                      const topTool = sortedTools[0]
                      const top3 = sortedTools.slice(0, 3)

                      return (
                        <tr key={day.date}>
                          <td>
                            <code className="small">{day.date}</code>
                          </td>
                          <td>{day.total_calls}</td>
                          <td>
                            {topTool ? (
                              <>
                                <code className="small">{topTool[0]}</code>
                                <Badge bg="secondary" className="ms-1">{topTool[1]}</Badge>
                              </>
                            ) : (
                              <span className="text-muted">-</span>
                            )}
                          </td>
                          <td>
                            {top3.length > 0 ? (
                              top3.map(([tool, count], idx) => (
                                <span key={tool}>
                                  {idx > 0 && ', '}
                                  <code className="small">{tool}</code>
                                  <span className="text-muted small ms-1">({count})</span>
                                </span>
                              ))
                            ) : (
                              <span className="text-muted">-</span>
                            )}
                          </td>
                        </tr>
                      )
                    })}
                </tbody>
              </Table>
              {activityHistory.daily_activity.filter(d => d.total_calls > 0).length === 0 && (
                <p className="text-muted small text-center mt-2">
                  No tool usage recorded in this period.
                </p>
              )}
            </>
          )}
        </Card.Body>
      </Card>
    </div>
  )
}

export default Admin
