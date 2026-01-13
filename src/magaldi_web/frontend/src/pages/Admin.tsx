import { useQuery } from '@tanstack/react-query'
import {
  Row,
  Col,
  Card,
  Badge,
  Spinner,
  Alert,
  Table,
} from 'react-bootstrap'
import { getHealth, getIndexStats, getDashboard } from '../api'

function Admin() {
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

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
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
                  <Col md={3} className="text-center">
                    <h3 className="text-primary">
                      {indexStats.document_count.toLocaleString()}
                    </h3>
                    <p className="text-muted mb-0">Documents</p>
                  </Col>
                  <Col md={3} className="text-center">
                    <h3 className="text-info">{indexStats.size_human}</h3>
                    <p className="text-muted mb-0">Index Size</p>
                  </Col>
                  <Col md={3} className="text-center">
                    <h3 className="text-success">
                      {indexStats.with_vectors.toLocaleString()}
                    </h3>
                    <p className="text-muted mb-0">With Vectors</p>
                  </Col>
                  <Col md={3} className="text-center">
                    <h3 className="text-warning">
                      {indexStats.vector_coverage_pct}%
                    </h3>
                    <p className="text-muted mb-0">Coverage</p>
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
    </div>
  )
}

export default Admin
