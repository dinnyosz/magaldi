import { useQuery } from '@tanstack/react-query'
import {
  Row,
  Col,
  Card,
  Badge,
  Spinner,
  Alert,
  Table,
  ProgressBar,
} from 'react-bootstrap'
import { getHealth, getJobs, getIndexStats } from '../api'

function Admin() {
  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: 10000,
  })

  const { data: jobs, isLoading: jobsLoading } = useQuery({
    queryKey: ['jobs'],
    queryFn: getJobs,
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

      {/* Job Queue Stats */}
      <Card>
        <Card.Header>
          <i className="bi bi-list-task me-2"></i>
          Job Queues
        </Card.Header>
        <Card.Body>
          {jobsLoading ? (
            <div className="text-center py-3">
              <Spinner animation="border" size="sm" />
            </div>
          ) : jobs ? (
            <Row>
              <Col md={6}>
                <h6>Summarization</h6>
                <Table size="sm" bordered>
                  <tbody>
                    <tr>
                      <td>Pending</td>
                      <td className="text-end">
                        <Badge bg="secondary">{jobs.summarization.pending}</Badge>
                      </td>
                    </tr>
                    <tr>
                      <td>Running</td>
                      <td className="text-end">
                        <Badge bg="primary">{jobs.summarization.running}</Badge>
                      </td>
                    </tr>
                    <tr>
                      <td>Completed</td>
                      <td className="text-end">
                        <Badge bg="success">{jobs.summarization.completed}</Badge>
                      </td>
                    </tr>
                    <tr>
                      <td>Failed</td>
                      <td className="text-end">
                        <Badge bg="danger">{jobs.summarization.failed}</Badge>
                      </td>
                    </tr>
                  </tbody>
                </Table>
              </Col>
              <Col md={6}>
                <h6>Embedding</h6>
                <Table size="sm" bordered>
                  <tbody>
                    <tr>
                      <td>Pending</td>
                      <td className="text-end">
                        <Badge bg="secondary">{jobs.embedding.pending}</Badge>
                      </td>
                    </tr>
                    <tr>
                      <td>Running</td>
                      <td className="text-end">
                        <Badge bg="primary">{jobs.embedding.running}</Badge>
                      </td>
                    </tr>
                    <tr>
                      <td>Completed</td>
                      <td className="text-end">
                        <Badge bg="success">{jobs.embedding.completed}</Badge>
                      </td>
                    </tr>
                    <tr>
                      <td>Failed</td>
                      <td className="text-end">
                        <Badge bg="danger">{jobs.embedding.failed}</Badge>
                      </td>
                    </tr>
                  </tbody>
                </Table>
              </Col>
            </Row>
          ) : (
            <p className="text-muted text-center mb-0">Unable to load job stats</p>
          )}
        </Card.Body>
      </Card>
    </div>
  )
}

export default Admin
