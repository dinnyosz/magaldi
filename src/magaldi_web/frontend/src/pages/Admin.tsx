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
    queryFn: () => getJobs(undefined, 20),
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
      case 'pending':
        return <Badge bg="secondary">Pending</Badge>
      case 'running':
      case 'in_progress':
        return <Badge bg="primary">Running</Badge>
      case 'completed':
      case 'success':
        return <Badge bg="success">Completed</Badge>
      case 'failed':
      case 'error':
        return <Badge bg="danger">Failed</Badge>
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

  const formatDate = (dateStr: string | null): string => {
    if (!dateStr) return '-'
    return new Date(dateStr).toLocaleString()
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
                    {getStatusBadge(health.status)}
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
                          {getStatusBadge(health.services.elasticsearch.status)}
                        </td>
                      </tr>
                      <tr>
                        <td>
                          <i className="bi bi-robot me-2"></i>
                          LLM
                        </td>
                        <td className="text-end">
                          {getStatusBadge(health.services.llm.status)}
                        </td>
                      </tr>
                      <tr>
                        <td>
                          <i className="bi bi-hdd-stack me-2"></i>
                          Redis
                        </td>
                        <td className="text-end">
                          {getStatusBadge(health.services.redis.status)}
                        </td>
                      </tr>
                    </tbody>
                  </Table>
                  <hr />
                  <small className="text-muted">
                    Last check: {formatDate(health.timestamp)}
                  </small>
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
                  <Col md={4} className="text-center">
                    <h2 className="text-primary">
                      {indexStats.total_documents.toLocaleString()}
                    </h2>
                    <p className="text-muted mb-0">Total Documents</p>
                  </Col>
                  <Col md={4} className="text-center">
                    <h2 className="text-info">
                      {formatBytes(indexStats.index_size_bytes)}
                    </h2>
                    <p className="text-muted mb-0">Index Size</p>
                  </Col>
                  <Col md={4} className="text-center">
                    <h2
                      className={
                        indexStats.shards.failed > 0
                          ? 'text-danger'
                          : 'text-success'
                      }
                    >
                      {indexStats.shards.successful}/{indexStats.shards.total}
                    </h2>
                    <p className="text-muted mb-0">Healthy Shards</p>
                  </Col>
                </Row>
              ) : (
                <Alert variant="warning">Unable to load index stats</Alert>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Jobs */}
      <Card>
        <Card.Header className="d-flex justify-content-between align-items-center">
          <span>
            <i className="bi bi-list-task me-2"></i>
            Recent Jobs
          </span>
          {jobs && (
            <small className="text-muted">{jobs.total} total jobs</small>
          )}
        </Card.Header>
        <Card.Body>
          {jobsLoading ? (
            <div className="text-center py-3">
              <Spinner animation="border" size="sm" />
            </div>
          ) : jobs && jobs.jobs.length > 0 ? (
            <Table hover responsive>
              <thead>
                <tr>
                  <th>Job ID</th>
                  <th>Type</th>
                  <th>Status</th>
                  <th>Progress</th>
                  <th>Created</th>
                  <th>Completed</th>
                </tr>
              </thead>
              <tbody>
                {jobs.jobs.map((job) => (
                  <tr key={job.job_id}>
                    <td>
                      <code className="small">{job.job_id.slice(0, 8)}...</code>
                    </td>
                    <td>
                      <Badge bg="secondary">{job.job_type}</Badge>
                    </td>
                    <td>{getStatusBadge(job.status)}</td>
                    <td style={{ minWidth: 100 }}>
                      <ProgressBar
                        now={job.progress * 100}
                        label={`${Math.round(job.progress * 100)}%`}
                        variant={
                          job.status === 'failed'
                            ? 'danger'
                            : job.progress === 1
                            ? 'success'
                            : 'primary'
                        }
                        style={{ height: 20 }}
                      />
                    </td>
                    <td>
                      <small>{formatDate(job.created_at)}</small>
                    </td>
                    <td>
                      <small>{formatDate(job.completed_at)}</small>
                    </td>
                  </tr>
                ))}
              </tbody>
            </Table>
          ) : (
            <p className="text-muted text-center mb-0">No jobs found</p>
          )}
        </Card.Body>
      </Card>
    </div>
  )
}

export default Admin
