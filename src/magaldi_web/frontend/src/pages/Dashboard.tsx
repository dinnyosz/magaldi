import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  Row,
  Col,
  Card,
  Form,
  Button,
  InputGroup,
  Badge,
  Spinner,
  Alert,
  Table,
} from 'react-bootstrap'
import { getDashboard } from '../api'

function Dashboard() {
  const navigate = useNavigate()
  const [searchQuery, setSearchQuery] = useState('')

  const { data: dashboard, isLoading, error } = useQuery({
    queryKey: ['dashboard'],
    queryFn: getDashboard,
    refetchInterval: 10000,
  })

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (searchQuery.trim()) {
      navigate(`/search?q=${encodeURIComponent(searchQuery)}`)
    }
  }

  const getHealthBadge = (status: string | undefined) => {
    switch (status) {
      case 'healthy':
        return <Badge bg="success">Healthy</Badge>
      case 'unhealthy':
        return <Badge bg="danger">Unhealthy</Badge>
      default:
        return <Badge bg="secondary">Unknown</Badge>
    }
  }

  if (isLoading) {
    return (
      <div className="text-center py-5">
        <Spinner animation="border" role="status">
          <span className="visually-hidden">Loading...</span>
        </Spinner>
      </div>
    )
  }

  if (error) {
    return (
      <Alert variant="danger">
        Failed to load dashboard: {(error as Error).message}
      </Alert>
    )
  }

  return (
    <div>
      <h1 className="mb-4">Dashboard</h1>

      {/* Search Bar */}
      <Card className="mb-4">
        <Card.Body>
          <Form onSubmit={handleSearch}>
            <InputGroup size="lg">
              <Form.Control
                type="search"
                placeholder="Search code semantically..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              <Button type="submit" variant="primary">
                <i className="bi bi-search me-2"></i>
                Search
              </Button>
            </InputGroup>
          </Form>
        </Card.Body>
      </Card>

      {/* Stats Cards - Row 1: Primary types */}
      <Row className="mb-3">
        <Col>
          <Card className="text-center h-100">
            <Card.Body className="py-3">
              <h3 className="mb-0 text-primary">
                {dashboard?.stats?.repository_count ?? 0}
              </h3>
              <small className="text-muted">Repos</small>
            </Card.Body>
          </Card>
        </Col>
        <Col>
          <Card
            className="text-center h-100"
            style={{ cursor: 'pointer' }}
            onClick={() => navigate('/explorer?type=file')}
          >
            <Card.Body className="py-3">
              <h3 className="mb-0 text-info">
                {dashboard?.stats?.file_count?.toLocaleString() ?? 0}
              </h3>
              <small className="text-muted">Files</small>
            </Card.Body>
          </Card>
        </Col>
        <Col>
          <Card
            className="text-center h-100"
            style={{ cursor: 'pointer' }}
            onClick={() => navigate('/explorer?type=class')}
          >
            <Card.Body className="py-3">
              <h3 className="mb-0" style={{ color: '#6f42c1' }}>
                {dashboard?.stats?.class_count?.toLocaleString() ?? 0}
              </h3>
              <small className="text-muted">Classes</small>
            </Card.Body>
          </Card>
        </Col>
        <Col>
          <Card
            className="text-center h-100"
            style={{ cursor: 'pointer' }}
            onClick={() => navigate('/explorer?type=function')}
          >
            <Card.Body className="py-3">
              <h3 className="mb-0" style={{ color: '#0d6efd' }}>
                {dashboard?.stats?.function_count?.toLocaleString() ?? 0}
              </h3>
              <small className="text-muted">Functions</small>
            </Card.Body>
          </Card>
        </Col>
        <Col>
          <Card
            className="text-center h-100"
            style={{ cursor: 'pointer' }}
            onClick={() => navigate('/explorer?type=method')}
          >
            <Card.Body className="py-3">
              <h3 className="mb-0 text-success">
                {dashboard?.stats?.method_count?.toLocaleString() ?? 0}
              </h3>
              <small className="text-muted">Methods</small>
            </Card.Body>
          </Card>
        </Col>
        <Col>
          <Card
            className="text-center h-100"
            style={{ cursor: 'pointer' }}
            onClick={() => navigate('/explorer?type=variable')}
          >
            <Card.Body className="py-3">
              <h3 className="mb-0 text-warning">
                {dashboard?.stats?.variable_count?.toLocaleString() ?? 0}
              </h3>
              <small className="text-muted">Variables</small>
            </Card.Body>
          </Card>
        </Col>
        <Col>
          <Card
            className="text-center h-100"
            style={{ cursor: 'pointer' }}
            onClick={() => navigate('/explorer?type=constant')}
          >
            <Card.Body className="py-3">
              <h3 className="mb-0" style={{ color: '#fd7e14' }}>
                {dashboard?.stats?.constant_count?.toLocaleString() ?? 0}
              </h3>
              <small className="text-muted">Constants</small>
            </Card.Body>
          </Card>
        </Col>
        <Col>
          <Card className="text-center h-100">
            <Card.Body className="py-3">
              <h3 className="mb-0" style={{ color: '#20c997' }}>
                {dashboard?.stats?.feature_count?.toLocaleString() ?? 0}
              </h3>
              <small className="text-muted">Features</small>
            </Card.Body>
          </Card>
        </Col>
        <Col>
          <Card className="text-center h-100">
            <Card.Body className="py-3">
              <h3 className="mb-0" style={{ color: '#17a2b8' }}>
                {dashboard?.stats?.subfeature_count?.toLocaleString() ?? 0}
              </h3>
              <small className="text-muted">Subfeatures</small>
            </Card.Body>
          </Card>
        </Col>
        <Col>
          <Card
            className="text-center h-100 bg-light"
            style={{ cursor: 'pointer' }}
            onClick={() => navigate('/explorer')}
          >
            <Card.Body className="py-3">
              <h3 className="mb-0 text-secondary">
                {dashboard?.stats?.element_count?.toLocaleString() ?? 0}
              </h3>
              <small className="text-muted">Total</small>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row className="mb-4">
        {/* Service Health */}
        <Col md={3}>
          <Card className="h-100">
            <Card.Header>
              <i className="bi bi-heart-pulse me-2"></i>
              Service Health
            </Card.Header>
            <Card.Body>
              <Table borderless size="sm" className="mb-0">
                <tbody>
                  <tr>
                    <td>Elasticsearch</td>
                    <td className="text-end">
                      {getHealthBadge(dashboard?.health?.elasticsearch?.status)}
                    </td>
                  </tr>
                  <tr>
                    <td>LLM</td>
                    <td className="text-end">
                      {getHealthBadge(dashboard?.health?.llm?.status)}
                    </td>
                  </tr>
                  <tr>
                    <td>Redis</td>
                    <td className="text-end">
                      {getHealthBadge(dashboard?.health?.redis?.status)}
                    </td>
                  </tr>
                </tbody>
              </Table>
            </Card.Body>
          </Card>
        </Col>

        {/* Queue Status */}
        <Col md={9}>
          <Card className="h-100">
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
              {(dashboard?.queue_status?.total_pending ?? 0) === 0 &&
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
                  <Row>
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
                  <Row className="mb-3">
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
                  </Row>
                </>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Repositories */}
      <Card className="mb-4">
        <Card.Header>
          <i className="bi bi-folder2-open me-2"></i>
          Indexed Repositories
        </Card.Header>
        <Card.Body>
          {dashboard?.recent_repos?.length ? (
            <Table hover responsive size="sm">
              <thead>
                <tr>
                  <th>Repository</th>
                  <th className="text-end">Files</th>
                  <th className="text-end">Classes</th>
                  <th className="text-end">Functions</th>
                  <th className="text-end">Methods</th>
                  <th className="text-end">Variables</th>
                  <th className="text-end">Constants</th>
                  <th className="text-end">Features</th>
                  <th className="text-end">Total</th>
                  <th>Languages</th>
                </tr>
              </thead>
              <tbody>
                {dashboard.recent_repos.map((repo) => (
                  <tr
                    key={`${repo.scope}/${repo.name}`}
                    style={{ cursor: 'pointer' }}
                    onClick={() => navigate(`/repos/${repo.scope}/${repo.name}`)}
                  >
                    <td>
                      <code>{repo.scope}/{repo.name}</code>
                    </td>
                    <td className="text-end">{repo.file_count.toLocaleString()}</td>
                    <td className="text-end">{repo.class_count.toLocaleString()}</td>
                    <td className="text-end">{repo.function_count.toLocaleString()}</td>
                    <td className="text-end">{repo.method_count.toLocaleString()}</td>
                    <td className="text-end">{repo.variable_count.toLocaleString()}</td>
                    <td className="text-end">{repo.constant_count.toLocaleString()}</td>
                    <td className="text-end">
                      {repo.feature_count > 0 ? (
                        <Badge bg="success">{repo.feature_count}</Badge>
                      ) : (
                        <span className="text-muted">0</span>
                      )}
                    </td>
                    <td className="text-end fw-bold">{repo.element_count.toLocaleString()}</td>
                    <td>
                      {repo.languages?.map((lang) => (
                        <Badge key={lang} bg="secondary" className="me-1">
                          {lang}
                        </Badge>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </Table>
          ) : (
            <p className="text-muted mb-0">
              No repositories indexed yet. Run <code>magaldi parse</code> to index a repository.
            </p>
          )}
        </Card.Body>
      </Card>
    </div>
  )
}

export default Dashboard
