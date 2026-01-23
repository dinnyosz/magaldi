import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  Card,
  Badge,
  Spinner,
  Alert,
  Form,
  Row,
  Col,
  Breadcrumb,
  ListGroup,
  Table,
} from 'react-bootstrap'
import { useState } from 'react'
import { getDependencyGraph } from '../api'

function DependencyGraph() {
  const { scope, repo } = useParams<{ scope: string; repo: string }>()
  const [maxDepth, setMaxDepth] = useState(3)
  const [includeExternal, setIncludeExternal] = useState(false)

  const { data, isLoading, error } = useQuery({
    queryKey: ['dependency-graph', scope, repo, maxDepth, includeExternal],
    queryFn: () =>
      getDependencyGraph(scope!, repo!, {
        max_depth: maxDepth,
        include_external: includeExternal,
      }),
    enabled: !!scope && !!repo,
  })

  if (!scope || !repo) {
    return <Alert variant="warning">Repository not specified</Alert>
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
        Failed to load dependency graph: {(error as Error).message}
      </Alert>
    )
  }

  // Build adjacency list for display
  const adjacency: Record<string, string[]> = {}
  data?.edges.forEach((edge) => {
    const from = edge.from
    if (!adjacency[from]) adjacency[from] = []
    adjacency[from].push(edge.to)
  })

  // Sort modules by number of dependencies
  const sortedModules = [...(data?.nodes || [])].sort((a, b) => {
    return (adjacency[b]?.length || 0) - (adjacency[a]?.length || 0)
  })

  return (
    <div>
      {/* Breadcrumb */}
      <Breadcrumb className="mb-4">
        <Breadcrumb.Item linkAs={Link} linkProps={{ to: '/repos' }}>
          Repositories
        </Breadcrumb.Item>
        <Breadcrumb.Item linkAs={Link} linkProps={{ to: `/repos/${scope}/${repo}` }}>
          {scope}/{repo}
        </Breadcrumb.Item>
        <Breadcrumb.Item active>Dependency Graph</Breadcrumb.Item>
      </Breadcrumb>

      <h2 className="mb-4">
        <i className="bi bi-share me-2"></i>
        Dependency Graph
      </h2>

      {/* Stats Card */}
      <Card className="mb-4">
        <Card.Body>
          <Row>
            <Col md={3} className="text-center border-end">
              <h3 className="text-primary mb-0">{data?.stats.total_modules ?? 0}</h3>
              <small className="text-muted">Modules</small>
            </Col>
            <Col md={3} className="text-center border-end">
              <h3 className="text-success mb-0">{data?.stats.total_edges ?? 0}</h3>
              <small className="text-muted">Dependencies</small>
            </Col>
            <Col md={3} className="text-center border-end">
              <h3 className="text-info mb-0">{data?.stats.internal_edges ?? 0}</h3>
              <small className="text-muted">Internal</small>
            </Col>
            <Col md={3} className="text-center">
              <h3 className={`mb-0 ${(data?.cycles.length || 0) > 0 ? 'text-danger' : 'text-secondary'}`}>
                {data?.cycles.length ?? 0}
              </h3>
              <small className="text-muted">Cycles</small>
            </Col>
          </Row>
        </Card.Body>
      </Card>

      {/* Controls */}
      <Card className="mb-4">
        <Card.Body>
          <Row>
            <Col md={6}>
              <Form.Group>
                <Form.Label>Max Depth: {maxDepth}</Form.Label>
                <Form.Range
                  min={1}
                  max={10}
                  value={maxDepth}
                  onChange={(e) => setMaxDepth(parseInt(e.target.value))}
                />
              </Form.Group>
            </Col>
            <Col md={6} className="d-flex align-items-center">
              <Form.Check
                type="switch"
                id="include-external"
                label="Include external dependencies"
                checked={includeExternal}
                onChange={(e) => setIncludeExternal(e.target.checked)}
              />
            </Col>
          </Row>
        </Card.Body>
      </Card>

      {/* Cycles Warning */}
      {data?.cycles && data.cycles.length > 0 && (
        <Card className="mb-4 border-danger">
          <Card.Header className="bg-danger text-white">
            <i className="bi bi-exclamation-triangle me-2"></i>
            Circular Dependencies Detected
          </Card.Header>
          <Card.Body>
            <ListGroup variant="flush">
              {data.cycles.map((cycle, index) => (
                <ListGroup.Item key={index}>
                  <code>{cycle.join(' \u2192 ')}</code>
                </ListGroup.Item>
              ))}
            </ListGroup>
          </Card.Body>
        </Card>
      )}

      {/* Module List */}
      <Card>
        <Card.Header>
          <i className="bi bi-list-nested me-2"></i>
          Module Dependencies
        </Card.Header>
        <Table striped hover className="mb-0">
          <thead>
            <tr>
              <th>Module</th>
              <th style={{ width: '100px' }}>Dependencies</th>
              <th>Imports</th>
            </tr>
          </thead>
          <tbody>
            {sortedModules.map((module) => (
              <tr key={module}>
                <td>
                  <code>{module}</code>
                </td>
                <td>
                  <Badge bg="primary">{adjacency[module]?.length || 0}</Badge>
                </td>
                <td>
                  {adjacency[module]?.slice(0, 5).map((dep, i) => (
                    <Badge key={i} bg="secondary" className="me-1">
                      {dep.split('/').pop()}
                    </Badge>
                  ))}
                  {adjacency[module]?.length > 5 && (
                    <span className="text-muted small">
                      +{adjacency[module].length - 5} more
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Card>
    </div>
  )
}

export default DependencyGraph
