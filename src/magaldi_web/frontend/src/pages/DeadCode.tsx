import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  Card,
  Badge,
  Spinner,
  Alert,
  Table,
  Form,
  Row,
  Col,
  Breadcrumb,
} from 'react-bootstrap'
import { useState } from 'react'
import { getDeadCode } from '../api'

// Type configuration
const typeConfig: Record<string, { icon: string; color: string; label: string }> = {
  function: { icon: 'bi-braces', color: 'primary', label: 'Function' },
  method: { icon: 'bi-gear', color: 'success', label: 'Method' },
}

function getTypeConfig(type: string) {
  return typeConfig[type] || { icon: 'bi-dot', color: 'secondary', label: type }
}

function DeadCode() {
  const { scope, repo } = useParams<{ scope: string; repo: string }>()
  const [includeTests, setIncludeTests] = useState(false)

  const { data, isLoading, error } = useQuery({
    queryKey: ['dead-code', scope, repo, includeTests],
    queryFn: () => getDeadCode(scope!, repo!, { include_tests: includeTests }),
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
        Failed to analyze dead code: {(error as Error).message}
      </Alert>
    )
  }

  // Group by file
  const byFile: Record<string, typeof data.potentially_dead> = {}
  data?.potentially_dead.forEach((item) => {
    if (!byFile[item.file_path]) {
      byFile[item.file_path] = []
    }
    byFile[item.file_path].push(item)
  })

  const files = Object.keys(byFile).sort()

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
        <Breadcrumb.Item active>Dead Code Analysis</Breadcrumb.Item>
      </Breadcrumb>

      <h2 className="mb-4">
        <i className="bi bi-trash me-2"></i>
        Dead Code Analysis
      </h2>

      {/* Stats Card */}
      <Card className="mb-4">
        <Card.Body>
          <Row>
            <Col md={3} className="text-center border-end">
              <h3 className="text-primary mb-0">{data?.stats.total_functions ?? 0}</h3>
              <small className="text-muted">Total Functions</small>
            </Col>
            <Col md={3} className="text-center border-end">
              <h3 className="text-success mb-0">{data?.stats.called ?? 0}</h3>
              <small className="text-muted">Called</small>
            </Col>
            <Col md={3} className="text-center border-end">
              <h3 className="text-info mb-0">{data?.stats.excluded_entry_points ?? 0}</h3>
              <small className="text-muted">Entry Points</small>
            </Col>
            <Col md={3} className="text-center">
              <h3 className="text-warning mb-0">{data?.stats.potentially_dead ?? 0}</h3>
              <small className="text-muted">Potentially Dead</small>
            </Col>
          </Row>
        </Card.Body>
      </Card>

      {/* Filter */}
      <Card className="mb-4">
        <Card.Body>
          <Form.Check
            type="switch"
            id="include-tests"
            label="Include test functions"
            checked={includeTests}
            onChange={(e) => setIncludeTests(e.target.checked)}
          />
        </Card.Body>
      </Card>

      {/* Results */}
      {data?.potentially_dead.length === 0 ? (
        <Alert variant="success">
          <i className="bi bi-check-circle me-2"></i>
          No potentially dead code found!
        </Alert>
      ) : (
        files.map((filePath) => (
          <Card key={filePath} className="mb-3">
            <Card.Header>
              <i className="bi bi-file-code me-2"></i>
              {filePath}
              <Badge bg="warning" className="ms-2">{byFile[filePath].length}</Badge>
            </Card.Header>
            <Table striped hover className="mb-0">
              <tbody>
                {byFile[filePath].map((item) => {
                  const config = getTypeConfig(item.element_type)
                  return (
                    <tr key={item.element_id}>
                      <td style={{ width: '40px' }}>
                        <Badge bg={config.color} pill>
                          <i className={`bi ${config.icon}`}></i>
                        </Badge>
                      </td>
                      <td>
                        <Link to={`/element/${item.hash_id || item.element_id}`}>
                          {item.name}
                        </Link>
                        {item.is_test && (
                          <Badge bg="info" className="ms-2">test</Badge>
                        )}
                      </td>
                      <td className="text-muted" style={{ width: '80px' }}>
                        line {item.line}
                      </td>
                      <td className="text-muted text-truncate" style={{ maxWidth: '300px' }}>
                        {item.summary}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </Table>
          </Card>
        ))
      )}
    </div>
  )
}

export default DeadCode
