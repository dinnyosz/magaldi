import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  Row,
  Col,
  Card,
  Badge,
  Spinner,
  Alert,
  ListGroup,
  Breadcrumb,
} from 'react-bootstrap'
import { getElement, getSimilarElements } from '../api'

function Element() {
  const { elementId } = useParams<{ elementId: string }>()
  const decodedId = elementId ? decodeURIComponent(elementId) : ''

  const { data: element, isLoading, error } = useQuery({
    queryKey: ['element', decodedId],
    queryFn: () => getElement(decodedId),
    enabled: !!decodedId,
  })

  const { data: similar } = useQuery({
    queryKey: ['similar', decodedId],
    queryFn: () => getSimilarElements(decodedId, 10),
    enabled: !!decodedId,
  })

  const getTypeBadgeVariant = (type: string): string => {
    switch (type) {
      case 'file':
        return 'info'
      case 'class':
        return 'purple'
      case 'function':
        return 'primary'
      case 'method':
        return 'success'
      case 'variable':
        return 'danger'
      case 'constant':
        return 'warning'
      default:
        return 'secondary'
    }
  }

  const getTypeBadgeStyle = (type: string): React.CSSProperties => {
    if (type === 'class') {
      return { backgroundColor: '#6f42c1', color: 'white' }
    }
    return {}
  }

  // Parse element ID to get repo info
  const parseElementId = (id: string) => {
    const parts = id.split(':')
    if (parts.length >= 3) {
      return { scope: parts[0], repository: parts[1] }
    }
    return null
  }

  const repoInfo = element ? parseElementId(element.element_id) : null

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
        Failed to load element: {(error as Error).message}
      </Alert>
    )
  }

  if (!element) {
    return (
      <Alert variant="warning">Element not found</Alert>
    )
  }

  return (
    <div>
      <Breadcrumb className="mb-4">
        <Breadcrumb.Item linkAs={Link} linkProps={{ to: '/repos' }}>
          Repositories
        </Breadcrumb.Item>
        {repoInfo && (
          <Breadcrumb.Item
            linkAs={Link}
            linkProps={{ to: `/repos/${repoInfo.scope}/${repoInfo.repository}` }}
          >
            {repoInfo.scope}/{repoInfo.repository}
          </Breadcrumb.Item>
        )}
        <Breadcrumb.Item active>{element.name}</Breadcrumb.Item>
      </Breadcrumb>

      <Row>
        <Col lg={8}>
          {/* Element Info */}
          <Card className="mb-4">
            <Card.Header className="d-flex justify-content-between align-items-center">
              <div>
                <Badge
                  bg={getTypeBadgeVariant(element.element_type)}
                  style={getTypeBadgeStyle(element.element_type)}
                  className="me-2"
                >
                  {element.element_type}
                </Badge>
                <span className="fs-5">{element.name}</span>
              </div>
              <small className="text-muted">
                Lines {element.line_start}-{element.line_end}
              </small>
            </Card.Header>
            <Card.Body>
              <div className="mb-3">
                <small className="text-muted">
                  <i className="bi bi-file-earmark me-1"></i>
                  {element.file_path}
                </small>
              </div>

              {element.summary && (
                <Alert variant="info" className="mb-3">
                  <i className="bi bi-info-circle me-2"></i>
                  {element.summary}
                </Alert>
              )}

              <pre className="bg-light p-3 rounded" style={{ maxHeight: '50vh', overflowY: 'auto' }}>
                <code>{element.code}</code>
              </pre>
            </Card.Body>
          </Card>

          {/* Children */}
          {element.children && element.children.length > 0 && (
            <Card className="mb-4">
              <Card.Header>
                <i className="bi bi-diagram-3 me-2"></i>
                Children ({element.children.length})
              </Card.Header>
              <ListGroup variant="flush">
                {element.children.map((child) => (
                  <ListGroup.Item
                    key={child.element_id}
                    action
                    as={Link}
                    to={`/element/${encodeURIComponent(child.element_id)}`}
                  >
                    <Badge
                      bg={getTypeBadgeVariant(child.element_type)}
                      style={getTypeBadgeStyle(child.element_type)}
                      className="me-2"
                    >
                      {child.element_type}
                    </Badge>
                    {child.name}
                  </ListGroup.Item>
                ))}
              </ListGroup>
            </Card>
          )}
        </Col>

        <Col lg={4}>
          {/* Parent */}
          {element.parent_id && (
            <Card className="mb-4">
              <Card.Header>
                <i className="bi bi-arrow-up-circle me-2"></i>
                Parent
              </Card.Header>
              <Card.Body>
                <Link to={`/element/${encodeURIComponent(element.parent_id)}`}>
                  <code>{element.parent_id.split(':').slice(-2).join(':')}</code>
                </Link>
              </Card.Body>
            </Card>
          )}

          {/* Similar Elements */}
          {similar && similar.length > 0 && (
            <Card className="mb-4">
              <Card.Header>
                <i className="bi bi-collection me-2"></i>
                Similar Elements
              </Card.Header>
              <ListGroup variant="flush">
                {similar.map((sim) => (
                  <ListGroup.Item
                    key={sim.element_id}
                    action
                    as={Link}
                    to={`/element/${encodeURIComponent(sim.element_id)}`}
                    className="d-flex justify-content-between align-items-start"
                  >
                    <div className="text-truncate me-2">
                      <Badge
                        bg={getTypeBadgeVariant(sim.element_type)}
                        style={getTypeBadgeStyle(sim.element_type)}
                        className="me-1"
                        pill
                      >
                        {sim.element_type.charAt(0)}
                      </Badge>
                      <small>{sim.name}</small>
                    </div>
                    <Badge bg="secondary" pill>
                      {(sim.similarity * 100).toFixed(0)}%
                    </Badge>
                  </ListGroup.Item>
                ))}
              </ListGroup>
            </Card>
          )}

          {/* Quick Info */}
          <Card>
            <Card.Header>
              <i className="bi bi-info-circle me-2"></i>
              Info
            </Card.Header>
            <ListGroup variant="flush">
              <ListGroup.Item className="d-flex justify-content-between">
                <span className="text-muted">Type</span>
                <Badge
                  bg={getTypeBadgeVariant(element.element_type)}
                  style={getTypeBadgeStyle(element.element_type)}
                >
                  {element.element_type}
                </Badge>
              </ListGroup.Item>
              <ListGroup.Item className="d-flex justify-content-between">
                <span className="text-muted">Lines</span>
                <span>{element.line_end - element.line_start + 1}</span>
              </ListGroup.Item>
              <ListGroup.Item className="d-flex justify-content-between">
                <span className="text-muted">Children</span>
                <span>{element.children?.length ?? 0}</span>
              </ListGroup.Item>
            </ListGroup>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default Element
