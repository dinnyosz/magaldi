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
  Tab,
  Tabs,
} from 'react-bootstrap'
import { getElement, getSimilarElements, ElementDetail } from '../api'

// Type configuration
const typeConfig: Record<string, { icon: string; color: string; label: string }> = {
  file: { icon: 'bi-file-code', color: 'info', label: 'File' },
  class: { icon: 'bi-box', color: 'purple', label: 'Class' },
  function: { icon: 'bi-braces', color: 'primary', label: 'Function' },
  method: { icon: 'bi-gear', color: 'success', label: 'Method' },
  variable: { icon: 'bi-x-diamond', color: 'secondary', label: 'Variable' },
  constant: { icon: 'bi-hash', color: 'warning', label: 'Constant' },
}

function getTypeConfig(type: string) {
  return typeConfig[type] || { icon: 'bi-dot', color: 'secondary', label: type }
}

function getTypeBadgeStyle(type: string): React.CSSProperties {
  if (type === 'class') {
    return { backgroundColor: '#6f42c1', color: 'white' }
  }
  return {}
}

// Language icons
const languageIcons: Record<string, string> = {
  python: 'bi-filetype-py',
  javascript: 'bi-filetype-js',
  typescript: 'bi-filetype-tsx',
  php: 'bi-filetype-php',
  rust: 'bi-filetype-rs',
}

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

  const config = getTypeConfig(element.element_type)
  const langIcon = languageIcons[element.language] || 'bi-file-code'
  const isCallable = ['function', 'method'].includes(element.element_type)
  const isContainer = ['file', 'class'].includes(element.element_type)

  return (
    <div>
      {/* Breadcrumb */}
      <Breadcrumb className="mb-4">
        <Breadcrumb.Item linkAs={Link} linkProps={{ to: '/repos' }}>
          Repositories
        </Breadcrumb.Item>
        <Breadcrumb.Item
          linkAs={Link}
          linkProps={{ to: `/repos/${element.repository.scope}/${element.repository.name}` }}
        >
          {element.repository.scope}/{element.repository.name}
        </Breadcrumb.Item>
        {element.context.file && element.element_type !== 'file' && (
          <Breadcrumb.Item
            linkAs={Link}
            linkProps={{ to: `/element/${element.context.file.hash_id || element.context.file.element_id}` }}
          >
            {element.context.file.name}
          </Breadcrumb.Item>
        )}
        {element.context.parent && (
          <Breadcrumb.Item
            linkAs={Link}
            linkProps={{ to: `/element/${element.context.parent.hash_id || element.context.parent.element_id}` }}
          >
            {element.context.parent.name}
          </Breadcrumb.Item>
        )}
        <Breadcrumb.Item active>{element.name}</Breadcrumb.Item>
      </Breadcrumb>

      <Row>
        <Col lg={8}>
          {/* Main Element Card */}
          <Card className="mb-4">
            <Card.Header>
              {/* Decorators */}
              {element.decorators && element.decorators.length > 0 && (
                <div className="mb-2">
                  {element.decorators.map((d, i) => (
                    <code key={i} className="me-2 text-info">@{d}</code>
                  ))}
                </div>
              )}

              {/* Name and badges */}
              <div className="d-flex justify-content-between align-items-start">
                <div>
                  <Badge
                    bg={config.color}
                    style={getTypeBadgeStyle(element.element_type)}
                    className="me-2"
                  >
                    <i className={`bi ${config.icon} me-1`}></i>
                    {config.label}
                  </Badge>
                  <span className="fs-4 fw-bold">{element.name}</span>

                  {/* Modifier badges */}
                  {element.visibility && element.visibility !== 'public' && (
                    <Badge bg="secondary" className="ms-2" pill>
                      {element.visibility}
                    </Badge>
                  )}
                  {element.is_async && (
                    <Badge bg="info" className="ms-2" pill>
                      async
                    </Badge>
                  )}
                </div>

                <div className="text-end">
                  <Badge bg="light" text="dark" className="me-2">
                    <i className={`bi ${langIcon} me-1`}></i>
                    {element.language}
                  </Badge>
                  <small className="text-muted">
                    L{element.line_start}
                    {element.line_end && element.line_end !== element.line_start && `-${element.line_end}`}
                  </small>
                </div>
              </div>

              {/* File path */}
              <div className="mt-2">
                <small className="text-muted">
                  <i className="bi bi-folder me-1"></i>
                  {element.file_path}
                </small>
              </div>
            </Card.Header>

            <Card.Body>
              {/* Signature for functions/methods */}
              {isCallable && element.signature && (
                <div className="mb-3">
                  <small className="text-uppercase text-muted fw-bold d-block mb-1">Signature</small>
                  <code className="d-block bg-light p-2 rounded">{element.signature}</code>
                </div>
              )}

              {/* Summary */}
              {element.summary && (
                <Alert variant="light" className="border mb-3">
                  <i className="bi bi-lightbulb me-2 text-warning"></i>
                  {element.summary}
                </Alert>
              )}

              {/* Docstring */}
              {element.docstring && (
                <div className="mb-3">
                  <small className="text-uppercase text-muted fw-bold d-block mb-1">Documentation</small>
                  <div className="bg-light p-3 rounded border-start border-4 border-info">
                    <pre className="mb-0" style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit' }}>
                      {element.docstring}
                    </pre>
                  </div>
                </div>
              )}

              {/* Code */}
              {element.raw_code && (
                <div>
                  <small className="text-uppercase text-muted fw-bold d-block mb-1">Source Code</small>
                  <pre className="bg-dark text-light p-3 rounded" style={{ maxHeight: '50vh', overflowY: 'auto' }}>
                    <code>{element.raw_code}</code>
                  </pre>
                </div>
              )}
            </Card.Body>
          </Card>

          {/* Children - shown for files and classes */}
          {isContainer && element.context.children && element.context.children.length > 0 && (
            <Card className="mb-4">
              <Card.Header>
                <i className="bi bi-diagram-3 me-2"></i>
                Contents ({element.context.children.length})
              </Card.Header>
              <Card.Body className="p-0">
                <Tabs defaultActiveKey="all" className="px-3 pt-2">
                  <Tab eventKey="all" title="All">
                    <ListGroup variant="flush">
                      {element.context.children.map((child) => {
                        const childConfig = getTypeConfig(child.element_type)
                        return (
                          <ListGroup.Item
                            key={child.element_id}
                            action
                            as={Link}
                            to={`/element/${child.hash_id || child.element_id}`}
                            className="d-flex justify-content-between align-items-start"
                          >
                            <div>
                              <Badge
                                bg={childConfig.color}
                                style={getTypeBadgeStyle(child.element_type)}
                                className="me-2"
                              >
                                <i className={`bi ${childConfig.icon}`}></i>
                              </Badge>
                              <span className="fw-medium">{child.name}</span>
                              {child.signature && (
                                <code className="ms-2 small text-muted">
                                  {child.signature.length > 40 ? child.signature.substring(0, 40) + '...' : child.signature}
                                </code>
                              )}
                              {child.summary && (
                                <small className="d-block text-muted ms-4 mt-1">{child.summary}</small>
                              )}
                            </div>
                            <small className="text-muted">L{child.line}</small>
                          </ListGroup.Item>
                        )
                      })}
                    </ListGroup>
                  </Tab>
                  {['class', 'function', 'method', 'variable', 'constant'].map((type) => {
                    const items = element.context.children.filter(c => c.element_type === type)
                    if (items.length === 0) return null
                    const tc = getTypeConfig(type)
                    return (
                      <Tab key={type} eventKey={type} title={`${tc.label}s (${items.length})`}>
                        <ListGroup variant="flush">
                          {items.map((child) => (
                            <ListGroup.Item
                              key={child.element_id}
                              action
                              as={Link}
                              to={`/element/${child.hash_id || child.element_id}`}
                              className="d-flex justify-content-between align-items-start"
                            >
                              <div>
                                <span className="fw-medium">{child.name}</span>
                                {child.signature && (
                                  <code className="ms-2 small text-muted">{child.signature}</code>
                                )}
                                {child.summary && (
                                  <small className="d-block text-muted mt-1">{child.summary}</small>
                                )}
                              </div>
                              <small className="text-muted">L{child.line}</small>
                            </ListGroup.Item>
                          ))}
                        </ListGroup>
                      </Tab>
                    )
                  })}
                </Tabs>
              </Card.Body>
            </Card>
          )}

          {/* Siblings - shown for non-file elements */}
          {element.element_type !== 'file' && element.context.siblings && element.context.siblings.length > 0 && (
            <Card className="mb-4">
              <Card.Header>
                <i className="bi bi-people me-2"></i>
                Siblings ({element.context.siblings.length})
              </Card.Header>
              <ListGroup variant="flush">
                {element.context.siblings.slice(0, 10).map((sibling) => {
                  const sibConfig = getTypeConfig(sibling.element_type)
                  return (
                    <ListGroup.Item
                      key={sibling.element_id}
                      action
                      as={Link}
                      to={`/element/${sibling.hash_id || sibling.element_id}`}
                      className="d-flex justify-content-between align-items-start"
                    >
                      <div>
                        <Badge
                          bg={sibConfig.color}
                          style={getTypeBadgeStyle(sibling.element_type)}
                          className="me-2"
                          pill
                        >
                          <i className={`bi ${sibConfig.icon}`}></i>
                        </Badge>
                        {sibling.name}
                        {sibling.summary && (
                          <small className="d-block text-muted ms-4">{sibling.summary}</small>
                        )}
                      </div>
                      <small className="text-muted">L{sibling.line}</small>
                    </ListGroup.Item>
                  )
                })}
                {element.context.siblings.length > 10 && (
                  <ListGroup.Item className="text-center text-muted">
                    +{element.context.siblings.length - 10} more
                  </ListGroup.Item>
                )}
              </ListGroup>
            </Card>
          )}
        </Col>

        <Col lg={4}>
          {/* Context Card */}
          {(element.context.parent || element.context.file) && (
            <Card className="mb-4">
              <Card.Header>
                <i className="bi bi-diagram-2 me-2"></i>
                Context
              </Card.Header>
              <ListGroup variant="flush">
                {element.context.file && element.element_type !== 'file' && (
                  <ListGroup.Item>
                    <small className="text-uppercase text-muted d-block">File</small>
                    <Link to={`/element/${element.context.file.element_id}`}>
                      <i className="bi bi-file-code me-1"></i>
                      {element.context.file.name}
                    </Link>
                    {element.context.file.summary && (
                      <small className="d-block text-muted mt-1">{element.context.file.summary}</small>
                    )}
                  </ListGroup.Item>
                )}
                {element.context.parent && (
                  <ListGroup.Item>
                    <small className="text-uppercase text-muted d-block">Parent</small>
                    <Link to={`/element/${element.context.parent.element_id}`}>
                      <Badge
                        bg={getTypeConfig(element.context.parent.element_type).color}
                        style={getTypeBadgeStyle(element.context.parent.element_type)}
                        className="me-1"
                        pill
                      >
                        <i className={`bi ${getTypeConfig(element.context.parent.element_type).icon}`}></i>
                      </Badge>
                      {element.context.parent.name}
                    </Link>
                    {element.context.parent.summary && (
                      <small className="d-block text-muted mt-1">{element.context.parent.summary}</small>
                    )}
                  </ListGroup.Item>
                )}
              </ListGroup>
            </Card>
          )}

          {/* Related Code */}
          {similar && similar.length > 0 && (
            <Card className="mb-4">
              <Card.Header>
                <i className="bi bi-diagram-3 me-2"></i>
                Related Code
              </Card.Header>
              <ListGroup variant="flush">
                {similar.map((sim) => {
                  const simConfig = getTypeConfig(sim.element_type)
                  return (
                    <ListGroup.Item
                      key={sim.element_id}
                      action
                      as={Link}
                      to={`/element/${sim.hash_id || sim.element_id}`}
                      className="d-flex justify-content-between align-items-start"
                    >
                      <div className="text-truncate me-2">
                        <Badge
                          bg={simConfig.color}
                          style={getTypeBadgeStyle(sim.element_type)}
                          className="me-1"
                          pill
                        >
                          <i className={`bi ${simConfig.icon}`}></i>
                        </Badge>
                        <span>{sim.name}</span>
                        {sim.summary && (
                          <small className="d-block text-muted text-truncate">{sim.summary}</small>
                        )}
                      </div>
                      <Badge bg="secondary" pill>
                        {(sim.similarity * 100).toFixed(0)}%
                      </Badge>
                    </ListGroup.Item>
                  )
                })}
              </ListGroup>
            </Card>
          )}

          {/* Info Card */}
          <Card>
            <Card.Header>
              <i className="bi bi-info-circle me-2"></i>
              Properties
            </Card.Header>
            <ListGroup variant="flush">
              <ListGroup.Item className="d-flex justify-content-between">
                <span className="text-muted">Type</span>
                <Badge
                  bg={config.color}
                  style={getTypeBadgeStyle(element.element_type)}
                >
                  {config.label}
                </Badge>
              </ListGroup.Item>
              <ListGroup.Item className="d-flex justify-content-between">
                <span className="text-muted">Language</span>
                <span>
                  <i className={`bi ${langIcon} me-1`}></i>
                  {element.language}
                </span>
              </ListGroup.Item>
              <ListGroup.Item className="d-flex justify-content-between">
                <span className="text-muted">Lines</span>
                <span>
                  {element.line_start}
                  {element.line_end && element.line_end !== element.line_start && `-${element.line_end}`}
                  {element.line_end && (
                    <small className="text-muted ms-1">
                      ({element.line_end - element.line_start + 1} lines)
                    </small>
                  )}
                </span>
              </ListGroup.Item>
              {element.visibility && (
                <ListGroup.Item className="d-flex justify-content-between">
                  <span className="text-muted">Visibility</span>
                  <Badge bg={element.visibility === 'public' ? 'success' : 'secondary'}>
                    {element.visibility}
                  </Badge>
                </ListGroup.Item>
              )}
              {isCallable && (
                <ListGroup.Item className="d-flex justify-content-between">
                  <span className="text-muted">Async</span>
                  <span>{element.is_async ? 'Yes' : 'No'}</span>
                </ListGroup.Item>
              )}
              {element.decorators && element.decorators.length > 0 && (
                <ListGroup.Item className="d-flex justify-content-between">
                  <span className="text-muted">Decorators</span>
                  <span>{element.decorators.length}</span>
                </ListGroup.Item>
              )}
              {isContainer && (
                <ListGroup.Item className="d-flex justify-content-between">
                  <span className="text-muted">Children</span>
                  <span>{element.context.children?.length ?? 0}</span>
                </ListGroup.Item>
              )}
              {element.docstring && (
                <ListGroup.Item className="d-flex justify-content-between">
                  <span className="text-muted">Documented</span>
                  <Badge bg="success">
                    <i className="bi bi-check"></i> Yes
                  </Badge>
                </ListGroup.Item>
              )}
            </ListGroup>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default Element
