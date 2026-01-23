import { useParams, useSearchParams } from 'react-router-dom'
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
} from 'react-bootstrap'
import { useState } from 'react'
import { getCallChain, type CallChainNode } from '../api'

// Type configuration
const typeConfig: Record<string, { icon: string; color: string }> = {
  function: { icon: 'bi-braces', color: 'primary' },
  method: { icon: 'bi-gear', color: 'success' },
}

function getTypeConfig(type: string | null) {
  if (!type) return { icon: 'bi-question', color: 'secondary' }
  return typeConfig[type] || { icon: 'bi-dot', color: 'secondary' }
}

// Recursive component for call chain tree
function CallChainTree({
  nodes,
  direction,
  depth = 0,
}: {
  nodes: CallChainNode[]
  direction: 'callers' | 'callees'
  depth?: number
}) {
  if (nodes.length === 0) return null

  return (
    <ListGroup variant="flush" className={depth > 0 ? 'ms-4' : ''}>
      {nodes.map((node, index) => {
        const config = getTypeConfig(node.element_type || null)
        const children = direction === 'callers' ? node.callers : node.callees

        return (
          <div key={`${node.name}-${index}`}>
            <ListGroup.Item className="border-0 py-2">
              <div className="d-flex align-items-center">
                {/* Indentation indicator */}
                {depth > 0 && (
                  <span className="text-muted me-2" style={{ opacity: 0.5 }}>
                    {direction === 'callers' ? '\u2190' : '\u2192'}
                  </span>
                )}

                {/* Type badge */}
                <Badge bg={config.color} className="me-2" pill>
                  <i className={`bi ${config.icon}`}></i>
                </Badge>

                {/* Name with link if resolved */}
                {node.element_id && !node.unresolved && !node.missing ? (
                  <Link
                    to={`/element/${node.hash_id || node.element_id}`}
                    className="fw-medium"
                  >
                    {node.name}
                  </Link>
                ) : (
                  <span className="fw-medium text-muted">{node.name}</span>
                )}

                {/* Status badges */}
                {node.cycle && (
                  <Badge bg="warning" className="ms-2">
                    cycle
                  </Badge>
                )}
                {node.unresolved && (
                  <Badge bg="secondary" className="ms-2">
                    unresolved
                  </Badge>
                )}
                {node.missing && (
                  <Badge bg="danger" className="ms-2">
                    missing
                  </Badge>
                )}

                {/* File info */}
                {node.file_path && (
                  <small className="text-muted ms-2">
                    {node.file_path}
                    {node.line && `:${node.line}`}
                  </small>
                )}
              </div>

              {/* Summary */}
              {node.summary && (
                <small className="d-block text-muted text-truncate mt-1 ms-4">
                  {node.summary}
                </small>
              )}
            </ListGroup.Item>

            {/* Recursive children */}
            {children && children.length > 0 && (
              <CallChainTree
                nodes={children}
                direction={direction}
                depth={depth + 1}
              />
            )}
          </div>
        )
      })}
    </ListGroup>
  )
}

function CallExplorer() {
  const { scope, repo } = useParams<{ scope: string; repo: string }>()
  const [searchParams] = useSearchParams()
  const elementId = searchParams.get('element')

  const [direction, setDirection] = useState<'both' | 'callers' | 'callees'>('both')
  const [maxDepth, setMaxDepth] = useState(3)

  const { data, isLoading, error } = useQuery({
    queryKey: ['call-chain', elementId, direction, maxDepth],
    queryFn: () =>
      getCallChain(elementId!, { direction, max_depth: maxDepth }),
    enabled: !!elementId,
  })

  if (!scope || !repo) {
    return <Alert variant="warning">Repository not specified</Alert>
  }

  if (!elementId) {
    return (
      <div>
        <Breadcrumb className="mb-4">
          <Breadcrumb.Item linkAs={Link} linkProps={{ to: '/repos' }}>
            Repositories
          </Breadcrumb.Item>
          <Breadcrumb.Item linkAs={Link} linkProps={{ to: `/repos/${scope}/${repo}` }}>
            {scope}/{repo}
          </Breadcrumb.Item>
          <Breadcrumb.Item active>Call Explorer</Breadcrumb.Item>
        </Breadcrumb>

        <Alert variant="info">
          <i className="bi bi-info-circle me-2"></i>
          Select an element to explore its call chain. You can access this from any function or method detail page.
        </Alert>
      </div>
    )
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
        Failed to load call chain: {(error as Error).message}
      </Alert>
    )
  }

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
        <Breadcrumb.Item active>Call Explorer</Breadcrumb.Item>
      </Breadcrumb>

      <h2 className="mb-4">
        <i className="bi bi-diagram-3 me-2"></i>
        Call Explorer
      </h2>

      {/* Root Element */}
      {data?.root && (
        <Card className="mb-4">
          <Card.Header>
            <i className="bi bi-bullseye me-2"></i>
            Root Element
          </Card.Header>
          <Card.Body>
            <h5>{data.root.name}</h5>
            <small className="text-muted">
              {data.root.file_path}:{data.root.line}
            </small>
            {data.root.summary && (
              <p className="mt-2 mb-0 text-muted">{data.root.summary}</p>
            )}
          </Card.Body>
        </Card>
      )}

      {/* Controls */}
      <Card className="mb-4">
        <Card.Body>
          <Row>
            <Col md={6}>
              <Form.Group>
                <Form.Label>Direction</Form.Label>
                <Form.Select
                  value={direction}
                  onChange={(e) => setDirection(e.target.value as 'both' | 'callers' | 'callees')}
                >
                  <option value="both">Both (Callers & Callees)</option>
                  <option value="callers">Callers Only</option>
                  <option value="callees">Callees Only</option>
                </Form.Select>
              </Form.Group>
            </Col>
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
          </Row>
        </Card.Body>
      </Card>

      {/* Call Chain Results */}
      <Row>
        {(direction === 'both' || direction === 'callers') && (
          <Col md={direction === 'both' ? 6 : 12}>
            <Card className="mb-4">
              <Card.Header>
                <i className="bi bi-arrow-left me-2"></i>
                Callers (Who calls this?)
                {data?.callers && (
                  <Badge bg="primary" className="ms-2">{data.callers.length}</Badge>
                )}
              </Card.Header>
              <Card.Body className="p-0">
                {data?.callers && data.callers.length > 0 ? (
                  <CallChainTree nodes={data.callers} direction="callers" />
                ) : (
                  <Alert variant="light" className="m-3 text-center text-muted">
                    No callers found
                  </Alert>
                )}
              </Card.Body>
            </Card>
          </Col>
        )}

        {(direction === 'both' || direction === 'callees') && (
          <Col md={direction === 'both' ? 6 : 12}>
            <Card className="mb-4">
              <Card.Header>
                <i className="bi bi-arrow-right me-2"></i>
                Callees (What does this call?)
                {data?.callees && (
                  <Badge bg="success" className="ms-2">{data.callees.length}</Badge>
                )}
              </Card.Header>
              <Card.Body className="p-0">
                {data?.callees && data.callees.length > 0 ? (
                  <CallChainTree nodes={data.callees} direction="callees" />
                ) : (
                  <Alert variant="light" className="m-3 text-center text-muted">
                    No callees found
                  </Alert>
                )}
              </Card.Body>
            </Card>
          </Col>
        )}
      </Row>
    </div>
  )
}

export default CallExplorer
