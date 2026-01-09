import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useSearchParams, Link } from 'react-router-dom'
import {
  Row,
  Col,
  Card,
  Form,
  Table,
  Badge,
  Spinner,
  Alert,
  Pagination,
  Button,
  Collapse,
  OverlayTrigger,
  Tooltip,
} from 'react-bootstrap'
import {
  getBrowseFilters,
  browseElements,
  getBrowseStats,
  getElementChildren,
  BrowseElement,
} from '../api'

// Element type icons and colors
const typeConfig: Record<string, { icon: string; color: string }> = {
  file: { icon: 'bi-file-code', color: 'info' },
  class: { icon: 'bi-box', color: 'primary' },
  function: { icon: 'bi-braces', color: 'success' },
  method: { icon: 'bi-gear', color: 'warning' },
  variable: { icon: 'bi-x-diamond', color: 'secondary' },
  constant: { icon: 'bi-hash', color: 'danger' },
  feature: { icon: 'bi-collection', color: 'info' },
  subfeature: { icon: 'bi-collection-fill', color: 'info' },
}

function ElementRow({ element, username }: { element: BrowseElement; username: string }) {
  const [expanded, setExpanded] = useState(false)

  const { data: children, isLoading: childrenLoading } = useQuery({
    queryKey: ['element-children', element.element_id, username],
    queryFn: () => getElementChildren(element.element_id, username),
    enabled: expanded && (element.element_type === 'file' || element.element_type === 'class'),
  })

  const config = typeConfig[element.element_type] || { icon: 'bi-dot', color: 'secondary' }
  const hasChildren = element.element_type === 'file' || element.element_type === 'class'

  return (
    <>
      <tr>
        <td style={{ width: '30px' }}>
          {hasChildren && (
            <Button
              variant="link"
              size="sm"
              className="p-0 text-muted"
              onClick={() => setExpanded(!expanded)}
            >
              <i className={`bi ${expanded ? 'bi-chevron-down' : 'bi-chevron-right'}`}></i>
            </Button>
          )}
        </td>
        <td>
          <i className={`bi ${config.icon} me-2 text-${config.color}`}></i>
          <Link to={`/element/${encodeURIComponent(element.element_id)}`} className="text-decoration-none">
            <strong>{element.name}</strong>
          </Link>
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
        </td>
        <td>
          <Badge bg={config.color}>{element.element_type}</Badge>
        </td>
        <td>
          <code className="small text-muted">{element.file_path}</code>
          {element.line_start > 0 && (
            <span className="text-muted small ms-1">:{element.line_start}</span>
          )}
        </td>
        <td>
          <Badge bg="secondary">{element.language}</Badge>
        </td>
        <td className="small text-muted" style={{ maxWidth: '300px' }}>
          {element.summary ? (
            <OverlayTrigger
              placement="top"
              overlay={<Tooltip>{element.summary}</Tooltip>}
            >
              <span className="text-truncate d-inline-block" style={{ maxWidth: '280px' }}>
                {element.summary}
              </span>
            </OverlayTrigger>
          ) : (
            <span className="text-muted fst-italic">No summary</span>
          )}
        </td>
      </tr>
      {hasChildren && (
        <tr>
          <td colSpan={6} className="p-0 border-0">
            <Collapse in={expanded}>
              <div>
                {childrenLoading ? (
                  <div className="p-3 ps-5 bg-light">
                    <Spinner size="sm" animation="border" /> Loading children...
                  </div>
                ) : children && children.total_children > 0 ? (
                  <div className="ps-5 py-2 bg-light border-bottom">
                    {Object.entries(children.children).map(([type, items]) => (
                      <div key={type} className="mb-2">
                        <small className="text-uppercase text-muted fw-bold">{type}s ({items.length})</small>
                        <div className="ms-3">
                          {items.map((child) => {
                            const childConfig = typeConfig[child.element_type] || { icon: 'bi-dot', color: 'secondary' }
                            return (
                              <div key={child.element_id} className="py-1 border-bottom border-light">
                                <i className={`bi ${childConfig.icon} me-2 text-${childConfig.color}`}></i>
                                <Link
                                  to={`/element/${encodeURIComponent(child.element_id)}`}
                                  className="text-decoration-none"
                                >
                                  {child.name}
                                </Link>
                                {child.visibility && child.visibility !== 'public' && (
                                  <Badge bg="secondary" className="ms-2" pill style={{ fontSize: '0.65em' }}>
                                    {child.visibility}
                                  </Badge>
                                )}
                                {child.is_async && (
                                  <Badge bg="info" className="ms-2" pill style={{ fontSize: '0.65em' }}>
                                    async
                                  </Badge>
                                )}
                                {child.signature && (
                                  <code className="ms-2 small text-muted">{child.signature}</code>
                                )}
                                {child.summary && (
                                  <span className="ms-2 small text-muted">- {child.summary}</span>
                                )}
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : expanded ? (
                  <div className="p-3 ps-5 bg-light text-muted small">
                    No child elements
                  </div>
                ) : null}
              </div>
            </Collapse>
          </td>
        </tr>
      )}
    </>
  )
}

function Explorer() {
  const [searchParams, setSearchParams] = useSearchParams()

  // Get filter values from URL params
  const scope = searchParams.get('scope') || ''
  const repository = searchParams.get('repository') || ''
  const username = searchParams.get('user') || 'main'
  const elementType = searchParams.get('type') || ''
  const language = searchParams.get('lang') || ''
  const page = parseInt(searchParams.get('page') || '1', 10)

  // Fetch filter options
  const { data: filters, isLoading: filtersLoading } = useQuery({
    queryKey: ['browse-filters'],
    queryFn: getBrowseFilters,
  })

  // Fetch stats
  const { data: stats } = useQuery({
    queryKey: ['browse-stats', scope, repository, username],
    queryFn: () => getBrowseStats({ scope: scope || undefined, repository: repository || undefined, username }),
    enabled: true,
  })

  // Fetch elements
  const { data: elements, isLoading: elementsLoading, error } = useQuery({
    queryKey: ['browse-elements', scope, repository, username, elementType, language, page],
    queryFn: () =>
      browseElements({
        scope: scope || undefined,
        repository: repository || undefined,
        username,
        element_type: elementType || undefined,
        language: language || undefined,
        page,
        limit: 50,
      }),
  })

  // Get filtered repositories based on selected scope
  const filteredRepos = filters?.repositories.filter(
    (r) => !scope || r.scope === scope
  ) || []

  // Update URL when filters change
  const updateFilter = (key: string, value: string) => {
    const newParams = new URLSearchParams(searchParams)
    if (value) {
      newParams.set(key, value)
    } else {
      newParams.delete(key)
    }
    // Reset page when filters change (except for page itself)
    if (key !== 'page') {
      newParams.delete('page')
    }
    setSearchParams(newParams)
  }

  // Reset repository when scope changes
  useEffect(() => {
    if (scope && repository) {
      const repoExists = filters?.repositories.some(
        (r) => r.scope === scope && r.repository === repository
      )
      if (!repoExists) {
        updateFilter('repository', '')
      }
    }
  }, [scope, filters])

  if (filtersLoading) {
    return (
      <div className="text-center py-5">
        <Spinner animation="border" role="status">
          <span className="visually-hidden">Loading...</span>
        </Spinner>
      </div>
    )
  }

  return (
    <div>
      <h1 className="mb-4">
        <i className="bi bi-folder2-open me-2"></i>
        Code Explorer
      </h1>

      {/* Filters */}
      <Card className="mb-4">
        <Card.Body>
          <Row>
            <Col md={2}>
              <Form.Group>
                <Form.Label className="small text-muted">Scope</Form.Label>
                <Form.Select
                  size="sm"
                  value={scope}
                  onChange={(e) => updateFilter('scope', e.target.value)}
                >
                  <option value="">All Scopes</option>
                  {filters?.scopes.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </Form.Select>
              </Form.Group>
            </Col>
            <Col md={3}>
              <Form.Group>
                <Form.Label className="small text-muted">Repository</Form.Label>
                <Form.Select
                  size="sm"
                  value={repository}
                  onChange={(e) => updateFilter('repository', e.target.value)}
                >
                  <option value="">All Repositories</option>
                  {filteredRepos.map((r) => (
                    <option key={`${r.scope}/${r.repository}`} value={r.repository}>
                      {scope ? r.repository : `${r.scope}/${r.repository}`}
                    </option>
                  ))}
                </Form.Select>
              </Form.Group>
            </Col>
            <Col md={2}>
              <Form.Group>
                <Form.Label className="small text-muted">User Branch</Form.Label>
                <Form.Select
                  size="sm"
                  value={username}
                  onChange={(e) => updateFilter('user', e.target.value)}
                >
                  {filters?.usernames.map((u) => (
                    <option key={u} value={u}>
                      {u}
                    </option>
                  ))}
                </Form.Select>
              </Form.Group>
            </Col>
            <Col md={2}>
              <Form.Group>
                <Form.Label className="small text-muted">Element Type</Form.Label>
                <Form.Select
                  size="sm"
                  value={elementType}
                  onChange={(e) => updateFilter('type', e.target.value)}
                >
                  <option value="">All Types</option>
                  {filters?.element_types
                    .filter((t) => !['feature', 'subfeature'].includes(t))
                    .map((t) => (
                      <option key={t} value={t}>
                        {t} {stats?.type_counts[t] ? `(${stats.type_counts[t].toLocaleString()})` : ''}
                      </option>
                    ))}
                </Form.Select>
              </Form.Group>
            </Col>
            <Col md={2}>
              <Form.Group>
                <Form.Label className="small text-muted">Language</Form.Label>
                <Form.Select
                  size="sm"
                  value={language}
                  onChange={(e) => updateFilter('lang', e.target.value)}
                >
                  <option value="">All Languages</option>
                  {filters?.languages.map((l) => (
                    <option key={l} value={l}>
                      {l} {stats?.language_counts[l] ? `(${stats.language_counts[l].toLocaleString()})` : ''}
                    </option>
                  ))}
                </Form.Select>
              </Form.Group>
            </Col>
            <Col md={1} className="d-flex align-items-end">
              <Button
                variant="outline-secondary"
                size="sm"
                onClick={() => setSearchParams(new URLSearchParams())}
                className="w-100"
              >
                Clear
              </Button>
            </Col>
          </Row>
        </Card.Body>
      </Card>

      {/* Stats Summary */}
      {stats && (
        <Row className="mb-3 g-2">
          {Object.entries(stats.type_counts)
            .filter(([type]) => !['feature', 'subfeature'].includes(type))
            .map(([type, count]) => {
              const config = typeConfig[type] || { icon: 'bi-dot', color: 'secondary' }
              const isActive = elementType === type
              return (
                <Col key={type}>
                  <Card
                    className={`text-center h-100 ${isActive ? 'border-primary' : ''}`}
                    style={{ cursor: 'pointer' }}
                    onClick={() => updateFilter('type', isActive ? '' : type)}
                  >
                    <Card.Body className="py-2">
                      <i className={`bi ${config.icon} text-${config.color}`}></i>
                      <h5 className="mb-0">{count.toLocaleString()}</h5>
                      <small className="text-muted text-capitalize">{type}s</small>
                    </Card.Body>
                  </Card>
                </Col>
              )
            })}
        </Row>
      )}

      {/* Error */}
      {error && (
        <Alert variant="danger">
          Failed to load elements: {(error as Error).message}
        </Alert>
      )}

      {/* Results */}
      <Card>
        <Card.Header className="d-flex justify-content-between align-items-center">
          <span>
            {elementsLoading ? (
              <Spinner size="sm" animation="border" className="me-2" />
            ) : (
              <Badge bg="primary" className="me-2">
                {elements?.total.toLocaleString() || 0}
              </Badge>
            )}
            Elements
          </span>
          {elements && elements.total_pages > 1 && (
            <small className="text-muted">
              Page {elements.page} of {elements.total_pages}
            </small>
          )}
        </Card.Header>
        <Card.Body className="p-0">
          {elementsLoading ? (
            <div className="text-center py-5">
              <Spinner animation="border" />
            </div>
          ) : elements?.elements.length ? (
            <Table hover responsive className="mb-0">
              <thead className="bg-light">
                <tr>
                  <th style={{ width: '30px' }}></th>
                  <th>Name</th>
                  <th style={{ width: '100px' }}>Type</th>
                  <th>Location</th>
                  <th style={{ width: '100px' }}>Language</th>
                  <th>Summary</th>
                </tr>
              </thead>
              <tbody>
                {elements.elements.map((el) => (
                  <ElementRow key={el.element_id} element={el} username={username} />
                ))}
              </tbody>
            </Table>
          ) : (
            <div className="text-center py-5 text-muted">
              <i className="bi bi-inbox fs-1 d-block mb-2"></i>
              No elements found
            </div>
          )}
        </Card.Body>
        {elements && elements.total_pages > 1 && (
          <Card.Footer className="d-flex justify-content-center">
            <Pagination className="mb-0">
              <Pagination.First
                disabled={page === 1}
                onClick={() => updateFilter('page', '1')}
              />
              <Pagination.Prev
                disabled={page === 1}
                onClick={() => updateFilter('page', String(page - 1))}
              />
              {Array.from({ length: Math.min(5, elements.total_pages) }, (_, i) => {
                let pageNum: number
                if (elements.total_pages <= 5) {
                  pageNum = i + 1
                } else if (page <= 3) {
                  pageNum = i + 1
                } else if (page >= elements.total_pages - 2) {
                  pageNum = elements.total_pages - 4 + i
                } else {
                  pageNum = page - 2 + i
                }
                return (
                  <Pagination.Item
                    key={pageNum}
                    active={pageNum === page}
                    onClick={() => updateFilter('page', String(pageNum))}
                  >
                    {pageNum}
                  </Pagination.Item>
                )
              })}
              <Pagination.Next
                disabled={page === elements.total_pages}
                onClick={() => updateFilter('page', String(page + 1))}
              />
              <Pagination.Last
                disabled={page === elements.total_pages}
                onClick={() => updateFilter('page', String(elements.total_pages))}
              />
            </Pagination>
          </Card.Footer>
        )}
      </Card>
    </div>
  )
}

export default Explorer
