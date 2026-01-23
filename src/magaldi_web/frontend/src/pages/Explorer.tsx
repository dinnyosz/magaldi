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
  getElementDetails,
  type BrowseElement,
  type CallGraphEntry,
} from '../api'

// Element type icons, colors, and behavior flags
type TypeConfig = {
  icon: string
  color: string
  canHaveChildren?: boolean
  canHaveCallGraph?: boolean
}

const typeConfig: Record<string, TypeConfig> = {
  file: { icon: 'bi-file-code', color: 'info', canHaveChildren: true },
  class: { icon: 'bi-box', color: 'primary', canHaveChildren: true },
  function: { icon: 'bi-braces', color: 'success', canHaveCallGraph: true },
  method: { icon: 'bi-gear', color: 'warning', canHaveCallGraph: true },
  variable: { icon: 'bi-x-diamond', color: 'secondary' },
  constant: { icon: 'bi-hash', color: 'danger' },
  feature: { icon: 'bi-collection', color: 'info' },
  subfeature: { icon: 'bi-collection-fill', color: 'info' },
}

const defaultTypeConfig: TypeConfig = { icon: 'bi-dot', color: 'secondary' }

// Simple pluralization for element types
function pluralize(type: string): string {
  if (type === 'class') return 'Classes'
  return `${type}s`
}

function ElementRow({ element, username }: { element: BrowseElement; username: string }) {
  const [expanded, setExpanded] = useState(false)
  const [showDetails, setShowDetails] = useState(false)

  const { data: children, isLoading: childrenLoading } = useQuery({
    queryKey: ['element-children', element.hash_id, username],
    queryFn: () => element.hash_id ? getElementChildren(element.hash_id, username) : Promise.resolve({ element_id: '', hash_id: null, children: {}, total_children: 0 }),
    enabled: expanded && !!element.hash_id,
  })

  const { data: details, isLoading: detailsLoading } = useQuery({
    queryKey: ['element-details', element.hash_id, username],
    queryFn: () => element.hash_id ? getElementDetails(element.hash_id, { includeCallGraph: true, username }) : Promise.resolve({ error: 'No hash_id' } as any),
    enabled: showDetails && !!element.hash_id,
  })

  const config = typeConfig[element.element_type] || defaultTypeConfig
  // Use type config flags to determine expand behavior
  const hasChildren = !!config.canHaveChildren && !!element.hash_id
  const canShowDetails = !!config.canHaveCallGraph && !!element.hash_id

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
          {canShowDetails && (
            <Button
              variant="link"
              size="sm"
              className="p-0 text-muted"
              onClick={() => setShowDetails(!showDetails)}
            >
              <i className={`bi ${showDetails ? 'bi-chevron-down' : 'bi-chevron-right'}`}></i>
            </Button>
          )}
        </td>
        <td>
          {/* Decorators */}
          {element.decorators && element.decorators.length > 0 && (
            <div className="small text-muted mb-1">
              {element.decorators.map((d, i) => (
                <span key={i} className="me-1 text-info">@{d}</span>
              ))}
            </div>
          )}
          <i className={`bi ${config.icon} me-2 text-${config.color}`}></i>
          <Link to={`/element/${encodeURIComponent(element.hash_id || element.element_id)}`} className="text-decoration-none">
            <strong>{element.name}</strong>
          </Link>
          {/* Signature for functions/methods */}
          {element.signature && (element.element_type === 'function' || element.element_type === 'method') && (
            <code className="ms-1 small text-muted">
              {element.signature.length > 60 ? element.signature.substring(0, 60) + '...' : element.signature}
            </code>
          )}
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
          {element.has_docstring && (
            <Badge bg="success" className="ms-2" pill title="Has docstring">
              <i className="bi bi-file-text"></i>
            </Badge>
          )}
        </td>
        <td>
          <Badge bg={config.color}>{element.element_type}</Badge>
        </td>
        <td>
          {/* Container info for methods */}
          {element.container && (
            <div className="small">
              <i className={`bi ${typeConfig[element.container.element_type]?.icon || 'bi-dot'} me-1 text-${typeConfig[element.container.element_type]?.color || 'secondary'}`}></i>
              <Link
                to={`/element/${element.container.element_id}`}
                className="text-decoration-none text-primary"
              >
                {element.container.name}
              </Link>
            </div>
          )}
          <code className="small text-muted">{element.file_path}</code>
          {element.line_start > 0 && (
            <span className="text-muted small ms-1">:{element.line_start}</span>
          )}
          {element.line_end && element.line_end !== element.line_start && (
            <span className="text-muted small">-{element.line_end}</span>
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

      {/* Children panel for files/classes */}
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
                                  to={`/element/${encodeURIComponent(child.hash_id || child.element_id)}`}
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

      {/* Details panel for functions/methods with call graph */}
      {canShowDetails && (
        <tr>
          <td colSpan={6} className="p-0 border-0">
            <Collapse in={showDetails}>
              <div className="ps-5 py-2 bg-light border-bottom">
                {detailsLoading ? (
                  <div className="p-2">
                    <Spinner size="sm" animation="border" /> Loading details...
                  </div>
                ) : details && !details.error ? (
                  <div className="row">
                    {/* Containers (parent hierarchy) */}
                    {details.containers && details.containers.length > 0 && (
                      <div className="col-md-3 mb-2">
                        <small className="text-uppercase text-muted fw-bold d-block mb-1">Container Hierarchy</small>
                        {details.containers.map((c: CallGraphEntry, i: number) => (
                          <div key={c.element_id} className="small" style={{ paddingLeft: `${i * 10}px` }}>
                            <i className={`bi ${typeConfig[c.element_type]?.icon || 'bi-dot'} me-1 text-${typeConfig[c.element_type]?.color || 'secondary'}`}></i>
                            <Link to={`/element/${encodeURIComponent(c.hash_id || c.element_id)}`} className="text-decoration-none">
                              {c.name}
                            </Link>
                            <span className="text-muted ms-1">({c.element_type})</span>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Callers */}
                    <div className="col-md-4 mb-2">
                      <small className="text-uppercase text-muted fw-bold d-block mb-1">
                        <i className="bi bi-arrow-left-circle me-1"></i>
                        Called by ({details.callers?.length || 0})
                      </small>
                      {details.callers && details.callers.length > 0 ? (
                        <div className="small" style={{ maxHeight: '150px', overflowY: 'auto' }}>
                          {details.callers.map((caller: CallGraphEntry) => (
                            <div key={caller.element_id} className="py-1">
                              <Link
                                to={`/element/${encodeURIComponent(caller.hash_id || caller.element_id)}`}
                                className="text-decoration-none"
                              >
                                {caller.name}
                              </Link>
                              <span className="text-muted ms-1">
                                in {caller.file_path.split('/').pop()}:{caller.line_start}
                              </span>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <span className="text-muted small fst-italic">No callers found</span>
                      )}
                    </div>

                    {/* Callees */}
                    <div className="col-md-4 mb-2">
                      <small className="text-uppercase text-muted fw-bold d-block mb-1">
                        <i className="bi bi-arrow-right-circle me-1"></i>
                        Calls ({details.callees?.length || 0})
                      </small>
                      {details.callees && details.callees.length > 0 ? (
                        <div className="small" style={{ maxHeight: '150px', overflowY: 'auto' }}>
                          {details.callees.map((callee: CallGraphEntry) => (
                            <div key={callee.element_id} className="py-1">
                              <Link
                                to={`/element/${encodeURIComponent(callee.hash_id || callee.element_id)}`}
                                className="text-decoration-none"
                              >
                                {callee.name}
                              </Link>
                              <span className="text-muted ms-1">
                                in {callee.file_path.split('/').pop()}:{callee.line_start}
                              </span>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <span className="text-muted small fst-italic">No calls found</span>
                      )}
                    </div>
                  </div>
                ) : showDetails ? (
                  <div className="p-2 text-muted small">
                    {details?.error || 'Could not load details'}
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
        Explorer
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
                  {filters?.element_types.map((t) => (
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
          {Object.entries(stats.type_counts).map(([type, count]) => {
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
                      <small className="text-muted text-capitalize">{pluralize(type)}</small>
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
