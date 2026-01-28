import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  Row,
  Col,
  Card,
  Badge,
  Spinner,
  Alert,
  Breadcrumb,
  ListGroup,
  Tab,
  Tabs,
  Form,
} from 'react-bootstrap'
import { useState } from 'react'
import {
  getComplexFunctions,
  getSecurityIssues,
  getUndocumentedFunctions,
  getEnvUsage,
  getAsyncCode,
} from '../api'

// Type configuration
const typeConfig: Record<string, { icon: string; color: string }> = {
  function: { icon: 'bi-braces', color: 'primary' },
  method: { icon: 'bi-gear', color: 'success' },
  class: { icon: 'bi-box', color: 'purple' },
  variable: { icon: 'bi-x-diamond', color: 'secondary' },
}

function getTypeConfig(type: string) {
  return typeConfig[type] || { icon: 'bi-dot', color: 'secondary' }
}

function CodeHealth() {
  const { scope, repo } = useParams<{ scope: string; repo: string }>()
  const [activeTab, setActiveTab] = useState('complexity')
  const [minComplexity, setMinComplexity] = useState(5)
  const [maxCoverage, setMaxCoverage] = useState(0.5)
  const [includeTests, setIncludeTests] = useState(false)

  // Fetch complexity data
  const { data: complexFunctions, isLoading: complexLoading } = useQuery({
    queryKey: ['complexFunctions', scope, repo, minComplexity, includeTests],
    queryFn: () => getComplexFunctions(scope!, repo!, {
      min_complexity: minComplexity,
      limit: 100,
      include_tests: includeTests
    }),
    enabled: !!scope && !!repo && activeTab === 'complexity',
  })

  // Fetch security issues
  const { data: securityIssues, isLoading: securityLoading } = useQuery({
    queryKey: ['securityIssues', scope, repo],
    queryFn: () => getSecurityIssues(scope!, repo!, { limit: 100 }),
    enabled: !!scope && !!repo && activeTab === 'security',
  })

  // Fetch undocumented functions
  const { data: undocumented, isLoading: undocLoading } = useQuery({
    queryKey: ['undocumented', scope, repo, maxCoverage, includeTests],
    queryFn: () => getUndocumentedFunctions(scope!, repo!, {
      max_coverage: maxCoverage,
      limit: 100,
      include_tests: includeTests
    }),
    enabled: !!scope && !!repo && activeTab === 'documentation',
  })

  // Fetch environment variable usage
  const { data: envUsage, isLoading: envLoading } = useQuery({
    queryKey: ['envUsage', scope, repo],
    queryFn: () => getEnvUsage(scope!, repo!, { limit: 200 }),
    enabled: !!scope && !!repo && activeTab === 'env',
  })

  // Fetch async code
  const { data: asyncCode, isLoading: asyncLoading } = useQuery({
    queryKey: ['asyncCode', scope, repo],
    queryFn: () => getAsyncCode(scope!, repo!, { limit: 100 }),
    enabled: !!scope && !!repo && activeTab === 'async',
  })

  if (!scope || !repo) {
    return <Alert variant="warning">No repo selected</Alert>
  }

  return (
    <div>
      <Breadcrumb className="mb-4">
        <Breadcrumb.Item linkAs={Link} linkProps={{ to: '/repos' }}>
          Repositories
        </Breadcrumb.Item>
        <Breadcrumb.Item
          linkAs={Link}
          linkProps={{ to: `/repos/${scope}/${repo}` }}
        >
          {scope}/{repo}
        </Breadcrumb.Item>
        <Breadcrumb.Item active>Code Health</Breadcrumb.Item>
      </Breadcrumb>

      <h2 className="mb-4">
        <i className="bi bi-heart-pulse me-2"></i>
        Code Health
      </h2>

      <Tabs activeKey={activeTab} onSelect={(k) => setActiveTab(k || 'complexity')} className="mb-4">
        {/* Complexity Tab */}
        <Tab eventKey="complexity" title={<><i className="bi bi-speedometer2 me-1"></i> Complexity</>}>
          <Row className="mb-3">
            <Col md={6}>
              <Form.Group className="d-flex align-items-center">
                <Form.Label className="me-2 mb-0">Min Complexity:</Form.Label>
                <Form.Range
                  min={1}
                  max={20}
                  value={minComplexity}
                  onChange={(e) => setMinComplexity(parseInt(e.target.value))}
                  style={{ width: '150px' }}
                />
                <Badge bg="secondary" className="ms-2">{minComplexity}</Badge>
              </Form.Group>
            </Col>
            <Col md={6}>
              <Form.Check
                type="switch"
                label="Include test functions"
                checked={includeTests}
                onChange={(e) => setIncludeTests(e.target.checked)}
              />
            </Col>
          </Row>

          {complexLoading ? (
            <div className="text-center py-5"><Spinner animation="border" /></div>
          ) : complexFunctions?.functions.length ? (
            <Card>
              <Card.Header>
                <div className="d-flex justify-content-between">
                  <span>Complex Functions ({complexFunctions.count})</span>
                  <small className="text-muted">Complexity &ge; {complexFunctions.min_complexity}</small>
                </div>
              </Card.Header>
              <ListGroup variant="flush">
                {complexFunctions.functions.map((fn) => (
                  <ListGroup.Item
                    key={fn.element_id}
                    action
                    as={Link}
                    to={`/element/${encodeURIComponent(fn.hash_id || fn.element_id)}`}
                  >
                    <div className="d-flex justify-content-between align-items-start">
                      <div>
                        <Badge
                          bg={getTypeConfig(fn.element_type).color}
                          style={fn.element_type === 'class' ? { backgroundColor: '#6f42c1' } : {}}
                          className="me-2"
                        >
                          <i className={`bi ${getTypeConfig(fn.element_type).icon}`}></i>
                        </Badge>
                        <span className="fw-medium">{fn.name}</span>
                        {fn.is_test && <Badge bg="warning" text="dark" className="ms-2">test</Badge>}
                        <small className="text-muted d-block ms-4 mt-1">
                          {fn.file_path}:{fn.line}
                        </small>
                        {fn.summary && (
                          <small className="text-muted d-block ms-4">{fn.summary}</small>
                        )}
                      </div>
                      <div className="text-end">
                        <Badge bg={fn.cyclomatic > 15 ? 'danger' : fn.cyclomatic > 10 ? 'warning' : 'info'}>
                          CC: {fn.cyclomatic}
                        </Badge>
                        <small className="text-muted d-block mt-1">
                          {fn.line_count} lines, depth {fn.nesting_depth}
                        </small>
                      </div>
                    </div>
                  </ListGroup.Item>
                ))}
              </ListGroup>
            </Card>
          ) : (
            <Alert variant="success">
              <i className="bi bi-check-circle me-2"></i>
              No functions with complexity &ge; {minComplexity} found.
            </Alert>
          )}
        </Tab>

        {/* Security Tab */}
        <Tab eventKey="security" title={<><i className="bi bi-shield-exclamation me-1"></i> Security</>}>
          {securityLoading ? (
            <div className="text-center py-5"><Spinner animation="border" /></div>
          ) : securityIssues?.issues.length ? (
            <>
              {/* Summary Cards */}
              <Row className="mb-4">
                {Object.entries(securityIssues.by_severity)
                  .sort(([a], [b]) => {
                    const order = ['critical', 'high', 'medium', 'low', 'info']
                    return order.indexOf(a) - order.indexOf(b)
                  })
                  .map(([severity, count]) => (
                    <Col key={severity} xs={6} md={2}>
                      <Card className="text-center">
                        <Card.Body className="py-2">
                          <h4 className={`mb-0 text-${severity === 'critical' ? 'danger' : severity === 'high' ? 'warning' : severity === 'medium' ? 'info' : 'secondary'}`}>
                            {count}
                          </h4>
                          <small className="text-muted text-capitalize">{severity}</small>
                        </Card.Body>
                      </Card>
                    </Col>
                  ))}
              </Row>

              <Card>
                <Card.Header>Security Issues ({securityIssues.count})</Card.Header>
                <ListGroup variant="flush">
                  {securityIssues.issues.map((issue, idx) => (
                    <ListGroup.Item
                      key={`${issue.element_id}-${idx}`}
                      action
                      as={Link}
                      to={`/element/${encodeURIComponent(issue.hash_id || issue.element_id)}`}
                    >
                      <div className="d-flex justify-content-between align-items-start">
                        <div>
                          <Badge
                            bg={issue.severity === 'critical' ? 'danger' : issue.severity === 'high' ? 'warning' : issue.severity === 'medium' ? 'info' : 'secondary'}
                            className="me-2"
                          >
                            {issue.severity}
                          </Badge>
                          <span className="fw-medium">{issue.function_name}</span>
                          <Badge bg="light" text="dark" className="ms-2">{issue.kind}</Badge>
                          <small className="text-muted d-block ms-4 mt-1">
                            {issue.file_path}:{issue.issue_line}
                          </small>
                          {issue.message && (
                            <small className="d-block ms-4 mt-1">{issue.message}</small>
                          )}
                        </div>
                      </div>
                    </ListGroup.Item>
                  ))}
                </ListGroup>
              </Card>
            </>
          ) : (
            <Alert variant="success">
              <i className="bi bi-shield-check me-2"></i>
              No security issues detected.
            </Alert>
          )}
        </Tab>

        {/* Documentation Tab */}
        <Tab eventKey="documentation" title={<><i className="bi bi-file-text me-1"></i> Documentation</>}>
          <Row className="mb-3">
            <Col md={6}>
              <Form.Group className="d-flex align-items-center">
                <Form.Label className="me-2 mb-0">Max Coverage:</Form.Label>
                <Form.Range
                  min={0}
                  max={100}
                  step={10}
                  value={maxCoverage * 100}
                  onChange={(e) => setMaxCoverage(parseInt(e.target.value) / 100)}
                  style={{ width: '150px' }}
                />
                <Badge bg="secondary" className="ms-2">{Math.round(maxCoverage * 100)}%</Badge>
              </Form.Group>
            </Col>
            <Col md={6}>
              <Form.Check
                type="switch"
                label="Include test functions"
                checked={includeTests}
                onChange={(e) => setIncludeTests(e.target.checked)}
              />
            </Col>
          </Row>

          {undocLoading ? (
            <div className="text-center py-5"><Spinner animation="border" /></div>
          ) : undocumented?.functions.length ? (
            <Card>
              <Card.Header>
                <div className="d-flex justify-content-between">
                  <span>Under-documented Functions ({undocumented.count})</span>
                  <small className="text-muted">Coverage &le; {Math.round(undocumented.max_coverage * 100)}%</small>
                </div>
              </Card.Header>
              <ListGroup variant="flush">
                {undocumented.functions.map((fn) => (
                  <ListGroup.Item
                    key={fn.element_id}
                    action
                    as={Link}
                    to={`/element/${encodeURIComponent(fn.hash_id || fn.element_id)}`}
                  >
                    <div className="d-flex justify-content-between align-items-start">
                      <div>
                        <Badge
                          bg={getTypeConfig(fn.element_type).color}
                          style={fn.element_type === 'class' ? { backgroundColor: '#6f42c1' } : {}}
                          className="me-2"
                        >
                          <i className={`bi ${getTypeConfig(fn.element_type).icon}`}></i>
                        </Badge>
                        <span className="fw-medium">{fn.name}</span>
                        {fn.visibility && fn.visibility !== 'public' && (
                          <Badge bg="secondary" className="ms-2">{fn.visibility}</Badge>
                        )}
                        <small className="text-muted d-block ms-4 mt-1">
                          {fn.file_path}:{fn.line}
                        </small>
                      </div>
                      <div className="text-end">
                        <Badge bg={fn.coverage >= 0.5 ? 'warning' : fn.coverage > 0 ? 'secondary' : 'danger'}>
                          {Math.round(fn.coverage * 100)}%
                        </Badge>
                        <small className="text-muted d-block mt-1">
                          {fn.param_count} params
                          {!fn.has_docstring && ', no docstring'}
                        </small>
                      </div>
                    </div>
                  </ListGroup.Item>
                ))}
              </ListGroup>
            </Card>
          ) : (
            <Alert variant="success">
              <i className="bi bi-book-half me-2"></i>
              All functions have documentation coverage &gt; {Math.round(maxCoverage * 100)}%.
            </Alert>
          )}
        </Tab>

        {/* Environment Variables Tab */}
        <Tab eventKey="env" title={<><i className="bi bi-gear-wide-connected me-1"></i> Env Vars</>}>
          {envLoading ? (
            <div className="text-center py-5"><Spinner animation="border" /></div>
          ) : envUsage?.usages.length ? (
            <>
              {/* Summary by Variable */}
              <Row className="mb-4">
                {Object.entries(envUsage.by_var).slice(0, 8).map(([varName, count]) => (
                  <Col key={varName} xs={6} md={3} className="mb-2">
                    <Card className="text-center">
                      <Card.Body className="py-2">
                        <code>{varName}</code>
                        <Badge bg="secondary" className="ms-2">{count}</Badge>
                      </Card.Body>
                    </Card>
                  </Col>
                ))}
              </Row>

              <Card>
                <Card.Header>
                  Environment Variable Usage ({envUsage.count} usages, {envUsage.unique_vars} unique vars)
                </Card.Header>
                <ListGroup variant="flush">
                  {envUsage.usages.map((usage, idx) => (
                    <ListGroup.Item
                      key={`${usage.element_id}-${idx}`}
                      action
                      as={Link}
                      to={`/element/${encodeURIComponent(usage.hash_id || usage.element_id)}`}
                    >
                      <div className="d-flex justify-content-between align-items-start">
                        <div>
                          <code className="me-2">{usage.env_name}</code>
                          <span className="text-muted">in</span>
                          <span className="ms-2 fw-medium">{usage.function_name}</span>
                          <small className="text-muted d-block mt-1">
                            {usage.file_path}:{usage.env_line}
                          </small>
                        </div>
                        <Badge bg="light" text="dark">{usage.access_type}</Badge>
                      </div>
                    </ListGroup.Item>
                  ))}
                </ListGroup>
              </Card>
            </>
          ) : (
            <Alert variant="info">
              <i className="bi bi-info-circle me-2"></i>
              No environment variable usage detected.
            </Alert>
          )}
        </Tab>

        {/* Async/Concurrency Tab */}
        <Tab eventKey="async" title={<><i className="bi bi-lightning me-1"></i> Async</>}>
          {asyncLoading ? (
            <div className="text-center py-5"><Spinner animation="border" /></div>
          ) : asyncCode?.functions.length ? (
            <>
              {/* Summary by Pattern */}
              <Row className="mb-4">
                {Object.entries(asyncCode.by_pattern).map(([pattern, count]) => (
                  <Col key={pattern} xs={6} md={2} className="mb-2">
                    <Card className="text-center">
                      <Card.Body className="py-2">
                        <Badge bg="info">{count}</Badge>
                        <small className="d-block text-muted text-capitalize">{pattern}</small>
                      </Card.Body>
                    </Card>
                  </Col>
                ))}
              </Row>

              <Card>
                <Card.Header>
                  Async/Concurrent Code ({asyncCode.count})
                </Card.Header>
                <ListGroup variant="flush">
                  {asyncCode.functions.map((fn) => (
                    <ListGroup.Item
                      key={fn.element_id}
                      action
                      as={Link}
                      to={`/element/${encodeURIComponent(fn.hash_id || fn.element_id)}`}
                    >
                      <div className="d-flex justify-content-between align-items-start">
                        <div>
                          <Badge
                            bg={getTypeConfig(fn.element_type).color}
                            style={fn.element_type === 'class' ? { backgroundColor: '#6f42c1' } : {}}
                            className="me-2"
                          >
                            <i className={`bi ${getTypeConfig(fn.element_type).icon}`}></i>
                          </Badge>
                          <span className="fw-medium">{fn.name}</span>
                          <small className="text-muted d-block ms-4 mt-1">
                            {fn.file_path}:{fn.line}
                          </small>
                          {fn.summary && (
                            <small className="text-muted d-block ms-4">{fn.summary}</small>
                          )}
                        </div>
                        <div className="d-flex flex-wrap gap-1 justify-content-end" style={{ maxWidth: '200px' }}>
                          {fn.is_async && <Badge bg="info">async</Badge>}
                          {fn.uses_threads && <Badge bg="warning" text="dark">threads</Badge>}
                          {fn.uses_locks && <Badge bg="secondary">locks</Badge>}
                          {fn.patterns.filter(p => !['async'].includes(p)).map((p, i) => (
                            <Badge key={i} bg="light" text="dark">{p}</Badge>
                          ))}
                        </div>
                      </div>
                    </ListGroup.Item>
                  ))}
                </ListGroup>
              </Card>
            </>
          ) : (
            <Alert variant="info">
              <i className="bi bi-info-circle me-2"></i>
              No async or concurrent code detected.
            </Alert>
          )}
        </Tab>
      </Tabs>
    </div>
  )
}

export default CodeHealth
