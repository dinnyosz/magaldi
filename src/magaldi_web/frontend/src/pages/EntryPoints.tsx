import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  Card,
  Badge,
  Spinner,
  Alert,
  Tab,
  Tabs,
  ListGroup,
  Row,
  Col,
  Breadcrumb,
} from 'react-bootstrap'
import { getEntryPoints, type EntryPointItem } from '../api'

// Type configuration
const typeConfig: Record<string, { icon: string; color: string; label: string }> = {
  function: { icon: 'bi-braces', color: 'primary', label: 'Function' },
  method: { icon: 'bi-gear', color: 'success', label: 'Method' },
}

function getTypeConfig(type: string) {
  return typeConfig[type] || { icon: 'bi-dot', color: 'secondary', label: type }
}

// Category configuration
const categoryConfig: Record<string, { icon: string; color: string; label: string }> = {
  http: { icon: 'bi-globe', color: 'primary', label: 'HTTP Handlers' },
  cli: { icon: 'bi-terminal', color: 'success', label: 'CLI Commands' },
  test: { icon: 'bi-check2-square', color: 'info', label: 'Test Fixtures' },
  main: { icon: 'bi-play-circle', color: 'warning', label: 'Main Functions' },
  async_tasks: { icon: 'bi-clock-history', color: 'secondary', label: 'Async Tasks' },
}

function EntryPointList({ items, category }: { items: EntryPointItem[]; category: string }) {
  const config = categoryConfig[category]

  if (items.length === 0) {
    return (
      <Alert variant="light" className="text-center text-muted">
        <i className={`bi ${config.icon} me-2`}></i>
        No {config.label.toLowerCase()} found
      </Alert>
    )
  }

  return (
    <ListGroup variant="flush">
      {items.map((item) => {
        const typeConf = getTypeConfig(item.element_type)
        return (
          <ListGroup.Item
            key={item.element_id}
            action
            as={Link}
            to={`/element/${encodeURIComponent(item.hash_id || item.element_id)}`}
          >
            <div className="d-flex justify-content-between align-items-start">
              <div className="me-2">
                <Badge bg={typeConf.color} className="me-1" pill>
                  <i className={`bi ${typeConf.icon}`}></i>
                </Badge>
                <span className="fw-medium">{item.name}</span>
                <small className="text-muted ms-2">{item.file_path}:{item.line}</small>
              </div>
            </div>
            {/* HTTP Routes - show method and path */}
            {category === 'http' && item.http_routes && item.http_routes.length > 0 && (
              <div className="mt-1">
                {item.http_routes.map((route, i) => (
                  <span key={i} className="me-2">
                    <Badge bg="primary" className="me-1">{route.method}</Badge>
                    <code className="text-success">{route.path}</code>
                    {route.framework && (
                      <small className="text-muted ms-1">({route.framework})</small>
                    )}
                  </span>
                ))}
              </div>
            )}
            {/* CLI Commands - show command name (for both CLI and Main categories) */}
            {(category === 'cli' || category === 'main') && item.cli_commands && item.cli_commands.length > 0 && (
              <div className="mt-1">
                {item.cli_commands.map((cmd, i) => (
                  <span key={i} className="me-2">
                    <code className="text-success">$ {cmd.name}</code>
                    {cmd.framework && (
                      <small className="text-muted ms-1">({cmd.framework})</small>
                    )}
                    {cmd.options && cmd.options.length > 0 && (
                      <small className="text-muted ms-1">
                        [{cmd.options.length} options]
                      </small>
                    )}
                  </span>
                ))}
              </div>
            )}
            {/* Fallback to decorators for other categories or when routes/commands empty */}
            {((category !== 'http' && category !== 'cli' && category !== 'main') ||
              (category === 'http' && (!item.http_routes || item.http_routes.length === 0)) ||
              ((category === 'cli' || category === 'main') && (!item.cli_commands || item.cli_commands.length === 0))) &&
              item.decorators && item.decorators.length > 0 && (
              <div className="mt-1">
                {item.decorators.slice(0, 3).map((d, i) => (
                  <code key={i} className="me-2 text-info small">@{d}</code>
                ))}
                {item.decorators.length > 3 && (
                  <span className="text-muted small">+{item.decorators.length - 3} more</span>
                )}
              </div>
            )}
            {/* Summary - allow multi-line */}
            {item.summary && (
              <small
                className="d-block text-muted mt-1"
                style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
              >
                {item.summary}
              </small>
            )}
          </ListGroup.Item>
        )
      })}
    </ListGroup>
  )
}

function EntryPoints() {
  const { scope, repo } = useParams<{ scope: string; repo: string }>()

  const { data, isLoading, error } = useQuery({
    queryKey: ['entry-points', scope, repo],
    queryFn: () => getEntryPoints(scope!, repo!),
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
        Failed to load entry points: {(error as Error).message}
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
        <Breadcrumb.Item active>Entry Points</Breadcrumb.Item>
      </Breadcrumb>

      <h2 className="mb-4">
        <i className="bi bi-door-open me-2"></i>
        Entry Points
      </h2>

      {/* Stats Card */}
      <Card className="mb-4">
        <Card.Body>
          <Row>
            <Col className="text-center border-end">
              <h3 className="text-primary mb-0">{data?.stats.total_http ?? 0}</h3>
              <small className="text-muted">HTTP</small>
            </Col>
            <Col className="text-center border-end">
              <h3 className="text-success mb-0">{data?.stats.total_cli ?? 0}</h3>
              <small className="text-muted">CLI</small>
            </Col>
            <Col className="text-center border-end">
              <h3 className="text-info mb-0">{data?.stats.total_test ?? 0}</h3>
              <small className="text-muted">Test</small>
            </Col>
            <Col className="text-center border-end">
              <h3 className="text-warning mb-0">{data?.stats.total_main ?? 0}</h3>
              <small className="text-muted">Main</small>
            </Col>
            <Col className="text-center">
              <h3 className="text-secondary mb-0">{data?.stats.total_async ?? 0}</h3>
              <small className="text-muted">Async</small>
            </Col>
          </Row>
        </Card.Body>
      </Card>

      {/* Tabbed Entry Points */}
      <Card>
        <Card.Body>
          <Tabs defaultActiveKey="http" className="mb-3">
            <Tab
              eventKey="http"
              title={
                <>
                  <i className="bi bi-globe me-1"></i>
                  HTTP
                  {data?.http.length ? <Badge bg="primary" className="ms-1">{data.http.length}</Badge> : null}
                </>
              }
            >
              <EntryPointList items={data?.http || []} category="http" />
            </Tab>
            <Tab
              eventKey="cli"
              title={
                <>
                  <i className="bi bi-terminal me-1"></i>
                  CLI
                  {data?.cli.length ? <Badge bg="success" className="ms-1">{data.cli.length}</Badge> : null}
                </>
              }
            >
              <EntryPointList items={data?.cli || []} category="cli" />
            </Tab>
            <Tab
              eventKey="test"
              title={
                <>
                  <i className="bi bi-check2-square me-1"></i>
                  Test
                  {data?.test.length ? <Badge bg="info" className="ms-1">{data.test.length}</Badge> : null}
                </>
              }
            >
              <EntryPointList items={data?.test || []} category="test" />
            </Tab>
            <Tab
              eventKey="main"
              title={
                <>
                  <i className="bi bi-play-circle me-1"></i>
                  Main
                  {data?.main.length ? <Badge bg="warning" className="ms-1">{data.main.length}</Badge> : null}
                </>
              }
            >
              <EntryPointList items={data?.main || []} category="main" />
            </Tab>
            <Tab
              eventKey="async"
              title={
                <>
                  <i className="bi bi-clock-history me-1"></i>
                  Async
                  {data?.async_tasks.length ? <Badge bg="secondary" className="ms-1">{data.async_tasks.length}</Badge> : null}
                </>
              }
            >
              <EntryPointList items={data?.async_tasks || []} category="async_tasks" />
            </Tab>
          </Tabs>
        </Card.Body>
      </Card>
    </div>
  )
}

export default EntryPoints
