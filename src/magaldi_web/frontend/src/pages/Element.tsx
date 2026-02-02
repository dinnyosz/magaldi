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
import ReactMarkdown from 'react-markdown'
import { getElement, getSimilarElements, explainElement, getGlossaryTermsForFeature, type ElementDetail as _ElementDetail } from '../api'

// Type configuration
const typeConfig: Record<string, { icon: string; color: string; label: string }> = {
  file: { icon: 'bi-file-code', color: 'info', label: 'File' },
  class: { icon: 'bi-box', color: 'purple', label: 'Class' },
  interface: { icon: 'bi-layers', color: 'info', label: 'Interface' },
  type_alias: { icon: 'bi-type', color: 'info', label: 'Type Alias' },
  trait: { icon: 'bi-diagram-3', color: 'info', label: 'Trait' },
  enum: { icon: 'bi-list-ol', color: 'warning', label: 'Enum' },
  function: { icon: 'bi-braces', color: 'primary', label: 'Function' },
  method: { icon: 'bi-gear', color: 'success', label: 'Method' },
  variable: { icon: 'bi-x-diamond', color: 'secondary', label: 'Variable' },
  constant: { icon: 'bi-hash', color: 'warning', label: 'Constant' },
  feature: { icon: 'bi-collection', color: 'info', label: 'Feature' },
  subfeature: { icon: 'bi-collection-fill', color: 'info', label: 'Subfeature' },
  glossary: { icon: 'bi-book', color: 'primary', label: 'Glossary Term' },
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

  // Fetch call analysis data for functions/methods
  const { data: explanation } = useQuery({
    queryKey: ['explain', decodedId],
    queryFn: () => explainElement(decodedId),
    enabled: !!decodedId && !!element && ['function', 'method'].includes(element.element_type),
  })

  // Fetch glossary terms for features/subfeatures
  const { data: glossaryTerms } = useQuery({
    queryKey: ['glossaryForFeature', decodedId],
    queryFn: () => getGlossaryTermsForFeature(decodedId),
    enabled: !!decodedId && !!element && ['feature', 'subfeature'].includes(element.element_type),
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
  const isFeature = ['feature', 'subfeature'].includes(element.element_type)
  const isGlossary = element.element_type === 'glossary'

  // Helper to extract clean value from decorator args like ("serve") -> serve
  const extractCleanArg = (args: string): string => {
    // Strip whitespace and outer parentheses
    let cleaned = args.trim()
    if (cleaned.startsWith('(') && cleaned.endsWith(')')) {
      cleaned = cleaned.slice(1, -1).trim()
    }
    // Extract first quoted string
    const match = cleaned.match(/^["']([^"']+)["']/)
    if (match) {
      return match[1]
    }
    return cleaned
  }

  // Detect entry point type from decorators (multi-language support)
  const entryPointPatterns: Record<string, { decorators: string[], label: string, icon: string, color: string }> = {
    http: {
      decorators: [
        // Python: Flask, FastAPI, Django REST
        'app.route', 'router.get', 'router.post', 'router.put', 'router.delete', 'router.patch',
        'api_view', 'action',
        // JavaScript/TypeScript: NestJS, Express decorators
        '@get', '@post', '@put', '@delete', '@patch', '@controller', '@requestmapping',
        // PHP: Symfony attributes
        '#[route', '#[get', '#[post', '#[put', '#[delete',
        // Rust: Actix-web, Rocket, Axum
        '#[get(', '#[post(', '#[put(', '#[delete(', '#[route(',
        'actix_web::get', 'actix_web::post', 'rocket::get', 'rocket::post',
        // Generic patterns
        'route', 'endpoint', 'api',
      ],
      label: 'HTTP Endpoint',
      icon: 'bi-globe',
      color: 'primary',
    },
    cli: {
      decorators: [
        // Python: Click, Typer
        'click.command', 'click.group', 'typer.command',
        // Rust: Clap
        '#[command', '#[clap',
        // Generic
        'command', 'subcommand',
      ],
      label: 'CLI Command',
      icon: 'bi-terminal',
      color: 'success',
    },
    test: {
      decorators: [
        // Python: pytest
        'pytest.fixture', 'fixture',
        // Rust
        '#[test', '#[tokio::test', '#[async_std::test',
        // PHP: PHPUnit
        '@test', '#[test',
        // JavaScript/TypeScript: Jest, Mocha (usually function names, but some use decorators)
        '@test', '@it', '@describe',
      ],
      label: 'Test',
      icon: 'bi-check2-circle',
      color: 'info',
    },
    async_task: {
      decorators: [
        // Python: Celery, RQ, Dramatiq
        'celery.task', 'dramatiq.actor', 'rq.job',
        // JavaScript: Bull, Agenda
        '@processor', '@process', '@queue',
        // Generic
        'task', 'job', 'worker', 'background',
      ],
      label: 'Async Task',
      icon: 'bi-lightning',
      color: 'warning',
    },
    event: {
      decorators: [
        // JavaScript/TypeScript: EventEmitter, NestJS
        '@on', '@subscribe', '@eventhandler', '@listener',
        // PHP: Symfony
        '#[aseventslistener',
        // Generic
        'event', 'handler', 'listener',
      ],
      label: 'Event Handler',
      icon: 'bi-broadcast',
      color: 'secondary',
    },
    scheduled: {
      decorators: [
        // Python: APScheduler, Celery beat
        '@scheduled', 'cron', 'interval',
        // JavaScript: NestJS
        '@cron', '@interval',
        // Generic
        'schedule', 'periodic',
      ],
      label: 'Scheduled',
      icon: 'bi-clock',
      color: 'dark',
    },
  }

  let entryPointType: { label: string, icon: string, color: string, args?: string } | null = null
  if (element.decorators && element.decorators.length > 0) {
    for (const [, epConfig] of Object.entries(entryPointPatterns)) {
      for (const decorator of element.decorators) {
        const decoratorLower = decorator.toLowerCase()
        if (epConfig.decorators.some(d => decoratorLower.includes(d.toLowerCase()))) {
          // Try to find matching decorator_details to get args
          let args: string | undefined
          if (element.decorator_details) {
            const matchingDetail = element.decorator_details.find(
              dd => dd.name && decorator.toLowerCase().includes(dd.name.toLowerCase())
            )
            if (matchingDetail?.args) {
              args = extractCleanArg(matchingDetail.args)
            }
          }
          entryPointType = { label: epConfig.label, icon: epConfig.icon, color: epConfig.color, args }
          break
        }
      }
      if (entryPointType) break
    }
  }
  // Also check for main function (common across languages)
  if (!entryPointType && element.element_type === 'function') {
    const mainNames = ['main', '__main__', 'run', 'start', 'bootstrap', 'init']
    if (mainNames.includes(element.name.toLowerCase())) {
      entryPointType = { label: 'Main Entry', icon: 'bi-play-circle', color: 'danger' }
    }
  }

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
            linkProps={{ to: `/element/${encodeURIComponent(element.context.file.hash_id || element.context.file.element_id)}` }}
          >
            {element.context.file.name}
          </Breadcrumb.Item>
        )}
        {element.context.parent && element.context.parent.element_type !== 'file' && (
          <Breadcrumb.Item
            linkAs={Link}
            linkProps={{ to: `/element/${encodeURIComponent(element.context.parent.hash_id || element.context.parent.element_id)}` }}
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
                  {element.decorator_details && element.decorator_details.length > 0
                    ? element.decorator_details.map((dd, i) => (
                        <code key={i} className="me-2 text-info">@{dd.full || dd.name}</code>
                      ))
                    : element.decorators.map((d, i) => (
                        <code key={i} className="me-2 text-info">@{d}</code>
                      ))
                  }
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
                  {element.is_test && (
                    <Badge bg="warning" text="dark" className="ms-2" pill>
                      test
                    </Badge>
                  )}
                  {entryPointType && (
                    <Badge bg={entryPointType.color} className="ms-2" pill>
                      <i className={`bi ${entryPointType.icon} me-1`}></i>
                      {entryPointType.label}
                      {entryPointType.args && (
                        <code className="ms-1 fw-normal" style={{ opacity: 0.9 }}>{entryPointType.args}</code>
                      )}
                    </Badge>
                  )}
                </div>

                <div className="text-end">
                  {element.language && (
                    <Badge bg="light" text="dark" className="me-2">
                      <i className={`bi ${langIcon} me-1`}></i>
                      {element.language}
                    </Badge>
                  )}
                  {element.line_start > 0 && (
                    <small className="text-muted">
                      L{element.line_start}
                      {element.line_end && element.line_end !== element.line_start && `-${element.line_end}`}
                    </small>
                  )}
                  {isFeature && element.feature_info && (
                    <Badge bg="secondary" className="ms-2">
                      {element.feature_info.member_count} members
                    </Badge>
                  )}
                  {isGlossary && element.glossary_info && element.glossary_info.feature_count > 0 && (
                    <Badge bg="secondary" className="ms-2">
                      {element.glossary_info.feature_count} source features
                    </Badge>
                  )}
                </div>
              </div>

              {/* File path - only shown for code elements */}
              {element.file_path && (
                <div className="mt-2">
                  <small className="text-muted">
                    <i className="bi bi-folder me-1"></i>
                    {element.file_path}
                  </small>
                </div>
              )}
            </Card.Header>

            <Card.Body>
              {/* Signature for functions/methods */}
              {isCallable && element.signature && (
                <div className="mb-3">
                  <small className="text-uppercase text-muted fw-bold d-block mb-1">Signature</small>
                  <code className="d-block bg-light p-2 rounded">{element.signature}</code>
                </div>
              )}

              {/* Parameters for functions/methods */}
              {isCallable && element.parameters && element.parameters.length > 0 && (
                <div className="mb-3">
                  <small className="text-uppercase text-muted fw-bold d-block mb-1">
                    <i className="bi bi-input-cursor me-1"></i>
                    Parameters
                  </small>
                  <div className="bg-light p-2 rounded">
                    {element.parameters.map((param, i) => (
                      <div key={i} className="d-flex align-items-center py-1">
                        <code className="me-2">{param.name}</code>
                        {param.type && <small className="text-muted">: {param.type}</small>}
                        {param.default && <small className="text-muted ms-2">= {param.default}</small>}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Return Type for functions/methods */}
              {isCallable && element.return_type && (
                <div className="mb-3">
                  <small className="text-uppercase text-muted fw-bold d-block mb-1">
                    <i className="bi bi-arrow-return-right me-1"></i>
                    Returns
                  </small>
                  <code className="bg-light p-2 rounded d-inline-block">{element.return_type}</code>
                </div>
              )}

              {/* HTTP Routes */}
              {element.http_routes && element.http_routes.length > 0 && (
                <div className="mb-3">
                  <small className="text-uppercase text-muted fw-bold d-block mb-1">
                    <i className="bi bi-globe me-1"></i>
                    HTTP Routes
                  </small>
                  <div className="bg-light p-2 rounded">
                    {element.http_routes.map((route, i) => (
                      <div key={i} className="d-flex align-items-center py-1">
                        <Badge bg={route.method === 'GET' ? 'success' : route.method === 'POST' ? 'primary' : route.method === 'DELETE' ? 'danger' : 'secondary'} className="me-2">
                          {route.method}
                        </Badge>
                        <code className="me-2">{route.path}</code>
                        <small className="text-muted">({route.framework})</small>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* CLI Commands */}
              {element.cli_commands && element.cli_commands.length > 0 && (
                <div className="mb-3">
                  <small className="text-uppercase text-muted fw-bold d-block mb-1">
                    <i className="bi bi-terminal me-1"></i>
                    CLI Commands
                  </small>
                  <div className="bg-light p-2 rounded">
                    {element.cli_commands.map((cmd, i) => (
                      <div key={i} className="d-flex align-items-center py-1">
                        <code className="me-2">{cmd.name}</code>
                        <small className="text-muted">({cmd.framework})</small>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Detected Patterns */}
              {element.detected_patterns && element.detected_patterns.length > 0 && (
                <div className="mb-3">
                  <small className="text-uppercase text-muted fw-bold d-block mb-1">
                    <i className="bi bi-puzzle me-1"></i>
                    Design Patterns
                  </small>
                  <div className="d-flex flex-wrap gap-2">
                    {element.detected_patterns.map((pattern, i) => (
                      <Badge key={i} bg="info" className="fw-normal">
                        {pattern}
                        {element.pattern_confidence?.[pattern] && (
                          <small className="ms-1 opacity-75">
                            ({Math.round(element.pattern_confidence[pattern] * 100)}%)
                          </small>
                        )}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Purity */}
              {element.purity && (
                <div className="mb-3">
                  <small className="text-uppercase text-muted fw-bold d-block mb-1">
                    <i className="bi bi-shield-check me-1"></i>
                    Purity
                  </small>
                  <Badge bg={element.purity.level === 'pure' ? 'success' : element.purity.level === 'read_only' ? 'info' : 'warning'} className="me-2">
                    {element.purity.level}
                  </Badge>
                  <small className="text-muted">({element.purity.confidence} confidence)</small>
                  {element.purity.reasons.length > 0 && (
                    <div className="mt-1">
                      {element.purity.reasons.map((reason, i) => (
                        <small key={i} className="d-block text-muted">{reason}</small>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Side Effects */}
              {element.side_effects && element.side_effects.length > 0 && (
                <div className="mb-3">
                  <small className="text-uppercase text-muted fw-bold d-block mb-1">
                    <i className="bi bi-exclamation-diamond me-1"></i>
                    Side Effects
                  </small>
                  <div className="bg-light p-2 rounded">
                    {element.side_effects.map((effect, i) => (
                      <div key={i} className="d-flex align-items-center py-1">
                        <Badge bg="warning" text="dark" className="me-2">{effect.kind}</Badge>
                        {effect.target && <code className="me-2">{effect.target}</code>}
                        <small className="text-muted ms-auto">L{effect.line}</small>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* TODOs */}
              {element.todos && element.todos.length > 0 && (
                <div className="mb-3">
                  <small className="text-uppercase text-muted fw-bold d-block mb-1">
                    <i className="bi bi-check2-square me-1"></i>
                    TODOs ({element.todos.length})
                  </small>
                  <div className="bg-light p-2 rounded" style={{ maxHeight: '200px', overflowY: 'auto' }}>
                    {element.todos.map((todo, i) => (
                      <div key={i} className="py-1 border-bottom">
                        <Badge bg={todo.kind === 'FIXME' ? 'danger' : todo.kind === 'BUG' ? 'warning' : 'secondary'} className="me-2">
                          {todo.kind}
                        </Badge>
                        <span>{todo.text}</span>
                        {todo.assignee && <small className="text-muted ms-2">@{todo.assignee}</small>}
                        <small className="text-muted ms-auto float-end">L{todo.line}</small>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Summary */}
              {element.summary && (
                <Alert variant="light" className="border mb-3">
                  <i className="bi bi-lightbulb me-2 text-warning"></i>
                  {isGlossary ? (
                    <div className="glossary-summary"><ReactMarkdown>{element.summary}</ReactMarkdown></div>
                  ) : (
                    element.summary
                  )}
                </Alert>
              )}

              {/* Glossary Info */}
              {isGlossary && element.glossary_info && (
                <>
                  {/* Source Features */}
                  {element.glossary_info.feature_associations.length > 0 && (
                    <Card className="mb-3">
                      <Card.Header>
                        <i className="bi bi-diagram-3 me-2"></i>
                        Extracted From ({element.glossary_info.feature_associations.length} features)
                      </Card.Header>
                      <ListGroup variant="flush">
                        {element.glossary_info.feature_associations.map((assoc) => (
                          <ListGroup.Item
                            key={assoc.feature_id}
                            action
                            as={Link}
                            to={`/element/${encodeURIComponent(assoc.hash_id || assoc.feature_id)}`}
                            className="d-flex justify-content-between align-items-start"
                          >
                            <div>
                              <Badge bg="info" className="me-2">
                                <i className="bi bi-collection"></i>
                              </Badge>
                              <span className="fw-medium">{assoc.feature_label}</span>
                              {assoc.summary && (
                                <small className="d-block text-muted ms-4 mt-1">{assoc.summary}</small>
                              )}
                            </div>
                            {assoc.member_count > 0 && (
                              <Badge bg="secondary" pill>
                                {assoc.member_count} members
                              </Badge>
                            )}
                          </ListGroup.Item>
                        ))}
                      </ListGroup>
                    </Card>
                  )}

                  {/* File Paths - only show if we have data */}
                  {element.glossary_info.file_paths && element.glossary_info.file_paths.length > 0 && (
                    <div className="mb-3">
                      <small className="text-uppercase text-muted fw-bold d-block mb-1">
                        <i className="bi bi-file-earmark-code me-1"></i>
                        Files ({element.glossary_info.file_paths.length})
                      </small>
                      <div className="bg-light p-2 rounded" style={{ maxHeight: '200px', overflowY: 'auto' }}>
                        {element.glossary_info.file_paths.slice(0, 20).map((path, i) => (
                          <div key={i} className="py-1">
                            <code className="small text-muted">{path}</code>
                          </div>
                        ))}
                        {element.glossary_info.file_paths.length > 20 && (
                          <div className="py-1 text-muted small">
                            ... and {element.glossary_info.file_paths.length - 20} more files
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </>
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

              {/* Base Classes - for classes */}
              {element.element_type === 'class' && element.base_classes && element.base_classes.length > 0 && (
                <div className="mb-3">
                  <small className="text-uppercase text-muted fw-bold d-block mb-1">
                    <i className="bi bi-diagram-2 me-1"></i>
                    Inherits From
                  </small>
                  <div className="d-flex flex-wrap gap-2">
                    {element.base_classes.map((base, i) => (
                      <Badge key={i} bg="secondary" className="fw-normal">
                        {base}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Class Attributes - for classes */}
              {element.element_type === 'class' && element.class_attributes && element.class_attributes.length > 0 && (
                <div className="mb-3">
                  <small className="text-uppercase text-muted fw-bold d-block mb-1">
                    <i className="bi bi-box me-1"></i>
                    Instance Attributes
                  </small>
                  <div className="bg-light p-2 rounded">
                    {element.class_attributes.map((attr, i) => (
                      <div key={i} className="d-flex align-items-center py-1">
                        <code className="me-2">{attr.name}</code>
                        {attr.type && <small className="text-muted">: {attr.type}</small>}
                        {attr.line && <small className="text-muted ms-auto">L{attr.line}</small>}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Exceptions Raised - for functions/methods */}
              {isCallable && element.exceptions_raised && element.exceptions_raised.length > 0 && (
                <div className="mb-3">
                  <small className="text-uppercase text-muted fw-bold d-block mb-1">
                    <i className="bi bi-exclamation-triangle me-1"></i>
                    Raises
                  </small>
                  <div className="d-flex flex-wrap gap-2">
                    {element.exceptions_raised.map((exc, i) => (
                      <Badge key={i} bg="danger" className="fw-normal">
                        {exc}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Attributes Modified - for methods */}
              {element.element_type === 'method' && element.attributes_modified && element.attributes_modified.length > 0 && (
                <div className="mb-3">
                  <small className="text-uppercase text-muted fw-bold d-block mb-1">
                    <i className="bi bi-pencil me-1"></i>
                    Modifies Attributes
                  </small>
                  <div className="d-flex flex-wrap gap-2">
                    {element.attributes_modified.map((attr, i) => (
                      <Badge key={i} bg="warning" text="dark" className="fw-normal">
                        self.{attr}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Imports - for file elements */}
              {element.element_type === 'file' && element.imports && element.imports.length > 0 && (
                <div className="mb-3">
                  <small className="text-uppercase text-muted fw-bold d-block mb-1">
                    <i className="bi bi-box-arrow-in-down me-1"></i>
                    Imports ({element.imports.length})
                  </small>
                  <div className="bg-body-tertiary border rounded p-2" style={{ maxHeight: '200px', overflowY: 'auto' }}>
                    {element.imports.map((imp, i) => (
                      <div key={i} className="d-flex align-items-center py-1">
                        <code className="text-primary me-2">
                          {imp.alias ? `${imp.name} as ${imp.alias}` : imp.name}
                        </code>
                        <small className="text-body-secondary">from {imp.module}</small>
                        {imp.line && <small className="text-body-secondary ms-auto">L{imp.line}</small>}
                      </div>
                    ))}
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
                            to={`/element/${encodeURIComponent(child.hash_id || child.element_id)}`}
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
                              to={`/element/${encodeURIComponent(child.hash_id || child.element_id)}`}
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

          {/* Subfeatures - shown for features (above members) */}
          {element.element_type === 'feature' && element.feature_info && element.feature_info.subfeatures && element.feature_info.subfeatures.length > 0 && (
            <Card className="mb-4">
              <Card.Header>
                <i className="bi bi-collection-fill me-2"></i>
                Subfeatures ({element.feature_info.subfeatures.length})
              </Card.Header>
              <ListGroup variant="flush">
                {element.feature_info.subfeatures.map((subfeature) => (
                  <ListGroup.Item
                    key={subfeature.element_id}
                    action
                    as={Link}
                    to={`/element/${encodeURIComponent(subfeature.hash_id || subfeature.element_id)}`}
                    className="d-flex justify-content-between align-items-start"
                  >
                    <div>
                      <Badge bg="info" className="me-2">
                        <i className="bi bi-collection-fill"></i>
                      </Badge>
                      <span className="fw-medium">{subfeature.label}</span>
                      {subfeature.summary && (
                        <small className="d-block text-muted ms-4 mt-1">{subfeature.summary}</small>
                      )}
                    </div>
                    <Badge bg="secondary" pill>
                      {subfeature.member_count} members
                    </Badge>
                  </ListGroup.Item>
                ))}
              </ListGroup>
            </Card>
          )}

          {/* Members - shown for features and subfeatures */}
          {isFeature && element.feature_info && (
            <Card className="mb-4">
              <Card.Header>
                <i className="bi bi-people me-2"></i>
                Members ({element.feature_info.member_count})
                {element.feature_info.member_count > element.feature_info.members.length && (
                  <small className="text-muted ms-2">
                    showing first {element.feature_info.members.length}
                  </small>
                )}
              </Card.Header>
              {element.feature_info.parent_feature && (
                <Card.Body className="bg-body-secondary border-bottom py-2">
                  <small className="text-uppercase text-muted fw-bold d-block mb-1">Parent Feature</small>
                  <div>
                    <i className="bi bi-collection me-2 text-info"></i>
                    <strong>{element.feature_info.parent_feature.label}</strong>
                    {element.feature_info.parent_feature.summary && (
                      <small className="d-block text-muted ms-4">{element.feature_info.parent_feature.summary}</small>
                    )}
                  </div>
                </Card.Body>
              )}
              <Card.Body className="p-0 bg-body-secondary">
                {element.feature_info.members.length > 0 ? (
                  <Tabs defaultActiveKey="all" className="px-3 pt-2">
                    <Tab eventKey="all" title="All">
                      <ListGroup variant="flush">
                        {element.feature_info.members.map((member) => {
                          const memberConfig = getTypeConfig(member.element_type)
                          return (
                            <ListGroup.Item
                              key={member.element_id}
                              action
                              as={Link}
                              to={`/element/${encodeURIComponent(member.hash_id || member.element_id)}`}
                              className="d-flex justify-content-between align-items-start"
                            >
                              <div>
                                <Badge
                                  bg={memberConfig.color}
                                  style={getTypeBadgeStyle(member.element_type)}
                                  className="me-2"
                                >
                                  <i className={`bi ${memberConfig.icon}`}></i>
                                </Badge>
                                <span className="fw-medium">{member.name}</span>
                                {member.signature && (
                                  <code className="ms-2 small text-muted">
                                    {member.signature.length > 40 ? member.signature.substring(0, 40) + '...' : member.signature}
                                  </code>
                                )}
                                {member.summary && (
                                  <small className="d-block text-muted ms-4 mt-1">{member.summary}</small>
                                )}
                              </div>
                              <div className="text-end">
                                <small className="text-muted d-block">{member.file_path.split('/').pop()}</small>
                                <small className="text-muted">L{member.line}</small>
                              </div>
                            </ListGroup.Item>
                          )
                        })}
                      </ListGroup>
                    </Tab>
                    {['function', 'method', 'class'].map((type) => {
                      const items = element.feature_info!.members.filter(m => m.element_type === type)
                      if (items.length === 0) return null
                      const tc = getTypeConfig(type)
                      return (
                        <Tab key={type} eventKey={type} title={`${tc.label}s (${items.length})`}>
                          <ListGroup variant="flush">
                            {items.map((member) => (
                              <ListGroup.Item
                                key={member.element_id}
                                action
                                as={Link}
                                to={`/element/${encodeURIComponent(member.hash_id || member.element_id)}`}
                                className="d-flex justify-content-between align-items-start"
                              >
                                <div>
                                  <span className="fw-medium">{member.name}</span>
                                  {member.signature && (
                                    <code className="ms-2 small text-muted">{member.signature}</code>
                                  )}
                                  {member.summary && (
                                    <small className="d-block text-muted mt-1">{member.summary}</small>
                                  )}
                                </div>
                                <div className="text-end">
                                  <small className="text-muted d-block">{member.file_path.split('/').pop()}</small>
                                  <small className="text-muted">L{member.line}</small>
                                </div>
                              </ListGroup.Item>
                            ))}
                          </ListGroup>
                        </Tab>
                      )
                    })}
                  </Tabs>
                ) : (
                  <div className="text-center py-4 text-muted">
                    <i className="bi bi-inbox fs-3 d-block mb-2"></i>
                    No members found
                  </div>
                )}
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
                      to={`/element/${encodeURIComponent(sibling.hash_id || sibling.element_id)}`}
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
                    <Link to={`/element/${encodeURIComponent(element.context.file.hash_id || element.context.file.element_id)}`}>
                      <i className="bi bi-file-code me-1"></i>
                      {element.context.file.name}
                    </Link>
                    {element.context.file.summary && (
                      <small className="d-block text-muted mt-1">{element.context.file.summary}</small>
                    )}
                  </ListGroup.Item>
                )}
                {/* Only show parent if it's different from file (not a file type itself) */}
                {element.context.parent && element.context.parent.element_type !== 'file' && (
                  <ListGroup.Item>
                    <small className="text-uppercase text-muted d-block">Parent</small>
                    <Link to={`/element/${encodeURIComponent(element.context.parent.hash_id || element.context.parent.element_id)}`}>
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

          {/* Glossary Terms - shown for features and subfeatures */}
          {isFeature && glossaryTerms && glossaryTerms.terms.length > 0 && (
            <Card className="mb-4">
              <Card.Header>
                <i className="bi bi-book me-2"></i>
                Domain Terms
                <Badge bg="secondary" className="ms-2">{glossaryTerms.terms.length}</Badge>
              </Card.Header>
              <ListGroup variant="flush">
                {glossaryTerms.terms.map((term) => (
                  <ListGroup.Item
                    key={term.term}
                    action
                    as={Link}
                    to={term.hash_id ? `/element/${term.hash_id}` : `/glossary/${element.repository.scope}/${element.repository.name}?term=${encodeURIComponent(term.term)}`}
                    className="py-2"
                  >
                    <div className="d-flex justify-content-between align-items-start">
                      <span className="fw-medium">{term.term}</span>
                      {term.feature_count > 1 && (
                        <Badge bg="secondary" pill className="ms-2">
                          {term.feature_count}
                        </Badge>
                      )}
                    </div>
                    {term.description && (
                      <small className="text-muted d-block mt-1 text-truncate glossary-description">
                        <ReactMarkdown>{term.description.split('\n')[0]}</ReactMarkdown>
                      </small>
                    )}
                  </ListGroup.Item>
                ))}
              </ListGroup>
            </Card>
          )}

          {/* Callers - who calls this function */}
          {explanation && explanation.callers && explanation.callers.length > 0 && (
            <Card className="mb-4">
              <Card.Header className="d-flex justify-content-between align-items-center">
                <span>
                  <i className="bi bi-arrow-down-left me-2"></i>
                  Callers
                  <Badge bg="secondary" className="ms-2">{explanation.callers.length}</Badge>
                </span>
                <Link
                  to={`/repos/${element.repository.scope}/${element.repository.name}/call-explorer?element=${encodeURIComponent(decodedId)}`}
                  className="btn btn-sm btn-outline-primary"
                >
                  <i className="bi bi-diagram-3 me-1"></i>
                  Explorer
                </Link>
              </Card.Header>
              <ListGroup variant="flush">
                {explanation.callers.slice(0, 10).map((caller) => {
                  const callerConfig = getTypeConfig(caller.element_type)
                  return (
                    <ListGroup.Item
                      key={caller.element_id}
                      action
                      as={Link}
                      to={`/element/${encodeURIComponent(caller.hash_id || caller.element_id)}`}
                    >
                      <div className="d-flex justify-content-between align-items-start">
                        <div className="text-truncate me-2">
                          <Badge
                            bg={callerConfig.color}
                            style={getTypeBadgeStyle(caller.element_type)}
                            className="me-1"
                            pill
                          >
                            <i className={`bi ${callerConfig.icon}`}></i>
                          </Badge>
                          <span>{caller.name}</span>
                          <small className="text-muted ms-2">{caller.file_path}:{caller.line}</small>
                        </div>
                      </div>
                      {caller.summary && (
                        <small className="d-block text-muted text-truncate mt-1">{caller.summary}</small>
                      )}
                    </ListGroup.Item>
                  )
                })}
              </ListGroup>
            </Card>
          )}

          {/* Callees - what this function calls */}
          {explanation && explanation.callees && explanation.callees.length > 0 && (
            <Card className="mb-4">
              <Card.Header className="d-flex justify-content-between align-items-center">
                <span>
                  <i className="bi bi-arrow-up-right me-2"></i>
                  Calls
                  <Badge bg="secondary" className="ms-2">{explanation.callees.length}</Badge>
                </span>
                <Link
                  to={`/repos/${element.repository.scope}/${element.repository.name}/call-explorer?element=${encodeURIComponent(decodedId)}`}
                  className="btn btn-sm btn-outline-primary"
                >
                  <i className="bi bi-diagram-3 me-1"></i>
                  Explorer
                </Link>
              </Card.Header>
              <ListGroup variant="flush">
                {explanation.callees.slice(0, 10).map((callee, idx) => {
                  const calleeConfig = callee.element_type ? getTypeConfig(callee.element_type) : { icon: 'bi-question', color: 'secondary', label: 'Unknown' }
                  const isResolved = !!callee.element_id
                  const calleeContent = (
                    <>
                      <div className="d-flex justify-content-between align-items-start">
                        <div className="text-truncate me-2">
                          {isResolved ? (
                            <Badge
                              bg={calleeConfig.color}
                              style={getTypeBadgeStyle(callee.element_type || '')}
                              className="me-1"
                              pill
                            >
                              <i className={`bi ${calleeConfig.icon}`}></i>
                            </Badge>
                          ) : (
                            <Badge bg="secondary" className="me-1" pill>
                              <i className="bi bi-question"></i>
                            </Badge>
                          )}
                          <span>
                            {callee.receiver && <span className="text-muted">{callee.receiver}.</span>}
                            {callee.name}()
                          </span>
                          <small className="text-muted ms-2">line {callee.line}</small>
                        </div>
                        {!isResolved && (
                          <Badge bg="warning" text="dark" pill>unresolved</Badge>
                        )}
                      </div>
                      {callee.summary && (
                        <small className="d-block text-muted text-truncate mt-1">{callee.summary}</small>
                      )}
                    </>
                  )
                  return isResolved ? (
                    <ListGroup.Item
                      key={`${callee.name}-${idx}`}
                      action
                      as={Link}
                      to={`/element/${encodeURIComponent(callee.hash_id || callee.element_id || '')}`}
                    >
                      {calleeContent}
                    </ListGroup.Item>
                  ) : (
                    <ListGroup.Item key={`${callee.name}-${idx}`}>
                      {calleeContent}
                    </ListGroup.Item>
                  )
                })}
              </ListGroup>
            </Card>
          )}

          {/* Embedding Status */}
          {explanation && explanation.embedding_status && (
            <Card className="mb-4">
              <Card.Header>
                <i className="bi bi-cpu me-2"></i>
                Embedding Status
              </Card.Header>
              <Card.Body>
                <div className="d-flex gap-3">
                  <Badge bg={explanation.embedding_status.has_summary ? 'success' : 'secondary'}>
                    <i className={`bi ${explanation.embedding_status.has_summary ? 'bi-check' : 'bi-x'} me-1`}></i>
                    Summary Embedding
                  </Badge>
                  <Badge bg={explanation.embedding_status.has_code ? 'success' : 'secondary'}>
                    <i className={`bi ${explanation.embedding_status.has_code ? 'bi-check' : 'bi-x'} me-1`}></i>
                    Code Embedding
                  </Badge>
                </div>
              </Card.Body>
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
                      to={`/element/${encodeURIComponent(sim.hash_id || sim.element_id)}`}
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

          {/* Code Metrics Card - for functions/methods */}
          {isCallable && (element.complexity || element.security_issues?.length > 0 || element.env_vars?.length > 0 || element.concurrency) && (
            <Card className="mb-4">
              <Card.Header>
                <i className="bi bi-speedometer2 me-2"></i>
                Code Metrics
              </Card.Header>
              <ListGroup variant="flush">
                {/* Complexity */}
                {element.complexity && (
                  <>
                    <ListGroup.Item className="d-flex justify-content-between">
                      <span className="text-muted">Cyclomatic Complexity</span>
                      <Badge bg={element.complexity.cyclomatic > 10 ? 'danger' : element.complexity.cyclomatic > 5 ? 'warning' : 'success'}>
                        {element.complexity.cyclomatic}
                      </Badge>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between">
                      <span className="text-muted">Nesting Depth</span>
                      <Badge bg={element.complexity.nesting_depth > 4 ? 'warning' : 'secondary'}>
                        {element.complexity.nesting_depth}
                      </Badge>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between">
                      <span className="text-muted">Branch Count</span>
                      <span>{element.complexity.branch_count}</span>
                    </ListGroup.Item>
                  </>
                )}
                {/* Code Metrics */}
                {element.code_metrics && (
                  <ListGroup.Item className="d-flex justify-content-between">
                    <span className="text-muted">Lines of Code</span>
                    <span>{element.code_metrics.line_count}</span>
                  </ListGroup.Item>
                )}
                {/* Docstring Quality */}
                {element.docstring_quality && (
                  <ListGroup.Item className="d-flex justify-content-between">
                    <span className="text-muted">Documentation</span>
                    <Badge bg={element.docstring_quality.coverage >= 0.75 ? 'success' : element.docstring_quality.coverage >= 0.5 ? 'warning' : 'secondary'}>
                      {Math.round(element.docstring_quality.coverage * 100)}%
                    </Badge>
                  </ListGroup.Item>
                )}
                {/* Concurrency */}
                {element.concurrency && element.concurrency.patterns.length > 0 && (
                  <ListGroup.Item>
                    <div className="d-flex justify-content-between mb-1">
                      <span className="text-muted">Concurrency</span>
                    </div>
                    <div className="d-flex flex-wrap gap-1">
                      {element.concurrency.is_async && <Badge bg="info">async</Badge>}
                      {element.concurrency.uses_threads && <Badge bg="warning" text="dark">threads</Badge>}
                      {element.concurrency.uses_locks && <Badge bg="secondary">locks</Badge>}
                      {element.concurrency.patterns.filter(p => !['async'].includes(p)).map((p, i) => (
                        <Badge key={i} bg="light" text="dark">{p}</Badge>
                      ))}
                    </div>
                  </ListGroup.Item>
                )}
                {/* Environment Variables */}
                {element.env_vars && element.env_vars.length > 0 && (
                  <ListGroup.Item>
                    <div className="d-flex justify-content-between mb-1">
                      <span className="text-muted">Environment Variables</span>
                      <Badge bg="secondary">{element.env_vars.length}</Badge>
                    </div>
                    <div className="d-flex flex-wrap gap-1">
                      {element.env_vars.map((ev, i) => (
                        <code key={i} className="small bg-light px-1 rounded">{ev.name}</code>
                      ))}
                    </div>
                  </ListGroup.Item>
                )}
                {/* Security Issues */}
                {element.security_issues && element.security_issues.length > 0 && (
                  <ListGroup.Item>
                    <div className="d-flex justify-content-between mb-1">
                      <span className="text-muted">Security Issues</span>
                      <Badge bg="danger">{element.security_issues.length}</Badge>
                    </div>
                    <div className="mt-2">
                      {element.security_issues.map((issue, i) => (
                        <div key={i} className="d-flex align-items-start mb-2">
                          <Badge
                            bg={issue.severity === 'critical' ? 'danger' : issue.severity === 'high' ? 'warning' : issue.severity === 'medium' ? 'info' : 'secondary'}
                            className="me-2"
                            style={{ minWidth: '60px' }}
                          >
                            {issue.severity}
                          </Badge>
                          <div>
                            <small className="d-block">{issue.message || issue.kind}</small>
                            <small className="text-muted">Line {issue.line}</small>
                          </div>
                        </div>
                      ))}
                    </div>
                  </ListGroup.Item>
                )}
              </ListGroup>
            </Card>
          )}

          {/* Code Health Card - for files/classes */}
          {isContainer && element.metrics_summary && (
            <Card className="mb-4">
              <Card.Header>
                <i className="bi bi-heart-pulse me-2"></i>
                Code Health
              </Card.Header>
              <ListGroup variant="flush">
                <ListGroup.Item className="d-flex justify-content-between">
                  <span className="text-muted">Total Functions</span>
                  <span>{element.metrics_summary.total_functions}</span>
                </ListGroup.Item>
                <ListGroup.Item className="d-flex justify-content-between">
                  <span className="text-muted">Avg Complexity</span>
                  <Badge bg={element.metrics_summary.avg_complexity > 10 ? 'danger' : element.metrics_summary.avg_complexity > 5 ? 'warning' : 'success'}>
                    {element.metrics_summary.avg_complexity.toFixed(1)}
                  </Badge>
                </ListGroup.Item>
                <ListGroup.Item className="d-flex justify-content-between">
                  <span className="text-muted">Max Complexity</span>
                  <Badge bg={element.metrics_summary.max_complexity > 15 ? 'danger' : element.metrics_summary.max_complexity > 10 ? 'warning' : 'secondary'}>
                    {element.metrics_summary.max_complexity}
                  </Badge>
                </ListGroup.Item>
                <ListGroup.Item className="d-flex justify-content-between">
                  <span className="text-muted">Total Lines</span>
                  <span>{element.metrics_summary.total_lines}</span>
                </ListGroup.Item>
                <ListGroup.Item className="d-flex justify-content-between">
                  <span className="text-muted">Documentation</span>
                  <Badge bg={element.metrics_summary.documented_pct >= 75 ? 'success' : element.metrics_summary.documented_pct >= 50 ? 'warning' : 'secondary'}>
                    {element.metrics_summary.documented_pct.toFixed(0)}%
                  </Badge>
                </ListGroup.Item>
                {element.metrics_summary.async_count > 0 && (
                  <ListGroup.Item className="d-flex justify-content-between">
                    <span className="text-muted">Async Functions</span>
                    <Badge bg="info">{element.metrics_summary.async_count}</Badge>
                  </ListGroup.Item>
                )}
                {element.metrics_summary.security_issue_count > 0 && (
                  <ListGroup.Item>
                    <div className="d-flex justify-content-between">
                      <span className="text-muted">Security Issues</span>
                      <Badge bg="danger">{element.metrics_summary.security_issue_count}</Badge>
                    </div>
                    {Object.keys(element.metrics_summary.security_by_severity).length > 0 && (
                      <div className="d-flex flex-wrap gap-1 mt-1">
                        {Object.entries(element.metrics_summary.security_by_severity)
                          .sort(([a], [b]) => {
                            const order = ['critical', 'high', 'medium', 'low', 'info'];
                            return order.indexOf(a) - order.indexOf(b);
                          })
                          .map(([severity, count]) => (
                            <Badge
                              key={severity}
                              bg={severity === 'critical' ? 'danger' : severity === 'high' ? 'warning' : severity === 'medium' ? 'info' : 'secondary'}
                            >
                              {count} {severity}
                            </Badge>
                          ))
                        }
                      </div>
                    )}
                  </ListGroup.Item>
                )}
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
              {element.language && (
                <ListGroup.Item className="d-flex justify-content-between">
                  <span className="text-muted">Language</span>
                  <span>
                    <i className={`bi ${langIcon} me-1`}></i>
                    {element.language}
                  </span>
                </ListGroup.Item>
              )}
              {element.line_start > 0 && (
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
              )}
              {isFeature && element.feature_info && (
                <ListGroup.Item className="d-flex justify-content-between">
                  <span className="text-muted">Members</span>
                  <span>{element.feature_info.member_count}</span>
                </ListGroup.Item>
              )}
              {element.element_type === 'feature' && element.feature_info && element.feature_info.subfeatures && element.feature_info.subfeatures.length > 0 && (
                <ListGroup.Item className="d-flex justify-content-between">
                  <span className="text-muted">Subfeatures</span>
                  <span>{element.feature_info.subfeatures.length}</span>
                </ListGroup.Item>
              )}
              {isGlossary && element.glossary_info && (
                <>
                  <ListGroup.Item className="d-flex justify-content-between">
                    <span className="text-muted">Source Features</span>
                    <span>{element.glossary_info.feature_count}</span>
                  </ListGroup.Item>
                  {element.glossary_info.file_paths && element.glossary_info.file_paths.length > 0 && (
                    <ListGroup.Item className="d-flex justify-content-between">
                      <span className="text-muted">Files</span>
                      <span>{element.glossary_info.file_paths.length}</span>
                    </ListGroup.Item>
                  )}
                </>
              )}
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
                <ListGroup.Item>
                  <div className="d-flex justify-content-between mb-1">
                    <span className="text-muted">Decorators</span>
                    <small className="text-muted">{element.decorators.length}</small>
                  </div>
                  <div className="d-flex flex-wrap gap-1">
                    {element.decorator_details && element.decorator_details.length > 0
                      ? element.decorator_details.map((dd, i) => (
                          <code key={i} className="small bg-light px-1 rounded">@{dd.full || dd.name}</code>
                        ))
                      : element.decorators.map((d, i) => (
                          <code key={i} className="small bg-light px-1 rounded">@{d}</code>
                        ))
                    }
                  </div>
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
              {element.is_test && (
                <ListGroup.Item className="d-flex justify-content-between">
                  <span className="text-muted">Test Code</span>
                  <Badge bg="warning" text="dark">
                    <i className="bi bi-check"></i> Yes
                  </Badge>
                </ListGroup.Item>
              )}
              {entryPointType && (
                <ListGroup.Item>
                  <div className="d-flex justify-content-between align-items-center">
                    <span className="text-muted">Entry Point</span>
                    <Badge bg={entryPointType.color}>
                      <i className={`bi ${entryPointType.icon} me-1`}></i>
                      {entryPointType.label}
                    </Badge>
                  </div>
                  {entryPointType.args && (
                    <code className="d-block mt-1 small text-break">{entryPointType.args}</code>
                  )}
                </ListGroup.Item>
              )}
              {element.element_type === 'file' && element.element_count !== null && element.element_count !== undefined && (
                <ListGroup.Item className="d-flex justify-content-between">
                  <span className="text-muted">Total Elements</span>
                  <span>{element.element_count}</span>
                </ListGroup.Item>
              )}
              {element.element_type === 'class' && element.base_classes && element.base_classes.length > 0 && (
                <ListGroup.Item className="d-flex justify-content-between">
                  <span className="text-muted">Base Classes</span>
                  <span>{element.base_classes.length}</span>
                </ListGroup.Item>
              )}
              {element.element_type === 'class' && element.class_attributes && element.class_attributes.length > 0 && (
                <ListGroup.Item className="d-flex justify-content-between">
                  <span className="text-muted">Attributes</span>
                  <span>{element.class_attributes.length}</span>
                </ListGroup.Item>
              )}
              {isCallable && element.exceptions_raised && element.exceptions_raised.length > 0 && (
                <ListGroup.Item className="d-flex justify-content-between">
                  <span className="text-muted">Exceptions</span>
                  <span>{element.exceptions_raised.length}</span>
                </ListGroup.Item>
              )}
              {element.element_type === 'file' && element.imports && element.imports.length > 0 && (
                <ListGroup.Item className="d-flex justify-content-between">
                  <span className="text-muted">Imports</span>
                  <span>{element.imports.length}</span>
                </ListGroup.Item>
              )}
              {element.indexed_at && (
                <ListGroup.Item className="d-flex justify-content-between">
                  <span className="text-muted">Indexed</span>
                  <small>{new Date(element.indexed_at).toLocaleDateString()}</small>
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
