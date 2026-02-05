/**
 * Sidebar components for code metrics and code health
 */

import { Card, Badge, ListGroup } from 'react-bootstrap'
import type { ElementDetail } from '../../../api'
import { isCallable, isContainer } from '../config'

interface Props {
  element: ElementDetail
}

export function CodeMetricsSidebar({ element }: Props) {
  const isCallableType = isCallable(element.element_type)

  if (
    !isCallableType ||
    (!element.complexity &&
      !(element.security_issues && element.security_issues.length > 0) &&
      !(element.env_vars && element.env_vars.length > 0) &&
      !element.concurrency)
  ) {
    return null
  }

  return (
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
              <Badge
                bg={
                  element.complexity.cyclomatic > 10
                    ? 'danger'
                    : element.complexity.cyclomatic > 5
                      ? 'warning'
                      : 'success'
                }
              >
                {element.complexity.cyclomatic}
              </Badge>
            </ListGroup.Item>
            <ListGroup.Item className="d-flex justify-content-between">
              <span className="text-muted">Nesting Depth</span>
              <Badge
                bg={
                  element.complexity.nesting_depth > 4 ? 'warning' : 'secondary'
                }
              >
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
            <Badge
              bg={
                element.docstring_quality.coverage >= 0.75
                  ? 'success'
                  : element.docstring_quality.coverage >= 0.5
                    ? 'warning'
                    : 'secondary'
              }
            >
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
              {element.concurrency.uses_threads && (
                <Badge bg="warning" text="dark">
                  threads
                </Badge>
              )}
              {element.concurrency.uses_locks && (
                <Badge bg="secondary">locks</Badge>
              )}
              {element.concurrency.patterns
                .filter((p) => !['async'].includes(p))
                .map((p, i) => (
                  <Badge key={i} bg="light" text="dark">
                    {p}
                  </Badge>
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
                <code key={i} className="small bg-light px-1 rounded">
                  {ev.name}
                </code>
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
                    bg={
                      issue.severity === 'critical'
                        ? 'danger'
                        : issue.severity === 'high'
                          ? 'warning'
                          : issue.severity === 'medium'
                            ? 'info'
                            : 'secondary'
                    }
                    className="me-2"
                    style={{ minWidth: '60px' }}
                  >
                    {issue.severity}
                  </Badge>
                  <div>
                    <small className="d-block">
                      {issue.message || issue.kind}
                    </small>
                    <small className="text-muted">Line {issue.line}</small>
                  </div>
                </div>
              ))}
            </div>
          </ListGroup.Item>
        )}
      </ListGroup>
    </Card>
  )
}

export function CodeHealthSidebar({ element }: Props) {
  const isContainerType = isContainer(element.element_type)

  if (!isContainerType || !element.metrics_summary) {
    return null
  }

  return (
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
          <Badge
            bg={
              element.metrics_summary.avg_complexity > 10
                ? 'danger'
                : element.metrics_summary.avg_complexity > 5
                  ? 'warning'
                  : 'success'
            }
          >
            {element.metrics_summary.avg_complexity.toFixed(1)}
          </Badge>
        </ListGroup.Item>
        <ListGroup.Item className="d-flex justify-content-between">
          <span className="text-muted">Max Complexity</span>
          <Badge
            bg={
              element.metrics_summary.max_complexity > 15
                ? 'danger'
                : element.metrics_summary.max_complexity > 10
                  ? 'warning'
                  : 'secondary'
            }
          >
            {element.metrics_summary.max_complexity}
          </Badge>
        </ListGroup.Item>
        <ListGroup.Item className="d-flex justify-content-between">
          <span className="text-muted">Total Lines</span>
          <span>{element.metrics_summary.total_lines}</span>
        </ListGroup.Item>
        <ListGroup.Item className="d-flex justify-content-between">
          <span className="text-muted">Documentation</span>
          <Badge
            bg={
              element.metrics_summary.documented_pct >= 75
                ? 'success'
                : element.metrics_summary.documented_pct >= 50
                  ? 'warning'
                  : 'secondary'
            }
          >
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
              <Badge bg="danger">
                {element.metrics_summary.security_issue_count}
              </Badge>
            </div>
            {Object.keys(element.metrics_summary.security_by_severity).length >
              0 && (
              <div className="d-flex flex-wrap gap-1 mt-1">
                {Object.entries(element.metrics_summary.security_by_severity)
                  .sort(([a], [b]) => {
                    const order = ['critical', 'high', 'medium', 'low', 'info']
                    return order.indexOf(a) - order.indexOf(b)
                  })
                  .map(([severity, count]) => (
                    <Badge
                      key={severity}
                      bg={
                        severity === 'critical'
                          ? 'danger'
                          : severity === 'high'
                            ? 'warning'
                            : severity === 'medium'
                              ? 'info'
                              : 'secondary'
                      }
                    >
                      {count} {severity}
                    </Badge>
                  ))}
              </div>
            )}
          </ListGroup.Item>
        )}
      </ListGroup>
    </Card>
  )
}
