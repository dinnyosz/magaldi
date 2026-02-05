/**
 * Children/Contents section for container elements (files, classes)
 */

import { Link } from 'react-router-dom'
import { Card, Badge, ListGroup, Tab, Tabs } from 'react-bootstrap'
import {
  getTypeConfig,
  getTypeBadgeStyle,
} from '../../../components/element'
import type { ElementDetail } from '../../../api'
import { isContainer } from '../config'

interface Props {
  element: ElementDetail
}

export function ChildrenSection({ element }: Props) {
  const isContainerType = isContainer(element.element_type)

  if (
    !isContainerType ||
    !element.context.children ||
    element.context.children.length === 0
  ) {
    return null
  }

  return (
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
                          {child.signature.length > 40
                            ? child.signature.substring(0, 40) + '...'
                            : child.signature}
                        </code>
                      )}
                      {child.summary && (
                        <small className="d-block text-muted ms-4 mt-1">
                          {child.summary}
                        </small>
                      )}
                    </div>
                    <small className="text-muted">L{child.line}</small>
                  </ListGroup.Item>
                )
              })}
            </ListGroup>
          </Tab>
          {['class', 'function', 'method', 'variable', 'constant'].map(
            (type) => {
              const items = element.context.children.filter(
                (c) => c.element_type === type
              )
              if (items.length === 0) return null
              const tc = getTypeConfig(type)
              return (
                <Tab
                  key={type}
                  eventKey={type}
                  title={`${tc.label}s (${items.length})`}
                >
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
                            <code className="ms-2 small text-muted">
                              {child.signature}
                            </code>
                          )}
                          {child.summary && (
                            <small className="d-block text-muted mt-1">
                              {child.summary}
                            </small>
                          )}
                        </div>
                        <small className="text-muted">L{child.line}</small>
                      </ListGroup.Item>
                    ))}
                  </ListGroup>
                </Tab>
              )
            }
          )}
        </Tabs>
      </Card.Body>
    </Card>
  )
}
