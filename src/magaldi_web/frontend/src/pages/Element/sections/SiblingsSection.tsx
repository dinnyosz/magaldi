/**
 * Siblings section showing sibling elements
 */

import { Link } from 'react-router-dom'
import { Card, Badge, ListGroup } from 'react-bootstrap'
import {
  getTypeConfig,
  getTypeBadgeStyle,
} from '../../../components/element'
import type { ElementDetail } from '../../../api'

interface Props {
  element: ElementDetail
}

export function SiblingsSection({ element }: Props) {
  if (
    element.element_type === 'file' ||
    !element.context.siblings ||
    element.context.siblings.length === 0
  ) {
    return null
  }

  return (
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
                  <small className="d-block text-muted ms-4">
                    {sibling.summary}
                  </small>
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
  )
}
