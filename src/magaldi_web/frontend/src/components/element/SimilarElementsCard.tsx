/**
 * Card showing similar code elements
 */

import { Link } from 'react-router-dom'
import { Card, Badge, ListGroup } from 'react-bootstrap'
import { getTypeConfig, getTypeBadgeStyle } from './typeConfig'
import type { SimilarElement } from '../../api'

interface Props {
  similar: SimilarElement[]
}

export function SimilarElementsCard({ similar }: Props) {
  if (!similar || similar.length === 0) {
    return null
  }

  return (
    <Card className="mb-4">
      <Card.Header>
        <i className="bi bi-diagram-3 me-2"></i>
        Related Code
      </Card.Header>
      <ListGroup variant="flush">
        {similar.map((s) => {
          const simConfig = getTypeConfig(s.element_type)
          return (
            <ListGroup.Item
              key={s.element_id}
              action
              as={Link}
              to={`/element/${encodeURIComponent(s.hash_id || s.element_id)}`}
              className="d-flex justify-content-between align-items-start"
            >
              <div className="text-truncate me-2">
                <Badge
                  bg={simConfig.color}
                  style={getTypeBadgeStyle(s.element_type)}
                  className="me-1"
                  pill
                >
                  <i className={`bi ${simConfig.icon}`}></i>
                </Badge>
                <span>{s.name}</span>
                {s.summary && (
                  <small className="d-block text-muted text-truncate">{s.summary}</small>
                )}
              </div>
              <Badge bg="secondary" pill>
                {(s.similarity * 100).toFixed(0)}%
              </Badge>
            </ListGroup.Item>
          )
        })}
      </ListGroup>
    </Card>
  )
}
