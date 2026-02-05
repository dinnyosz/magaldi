/**
 * Card showing element context (file, parent)
 */

import { Link } from 'react-router-dom'
import { Card, Badge, ListGroup } from 'react-bootstrap'
import { getTypeConfig, getTypeBadgeStyle } from './typeConfig'
import type { ElementDetailFile, ElementDetailParent } from '../../api'

interface Props {
  file: ElementDetailFile | null
  parent: ElementDetailParent | null
  elementType: string
}

export function ContextCard({ file, parent, elementType }: Props) {
  const hasFile = file && elementType !== 'file'
  const hasParent = parent && parent.element_type !== 'file'

  if (!hasFile && !hasParent) {
    return null
  }

  return (
    <Card className="mb-4">
      <Card.Header>
        <i className="bi bi-diagram-2 me-2"></i>
        Context
      </Card.Header>
      <ListGroup variant="flush">
        {hasFile && (
          <ListGroup.Item>
            <small className="text-uppercase text-muted d-block">File</small>
            <Link to={`/element/${encodeURIComponent(file!.hash_id || file!.element_id)}`}>
              <i className="bi bi-file-code me-1"></i>
              {file!.name}
            </Link>
            {file!.summary && (
              <small className="d-block text-muted mt-1">{file!.summary}</small>
            )}
          </ListGroup.Item>
        )}
        {hasParent && (
          <ListGroup.Item>
            <small className="text-uppercase text-muted d-block">Parent</small>
            <Link to={`/element/${encodeURIComponent(parent!.hash_id || parent!.element_id)}`}>
              <Badge
                bg={getTypeConfig(parent!.element_type).color}
                style={getTypeBadgeStyle(parent!.element_type)}
                className="me-1"
                pill
              >
                <i className={`bi ${getTypeConfig(parent!.element_type).icon}`}></i>
              </Badge>
              {parent!.name}
            </Link>
            {parent!.summary && (
              <small className="d-block text-muted mt-1">{parent!.summary}</small>
            )}
          </ListGroup.Item>
        )}
      </ListGroup>
    </Card>
  )
}
