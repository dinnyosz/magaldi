/**
 * Call graph sidebar showing callers and callees
 */

import { Link } from 'react-router-dom'
import { Card, Badge, ListGroup } from 'react-bootstrap'
import {
  getTypeConfig,
  getTypeBadgeStyle,
} from '../../../components/element'
import type { ExplainElementResponse, ElementDetail } from '../../../api'

interface Props {
  element: ElementDetail
  explanation: ExplainElementResponse | undefined
  decodedId: string
}

export function CallersSidebar({ element, explanation, decodedId }: Props) {
  if (!explanation || !explanation.callers || explanation.callers.length === 0) {
    return null
  }

  return (
    <Card className="mb-4">
      <Card.Header className="d-flex justify-content-between align-items-center">
        <span>
          <i className="bi bi-arrow-down-left me-2"></i>
          Callers
          <Badge bg="secondary" className="ms-2">
            {explanation.callers.length}
          </Badge>
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
                  <small className="text-muted ms-2">
                    {caller.file_path}:{caller.line}
                  </small>
                </div>
              </div>
              {caller.summary && (
                <small className="d-block text-muted text-truncate mt-1">
                  {caller.summary}
                </small>
              )}
            </ListGroup.Item>
          )
        })}
      </ListGroup>
    </Card>
  )
}

export function CalleesSidebar({ element, explanation, decodedId }: Props) {
  if (!explanation || !explanation.callees || explanation.callees.length === 0) {
    return null
  }

  return (
    <Card className="mb-4">
      <Card.Header className="d-flex justify-content-between align-items-center">
        <span>
          <i className="bi bi-arrow-up-right me-2"></i>
          Calls
          <Badge bg="secondary" className="ms-2">
            {explanation.callees.length}
          </Badge>
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
          const calleeConfig = callee.element_type
            ? getTypeConfig(callee.element_type)
            : { icon: 'bi-question', color: 'secondary', label: 'Unknown' }
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
                    {callee.receiver && (
                      <span className="text-muted">{callee.receiver}.</span>
                    )}
                    {callee.name}()
                  </span>
                  <small className="text-muted ms-2">line {callee.line}</small>
                </div>
                {!isResolved && (
                  <Badge bg="warning" text="dark" pill>
                    unresolved
                  </Badge>
                )}
              </div>
              {callee.summary && (
                <small className="d-block text-muted text-truncate mt-1">
                  {callee.summary}
                </small>
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
  )
}

export function EmbeddingStatusSidebar({
  explanation,
}: {
  explanation: ExplainElementResponse | undefined
}) {
  if (!explanation || !explanation.embedding_status) {
    return null
  }

  return (
    <Card className="mb-4">
      <Card.Header>
        <i className="bi bi-cpu me-2"></i>
        Embedding Status
      </Card.Header>
      <Card.Body>
        <div className="d-flex gap-3">
          <Badge
            bg={
              explanation.embedding_status.has_summary ? 'success' : 'secondary'
            }
          >
            <i
              className={`bi ${explanation.embedding_status.has_summary ? 'bi-check' : 'bi-x'} me-1`}
            ></i>
            Summary Embedding
          </Badge>
          <Badge
            bg={explanation.embedding_status.has_code ? 'success' : 'secondary'}
          >
            <i
              className={`bi ${explanation.embedding_status.has_code ? 'bi-check' : 'bi-x'} me-1`}
            ></i>
            Code Embedding
          </Badge>
        </div>
      </Card.Body>
    </Card>
  )
}
