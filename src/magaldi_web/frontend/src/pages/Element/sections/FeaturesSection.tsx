/**
 * Features section showing subfeatures, connected features, and members
 */

import { Link } from 'react-router-dom'
import { Card, Badge, ListGroup, Tab, Tabs } from 'react-bootstrap'
import {
  getTypeConfig,
  getTypeBadgeStyle,
} from '../../../components/element'
import type { ElementDetail } from '../../../api'
import { isFeature } from '../config'

interface Props {
  element: ElementDetail
}

export function SubfeaturesSection({ element }: Props) {
  if (
    element.element_type !== 'feature' ||
    !element.feature_info ||
    !element.feature_info.subfeatures ||
    element.feature_info.subfeatures.length === 0
  ) {
    return null
  }

  return (
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
                <small className="d-block text-muted ms-4 mt-1">
                  {subfeature.summary}
                </small>
              )}
            </div>
            <Badge bg="secondary" pill>
              {subfeature.member_count} members
            </Badge>
          </ListGroup.Item>
        ))}
      </ListGroup>
    </Card>
  )
}

export function ConnectedFeaturesSection({ element }: Props) {
  if (
    element.element_type !== 'feature' ||
    !element.feature_info ||
    !element.feature_info.connected_features ||
    element.feature_info.connected_features.length === 0
  ) {
    return null
  }

  return (
    <Card className="mb-4">
      <Card.Header>
        <i className="bi bi-diagram-3 me-2"></i>
        Related Features ({element.feature_info.connected_features.length})
      </Card.Header>
      <ListGroup variant="flush">
        {element.feature_info.connected_features.map((connected) => (
          <ListGroup.Item
            key={connected.feature_id}
            action
            as={Link}
            to={`/element/${encodeURIComponent(connected.hash_id || connected.feature_id)}`}
          >
            <div className="d-flex justify-content-between align-items-start">
              <div>
                <Badge bg="info" className="me-2">
                  <i className="bi bi-collection"></i>
                </Badge>
                <span className="fw-medium">{connected.label}</span>
              </div>
              {connected.shared_member_count > 0 && (
                <Badge bg="secondary" pill>
                  {connected.shared_member_count} shared
                </Badge>
              )}
            </div>
            {connected.common_glossary_terms.length > 0 && (
              <small className="d-block text-muted mt-1 ms-4">
                <i className="bi bi-book me-1"></i>
                {connected.common_glossary_terms.join(', ')}
              </small>
            )}
          </ListGroup.Item>
        ))}
      </ListGroup>
    </Card>
  )
}

export function MembersSection({ element }: Props) {
  const isFeatureType = isFeature(element.element_type)

  if (!isFeatureType || !element.feature_info) {
    return null
  }

  return (
    <Card className="mb-4">
      <Card.Header>
        <i className="bi bi-people me-2"></i>
        Members ({element.feature_info.member_count})
        {element.feature_info.member_count >
          element.feature_info.members.length && (
          <small className="text-muted ms-2">
            showing first {element.feature_info.members.length}
          </small>
        )}
      </Card.Header>
      {element.feature_info.parent_feature && (
        <Card.Body className="bg-body-secondary border-bottom py-2">
          <small className="text-uppercase text-muted fw-bold d-block mb-1">
            Parent Feature
          </small>
          <div>
            <i className="bi bi-collection me-2 text-info"></i>
            <strong>{element.feature_info.parent_feature.label}</strong>
            {element.feature_info.parent_feature.summary && (
              <small className="d-block text-muted ms-4">
                {element.feature_info.parent_feature.summary}
              </small>
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
                            {member.signature.length > 40
                              ? member.signature.substring(0, 40) + '...'
                              : member.signature}
                          </code>
                        )}
                        {member.summary && (
                          <small className="d-block text-muted ms-4 mt-1">
                            {member.summary}
                          </small>
                        )}
                      </div>
                      <div className="text-end">
                        <small className="text-muted d-block">
                          {member.file_path.split('/').pop()}
                        </small>
                        <small className="text-muted">L{member.line}</small>
                      </div>
                    </ListGroup.Item>
                  )
                })}
              </ListGroup>
            </Tab>
            {['function', 'method', 'class'].map((type) => {
              const items = element.feature_info!.members.filter(
                (m) => m.element_type === type
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
                            <code className="ms-2 small text-muted">
                              {member.signature}
                            </code>
                          )}
                          {member.summary && (
                            <small className="d-block text-muted mt-1">
                              {member.summary}
                            </small>
                          )}
                        </div>
                        <div className="text-end">
                          <small className="text-muted d-block">
                            {member.file_path.split('/').pop()}
                          </small>
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
  )
}
