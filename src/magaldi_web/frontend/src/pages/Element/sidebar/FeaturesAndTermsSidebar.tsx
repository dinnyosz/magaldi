/**
 * Sidebar components for glossary terms and connected features
 */

import { Link } from 'react-router-dom'
import { Card, Badge, ListGroup } from 'react-bootstrap'
import ReactMarkdown from 'react-markdown'
import type { ElementDetail, ElementFeaturesResponse } from '../../../api'
import { isFeature } from '../config'

interface GlossaryTerm {
  term: string
  hash_id: string | null
  description: string | null
  feature_count: number
}

interface GlossaryTermsForFeatureResponse {
  terms: GlossaryTerm[]
}

interface GlossaryTermsProps {
  element: ElementDetail
  glossaryTerms: GlossaryTermsForFeatureResponse | undefined
}

export function GlossaryTermsSidebar({
  element,
  glossaryTerms,
}: GlossaryTermsProps) {
  const isFeatureType = isFeature(element.element_type)

  if (!isFeatureType || !glossaryTerms || glossaryTerms.terms.length === 0) {
    return null
  }

  return (
    <Card className="mb-4">
      <Card.Header>
        <i className="bi bi-book me-2"></i>
        Domain Terms
        <Badge bg="secondary" className="ms-2">
          {glossaryTerms.terms.length}
        </Badge>
      </Card.Header>
      <ListGroup variant="flush">
        {glossaryTerms.terms.map((term) => (
          <ListGroup.Item
            key={term.term}
            action
            as={Link}
            to={
              term.hash_id
                ? `/element/${term.hash_id}`
                : `/glossary/${element.repository.scope}/${element.repository.name}?term=${encodeURIComponent(term.term)}`
            }
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
  )
}

interface ElementFeaturesProps {
  elementFeatures: ElementFeaturesResponse | undefined
}

export function ElementFeaturesSidebar({
  elementFeatures,
}: ElementFeaturesProps) {
  if (!elementFeatures || elementFeatures.features.length === 0) {
    return null
  }

  return (
    <Card className="mb-4">
      <Card.Header>
        <i className="bi bi-collection me-2"></i>
        Part of Features
        <Badge bg="secondary" className="ms-2">
          {elementFeatures.features.length}
        </Badge>
      </Card.Header>
      <ListGroup variant="flush">
        {elementFeatures.features.map((feature) => {
          const isSubfeature = feature.element_type === 'subfeature'
          return (
            <ListGroup.Item
              key={feature.feature_id}
              action
              as={Link}
              to={`/element/${encodeURIComponent(feature.hash_id || feature.feature_id)}`}
            >
              <div className="d-flex justify-content-between align-items-start">
                <div className="text-truncate me-2">
                  <Badge bg="info" className="me-1" pill>
                    <i
                      className={`bi ${isSubfeature ? 'bi-collection-fill' : 'bi-collection'}`}
                    ></i>
                  </Badge>
                  <span className="fw-medium">{feature.label}</span>
                  {isSubfeature && feature.parent_feature_label && (
                    <small className="text-muted ms-2">
                      (in {feature.parent_feature_label})
                    </small>
                  )}
                </div>
                <Badge bg="secondary" pill>
                  {feature.member_count}
                </Badge>
              </div>
              {feature.summary && (
                <small className="d-block text-muted text-truncate mt-1">
                  {feature.summary}
                </small>
              )}
            </ListGroup.Item>
          )
        })}
      </ListGroup>
    </Card>
  )
}
