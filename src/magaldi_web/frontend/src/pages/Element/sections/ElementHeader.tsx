/**
 * Element header section with decorators, name, badges, and file path
 */

import { Card, Badge } from 'react-bootstrap'
import {
  getTypeConfig,
  getTypeBadgeStyle,
  languageIcons,
} from '../../../components/element'
import type { ElementDetail } from '../../../api'
import type { DetectedEntryPoint } from '../config'
import { isFeature, isGlossary } from '../config'

interface Props {
  element: ElementDetail
  entryPointType: DetectedEntryPoint | null
}

export function ElementHeader({ element, entryPointType }: Props) {
  const config = getTypeConfig(element.element_type)
  const langIcon = languageIcons[element.language] || 'bi-file-code'
  const isFeatureType = isFeature(element.element_type)
  const isGlossaryType = isGlossary(element.element_type)

  return (
    <Card.Header>
      {/* Decorators */}
      {element.decorators && element.decorators.length > 0 && (
        <div className="mb-2">
          {element.decorator_details && element.decorator_details.length > 0
            ? element.decorator_details.map((dd, i) => (
                <code key={i} className="me-2 text-info">
                  @{dd.full || dd.name}
                </code>
              ))
            : element.decorators.map((d, i) => (
                <code key={i} className="me-2 text-info">
                  @{d}
                </code>
              ))}
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
          {element.is_public_api && (
            <Badge bg="success" className="ms-2" pill>
              <i className="bi bi-globe2 me-1"></i>
              public API
            </Badge>
          )}
          {entryPointType && (
            <Badge bg={entryPointType.color} className="ms-2" pill>
              <i className={`bi ${entryPointType.icon} me-1`}></i>
              {entryPointType.label}
              {entryPointType.args && (
                <code className="ms-1 fw-normal" style={{ opacity: 0.9 }}>
                  {entryPointType.args}
                </code>
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
              {element.line_end &&
                element.line_end !== element.line_start &&
                `-${element.line_end}`}
            </small>
          )}
          {isFeatureType && element.feature_info && (
            <Badge bg="secondary" className="ms-2">
              {element.feature_info.member_count} members
            </Badge>
          )}
          {isGlossaryType &&
            element.glossary_info &&
            element.glossary_info.feature_count > 0 && (
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
  )
}
