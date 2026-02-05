/**
 * Properties sidebar showing element info
 */

import { Card, Badge, ListGroup } from 'react-bootstrap'
import {
  getTypeConfig,
  getTypeBadgeStyle,
  languageIcons,
} from '../../../components/element'
import type { ElementDetail } from '../../../api'
import type { DetectedEntryPoint } from '../config'
import { isCallable, isContainer, isFeature, isGlossary } from '../config'

interface Props {
  element: ElementDetail
  entryPointType: DetectedEntryPoint | null
}

export function PropertiesSidebar({ element, entryPointType }: Props) {
  const config = getTypeConfig(element.element_type)
  const langIcon = languageIcons[element.language] || 'bi-file-code'
  const isCallableType = isCallable(element.element_type)
  const isContainerType = isContainer(element.element_type)
  const isFeatureType = isFeature(element.element_type)
  const isGlossaryType = isGlossary(element.element_type)

  return (
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
              {element.line_end &&
                element.line_end !== element.line_start &&
                `-${element.line_end}`}
              {element.line_end && (
                <small className="text-muted ms-1">
                  ({element.line_end - element.line_start + 1} lines)
                </small>
              )}
            </span>
          </ListGroup.Item>
        )}
        {isFeatureType && element.feature_info && (
          <ListGroup.Item className="d-flex justify-content-between">
            <span className="text-muted">Members</span>
            <span>{element.feature_info.member_count}</span>
          </ListGroup.Item>
        )}
        {element.element_type === 'feature' &&
          element.feature_info &&
          element.feature_info.subfeatures &&
          element.feature_info.subfeatures.length > 0 && (
            <ListGroup.Item className="d-flex justify-content-between">
              <span className="text-muted">Subfeatures</span>
              <span>{element.feature_info.subfeatures.length}</span>
            </ListGroup.Item>
          )}
        {isGlossaryType && element.glossary_info && (
          <>
            <ListGroup.Item className="d-flex justify-content-between">
              <span className="text-muted">Source Features</span>
              <span>{element.glossary_info.feature_count}</span>
            </ListGroup.Item>
            {element.glossary_info.file_paths &&
              element.glossary_info.file_paths.length > 0 && (
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
            <Badge
              bg={element.visibility === 'public' ? 'success' : 'secondary'}
            >
              {element.visibility}
            </Badge>
          </ListGroup.Item>
        )}
        {isCallableType && (
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
                    <code key={i} className="small bg-light px-1 rounded">
                      @{dd.full || dd.name}
                    </code>
                  ))
                : element.decorators.map((d, i) => (
                    <code key={i} className="small bg-light px-1 rounded">
                      @{d}
                    </code>
                  ))}
            </div>
          </ListGroup.Item>
        )}
        {isContainerType && (
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
              <code className="d-block mt-1 small text-break">
                {entryPointType.args}
              </code>
            )}
          </ListGroup.Item>
        )}
        {element.element_type === 'file' &&
          element.element_count !== null &&
          element.element_count !== undefined && (
            <ListGroup.Item className="d-flex justify-content-between">
              <span className="text-muted">Total Elements</span>
              <span>{element.element_count}</span>
            </ListGroup.Item>
          )}
        {element.element_type === 'class' &&
          element.base_classes &&
          element.base_classes.length > 0 && (
            <ListGroup.Item className="d-flex justify-content-between">
              <span className="text-muted">Base Classes</span>
              <span>{element.base_classes.length}</span>
            </ListGroup.Item>
          )}
        {element.element_type === 'class' &&
          element.class_attributes &&
          element.class_attributes.length > 0 && (
            <ListGroup.Item className="d-flex justify-content-between">
              <span className="text-muted">Attributes</span>
              <span>{element.class_attributes.length}</span>
            </ListGroup.Item>
          )}
        {isCallableType &&
          element.exceptions_raised &&
          element.exceptions_raised.length > 0 && (
            <ListGroup.Item className="d-flex justify-content-between">
              <span className="text-muted">Exceptions</span>
              <span>{element.exceptions_raised.length}</span>
            </ListGroup.Item>
          )}
        {element.element_type === 'file' &&
          element.imports &&
          element.imports.length > 0 && (
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
  )
}
