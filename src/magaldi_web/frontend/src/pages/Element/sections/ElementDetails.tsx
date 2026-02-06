/**
 * Element details section with signature, parameters, return type,
 * metadata, code, and documentation
 */

import { Link } from 'react-router-dom'
import { Card, Badge, Alert, ListGroup } from 'react-bootstrap'
import ReactMarkdown from 'react-markdown'
import type { ElementDetail } from '../../../api'
import { isCallable, isGlossary } from '../config'

interface Props {
  element: ElementDetail
}

export function ElementDetails({ element }: Props) {
  const isCallableType = isCallable(element.element_type)
  const isGlossaryType = isGlossary(element.element_type)

  return (
    <Card.Body>
      {/* Signature for functions/methods */}
      {isCallableType && element.signature && (
        <div className="mb-3">
          <small className="text-uppercase text-muted fw-bold d-block mb-1">
            Signature
          </small>
          <code className="d-block bg-light p-2 rounded">
            {element.signature}
          </code>
        </div>
      )}

      {/* Parameters for functions/methods */}
      {isCallableType && element.parameters && element.parameters.length > 0 && (
        <div className="mb-3">
          <small className="text-uppercase text-muted fw-bold d-block mb-1">
            <i className="bi bi-input-cursor me-1"></i>
            Parameters
          </small>
          <div className="bg-light p-2 rounded">
            {element.parameters.map((param, i) => (
              <div key={i} className="d-flex align-items-center py-1">
                <code className="me-2">{param.name}</code>
                {param.type && (
                  <small className="text-muted">: {param.type}</small>
                )}
                {param.default && (
                  <small className="text-muted ms-2">= {param.default}</small>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Return Type for functions/methods */}
      {isCallableType && element.return_type && (
        <div className="mb-3">
          <small className="text-uppercase text-muted fw-bold d-block mb-1">
            <i className="bi bi-arrow-return-right me-1"></i>
            Returns
          </small>
          <code className="bg-light p-2 rounded d-inline-block">
            {element.return_type}
          </code>
        </div>
      )}

      {/* Type Annotations */}
      {element.type_annotations && element.type_annotations.length > 0 && (
        <div className="mb-3">
          <small className="text-uppercase text-muted fw-bold d-block mb-1">
            <i className="bi bi-braces-asterisk me-1"></i>
            Type Annotations ({element.type_annotations.length})
          </small>
          <div
            className="bg-light p-2 rounded"
            style={{ maxHeight: '150px', overflowY: 'auto' }}
          >
            {element.type_annotations.map((ann, i) => (
              <div key={i} className="d-flex align-items-center py-1">
                <code className="me-2">{ann.name}</code>
                <Badge bg="light" text="dark" className="me-2">
                  {ann.kind}
                </Badge>
                {ann.location && (
                  <small className="text-muted">{ann.location}</small>
                )}
                {ann.line && (
                  <small className="text-muted ms-auto">L{ann.line}</small>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* HTTP Routes */}
      {element.http_routes && element.http_routes.length > 0 && (
        <div className="mb-3">
          <small className="text-uppercase text-muted fw-bold d-block mb-1">
            <i className="bi bi-globe me-1"></i>
            HTTP Routes
          </small>
          <div className="bg-light p-2 rounded">
            {element.http_routes.map((route, i) => (
              <div key={i} className="d-flex align-items-center py-1">
                <Badge
                  bg={
                    route.method === 'GET'
                      ? 'success'
                      : route.method === 'POST'
                        ? 'primary'
                        : route.method === 'DELETE'
                          ? 'danger'
                          : 'secondary'
                  }
                  className="me-2"
                >
                  {route.method}
                </Badge>
                <code className="me-2">{route.path}</code>
                <small className="text-muted">({route.framework})</small>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* CLI Commands */}
      {element.cli_commands && element.cli_commands.length > 0 && (
        <div className="mb-3">
          <small className="text-uppercase text-muted fw-bold d-block mb-1">
            <i className="bi bi-terminal me-1"></i>
            CLI Commands
          </small>
          <div className="bg-light p-2 rounded">
            {element.cli_commands.map((cmd, i) => (
              <div key={i} className="d-flex align-items-center py-1">
                <code className="me-2">{cmd.name}</code>
                <small className="text-muted">({cmd.framework})</small>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Detected Patterns */}
      {element.detected_patterns && element.detected_patterns.length > 0 && (
        <div className="mb-3">
          <small className="text-uppercase text-muted fw-bold d-block mb-1">
            <i className="bi bi-puzzle me-1"></i>
            Design Patterns
          </small>
          <div className="d-flex flex-wrap gap-2">
            {element.detected_patterns.map((pattern, i) => (
              <Badge key={i} bg="info" className="fw-normal">
                {pattern}
                {element.pattern_confidence?.[pattern] && (
                  <small className="ms-1 opacity-75">
                    ({Math.round(element.pattern_confidence[pattern] * 100)}%)
                  </small>
                )}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {/* Purity */}
      {element.purity && (
        <div className="mb-3">
          <small className="text-uppercase text-muted fw-bold d-block mb-1">
            <i className="bi bi-shield-check me-1"></i>
            Purity
          </small>
          <Badge
            bg={
              element.purity.level === 'pure'
                ? 'success'
                : element.purity.level === 'read_only'
                  ? 'info'
                  : 'warning'
            }
            className="me-2"
          >
            {element.purity.level}
          </Badge>
          <small className="text-muted">
            ({element.purity.confidence} confidence)
          </small>
          {element.purity.reasons.length > 0 && (
            <div className="mt-1">
              {element.purity.reasons.map((reason, i) => (
                <small key={i} className="d-block text-muted">
                  {reason}
                </small>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Side Effects */}
      {element.side_effects && element.side_effects.length > 0 && (
        <div className="mb-3">
          <small className="text-uppercase text-muted fw-bold d-block mb-1">
            <i className="bi bi-exclamation-diamond me-1"></i>
            Side Effects
          </small>
          <div className="bg-light p-2 rounded">
            {element.side_effects.map((effect, i) => (
              <div key={i} className="d-flex align-items-center py-1">
                <Badge bg="warning" text="dark" className="me-2">
                  {effect.kind}
                </Badge>
                {effect.target && <code className="me-2">{effect.target}</code>}
                <small className="text-muted ms-auto">L{effect.line}</small>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Mutated State */}
      {element.mutated_state && element.mutated_state.length > 0 && (
        <div className="mb-3">
          <small className="text-uppercase text-muted fw-bold d-block mb-1">
            <i className="bi bi-pencil-square me-1"></i>
            Mutated State
          </small>
          <div className="d-flex flex-wrap gap-1">
            {element.mutated_state.map((state, i) => (
              <Badge key={i} bg="secondary" className="fw-normal">
                {state}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {/* TODOs */}
      {element.todos && element.todos.length > 0 && (
        <div className="mb-3">
          <small className="text-uppercase text-muted fw-bold d-block mb-1">
            <i className="bi bi-check2-square me-1"></i>
            TODOs ({element.todos.length})
          </small>
          <div
            className="bg-light p-2 rounded"
            style={{ maxHeight: '200px', overflowY: 'auto' }}
          >
            {element.todos.map((todo, i) => (
              <div key={i} className="py-1 border-bottom">
                <Badge
                  bg={
                    todo.kind === 'FIXME'
                      ? 'danger'
                      : todo.kind === 'BUG'
                        ? 'warning'
                        : 'secondary'
                  }
                  className="me-2"
                >
                  {todo.kind}
                </Badge>
                <span>{todo.text}</span>
                {todo.assignee && (
                  <small className="text-muted ms-2">@{todo.assignee}</small>
                )}
                <small className="text-muted ms-auto float-end">
                  L{todo.line}
                </small>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Context Usages (for variables) */}
      {element.context_usages && element.context_usages.length > 0 && (
        <div className="mb-3">
          <small className="text-uppercase text-muted fw-bold d-block mb-1">
            <i className="bi bi-signpost-split me-1"></i>
            Used In ({element.context_usages.length})
          </small>
          <div
            className="bg-light p-2 rounded"
            style={{ maxHeight: '150px', overflowY: 'auto' }}
          >
            {element.context_usages.map((usage, i) => (
              <div key={i} className="py-1">
                <code className="small">{usage}</code>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Summary */}
      {element.summary && (
        <Alert variant="light" className="border mb-3">
          <i className="bi bi-lightbulb me-2 text-warning"></i>
          {isGlossaryType ? (
            <div className="glossary-summary">
              <ReactMarkdown>{element.summary}</ReactMarkdown>
            </div>
          ) : (
            element.summary
          )}
        </Alert>
      )}

      {/* Glossary Info */}
      {isGlossaryType && element.glossary_info && (
        <>
          {/* Source Features */}
          {element.glossary_info.feature_associations.length > 0 && (
            <Card className="mb-3">
              <Card.Header>
                <i className="bi bi-diagram-3 me-2"></i>
                Extracted From ({element.glossary_info.feature_associations.length}{' '}
                features)
              </Card.Header>
              <ListGroup variant="flush">
                {element.glossary_info.feature_associations.map((assoc) => (
                  <ListGroup.Item
                    key={assoc.feature_id}
                    action
                    as={Link}
                    to={`/element/${encodeURIComponent(assoc.hash_id || assoc.feature_id)}`}
                    className="d-flex justify-content-between align-items-start"
                  >
                    <div>
                      <Badge bg="info" className="me-2">
                        <i className="bi bi-collection"></i>
                      </Badge>
                      <span className="fw-medium">{assoc.feature_label}</span>
                      {assoc.summary && (
                        <small className="d-block text-muted ms-4 mt-1">
                          {assoc.summary}
                        </small>
                      )}
                    </div>
                    {assoc.member_count > 0 && (
                      <Badge bg="secondary" pill>
                        {assoc.member_count} members
                      </Badge>
                    )}
                  </ListGroup.Item>
                ))}
              </ListGroup>
            </Card>
          )}

          {/* File Paths - only show if we have data */}
          {element.glossary_info.file_paths &&
            element.glossary_info.file_paths.length > 0 && (
              <div className="mb-3">
                <small className="text-uppercase text-muted fw-bold d-block mb-1">
                  <i className="bi bi-file-earmark-code me-1"></i>
                  Files ({element.glossary_info.file_paths.length})
                </small>
                <div
                  className="bg-light p-2 rounded"
                  style={{ maxHeight: '200px', overflowY: 'auto' }}
                >
                  {element.glossary_info.file_paths.slice(0, 20).map((path, i) => (
                    <div key={i} className="py-1">
                      <code className="small text-muted">{path}</code>
                    </div>
                  ))}
                  {element.glossary_info.file_paths.length > 20 && (
                    <div className="py-1 text-muted small">
                      ... and {element.glossary_info.file_paths.length - 20} more
                      files
                    </div>
                  )}
                </div>
              </div>
            )}
        </>
      )}

      {/* Docstring */}
      {element.docstring && (
        <div className="mb-3">
          <small className="text-uppercase text-muted fw-bold d-block mb-1">
            Documentation
          </small>
          <div className="bg-light p-3 rounded border-start border-4 border-info">
            <pre
              className="mb-0"
              style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit' }}
            >
              {element.docstring}
            </pre>
          </div>
        </div>
      )}

      {/* Base Classes - for classes */}
      {element.element_type === 'class' &&
        element.base_classes &&
        element.base_classes.length > 0 && (
          <div className="mb-3">
            <small className="text-uppercase text-muted fw-bold d-block mb-1">
              <i className="bi bi-diagram-2 me-1"></i>
              Inherits From
            </small>
            <div className="d-flex flex-wrap gap-2">
              {element.base_classes.map((base, i) => (
                <Badge key={i} bg="secondary" className="fw-normal">
                  {base}
                </Badge>
              ))}
            </div>
          </div>
        )}

      {/* Class Attributes - for classes */}
      {element.element_type === 'class' &&
        element.class_attributes &&
        element.class_attributes.length > 0 && (
          <div className="mb-3">
            <small className="text-uppercase text-muted fw-bold d-block mb-1">
              <i className="bi bi-box me-1"></i>
              Instance Attributes
            </small>
            <div className="bg-light p-2 rounded">
              {element.class_attributes.map((attr, i) => (
                <div key={i} className="d-flex align-items-center py-1">
                  <code className="me-2">{attr.name}</code>
                  {attr.type && (
                    <small className="text-muted">: {attr.type}</small>
                  )}
                  {attr.line && (
                    <small className="text-muted ms-auto">L{attr.line}</small>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

      {/* Exceptions Raised - for functions/methods */}
      {isCallableType &&
        element.exceptions_raised &&
        element.exceptions_raised.length > 0 && (
          <div className="mb-3">
            <small className="text-uppercase text-muted fw-bold d-block mb-1">
              <i className="bi bi-exclamation-triangle me-1"></i>
              Raises
            </small>
            <div className="d-flex flex-wrap gap-2">
              {element.exceptions_raised.map((exc, i) => (
                <Badge key={i} bg="danger" className="fw-normal">
                  {exc}
                </Badge>
              ))}
            </div>
          </div>
        )}

      {/* Attributes Modified - for methods */}
      {element.element_type === 'method' &&
        element.attributes_modified &&
        element.attributes_modified.length > 0 && (
          <div className="mb-3">
            <small className="text-uppercase text-muted fw-bold d-block mb-1">
              <i className="bi bi-pencil me-1"></i>
              Modifies Attributes
            </small>
            <div className="d-flex flex-wrap gap-2">
              {element.attributes_modified.map((attr, i) => (
                <Badge key={i} bg="warning" text="dark" className="fw-normal">
                  self.{attr}
                </Badge>
              ))}
            </div>
          </div>
        )}

      {/* Imports - for file elements */}
      {element.element_type === 'file' &&
        element.imports &&
        element.imports.length > 0 && (
          <div className="mb-3">
            <small className="text-uppercase text-muted fw-bold d-block mb-1">
              <i className="bi bi-box-arrow-in-down me-1"></i>
              Imports ({element.imports.length})
            </small>
            <div
              className="bg-body-tertiary border rounded p-2"
              style={{ maxHeight: '200px', overflowY: 'auto' }}
            >
              {element.imports.map((imp, i) => (
                <div key={i} className="d-flex align-items-center py-1">
                  <code className="text-primary me-2">
                    {imp.alias ? `${imp.name} as ${imp.alias}` : imp.name}
                  </code>
                  <small className="text-body-secondary">from {imp.module}</small>
                  {imp.line && (
                    <small className="text-body-secondary ms-auto">
                      L{imp.line}
                    </small>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

      {/* Section Markers - for file elements */}
      {element.element_type === 'file' &&
        element.section_markers &&
        element.section_markers.length > 0 && (
          <div className="mb-3">
            <small className="text-uppercase text-muted fw-bold d-block mb-1">
              <i className="bi bi-signpost-2 me-1"></i>
              Code Sections ({element.section_markers.length})
            </small>
            <div
              className="bg-light p-2 rounded"
              style={{ maxHeight: '150px', overflowY: 'auto' }}
            >
              {element.section_markers.map((marker, i) => (
                <div key={i} className="d-flex align-items-center py-1">
                  <Badge bg="light" text="dark" className="me-2">
                    {marker.style === 'banner'
                      ? '==='
                      : marker.style === 'box'
                        ? '+-+'
                        : '---'}
                  </Badge>
                  <span>{marker.label}</span>
                  <small className="text-muted ms-auto">L{marker.line}</small>
                </div>
              ))}
            </div>
          </div>
        )}

      {/* Document Sections - for markdown/doc file elements */}
      {element.element_type === 'file' &&
        element.document_sections &&
        element.document_sections.length > 0 && (
          <div className="mb-3">
            <small className="text-uppercase text-muted fw-bold d-block mb-1">
              <i className="bi bi-list-nested me-1"></i>
              Document Structure ({element.document_sections.length})
            </small>
            <div
              className="border rounded p-2"
              style={{ maxHeight: '200px', overflowY: 'auto' }}
            >
              {element.document_sections.map((section, i) => (
                <div
                  key={i}
                  className="d-flex align-items-center py-1"
                  style={{ paddingLeft: `${(section.level - 1) * 16}px` }}
                >
                  <Badge bg="secondary" className="me-2">
                    H{section.level}
                  </Badge>
                  <span className="text-body">{section.title}</span>
                  <small className="text-body-secondary ms-auto text-nowrap ms-2">
                    L{section.line_start}-{section.line_end}
                  </small>
                </div>
              ))}
            </div>
          </div>
        )}

      {/* Code */}
      {element.raw_code && (
        <div>
          <small className="text-uppercase text-muted fw-bold d-block mb-1">
            Source Code
          </small>
          <pre
            className="bg-dark text-light p-3 rounded"
            style={{ maxHeight: '50vh', overflowY: 'auto' }}
          >
            <code>{element.raw_code}</code>
          </pre>
        </div>
      )}
    </Card.Body>
  )
}
