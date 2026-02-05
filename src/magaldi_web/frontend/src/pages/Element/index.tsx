/**
 * Element detail page
 *
 * Displays comprehensive information about a code element including:
 * - Element metadata (type, name, signature, decorators)
 * - Documentation and source code
 * - Context (file, parent, children, siblings)
 * - Call graph (callers/callees for functions)
 * - Related code elements
 * - Code metrics and health
 */

import { useParams } from 'react-router-dom'
import { Row, Col, Card } from 'react-bootstrap'
import {
  SimilarElementsCard,
  ContextCard,
} from '../../components/element'

// Hooks
import { useElementData } from './hooks'

// Config
import { detectEntryPointType } from './config'

// Components
import {
  LoadingSpinner,
  ErrorAlert,
  NotFoundAlert,
  BreadcrumbNav,
} from './components'

// Sections
import {
  ElementHeader,
  ElementDetails,
  ChildrenSection,
  SubfeaturesSection,
  ConnectedFeaturesSection,
  MembersSection,
  SiblingsSection,
} from './sections'

// Sidebar
import {
  CallersSidebar,
  CalleesSidebar,
  EmbeddingStatusSidebar,
  GlossaryTermsSidebar,
  ElementFeaturesSidebar,
  CodeMetricsSidebar,
  CodeHealthSidebar,
  PropertiesSidebar,
} from './sidebar'

function Element() {
  const { elementId } = useParams<{ elementId: string }>()
  const decodedId = elementId ? decodeURIComponent(elementId) : ''

  const {
    element,
    isLoading,
    error,
    similar,
    explanation,
    glossaryTerms,
    elementFeatures,
  } = useElementData(decodedId)

  if (isLoading) {
    return <LoadingSpinner />
  }

  if (error) {
    return <ErrorAlert error={error as Error} />
  }

  if (!element) {
    return <NotFoundAlert />
  }

  const entryPointType = detectEntryPointType(element)

  return (
    <div>
      {/* Breadcrumb */}
      <BreadcrumbNav element={element} />

      <Row>
        <Col lg={8}>
          {/* Main Element Card */}
          <Card className="mb-4">
            <ElementHeader element={element} entryPointType={entryPointType} />
            <ElementDetails element={element} />
          </Card>

          {/* Children - shown for files and classes */}
          <ChildrenSection element={element} />

          {/* Subfeatures - shown for features (above members) */}
          <SubfeaturesSection element={element} />

          {/* Connected Features - shown for features with soft clustering connections */}
          <ConnectedFeaturesSection element={element} />

          {/* Members - shown for features and subfeatures */}
          <MembersSection element={element} />

          {/* Siblings - shown for non-file elements */}
          <SiblingsSection element={element} />
        </Col>

        <Col lg={4}>
          {/* Context Card */}
          <ContextCard
            file={element.context.file}
            parent={element.context.parent}
            elementType={element.element_type}
          />

          {/* Glossary Terms - shown for features and subfeatures */}
          <GlossaryTermsSidebar
            element={element}
            glossaryTerms={glossaryTerms}
          />

          {/* Callers - who calls this function */}
          <CallersSidebar
            element={element}
            explanation={explanation}
            decodedId={decodedId}
          />

          {/* Callees - what this function calls */}
          <CalleesSidebar
            element={element}
            explanation={explanation}
            decodedId={decodedId}
          />

          {/* Embedding Status */}
          <EmbeddingStatusSidebar explanation={explanation} />

          {/* Connected Features/Subfeatures */}
          <ElementFeaturesSidebar elementFeatures={elementFeatures} />

          {/* Related Code */}
          <SimilarElementsCard similar={similar || []} />

          {/* Code Metrics Card - for functions/methods */}
          <CodeMetricsSidebar element={element} />

          {/* Code Health Card - for files/classes */}
          <CodeHealthSidebar element={element} />

          {/* Info Card */}
          <PropertiesSidebar
            element={element}
            entryPointType={entryPointType}
          />
        </Col>
      </Row>
    </div>
  )
}

export default Element
