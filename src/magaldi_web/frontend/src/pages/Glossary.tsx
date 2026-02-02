import { useState } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  Row,
  Col,
  Card,
  Badge,
  Spinner,
  Alert,
  Breadcrumb,
  ListGroup,
  Form,
  InputGroup,
} from 'react-bootstrap'
import ReactMarkdown from 'react-markdown'
import {
  getRepositories,
  getGlossaryTerms,
  getGlossaryTerm,
  searchGlossaryTerms,
} from '../api'

function TermDetail({ term, scope, repository }: { term: string; scope: string; repository: string }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['glossaryTerm', scope, repository, term],
    queryFn: () => getGlossaryTerm(scope, repository, term),
  })

  if (isLoading) {
    return (
      <div className="text-center py-4">
        <Spinner animation="border" size="sm" />
      </div>
    )
  }

  if (error || !data) {
    return <Alert variant="warning">Failed to load term details</Alert>
  }

  return (
    <div>
      <h4 className="mb-3">
        <Badge bg="primary" className="me-2">{data.term}</Badge>
        <small className="text-muted">{data.total_count} occurrences</small>
      </h4>

      {/* Description */}
      {data.description && (
        <Card className="mb-3 border-start border-primary border-3">
          <Card.Body className="py-2">
            <small className="text-muted d-block mb-1">
              <i className="bi bi-info-circle me-1"></i>
              Definition
            </small>
            <div className="glossary-description">
              <ReactMarkdown>{data.description}</ReactMarkdown>
            </div>
          </Card.Body>
        </Card>
      )}

      {/* Files */}
      <Card className="mb-3">
        <Card.Header>
          <i className="bi bi-file-earmark-code me-2"></i>
          Files ({data.file_paths.length})
        </Card.Header>
        <ListGroup variant="flush">
          {data.file_paths.slice(0, 10).map((path) => (
            <ListGroup.Item key={path} className="py-2">
              <code className="text-muted">{path}</code>
            </ListGroup.Item>
          ))}
          {data.file_paths.length > 10 && (
            <ListGroup.Item className="text-muted">
              ... and {data.file_paths.length - 10} more files
            </ListGroup.Item>
          )}
        </ListGroup>
      </Card>

      {/* Source Features - features/subfeatures this term was extracted from */}
      {data.feature_associations.length > 0 && (
        <Card>
          <Card.Header>
            <i className="bi bi-diagram-3 me-2"></i>
            Extracted From ({data.feature_associations.length} features)
          </Card.Header>
          <ListGroup variant="flush">
            {data.feature_associations.map((assoc) => (
              <ListGroup.Item key={assoc.feature_id} className="py-2">
                <i className="bi bi-tag me-2 text-muted"></i>
                <Link to={`/element/${encodeURIComponent(assoc.feature_id)}`}>
                  {assoc.feature_label}
                </Link>
              </ListGroup.Item>
            ))}
          </ListGroup>
        </Card>
      )}
    </div>
  )
}

function Glossary() {
  const { scope, repository } = useParams<{ scope?: string; repository?: string }>()
  const navigate = useNavigate()
  const [selectedTerm, setSelectedTerm] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [minCount, setMinCount] = useState(2)

  const { data: repos, isLoading: reposLoading } = useQuery({
    queryKey: ['repositories'],
    queryFn: getRepositories,
    enabled: !scope || !repository,
  })

  const { data: glossaryData, isLoading: glossaryLoading } = useQuery({
    queryKey: ['glossary', scope, repository, minCount],
    queryFn: () => getGlossaryTerms(scope!, repository!, minCount),
    enabled: !!scope && !!repository && !searchQuery,
  })

  const { data: searchData, isLoading: searchLoading } = useQuery({
    queryKey: ['glossarySearch', scope, repository, searchQuery],
    queryFn: () => searchGlossaryTerms(scope!, repository!, searchQuery),
    enabled: !!scope && !!repository && !!searchQuery,
  })

  const terms = searchQuery ? searchData?.terms : glossaryData?.terms
  const isLoading = searchQuery ? searchLoading : glossaryLoading

  // Repository selection view
  if (!scope || !repository) {
    if (reposLoading) {
      return (
        <div className="text-center py-5">
          <Spinner animation="border" />
        </div>
      )
    }

    return (
      <div>
        <h1 className="mb-4">
          <i className="bi bi-book me-2"></i>
          Glossary
        </h1>
        <p className="text-muted mb-4">Select a repository to explore its domain terminology.</p>
        {repos?.length ? (
          <Row>
            {repos.map((repo) => (
              <Col key={`${repo.scope}/${repo.name}`} md={4} className="mb-4">
                <Card
                  className="h-100"
                  style={{ cursor: 'pointer' }}
                  onClick={() => navigate(`/glossary/${repo.scope}/${repo.name}`)}
                >
                  <Card.Body>
                    <Card.Title>
                      <i className="bi bi-folder2-open me-2 text-primary"></i>
                      {repo.name}
                    </Card.Title>
                    <Card.Subtitle className="mb-2 text-muted">{repo.scope}</Card.Subtitle>
                    <Badge bg="secondary">{repo.element_count.toLocaleString()} elements</Badge>
                  </Card.Body>
                </Card>
              </Col>
            ))}
          </Row>
        ) : (
          <Alert variant="info">No repositories indexed yet.</Alert>
        )}
      </div>
    )
  }

  // Glossary browser view
  return (
    <div>
      <Breadcrumb className="mb-4">
        <Breadcrumb.Item linkAs={Link} linkProps={{ to: '/glossary' }}>
          Glossary
        </Breadcrumb.Item>
        <Breadcrumb.Item active>
          {scope}/{repository}
        </Breadcrumb.Item>
      </Breadcrumb>

      <Row>
        {/* Term List */}
        <Col md={4} lg={3}>
          <Card className="mb-4">
            <Card.Header>
              <i className="bi bi-book me-2"></i>
              Domain Terms
            </Card.Header>
            <Card.Body>
              {/* Search */}
              <InputGroup className="mb-3">
                <Form.Control
                  placeholder="Search terms..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
                {searchQuery && (
                  <InputGroup.Text
                    style={{ cursor: 'pointer' }}
                    onClick={() => setSearchQuery('')}
                  >
                    <i className="bi bi-x"></i>
                  </InputGroup.Text>
                )}
              </InputGroup>

              {/* Min count filter */}
              {!searchQuery && (
                <Form.Group className="mb-3">
                  <Form.Label className="small text-muted">
                    Min occurrences: {minCount}
                  </Form.Label>
                  <Form.Range
                    min={1}
                    max={10}
                    value={minCount}
                    onChange={(e) => setMinCount(Number(e.target.value))}
                  />
                </Form.Group>
              )}
            </Card.Body>

            {/* Term List */}
            <ListGroup variant="flush" style={{ maxHeight: '50vh', overflowY: 'auto' }}>
              {isLoading ? (
                <ListGroup.Item className="text-center py-3">
                  <Spinner animation="border" size="sm" />
                </ListGroup.Item>
              ) : terms?.length ? (
                terms.map((t) => (
                  <ListGroup.Item
                    key={t.term}
                    action
                    active={selectedTerm === t.term}
                    onClick={() => setSelectedTerm(t.term)}
                    className="py-2"
                  >
                    <div className="d-flex justify-content-between align-items-center">
                      <span>{t.term}</span>
                      <div>
                        <Badge bg="secondary" className="me-1">{t.total_count}</Badge>
                        {t.feature_count > 0 && (
                          <Badge bg="info" title="Related features">{t.feature_count}</Badge>
                        )}
                      </div>
                    </div>
                    {t.description && (
                      <small className={`d-block mt-1 ${selectedTerm === t.term ? 'text-light opacity-75' : 'text-muted'}`} style={{ fontSize: '0.75rem' }}>
                        {t.description.length > 60 ? `${t.description.substring(0, 60)}...` : t.description}
                      </small>
                    )}
                  </ListGroup.Item>
                ))
              ) : (
                <ListGroup.Item className="text-muted">
                  No terms found
                </ListGroup.Item>
              )}
            </ListGroup>

            <Card.Footer className="text-muted small">
              {terms?.length || 0} terms
            </Card.Footer>
          </Card>
        </Col>

        {/* Term Detail */}
        <Col md={8} lg={9}>
          {selectedTerm ? (
            <TermDetail term={selectedTerm} scope={scope} repository={repository} />
          ) : (
            <Card className="text-center py-5">
              <Card.Body>
                <i className="bi bi-book display-1 text-muted mb-3 d-block"></i>
                <h5>Select a term</h5>
                <p className="text-muted">
                  Click on a term to see which files and features use it.
                </p>
              </Card.Body>
            </Card>
          )}
        </Col>
      </Row>
    </div>
  )
}

export default Glossary
