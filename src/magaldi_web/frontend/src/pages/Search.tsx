import { useState, useEffect } from 'react'
import { useSearchParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  Row,
  Col,
  Card,
  Form,
  Button,
  InputGroup,
  Badge,
  Spinner,
  Alert,
  ListGroup,
} from 'react-bootstrap'
import { search, getRepositories, type SearchResult } from '../api'

const ELEMENT_TYPES = ['file', 'class', 'function', 'method', 'variable', 'constant']

function Search() {
  const [searchParams, setSearchParams] = useSearchParams()
  const initialQuery = searchParams.get('q') || ''

  const [query, setQuery] = useState(initialQuery)
  const [selectedScope, setSelectedScope] = useState<string>('')
  const [selectedRepo, setSelectedRepo] = useState<string>('')
  const [selectedTypes, setSelectedTypes] = useState<string[]>([])
  const [limit, setLimit] = useState(20)

  const { data: repos } = useQuery({
    queryKey: ['repositories'],
    queryFn: getRepositories,
  })

  const { data: searchResult, isLoading, error, refetch } = useQuery({
    queryKey: ['search', query, selectedScope, selectedRepo, selectedTypes, limit],
    queryFn: () =>
      search({
        query,
        scope: selectedScope || undefined,
        repository: selectedRepo || undefined,
        element_types: selectedTypes.length > 0 ? selectedTypes : undefined,
        limit,
      }),
    enabled: query.length > 0,
  })

  useEffect(() => {
    if (initialQuery && initialQuery !== query) {
      setQuery(initialQuery)
    }
  }, [initialQuery])

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      setSearchParams({ q: query })
      refetch()
    }
  }

  const toggleType = (type: string) => {
    setSelectedTypes((prev) =>
      prev.includes(type) ? prev.filter((t) => t !== type) : [...prev, type]
    )
  }

  const getTypeBadgeVariant = (type: string): string => {
    switch (type) {
      case 'file':
        return 'info'
      case 'class':
        return 'purple'
      case 'function':
        return 'primary'
      case 'method':
        return 'success'
      case 'variable':
        return 'danger'
      case 'constant':
        return 'warning'
      default:
        return 'secondary'
    }
  }

  const getTypeBadgeStyle = (type: string): React.CSSProperties => {
    if (type === 'class') {
      return { backgroundColor: '#6f42c1', color: 'white' }
    }
    return {}
  }

  return (
    <div>
      <h1 className="mb-4">Search</h1>

      <Row>
        {/* Filters Sidebar */}
        <Col md={3}>
          <Card className="mb-4">
            <Card.Header>Filters</Card.Header>
            <Card.Body>
              <Form.Group className="mb-3">
                <Form.Label>Repository</Form.Label>
                <Form.Select
                  value={`${selectedScope}/${selectedRepo}`}
                  onChange={(e) => {
                    const [scope, repo] = e.target.value.split('/')
                    setSelectedScope(scope || '')
                    setSelectedRepo(repo || '')
                  }}
                >
                  <option value="">All repositories</option>
                  {repos?.map((r) => (
                    <option key={`${r.scope}/${r.name}`} value={`${r.scope}/${r.name}`}>
                      {r.scope}/{r.name}
                    </option>
                  ))}
                </Form.Select>
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>Element Types</Form.Label>
                <div>
                  {ELEMENT_TYPES.map((type) => (
                    <Form.Check
                      key={type}
                      type="checkbox"
                      id={`type-${type}`}
                      label={type}
                      checked={selectedTypes.includes(type)}
                      onChange={() => toggleType(type)}
                    />
                  ))}
                </div>
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>Results Limit</Form.Label>
                <Form.Select
                  value={limit}
                  onChange={(e) => setLimit(Number(e.target.value))}
                >
                  <option value={10}>10</option>
                  <option value={20}>20</option>
                  <option value={50}>50</option>
                  <option value={100}>100</option>
                </Form.Select>
              </Form.Group>
            </Card.Body>
          </Card>
        </Col>

        {/* Search Results */}
        <Col md={9}>
          <Form onSubmit={handleSearch} className="mb-4">
            <InputGroup size="lg">
              <Form.Control
                type="search"
                placeholder="Search code semantically... (e.g., 'handle user authentication')"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
              <Button type="submit" variant="primary">
                <i className="bi bi-search me-2"></i>
                Search
              </Button>
            </InputGroup>
          </Form>

          {isLoading && (
            <div className="text-center py-5">
              <Spinner animation="border" role="status">
                <span className="visually-hidden">Searching...</span>
              </Spinner>
            </div>
          )}

          {error && (
            <Alert variant="danger">
              Search failed: {(error as Error).message}
            </Alert>
          )}

          {searchResult && (
            <>
              <div className="d-flex justify-content-between align-items-center mb-3">
                <span className="text-muted">
                  Found {searchResult.total} results in {searchResult.took_ms}ms
                </span>
              </div>

              {searchResult.results.length > 0 ? (
                <ListGroup>
                  {searchResult.results.map((result: SearchResult) => (
                    <ListGroup.Item
                      key={result.element_id}
                      action
                      as={Link}
                      to={`/element/${encodeURIComponent(result.element_id)}`}
                      className="d-flex flex-column"
                    >
                      <div className="d-flex justify-content-between align-items-start">
                        <div>
                          <h6 className="mb-1">
                            <Badge
                              bg={getTypeBadgeVariant(result.element_type)}
                              style={getTypeBadgeStyle(result.element_type)}
                              className="me-2"
                            >
                              {result.element_type}
                            </Badge>
                            <code>{result.name}</code>
                          </h6>
                          <small className="text-muted">
                            {result.file_path}:{result.line}
                          </small>
                        </div>
                        <Badge bg="secondary">
                          {(result.score * 100).toFixed(1)}%
                        </Badge>
                      </div>
                      {result.summary && (
                        <p className="mb-1 mt-2 text-muted small">{result.summary}</p>
                      )}
                      {result.code_snippet && (
                        <pre className="mb-0 mt-2 bg-light p-2 rounded small">
                          <code>{result.code_snippet}</code>
                        </pre>
                      )}
                    </ListGroup.Item>
                  ))}
                </ListGroup>
              ) : (
                <Alert variant="info">
                  No results found for "{searchResult.query}"
                </Alert>
              )}
            </>
          )}

          {!query && !searchResult && (
            <Card className="text-center py-5">
              <Card.Body>
                <i className="bi bi-search display-1 text-muted mb-3 d-block"></i>
                <h5>Semantic Code Search</h5>
                <p className="text-muted">
                  Search your codebase using natural language. Describe what you're looking for
                  and find relevant code elements.
                </p>
              </Card.Body>
            </Card>
          )}
        </Col>
      </Row>
    </div>
  )
}

export default Search
