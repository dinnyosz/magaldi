import { useState, useEffect, useRef } from 'react'
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
  OverlayTrigger,
  Tooltip,
} from 'react-bootstrap'
import ReactMarkdown from 'react-markdown'
import { search, generateSearchSummary, getRepositories, getBrowseFilters, type SearchResult } from '../api'

const ELEMENT_TYPES = ['file', 'class', 'function', 'method', 'variable', 'constant', 'feature', 'subfeature']
const DEBOUNCE_MS = 300

function Search() {
  const [searchParams, setSearchParams] = useSearchParams()
  const initialQuery = searchParams.get('q') || ''

  const [query, setQuery] = useState(initialQuery)
  const [debouncedQuery, setDebouncedQuery] = useState(initialQuery)
  const [selectedScope, setSelectedScope] = useState<string>('')
  const [selectedRepo, setSelectedRepo] = useState<string>('')
  const [selectedTypes, setSelectedTypes] = useState<string[]>([])
  const [selectedUsername, setSelectedUsername] = useState<string>('')
  const [limit, setLimit] = useState(20)
  const [useTextSearch, setUseTextSearch] = useState(true)
  const [useVectorSearch, setUseVectorSearch] = useState(true)
  const [includeTests, setIncludeTests] = useState(false)
  const [generateSummary, setGenerateSummary] = useState(false)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const { data: repos } = useQuery({
    queryKey: ['repositories'],
    queryFn: getRepositories,
  })

  const { data: filters } = useQuery({
    queryKey: ['browse-filters'],
    queryFn: getBrowseFilters,
  })

  const { data: searchResult, isLoading, error } = useQuery({
    queryKey: ['search', debouncedQuery, selectedScope, selectedRepo, selectedTypes, selectedUsername, limit, useTextSearch, useVectorSearch, includeTests],
    queryFn: () =>
      search({
        query: debouncedQuery,
        scope: selectedScope || undefined,
        repository: selectedRepo || undefined,
        element_types: selectedTypes.length > 0 ? selectedTypes : undefined,
        username: selectedUsername || undefined,
        limit,
        use_text_search: useTextSearch,
        use_vector_search: useVectorSearch,
        include_tests: includeTests,
      }),
    enabled: debouncedQuery.length > 0 && (useTextSearch || useVectorSearch),
  })

  // Separate query for AI summary - only runs after search results are available
  const { data: summaryResult, isLoading: isSummaryLoading } = useQuery({
    queryKey: ['search-summary', debouncedQuery, searchResult?.results?.slice(0, 50).map(r => r.element_id).join(',')],
    queryFn: () =>
      generateSearchSummary({
        query: debouncedQuery,
        results: (searchResult?.results || []).slice(0, 50).map(r => ({
          name: r.name,
          element_type: r.element_type,
          file_path: r.file_path,
          line: r.line,
          summary: r.summary,
          signature: null,
        })),
      }),
    enabled: generateSummary && !!searchResult?.results?.length,
  })

  // Debounce query changes
  useEffect(() => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current)
    }
    debounceRef.current = setTimeout(() => {
      setDebouncedQuery(query)
      if (query.trim()) {
        setSearchParams({ q: query })
      }
    }, DEBOUNCE_MS)
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current)
      }
    }
  }, [query, setSearchParams])

  useEffect(() => {
    if (initialQuery && initialQuery !== query) {
      setQuery(initialQuery)
      setDebouncedQuery(initialQuery)
    }
  }, [initialQuery])

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      // Immediate search on submit
      if (debounceRef.current) {
        clearTimeout(debounceRef.current)
      }
      setDebouncedQuery(query)
      setSearchParams({ q: query })
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
            <Card.Header>Search Mode</Card.Header>
            <Card.Body className="pb-2">
              <Form.Check
                type="switch"
                id="text-search"
                label="TextRank"
                checked={useTextSearch}
                onChange={(e) => {
                  if (!e.target.checked && !useVectorSearch) return
                  setUseTextSearch(e.target.checked)
                }}
                className="mb-2"
              />
              <Form.Check
                type="switch"
                id="vector-search"
                label="Semantic"
                checked={useVectorSearch}
                onChange={(e) => {
                  if (!e.target.checked && !useTextSearch) return
                  setUseVectorSearch(e.target.checked)
                }}
                className="mb-2"
              />
              <hr className="my-2" />
              <Form.Check
                type="switch"
                id="generate-summary"
                label="AI Summary"
                checked={generateSummary}
                onChange={(e) => setGenerateSummary(e.target.checked)}
              />
              <Form.Text className="text-muted small">
                Generate overview from top results
              </Form.Text>
              <hr className="my-2" />
              <Form.Check
                type="switch"
                id="include-tests"
                label="Include Tests"
                checked={includeTests}
                onChange={(e) => setIncludeTests(e.target.checked)}
              />
            </Card.Body>
          </Card>

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
                <Form.Label>User Branch</Form.Label>
                <Form.Select
                  value={selectedUsername}
                  onChange={(e) => setSelectedUsername(e.target.value)}
                >
                  <option value="">All users (main + branches)</option>
                  {filters?.usernames?.map((username) => (
                    <option key={username} value={username}>
                      {username}
                    </option>
                  ))}
                </Form.Select>
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
              {useVectorSearch && !searchResult.vector_search_used && searchResult.embedding_error && (
                <Alert variant="warning" className="mb-3">
                  <i className="bi bi-exclamation-triangle me-2"></i>
                  <strong>Vector search failed:</strong> {searchResult.embedding_error}
                </Alert>
              )}
              <div className="d-flex justify-content-between align-items-center mb-3">
                <span className="text-muted">
                  Found {searchResult.total} results in {searchResult.took_ms}ms
                </span>
                <div className="d-flex gap-1">
                  {searchResult.text_search_used && (
                    <Badge bg="secondary">TextRank</Badge>
                  )}
                  {searchResult.vector_search_used && (
                    <Badge bg="primary">Semantic</Badge>
                  )}
                </div>
              </div>

              {/* AI Summary */}
              {generateSummary && (
                <Card className="mb-4 border-info">
                  <Card.Header className="bg-info bg-opacity-10 d-flex align-items-center">
                    <i className="bi bi-stars me-2"></i>
                    <strong>AI Summary</strong>
                    {isSummaryLoading && (
                      <Spinner animation="border" size="sm" className="ms-2" />
                    )}
                  </Card.Header>
                  <Card.Body>
                    {isSummaryLoading && (
                      <div className="text-muted">
                        <div className="placeholder-glow">
                          <span className="placeholder col-12 mb-2"></span>
                          <span className="placeholder col-11 mb-2"></span>
                          <span className="placeholder col-10 mb-2"></span>
                          <span className="placeholder col-12 mb-3"></span>
                          <span className="placeholder col-11 mb-2"></span>
                          <span className="placeholder col-9"></span>
                        </div>
                      </div>
                    )}
                    {!isSummaryLoading && summaryResult?.summary && (
                      <ReactMarkdown>{summaryResult.summary}</ReactMarkdown>
                    )}
                    {!isSummaryLoading && summaryResult?.error && (
                      <Alert variant="warning" className="mb-0">
                        <i className="bi bi-exclamation-triangle me-2"></i>
                        <strong>Failed:</strong> {summaryResult.error}
                      </Alert>
                    )}
                  </Card.Body>
                </Card>
              )}

              {searchResult.results.length > 0 ? (
                <ListGroup>
                  {searchResult.results.map((result: SearchResult) => (
                    <ListGroup.Item
                      key={result.element_id}
                      action
                      as={Link}
                      to={`/element/${result.hash_id || result.element_id}`}
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
                        <div className="d-flex gap-1">
                          {result.combined_score !== null && (
                            <OverlayTrigger
                              placement="top"
                              overlay={<Tooltip>Combined score (sum of enabled search modes)</Tooltip>}
                            >
                              <Badge bg="dark">{result.combined_score.toFixed(3)}</Badge>
                            </OverlayTrigger>
                          )}
                          {useTextSearch && result.text_score !== null && (
                            <OverlayTrigger
                              placement="top"
                              overlay={<Tooltip>TextRank score (0-1)</Tooltip>}
                            >
                              <Badge bg="secondary">T:{result.text_score.toFixed(2)}</Badge>
                            </OverlayTrigger>
                          )}
                          {useVectorSearch && result.vector_score !== null && (
                            <OverlayTrigger
                              placement="top"
                              overlay={<Tooltip>Semantic similarity (0-1)</Tooltip>}
                            >
                              <Badge bg="primary">S:{result.vector_score.toFixed(2)}</Badge>
                            </OverlayTrigger>
                          )}
                        </div>
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
