import { useState, useCallback } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  Row,
  Col,
  Card,
  Form,
  Badge,
  Spinner,
  Alert,
  ListGroup,
  Breadcrumb,
  Accordion,
} from 'react-bootstrap'
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import { getRepositories, getVectorMap, getClusters, type VectorPoint } from '../api'

const ELEMENT_COLORS: Record<string, string> = {
  file: '#17a2b8',
  class: '#6f42c1',
  function: '#0d6efd',
  method: '#198754',
  variable: '#dc3545',
  constant: '#ffc107',
  feature: '#fd7e14',
}

const ELEMENT_TYPES = ['class', 'function', 'method']

function VectorMap() {
  const { scope, repository } = useParams<{ scope?: string; repository?: string }>()
  const navigate = useNavigate()

  const [selectedTypes, setSelectedTypes] = useState<string[]>(ELEMENT_TYPES)
  const [algorithm, setAlgorithm] = useState<'umap' | 'tsne'>('umap')
  const [limit, setLimit] = useState(1000)
  const [hoveredPoint, setHoveredPoint] = useState<VectorPoint | null>(null)

  const { data: repos } = useQuery({
    queryKey: ['repositories'],
    queryFn: getRepositories,
  })

  const { data: vectorMap, isLoading: mapLoading, error: mapError } = useQuery({
    queryKey: ['vectorMap', scope, repository, selectedTypes, algorithm, limit],
    queryFn: () =>
      getVectorMap(scope!, repository!, {
        element_types: selectedTypes,
        algorithm,
        limit,
      }),
    enabled: !!scope && !!repository,
  })

  const { data: clusters } = useQuery({
    queryKey: ['clusters', scope, repository],
    queryFn: () => getClusters(scope!, repository!),
    enabled: !!scope && !!repository,
  })

  const toggleType = (type: string) => {
    setSelectedTypes((prev) =>
      prev.includes(type) ? prev.filter((t) => t !== type) : [...prev, type]
    )
  }

  const handlePointClick = useCallback(
    (point: VectorPoint) => {
      navigate(`/element/${encodeURIComponent(point.element_id)}`)
    },
    [navigate]
  )

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length > 0) {
      const point = payload[0].payload as VectorPoint
      return (
        <Card className="shadow-sm" style={{ maxWidth: 300 }}>
          <Card.Body className="p-2">
            <Badge
              style={{ backgroundColor: ELEMENT_COLORS[point.element_type] }}
              className="mb-1"
            >
              {point.element_type}
            </Badge>
            <div className="fw-bold">{point.name}</div>
            <small className="text-muted d-block">
              {point.file_path}:{point.line}
            </small>
            {point.summary && (
              <small className="text-muted d-block mt-1">{point.summary}</small>
            )}
          </Card.Body>
        </Card>
      )
    }
    return null
  }

  // Repository selection view
  if (!scope || !repository) {
    return (
      <div>
        <h1 className="mb-4">Vector Space Visualization</h1>
        <p className="text-muted mb-4">
          Select a repository to visualize its embedding space.
        </p>
        {repos?.length ? (
          <Row>
            {repos.map((repo) => (
              <Col key={`${repo.scope}/${repo.repository}`} md={4} className="mb-4">
                <Card
                  className="h-100"
                  style={{ cursor: 'pointer' }}
                  onClick={() => navigate(`/vector-map/${repo.scope}/${repo.repository}`)}
                >
                  <Card.Body>
                    <Card.Title>
                      <i className="bi bi-diagram-3 me-2 text-primary"></i>
                      {repo.repository}
                    </Card.Title>
                    <Card.Subtitle className="mb-2 text-muted">
                      {repo.scope}
                    </Card.Subtitle>
                    <Badge bg="secondary">
                      {repo.element_count.toLocaleString()} elements
                    </Badge>
                  </Card.Body>
                </Card>
              </Col>
            ))}
          </Row>
        ) : (
          <Alert variant="info">
            No repositories indexed yet.
          </Alert>
        )}
      </div>
    )
  }

  return (
    <div>
      <Breadcrumb className="mb-4">
        <Breadcrumb.Item linkAs={Link} linkProps={{ to: '/vector-map' }}>
          Vector Map
        </Breadcrumb.Item>
        <Breadcrumb.Item active>
          {scope}/{repository}
        </Breadcrumb.Item>
      </Breadcrumb>

      <Row>
        {/* Controls */}
        <Col md={3}>
          <Card className="mb-4">
            <Card.Header>Controls</Card.Header>
            <Card.Body>
              <Form.Group className="mb-3">
                <Form.Label>Element Types</Form.Label>
                <div>
                  {['class', 'function', 'method', 'file'].map((type) => (
                    <Form.Check
                      key={type}
                      type="checkbox"
                      id={`vtype-${type}`}
                      label={
                        <span>
                          <span
                            className="d-inline-block rounded me-1"
                            style={{
                              width: 12,
                              height: 12,
                              backgroundColor: ELEMENT_COLORS[type],
                            }}
                          ></span>
                          {type}
                        </span>
                      }
                      checked={selectedTypes.includes(type)}
                      onChange={() => toggleType(type)}
                    />
                  ))}
                </div>
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>Algorithm</Form.Label>
                <Form.Select
                  value={algorithm}
                  onChange={(e) => setAlgorithm(e.target.value as 'umap' | 'tsne')}
                >
                  <option value="umap">UMAP</option>
                  <option value="tsne">t-SNE</option>
                </Form.Select>
              </Form.Group>

              <Form.Group className="mb-3">
                <Form.Label>Max Points</Form.Label>
                <Form.Select
                  value={limit}
                  onChange={(e) => setLimit(Number(e.target.value))}
                >
                  <option value={500}>500</option>
                  <option value={1000}>1000</option>
                  <option value={2000}>2000</option>
                  <option value={5000}>5000</option>
                </Form.Select>
              </Form.Group>
            </Card.Body>
          </Card>

          {/* Clusters/Features */}
          {clusters && clusters.clusters.length > 0 && (
            <Card className="mb-4">
              <Card.Header>
                <i className="bi bi-collection me-2"></i>
                Features ({clusters.clusters.length})
              </Card.Header>
              <Accordion
                flush
                style={{ maxHeight: '400px', overflowY: 'auto' }}
              >
                {clusters.clusters.map((cluster) => (
                  <Accordion.Item
                    key={cluster.cluster_id}
                    eventKey={String(cluster.cluster_id)}
                  >
                    <Accordion.Header>
                      <div className="d-flex flex-column">
                        <span className="fw-bold">{cluster.representative.name}</span>
                        {cluster.subfeatures.length > 0 && (
                          <small className="text-muted">
                            {cluster.subfeatures.length} sub-features
                          </small>
                        )}
                      </div>
                    </Accordion.Header>
                    <Accordion.Body className="p-2">
                      {cluster.representative.summary && (
                        <p className="text-muted small mb-2">
                          {cluster.representative.summary}
                        </p>
                      )}
                      {cluster.subfeatures.length > 0 && (
                        <ListGroup variant="flush" className="small">
                          {cluster.subfeatures.map((sub) => (
                            <ListGroup.Item
                              key={sub.subfeature_id}
                              className="py-1 px-2"
                            >
                              <div className="d-flex justify-content-between align-items-start">
                                <span>{sub.label}</span>
                                <Badge bg="secondary" pill className="ms-2">
                                  {sub.member_count}
                                </Badge>
                              </div>
                              {sub.summary && (
                                <small className="text-muted d-block">
                                  {sub.summary.slice(0, 80)}...
                                </small>
                              )}
                            </ListGroup.Item>
                          ))}
                        </ListGroup>
                      )}
                    </Accordion.Body>
                  </Accordion.Item>
                ))}
              </Accordion>
            </Card>
          )}

          {/* Legend */}
          <Card>
            <Card.Header>Legend</Card.Header>
            <Card.Body>
              {Object.entries(ELEMENT_COLORS).map(([type, color]) => (
                <div key={type} className="d-flex align-items-center mb-1">
                  <span
                    className="d-inline-block rounded-circle me-2"
                    style={{
                      width: 12,
                      height: 12,
                      backgroundColor: color,
                    }}
                  ></span>
                  <small>{type}</small>
                </div>
              ))}
            </Card.Body>
          </Card>
        </Col>

        {/* Visualization */}
        <Col md={9}>
          <Card className="vector-map-container">
            <Card.Body style={{ height: '70vh' }}>
              {mapLoading ? (
                <div className="d-flex justify-content-center align-items-center h-100">
                  <div className="text-center">
                    <Spinner animation="border" className="mb-2" />
                    <p className="text-muted">
                      Computing {algorithm.toUpperCase()} projection...
                    </p>
                  </div>
                </div>
              ) : mapError ? (
                <Alert variant="danger">
                  Failed to load vector map: {(mapError as Error).message}
                </Alert>
              ) : vectorMap && vectorMap.points.length > 0 ? (
                <>
                  <div className="d-flex justify-content-between align-items-center mb-2">
                    <small className="text-muted">
                      Showing {vectorMap.element_count.toLocaleString()} elements
                    </small>
                    <Badge bg="info">{vectorMap.algorithm.toUpperCase()}</Badge>
                  </div>
                  <ResponsiveContainer width="100%" height="95%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <XAxis
                        type="number"
                        dataKey="x"
                        domain={vectorMap.bounds.x}
                        tick={false}
                        axisLine={false}
                      />
                      <YAxis
                        type="number"
                        dataKey="y"
                        domain={vectorMap.bounds.y}
                        tick={false}
                        axisLine={false}
                      />
                      <ZAxis range={[20, 100]} />
                      <Tooltip content={<CustomTooltip />} />
                      <Scatter
                        data={vectorMap.points}
                        onClick={(data) => handlePointClick(data as unknown as VectorPoint)}
                        onMouseEnter={(data) => setHoveredPoint(data as unknown as VectorPoint)}
                        onMouseLeave={() => setHoveredPoint(null)}
                        style={{ cursor: 'pointer' }}
                      >
                        {vectorMap.points.map((point, index) => (
                          <Cell
                            key={index}
                            fill={ELEMENT_COLORS[point.element_type] || '#6c757d'}
                            opacity={
                              hoveredPoint
                                ? hoveredPoint.element_id === point.element_id
                                  ? 1
                                  : 0.3
                                : 0.7
                            }
                          />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </>
              ) : (
                <div className="d-flex justify-content-center align-items-center h-100">
                  <div className="text-center">
                    <i className="bi bi-diagram-3 display-1 text-muted mb-3 d-block"></i>
                    <h5>No elements found</h5>
                    <p className="text-muted">
                      Try adjusting the filters or check if the repository has embeddings.
                    </p>
                  </div>
                </div>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default VectorMap
