import { useState, useCallback, useRef, useEffect } from 'react'
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
  Button,
  ButtonGroup,
} from 'react-bootstrap'
import * as d3 from 'd3'
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

// Minimum zoom level to show labels
const LABEL_ZOOM_THRESHOLD = 3

interface ZoomableScatterPlotProps {
  points: VectorPoint[]
  bounds: { x: [number, number]; y: [number, number] }
  onPointClick: (point: VectorPoint) => void
  clusters?: Array<{
    cluster_id: number
    representative: { name: string }
    members: Array<{ element_id: string }>
  }>
}

function ZoomableScatterPlot({ points, bounds, onPointClick, clusters }: ZoomableScatterPlotProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [zoomLevel, setZoomLevel] = useState(1)
  const [hoveredPoint, setHoveredPoint] = useState<VectorPoint | null>(null)
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown>>()
  const gRef = useRef<d3.Selection<SVGGElement, unknown, null, undefined>>()
  const currentZoomRef = useRef(1) // Track current zoom level for event handlers

  // Compute cluster centroids and radii for drawing circles
  const clusterBounds = useCallback(() => {
    if (!clusters || clusters.length === 0) return []

    const memberIds = new Set<string>()
    const clusterData: Array<{
      id: number
      name: string
      cx: number
      cy: number
      radius: number
      color: string
    }> = []

    // Create a map of element_id to point for quick lookup
    const pointMap = new Map<string, VectorPoint>()
    points.forEach(p => pointMap.set(p.element_id, p))

    clusters.forEach((cluster, idx) => {
      const clusterPoints: VectorPoint[] = []
      cluster.members.forEach(m => {
        const point = pointMap.get(m.element_id)
        if (point && !memberIds.has(m.element_id)) {
          clusterPoints.push(point)
          memberIds.add(m.element_id)
        }
      })

      if (clusterPoints.length > 0) {
        // Calculate centroid
        const cx = d3.mean(clusterPoints, d => d.x) || 0
        const cy = d3.mean(clusterPoints, d => d.y) || 0

        // Calculate radius as max distance from centroid + padding
        const maxDist = d3.max(clusterPoints, d =>
          Math.sqrt((d.x - cx) ** 2 + (d.y - cy) ** 2)
        ) || 1

        // Generate consistent color based on cluster index
        const hue = (idx * 137.508) % 360 // Golden angle for good distribution

        clusterData.push({
          id: cluster.cluster_id,
          name: cluster.representative.name,
          cx,
          cy,
          radius: Math.max(maxDist * 1.3, 0.5), // 30% padding, minimum radius
          color: `hsla(${hue}, 70%, 50%, 0.2)`,
        })
      }
    })

    return clusterData
  }, [clusters, points])

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || points.length === 0) return

    const container = containerRef.current
    const width = container.clientWidth
    const height = container.clientHeight
    const margin = { top: 20, right: 20, bottom: 20, left: 20 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    // Create scales
    const xScale = d3.scaleLinear()
      .domain(bounds.x)
      .range([0, innerWidth])

    const yScale = d3.scaleLinear()
      .domain(bounds.y)
      .range([innerHeight, 0])

    // Clear previous content
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    gRef.current = g

    // Create clip path
    svg.append('defs')
      .append('clipPath')
      .attr('id', 'clip')
      .append('rect')
      .attr('width', innerWidth)
      .attr('height', innerHeight)

    const clippedG = g.append('g').attr('clip-path', 'url(#clip)')

    // Draw cluster circles
    const clusterCircles = clusterBounds()
    const clustersGroup = clippedG.append('g').attr('class', 'clusters')

    clustersGroup.selectAll('circle.cluster')
      .data(clusterCircles)
      .join('circle')
      .attr('class', 'cluster')
      .attr('cx', d => xScale(d.cx))
      .attr('cy', d => yScale(d.cy))
      .attr('r', d => Math.max(Math.abs(xScale(d.cx + d.radius) - xScale(d.cx)), 20))
      .attr('fill', d => d.color)
      .attr('stroke', d => d.color.replace('0.2', '0.5'))
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5')

    // Draw cluster labels
    clustersGroup.selectAll('text.cluster-label')
      .data(clusterCircles)
      .join('text')
      .attr('class', 'cluster-label')
      .attr('x', d => xScale(d.cx))
      .attr('y', d => yScale(d.cy) - Math.abs(xScale(d.cx + d.radius) - xScale(d.cx)) - 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '11px')
      .attr('fill', '#666')
      .attr('font-weight', 'bold')
      .text(d => d.name)

    // Create points group
    const pointsGroup = clippedG.append('g').attr('class', 'points')

    // Draw points
    pointsGroup.selectAll('circle.point')
      .data(points)
      .join('circle')
      .attr('class', 'point')
      .attr('cx', d => xScale(d.x))
      .attr('cy', d => yScale(d.y))
      .attr('r', 4)
      .attr('fill', d => ELEMENT_COLORS[d.element_type] || '#6c757d')
      .attr('opacity', 0.7)
      .attr('cursor', 'pointer')
      .on('mouseenter', function(_event, d) {
        d3.select(this).attr('r', 8).attr('opacity', 1)
        setHoveredPoint(d)
      })
      .on('mouseleave', function() {
        d3.select(this).attr('r', 4 / currentZoomRef.current).attr('opacity', 0.7)
        setHoveredPoint(null)
      })
      .on('click', (event, d) => {
        event.stopPropagation()
        onPointClick(d)
      })

    // Create labels group (hidden initially, OUTSIDE the transformed group)
    // Labels are rendered in screen space to avoid scaling issues
    const labelsGroup = g.append('g')
      .attr('class', 'labels')
      .attr('clip-path', 'url(#clip)')
      .style('opacity', 0)

    labelsGroup.selectAll('text.label')
      .data(points)
      .join('text')
      .attr('class', 'label')
      .attr('x', d => xScale(d.x) + 8)
      .attr('y', d => yScale(d.y) - 5)
      .attr('font-size', '9px')
      .attr('fill', '#333')
      .attr('pointer-events', 'none')
      .text(d => d.name.length > 25 ? d.name.slice(0, 25) + '...' : d.name)

    // Setup zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 20])
      .on('zoom', (event) => {
        const transform = event.transform
        currentZoomRef.current = transform.k
        setZoomLevel(transform.k)

        // Apply transform to clipped group (points and clusters)
        clippedG.attr('transform', transform.toString())

        // Adjust point sizes based on zoom
        pointsGroup.selectAll('circle.point')
          .attr('r', 4 / transform.k)

        // Adjust cluster strokes
        clustersGroup.selectAll('circle.cluster')
          .attr('stroke-width', 2 / transform.k)

        // Update label positions based on transform (labels are in screen space)
        if (transform.k >= LABEL_ZOOM_THRESHOLD) {
          labelsGroup.style('opacity', Math.min(1, (transform.k - LABEL_ZOOM_THRESHOLD) / 2))
          labelsGroup.selectAll<SVGTextElement, VectorPoint>('text.label')
            .attr('x', d => transform.applyX(xScale(d.x)) + 8)
            .attr('y', d => transform.applyY(yScale(d.y)) - 5)
        } else {
          labelsGroup.style('opacity', 0)
        }

        // Adjust cluster label size
        clustersGroup.selectAll('text.cluster-label')
          .attr('font-size', `${11 / transform.k}px`)
      })

    svg.call(zoom)
    zoomRef.current = zoom

    // Store initial transform
    svg.call(zoom.transform, d3.zoomIdentity)

  }, [points, bounds, onPointClick, clusterBounds])

  const handleZoomIn = () => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current).transition().duration(300).call(
        zoomRef.current.scaleBy, 1.5
      )
    }
  }

  const handleZoomOut = () => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current).transition().duration(300).call(
        zoomRef.current.scaleBy, 0.67
      )
    }
  }

  const handleReset = () => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current).transition().duration(300).call(
        zoomRef.current.transform, d3.zoomIdentity
      )
    }
  }

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* Zoom Controls */}
      <div style={{ position: 'absolute', top: 10, right: 10, zIndex: 10 }}>
        <ButtonGroup vertical size="sm">
          <Button variant="outline-secondary" onClick={handleZoomIn} title="Zoom In">
            <i className="bi bi-zoom-in"></i>
          </Button>
          <Button variant="outline-secondary" onClick={handleZoomOut} title="Zoom Out">
            <i className="bi bi-zoom-out"></i>
          </Button>
          <Button variant="outline-secondary" onClick={handleReset} title="Reset">
            <i className="bi bi-arrows-angle-contract"></i>
          </Button>
        </ButtonGroup>
      </div>

      {/* Zoom Level Indicator */}
      <div style={{ position: 'absolute', bottom: 10, right: 10, zIndex: 10 }}>
        <Badge bg="secondary" style={{ fontSize: '0.75rem' }}>
          {zoomLevel.toFixed(1)}x
          {zoomLevel >= LABEL_ZOOM_THRESHOLD && ' (labels visible)'}
        </Badge>
      </div>

      {/* SVG Canvas */}
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        style={{ cursor: 'grab' }}
      />

      {/* Tooltip */}
      {hoveredPoint && (
        <div
          style={{
            position: 'absolute',
            top: 10,
            left: 10,
            zIndex: 20,
            maxWidth: 300,
          }}
        >
          <Card className="shadow-sm">
            <Card.Body className="p-2">
              <Badge
                style={{ backgroundColor: ELEMENT_COLORS[hoveredPoint.element_type] }}
                className="mb-1"
              >
                {hoveredPoint.element_type}
              </Badge>
              <div className="fw-bold">{hoveredPoint.name}</div>
              <small className="text-muted d-block">
                {hoveredPoint.file_path}:{hoveredPoint.line}
              </small>
              {hoveredPoint.summary && (
                <small className="text-muted d-block mt-1">{hoveredPoint.summary}</small>
              )}
            </Card.Body>
          </Card>
        </div>
      )}
    </div>
  )
}

function VectorMap() {
  const { scope, repository } = useParams<{ scope?: string; repository?: string }>()
  const navigate = useNavigate()

  const [selectedTypes, setSelectedTypes] = useState<string[]>(ELEMENT_TYPES)
  const [algorithm, setAlgorithm] = useState<'umap' | 'tsne'>('umap')
  const [limit, setLimit] = useState(1000)

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
      navigate(`/element/${encodeURIComponent(point.hash_id || point.element_id)}`)
    },
    [navigate]
  )

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
              <Col key={`${repo.scope}/${repo.name}`} md={4} className="mb-4">
                <Card
                  className="h-100"
                  style={{ cursor: 'pointer' }}
                  onClick={() => navigate(`/vector-map/${repo.scope}/${repo.name}`)}
                >
                  <Card.Body>
                    <Card.Title>
                      <i className="bi bi-diagram-3 me-2 text-primary"></i>
                      {repo.name}
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

              <div className="text-muted small mt-3">
                <i className="bi bi-info-circle me-1"></i>
                Scroll to zoom, drag to pan. Labels appear at 3x zoom.
              </div>
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
            <Card.Body style={{ height: '70vh', padding: 0 }}>
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
                <Alert variant="danger" className="m-3">
                  Failed to load vector map: {(mapError as Error).message}
                </Alert>
              ) : vectorMap && vectorMap.points.length > 0 ? (
                <>
                  <div
                    className="d-flex justify-content-between align-items-center px-3 py-2"
                    style={{ borderBottom: '1px solid #dee2e6' }}
                  >
                    <small className="text-muted">
                      Showing {vectorMap.element_count.toLocaleString()} elements
                    </small>
                    <Badge bg="info">{vectorMap.algorithm.toUpperCase()}</Badge>
                  </div>
                  <div style={{ height: 'calc(100% - 40px)' }}>
                    <ZoomableScatterPlot
                      points={vectorMap.points}
                      bounds={vectorMap.bounds}
                      onPointClick={handlePointClick}
                      clusters={clusters?.clusters}
                    />
                  </div>
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
