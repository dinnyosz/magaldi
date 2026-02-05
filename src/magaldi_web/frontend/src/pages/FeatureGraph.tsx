import { useState, useCallback, useRef, useEffect, useMemo } from 'react'
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
  Breadcrumb,
  Button,
  ButtonGroup,
  OverlayTrigger,
  Tooltip,
} from 'react-bootstrap'
import * as d3 from 'd3'
import { getRepositories, getFeatureGraph, type FeatureGraphNode, type FeatureConnection } from '../api'

// Color palette for features
const FEATURE_COLORS = [
  '#6366f1', // indigo
  '#8b5cf6', // violet
  '#a855f7', // purple
  '#d946ef', // fuchsia
  '#ec4899', // pink
  '#f43f5e', // rose
  '#ef4444', // red
  '#f97316', // orange
  '#f59e0b', // amber
  '#eab308', // yellow
  '#84cc16', // lime
  '#22c55e', // green
  '#10b981', // emerald
  '#14b8a6', // teal
  '#06b6d4', // cyan
  '#0ea5e9', // sky
  '#3b82f6', // blue
]

// Zoom thresholds for text visibility
const LABEL_ZOOM_THRESHOLD = 0.5

interface GraphNode extends d3.SimulationNodeDatum {
  id: string
  hash_id: string | null
  label: string
  node_type: 'feature' | 'subfeature'
  summary: string | null
  member_count: number
  parent_id: string | null
  color: string
  radius: number
}

interface GraphLink extends d3.SimulationLinkDatum<GraphNode> {
  source: string | GraphNode
  target: string | GraphNode
  affinity: number
  shared_member_count: number
  linkType: 'parent-child' | 'feature-connection'
}

interface FeatureGraphVisualizationProps {
  nodes: FeatureGraphNode[]
  connections: FeatureConnection[]
  onNodeClick: (node: GraphNode) => void
}

function FeatureGraphVisualization({ nodes, connections, onNodeClick }: FeatureGraphVisualizationProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [zoomLevel, setZoomLevel] = useState(1)
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null)
  const [minSharedMembers, setMinSharedMembers] = useState(5)
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown>>()
  const simulationRef = useRef<d3.Simulation<GraphNode, GraphLink> | null>(null)
  const featureLinksRef = useRef<d3.Selection<SVGLineElement | d3.BaseType, GraphLink, SVGGElement, unknown> | null>(null)

  // Process nodes and create graph data
  const { graphNodes, graphLinks } = useMemo(() => {
    const labelToId = new Map<string, string>()
    const featureLabelToColorMap = new Map<string, string>()
    const graphNodes: GraphNode[] = []
    const graphLinks: GraphLink[] = []

    // First pass: create feature nodes and assign colors
    const features = nodes.filter(n => n.node_type === 'feature')
    features.forEach((node, idx) => {
      const color = FEATURE_COLORS[idx % FEATURE_COLORS.length]
      featureLabelToColorMap.set(node.label, color)
      labelToId.set(node.label, node.id)

      graphNodes.push({
        id: node.id,
        hash_id: node.hash_id,
        label: node.label,
        node_type: node.node_type,
        summary: node.summary,
        member_count: node.member_count,
        parent_id: node.parent_id,
        color,
        radius: Math.max(20, Math.min(50, 15 + Math.sqrt(node.member_count) * 3)),
      })
    })

    // Second pass: create subfeature nodes
    const subfeatures = nodes.filter(n => n.node_type === 'subfeature')
    subfeatures.forEach((node) => {
      const parentId = node.parent_id
      let parentLabel = ''

      // Find parent label from parent_id
      const parentNode = features.find(f => f.id === parentId)
      if (parentNode) {
        parentLabel = parentNode.label
      }

      const color = featureLabelToColorMap.get(parentLabel) || '#94a3b8'

      graphNodes.push({
        id: node.id,
        hash_id: node.hash_id,
        label: node.label,
        node_type: node.node_type,
        summary: node.summary,
        member_count: node.member_count,
        parent_id: node.parent_id,
        color,
        radius: Math.max(8, Math.min(20, 6 + Math.sqrt(node.member_count) * 1.5)),
      })

      // Create parent-child link
      if (parentId) {
        graphLinks.push({
          source: parentId,
          target: node.id,
          affinity: 1,
          shared_member_count: 0,
          linkType: 'parent-child',
        })
      }
    })

    // Create feature-to-feature connections
    const seenConnections = new Set<string>()
    connections.forEach((conn) => {
      const sourceId = labelToId.get(conn.source)
      const targetId = labelToId.get(conn.target)

      if (sourceId && targetId && sourceId !== targetId) {
        // Avoid duplicate connections (A->B and B->A)
        const key = [sourceId, targetId].sort().join('|')
        if (!seenConnections.has(key)) {
          seenConnections.add(key)
          graphLinks.push({
            source: sourceId,
            target: targetId,
            affinity: conn.affinity,
            shared_member_count: conn.shared_member_count,
            linkType: 'feature-connection',
          })
        }
      }
    })

    return { graphNodes, graphLinks }
  }, [nodes, connections])

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || graphNodes.length === 0) return

    const container = containerRef.current
    const width = container.clientWidth
    const height = container.clientHeight

    // Clear previous content
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    // Create main group for zoom
    const g = svg.append('g')

    // Create arrow marker for connections
    svg.append('defs')
      .append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '-0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .append('path')
      .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
      .attr('fill', '#94a3b8')

    // Create force simulation - spread features evenly, cluster subfeatures with parents
    const simulation = d3.forceSimulation<GraphNode>(graphNodes)
      .force('link', d3.forceLink<GraphNode, GraphLink>(graphLinks)
        .id(d => d.id)
        .distance(d => {
          if (d.linkType === 'parent-child') return 50
          // Feature connections don't pull nodes together (visual only)
          return 300
        })
        .strength(d => {
          if (d.linkType === 'parent-child') return 0.8
          // Don't use connection links for layout
          return 0
        })
      )
      .force('charge', d3.forceManyBody()
        // Strong repulsion to spread nodes apart evenly
        .strength(d => (d as GraphNode).node_type === 'feature' ? -1500 : -100)
        .distanceMax(600)
      )
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide<GraphNode>()
        // Generous padding between nodes
        .radius(d => d.radius + 40)
        .strength(1)
      )
      // Spread features in a wide ring
      .force('radial', d3.forceRadial<GraphNode>(
        d => d.node_type === 'feature' ? Math.min(width, height) * 0.35 : 40,
        width / 2,
        height / 2
      ).strength(d => d.node_type === 'feature' ? 0.3 : 0.5))

    simulationRef.current = simulation

    // Create links
    const linkGroup = g.append('g').attr('class', 'links')

    // Parent-child links (thinner, solid, teal color)
    const parentChildLinks = linkGroup.selectAll('line.parent-child')
      .data(graphLinks.filter(l => l.linkType === 'parent-child'))
      .join('line')
      .attr('class', 'parent-child')
      .attr('stroke', '#14b8a6')  // Teal color
      .attr('stroke-width', 2)
      .attr('stroke-opacity', 0.5)

    // Feature connection links - based on shared soft clustering members
    // Filter based on threshold
    const featureConnectionData = graphLinks.filter(
      l => l.linkType === 'feature-connection' && l.shared_member_count >= minSharedMembers
    )

    // Calculate max shared count for normalization
    const maxShared = Math.max(1, ...featureConnectionData.map(d => d.shared_member_count))

    // Use curved paths instead of straight lines to avoid crossing through nodes
    const featureLinks = linkGroup.selectAll('path.feature-connection')
      .data(featureConnectionData)
      .join('path')
      .attr('class', 'feature-connection')
      .attr('fill', 'none')
      .attr('stroke', '#f97316')  // Orange color for visibility
      .attr('stroke-width', d => 1.5 + (d.shared_member_count / maxShared) * 6)
      .attr('stroke-opacity', 0.1)  // Subtle hint, full opacity on hover

    featureLinksRef.current = featureLinks as unknown as d3.Selection<SVGLineElement | d3.BaseType, GraphLink, SVGGElement, unknown>

    // Create node group
    const nodeGroup = g.append('g').attr('class', 'nodes')

    // Create nodes
    const nodeElements = nodeGroup.selectAll<SVGGElement, GraphNode>('g.node')
      .data(graphNodes)
      .join('g')
      .attr('class', 'node')
      .attr('cursor', 'pointer')
      .call(d3.drag<SVGGElement, GraphNode>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart()
          d.fx = d.x
          d.fy = d.y
        })
        .on('drag', (event, d) => {
          d.fx = event.x
          d.fy = event.y
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0)
          d.fx = null
          d.fy = null
          // Reset labels - get current zoom from SVG transform
          const currentTransform = d3.zoomTransform(svg.node()!)
          const currentZoom = currentTransform.k
          labels.attr('opacity', currentZoom >= LABEL_ZOOM_THRESHOLD
            ? Math.min(1, (currentZoom - LABEL_ZOOM_THRESHOLD) / 0.5)
            : 0
          )
        })
      )

    // Add circles to nodes
    nodeElements.append('circle')
      .attr('r', d => d.radius)
      .attr('fill', d => d.color)
      .attr('fill-opacity', d => d.node_type === 'feature' ? 0.9 : 0.7)
      .attr('stroke', d => d.node_type === 'feature' ? d3.color(d.color)?.darker(0.5)?.toString() || d.color : 'none')
      .attr('stroke-width', d => d.node_type === 'feature' ? 3 : 0)

    // Add labels (initially hidden based on zoom)
    const labels = nodeElements.append('text')
      .attr('class', 'node-label')
      .attr('text-anchor', 'middle')
      .attr('dy', d => d.radius + 14)
      .attr('font-size', d => d.node_type === 'feature' ? '12px' : '10px')
      .attr('font-weight', d => d.node_type === 'feature' ? 'bold' : 'normal')
      .attr('fill', '#1e293b')
      .attr('opacity', 0)
      .text(d => d.label)

    // Add member count badge
    nodeElements.filter(d => d.node_type === 'feature')
      .append('text')
      .attr('class', 'member-count')
      .attr('text-anchor', 'middle')
      .attr('dy', 4)
      .attr('font-size', '11px')
      .attr('font-weight', 'bold')
      .attr('fill', 'white')
      .text(d => d.member_count)

    // Node interactions - highlight connections and dim non-connected nodes on hover
    nodeElements
      .on('mouseenter', function(_event, d) {
        setHoveredNode(d)

        // Interrupt any ongoing transitions to prevent race conditions
        featureLinks.interrupt()
        nodeElements.interrupt()
        nodeElements.select('circle').interrupt()
        nodeElements.select('text.node-label').interrupt()

        // Find connected node IDs
        const connectedIds = new Set<string>()
        connectedIds.add(d.id)  // Include self

        if (d.node_type === 'feature') {
          featureConnectionData.forEach(link => {
            const sourceId = typeof link.source === 'string' ? link.source : link.source.id
            const targetId = typeof link.target === 'string' ? link.target : link.target.id
            if (sourceId === d.id) connectedIds.add(targetId)
            if (targetId === d.id) connectedIds.add(sourceId)
          })

          // Highlight connections for this node, dim others
          featureLinks
            .transition()
            .duration(150)
            .attr('stroke-opacity', link => {
              const sourceId = typeof link.source === 'string' ? link.source : link.source.id
              const targetId = typeof link.target === 'string' ? link.target : link.target.id
              if (sourceId === d.id || targetId === d.id) {
                return 0.6 + (link.shared_member_count / maxShared) * 0.3
              }
              return 0.05  // Dim non-connected
            })
        }

        // Dim non-connected nodes, highlight connected ones
        nodeElements
          .transition()
          .duration(150)
          .attr('opacity', (node: unknown) => {
            const n = node as GraphNode
            return connectedIds.has(n.id) ? 1 : 0.3
          })

        // Show labels for connected nodes
        nodeElements.select('text.node-label')
          .transition()
          .duration(150)
          .attr('opacity', (node: unknown) => {
            const n = node as GraphNode
            return connectedIds.has(n.id) ? 1 : 0
          })

        // Highlight hovered and connected nodes' borders
        nodeElements.select('circle')
          .transition()
          .duration(150)
          .attr('stroke-width', (node: unknown) => {
            const n = node as GraphNode
            if (n.id === d.id) return 5  // Hovered node
            if (connectedIds.has(n.id)) return 5  // Connected nodes
            return n.node_type === 'feature' ? 3 : 0  // Default
          })

        // Reset all node sizes first, then enlarge hovered node
        nodeElements.each(function(node) {
          d3.select(this).select('circle').attr('r', node.radius)
        })
        d3.select(this).select('circle')
          .transition()
          .duration(150)
          .attr('r', d.radius * 1.1)
      })
      .on('mouseleave', function(_event, d) {
        setHoveredNode(null)

        // Reset hovered node size
        d3.select(this).select('circle')
          .transition()
          .duration(150)
          .attr('r', d.radius)

        // Return all connections to subtle opacity
        featureLinks
          .transition()
          .duration(150)
          .attr('stroke-opacity', 0.1)

        // Reset all nodes opacity and borders
        nodeElements
          .transition()
          .duration(150)
          .attr('opacity', 1)

        nodeElements.select('circle')
          .transition()
          .duration(150)
          .attr('stroke-width', (node: unknown) =>
            (node as GraphNode).node_type === 'feature' ? 3 : 0
          )

        // Hide labels (unless zoom level shows them)
        nodeElements.select('text.node-label')
          .transition()
          .duration(150)
          .attr('opacity', zoomLevel >= LABEL_ZOOM_THRESHOLD
            ? Math.min(1, (zoomLevel - LABEL_ZOOM_THRESHOLD) / 0.5)
            : 0
          )
      })

    // Reset everything when mouse leaves the graph area
    const resetHoverState = () => {
      setHoveredNode(null)

      featureLinks
        .transition()
        .duration(150)
        .attr('stroke-opacity', 0.1)

      nodeElements
        .transition()
        .duration(150)
        .attr('opacity', 1)

      nodeElements.each(function(d) {
        d3.select(this).select('circle')
          .transition()
          .duration(150)
          .attr('stroke-width', d.node_type === 'feature' ? 3 : 0)
          .attr('r', d.radius)
      })

      nodeElements.select('text.node-label')
        .transition()
        .duration(150)
        .attr('opacity', zoomLevel >= LABEL_ZOOM_THRESHOLD
          ? Math.min(1, (zoomLevel - LABEL_ZOOM_THRESHOLD) / 0.5)
          : 0
        )
    }

    svg.on('mouseleave', resetHoverState)
    g.on('mouseleave', resetHoverState)

    nodeElements.on('click', (event, d) => {
      event.stopPropagation()
      onNodeClick(d)
    })

    // Update positions on simulation tick
    simulation.on('tick', () => {
      parentChildLinks
        .attr('x1', d => (d.source as GraphNode).x || 0)
        .attr('y1', d => (d.source as GraphNode).y || 0)
        .attr('x2', d => (d.target as GraphNode).x || 0)
        .attr('y2', d => (d.target as GraphNode).y || 0)

      // Draw curved paths for feature connections - gentle curves that bundle together
      featureLinks.attr('d', d => {
        const source = d.source as GraphNode
        const target = d.target as GraphNode
        const x1 = source.x || 0
        const y1 = source.y || 0
        const x2 = target.x || 0
        const y2 = target.y || 0

        // Calculate midpoint
        const midX = (x1 + x2) / 2
        const midY = (y1 + y2) / 2

        // Calculate perpendicular offset for curve
        const dx = x2 - x1
        const dy = y2 - y1
        const dist = Math.sqrt(dx * dx + dy * dy)

        if (dist < 1) return `M ${x1} ${y1} L ${x2} ${y2}`

        // Perpendicular direction (normalized)
        const perpX = -dy / dist
        const perpY = dx / dist

        // Always curve AWAY from graph center (outward)
        const centerX = width / 2
        const centerY = height / 2

        // Vector from center to midpoint
        const fromCenterX = midX - centerX
        const fromCenterY = midY - centerY

        // Determine which direction is "outward"
        const dot = fromCenterX * perpX + fromCenterY * perpY
        const outwardSign = dot > 0 ? 1 : -1

        // Smaller curve amount - keeps edges bundled closer
        const curveOffset = dist * 0.15

        const ctrlX = midX + perpX * curveOffset * outwardSign
        const ctrlY = midY + perpY * curveOffset * outwardSign

        return `M ${x1} ${y1} Q ${ctrlX} ${ctrlY} ${x2} ${y2}`
      })

      nodeElements.attr('transform', d => `translate(${d.x || 0},${d.y || 0})`)
    })

    // Setup zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 10])
      .on('zoom', (event) => {
        const transform = event.transform
        setZoomLevel(transform.k)
        g.attr('transform', transform.toString())

        // Update label visibility based on zoom
        const labelOpacity = transform.k >= LABEL_ZOOM_THRESHOLD
          ? Math.min(1, (transform.k - LABEL_ZOOM_THRESHOLD) / 0.5)
          : 0
        labels.attr('opacity', labelOpacity)

        // Don't scale node radius - let them naturally get smaller when zoomed out
        // This ensures accurate hit detection at all zoom levels

        nodeElements.selectAll('text.node-label')
          .attr('dy', (d: unknown) => (d as GraphNode).radius + 14)
          .attr('font-size', (d: unknown) => {
            const base = (d as GraphNode).node_type === 'feature' ? 12 : 10
            return `${base}px`
          })

        nodeElements.selectAll('text.member-count')
          .attr('font-size', `${11 / transform.k}px`)

        // Adjust link widths
        parentChildLinks.attr('stroke-width', 2 / transform.k)
        const zoomMaxShared = Math.max(1, ...featureConnectionData.map(d => d.shared_member_count))
        featureLinks.attr('stroke-width', (d: GraphLink) =>
          (1.5 + (d.shared_member_count / zoomMaxShared) * 4) / transform.k
        )
      })

    svg.call(zoom)
    zoomRef.current = zoom

    // Initial zoom to fit
    const bounds = g.node()?.getBBox()
    if (bounds) {
      const dx = bounds.width
      const dy = bounds.height
      const x = bounds.x + dx / 2
      const y = bounds.y + dy / 2
      const scale = Math.min(0.8, 0.8 / Math.max(dx / width, dy / height))
      const translate = [width / 2 - scale * x, height / 2 - scale * y]

      svg.call(zoom.transform, d3.zoomIdentity
        .translate(translate[0], translate[1])
        .scale(scale))
    }

    return () => {
      simulation.stop()
    }
  }, [graphNodes, graphLinks, onNodeClick, minSharedMembers])

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
    if (svgRef.current && zoomRef.current && containerRef.current) {
      const width = containerRef.current.clientWidth
      const height = containerRef.current.clientHeight
      d3.select(svgRef.current).transition().duration(300).call(
        zoomRef.current.transform,
        d3.zoomIdentity.translate(width / 2, height / 2).scale(0.8)
      )
    }
  }

  // Count connections at current threshold
  const connectionCount = graphLinks.filter(
    l => l.linkType === 'feature-connection' && l.shared_member_count >= minSharedMembers
  ).length

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* Connection Controls */}
      <div style={{ position: 'absolute', top: 10, left: 10, zIndex: 10, maxWidth: 180 }}>
        <Card className="shadow-sm" style={{ fontSize: '0.85rem' }}>
          <Card.Body className="p-2">
            <Form.Label className="mb-1 d-block">
              <small>Min shared members: {minSharedMembers}</small>
            </Form.Label>
            <Form.Range
              min={1}
              max={20}
              value={minSharedMembers}
              onChange={(e) => setMinSharedMembers(Number(e.target.value))}
              className="mb-1"
            />
            <small className="text-muted d-block">
              {connectionCount} connections
            </small>
          </Card.Body>
        </Card>
      </div>

      {/* Zoom Controls */}
      <div style={{ position: 'absolute', top: 10, right: 10, zIndex: 10 }}>
        <ButtonGroup vertical size="sm">
          <OverlayTrigger placement="left" overlay={<Tooltip>Zoom In</Tooltip>}>
            <Button variant="outline-secondary" onClick={handleZoomIn}>
              <i className="bi bi-zoom-in"></i>
            </Button>
          </OverlayTrigger>
          <OverlayTrigger placement="left" overlay={<Tooltip>Zoom Out</Tooltip>}>
            <Button variant="outline-secondary" onClick={handleZoomOut}>
              <i className="bi bi-zoom-out"></i>
            </Button>
          </OverlayTrigger>
          <OverlayTrigger placement="left" overlay={<Tooltip>Reset View</Tooltip>}>
            <Button variant="outline-secondary" onClick={handleReset}>
              <i className="bi bi-arrows-angle-contract"></i>
            </Button>
          </OverlayTrigger>
        </ButtonGroup>
      </div>

      {/* Zoom Level Indicator */}
      <div style={{ position: 'absolute', bottom: 10, right: 10, zIndex: 10 }}>
        <Badge bg="secondary" style={{ fontSize: '0.75rem' }}>
          {zoomLevel.toFixed(1)}x
        </Badge>
      </div>

      {/* SVG Canvas */}
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        style={{ cursor: 'grab', background: '#f8fafc' }}
      />

      {/* Tooltip */}
      {hoveredNode && (
        <div
          style={{
            position: 'absolute',
            top: 10,
            left: 10,
            zIndex: 20,
            maxWidth: 350,
          }}
        >
          <Card className="shadow-sm">
            <Card.Body className="p-2">
              <Badge
                bg={hoveredNode.node_type === 'feature' ? 'primary' : 'secondary'}
                className="mb-1"
              >
                {hoveredNode.node_type}
              </Badge>
              <span
                className="ms-2 d-inline-block rounded"
                style={{
                  width: 12,
                  height: 12,
                  backgroundColor: hoveredNode.color,
                  verticalAlign: 'middle',
                }}
              ></span>
              <div className="fw-bold">{hoveredNode.label}</div>
              <small className="text-muted d-block">
                {hoveredNode.member_count} members
              </small>
              {hoveredNode.summary && (
                <small className="text-muted d-block mt-1">
                  {hoveredNode.summary.slice(0, 150)}
                  {hoveredNode.summary.length > 150 ? '...' : ''}
                </small>
              )}
            </Card.Body>
          </Card>
        </div>
      )}
    </div>
  )
}

function FeatureGraph() {
  const { scope, repository } = useParams<{ scope?: string; repository?: string }>()
  const navigate = useNavigate()

  const { data: repos } = useQuery({
    queryKey: ['repositories'],
    queryFn: getRepositories,
  })

  const { data: featureGraph, isLoading, error } = useQuery({
    queryKey: ['featureGraph', scope, repository],
    queryFn: () => getFeatureGraph(scope!, repository!),
    enabled: !!scope && !!repository,
  })

  const handleNodeClick = useCallback(
    (node: GraphNode) => {
      if (node.hash_id) {
        navigate(`/element/${encodeURIComponent(node.hash_id)}`)
      }
    },
    [navigate]
  )

  // Repository selection view
  if (!scope || !repository) {
    return (
      <div>
        <h1 className="mb-4">
          <i className="bi bi-diagram-3 me-2"></i>
          Feature Graph
        </h1>
        <p className="text-muted mb-4">
          Visualize features, subfeatures, and their connections. Select a repository to explore.
        </p>
        {repos?.length ? (
          <Row>
            {repos.map((repo) => (
              <Col key={`${repo.scope}/${repo.name}`} md={4} className="mb-4">
                <Card
                  className="h-100 feature-card"
                  style={{ cursor: 'pointer' }}
                  onClick={() => navigate(`/feature-graph/${repo.scope}/${repo.name}`)}
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
        <Breadcrumb.Item linkAs={Link} linkProps={{ to: '/feature-graph' }}>
          Feature Graph
        </Breadcrumb.Item>
        <Breadcrumb.Item active>
          {scope}/{repository}
        </Breadcrumb.Item>
      </Breadcrumb>

      <Row>
        {/* Controls */}
        <Col md={3}>
          <Card className="mb-4">
            <Card.Header>
              <i className="bi bi-info-circle me-2"></i>
              Graph Info
            </Card.Header>
            <Card.Body>
              {featureGraph && (
                <>
                  <div className="mb-2">
                    <strong>Features:</strong> {featureGraph.feature_count}
                  </div>
                  <div className="mb-3">
                    <strong>Subfeatures:</strong> {featureGraph.subfeature_count}
                  </div>
                  <div className="mb-3">
                    <strong>Connections:</strong> {featureGraph.connections.length}
                  </div>
                </>
              )}
              <hr />
              <div className="text-muted small">
                <p className="mb-2">
                  <i className="bi bi-cursor me-1"></i>
                  <strong>Hover</strong> a feature to see connections
                </p>
                <p className="mb-2">
                  <i className="bi bi-hand-index me-1"></i>
                  <strong>Drag</strong> nodes to rearrange
                </p>
                <p className="mb-2">
                  <i className="bi bi-mouse me-1"></i>
                  <strong>Scroll</strong> to zoom in/out
                </p>
                <p className="mb-0">
                  <i className="bi bi-zoom-in me-1"></i>
                  Labels appear at <strong>0.5x</strong> zoom
                </p>
              </div>
            </Card.Body>
          </Card>

          {/* Legend */}
          <Card className="mb-4">
            <Card.Header>
              <i className="bi bi-palette me-2"></i>
              Legend
            </Card.Header>
            <Card.Body>
              <div className="d-flex align-items-center mb-2">
                <div
                  className="rounded-circle me-2"
                  style={{
                    width: 20,
                    height: 20,
                    backgroundColor: '#6366f1',
                    border: '2px solid #4338ca',
                  }}
                ></div>
                <span>Feature (larger node)</span>
              </div>
              <div className="d-flex align-items-center mb-3">
                <div
                  className="rounded-circle me-2"
                  style={{
                    width: 12,
                    height: 12,
                    backgroundColor: '#94a3b8',
                  }}
                ></div>
                <span>Subfeature (smaller node)</span>
              </div>
              <div className="d-flex align-items-center mb-2">
                <div
                  className="me-2"
                  style={{
                    width: 30,
                    height: 2,
                    backgroundColor: '#14b8a6',
                  }}
                ></div>
                <span className="small">Parent-child link</span>
              </div>
              <div className="d-flex align-items-center">
                <div
                  className="me-2"
                  style={{
                    width: 30,
                    height: 4,
                    backgroundColor: '#f97316',
                    opacity: 0.6,
                  }}
                ></div>
                <span className="small">Shared members (hover to see)</span>
              </div>
            </Card.Body>
          </Card>

          {/* Repository Selector */}
          <Card>
            <Card.Header>
              <i className="bi bi-folder me-2"></i>
              Repository
            </Card.Header>
            <Card.Body>
              <Form.Select
                value={`${scope}/${repository}`}
                onChange={(e) => {
                  const [newScope, newRepo] = e.target.value.split('/')
                  navigate(`/feature-graph/${newScope}/${newRepo}`)
                }}
              >
                {repos?.map((repo) => (
                  <option key={`${repo.scope}/${repo.name}`} value={`${repo.scope}/${repo.name}`}>
                    {repo.scope}/{repo.name}
                  </option>
                ))}
              </Form.Select>
            </Card.Body>
          </Card>
        </Col>

        {/* Visualization */}
        <Col md={9}>
          <Card className="feature-graph-container">
            <Card.Body style={{ height: '75vh', padding: 0 }}>
              {isLoading ? (
                <div className="d-flex justify-content-center align-items-center h-100">
                  <div className="text-center">
                    <Spinner animation="border" className="mb-2" />
                    <p className="text-muted">Loading feature graph...</p>
                  </div>
                </div>
              ) : error ? (
                <Alert variant="danger" className="m-3">
                  Failed to load feature graph: {(error as Error).message}
                </Alert>
              ) : featureGraph && featureGraph.nodes.length > 0 ? (
                <>
                  <div
                    className="d-flex justify-content-between align-items-center px-3 py-2"
                    style={{ borderBottom: '1px solid #dee2e6' }}
                  >
                    <small className="text-muted">
                      {featureGraph.feature_count} features, {featureGraph.subfeature_count} subfeatures
                    </small>
                    <Badge bg="info">
                      {featureGraph.connections.length} connections
                    </Badge>
                  </div>
                  <div style={{ height: 'calc(100% - 40px)' }}>
                    <FeatureGraphVisualization
                      nodes={featureGraph.nodes}
                      connections={featureGraph.connections}
                      onNodeClick={handleNodeClick}
                    />
                  </div>
                </>
              ) : (
                <div className="d-flex justify-content-center align-items-center h-100">
                  <div className="text-center">
                    <i className="bi bi-diagram-3 display-1 text-muted mb-3 d-block"></i>
                    <h5>No features found</h5>
                    <p className="text-muted">
                      This repository doesn't have any features extracted yet.
                      <br />
                      Run the feature extraction pipeline to generate features.
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

export default FeatureGraph
