import { useState } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  Row,
  Col,
  Card,
  ListGroup,
  Badge,
  Spinner,
  Alert,
  Breadcrumb,
} from 'react-bootstrap'
import { getRepositories, getFileTree, getFileDetail, type FileTreeNode } from '../api'

function FileTree({
  nodes,
  onSelect,
  selectedPath,
  level = 0
}: {
  nodes: FileTreeNode[]
  onSelect: (path: string) => void
  selectedPath: string | null
  level?: number
}) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set())

  const toggleExpand = (path: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(path)) {
        next.delete(path)
      } else {
        next.add(path)
      }
      return next
    })
  }

  return (
    <div style={{ paddingLeft: level > 0 ? '1rem' : 0 }}>
      {nodes.map((node) => (
        <div key={node.path}>
          <div
            className={`file-tree-item d-flex align-items-center ${
              selectedPath === node.path ? 'active' : ''
            }`}
            onClick={() => node.type === 'file' && onSelect(node.path)}
            style={{ cursor: node.type === 'file' ? 'pointer' : 'default' }}
          >
            {node.type === 'directory' && (
              <i
                className={`bi bi-chevron-${expanded.has(node.path) ? 'down' : 'right'} me-1`}
                style={{ cursor: 'pointer', width: '16px' }}
                onClick={(e) => toggleExpand(node.path, e)}
              ></i>
            )}
            {node.type === 'file' && <span style={{ width: '16px' }}></span>}
            <i
              className={`bi ${
                node.type === 'directory' ? 'bi-folder-fill text-warning' : 'bi-file-earmark-code text-info'
              } me-2`}
            ></i>
            <span className="flex-grow-1 text-truncate">{node.name}</span>
            {node.element_count !== undefined && node.element_count > 0 && (
              <Badge bg="secondary" pill className="ms-2">
                {node.element_count}
              </Badge>
            )}
          </div>
          {node.type === 'directory' && expanded.has(node.path) && node.children && (
            <FileTree
              nodes={node.children}
              onSelect={onSelect}
              selectedPath={selectedPath}
              level={level + 1}
            />
          )}
        </div>
      ))}
    </div>
  )
}

function Repository() {
  const { scope, repository } = useParams<{ scope?: string; repository?: string }>()
  const navigate = useNavigate()
  const [selectedFile, setSelectedFile] = useState<string | null>(null)

  const { data: repos, isLoading: reposLoading } = useQuery({
    queryKey: ['repositories'],
    queryFn: getRepositories,
    enabled: !scope || !repository,
  })

  const { data: fileTree, isLoading: treeLoading } = useQuery({
    queryKey: ['fileTree', scope, repository],
    queryFn: () => getFileTree(scope!, repository!),
    enabled: !!scope && !!repository,
  })

  const { data: fileDetail, isLoading: fileLoading } = useQuery({
    queryKey: ['fileDetail', scope, repository, selectedFile],
    queryFn: () => getFileDetail(scope!, repository!, selectedFile!),
    enabled: !!scope && !!repository && !!selectedFile,
  })

  // Repository list view
  if (!scope || !repository) {
    if (reposLoading) {
      return (
        <div className="text-center py-5">
          <Spinner animation="border" role="status">
            <span className="visually-hidden">Loading...</span>
          </Spinner>
        </div>
      )
    }

    return (
      <div>
        <h1 className="mb-4">Repositories</h1>
        {repos?.length ? (
          <Row>
            {repos.map((repo) => (
              <Col key={`${repo.scope}/${repo.name}`} md={4} className="mb-4">
                <Card
                  className="h-100"
                  style={{ cursor: 'pointer' }}
                  onClick={() => navigate(`/repos/${repo.scope}/${repo.name}`)}
                >
                  <Card.Body>
                    <Card.Title>
                      <i className="bi bi-folder2-open me-2 text-primary"></i>
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
            No repositories indexed yet. Run <code>magaldi parse</code> to index a repository.
          </Alert>
        )}
      </div>
    )
  }

  // Repository browser view
  return (
    <div>
      <Breadcrumb className="mb-4">
        <Breadcrumb.Item linkAs={Link} linkProps={{ to: '/repos' }}>
          Repositories
        </Breadcrumb.Item>
        <Breadcrumb.Item active>
          {scope}/{repository}
        </Breadcrumb.Item>
      </Breadcrumb>

      <Row>
        {/* File Tree */}
        <Col md={4} lg={3}>
          <Card className="mb-4">
            <Card.Header className="d-flex justify-content-between align-items-center">
              <span>
                <i className="bi bi-folder-fill me-2"></i>
                Files
              </span>
              {fileTree && (
                <small className="text-muted">
                  {fileTree.total_files} files
                </small>
              )}
            </Card.Header>
            <Card.Body className="file-tree" style={{ maxHeight: '70vh', overflowY: 'auto' }}>
              {treeLoading ? (
                <div className="text-center py-3">
                  <Spinner animation="border" size="sm" />
                </div>
              ) : fileTree?.tree?.length ? (
                <FileTree
                  nodes={fileTree.tree}
                  onSelect={setSelectedFile}
                  selectedPath={selectedFile}
                />
              ) : (
                <p className="text-muted mb-0">No files found</p>
              )}
            </Card.Body>
          </Card>
        </Col>

        {/* File Content */}
        <Col md={8} lg={9}>
          {selectedFile ? (
            <Card>
              <Card.Header className="d-flex justify-content-between align-items-center">
                <code>{selectedFile}</code>
                {fileDetail && (
                  <Badge bg="info">{fileDetail.language}</Badge>
                )}
              </Card.Header>
              <Card.Body>
                {fileLoading ? (
                  <div className="text-center py-5">
                    <Spinner animation="border" />
                  </div>
                ) : fileDetail ? (
                  <>
                    {fileDetail.elements?.length > 0 && (
                      <div className="mb-3">
                        <h6>Elements in this file:</h6>
                        <ListGroup horizontal className="flex-wrap">
                          {fileDetail.elements.map((el) => (
                            <ListGroup.Item
                              key={el.element_id}
                              action
                              as={Link}
                              to={`/element/${el.hash_id || el.element_id}`}
                              className="py-1 px-2"
                            >
                              <Badge
                                bg={
                                  el.element_type === 'class'
                                    ? 'purple'
                                    : el.element_type === 'function'
                                    ? 'primary'
                                    : el.element_type === 'method'
                                    ? 'success'
                                    : 'secondary'
                                }
                                style={
                                  el.element_type === 'class'
                                    ? { backgroundColor: '#6f42c1' }
                                    : {}
                                }
                                className="me-1"
                              >
                                {el.element_type}
                              </Badge>
                              <small>{el.name}</small>
                            </ListGroup.Item>
                          ))}
                        </ListGroup>
                      </div>
                    )}
                    <pre className="bg-light p-3 rounded" style={{ maxHeight: '60vh', overflowY: 'auto' }}>
                      <code>{fileDetail.content}</code>
                    </pre>
                  </>
                ) : (
                  <Alert variant="warning">Failed to load file</Alert>
                )}
              </Card.Body>
            </Card>
          ) : (
            <Card className="text-center py-5">
              <Card.Body>
                <i className="bi bi-file-earmark-code display-1 text-muted mb-3 d-block"></i>
                <h5>Select a file</h5>
                <p className="text-muted">
                  Click on a file in the tree to view its contents and elements.
                </p>
              </Card.Body>
            </Card>
          )}
        </Col>
      </Row>
    </div>
  )
}

export default Repository
