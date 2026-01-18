import { Routes, Route, Link, useLocation } from 'react-router-dom'
import { Container, Nav, Navbar } from 'react-bootstrap'
import Dashboard from './pages/Dashboard'
import Search from './pages/Search'
import Repository from './pages/Repository'
import Element from './pages/Element'
import Explorer from './pages/Explorer'
import VectorMap from './pages/VectorMap'
import Glossary from './pages/Glossary'
import Admin from './pages/Admin'

function App() {
  const location = useLocation()

  return (
    <div className="d-flex flex-column min-vh-100">
      <Navbar bg="dark" variant="dark" expand="lg" className="mb-3">
        <Container fluid>
          <Navbar.Brand as={Link} to="/">
            <i className="bi bi-compass me-2"></i>
            Magaldi
          </Navbar.Brand>
          <Navbar.Toggle aria-controls="main-nav" />
          <Navbar.Collapse id="main-nav">
            <Nav className="me-auto">
              <Nav.Link
                as={Link}
                to="/"
                active={location.pathname === '/'}
              >
                Dashboard
              </Nav.Link>
              <Nav.Link
                as={Link}
                to="/search"
                active={location.pathname.startsWith('/search')}
              >
                Search
              </Nav.Link>
              <Nav.Link
                as={Link}
                to="/explorer"
                active={location.pathname.startsWith('/explorer')}
              >
                Explorer
              </Nav.Link>
              <Nav.Link
                as={Link}
                to="/repos"
                active={location.pathname.startsWith('/repos')}
              >
                Repositories
              </Nav.Link>
              <Nav.Link
                as={Link}
                to="/vector-map"
                active={location.pathname.startsWith('/vector-map')}
              >
                Vector Map
              </Nav.Link>
              <Nav.Link
                as={Link}
                to="/glossary"
                active={location.pathname.startsWith('/glossary')}
              >
                Glossary
              </Nav.Link>
              <Nav.Link
                as={Link}
                to="/admin"
                active={location.pathname.startsWith('/admin')}
              >
                Admin
              </Nav.Link>
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>

      <Container fluid className="flex-grow-1 pb-4">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/search" element={<Search />} />
          <Route path="/explorer" element={<Explorer />} />
          <Route path="/repos" element={<Repository />} />
          <Route path="/repos/:scope/:repository/*" element={<Repository />} />
          <Route path="/element/:elementId" element={<Element />} />
          <Route path="/vector-map" element={<VectorMap />} />
          <Route path="/vector-map/:scope/:repository" element={<VectorMap />} />
          <Route path="/glossary" element={<Glossary />} />
          <Route path="/glossary/:scope/:repository" element={<Glossary />} />
          <Route path="/admin" element={<Admin />} />
        </Routes>
      </Container>

      <footer className="bg-light py-3 mt-auto border-top">
        <Container fluid>
          <small className="text-muted">
            Magaldi - Code Discovery Engine
          </small>
        </Container>
      </footer>
    </div>
  )
}

export default App
