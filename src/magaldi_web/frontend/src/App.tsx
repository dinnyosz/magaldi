import { Routes, Route, Link, useLocation } from 'react-router-dom'
import { Container, Nav, Navbar, Button } from 'react-bootstrap'
import { useDarkMode } from './hooks/useDarkMode'
import Dashboard from './pages/Dashboard'
import Search from './pages/Search'
import Repository from './pages/Repository'
import Element from './pages/Element'
import Explorer from './pages/Explorer'
import VectorMap from './pages/VectorMap'
import Glossary from './pages/Glossary'
import Admin from './pages/Admin'
import DeadCode from './pages/DeadCode'
import EntryPoints from './pages/EntryPoints'
import CallExplorer from './pages/CallExplorer'
import DependencyGraph from './pages/DependencyGraph'

function App() {
  const location = useLocation()
  const [theme, toggleTheme] = useDarkMode()

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
            <Button
              variant="outline-light"
              size="sm"
              onClick={toggleTheme}
              className="ms-2"
              aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
            >
              <i className={`bi ${theme === 'light' ? 'bi-moon-fill' : 'bi-sun-fill'}`}></i>
            </Button>
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
          <Route path="/repos/:scope/:repo/dead-code" element={<DeadCode />} />
          <Route path="/repos/:scope/:repo/entry-points" element={<EntryPoints />} />
          <Route path="/repos/:scope/:repo/call-explorer" element={<CallExplorer />} />
          <Route path="/repos/:scope/:repo/dependency-graph" element={<DependencyGraph />} />
          <Route path="/element/:elementId" element={<Element />} />
          <Route path="/vector-map" element={<VectorMap />} />
          <Route path="/vector-map/:scope/:repository" element={<VectorMap />} />
          <Route path="/glossary" element={<Glossary />} />
          <Route path="/glossary/:scope/:repository" element={<Glossary />} />
          <Route path="/admin" element={<Admin />} />
        </Routes>
      </Container>

      <footer className="bg-body-tertiary py-3 mt-auto border-top">
        <Container fluid>
          <small className="text-body-secondary">
            Magaldi - Code Discovery Engine
          </small>
        </Container>
      </footer>
    </div>
  )
}

export default App
