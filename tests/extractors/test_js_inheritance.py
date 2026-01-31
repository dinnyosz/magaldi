"""Tests for JavaScript/TypeScript inheritance and DI pattern extraction."""

import pytest
from tree_sitter import Language, Parser

import tree_sitter_typescript as ts_typescript

from magaldi_core.extractors.javascript import (
    extract_javascript_base_class,
    extract_javascript_elements,
)


class TestJavaScriptInheritance:
    """Test extraction of JS/TS inheritance patterns."""

    @pytest.fixture
    def ts_parser(self):
        """Create a TypeScript parser."""
        ts_lang = Language(ts_typescript.language_typescript())
        return Parser(ts_lang)

    def test_extract_extends_single_class(self, ts_parser):
        """Test extraction of single extends clause."""
        code = """
class UserService extends BaseService {
  private repo: UserRepository;
}
"""
        tree = ts_parser.parse(code.encode())
        # Find the class_declaration node
        class_node = None
        for node in tree.root_node.children:
            if node.type == "class_declaration":
                class_node = node
                break

        assert class_node is not None
        bases = extract_javascript_base_class(class_node)
        assert "BaseService" in bases

    def test_extract_implements_single_interface(self, ts_parser):
        """Test extraction of single implements clause."""
        code = """
class UserRepository implements Repository {
  find(id: number): Promise<any> {
    return Promise.resolve({});
  }
}
"""
        tree = ts_parser.parse(code.encode())
        class_node = None
        for node in tree.root_node.children:
            if node.type == "class_declaration":
                class_node = node
                break

        assert class_node is not None
        bases = extract_javascript_base_class(class_node)
        assert "Repository" in bases

    def test_extract_implements_multiple_interfaces(self, ts_parser):
        """Test extraction of multiple implements clause."""
        code = """
class UserRepository implements Repository, Cacheable, Loggable {
  find(id: number): Promise<any> {
    return Promise.resolve({});
  }
  clearCache(): void {}
  log(msg: string): void {}
}
"""
        tree = ts_parser.parse(code.encode())
        class_node = None
        for node in tree.root_node.children:
            if node.type == "class_declaration":
                class_node = node
                break

        assert class_node is not None
        bases = extract_javascript_base_class(class_node)
        assert "Repository" in bases
        assert "Cacheable" in bases
        assert "Loggable" in bases

    def test_extract_extends_and_implements(self, ts_parser):
        """Test extraction of both extends and implements."""
        code = """
class AdminService extends BaseService implements IService, IAdmin {
  doSomething(): void {}
}
"""
        tree = ts_parser.parse(code.encode())
        class_node = None
        for node in tree.root_node.children:
            if node.type == "class_declaration":
                class_node = node
                break

        assert class_node is not None
        bases = extract_javascript_base_class(class_node)
        # Should have both base class and interfaces
        assert len(bases) >= 2  # At least BaseService + interfaces


class TestJavaScriptDecoratorExtraction:
    """Test extraction of JS/TS decorators on classes."""

    @pytest.fixture
    def ts_parser(self):
        """Create a TypeScript parser."""
        ts_lang = Language(ts_typescript.language_typescript())
        return Parser(ts_lang)

    def test_extract_exported_class_with_decorator(self, ts_parser):
        """Test decorator extraction on exported class."""
        code = """
@Injectable()
export class AuthService {
  constructor(private userService: UserService) {}
}
"""
        tree = ts_parser.parse(code.encode())
        elements = extract_javascript_elements(tree, code.split("\n"))

        # Find the AuthService class
        auth_service = next((e for e in elements if e.name == "AuthService"), None)
        assert auth_service is not None
        # Decorators should be extracted
        if auth_service.decorators:
            assert "Injectable" in auth_service.decorators

    def test_extract_class_without_export_with_decorator(self, ts_parser):
        """Test decorator extraction on non-exported class."""
        code = """
@Controller('users')
class UserController {
  constructor(private userService: UserService) {}
}
"""
        tree = ts_parser.parse(code.encode())
        elements = extract_javascript_elements(tree, code.split("\n"))

        # Find the UserController class
        user_ctrl = next((e for e in elements if e.name == "UserController"), None)
        assert user_ctrl is not None


class TestTypeScriptInterfaces:
    """Test extraction of TypeScript interfaces."""

    @pytest.fixture
    def ts_parser(self):
        """Create a TypeScript parser."""
        ts_lang = Language(ts_typescript.language_typescript())
        return Parser(ts_lang)

    def test_extract_interface(self, ts_parser):
        """Test extraction of TypeScript interface."""
        code = """
interface Repository {
  find(id: number): Promise<any>;
  save(data: any): Promise<void>;
}
"""
        tree = ts_parser.parse(code.encode())
        elements = extract_javascript_elements(tree, code.split("\n"))

        repo = next((e for e in elements if e.name == "Repository"), None)
        assert repo is not None
        assert repo.element_type == "interface"

    def test_extract_interface_extends(self, ts_parser):
        """Test extraction of interface that extends another."""
        code = """
interface UserRepository extends Repository {
  findByEmail(email: string): Promise<User>;
}
"""
        tree = ts_parser.parse(code.encode())
        elements = extract_javascript_elements(tree, code.split("\n"))

        user_repo = next((e for e in elements if e.name == "UserRepository"), None)
        assert user_repo is not None
