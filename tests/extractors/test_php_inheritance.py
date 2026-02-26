"""Tests for PHP inheritance and trait extraction."""

import pytest
import tree_sitter_php as ts_php
from tree_sitter import Language, Parser

from magaldi_core.extractors.php import (
    extract_php_base_class,
)


class TestPHPInheritance:
    """Test extraction of PHP inheritance patterns."""

    @pytest.fixture
    def php_parser(self):
        """Create a PHP parser."""
        php_lang = Language(ts_php.language_php())
        return Parser(php_lang)

    def test_extract_extends(self, php_parser):
        """Test extraction of extends clause."""
        code = """<?php
class UserService extends BaseService {
    private $repo;
}
"""
        tree = php_parser.parse(code.encode())
        class_node = None
        for node in tree.root_node.children:
            if node.type == "class_declaration":
                class_node = node
                break

        assert class_node is not None
        bases = extract_php_base_class(class_node)
        assert "BaseService" in bases

    def test_extract_implements_single(self, php_parser):
        """Test extraction of single implements clause."""
        code = """<?php
class UserRepository implements Repository {
    public function find(int $id): array {
        return [];
    }
}
"""
        tree = php_parser.parse(code.encode())
        class_node = None
        for node in tree.root_node.children:
            if node.type == "class_declaration":
                class_node = node
                break

        assert class_node is not None
        bases = extract_php_base_class(class_node)
        assert "Repository" in bases

    def test_extract_implements_multiple(self, php_parser):
        """Test extraction of multiple implements clause."""
        code = """<?php
class CachedRepository implements Repository, Cacheable, Loggable {
    public function find(int $id): array {
        return [];
    }
}
"""
        tree = php_parser.parse(code.encode())
        class_node = None
        for node in tree.root_node.children:
            if node.type == "class_declaration":
                class_node = node
                break

        assert class_node is not None
        bases = extract_php_base_class(class_node)
        assert "Repository" in bases
        assert "Cacheable" in bases
        assert "Loggable" in bases

    def test_extract_extends_and_implements(self, php_parser):
        """Test extraction of both extends and implements."""
        code = """<?php
class AdminService extends BaseService implements ServiceInterface, Loggable {
    public function doSomething(): void {}
}
"""
        tree = php_parser.parse(code.encode())
        class_node = None
        for node in tree.root_node.children:
            if node.type == "class_declaration":
                class_node = node
                break

        assert class_node is not None
        bases = extract_php_base_class(class_node)
        assert "BaseService" in bases
        assert "ServiceInterface" in bases
        assert "Loggable" in bases

    def test_extract_trait_usage(self, php_parser):
        """Test extraction of trait usage."""
        code = """<?php
class MyService {
    use LoggerTrait;
    use CacheableTrait;

    public function doWork(): void {}
}
"""
        tree = php_parser.parse(code.encode())
        class_node = None
        for node in tree.root_node.children:
            if node.type == "class_declaration":
                class_node = node
                break

        assert class_node is not None
        bases = extract_php_base_class(class_node)
        assert "LoggerTrait" in bases
        assert "CacheableTrait" in bases

    def test_extract_combined_inheritance(self, php_parser):
        """Test extraction of extends + implements + traits."""
        code = """<?php
class UserController extends Controller implements Requestable {
    use LoggerTrait;

    public function index(): void {}
}
"""
        tree = php_parser.parse(code.encode())
        class_node = None
        for node in tree.root_node.children:
            if node.type == "class_declaration":
                class_node = node
                break

        assert class_node is not None
        bases = extract_php_base_class(class_node)
        assert "Controller" in bases
        assert "Requestable" in bases
        assert "LoggerTrait" in bases
