"""Tests for PHP 8+ modern feature extraction."""

import pytest
from tree_sitter import Language, Parser

import tree_sitter_php as ts_php

from magaldi_core.extractors.php import (
    extract_php_elements,
    extract_php_enum_members,
)


class TestPhpEnumExtraction:
    """Test extraction of PHP 8.1 enums."""

    @pytest.fixture
    def php_parser(self):
        """Create a PHP parser."""
        php_lang = Language(ts_php.language_php())
        return Parser(php_lang)

    def test_backed_enum_extraction(self, php_parser):
        """Test extraction of backed enum with string type."""
        code = """<?php
enum Suit: string
{
    case Hearts = 'H';
    case Diamonds = 'D';
}
"""
        tree = php_parser.parse(code.encode())
        elements = extract_php_elements(tree, code.split("\n"))

        suit = next((e for e in elements if e.name == "Suit"), None)
        assert suit is not None
        assert suit.element_type == "enum"
        assert suit.return_type == "string"  # backing type

    def test_pure_enum_extraction(self, php_parser):
        """Test extraction of pure enum without backing type."""
        code = """<?php
enum Size
{
    case Small;
    case Medium;
    case Large;
}
"""
        tree = php_parser.parse(code.encode())
        elements = extract_php_elements(tree, code.split("\n"))

        size = next((e for e in elements if e.name == "Size"), None)
        assert size is not None
        assert size.element_type == "enum"
        assert size.return_type is None  # no backing type

    def test_enum_member_extraction(self, php_parser):
        """Test extraction of methods and cases from enum."""
        code = """<?php
enum Suit: string
{
    case Hearts = 'H';
    case Diamonds = 'D';

    public static function values(): array
    {
        return array_column(self::cases(), 'value');
    }
}
"""
        tree = php_parser.parse(code.encode())
        elements = extract_php_elements(tree, code.split("\n"))

        suit = next((e for e in elements if e.name == "Suit"), None)
        assert suit is not None

        methods, cases = extract_php_enum_members(suit.node, code.split("\n"))

        # Check methods
        assert len(methods) == 1
        assert methods[0].name == "values"
        assert "static" in methods[0].decorators

        # Check cases
        assert len(cases) == 2
        case_names = [c.name for c in cases]
        assert "Hearts" in case_names
        assert "Diamonds" in case_names


class TestPhpArrowFunctionExtraction:
    """Test extraction of PHP 7.4+ arrow functions."""

    @pytest.fixture
    def php_parser(self):
        """Create a PHP parser."""
        php_lang = Language(ts_php.language_php())
        return Parser(php_lang)

    def test_arrow_function_extraction(self, php_parser):
        """Test extraction of arrow function."""
        code = """<?php
$multiply = fn($x, $y) => $x * $y;
"""
        tree = php_parser.parse(code.encode())
        elements = extract_php_elements(tree, code.split("\n"))

        multiply = next((e for e in elements if e.name == "multiply"), None)
        assert multiply is not None
        assert multiply.element_type == "function"
        assert multiply.decorators == ["arrow"]
        assert "fn($x, $y)" in multiply.signature

    def test_arrow_function_with_types(self, php_parser):
        """Test extraction of typed arrow function."""
        code = """<?php
$double = fn(int $x): int => $x * 2;
"""
        tree = php_parser.parse(code.encode())
        elements = extract_php_elements(tree, code.split("\n"))

        double = next((e for e in elements if e.name == "double"), None)
        assert double is not None
        assert double.decorators == ["arrow"]


class TestPhpClosureExtraction:
    """Test extraction of PHP closures."""

    @pytest.fixture
    def php_parser(self):
        """Create a PHP parser."""
        php_lang = Language(ts_php.language_php())
        return Parser(php_lang)

    def test_closure_extraction(self, php_parser):
        """Test extraction of traditional closure."""
        code = """<?php
$add = function($a, $b) {
    return $a + $b;
};
"""
        tree = php_parser.parse(code.encode())
        elements = extract_php_elements(tree, code.split("\n"))

        add = next((e for e in elements if e.name == "add"), None)
        assert add is not None
        assert add.element_type == "function"
        assert add.decorators == ["closure"]
        assert "function($a, $b)" in add.signature

    def test_closure_with_use(self, php_parser):
        """Test extraction of closure with use clause."""
        code = """<?php
$greeting = "Hello";
$sayHello = function($name) use ($greeting) {
    return "$greeting, $name!";
};
"""
        tree = php_parser.parse(code.encode())
        elements = extract_php_elements(tree, code.split("\n"))

        say_hello = next((e for e in elements if e.name == "sayHello"), None)
        assert say_hello is not None
        assert say_hello.decorators == ["closure"]


class TestPhpReadonlyClass:
    """Test extraction of PHP 8.2 readonly classes."""

    @pytest.fixture
    def php_parser(self):
        """Create a PHP parser."""
        php_lang = Language(ts_php.language_php())
        return Parser(php_lang)

    def test_readonly_class_extraction(self, php_parser):
        """Test extraction of readonly class with modifier."""
        code = """<?php
readonly class Point
{
    public function __construct(
        public int $x,
        public int $y,
    ) {}
}
"""
        tree = php_parser.parse(code.encode())
        elements = extract_php_elements(tree, code.split("\n"))

        point = next((e for e in elements if e.name == "Point"), None)
        assert point is not None
        assert point.element_type == "class"
        assert "readonly" in point.decorators
