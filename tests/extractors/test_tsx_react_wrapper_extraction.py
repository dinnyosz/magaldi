"""Tests for React wrapper component extraction (memo, forwardRef, lazy)."""

import pytest
import tree_sitter_typescript as ts_tsx
from tree_sitter import Language, Parser

from magaldi_core.extractors.javascript import extract_javascript_elements


class TestReactWrapperExtraction:
    """Test extraction of React wrapper patterns."""

    @pytest.fixture
    def tsx_parser(self):
        """Create a TSX parser."""
        tsx_lang = Language(ts_tsx.language_tsx())
        return Parser(tsx_lang)

    def test_memo_wrapper_extraction(self, tsx_parser):
        """Test extraction of memo-wrapped components."""
        code = """
import { memo } from 'react';

const Greeting = memo(function Greeting({ name }) {
  return <h3>Hello {name}</h3>;
});
"""
        tree = tsx_parser.parse(code.encode())
        elements = extract_javascript_elements(tree, code.split("\n"))

        # Find the Greeting component
        greeting = next((e for e in elements if e.name == "Greeting"), None)
        assert greeting is not None
        assert greeting.element_type == "function"
        assert greeting.decorators == ["memo"]

    def test_forwardref_wrapper_extraction(self, tsx_parser):
        """Test extraction of forwardRef-wrapped components."""
        code = """
import { forwardRef } from 'react';

const MyInput = forwardRef(function MyInput(props, ref) {
  return <input ref={ref} />;
});
"""
        tree = tsx_parser.parse(code.encode())
        elements = extract_javascript_elements(tree, code.split("\n"))

        # Find the MyInput component
        my_input = next((e for e in elements if e.name == "MyInput"), None)
        assert my_input is not None
        assert my_input.element_type == "function"
        assert my_input.decorators == ["forwardRef"]

    def test_react_dot_memo_extraction(self, tsx_parser):
        """Test extraction of React.memo-wrapped components."""
        code = """
import React from 'react';

const Greeting = React.memo(function Greeting({ name }) {
  return <h3>Hello {name}</h3>;
});
"""
        tree = tsx_parser.parse(code.encode())
        elements = extract_javascript_elements(tree, code.split("\n"))

        greeting = next((e for e in elements if e.name == "Greeting"), None)
        assert greeting is not None
        assert greeting.element_type == "function"
        assert greeting.decorators == ["React.memo"]

    def test_react_dot_forwardref_extraction(self, tsx_parser):
        """Test extraction of React.forwardRef-wrapped components."""
        code = """
import React from 'react';

const MyInput = React.forwardRef(function MyInput(props, ref) {
  return <input ref={ref} />;
});
"""
        tree = tsx_parser.parse(code.encode())
        elements = extract_javascript_elements(tree, code.split("\n"))

        my_input = next((e for e in elements if e.name == "MyInput"), None)
        assert my_input is not None
        assert my_input.element_type == "function"
        assert my_input.decorators == ["React.forwardRef"]

    def test_lazy_wrapper_extraction(self, tsx_parser):
        """Test extraction of lazy-wrapped components."""
        code = """
import { lazy } from 'react';

const LazyComponent = lazy(() => import('./Component'));
"""
        tree = tsx_parser.parse(code.encode())
        elements = extract_javascript_elements(tree, code.split("\n"))

        lazy_comp = next((e for e in elements if e.name == "LazyComponent"), None)
        assert lazy_comp is not None
        assert lazy_comp.element_type == "function"
        assert lazy_comp.decorators == ["lazy"]

    def test_regular_arrow_not_wrapped(self, tsx_parser):
        """Test that regular arrow functions don't have decorators."""
        code = """
const ArrowComponent = ({ x }) => <div>{x}</div>;
"""
        tree = tsx_parser.parse(code.encode())
        elements = extract_javascript_elements(tree, code.split("\n"))

        arrow = next((e for e in elements if e.name == "ArrowComponent"), None)
        assert arrow is not None
        assert arrow.element_type == "function"
        assert arrow.decorators is None

    def test_non_react_wrapper_not_extracted(self, tsx_parser):
        """Test that non-React wrapper calls are not extracted as components."""
        code = """
const result = someFunction(function inner() {
  return 42;
});
"""
        tree = tsx_parser.parse(code.encode())
        elements = extract_javascript_elements(tree, code.split("\n"))

        # Should not extract 'result' as a function since someFunction is not a React wrapper
        # It may be extracted as a variable, but never as a function
        result = next((e for e in elements if e.name == "result"), None)
        if result is not None:
            assert result.element_type != "function", \
                f"'result' should not be extracted as function, got: {result.element_type}"


class TestReactHookDetection:
    """Test detection of React custom hooks."""

    @pytest.fixture
    def tsx_parser(self):
        """Create a TSX parser."""
        tsx_lang = Language(ts_tsx.language_tsx())
        return Parser(tsx_lang)

    def test_custom_hook_function_declaration(self, tsx_parser):
        """Test that function declarations starting with 'use' are marked as hooks."""
        code = """
function useCounter(initialValue = 0) {
  const [count, setCount] = useState(initialValue);
  return count;
}
"""
        tree = tsx_parser.parse(code.encode())
        elements = extract_javascript_elements(tree, code.split("\n"))

        hook = next((e for e in elements if e.name == "useCounter"), None)
        assert hook is not None
        assert hook.decorators == ["hook"]

    def test_custom_hook_arrow_function(self, tsx_parser):
        """Test that arrow functions starting with 'use' are marked as hooks."""
        code = """
const useWindowSize = () => {
  const [size, setSize] = useState({ width: 0, height: 0 });
  return size;
};
"""
        tree = tsx_parser.parse(code.encode())
        elements = extract_javascript_elements(tree, code.split("\n"))

        hook = next((e for e in elements if e.name == "useWindowSize"), None)
        assert hook is not None
        assert hook.decorators == ["hook"]

    def test_regular_function_not_hook(self, tsx_parser):
        """Test that regular functions are not marked as hooks."""
        code = """
function calculateTotal(items) {
  return items.reduce((sum, item) => sum + item.price, 0);
}
"""
        tree = tsx_parser.parse(code.encode())
        elements = extract_javascript_elements(tree, code.split("\n"))

        func = next((e for e in elements if e.name == "calculateTotal"), None)
        assert func is not None
        assert func.decorators is None

    def test_useless_not_hook(self, tsx_parser):
        """Test that 'useless' (lowercase after 'use') is not a hook."""
        code = """
function useless() {
  return null;
}
"""
        tree = tsx_parser.parse(code.encode())
        elements = extract_javascript_elements(tree, code.split("\n"))

        func = next((e for e in elements if e.name == "useless"), None)
        assert func is not None
        assert func.decorators is None

    def test_use_only_not_hook(self, tsx_parser):
        """Test that 'use' alone is not a hook (too short)."""
        code = """
function use() {
  return null;
}
"""
        tree = tsx_parser.parse(code.encode())
        elements = extract_javascript_elements(tree, code.split("\n"))

        func = next((e for e in elements if e.name == "use"), None)
        assert func is not None
        assert func.decorators is None

    def test_user_not_hook(self, tsx_parser):
        """Test that 'user' is not a hook."""
        code = """
function user() {
  return { name: 'John' };
}
"""
        tree = tsx_parser.parse(code.encode())
        elements = extract_javascript_elements(tree, code.split("\n"))

        func = next((e for e in elements if e.name == "user"), None)
        assert func is not None
        assert func.decorators is None
