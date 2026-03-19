"""Tests for Svelte SFC parser (src/magaldi_core/parsers/svelte.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from magaldi_core.parsers.svelte import SvelteParser


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def parser():
    return SvelteParser()


@pytest.fixture
def file_info():
    fi = MagicMock()
    fi.relative_path = "src/components/Counter.svelte"
    fi.absolute_path = Path("/tmp/Counter.svelte")
    fi.language = "svelte"
    return fi


def _parse(parser, file_info, content: str):
    return parser.parse(content, file_info, "test", "repo", "user")


# =============================================================================
# FILE ELEMENT BASICS
# =============================================================================


class TestFileElement:
    def test_file_element_created(self, parser, file_info):
        content = """<script>
  let count = 0;
</script>

<div class="counter">
  <h1>Count: {count}</h1>
  <button on:click={() => count++}>Increment</button>
</div>

<style>
  .counter { padding: 1rem; }
</style>
"""
        elements = _parse(parser, file_info, content)
        file_elem = elements[0]
        assert file_elem.element_type == "file"
        assert file_elem.language == "svelte"
        assert file_elem.name == "Counter.svelte"

    def test_template_extracted_as_raw_code(self, parser, file_info):
        content = """<script>
  let name = 'World';
</script>

<div class="greeting">
  <h1>Hello {name}!</h1>
  <p>Welcome to Svelte</p>
</div>
"""
        elements = _parse(parser, file_info, content)
        raw = elements[0].raw_code
        # Template HTML (outside script/style) converted to markdown
        assert "Hello" in raw
        assert "Welcome to Svelte" in raw

    def test_css_classes_extracted(self, parser, file_info):
        content = """<div class="card shadow-lg">
  <nav id="main-nav" class="navbar">Links</nav>
</div>
"""
        elements = _parse(parser, file_info, content)
        metrics = elements[0].code_metrics
        assert "css_classes" in metrics
        assert "card" in metrics["css_classes"]
        assert "shadow-lg" in metrics["css_classes"]
        assert "navbar" in metrics["css_classes"]


# =============================================================================
# SVELTE SYNTAX STRIPPING
# =============================================================================


class TestSvelteSyntaxStripping:
    def test_if_block_stripped(self, parser, file_info):
        content = """{#if isVisible}
  <p>Visible</p>
{:else}
  <p>Hidden</p>
{/if}
"""
        elements = _parse(parser, file_info, content)
        raw = elements[0].raw_code
        assert "{#if" not in raw
        assert "{:else}" not in raw
        assert "{/if}" not in raw
        assert "Visible" in raw
        assert "Hidden" in raw

    def test_each_block_stripped(self, parser, file_info):
        content = """<ul>
{#each items as item}
  <li>{item.name}</li>
{/each}
</ul>
"""
        elements = _parse(parser, file_info, content)
        raw = elements[0].raw_code
        assert "{#each" not in raw
        assert "{/each}" not in raw

    def test_expression_interpolation_kept(self, parser, file_info):
        content = "<p>{userName}</p>"
        elements = _parse(parser, file_info, content)
        raw = elements[0].raw_code
        assert "userName" in raw

    def test_await_block_stripped(self, parser, file_info):
        content = """{#await promise}
  <p>Loading...</p>
{:then data}
  <p>{data.result}</p>
{:catch error}
  <p>Error: {error.message}</p>
{/await}
"""
        elements = _parse(parser, file_info, content)
        raw = elements[0].raw_code
        assert "{#await" not in raw
        assert "{:then" not in raw
        assert "{:catch" not in raw
        assert "Loading" in raw

    def test_html_tag_stripped(self, parser, file_info):
        content = '<div>{@html rawContent}</div>'
        elements = _parse(parser, file_info, content)
        raw = elements[0].raw_code
        assert "{@html" not in raw

    def test_class_directive_converted(self, parser, file_info):
        content = '<div class:active={isActive}>Content</div>'
        elements = _parse(parser, file_info, content)
        raw = elements[0].raw_code
        assert "class:" not in raw
        assert "Content" in raw


# =============================================================================
# SCRIPT SECTION PARSING
# =============================================================================


class TestScriptParsing:
    def test_functions_extracted(self, parser, file_info):
        content = """<script>
function handleClick() {
  console.log('clicked');
}

const formatNumber = (n) => {
  return n.toLocaleString();
};
</script>

<button on:click={handleClick}>Click me</button>
"""
        elements = _parse(parser, file_info, content)
        func_names = [e.name for e in elements if e.element_type == "function"]
        assert "handleClick" in func_names
        assert "formatNumber" in func_names

    def test_typescript_script(self, parser, file_info):
        content = """<script lang="ts">
interface User {
  name: string;
  age: number;
}

function getUser(): User {
  return { name: 'Alice', age: 30 };
}
</script>

<p>Hello</p>
"""
        elements = _parse(parser, file_info, content)
        func_names = [e.name for e in elements if e.element_type == "function"]
        assert "getUser" in func_names

    def test_script_line_numbers_adjusted(self, parser, file_info):
        content = """<script>
function myFunc() {
  return 42;
}
</script>

<div>Template</div>
"""
        elements = _parse(parser, file_info, content)
        func = next((e for e in elements if e.name == "myFunc"), None)
        assert func is not None
        # myFunc starts on line 2 in original file
        assert func.line_start >= 2

    def test_script_elements_have_svelte_language(self, parser, file_info):
        content = """<script>
const x = 42;
</script>
<p>Hello</p>
"""
        elements = _parse(parser, file_info, content)
        for elem in elements:
            assert elem.language == "svelte"


# =============================================================================
# STYLE SECTION PARSING
# =============================================================================


class TestStyleParsing:
    def test_css_rules_extracted(self, parser, file_info):
        content = """<div class="card">Hello</div>

<style>
.card {
  background: white;
  padding: 1rem;
  border-radius: 8px;
}
</style>
"""
        elements = _parse(parser, file_info, content)
        css_rules = [e for e in elements if e.element_type == "css_rule"]
        assert len(css_rules) >= 1

    def test_scss_style(self, parser, file_info):
        content = """<p>Hello</p>

<style lang="scss">
$color: #333;
.card {
  color: $color;
  &:hover { opacity: 0.8; }
}
</style>
"""
        elements = _parse(parser, file_info, content)
        non_file = [e for e in elements if e.element_type != "file"]
        assert len(non_file) >= 1


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    def test_template_only_no_script(self, parser, file_info):
        content = """<div class="simple">
  <h1>Hello World</h1>
  <p>No script here</p>
</div>
"""
        elements = _parse(parser, file_info, content)
        assert len(elements) >= 1
        assert elements[0].element_type == "file"
        assert "Hello World" in elements[0].raw_code

    def test_script_only_no_template(self, parser, file_info):
        content = """<script>
export const API_URL = 'https://api.example.com';
</script>
"""
        elements = _parse(parser, file_info, content)
        assert elements[0].element_type == "file"
        assert elements[0].language == "svelte"

    def test_empty_file(self, parser, file_info):
        elements = _parse(parser, file_info, "")
        assert len(elements) >= 1

    def test_full_component(self, parser, file_info):
        """Real-world-like Svelte component."""
        content = """<script lang="ts">
  export let title: string;
  export let items: string[] = [];

  function addItem(item: string) {
    items = [...items, item];
  }

  const removeItem = (index: number) => {
    items = items.filter((_, i) => i !== index);
  };
</script>

<div class="todo-list">
  <h2>{title}</h2>
  {#each items as item, i}
    <div class="todo-item">
      <span>{item}</span>
      <button on:click={() => removeItem(i)}>Remove</button>
    </div>
  {/each}
</div>

<style>
  .todo-list { max-width: 400px; margin: 0 auto; }
  .todo-item { display: flex; justify-content: space-between; padding: 0.5rem; }
</style>
"""
        elements = _parse(parser, file_info, content)
        # File element
        assert elements[0].element_type == "file"
        assert elements[0].language == "svelte"
        # Script functions
        func_names = [e.name for e in elements if e.element_type == "function"]
        assert "addItem" in func_names
        assert "removeItem" in func_names
        # CSS rules
        css_rules = [e for e in elements if e.element_type == "css_rule"]
        assert len(css_rules) >= 1
        # Template content
        assert "todo-list" in str(elements[0].code_metrics)


# =============================================================================
# DISCOVERY & REGISTRY
# =============================================================================


class TestDiscovery:
    def test_svelte_extension_supported(self):
        from magaldi_core.discovery import SUPPORTED_EXTENSIONS

        assert ".svelte" in SUPPORTED_EXTENSIONS
        assert SUPPORTED_EXTENSIONS[".svelte"] == "svelte"

    def test_svelte_parser_registered(self):
        from magaldi_core.code_parser import get_parser

        parser = get_parser("svelte")
        assert parser is not None
        assert isinstance(parser, SvelteParser)
