"""Tests for Vue SFC parser (src/magaldi_core/parsers/vue.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from magaldi_core.parsers.sfc_splitter import SFCSection, split_sfc
from magaldi_core.parsers.vue import VueParser


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def parser():
    return VueParser()


@pytest.fixture
def file_info():
    fi = MagicMock()
    fi.relative_path = "src/components/UserCard.vue"
    fi.absolute_path = Path("/tmp/UserCard.vue")
    fi.language = "vue"
    return fi


def _parse(parser, file_info, content: str):
    return parser.parse(content, file_info, "test", "repo", "user")


# =============================================================================
# SFC SPLITTER
# =============================================================================


class TestSFCSplitter:
    def test_splits_three_sections(self):
        content = """<template>
  <div>Hello</div>
</template>

<script>
export default { name: 'Comp' }
</script>

<style>
.card { color: red; }
</style>
"""
        sections = split_sfc(content)
        tags = [s.tag for s in sections]
        assert "template" in tags
        assert "script" in tags
        assert "style" in tags

    def test_section_content_correct(self):
        content = """<template>
  <p>Hello</p>
</template>

<script>
const x = 1;
</script>
"""
        sections = split_sfc(content)
        template = next(s for s in sections if s.tag == "template")
        script = next(s for s in sections if s.tag == "script")
        assert "<p>Hello</p>" in template.content
        assert "const x = 1" in script.content

    def test_section_line_offsets(self):
        content = """<template>
  <div>Hello</div>
</template>

<script>
const x = 1;
</script>
"""
        sections = split_sfc(content)
        template = next(s for s in sections if s.tag == "template")
        script = next(s for s in sections if s.tag == "script")
        # Template inner content starts on line 1 (after <template> tag on line 1)
        assert template.line_start == 1
        # Script inner content starts after the <script> tag
        assert script.line_start >= 5

    def test_attrs_parsed(self):
        content = '<script setup lang="ts">\nconst x = 1;\n</script>'
        sections = split_sfc(content)
        script = sections[0]
        assert script.attrs.get("setup") == ""
        assert script.attrs.get("lang") == "ts"

    def test_scoped_style(self):
        content = '<style scoped lang="scss">\n.card { }\n</style>'
        sections = split_sfc(content)
        style = sections[0]
        assert style.attrs.get("scoped") == ""
        assert style.attrs.get("lang") == "scss"

    def test_empty_file(self):
        sections = split_sfc("")
        assert sections == []

    def test_no_matching_sections(self):
        sections = split_sfc("<div>Just HTML</div>")
        assert sections == []


# =============================================================================
# FILE ELEMENT BASICS
# =============================================================================


class TestFileElement:
    def test_file_element_created(self, parser, file_info):
        content = """<template>
  <div class="card"><h1>Hello</h1></div>
</template>

<script>
export default { name: 'UserCard' }
</script>
"""
        elements = _parse(parser, file_info, content)
        file_elem = elements[0]
        assert file_elem.element_type == "file"
        assert file_elem.language == "vue"
        assert file_elem.name == "UserCard.vue"

    def test_template_becomes_raw_code(self, parser, file_info):
        content = """<template>
  <div class="container">
    <h1>Title</h1>
    <p>Description</p>
  </div>
</template>
"""
        elements = _parse(parser, file_info, content)
        raw = elements[0].raw_code
        assert "# Title" in raw
        assert "Description" in raw

    def test_css_classes_extracted_from_template(self, parser, file_info):
        content = """<template>
  <div class="card bg-white shadow">
    <nav id="main-nav" class="navbar">Links</nav>
  </div>
</template>
"""
        elements = _parse(parser, file_info, content)
        metrics = elements[0].code_metrics
        assert "css_classes" in metrics
        assert "card" in metrics["css_classes"]
        assert "bg-white" in metrics["css_classes"]
        assert "navbar" in metrics["css_classes"]
        assert "css_ids" in metrics
        assert "main-nav" in metrics["css_ids"]


# =============================================================================
# VUE TEMPLATE SYNTAX STRIPPING
# =============================================================================


class TestVueSyntaxStripping:
    def test_interpolation_kept_as_text(self, parser, file_info):
        content = "<template><p>{{ userName }}</p></template>"
        elements = _parse(parser, file_info, content)
        raw = elements[0].raw_code
        assert "userName" in raw
        assert "{{" not in raw

    def test_v_if_stripped(self, parser, file_info):
        content = '<template><p v-if="isVisible">Visible</p></template>'
        elements = _parse(parser, file_info, content)
        raw = elements[0].raw_code
        assert "v-if" not in raw
        assert "Visible" in raw

    def test_v_for_stripped(self, parser, file_info):
        content = '<template><li v-for="item in items">{{ item.name }}</li></template>'
        elements = _parse(parser, file_info, content)
        raw = elements[0].raw_code
        assert "v-for" not in raw

    def test_event_handler_stripped(self, parser, file_info):
        content = '<template><button @click="handleClick" class="btn">Click</button></template>'
        elements = _parse(parser, file_info, content)
        raw = elements[0].raw_code
        assert "@click" not in raw
        assert "Click" in raw

    def test_v_bind_converted(self, parser, file_info):
        content = '<template><img :src="imageUrl" :alt="altText"></template>'
        elements = _parse(parser, file_info, content)
        raw = elements[0].raw_code
        assert ":src" not in raw

    def test_v_model_stripped(self, parser, file_info):
        content = '<template><input v-model="searchQuery"></template>'
        elements = _parse(parser, file_info, content)
        raw = elements[0].raw_code
        assert "v-model" not in raw


# =============================================================================
# SCRIPT SECTION PARSING
# =============================================================================


class TestScriptParsing:
    def test_functions_extracted_from_script(self, parser, file_info):
        content = """<template>
  <div>Hello</div>
</template>

<script>
function greet(name) {
  return `Hello ${name}`;
}

export default {
  methods: { greet }
}
</script>
"""
        elements = _parse(parser, file_info, content)
        func_names = [e.name for e in elements if e.element_type == "function"]
        assert "greet" in func_names

    def test_arrow_functions_extracted(self, parser, file_info):
        content = """<template><div>Hello</div></template>

<script>
const formatDate = (date) => {
  return date.toISOString();
};
</script>
"""
        elements = _parse(parser, file_info, content)
        func_names = [e.name for e in elements if e.element_type == "function"]
        assert "formatDate" in func_names

    def test_typescript_script(self, parser, file_info):
        content = """<template><div>Hello</div></template>

<script lang="ts">
interface User {
  name: string;
  email: string;
}

function getUser(): User {
  return { name: 'Alice', email: 'alice@example.com' };
}
</script>
"""
        elements = _parse(parser, file_info, content)
        types = {e.element_type for e in elements if e.element_type != "file"}
        assert "interface" in types or "function" in types
        func_names = [e.name for e in elements if e.element_type == "function"]
        assert "getUser" in func_names

    def test_script_line_numbers_adjusted(self, parser, file_info):
        content = """<template>
  <div>Hello</div>
</template>

<script>
function myFunc() {
  return 42;
}
</script>
"""
        elements = _parse(parser, file_info, content)
        func = next(e for e in elements if e.name == "myFunc")
        # myFunc is on line 6 in the original file (after template + blank line + script tag)
        assert func.line_start >= 5

    def test_script_elements_have_vue_language(self, parser, file_info):
        content = """<template><div>Hi</div></template>
<script>
const x = 42;
</script>
"""
        elements = _parse(parser, file_info, content)
        for elem in elements:
            assert elem.language == "vue"


# =============================================================================
# STYLE SECTION PARSING
# =============================================================================


class TestStyleParsing:
    def test_css_rules_extracted(self, parser, file_info):
        content = """<template><div class="card">Hello</div></template>

<style>
.card {
  background: white;
  padding: 1rem;
}

.card:hover {
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
"""
        elements = _parse(parser, file_info, content)
        css_rules = [e for e in elements if e.element_type == "css_rule"]
        assert len(css_rules) >= 1

    def test_scss_style(self, parser, file_info):
        content = """<template><div>Hello</div></template>

<style lang="scss">
$primary: #333;

.card {
  color: $primary;

  &:hover {
    opacity: 0.8;
  }
}
</style>
"""
        elements = _parse(parser, file_info, content)
        # Should find at least the variable or rule
        non_file = [e for e in elements if e.element_type != "file"]
        assert len(non_file) >= 1

    def test_style_line_numbers_adjusted(self, parser, file_info):
        content = """<template>
  <div>Hello</div>
</template>

<script>
const x = 1;
</script>

<style>
.card {
  color: red;
}
</style>
"""
        elements = _parse(parser, file_info, content)
        css_rules = [e for e in elements if e.element_type == "css_rule"]
        if css_rules:
            # .card is on line 10 in original file
            assert css_rules[0].line_start >= 9


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    def test_template_only(self, parser, file_info):
        content = """<template>
  <div class="simple">Just HTML</div>
</template>
"""
        elements = _parse(parser, file_info, content)
        assert len(elements) >= 1
        assert elements[0].element_type == "file"
        assert "Just HTML" in elements[0].raw_code

    def test_empty_script(self, parser, file_info):
        content = """<template><div>Hello</div></template>
<script></script>
"""
        elements = _parse(parser, file_info, content)
        assert elements[0].element_type == "file"

    def test_empty_file(self, parser, file_info):
        elements = _parse(parser, file_info, "")
        assert len(elements) >= 1

    def test_no_template_section(self, parser, file_info):
        """Vue file with only script (no template)."""
        content = """<script>
export const API_URL = 'https://api.example.com';
</script>
"""
        elements = _parse(parser, file_info, content)
        assert elements[0].element_type == "file"
        assert elements[0].language == "vue"

    def test_multiple_script_sections(self, parser, file_info):
        """Vue 3 <script setup> + regular <script>."""
        content = """<template><div>Hello</div></template>

<script>
export default { name: 'MyComp' }
</script>

<script setup>
const count = ref(0);
</script>
"""
        elements = _parse(parser, file_info, content)
        # Both script sections should be parsed
        assert elements[0].element_type == "file"


# =============================================================================
# DISCOVERY & REGISTRY
# =============================================================================


class TestDiscovery:
    def test_vue_extension_supported(self):
        from magaldi_core.discovery import SUPPORTED_EXTENSIONS

        assert ".vue" in SUPPORTED_EXTENSIONS
        assert SUPPORTED_EXTENSIONS[".vue"] == "vue"

    def test_vue_parser_registered(self):
        from magaldi_core.code_parser import get_parser

        parser = get_parser("vue")
        assert parser is not None
        assert isinstance(parser, VueParser)
