"""Tests for HTML parser (src/magaldi_core/parsers/html.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from magaldi_core.parsers.html import HtmlParser

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def parser():
    return HtmlParser()


@pytest.fixture
def file_info():
    fi = MagicMock()
    fi.relative_path = "templates/index.html"
    fi.absolute_path = Path("/tmp/test.html")
    fi.language = "html"
    return fi


@pytest.fixture
def htm_file_info():
    fi = MagicMock()
    fi.relative_path = "pages/about.htm"
    fi.absolute_path = Path("/tmp/test.htm")
    fi.language = "html"
    return fi


def _parse(parser, file_info, html: str):
    return parser.parse(html, file_info, "test", "repo", "user")


# =============================================================================
# FILE ELEMENT BASICS
# =============================================================================


class TestFileElement:
    def test_file_element_created(self, parser, file_info):
        elements = _parse(parser, file_info, "<h1>Hello</h1><p>World</p>")
        assert len(elements) == 1
        file_elem = elements[0]
        assert file_elem.element_type == "file"
        assert file_elem.name == "index.html"
        assert file_elem.language == "html"
        assert file_elem.level == 0

    def test_element_id_format(self, parser, file_info):
        elements = _parse(parser, file_info, "<p>test</p>")
        eid = elements[0].element_id
        assert "test:repo:user:templates/index.html:file:" in eid

    def test_htm_extension(self, parser, htm_file_info):
        elements = _parse(parser, htm_file_info, "<p>About us</p>")
        assert elements[0].name == "about.htm"
        assert elements[0].language == "html"

    def test_empty_html(self, parser, file_info):
        elements = _parse(parser, file_info, "")
        assert len(elements) == 1
        assert elements[0].raw_code == ""

    def test_whitespace_only_html(self, parser, file_info):
        elements = _parse(parser, file_info, "   \n\n  ")
        assert len(elements) == 1
        assert elements[0].raw_code == ""


# =============================================================================
# HTML TO MARKDOWN CONVERSION
# =============================================================================


class TestHtmlToMarkdown:
    def test_headings_converted(self, parser, file_info):
        html = "<h1>Title</h1><h2>Subtitle</h2><p>Content</p>"
        elements = _parse(parser, file_info, html)
        raw = elements[0].raw_code
        assert "# Title" in raw
        assert "## Subtitle" in raw
        assert "Content" in raw

    def test_paragraphs_preserved(self, parser, file_info):
        html = "<p>First paragraph.</p><p>Second paragraph.</p>"
        elements = _parse(parser, file_info, html)
        raw = elements[0].raw_code
        assert "First paragraph." in raw
        assert "Second paragraph." in raw

    def test_links_converted(self, parser, file_info):
        html = '<a href="https://example.com">Example</a>'
        elements = _parse(parser, file_info, html)
        assert "[Example](https://example.com)" in elements[0].raw_code

    def test_bold_italic_converted(self, parser, file_info):
        html = "<strong>bold</strong> and <em>italic</em>"
        elements = _parse(parser, file_info, html)
        raw = elements[0].raw_code
        assert "**bold**" in raw
        assert "*italic*" in raw

    def test_lists_converted(self, parser, file_info):
        html = "<ul><li>One</li><li>Two</li></ul>"
        elements = _parse(parser, file_info, html)
        raw = elements[0].raw_code
        assert "One" in raw
        assert "Two" in raw

    def test_tables_converted(self, parser, file_info):
        html = (
            "<table><tr><th>Name</th><th>Value</th></tr>"
            "<tr><td>A</td><td>1</td></tr></table>"
        )
        elements = _parse(parser, file_info, html)
        raw = elements[0].raw_code
        assert "Name" in raw
        assert "Value" in raw

    def test_script_and_style_kept(self, parser, file_info):
        """Script and style content is kept (user chose no stripping)."""
        html = "<style>body{color:red}</style><script>alert(1)</script><p>Text</p>"
        elements = _parse(parser, file_info, html)
        raw = elements[0].raw_code
        # The converted markdown should contain the text content
        assert "Text" in raw

    def test_excessive_newlines_collapsed(self, parser, file_info):
        html = "<p>A</p>\n\n\n\n\n<p>B</p>"
        elements = _parse(parser, file_info, html)
        assert "\n\n\n" not in elements[0].raw_code

    def test_full_page_html(self, parser, file_info):
        """Full HTML document with doctype, head, body."""
        html = """<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
<h1>Welcome</h1>
<p>This is a test page with <a href="/link">a link</a>.</p>
<h2>Features</h2>
<ul>
<li>Feature one</li>
<li>Feature two</li>
</ul>
</body>
</html>"""
        elements = _parse(parser, file_info, html)
        raw = elements[0].raw_code
        assert "# Welcome" in raw
        assert "## Features" in raw
        assert "a link" in raw
        assert "Feature one" in raw

    def test_raw_code_is_markdown_not_html(self, parser, file_info):
        """Verify raw_code contains markdown, not original HTML tags."""
        html = "<div><h1>Title</h1><p>Para</p></div>"
        elements = _parse(parser, file_info, html)
        raw = elements[0].raw_code
        assert "<div>" not in raw
        assert "<h1>" not in raw
        assert "<p>" not in raw
        assert "# Title" in raw


# =============================================================================
# DOCUMENT SECTIONS EXTRACTION
# =============================================================================


class TestDocumentSections:
    def test_sections_extracted_from_headings(self, parser, file_info):
        html = "<h1>Introduction</h1><p>Text</p><h2>Details</h2><p>More</p>"
        elements = _parse(parser, file_info, html)
        sections = elements[0].document_sections
        assert sections is not None
        assert len(sections) >= 2
        titles = [s["title"] for s in sections]
        assert "Introduction" in titles
        assert "Details" in titles

    def test_section_levels(self, parser, file_info):
        html = "<h1>H1</h1><h2>H2</h2><h3>H3</h3>"
        elements = _parse(parser, file_info, html)
        sections = elements[0].document_sections
        levels = [s["level"] for s in sections]
        assert levels == [1, 2, 3]

    def test_no_sections_for_plain_text(self, parser, file_info):
        html = "<p>Just a paragraph without headings.</p>"
        elements = _parse(parser, file_info, html)
        sections = elements[0].document_sections
        # Should be empty list or None
        assert not sections

    def test_no_sections_for_empty(self, parser, file_info):
        elements = _parse(parser, file_info, "")
        assert not elements[0].document_sections


# =============================================================================
# CSS CLASSES AND IDS EXTRACTION
# =============================================================================


class TestCssClassesAndIds:
    def test_classes_extracted(self, parser, file_info):
        html = '<div class="container"><p class="text-lg bold">Hi</p></div>'
        elements = _parse(parser, file_info, html)
        metrics = elements[0].code_metrics
        assert metrics is not None
        assert "css_classes" in metrics
        assert "container" in metrics["css_classes"]
        assert "text-lg" in metrics["css_classes"]
        assert "bold" in metrics["css_classes"]

    def test_ids_extracted(self, parser, file_info):
        html = '<div id="root"><section id="features">content</section></div>'
        elements = _parse(parser, file_info, html)
        metrics = elements[0].code_metrics
        assert metrics is not None
        assert "css_ids" in metrics
        assert "root" in metrics["css_ids"]
        assert "features" in metrics["css_ids"]

    def test_classes_and_ids_together(self, parser, file_info):
        html = '<div id="app" class="wrapper"><nav class="sidebar">menu</nav></div>'
        elements = _parse(parser, file_info, html)
        metrics = elements[0].code_metrics
        assert "css_classes" in metrics
        assert "css_ids" in metrics
        assert "wrapper" in metrics["css_classes"]
        assert "sidebar" in metrics["css_classes"]
        assert "app" in metrics["css_ids"]

    def test_duplicate_classes_deduplicated(self, parser, file_info):
        html = '<div class="btn"><a class="btn">link</a></div>'
        elements = _parse(parser, file_info, html)
        classes = elements[0].code_metrics["css_classes"]
        assert classes.count("btn") == 1

    def test_classes_sorted(self, parser, file_info):
        html = '<div class="zebra alpha mango">text</div>'
        elements = _parse(parser, file_info, html)
        classes = elements[0].code_metrics["css_classes"]
        assert classes == sorted(classes)

    def test_no_metrics_for_plain_html(self, parser, file_info):
        html = "<p>No classes or IDs here</p>"
        elements = _parse(parser, file_info, html)
        assert not elements[0].code_metrics

    def test_no_metrics_for_empty(self, parser, file_info):
        elements = _parse(parser, file_info, "")
        assert not elements[0].code_metrics

    def test_tailwind_classes_extracted(self, parser, file_info):
        """Real-world Tailwind CSS classes are captured."""
        html = '<div class="flex items-center space-x-4 bg-blue-500 hover:bg-blue-700 p-4">content</div>'
        elements = _parse(parser, file_info, html)
        classes = elements[0].code_metrics["css_classes"]
        assert "flex" in classes
        assert "bg-blue-500" in classes
        assert "hover:bg-blue-700" in classes

    def test_classes_inlined_near_element(self, parser, file_info):
        """CSS classes appear inline next to their element, not at the top."""
        html = '<div id="app" class="container wrapper"><p>Hello</p></div>'
        elements = _parse(parser, file_info, html)
        raw = elements[0].raw_code
        # Annotations use CSS-selector syntax {#id .class}
        assert "{#app .container .wrapper}" in raw
        assert "Hello" in raw

    def test_link_classes_inlined(self, parser, file_info):
        """Link classes appear right after the markdown link."""
        html = '<a href="/home" class="nav-link active">Home</a>'
        elements = _parse(parser, file_info, html)
        raw = elements[0].raw_code
        assert "[Home](/home)" in raw
        assert "{.nav-link .active}" in raw

    def test_list_item_classes_inlined(self, parser, file_info):
        """List item classes appear on the same line as the item."""
        html = '<ul><li class="nav-item">Item</li></ul>'
        elements = _parse(parser, file_info, html)
        raw = elements[0].raw_code
        assert "Item" in raw
        assert "{.nav-item}" in raw

    def test_no_annotations_for_classless_html(self, parser, file_info):
        """Plain HTML without classes/IDs has no annotation markers."""
        html = "<p>Just text</p>"
        elements = _parse(parser, file_info, html)
        raw = elements[0].raw_code
        assert "{" not in raw
        assert "}" not in raw


# =============================================================================
# SIZE REDUCTION
# =============================================================================


class TestSizeReduction:
    def test_markdown_is_smaller_than_html(self, parser, file_info):
        """Converted markdown should be significantly smaller than raw HTML."""
        html = """
        <div class="container mx-auto px-4">
            <header class="bg-blue-500 text-white p-6">
                <h1 class="text-3xl font-bold">Welcome to Our Site</h1>
                <nav class="flex space-x-4 mt-2">
                    <a href="/" class="hover:underline">Home</a>
                    <a href="/about" class="hover:underline">About</a>
                    <a href="/contact" class="hover:underline">Contact</a>
                </nav>
            </header>
            <main class="py-8">
                <section class="mb-6">
                    <h2 class="text-2xl mb-4">Our Services</h2>
                    <p class="text-gray-700">We provide excellent services.</p>
                    <ul class="list-disc ml-6">
                        <li>Service one</li>
                        <li>Service two</li>
                        <li>Service three</li>
                    </ul>
                </section>
            </main>
        </div>
        """
        elements = _parse(parser, file_info, html)
        md_size = len(elements[0].raw_code)
        html_size = len(html)
        # Markdown should be at least 40% smaller than raw HTML
        assert md_size < html_size * 0.6, (
            f"Markdown ({md_size} chars) should be much smaller than HTML ({html_size} chars)"
        )


# =============================================================================
# DISCOVERY INTEGRATION
# =============================================================================


class TestDiscovery:
    def test_html_extension_supported(self):
        from magaldi_core.discovery import SUPPORTED_EXTENSIONS

        assert ".html" in SUPPORTED_EXTENSIONS
        assert SUPPORTED_EXTENSIONS[".html"] == "html"

    def test_htm_extension_supported(self):
        from magaldi_core.discovery import SUPPORTED_EXTENSIONS

        assert ".htm" in SUPPORTED_EXTENSIONS
        assert SUPPORTED_EXTENSIONS[".htm"] == "html"


class TestParserRegistry:
    def test_html_parser_registered(self):
        from magaldi_core.code_parser import get_parser

        parser = get_parser("html")
        assert parser is not None
        assert isinstance(parser, HtmlParser)
