"""Tests for generic TemplateParser (src/magaldi_core/parsers/template.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from magaldi_core.parsers.template import (
    TemplateParser,
    strip_template_syntax,
    TEMPLATE_CONFIGS,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def parser():
    return TemplateParser()


def _make_file_info(relative_path: str, language: str):
    fi = MagicMock()
    fi.relative_path = relative_path
    fi.absolute_path = Path(f"/tmp/{Path(relative_path).name}")
    fi.language = language
    return fi


def _parse(parser, file_info, content: str):
    return parser.parse(content, file_info, "test", "repo", "user")


# =============================================================================
# BLADE (LARAVEL)
# =============================================================================


class TestBlade:
    def test_blade_directives_stripped(self, parser):
        html = """
        @extends('layouts.app')
        @section('content')
        <div class="container">
            <h1>{{ $title }}</h1>
            @if($user->isAdmin())
                <p class="admin-badge">Admin</p>
            @else
                <p>Regular user</p>
            @endif
            <ul>
            @foreach($items as $item)
                <li>{{ $item->name }}</li>
            @endforeach
            </ul>
        </div>
        @endsection
        """
        fi = _make_file_info("resources/views/dashboard.blade.php", "blade")
        elements = _parse(parser, fi, html)
        raw = elements[0].raw_code
        # Directives stripped
        assert "@if" not in raw
        assert "@foreach" not in raw
        assert "@extends" not in raw
        assert "@endsection" not in raw
        # HTML content preserved
        assert "Admin" in raw
        assert "Regular user" in raw
        # Interpolation kept as text
        assert "$title" in raw or "title" in raw

    def test_blade_raw_output_stripped(self, parser):
        html = '<div>{!! $rawHtml !!}</div>'
        fi = _make_file_info("view.blade.php", "blade")
        elements = _parse(parser, fi, html)
        assert "{!!" not in elements[0].raw_code

    def test_blade_css_classes_extracted(self, parser):
        html = '<div class="container mx-auto"><p class="text-lg">{{ $msg }}</p></div>'
        fi = _make_file_info("view.blade.php", "blade")
        elements = _parse(parser, fi, html)
        metrics = elements[0].code_metrics
        assert "css_classes" in metrics
        assert "container" in metrics["css_classes"]
        assert "mx-auto" in metrics["css_classes"]

    def test_blade_language_preserved(self, parser):
        fi = _make_file_info("view.blade.php", "blade")
        elements = _parse(parser, fi, "<p>Hello</p>")
        assert elements[0].language == "blade"


# =============================================================================
# ERB (RAILS)
# =============================================================================


class TestErb:
    def test_erb_tags_stripped(self, parser):
        html = """
        <div class="products">
            <h1><%= @page_title %></h1>
            <% if @products.any? %>
                <ul>
                <% @products.each do |product| %>
                    <li class="product"><%= product.name %></li>
                <% end %>
                </ul>
            <% else %>
                <p>No products found.</p>
            <% end %>
        </div>
        """
        fi = _make_file_info("app/views/products/index.html.erb", "erb")
        elements = _parse(parser, fi, html)
        raw = elements[0].raw_code
        # Control flow stripped
        assert "<%" not in raw
        assert "%>" not in raw
        # Content preserved
        assert "No products found" in raw

    def test_erb_output_kept_as_text(self, parser):
        html = '<h1><%= @title %></h1>'
        fi = _make_file_info("view.html.erb", "erb")
        elements = _parse(parser, fi, html)
        raw = elements[0].raw_code
        assert "# " in raw  # h1 → markdown heading
        # Expression text should be preserved
        assert "@title" in raw or "title" in raw

    def test_erb_comment_stripped(self, parser):
        html = '<%# This is a comment %><p>Content</p>'
        fi = _make_file_info("view.erb", "erb")
        elements = _parse(parser, fi, html)
        assert "This is a comment" not in elements[0].raw_code
        assert "Content" in elements[0].raw_code

    def test_erb_language_preserved(self, parser):
        fi = _make_file_info("view.html.erb", "erb")
        elements = _parse(parser, fi, "<p>Hello</p>")
        assert elements[0].language == "erb"


# =============================================================================
# EJS (EXPRESS)
# =============================================================================


class TestEjs:
    def test_ejs_tags_stripped(self, parser):
        html = """
        <div class="homepage">
            <h1><%= title %></h1>
            <ul>
            <% users.forEach(function(user) { %>
                <li class="user-item"><%= user.name %></li>
            <% }); %>
            </ul>
        </div>
        """
        fi = _make_file_info("views/home.ejs", "ejs")
        elements = _parse(parser, fi, html)
        raw = elements[0].raw_code
        assert "<%" not in raw
        assert "forEach" not in raw
        assert "homepage" in raw or "user-item" in raw

    def test_ejs_comment_stripped(self, parser):
        html = '<%# comment %><p>Visible</p>'
        fi = _make_file_info("view.ejs", "ejs")
        elements = _parse(parser, fi, html)
        assert "comment" not in elements[0].raw_code
        assert "Visible" in elements[0].raw_code


# =============================================================================
# HANDLEBARS
# =============================================================================


class TestHandlebars:
    def test_handlebars_blocks_stripped(self, parser):
        html = """
        <div class="card">
            <h2>{{title}}</h2>
            {{#if isActive}}
                <span class="badge">Active</span>
            {{else}}
                <span>Inactive</span>
            {{/if}}
            {{#each items}}
                <p>{{this.name}}</p>
            {{/each}}
        </div>
        """
        fi = _make_file_info("templates/card.hbs", "handlebars")
        elements = _parse(parser, fi, html)
        raw = elements[0].raw_code
        assert "{{#if" not in raw
        assert "{{/if}}" not in raw
        assert "{{#each" not in raw
        assert "{{else}}" not in raw
        # Content preserved
        assert "Active" in raw
        assert "Inactive" in raw

    def test_handlebars_expression_kept(self, parser):
        html = '<h1>{{pageTitle}}</h1>'
        fi = _make_file_info("view.hbs", "handlebars")
        elements = _parse(parser, fi, html)
        assert "pageTitle" in elements[0].raw_code

    def test_handlebars_comment_stripped(self, parser):
        html = '{{!-- secret comment --}}<p>Visible</p>'
        fi = _make_file_info("view.hbs", "handlebars")
        elements = _parse(parser, fi, html)
        assert "secret comment" not in elements[0].raw_code
        assert "Visible" in elements[0].raw_code

    def test_handlebars_partial_stripped(self, parser):
        html = '{{> header}}<main>Content</main>{{> footer}}'
        fi = _make_file_info("view.hbs", "handlebars")
        elements = _parse(parser, fi, html)
        assert "{{>" not in elements[0].raw_code
        assert "Content" in elements[0].raw_code


# =============================================================================
# GO TEMPLATES
# =============================================================================


class TestGoTemplate:
    def test_go_template_stripped(self, parser):
        html = """
        <div class="profile">
            <h1>{{.User.Name}}</h1>
            {{if .User.IsActive}}
                <span class="badge active">Active</span>
            {{else}}
                <span class="badge">Inactive</span>
            {{end}}
            <ul>
            {{range .Posts}}
                <li class="post">{{.Title}}</li>
            {{end}}
            </ul>
        </div>
        """
        fi = _make_file_info("templates/user.gohtml", "gotemplate")
        elements = _parse(parser, fi, html)
        raw = elements[0].raw_code
        assert "{{if" not in raw
        assert "{{range" not in raw
        assert "{{end}}" not in raw
        assert "Active" in raw

    def test_go_template_expression_kept(self, parser):
        html = '<title>{{.PageTitle}}</title>'
        fi = _make_file_info("layout.gohtml", "gotemplate")
        elements = _parse(parser, fi, html)
        assert ".PageTitle" in elements[0].raw_code

    def test_go_template_comment_stripped(self, parser):
        html = '{{/* This is hidden */}}<p>Visible</p>'
        fi = _make_file_info("view.tmpl", "gotemplate")
        elements = _parse(parser, fi, html)
        assert "This is hidden" not in elements[0].raw_code
        assert "Visible" in elements[0].raw_code


# =============================================================================
# RAZOR (ASP.NET)
# =============================================================================


class TestRazor:
    def test_razor_directives_stripped(self, parser):
        html = """
        @model IEnumerable<Product>
        @using MyApp.Models
        <div class="container">
            <h1>Products</h1>
            <ul class="product-grid">
                <li class="item">Sample</li>
            </ul>
        </div>
        """
        fi = _make_file_info("Views/Products/Index.cshtml", "razor")
        elements = _parse(parser, fi, html)
        raw = elements[0].raw_code
        assert "@model" not in raw
        assert "@using" not in raw
        assert "Products" in raw

    def test_razor_comment_stripped(self, parser):
        html = '@* hidden comment *@<p>Visible</p>'
        fi = _make_file_info("view.cshtml", "razor")
        elements = _parse(parser, fi, html)
        assert "hidden comment" not in elements[0].raw_code
        assert "Visible" in elements[0].raw_code


# =============================================================================
# HEEX (PHOENIX)
# =============================================================================


class TestHeex:
    def test_heex_tags_stripped(self, parser):
        html = """
        <div class="card">
            <h2><%= @name %></h2>
            <p><%= @email %></p>
            <.link href={@profile_url} class="btn-primary">
                View Profile
            </.link>
        </div>
        """
        fi = _make_file_info("lib/components/user_card.html.heex", "heex")
        elements = _parse(parser, fi, html)
        raw = elements[0].raw_code
        assert "<%=" not in raw
        assert "View Profile" in raw

    def test_heex_code_block_stripped(self, parser):
        html = '<% for item <- @items do %><li><%= item.name %></li><% end %>'
        fi = _make_file_info("view.heex", "heex")
        elements = _parse(parser, fi, html)
        raw = elements[0].raw_code
        assert "<%" not in raw


# =============================================================================
# DISCOVERY: COMPOUND EXTENSIONS
# =============================================================================


class TestCompoundExtensions:
    def test_blade_php_detected(self):
        from magaldi_core.discovery import COMPOUND_EXTENSIONS

        assert ".blade.php" in COMPOUND_EXTENSIONS
        assert COMPOUND_EXTENSIONS[".blade.php"] == "blade"

    def test_html_erb_detected(self):
        from magaldi_core.discovery import COMPOUND_EXTENSIONS

        assert ".html.erb" in COMPOUND_EXTENSIONS
        assert COMPOUND_EXTENSIONS[".html.erb"] == "erb"

    def test_html_heex_detected(self):
        from magaldi_core.discovery import COMPOUND_EXTENSIONS

        assert ".html.heex" in COMPOUND_EXTENSIONS
        assert COMPOUND_EXTENSIONS[".html.heex"] == "heex"

    def test_detect_blade_language(self):
        from magaldi_core.discovery import _detect_file_language

        assert _detect_file_language(Path("dashboard.blade.php")) == "blade"

    def test_detect_erb_language(self):
        from magaldi_core.discovery import _detect_file_language

        assert _detect_file_language(Path("index.html.erb")) == "erb"

    def test_detect_heex_language(self):
        from magaldi_core.discovery import _detect_file_language

        assert _detect_file_language(Path("user_card.html.heex")) == "heex"

    def test_plain_php_not_blade(self):
        """Regular .php files should still be detected as PHP, not blade."""
        from magaldi_core.discovery import _detect_file_language

        assert _detect_file_language(Path("app.php")) == "php"

    def test_plain_erb_detected(self):
        """Standalone .erb files also work."""
        from magaldi_core.discovery import _detect_file_language

        assert _detect_file_language(Path("template.erb")) == "erb"


# =============================================================================
# DISCOVERY: SINGLE EXTENSIONS
# =============================================================================


class TestSingleExtensions:
    def test_ejs_detected(self):
        from magaldi_core.discovery import SUPPORTED_EXTENSIONS

        assert SUPPORTED_EXTENSIONS[".ejs"] == "ejs"

    def test_hbs_detected(self):
        from magaldi_core.discovery import SUPPORTED_EXTENSIONS

        assert SUPPORTED_EXTENSIONS[".hbs"] == "handlebars"

    def test_handlebars_detected(self):
        from magaldi_core.discovery import SUPPORTED_EXTENSIONS

        assert SUPPORTED_EXTENSIONS[".handlebars"] == "handlebars"

    def test_gohtml_detected(self):
        from magaldi_core.discovery import SUPPORTED_EXTENSIONS

        assert SUPPORTED_EXTENSIONS[".gohtml"] == "gotemplate"

    def test_tmpl_detected(self):
        from magaldi_core.discovery import SUPPORTED_EXTENSIONS

        assert SUPPORTED_EXTENSIONS[".tmpl"] == "gotemplate"

    def test_cshtml_detected(self):
        from magaldi_core.discovery import SUPPORTED_EXTENSIONS

        assert SUPPORTED_EXTENSIONS[".cshtml"] == "razor"

    def test_heex_detected(self):
        from magaldi_core.discovery import SUPPORTED_EXTENSIONS

        assert SUPPORTED_EXTENSIONS[".heex"] == "heex"


# =============================================================================
# PARSER REGISTRY
# =============================================================================


class TestParserRegistry:
    @pytest.mark.parametrize("language", [
        "blade", "erb", "ejs", "handlebars", "gotemplate", "razor", "heex",
    ])
    def test_template_parser_registered(self, language):
        from magaldi_core.code_parser import get_parser

        parser = get_parser(language)
        assert parser is not None
        assert isinstance(parser, TemplateParser)


# =============================================================================
# GENERIC BEHAVIOUR
# =============================================================================


class TestGenericBehaviour:
    def test_empty_template(self, parser):
        fi = _make_file_info("view.blade.php", "blade")
        elements = _parse(parser, fi, "")
        assert len(elements) == 1
        assert elements[0].raw_code == ""

    def test_pure_html_no_template_syntax(self, parser):
        """Template with no template tags should still work fine."""
        html = '<div class="container"><h1>Hello World</h1></div>'
        fi = _make_file_info("view.ejs", "ejs")
        elements = _parse(parser, fi, html)
        raw = elements[0].raw_code
        assert "Hello World" in raw
        assert "container" in raw or "{.container}" in raw

    def test_document_sections_extracted(self, parser):
        html = '<h1>Title</h1><p>Intro</p><h2>Details</h2><p>More info</p>'
        fi = _make_file_info("view.hbs", "handlebars")
        elements = _parse(parser, fi, html)
        sections = elements[0].document_sections
        assert sections is not None
        assert len(sections) >= 2

    def test_file_element_type(self, parser):
        fi = _make_file_info("view.gohtml", "gotemplate")
        elements = _parse(parser, fi, "<p>Hello</p>")
        assert elements[0].element_type == "file"

    def test_unknown_language_fallback(self, parser):
        """Unknown template language falls through to plain HTML parsing."""
        fi = _make_file_info("view.unknown", "unknown_template")
        elements = _parse(parser, fi, "<p>Fallback</p>")
        assert "Fallback" in elements[0].raw_code
