"""Generic template parser — strips template syntax, then delegates to HtmlParser.

Supports backend template formats: Blade, ERB, EJS, Handlebars, Go templates,
Razor, and HEEx. Each format has its own set of regex patterns that strip
template-specific syntax (control flow, interpolation, directives) leaving
clean HTML that the existing HtmlParser converts to markdown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from magaldi_core.parsers.html import HtmlParser

if TYPE_CHECKING:
    from magaldi_core.change_detection import FileInfo
    from magaldi_core.parsers.base import CodeElement


@dataclass
class TemplateConfig:
    """Regex patterns to strip template syntax from a specific format.

    Attributes:
        patterns: Compiled regexes whose matches are removed from the content.
        replacements: Compiled regexes with replacement strings (pattern, repl).
    """

    patterns: list[re.Pattern[str]] = field(default_factory=list)
    replacements: list[tuple[re.Pattern[str], str]] = field(default_factory=list)


# =============================================================================
# TEMPLATE FORMAT CONFIGURATIONS
# =============================================================================

# Blade (Laravel) — @if, @foreach, {{ }}, {!! !!}, @section/@yield etc.
_BLADE_CONFIG = TemplateConfig(
    patterns=[
        # Block directives: @if(...), @foreach(...), @section('name'), etc.
        re.compile(r"@(?:if|elseif|unless|for|foreach|forelse|while|switch|case|break|default)\s*\([^)]*\)", re.DOTALL),
        # End directives: @endif, @endforeach, etc.
        re.compile(r"@end(?:if|unless|for|foreach|forelse|while|switch|section|push|once|php|verbatim|component|slot)\b"),
        # Simple directives: @else, @csrf, @method, @yield, @extends, @include, @slot, etc.
        re.compile(r"@(?:else|csrf|method|yield|extends|include|slot|push|prepend|stack|once|php|verbatim|auth|guest|production|env|hasSection|sectionMissing|component|endcomponent|props|aware)\b(?:\s*\([^)]*\))?", re.DOTALL),
        # Raw output: {!! $var !!}
        re.compile(r"\{!!\s*.*?\s*!!\}", re.DOTALL),
    ],
    replacements=[
        # Escaped output: {{ $var }} → keep the expression text for context
        (re.compile(r"\{\{\s*(.*?)\s*\}\}", re.DOTALL), r"\1"),
    ],
)

# ERB (Rails) — <%= %>, <% %>, <%- %>, <%# %>
_ERB_CONFIG = TemplateConfig(
    patterns=[
        # Comment tags: <%# ... %>
        re.compile(r"<%#.*?%>", re.DOTALL),
        # Code-only tags (no output): <% code %>
        re.compile(r"<%(?!=)-?\s*(?:end|if|elsif|else|unless|for|while|do|begin|rescue|ensure|case|when)\b.*?-?%>", re.DOTALL),
    ],
    replacements=[
        # Output tags: <%= expr %> and <%- expr %> → keep expression
        (re.compile(r"<%-?\s*=?\s*(.*?)\s*-?%>", re.DOTALL), r"\1"),
    ],
)

# EJS (Express) — <%= %>, <% %>, <%- %>, <%# %>
_EJS_CONFIG = TemplateConfig(
    patterns=[
        # Comment tags: <%# ... %>
        re.compile(r"<%#.*?%>", re.DOTALL),
        # Code-only tags: <% code %> (control flow)
        re.compile(r"<%(?!=|-).*?%>", re.DOTALL),
    ],
    replacements=[
        # Output tags: <%= expr %> (escaped) and <%- expr %> (raw) → keep expression
        (re.compile(r"<%-?\s*=?\s*(.*?)\s*-?%>", re.DOTALL), r"\1"),
    ],
)

# Handlebars / Mustache — {{ }}, {{#if}}, {{/if}}, {{>partial}}, {{!comment}}
_HANDLEBARS_CONFIG = TemplateConfig(
    patterns=[
        # Comments: {{!-- ... --}} and {{! ... }}
        re.compile(r"\{\{!--.*?--\}\}", re.DOTALL),
        re.compile(r"\{\{!.*?\}\}", re.DOTALL),
        # Block helpers: {{#if}}, {{#each}}, {{#with}}, {{#unless}}, {{/...}}
        re.compile(r"\{\{[#/]\s*\w+.*?\}\}", re.DOTALL),
        # Partials: {{> partialName}}
        re.compile(r"\{\{>\s*\w+.*?\}\}", re.DOTALL),
        # Else: {{else}}
        re.compile(r"\{\{else\}\}"),
    ],
    replacements=[
        # Triple-stash (raw): {{{ expr }}} → keep expression
        (re.compile(r"\{\{\{\s*(.*?)\s*\}\}\}", re.DOTALL), r"\1"),
        # Double-stash (escaped): {{ expr }} → keep expression
        (re.compile(r"\{\{\s*(.*?)\s*\}\}", re.DOTALL), r"\1"),
    ],
)

# Go templates — {{ }}, {{if}}, {{range}}, {{end}}, {{template}}, {{define}}
_GOTEMPLATE_CONFIG = TemplateConfig(
    patterns=[
        # Comments: {{/* ... */}}
        re.compile(r"\{\{/\*.*?\*/\}\}", re.DOTALL),
        # Control flow: {{if}}, {{else}}, {{end}}, {{range}}, {{with}}, {{block}}, {{define}}, {{template}}
        re.compile(r"\{\{-?\s*(?:if|else|end|range|with|block|define|template)\b.*?-?\}\}", re.DOTALL),
    ],
    replacements=[
        # Output: {{ .Field }} or {{ funcCall .arg }} → keep expression
        (re.compile(r"\{\{-?\s*(.*?)\s*-?\}\}", re.DOTALL), r"\1"),
    ],
)

# Razor (ASP.NET) — @model, @{ }, @if, @foreach, @variable
_RAZOR_CONFIG = TemplateConfig(
    patterns=[
        # Code blocks: @{ ... } (multiline)
        re.compile(r"@\{[^}]*\}", re.DOTALL),
        # Directives: @model, @using, @inject, @page, @layout, @section, @addTagHelper
        re.compile(r"@(?:model|using|inject|page|layout|section|addTagHelper|implements|inherits|namespace|attribute)\b[^\n]*"),
        # Control flow: @if (...) { ... }, @foreach, @for, @while, @switch
        re.compile(r"@(?:if|else\s*if|foreach|for|while|switch|lock|try|catch|finally)\s*\([^)]*\)\s*\{[^}]*\}", re.DOTALL),
        # @else { ... }
        re.compile(r"@else\s*\{[^}]*\}", re.DOTALL),
        # Comments: @* ... *@
        re.compile(r"@\*.*?\*@", re.DOTALL),
    ],
    replacements=[
        # Inline expressions: @variable, @Model.Property, @DateTime.Now
        (re.compile(r"@([\w.]+(?:\(.*?\))?)"), r"\1"),
    ],
)

# HEEx (Phoenix/Elixir) — <%= %>, <% %>, <.component>, {:else}, ~H sigil
_HEEX_CONFIG = TemplateConfig(
    patterns=[
        # Code-only tags: <% code %>
        re.compile(r"<%(?!=).*?%>", re.DOTALL),
        # Elixir block markers: {:else}, {:ok}, {:error}
        re.compile(r"\{:\w+\}"),
        # Component self-closing tags: <.component_name attr="val" />
        re.compile(r"<\.\w+[^>]*/\s*>", re.DOTALL),
    ],
    replacements=[
        # Output tags: <%= expr %> → keep expression
        (re.compile(r"<%=\s*(.*?)\s*%>", re.DOTALL), r"\1"),
        # Component opening/closing tags: <.name> → name, </.name> → empty
        (re.compile(r"</\.\w+\s*>"), ""),
        (re.compile(r"<\.(\w+)"), r"<div data-component='\1'"),
    ],
)


TEMPLATE_CONFIGS: dict[str, TemplateConfig] = {
    "blade": _BLADE_CONFIG,
    "erb": _ERB_CONFIG,
    "ejs": _EJS_CONFIG,
    "handlebars": _HANDLEBARS_CONFIG,
    "gotemplate": _GOTEMPLATE_CONFIG,
    "razor": _RAZOR_CONFIG,
    "heex": _HEEX_CONFIG,
}


# =============================================================================
# TEMPLATE SYNTAX STRIPPING
# =============================================================================


def strip_template_syntax(content: str, config: TemplateConfig) -> str:
    """Strip template-specific syntax from content, leaving clean HTML.

    Applies removal patterns first (control flow, comments), then
    replacement patterns (interpolation → keep expression text).

    Args:
        content: Raw template file content.
        config: Template configuration with patterns.

    Returns:
        Cleaned HTML content with template syntax removed.
    """
    result = content

    # First apply removal patterns (control flow, comments, directives)
    for pattern in config.patterns:
        result = pattern.sub("", result)

    # Then apply replacement patterns (interpolation → keep expression)
    for pattern, repl in config.replacements:
        result = pattern.sub(repl, result)

    return result


# =============================================================================
# TEMPLATE PARSER
# =============================================================================


class TemplateParser:
    """Generic parser for backend template files.

    Strips template-specific syntax (Blade, ERB, EJS, Handlebars, Go, Razor,
    HEEx) and delegates to HtmlParser for HTML→markdown conversion and
    metadata extraction.
    """

    def __init__(self) -> None:
        self._html_parser = HtmlParser()

    def parse(
        self,
        content: str,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
    ) -> list[CodeElement]:
        """Parse template content by stripping syntax and delegating to HtmlParser.

        Args:
            content: Raw template file content.
            file_info: File metadata from change detection.
            scope: Repository scope.
            repository: Repository name.
            username: Username (branch).

        Returns:
            List with a single file-level CodeElement containing
            markdown-converted content as raw_code.
        """
        config = TEMPLATE_CONFIGS.get(file_info.language)
        if config is None:
            # Fallback: treat as plain HTML if no config found
            clean_html = content
        else:
            clean_html = strip_template_syntax(content, config)

        # Delegate to HtmlParser for HTML→markdown + metadata extraction
        elements = self._html_parser.parse(
            clean_html, file_info, scope, repository, username,
        )

        # Override language to keep the template language (not "html")
        for elem in elements:
            elem.language = file_info.language

        return elements
