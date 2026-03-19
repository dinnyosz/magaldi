"""Heuristic CSS/SCSS summary generation.

Converts CSS rules into human-readable text summaries without LLM.
Property names are mapped to visual/layout concepts for semantic search.

Examples:
  - ".navbar { display: flex; padding: 1rem; }" → "Styles .navbar with flex layout and padding"
  - "@media (max-width: 768px) { .sidebar { display: none; } }"
    → "Responsive styles for max-width 768px: hides .sidebar"
  - "@keyframes fadeIn { from { opacity: 0 } to { opacity: 1 } }"
    → "Animation 'fadeIn': transitions opacity"
"""

from __future__ import annotations

import re

# =============================================================================
# PROPERTY → CONCEPT MAPPING
# =============================================================================

# Maps CSS property names (or prefixes) to human-readable concepts.
# Order matters: first match wins for prefix-based lookups.
PROPERTY_CONCEPTS: dict[str, str] = {
    # Layout
    "display: flex": "flex layout",
    "display: grid": "grid layout",
    "display: inline-flex": "inline flex layout",
    "display: inline-grid": "inline grid layout",
    "display: none": "hidden",
    "display: block": "block display",
    "display: inline-block": "inline-block display",
    "display: inline": "inline display",
    "display: contents": "contents display",
    # Positioning
    "position: fixed": "fixed positioning",
    "position: sticky": "sticky positioning",
    "position: absolute": "absolute positioning",
    "position: relative": "relative positioning",
    # Flexbox
    "flex-direction": "flex direction",
    "flex-wrap": "flex wrapping",
    "flex-grow": "flex growth",
    "flex-shrink": "flex shrink",
    "flex-basis": "flex basis",
    "justify-content": "justify content",
    "align-items": "align items",
    "align-self": "align self",
    "align-content": "align content",
    "gap": "spacing gap",
    "row-gap": "row gap",
    "column-gap": "column gap",
    "order": "flex order",
    # Grid
    "grid-template-columns": "grid columns",
    "grid-template-rows": "grid rows",
    "grid-template-areas": "grid areas",
    "grid-column": "grid column placement",
    "grid-row": "grid row placement",
    "grid-area": "grid area placement",
    "grid-auto-flow": "grid auto flow",
    "grid-auto-columns": "auto grid columns",
    "grid-auto-rows": "auto grid rows",
    # Box model
    "margin": "margin",
    "padding": "padding",
    "border": "border",
    "border-radius": "border radius",
    "box-sizing": "box sizing",
    "width": "width",
    "height": "height",
    "max-width": "max width",
    "max-height": "max height",
    "min-width": "min width",
    "min-height": "min height",
    "overflow": "overflow",
    "overflow-x": "horizontal overflow",
    "overflow-y": "vertical overflow",
    # Typography
    "font-size": "font size",
    "font-weight": "font weight",
    "font-family": "font family",
    "font-style": "font style",
    "line-height": "line height",
    "letter-spacing": "letter spacing",
    "text-align": "text alignment",
    "text-decoration": "text decoration",
    "text-transform": "text transform",
    "white-space": "white space",
    "word-break": "word break",
    "word-wrap": "word wrap",
    "text-overflow": "text overflow",
    "vertical-align": "vertical alignment",
    # Colors & backgrounds
    "color": "text color",
    "background-color": "background color",
    "background-image": "background image",
    "background": "background",
    "opacity": "opacity",
    # Visual effects
    "box-shadow": "box shadow",
    "text-shadow": "text shadow",
    "filter": "filter effect",
    "backdrop-filter": "backdrop filter",
    "mix-blend-mode": "blend mode",
    "clip-path": "clip path",
    # Transitions & animations
    "transition": "transition",
    "animation": "animation",
    "animation-name": "animation name",
    "animation-duration": "animation duration",
    "animation-delay": "animation delay",
    "transform": "transform",
    # Other
    "cursor": "cursor",
    "pointer-events": "pointer events",
    "user-select": "user selection",
    "z-index": "z-index",
    "visibility": "visibility",
    "outline": "outline",
    "list-style": "list style",
    "content": "generated content",
    "resize": "resizable",
    "appearance": "appearance",
    "scroll-behavior": "scroll behavior",
    "scroll-snap-type": "scroll snapping",
    "aspect-ratio": "aspect ratio",
    "object-fit": "object fit",
    "object-position": "object position",
    "container-type": "container query type",
    "container-name": "container name",
}

# Display property+value combos that override the generic property match
_DISPLAY_VALUE_RE = re.compile(r"display\s*:\s*(\w[\w-]*)")


def _property_to_concept(prop: str, value: str | None = None) -> str | None:
    """Map a CSS property (and optionally its value) to a concept.

    Checks property+value combos first (e.g., "display: flex" → "flex layout"),
    then falls back to property-name-only matches.
    """
    prop = prop.strip().lower()

    # Try property+value combo for richer concepts
    if value is not None:
        value = value.strip().lower().rstrip(";").strip()
        combo = f"{prop}: {value}"
        if combo in PROPERTY_CONCEPTS:
            return PROPERTY_CONCEPTS[combo]

    # Fall back to property-name match
    if prop in PROPERTY_CONCEPTS:
        return PROPERTY_CONCEPTS[prop]

    # Prefix matches for vendor-prefixed or shorthand variants
    for prefix in ("-webkit-", "-moz-", "-ms-", "-o-"):
        if prop.startswith(prefix):
            unprefixed = prop[len(prefix) :]
            if unprefixed in PROPERTY_CONCEPTS:
                return PROPERTY_CONCEPTS[unprefixed]

    return None


# =============================================================================
# DECLARATION PARSING
# =============================================================================

_DECLARATION_RE = re.compile(
    r"([\w-]+)\s*:\s*([^;{}]+?)(?:;|$)", re.MULTILINE
)

_CUSTOM_PROP_RE = re.compile(
    r"(--[\w-]+)\s*:\s*([^;{}]+?)(?:;|$)", re.MULTILINE
)


def _parse_declarations(block_content: str) -> list[tuple[str, str]]:
    """Extract property: value pairs from a CSS block body."""
    return _DECLARATION_RE.findall(block_content)


def _parse_custom_properties(block_content: str) -> list[tuple[str, str]]:
    """Extract CSS custom properties (--*) from a block."""
    return _CUSTOM_PROP_RE.findall(block_content)


# =============================================================================
# SELECTOR HELPERS
# =============================================================================


def _clean_selector(selector: str) -> str:
    """Clean and normalize a CSS selector for display in summaries."""
    # Collapse whitespace
    selector = re.sub(r"\s+", " ", selector).strip()
    return selector


def _extract_selectors_from_block(block_content: str) -> list[str]:
    """Extract nested selectors from inside a @media or similar at-rule."""
    selectors: list[str] = []
    # Match "selector { ... }" patterns
    for m in re.finditer(r"([^{}]+?)\s*\{", block_content):
        sel = _clean_selector(m.group(1))
        if sel and not sel.startswith("@"):
            selectors.append(sel)
    return selectors


# =============================================================================
# SUMMARY GENERATORS BY ELEMENT KIND
# =============================================================================


def summarize_rule(selector: str, block_content: str) -> str:
    """Generate summary for a CSS qualified rule (.class { ... }).

    Args:
        selector: The CSS selector (e.g., ".navbar", "#main > .content").
        block_content: The raw text inside the braces.

    Returns:
        Human-readable summary like "Styles .navbar with flex layout, padding, and border radius"
    """
    selector = _clean_selector(selector)
    declarations = _parse_declarations(block_content)

    # Collect unique concepts with key value details
    concepts: list[str] = []
    seen: set[str] = set()
    seen_props: set[str] = set()  # Track base property names for dedup
    color_values: list[str] = []
    for prop, value in declarations:
        if prop.startswith("--"):
            continue  # Custom properties handled separately
        concept = _property_to_concept(prop, value)
        if concept and concept not in seen:
            concepts.append(concept)
            seen.add(concept)
            seen_props.add(prop)
        elif not concept and prop not in seen:
            # Skip shorthand variants if base property already matched
            # e.g., skip "padding-top" if "padding" already produced a concept
            base = prop.rsplit("-", 1)[0] if "-" in prop else None
            if base and base in seen_props:
                continue
            # For unmapped properties, include prop name for searchability
            concepts.append(prop)
            seen.add(prop)
            seen_props.add(prop)
        # Collect color values for extra detail
        if prop in ("color", "background-color", "background") and value.strip():
            val = value.strip().rstrip(";").strip()
            if len(val) <= 30:
                color_values.append(f"{prop}: {val}")

    if not concepts:
        # Fallback: count declarations
        n = len(declarations)
        if n == 0:
            return f"Empty rule for {selector}"
        return f"Styles {selector} with {n} {'property' if n == 1 else 'properties'}"

    base = f"Styles {selector} with {_join_list(concepts)}"
    # Append key color values for searchability (e.g., "color: #333")
    if color_values:
        base += f" ({', '.join(color_values[:3])})"
    return base


def summarize_media(prelude: str, block_content: str) -> str:
    """Generate summary for a @media rule.

    Args:
        prelude: The media query (e.g., "(max-width: 768px)").
        block_content: The raw text inside the braces.

    Returns:
        Summary like "Responsive styles for max-width 768px: adjusts .sidebar and .navbar"
    """
    prelude = prelude.strip()
    # Clean up parens for readability
    condition = prelude.strip("()")

    selectors = _extract_selectors_from_block(block_content)

    # Also extract key property changes for detail
    declarations = _parse_declarations(block_content)
    key_changes: list[str] = []
    seen_concepts: set[str] = set()
    for prop, value in declarations:
        concept = _property_to_concept(prop, value)
        if concept and concept not in seen_concepts:
            key_changes.append(concept)
            seen_concepts.add(concept)

    if selectors:
        sel_list = ", ".join(selectors[:5])
        if len(selectors) > 5:
            sel_list += f" and {len(selectors) - 5} more"
        base = f"Responsive styles for {condition}: adjusts {sel_list}"
        # Append key property changes for richer search
        if key_changes:
            base += f" ({_join_list(key_changes[:4])})"
        return base

    return f"Responsive styles for {condition}"


def summarize_keyframes(name: str, block_content: str) -> str:
    """Generate summary for a @keyframes rule.

    Args:
        name: Animation name (e.g., "fadeIn").
        block_content: The raw text inside the braces.

    Returns:
        Summary like "Animation 'fadeIn': transitions opacity and transform"
    """
    # Collect all properties used across keyframe steps
    declarations = _parse_declarations(block_content)
    props: list[str] = []
    seen: set[str] = set()
    for prop, value in declarations:
        concept = _property_to_concept(prop, value) or prop
        if concept not in seen:
            props.append(concept)
            seen.add(concept)

    if props:
        return f"Animation '{name}': transitions {_join_list(props)}"
    return f"Animation '{name}'"


def summarize_font_face(block_content: str) -> str:
    """Generate summary for a @font-face rule.

    Returns:
        Summary like "Font face declaration for 'Inter' (woff2)"
    """
    declarations = dict(_parse_declarations(block_content))
    font_family = declarations.get("font-family", "").strip("'\"") or "unknown"
    # Try to detect format from src
    src = declarations.get("src", "")
    fmt = ""
    fmt_match = re.search(r"format\(['\"]?(\w+)", src)
    if fmt_match:
        fmt = f" ({fmt_match.group(1)})"
    elif ".woff2" in src:
        fmt = " (woff2)"
    elif ".woff" in src:
        fmt = " (woff)"
    elif ".ttf" in src:
        fmt = " (ttf)"

    weight = declarations.get("font-weight", "")
    weight_part = f", weight {weight}" if weight else ""

    return f"Font face declaration for '{font_family}'{fmt}{weight_part}"


def summarize_custom_properties(selector: str, block_content: str) -> str:
    """Generate summary for a rule containing CSS custom properties.

    Typically :root { --color-primary: ...; --spacing-md: ...; }

    Args:
        selector: The selector (usually ":root" or "[data-theme='dark']").
        block_content: The raw text inside the braces.

    Returns:
        Summary like "CSS custom properties on :root: --color-primary, --spacing-md, --font-size-base"
    """
    custom_props = _parse_custom_properties(block_content)
    if not custom_props:
        return f"Custom properties on {_clean_selector(selector)}"

    prop_names = [name for name, _ in custom_props]
    if len(prop_names) <= 6:
        return f"CSS custom properties on {_clean_selector(selector)}: {', '.join(prop_names)}"
    shown = ", ".join(prop_names[:5])
    return f"CSS custom properties on {_clean_selector(selector)}: {shown} and {len(prop_names) - 5} more"


def summarize_scss_variable(name: str, value: str) -> str:
    """Generate summary for a SCSS variable ($var: value).

    Handles simple values, maps, and lists:
      $color: red         → "SCSS variable $color set to red"
      $breakpoints: (sm: 640px, md: 768px, lg: 1024px)
        → "SCSS map $breakpoints with keys sm, md, lg"
      $fonts: ("Helvetica", "Arial", sans-serif)
        → "SCSS list $fonts with 3 items"

    Returns:
        Human-readable summary for semantic search.
    """
    value = value.strip().rstrip(";").strip()

    # Detect map: (key: value, key: value, ...)
    if value.startswith("(") and value.endswith(")"):
        inner = value[1:-1].strip()
        # Check if it's a map (has key: value pairs)
        pairs = re.findall(r"([\w-]+)\s*:", inner)
        if pairs:
            keys = ", ".join(pairs[:8])
            extra = f" and {len(pairs) - 8} more" if len(pairs) > 8 else ""
            return f"SCSS map {name} with keys {keys}{extra}"
        # Otherwise it's a list
        items = [i.strip() for i in inner.split(",") if i.strip()]
        if items:
            if len(items) <= 4:
                return f"SCSS list {name}: {', '.join(items)}"
            return f"SCSS list {name} with {len(items)} items"

    # Simple value
    if len(value) > 60:
        value = value[:57] + "..."
    return f"SCSS variable {name} set to {value}"


def summarize_scss_mixin(name: str, params: str, block_content: str) -> str:
    """Generate summary for a SCSS @mixin.

    Returns:
        Summary like "Mixin responsive($breakpoint): applies media query wrapper with flex layout"
    """
    signature = f"{name}({params})" if params else name
    declarations = _parse_declarations(block_content)

    concepts: list[str] = []
    seen: set[str] = set()
    for prop, value in declarations:
        concept = _property_to_concept(prop, value)
        if concept and concept not in seen:
            concepts.append(concept)
            seen.add(concept)

    # Check for @media inside mixin
    if "@media" in block_content:
        concepts.insert(0, "media query wrapper")
    # Check for @include
    includes = re.findall(r"@include\s+([\w-]+)", block_content)
    for inc in includes:
        label = f"includes {inc}"
        if label not in seen:
            concepts.append(label)
            seen.add(label)

    if concepts:
        return f"Mixin {signature}: applies {_join_list(concepts)}"
    return f"Mixin {signature}"


def summarize_scss_function(name: str, params: str, block_content: str) -> str:
    """Generate summary for a SCSS @function.

    Returns:
        Summary like "Function rem($px): converts pixel values to rem units"
    """
    signature = f"{name}({params})" if params else name

    # Try to infer purpose from @return and body
    return_match = re.search(r"@return\s+(.+?)(?:;|$)", block_content)
    if return_match:
        return_expr = return_match.group(1).strip()
        if len(return_expr) <= 60:
            return f"Function {signature}: returns {return_expr}"
        return f"Function {signature}: returns computed value"

    return f"Function {signature}"


def summarize_at_rule(name: str, prelude: str, block_content: str | None) -> str:
    """Generate summary for a generic at-rule (@import, @charset, @layer, etc.).

    Args:
        name: The at-rule name (without @).
        prelude: The text between @ and {.
        block_content: The block content, or None for statement at-rules.

    Returns:
        Summary like "@import 'normalize.css'" or "@layer base: defines layer with 3 rules"
    """
    prelude = prelude.strip()

    if name == "import":
        return f"@import {prelude}"
    if name == "charset":
        return f"@charset {prelude}"
    if name == "namespace":
        return f"@namespace {prelude}"
    if name == "layer":
        if block_content:
            selectors = _extract_selectors_from_block(block_content)
            n = len(selectors)
            if n:
                return f"@layer {prelude}: defines layer with {n} {'rule' if n == 1 else 'rules'}"
        return f"@layer {prelude}"
    if name == "supports":
        if block_content:
            selectors = _extract_selectors_from_block(block_content)
            if selectors:
                sel_text = ", ".join(selectors[:4])
                return f"@supports {prelude}: styles {sel_text}"
        return f"@supports {prelude}"
    if name == "container":
        if block_content:
            selectors = _extract_selectors_from_block(block_content)
            if selectors:
                return f"Container query {prelude}: adjusts {', '.join(selectors[:4])}"
        return f"Container query {prelude}"

    # Generic fallback
    if block_content:
        return f"@{name} {prelude}"
    return f"@{name} {prelude}"


def summarize_file(filename: str, element_count: int, language: str = "css") -> str:  # noqa: ARG001
    """Generate summary for a CSS/SCSS file-level element.

    Args:
        filename: The file name (e.g., "styles.css").
        element_count: Number of child elements extracted.
        language: "css" or "scss".

    Returns:
        Summary like "CSS stylesheet with 12 rules"
    """
    lang_label = "SCSS stylesheet" if language == "scss" else "CSS stylesheet"
    if element_count == 0:
        return f"Empty {lang_label}"
    return f"{lang_label} with {element_count} {'section' if element_count == 1 else 'sections'}"


# =============================================================================
# HELPERS
# =============================================================================


def _join_list(items: list[str], max_items: int = 5) -> str:
    """Join items with commas and 'and', truncating if needed.

    Examples:
        ["a"] → "a"
        ["a", "b"] → "a and b"
        ["a", "b", "c"] → "a, b, and c"
        6+ items → "a, b, c, d, e, and 3 more"
    """
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    if len(items) <= max_items:
        return ", ".join(items[:-1]) + f", and {items[-1]}"
    shown = ", ".join(items[:max_items])
    remaining = len(items) - max_items
    return f"{shown}, and {remaining} more"
