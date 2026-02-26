"""Prompt templates and builders for code summarization.

This module contains:
1. Prompt templates for different element types (file, class, function, etc.)
2. Sentence range calculation based on element size
3. Context builder helpers for constructing prompt sections
4. Prompt building functions for both legacy and message-based formats
5. Summary cleaning utilities

The prompts are designed to:
- Guide LLMs to produce concise, actionable summaries for AI agent navigation
- Avoid verbose prefixes like "This function..." or "This class..."
- Scale summary length dynamically based on code element size
- Maximize Ollama KV cache reuse through system/user message separation
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from magaldi_core.code_parser import CodeElement


# =============================================================================
# DYNAMIC SENTENCE RANGES
# =============================================================================
# Sentence count scales with element size (line count) to avoid over-describing
# simple elements and under-describing complex ones.

LINE_THRESHOLDS: dict[str, dict[str, int]] = {
    "file":       {"tiny": 20,  "small": 50,  "medium": 200},
    "class":      {"tiny": 10,  "small": 30,  "medium": 100},
    "interface":  {"tiny": 5,   "small": 15,  "medium": 50},
    "trait":      {"tiny": 5,   "small": 15,  "medium": 50},
    "enum":       {"tiny": 3,   "small": 10,  "medium": 30},
    "type_alias": {"tiny": 1,   "small": 3,   "medium": 10},
    "function":   {"tiny": 5,   "small": 15,  "medium": 50},
    "method":     {"tiny": 5,   "small": 15,  "medium": 50},
    "constant":   {"tiny": 1,   "small": 3,   "medium": 5},
    "variable":   {"tiny": 1,   "small": 3,   "medium": 5},
    "import":     {"tiny": 1,   "small": 1,   "medium": 1},
}

SENTENCE_RANGES: dict[str, dict[str, tuple[int, int]]] = {
    "file": {
        "tiny":   (1, 2),
        "small":  (2, 3),
        "medium": (3, 4),
        "large":  (5, 6),
    },
    "class": {
        "tiny":   (1, 2),
        "small":  (2, 3),
        "medium": (3, 4),
        "large":  (5, 6),
    },
    "interface": {
        "tiny":   (1, 2),
        "small":  (2, 3),
        "medium": (3, 4),
        "large":  (4, 5),
    },
    "trait": {
        "tiny":   (1, 2),
        "small":  (2, 3),
        "medium": (3, 4),
        "large":  (4, 5),
    },
    "enum": {
        "tiny":   (1, 2),
        "small":  (2, 3),
        "medium": (3, 4),
        "large":  (4, 5),
    },
    "type_alias": {
        "tiny":   (1, 1),
        "small":  (1, 2),
        "medium": (2, 2),
        "large":  (2, 3),
    },
    "function": {
        "tiny":   (1, 2),
        "small":  (2, 3),
        "medium": (3, 4),
        "large":  (5, 6),
    },
    "method": {
        "tiny":   (1, 2),
        "small":  (2, 3),
        "medium": (3, 4),
        "large":  (4, 5),
    },
    "constant": {
        "tiny":   (1, 1),
        "small":  (1, 2),
        "medium": (2, 2),
        "large":  (2, 3),
    },
    "variable": {
        "tiny":   (1, 1),
        "small":  (1, 2),
        "medium": (2, 2),
        "large":  (2, 3),
    },
    "import": {
        "tiny":   (1, 2),
        "small":  (1, 2),
        "medium": (1, 2),
        "large":  (1, 2),
    },
}


def get_size_tier(element_type: str, line_count: int) -> str:
    """Determine size tier based on element type and line count.

    Args:
        element_type: Type of element (file, class, function, etc.)
        line_count: Number of lines in the element.

    Returns:
        Size tier: "tiny", "small", "medium", or "large".
    """
    thresholds = LINE_THRESHOLDS.get(element_type, LINE_THRESHOLDS["function"])
    if line_count <= thresholds["tiny"]:
        return "tiny"
    elif line_count <= thresholds["small"]:
        return "small"
    elif line_count <= thresholds["medium"]:
        return "medium"
    return "large"


def get_sentence_range(element_type: str, line_count: int) -> tuple[int, int]:
    """Get sentence range for an element based on its type and size.

    Args:
        element_type: Type of element (file, class, function, etc.)
        line_count: Number of lines in the element.

    Returns:
        Tuple of (min_sentences, max_sentences).
    """
    tier = get_size_tier(element_type, line_count)
    ranges = SENTENCE_RANGES.get(element_type, SENTENCE_RANGES["function"])
    return ranges.get(tier, (3, 4))


def format_sentence_range(element_type: str, line_count: int) -> str:
    """Format sentence range as a string for prompts.

    Args:
        element_type: Type of element (file, class, function, etc.)
        line_count: Number of lines in the element.

    Returns:
        Formatted string like "2-3" or "1" (if min==max).
    """
    min_s, max_s = get_sentence_range(element_type, line_count)
    if min_s == max_s:
        return str(min_s)
    return f"{min_s}-{max_s}"


# =============================================================================
# PROMPT TEMPLATES (Optimized for Prefix Caching)
# =============================================================================
# System messages are STATIC and get cached by Ollama's KV cache.
# User messages contain VARIABLE content (code, paths, context).
# This structure maximizes cache reuse across requests of the same type.

SYSTEM_PROMPTS = {
    "file": """You summarize code files for AI agents navigating codebases.

For each file, provide a {sentence_range} sentence summary answering:
1. PURPOSE: What is this module's primary job? What capability does it provide?
2. DOMAIN: What problem space does this code address?
3. ARCHITECTURE: What design patterns or abstractions does it use?
4. DISCOVERY: When should an agent look here? What questions lead to this file?
5. DEPENDENCIES: What external modules or systems does this integrate with?

Do NOT list individual classes/functions - those are documented separately.
Write ONLY the {sentence_range} sentence summary. No reasoning, explanations, or bullet points.
Start directly with what it does - never start with "This module...", "This file...", or similar.""",

    "class": """You summarize classes for AI agents navigating codebases.

For each class, provide a {sentence_range} sentence summary answering:
1. IDENTITY: What real-world concept or data structure does this class model?
2. RESPONSIBILITY: What problem does using this class solve?
3. INSTANTIATION: How do you create an instance? What parameters are required?
4. STATE: What key attributes does it hold? What makes an instance valid vs invalid?
5. COLLABORATORS: What other classes or modules does it work with?

Do NOT list methods - those are documented separately.
Write ONLY the {sentence_range} sentence summary. No reasoning, explanations, or bullet points.
Start directly with what it models/does - never start with "This class...", "The X class...", or similar.""",

    "interface": """You summarize interfaces for AI agents navigating codebases.

For each interface, provide a {sentence_range} sentence summary answering:
1. CONTRACT: What behavior or capability does this interface define?
2. IMPLEMENTERS: What types of classes should implement this?
3. METHODS: What key methods must implementers provide?
4. USAGE: When should code depend on this interface vs a concrete class?

Write ONLY the {sentence_range} sentence summary. No reasoning, explanations, or bullet points.
Start directly with what contract it defines - never start with "This interface...", "The X interface...", or similar.""",

    "trait": """You summarize traits for AI agents navigating codebases.

For each trait, provide a {sentence_range} sentence summary answering:
1. CAPABILITY: What shared behavior or capability does this trait provide?
2. IMPLEMENTERS: What types of structs/classes should implement this?
3. METHODS: What key methods does it define or require?
4. COMPOSITION: How does this trait combine with others?

Write ONLY the {sentence_range} sentence summary. No reasoning, explanations, or bullet points.
Start directly with what capability it provides - never start with "This trait...", "The X trait...", or similar.""",

    "enum": """You summarize enums for AI agents navigating codebases.

For each enum, provide a {sentence_range} sentence summary answering:
1. DOMAIN: What set of values or states does this enum represent?
2. VARIANTS: What are the key variants and what do they mean?
3. USAGE: Where and how is this enum used in the codebase?

Write ONLY the {sentence_range} sentence summary. No reasoning, explanations, or bullet points.
Start directly with what it represents - never start with "This enum...", "The X enum...", or similar.""",

    "type_alias": """You describe type aliases for AI agents navigating codebases.

For each type alias, provide a {sentence_range} sentence description answering:
1. MEANING: What concept or data shape does this type represent?
2. STRUCTURE: What is the underlying type structure?
3. USAGE: Where and why is this alias used instead of the raw type?

Write ONLY the {sentence_range} sentence description. No reasoning, explanations, or bullet points.
Start directly with what it represents - never start with "This type...", "The X type...", or similar.""",

    "function": """You describe functions for AI agents navigating codebases.

For each function, provide a {sentence_range} sentence description answering:
1. OPERATION: What does calling this function accomplish?
2. INTERFACE: What are the parameters and return value? What types?
3. WHEN TO USE: In what scenario should an agent call this?
4. SIDE EFFECTS: Does it modify external state, perform I/O, or raise exceptions?
5. EDGE CASES: What happens with empty/None inputs? What preconditions must hold?

Write ONLY the {sentence_range} sentence description. No reasoning, explanations, or bullet points.
Start with an action verb - never start with "This function...", "The X function...", or "This function is used to...".""",

    "method": """You describe methods for AI agents navigating codebases.

For each method, provide a {sentence_range} sentence description answering:
1. OPERATION: What does this method do to/for the object?
2. INTERFACE: What parameters does it take? What does it return?
3. STATE: Which instance attributes does it read or modify?
4. LIFECYCLE: Is this setup/init, cleanup/teardown, or called repeatedly?
5. ERRORS: What exceptions can it raise? What preconditions must hold?

Write ONLY the {sentence_range} sentence description. No reasoning, explanations, or bullet points.
Start with an action verb - never start with "This method...", "The X method...", or "This method is used to...".""",

    "constant": """You describe constants for AI agents navigating codebases.

For each constant, provide a {sentence_range} sentence description answering:
1. MEANING: What does this value represent or configure?
2. USAGE: Where in the system is it used?
3. CONSTRAINTS: What are valid values? Any min/max bounds?

Write ONLY the {sentence_range} sentence description. No reasoning, explanations, or bullet points.
Start directly with what it represents - never start with "This constant...", "The X constant...", or similar.""",

    "variable": """You describe variables for AI agents navigating codebases.

For each variable, provide a {sentence_range} sentence description answering:
1. DATA: What information does this variable hold?
2. LIFECYCLE: How is it initialized? When does it change?
3. ROLE: How does this variable influence its containing scope?

Write ONLY the {sentence_range} sentence description. No reasoning, explanations, or bullet points.
Start directly with what it holds - never start with "This variable...", "The X variable...", or similar.""",

    "import": """You describe imports for AI agents navigating codebases.

For each import, provide a {sentence_range} sentence description answering:
1. PROVIDES: What module/package is being imported and what does it provide?
2. PURPOSE: Why is this import needed in this file?

Write ONLY the {sentence_range} sentence description. No reasoning, explanations, or bullet points.
Start with what it imports - never start with "This import...", "The import...", or similar.""",
}

# User message templates - contain variable content
USER_PROMPTS = {
    "file": """Language: {language}
File: {file_path}
{imports_section}{document_sections_section}

Code:
{code}""",

    "class": """File context: {file_summary}

Class: {class_name}
{decorators}
{base_classes_section}
{attributes_section}
{collaborators_section}
{instantiation_hints}

Code:
{code}
{usages_section}""",

    "interface": """File context: {file_summary}

Interface: {name}
{base_classes_section}

Code:
{code}
{usages_section}""",

    "trait": """File context: {file_summary}

Trait: {name}
{base_classes_section}

Code:
{code}
{usages_section}""",

    "enum": """File context: {file_summary}

Enum: {name}

Code:
{code}
{usages_section}""",

    "type_alias": """File context: {file_summary}

Type alias: {name}

Code:
{code}
{usages_section}""",

    "function": """File context: {file_summary}
{class_context}

Function: {function_name}
Signature: {signature}
{docstring_section}
{exceptions_section}
{usage_examples}

Code:
{code}""",

    "method": """File context: {file_summary}
Class context: {class_summary}

Method: {method_name}
Signature: {signature}
{docstring_section}
{state_section}
{exceptions_section}

Code:
{code}
{usages_section}""",

    "constant": """File context: {file_summary}
{class_context}
{function_context}

Name: {name}
{usage_examples}

Value:
{code}""",

    "variable": """File context: {file_summary}
{class_context}
{function_context}

Name: {name}

Value:
{code}
{usages_section}""",

    "import": """File context: {file_summary}

Import: {name}

Code:
{code}""",
}

# Legacy single-prompt templates (kept for backwards compatibility)
PROMPTS = {
    "file": """Summarize this {language} file in {sentence_range} sentences for an AI agent navigating this codebase.

Answer these questions:
1. PURPOSE: What is this module's primary job? What capability does it provide?
2. DOMAIN: What problem space does this code address?
3. ARCHITECTURE: What design patterns or abstractions does it use? (e.g., factory, repository, decorator, event-driven)
4. DISCOVERY: When should an agent look here? What questions or tasks lead to this file?
5. DEPENDENCIES: What external modules or systems does this integrate with?
{imports_section}{document_sections_section}

Do NOT list individual classes/functions - those are documented separately.

File: {file_path}

Code:
{code}

Write ONLY the {sentence_range} sentence summary. No reasoning or bullet points.
Start directly with what it does - never start with "This module...", "This file...", or similar.

Summary:""",
    "class": """Summarize this {language} class in {sentence_range} sentences for an AI agent.

FOCUS on the class itself. Use file context only to understand how it fits in.

Answer these questions:
1. IDENTITY: What real-world concept or data structure does this class model?
2. RESPONSIBILITY: What problem does using this class solve?
3. INSTANTIATION: How do you create an instance? What parameters are required? Any factory methods or singletons?{instantiation_hints}
4. STATE: What key attributes does it hold? What makes an instance valid vs invalid?{attributes_section}
5. COLLABORATORS: What other classes or modules does it work with?{collaborators_section}

Do NOT list methods - those are documented separately.
{base_classes_section}

File context: {file_summary}

Class: {class_name}
{decorators}

Code:
{code}
{usages_section}

Write ONLY the {sentence_range} sentence summary. No reasoning or bullet points.
Start directly with what it models/does - never start with "This class...", "The X class...", or similar.

Summary:""",
    "interface": """Summarize this interface in {sentence_range} sentences for an AI agent.

FOCUS on the interface itself. Use file context only to understand how it fits in.

Answer these questions:
1. CONTRACT: What behavior or capability does this interface define?
2. IMPLEMENTERS: What types of classes should implement this?
3. METHODS: What key methods must implementers provide?
4. USAGE: When should code depend on this interface vs a concrete class?
{base_classes_section}

File context: {file_summary}

Interface: {name}

Code:
{code}
{usages_section}

Write ONLY the {sentence_range} sentence summary. No reasoning or bullet points.
Start directly with what contract it defines - never start with "This interface...", "The X interface...", or similar.

Summary:""",
    "trait": """Summarize this trait in {sentence_range} sentences for an AI agent.

FOCUS on the trait itself. Use file context only to understand how it fits in.

Answer these questions:
1. CAPABILITY: What shared behavior or capability does this trait provide?
2. IMPLEMENTERS: What types of structs/classes should implement this?
3. METHODS: What key methods does it define or require?
4. COMPOSITION: How does this trait combine with others?
{base_classes_section}

File context: {file_summary}

Trait: {name}

Code:
{code}
{usages_section}

Write ONLY the {sentence_range} sentence summary. No reasoning or bullet points.
Start directly with what capability it provides - never start with "This trait...", "The X trait...", or similar.

Summary:""",
    "enum": """Summarize this enum in {sentence_range} sentences for an AI agent.

FOCUS on the enum itself.

Answer these questions:
1. DOMAIN: What set of values or states does this enum represent?
2. VARIANTS: What are the key variants and what do they mean?
3. USAGE: Where and how is this enum used in the codebase?

File context: {file_summary}

Enum: {name}

Code:
{code}
{usages_section}

Write ONLY the {sentence_range} sentence summary. No reasoning or bullet points.
Start directly with what it represents - never start with "This enum...", "The X enum...", or similar.

Summary:""",
    "type_alias": """Describe this type alias in {sentence_range} sentences for an AI agent.

FOCUS on the type alias itself.

Answer these questions:
1. MEANING: What concept or data shape does this type represent?
2. STRUCTURE: What is the underlying type structure?
3. USAGE: Where and why is this alias used instead of the raw type?

File context: {file_summary}

Type alias: {name}

Code:
{code}
{usages_section}

Write ONLY the {sentence_range} sentence description. No reasoning or bullet points.
Start directly with what it represents - never start with "This type...", "The X type...", or similar.

Description:""",
    "function": """Describe this function in {sentence_range} sentences for an AI agent.

FOCUS on the function itself. Use context only to understand its role.

Answer these questions:
1. OPERATION: What does calling this function accomplish?
2. INTERFACE: What are the parameters and return value? What types?
3. WHEN TO USE: In what scenario should an agent call this? What task requires it?{usage_examples}
4. SIDE EFFECTS: Does it modify external state, perform I/O, or raise exceptions?{exceptions_section}
5. EDGE CASES: What happens with empty/None inputs? What preconditions must hold? What errors can occur?

File context: {file_summary}
{class_context}

Function: {function_name}
Signature: {signature}
{docstring_section}

Code:
{code}

Write ONLY the {sentence_range} sentence summary. No reasoning or bullet points.
Start with an action verb - never start with "This function...", "The X function...", or "This function is used to...".

Summary:""",
    "method": """Describe this method in {sentence_range} sentences for an AI agent.

FOCUS on the method itself. Use context only to understand its role.

Answer these questions:
1. OPERATION: What does this method do to/for the object?
2. INTERFACE: What parameters does it take? What does it return?
3. STATE: Which instance attributes does it read? Which does it modify?{state_section}
4. LIFECYCLE: Is this a setup/init method? Cleanup/teardown? Called once or repeatedly? Must it be called in a specific order relative to other methods?
5. ERRORS: What exceptions can it raise? What preconditions must hold?{exceptions_section}

File context: {file_summary}
Class context: {class_summary}

Method: {method_name}
Signature: {signature}
{docstring_section}

Code:
{code}
{usages_section}

Write ONLY the {sentence_range} sentence summary. No reasoning or bullet points.
Start with an action verb - never start with "This method...", "The X method...", or "This method is used to...".

Summary:""",
    "constant": """Describe this constant in {sentence_range} sentences for an AI agent.

FOCUS on the constant itself.

Answer these questions:
1. MEANING: What does this value represent or configure?
2. USAGE: Where in the system is it used?{usage_examples}
3. CONSTRAINTS: What are valid values? Any min/max bounds? Related constants that must stay consistent?

File context: {file_summary}
{function_context}

Name: {name}
Value:
{code}

Write ONLY the {sentence_range} sentence description. No reasoning or bullet points.
Start directly with what it represents - never start with "This constant...", "The X constant...", or similar.

Description:""",
    "variable": """Describe this variable in {sentence_range} sentences for an AI agent.

FOCUS on the variable itself.

Answer these questions:
1. DATA: What information does this variable hold?
2. LIFECYCLE: How is it initialized? When and why does it change?
3. ROLE: How does this variable influence the behavior of its containing scope?

File context: {file_summary}
{class_context}
{function_context}

Name: {name}
Value:
{code}
{usages_section}

Write ONLY the {sentence_range} sentence description. No reasoning or bullet points.
Start directly with what it holds - never start with "This variable...", "The X variable...", or similar.

Description:""",
    "import": """Describe this import in {sentence_range} sentences for an AI agent.

FOCUS on what this import provides.

Answer these questions:
1. PROVIDES: What module/package is being imported and what does it provide?
2. PURPOSE: Why is this import needed in this file?

File context: {file_summary}

Import: {name}

Code:
{code}

Write ONLY the {sentence_range} sentence description. No reasoning or bullet points.
Start with what it imports - never start with "This import...", "The import...", or similar.

Description:""",
}


# =============================================================================
# CONTEXT BUILDER HELPERS
# =============================================================================


def build_imports_section(element: CodeElement) -> str:
    """Build imports section for file prompt.

    Shows key external imports to indicate dependencies.
    """
    if not element.imports:
        return ""

    # Filter to external imports (not relative)
    external = [
        imp.module for imp in element.imports
        if not imp.module.startswith(".")
    ][:5]  # Limit to 5

    if not external:
        return ""

    return f"\nKey imports: {', '.join(external)}"


def build_attributes_section(element: CodeElement) -> str:
    """Build attributes section for class prompt.

    Shows instance attributes defined in __init__.
    """
    if not element.class_attributes:
        return ""

    attr_names = [a["name"] for a in element.class_attributes][:5]  # Limit to 5
    if not attr_names:
        return ""

    return f"\nInstance attributes: {', '.join(attr_names)}"


def build_base_classes_section(element: CodeElement) -> str:
    """Build base classes section for class prompt."""
    if not element.base_classes:
        return ""

    return f"\nInherits from: {', '.join(element.base_classes)}"


def build_collaborators_section(element: CodeElement) -> str:
    """Build collaborators section for class prompt.

    Shows types this class interacts with based on calls.
    """
    if not element.calls:
        return ""

    # Get unique receivers that aren't 'self'
    receivers = list({
        c.receiver for c in element.calls
        if c.receiver and c.receiver != "self"
    })[:3]  # Limit to 3

    if not receivers:
        return ""

    return f"\nUses: {', '.join(receivers)}"


def build_instantiation_hints(element: CodeElement) -> str:
    """Build instantiation hints for class prompt.

    Filters context_usages for instantiation examples.
    """
    if not element.context_usages:
        return ""

    # Filter for instantiation usages
    insts = [
        u for u in element.context_usages
        if "instantiated" in u.lower()
    ][:2]  # Limit to 2

    if not insts:
        return ""

    return "\nInstantiation examples:\n" + "\n".join(f"- {u}" for u in insts)


def build_exceptions_section(element: CodeElement) -> str:
    """Build exceptions section for function/method prompt."""
    if not element.exceptions_raised:
        return ""

    return f"\nRaises: {', '.join(element.exceptions_raised)}"


def build_state_section(element: CodeElement) -> str:
    """Build state section for method prompt.

    Shows which attributes this method modifies.
    """
    if not element.attributes_modified:
        return ""

    return f"\nModifies attributes: {', '.join(element.attributes_modified)}"


def build_usage_examples(element: CodeElement) -> str:
    """Build usage examples section for function/constant prompts."""
    if not element.context_usages:
        return ""

    examples = element.context_usages[:3]  # Limit to 3
    return "\nUsage examples:\n" + "\n".join(f"- {u}" for u in examples)


def build_document_sections_section(element: CodeElement) -> str:
    """Build document sections section for file prompt.

    Shows heading structure for markdown/doc files so the LLM
    understands the document organization.
    """
    if not element.document_sections:
        return ""

    headings = element.document_sections[:10]  # Limit to 10
    lines = [f"{'  ' * (h['level'] - 1)}- {h['title']}" for h in headings]
    return "\nDocument structure:\n" + "\n".join(lines)


# =============================================================================
# CODE TRUNCATION
# =============================================================================


def truncate_code(code: str, max_tokens: int = 4000) -> str:
    """Truncate code to fit context window.

    Args:
        code: Source code to truncate.
        max_tokens: Maximum tokens (rough estimate: 1 token ~= 4 chars).

    Returns:
        Truncated code with marker if truncated.
    """
    max_chars = max_tokens * 4

    if len(code) <= max_chars:
        return code

    truncated = code[:max_chars]

    # Try to end at a complete line
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.8:
        truncated = truncated[:last_newline]

    return truncated + "\n\n# ... (truncated)"


# =============================================================================
# PROMPT BUILDING
# =============================================================================


def build_prompt(
    element: CodeElement,
    parent_summaries: dict[str, str],
    max_code_tokens: int = 4000,
) -> str:
    """Build prompt with parent context and enhanced context sections.

    Args:
        element: Code element to summarize.
        parent_summaries: Dict with 'file' and/or 'class' summaries.
        max_code_tokens: Max tokens for code in prompt.

    Returns:
        Formatted prompt string.
    """
    element_type = element.element_type

    # Get template key
    if element_type == "method":
        template_key = "method"
    elif element_type in PROMPTS:
        template_key = element_type
    else:
        template_key = "function"

    template = PROMPTS[template_key]

    # Calculate line count for dynamic sentence range
    line_count = 1
    if element.line_end and element.line_start:
        line_count = max(1, element.line_end - element.line_start + 1)
    elif element.raw_code:
        line_count = element.raw_code.count("\n") + 1

    # Get sentence range based on element type and size
    sentence_range = format_sentence_range(template_key, line_count)

    # Build parent context sections
    file_summary = parent_summaries.get("file", "No file context available.")
    class_summary = parent_summaries.get("class", "")
    function_summary = parent_summaries.get("function", "")

    class_context = ""
    if class_summary and element_type in ("method", "function", "variable", "constant"):
        class_context = f"Class context: {class_summary}"

    function_context = ""
    if function_summary and element_type in ("variable", "constant"):
        function_context = f"Function/method context: {function_summary}"

    docstring_section = ""
    if element.docstring:
        docstring_section = f"Docstring: {element.docstring}"

    decorators = ""
    if element.decorators:
        decorators = f"Decorators: {', '.join(element.decorators)}"

    # Build usages section for variables/constants
    usages_section = ""
    if element.context_usages:
        usages_section = "\nUsed in:\n" + "\n".join(f"- {u}" for u in element.context_usages)

    # Build enhanced context sections based on element type
    imports_section = build_imports_section(element) if element_type == "file" else ""
    document_sections_section = build_document_sections_section(element) if element_type == "file" else ""
    attributes_section = build_attributes_section(element) if element_type == "class" else ""
    base_classes_section = build_base_classes_section(element) if element_type in ("class", "interface", "trait") else ""
    collaborators_section = build_collaborators_section(element) if element_type == "class" else ""
    instantiation_hints = build_instantiation_hints(element) if element_type == "class" else ""
    exceptions_section = build_exceptions_section(element) if element_type in ("function", "method") else ""
    state_section = build_state_section(element) if element_type == "method" else ""
    usage_examples = build_usage_examples(element) if element_type in ("function", "constant") else ""

    # Truncate code
    code = truncate_code(element.raw_code or "", max_code_tokens)

    return template.format(
        language=element.language or "code",
        file_path=element.relative_path,
        file_summary=file_summary,
        class_summary=class_summary,
        class_context=class_context,
        function_context=function_context,
        class_name=element.name if element_type == "class" else "",
        function_name=element.name,
        method_name=element.name,
        name=element.name,
        signature=element.signature or "",
        docstring_section=docstring_section,
        decorators=decorators,
        code=code,
        usages_section=usages_section,
        sentence_range=sentence_range,
        # Enhanced context sections
        imports_section=imports_section,
        document_sections_section=document_sections_section,
        attributes_section=attributes_section,
        base_classes_section=base_classes_section,
        collaborators_section=collaborators_section,
        instantiation_hints=instantiation_hints,
        exceptions_section=exceptions_section,
        state_section=state_section,
        usage_examples=usage_examples,
    )


def build_messages(
    element: CodeElement,
    parent_summaries: dict[str, str],
    max_code_tokens: int = 4000,
) -> list[dict[str, str]]:
    """Build messages optimized for Ollama KV cache prefix caching.

    Returns system + user messages where:
    - System message: Static instructions (cached after first request of each type)
    - User message: Variable content (code, paths, context)

    This structure maximizes cache reuse - the system message tokens are
    processed once and cached, then reused for all subsequent requests
    of the same element type.

    Args:
        element: Code element to summarize.
        parent_summaries: Dict with 'file' and/or 'class' summaries.
        max_code_tokens: Max tokens for code in prompt.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    element_type = element.element_type

    # Map element type to template key
    if element_type == "method":
        template_key = "method"
    elif element_type in SYSTEM_PROMPTS:
        template_key = element_type
    else:
        template_key = "function"

    # Calculate line count for dynamic sentence range
    line_count = 1
    if element.line_end and element.line_start:
        line_count = max(1, element.line_end - element.line_start + 1)
    elif element.raw_code:
        line_count = element.raw_code.count("\n") + 1

    # Get sentence range based on element type and size
    sentence_range = format_sentence_range(template_key, line_count)

    # Get system prompt and format with sentence range
    system_prompt = SYSTEM_PROMPTS[template_key].format(sentence_range=sentence_range)

    # Build variable content for user message
    file_summary = parent_summaries.get("file", "No file context available.")
    class_summary = parent_summaries.get("class", "")
    function_summary = parent_summaries.get("function", "")

    class_context = ""
    if class_summary and element_type in ("method", "function", "variable", "constant"):
        class_context = f"Class context: {class_summary}"

    function_context = ""
    if function_summary and element_type in ("variable", "constant"):
        function_context = f"Function/method context: {function_summary}"

    docstring_section = ""
    if element.docstring:
        docstring_section = f"Docstring: {element.docstring}"

    decorators = ""
    if element.decorators:
        decorators = f"Decorators: {', '.join(element.decorators)}"

    usages_section = ""
    if element.context_usages:
        usages_section = "\nUsed in:\n" + "\n".join(f"- {u}" for u in element.context_usages)

    # Build enhanced context sections
    imports_section = build_imports_section(element) if element_type == "file" else ""
    document_sections_section = build_document_sections_section(element) if element_type == "file" else ""
    attributes_section = build_attributes_section(element) if element_type == "class" else ""
    base_classes_section = build_base_classes_section(element) if element_type in ("class", "interface", "trait") else ""
    collaborators_section = build_collaborators_section(element) if element_type == "class" else ""
    instantiation_hints = build_instantiation_hints(element) if element_type == "class" else ""
    exceptions_section = build_exceptions_section(element) if element_type in ("function", "method") else ""
    state_section = build_state_section(element) if element_type == "method" else ""
    usage_examples = build_usage_examples(element) if element_type in ("function", "constant") else ""

    # Truncate code
    code = truncate_code(element.raw_code or "", max_code_tokens)

    # Format user message with variable content
    user_template = USER_PROMPTS[template_key]
    user_content = user_template.format(
        language=element.language or "code",
        file_path=element.relative_path,
        file_summary=file_summary,
        class_summary=class_summary,
        class_context=class_context,
        function_context=function_context,
        class_name=element.name if element_type == "class" else "",
        function_name=element.name,
        method_name=element.name,
        name=element.name,
        signature=element.signature or "",
        docstring_section=docstring_section,
        decorators=decorators,
        code=code,
        usages_section=usages_section,
        imports_section=imports_section,
        document_sections_section=document_sections_section,
        attributes_section=attributes_section,
        base_classes_section=base_classes_section,
        collaborators_section=collaborators_section,
        instantiation_hints=instantiation_hints,
        exceptions_section=exceptions_section,
        state_section=state_section,
        usage_examples=usage_examples,
    )

    # Clean up extra blank lines from empty optional sections
    user_content = "\n".join(line for line in user_content.split("\n") if line.strip() or line == "")
    user_content = user_content.strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


# =============================================================================
# SUMMARY CLEANING
# =============================================================================

# Chat template markers that indicate the model went past intended response
CHAT_MARKERS = [
    "<|im_start|>",      # ChatML format (Qwen, etc.)
    "<|im_end|>",        # ChatML end
    "<|im_sep|>",        # ChatML separator
    "<|assistant|>",     # Phi format
    "<|user|>",          # Phi format
    "<|end|>",           # Generic end
    "<|eot_id|>",        # Llama 3 format
    "<|start_header_id|>",  # Llama 3 format
    "[INST]",            # Mistral/Llama 2 format
    "[/INST]",           # Mistral/Llama 2 format
    "### ",              # Alpaca format (### Response:, ### Human:, etc.)
]

# Patterns for reasoning/thinking tags from models like nemotron-3-nano, DeepSeek, etc.
THINKING_PATTERNS = [
    r"<think>.*?</think>\s*",      # <think>...</think>
    r"<thinking>.*?</thinking>\s*", # <thinking>...</thinking>
    r"<reasoning>.*?</reasoning>\s*", # <reasoning>...</reasoning>
    r"<reflection>.*?</reflection>\s*", # <reflection>...</reflection>
]

# Patterns for unclosed thinking tags (content got cut off)
UNCLOSED_THINKING_PATTERNS = [
    r"<think>.*$",
    r"<thinking>.*$",
    r"<reasoning>.*$",
    r"<reflection>.*$",
]

# Common prefixes to remove from summaries
PREFIXES_TO_REMOVE = [
    "Summary:",
    "This function ",
    "This method ",
    "This class ",
    "This file ",
]


def clean_summary(summary: str) -> str:
    """Clean and normalize generated summary.

    Args:
        summary: Raw summary from LLM.

    Returns:
        Cleaned summary.
    """
    summary = summary.strip()

    if not summary:
        return summary

    # Truncate at chat template markers (model starting a new turn)
    # These indicate the model went past the intended response
    for marker in CHAT_MARKERS:
        if marker in summary:
            summary = summary.split(marker)[0]

    # Remove reasoning/thinking tags from models like nemotron-3-nano, DeepSeek, etc.
    # These models output chain-of-thought in <think>...</think> or similar tags
    for pattern in THINKING_PATTERNS:
        summary = re.sub(pattern, "", summary, flags=re.DOTALL | re.IGNORECASE)

    # Handle unclosed thinking tags (content got cut off)
    # Remove everything from opening tag to end if no closing tag
    for pattern in UNCLOSED_THINKING_PATTERNS:
        summary = re.sub(pattern, "", summary, flags=re.DOTALL | re.IGNORECASE)

    summary = summary.strip()

    if not summary:
        return summary

    # Remove common prefixes
    for prefix in PREFIXES_TO_REMOVE:
        if summary.lower().startswith(prefix.lower()):
            summary = summary[len(prefix):].strip()
            break

    # Ensure ends with period
    if summary and not summary.endswith("."):
        summary += "."

    # Capitalize first letter
    if summary:
        summary = summary[0].upper() + summary[1:]

    return summary
