"""Glossary extraction and linking for domain concept discovery."""

from shared.ai.glossary.ai_extractor import (
    GlossaryItem,
    call_llm_for_glossary,
    extract_glossary_from_feature,
)

__all__ = [
    "GlossaryItem",
    "call_llm_for_glossary",
    "extract_glossary_from_feature",
]
