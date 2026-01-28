"""Analysis modules for extracting semantic information from code.

This package provides analysis functions for:
- Comment and TODO extraction (semantic.py)
- Purity analysis (semantic.py)
- Type annotation extraction (semantic.py)
- API surface detection (api_detection.py)
- Design pattern detection (api_detection.py)
- Code metrics and complexity (metrics.py)
- Security pattern detection (security.py)
- Concurrency and env var detection (concurrency.py)
"""

from __future__ import annotations

from magaldi_core.analysis.api_detection import (
    detect_cli_commands,
    detect_http_routes,
    detect_patterns,
    detect_public_api,
)
from magaldi_core.analysis.concurrency import (
    detect_concurrency,
    detect_env_vars,
    summarize_concurrency,
)
from magaldi_core.analysis.metrics import (
    analyze_docstring,
    compute_code_metrics,
    compute_complexity,
)
from magaldi_core.analysis.security import (
    detect_security_issues,
    get_severity_score,
    summarize_security_issues,
)
from magaldi_core.analysis.semantic import (
    IMPURE_CALLS,
    analyze_purity,
    associate_comments,
    extract_comments,
    extract_section_markers,
    extract_side_effects,
    extract_todos,
    extract_type_annotations,
)

__all__ = [
    # Semantic analysis
    "extract_todos",
    "extract_section_markers",
    "extract_comments",
    "associate_comments",
    "analyze_purity",
    "extract_side_effects",
    "extract_type_annotations",
    "IMPURE_CALLS",
    # API detection
    "detect_http_routes",
    "detect_cli_commands",
    "detect_public_api",
    "detect_patterns",
    # Metrics
    "compute_complexity",
    "compute_code_metrics",
    "analyze_docstring",
    # Security
    "detect_security_issues",
    "summarize_security_issues",
    "get_severity_score",
    # Concurrency
    "detect_env_vars",
    "detect_concurrency",
    "summarize_concurrency",
]
