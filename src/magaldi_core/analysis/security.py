"""Security pattern detection for code analysis.

This module provides functions for detecting potential security issues:
- Hardcoded secrets (passwords, API keys, tokens)
- SQL injection vulnerabilities
- Command injection vulnerabilities
- Path traversal vulnerabilities
- Insecure random number generation
- Debug/development code in production
"""

from __future__ import annotations

import re
from typing import Any

# =============================================================================
# SECURITY PATTERNS
# =============================================================================

# Each pattern: (regex, severity)
# Severity: "critical", "high", "medium", "low", "info"

SECURITY_PATTERNS: dict[str, list[tuple[str, str, str]]] = {
    # Hardcoded secrets
    "hardcoded_secret": [
        (r"password\s*=\s*['\"][^'\"]+['\"]", "high", "Hardcoded password"),
        (r"passwd\s*=\s*['\"][^'\"]+['\"]", "high", "Hardcoded password"),
        (r"api_key\s*=\s*['\"][^'\"]+['\"]", "high", "Hardcoded API key"),
        (r"apikey\s*=\s*['\"][^'\"]+['\"]", "high", "Hardcoded API key"),
        (r"secret\s*=\s*['\"][^'\"]+['\"]", "high", "Hardcoded secret"),
        (r"secret_key\s*=\s*['\"][^'\"]+['\"]", "high", "Hardcoded secret key"),
        (r"private_key\s*=\s*['\"][^'\"]+['\"]", "critical", "Hardcoded private key"),
        (r"token\s*=\s*['\"][a-zA-Z0-9_-]{20,}['\"]", "high", "Hardcoded token"),
        (r"auth\s*=\s*['\"][^'\"]+['\"]", "medium", "Hardcoded auth value"),
        (r"Bearer\s+[a-zA-Z0-9_-]{20,}", "high", "Hardcoded bearer token"),
        (r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----", "critical", "Embedded private key"),
        (r"-----BEGIN\s+CERTIFICATE-----", "medium", "Embedded certificate"),
    ],

    # SQL injection
    "sql_injection": [
        (r"execute\s*\(\s*['\"].*%s", "high", "SQL with string formatting"),
        (r"execute\s*\(\s*['\"].*%\(", "high", "SQL with dict formatting"),
        (r'execute\s*\(\s*f["\']', "critical", "SQL with f-string"),
        (r"execute\s*\(\s*['\"].*\+\s*\w+", "high", "SQL with concatenation"),
        (r'cursor\.execute\s*\(\s*f["\']', "critical", "Cursor execute with f-string"),
        (r"raw\s*\(\s*f['\"].*SELECT", "critical", "Raw SQL with f-string"),
        (r"\.query\s*\(\s*['\"].*\$\{", "high", "SQL with template literal"),
    ],

    # Command injection
    "command_injection": [
        (r"os\.system\s*\(\s*f['\"]", "critical", "os.system with f-string"),
        (r"os\.system\s*\(\s*['\"].*%s", "critical", "os.system with string formatting"),
        (r"os\.popen\s*\(\s*f['\"]", "critical", "os.popen with f-string"),
        (r"subprocess\.(run|call|Popen)\s*\(\s*f['\"]", "critical", "subprocess with f-string"),
        (r"subprocess\.(run|call|Popen)\s*\([^)]*shell\s*=\s*True", "high", "subprocess with shell=True"),
        (r"exec\s*\(\s*f['\"]", "critical", "exec with f-string"),
        (r"eval\s*\(\s*\w+", "high", "eval with variable"),
        (r"child_process\.exec\s*\(\s*`", "critical", "child_process.exec with template literal"),
    ],

    # Path traversal
    "path_traversal": [
        (r"open\s*\(\s*f['\"].*\{.*\}", "medium", "File open with f-string path"),
        (r"open\s*\(\s*['\"].*\+\s*\w+", "medium", "File open with concatenated path"),
        (r"\.\.\/", "low", "Parent directory reference"),
        (r"Path\s*\(\s*f['\"]", "medium", "Path with f-string"),
    ],

    # Insecure randomness
    "insecure_random": [
        (r"random\.(random|randint|choice|shuffle)\s*\(", "info", "Non-cryptographic random (use secrets module for security)"),
        (r"Math\.random\s*\(", "info", "Non-cryptographic random in JS"),
        (r"rand\s*\(\s*\)", "info", "Non-cryptographic random"),
    ],

    # Debug/development code
    "debug_code": [
        (r"DEBUG\s*=\s*True", "medium", "Debug mode enabled"),
        (r"debugger;", "medium", "JavaScript debugger statement"),
        (r"console\.log\s*\(", "low", "Console.log in code"),
        (r"print\s*\(\s*['\"]DEBUG", "low", "Debug print statement"),
        (r"binding\.pry", "medium", "Ruby debugger"),
        (r"import\s+pdb", "medium", "Python debugger import"),
        (r"breakpoint\s*\(\s*\)", "medium", "Python breakpoint"),
    ],

    # Insecure crypto
    "weak_crypto": [
        (r"md5\s*\(", "medium", "MD5 hash (consider SHA-256+)"),
        (r"sha1\s*\(", "medium", "SHA1 hash (consider SHA-256+)"),
        (r"DES\.", "high", "DES encryption (use AES)"),
        (r"RC4", "high", "RC4 encryption (insecure)"),
    ],

    # XXE and deserialization
    "deserialization": [
        (r"pickle\.loads?\s*\(", "high", "Pickle deserialization (dangerous with untrusted data)"),
        (r"yaml\.load\s*\([^)]*\)", "high", "YAML load without safe_load"),
        (r"yaml\.unsafe_load", "critical", "Unsafe YAML load"),
        (r"unserialize\s*\(", "high", "PHP unserialize"),
        (r"JSON\.parse\s*\(.*\$", "medium", "JSON parse with user input"),
    ],

    # SSRF indicators
    "ssrf": [
        (r"requests\.(get|post|put|delete)\s*\(\s*f['\"]", "medium", "HTTP request with f-string URL"),
        (r"requests\.(get|post|put|delete)\s*\(\s*\w+\s*\)", "medium", "HTTP request with variable URL"),
        (r"urllib\.request\.urlopen\s*\(\s*\w+", "medium", "URL open with variable"),
        (r"fetch\s*\(\s*\w+", "medium", "Fetch with variable URL"),
    ],
}


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================


def detect_security_issues(raw_code: str, language: str) -> list[dict[str, Any]]:
    """Detect potential security issues in code.

    Args:
        raw_code: The source code to analyze.
        language: Programming language (for context).

    Returns:
        List of security issues: [{kind, pattern, line, severity, message}]
    """
    if not raw_code:
        return []

    issues: list[dict[str, Any]] = []
    lines = raw_code.split("\n")

    for kind, patterns in SECURITY_PATTERNS.items():
        for pattern_str, severity, message in patterns:
            try:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                for line_num, line in enumerate(lines, start=1):
                    # Skip comments
                    stripped = line.strip()
                    if _is_comment_line(stripped, language):
                        continue

                    if pattern.search(line):
                        # Check for false positives
                        if _is_false_positive(kind, line, language):
                            continue

                        issues.append({
                            "kind": kind,
                            "pattern": pattern_str,
                            "line": line_num,
                            "severity": severity,
                            "message": message,
                        })
            except re.error:
                # Invalid regex pattern, skip
                continue

    # Deduplicate by line and kind
    seen = set()
    unique_issues = []
    for issue in issues:
        key = (issue["line"], issue["kind"])
        if key not in seen:
            seen.add(key)
            unique_issues.append(issue)

    return unique_issues


def _is_comment_line(line: str, language: str) -> bool:
    """Check if a line is a comment."""
    if language == "python":
        return line.startswith("#")
    elif language in ("javascript", "typescript", "php", "rust"):
        return line.startswith("//") or line.startswith("/*") or line.startswith("*")
    return False


def _is_false_positive(kind: str, line: str, _language: str) -> bool:
    """Check for common false positive patterns."""
    line_lower = line.lower()

    # Hardcoded secrets: Skip if it's a variable declaration with empty/placeholder value
    if kind == "hardcoded_secret":
        # Skip empty strings or placeholders
        if re.search(r"=\s*['\"][\s]*['\"]", line):  # Empty string
            return True
        if re.search(r"=\s*['\"](\*+|x+|your[_-]?|CHANGEME|TODO|REPLACE)['\"]", line, re.IGNORECASE):
            return True
        # Skip environment variable lookups
        if "os.environ" in line or "getenv" in line or "process.env" in line:
            return True
        # Skip test files (heuristic: line contains "test" context)
        if "test" in line_lower and "=" in line:
            return True

    # SQL injection: Skip parameterized queries
    if kind == "sql_injection":
        # Skip if using parameterized form
        if re.search(r"execute\s*\([^)]*,\s*\[", line):  # execute(query, [params])
            return True
        if re.search(r"execute\s*\([^)]*,\s*\(", line):  # execute(query, (params))
            return True

    # Debug code: Skip test files
    if kind == "debug_code" and "test" in line_lower:
        return True

    return False


def get_severity_score(severity: str) -> int:
    """Convert severity to numeric score for sorting."""
    scores = {
        "critical": 5,
        "high": 4,
        "medium": 3,
        "low": 2,
        "info": 1,
    }
    return scores.get(severity, 0)


def summarize_security_issues(issues: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize security issues by severity and kind.

    Args:
        issues: List of security issues.

    Returns:
        Summary dict with counts by severity and kind.
    """
    by_severity: dict[str, int] = {}
    by_kind: dict[str, int] = {}

    for issue in issues:
        severity = issue.get("severity", "info")
        kind = issue.get("kind", "unknown")

        by_severity[severity] = by_severity.get(severity, 0) + 1
        by_kind[kind] = by_kind.get(kind, 0) + 1

    return {
        "total": len(issues),
        "by_severity": by_severity,
        "by_kind": by_kind,
        "has_critical": by_severity.get("critical", 0) > 0,
        "has_high": by_severity.get("high", 0) > 0,
    }
