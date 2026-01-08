"""Parser pipeline for Magaldi.

The parser pipeline consists of:
- Phase 1: Discovery - Path validation, config loading, language enumeration
- Phase 2: Change Detection - SHA256 hashing, diff logic
- Phase 3: Parsing - Tree-sitter extraction
- Phase 4: Storage - ES storage, job creation
"""

from magaldi.parser.discovery import (
    DEFAULT_EXCLUDE_DIRECTORIES,
    DEFAULT_EXCLUDE_FILES,
    SUPPORTED_EXTENSIONS,
    DiscoveryError,
    DiscoveryResult,
    LanguageStats,
    RepoConfig,
    discover,
)
from magaldi.parser.change_detection import (
    ChangeDetectionError,
    ChangeManifest,
    FileInfo,
    detect_changes,
)
from magaldi.parser.code_parser import (
    CodeElement,
    ParsedFile,
    ParsingError,
    ParsingResult,
    parse_file,
    parse_files,
)

__all__ = [
    # Phase 1: Discovery
    "discover",
    "DiscoveryResult",
    "DiscoveryError",
    "RepoConfig",
    "LanguageStats",
    "SUPPORTED_EXTENSIONS",
    "DEFAULT_EXCLUDE_DIRECTORIES",
    "DEFAULT_EXCLUDE_FILES",
    # Phase 2: Change Detection
    "detect_changes",
    "ChangeManifest",
    "ChangeDetectionError",
    "FileInfo",
    # Phase 3: Parsing
    "parse_file",
    "parse_files",
    "ParsedFile",
    "ParsingResult",
    "ParsingError",
    "CodeElement",
]
