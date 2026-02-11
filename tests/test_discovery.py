"""Tests for the discovery module (Phase 1 - TDD)."""

import os
from pathlib import Path

import pytest

from magaldi_core.discovery import (
    DEFAULT_EXCLUDE_DIRECTORIES,
    DEFAULT_EXCLUDE_FILES,
    SUPPORTED_EXTENSIONS,
    DiscoveryError,
    DiscoveryResult,
    LanguageStats,
    RepoConfig,
    discover,
    load_repo_config,
    resolve_username,
    validate_path,
    enumerate_languages,
)


# =============================================================================
# FIXTURES
# =============================================================================

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "repos"


@pytest.fixture
def valid_repo_path() -> Path:
    """Path to valid test repository with magaldi.yaml."""
    return FIXTURES_DIR / "valid_repo"


@pytest.fixture
def no_config_repo_path() -> Path:
    """Path to test repository without magaldi.yaml."""
    return FIXTURES_DIR / "no_config_repo"


@pytest.fixture
def clean_user_env():
    """Remove MAGALDI_USER environment variable."""
    original = os.environ.pop("MAGALDI_USER", None)
    yield
    if original is not None:
        os.environ["MAGALDI_USER"] = original


# =============================================================================
# 1.1 PATH VALIDATION
# =============================================================================


class TestValidatePath:
    """Tests for path validation."""

    def test_valid_directory_passes(self, valid_repo_path: Path):
        """Valid directory should pass validation."""
        result = validate_path(valid_repo_path)
        assert result == valid_repo_path.resolve()

    def test_valid_directory_as_string(self, valid_repo_path: Path):
        """String path should be converted to Path."""
        result = validate_path(str(valid_repo_path))
        assert isinstance(result, Path)
        assert result == valid_repo_path.resolve()

    def test_nonexistent_path_raises_error(self):
        """Nonexistent path should raise DiscoveryError."""
        with pytest.raises(DiscoveryError) as exc_info:
            validate_path("/nonexistent/path/to/repo")
        assert "does not exist" in str(exc_info.value).lower()

    def test_file_path_raises_error(self, valid_repo_path: Path):
        """Path to file (not directory) should raise DiscoveryError."""
        file_path = valid_repo_path / "magaldi.yaml"
        with pytest.raises(DiscoveryError) as exc_info:
            validate_path(file_path)
        assert "not a directory" in str(exc_info.value).lower()

    def test_returns_resolved_absolute_path(self, valid_repo_path: Path):
        """Should return resolved absolute path."""
        result = validate_path(valid_repo_path)
        assert result.is_absolute()
        assert not str(result).endswith("..")


# =============================================================================
# 1.2 REPO CONFIG LOADING
# =============================================================================


class TestLoadRepoConfig:
    """Tests for loading repository magaldi.yaml config."""

    def test_load_valid_config(self, valid_repo_path: Path):
        """Should load all fields from valid config."""
        config = load_repo_config(valid_repo_path)

        assert config.scope == "test-scope"
        assert config.name == "valid-repo"
        assert config.description == "A valid test repository"
        assert "test" in config.tags
        assert "fixture" in config.tags

    def test_exclude_directories_loaded(self, valid_repo_path: Path):
        """Should load exclude_directories from config."""
        config = load_repo_config(valid_repo_path)
        assert "node_modules" in config.exclude_directories
        assert "__pycache__" in config.exclude_directories

    def test_exclude_files_loaded(self, valid_repo_path: Path):
        """Should load exclude_files from config."""
        config = load_repo_config(valid_repo_path)
        assert "*.min.js" in config.exclude_files

    def test_missing_config_raises_error(self, no_config_repo_path: Path):
        """Missing magaldi.yaml should raise DiscoveryError."""
        with pytest.raises(DiscoveryError) as exc_info:
            load_repo_config(no_config_repo_path)
        assert "magaldi.yaml" in str(exc_info.value).lower()

    def test_missing_scope_raises_error(self, tmp_path: Path):
        """Config without scope should raise DiscoveryError."""
        config_file = tmp_path / "magaldi.yaml"
        config_file.write_text("name: test-repo\n")

        with pytest.raises(DiscoveryError) as exc_info:
            load_repo_config(tmp_path)
        assert "scope" in str(exc_info.value).lower()

    def test_name_defaults_to_dirname(self, tmp_path: Path):
        """Name should default to directory name if not specified."""
        config_file = tmp_path / "magaldi.yaml"
        config_file.write_text("scope: test-scope\n")

        config = load_repo_config(tmp_path)
        assert config.name == tmp_path.name

    def test_optional_fields_have_defaults(self, tmp_path: Path):
        """Optional fields should have sensible defaults."""
        config_file = tmp_path / "magaldi.yaml"
        config_file.write_text("scope: test-scope\n")

        config = load_repo_config(tmp_path)
        assert config.description is None or config.description == ""
        assert config.tags == []
        assert len(config.exclude_directories) > 0  # Should have defaults
        assert len(config.exclude_files) > 0  # Should have defaults

    def test_default_exclusions_included(self, tmp_path: Path):
        """Default exclusions should always be included."""
        config_file = tmp_path / "magaldi.yaml"
        config_file.write_text("scope: test-scope\n")

        config = load_repo_config(tmp_path)

        # Check some default directory exclusions
        assert ".git" in config.exclude_directories
        assert "node_modules" in config.exclude_directories
        assert "__pycache__" in config.exclude_directories

    def test_user_exclusions_merged_with_defaults(self, valid_repo_path: Path):
        """User exclusions should be merged with defaults."""
        config = load_repo_config(valid_repo_path)

        # User-specified
        assert "node_modules" in config.exclude_directories

        # Defaults should also be present
        assert ".git" in config.exclude_directories


class TestRepoConfigDataclass:
    """Tests for RepoConfig dataclass."""

    def test_repo_config_fields(self):
        """RepoConfig should have all expected fields."""
        config = RepoConfig(
            scope="test",
            name="repo",
            description="desc",
            tags=["a", "b"],
            user=None,
            exclude_directories=["dir1"],
            exclude_files=["*.txt"],
        )

        assert config.scope == "test"
        assert config.name == "repo"
        assert config.description == "desc"
        assert config.tags == ["a", "b"]
        assert config.user is None
        assert config.exclude_directories == ["dir1"]
        assert config.exclude_files == ["*.txt"]


# =============================================================================
# 1.3 USERNAME RESOLUTION
# =============================================================================


class TestResolveUsername:
    """Tests for username resolution."""

    def test_cli_arg_takes_precedence(self, clean_user_env):
        """CLI argument should override everything."""
        os.environ["MAGALDI_USER"] = "env-user"
        config = RepoConfig(scope="test", name="repo", user="config-user")

        result = resolve_username(cli_user="cli-user", config=config)
        assert result == "cli-user"

    def test_env_var_used_when_no_cli(self, clean_user_env):
        """Environment variable should be used when no CLI arg."""
        os.environ["MAGALDI_USER"] = "env-user"
        config = RepoConfig(scope="test", name="repo", user="config-user")

        result = resolve_username(cli_user=None, config=config)
        assert result == "env-user"

    def test_config_used_when_no_cli_or_env(self, clean_user_env):
        """Config user should be used when no CLI or env."""
        config = RepoConfig(scope="test", name="repo", user="config-user")

        result = resolve_username(cli_user=None, config=config)
        assert result == "config-user"

    def test_missing_username_raises_error(self, clean_user_env):
        """Missing username from all sources should raise error."""
        config = RepoConfig(scope="test", name="repo", user=None)

        with pytest.raises(DiscoveryError) as exc_info:
            resolve_username(cli_user=None, config=config)
        assert "username" in str(exc_info.value).lower() or "user" in str(exc_info.value).lower()

    def test_main_username_accepted(self, clean_user_env):
        """'main' should be accepted as a valid username."""
        config = RepoConfig(scope="test", name="repo")

        result = resolve_username(cli_user="main", config=config)
        assert result == "main"

    def test_whitespace_username_rejected(self, clean_user_env):
        """Whitespace-only username should be rejected."""
        config = RepoConfig(scope="test", name="repo")

        with pytest.raises(DiscoveryError):
            resolve_username(cli_user="   ", config=config)


# =============================================================================
# 1.4 LANGUAGE ENUMERATION
# =============================================================================


class TestEnumerateLanguages:
    """Tests for language enumeration."""

    def test_counts_python_files(self, valid_repo_path: Path):
        """Should count Python files."""
        config = RepoConfig(scope="test", name="repo")
        result = enumerate_languages(valid_repo_path, config)

        assert "python" in result
        assert result["python"].files >= 2  # main.py, utils.py

    def test_counts_javascript_files(self, valid_repo_path: Path):
        """Should count JavaScript files."""
        config = RepoConfig(scope="test", name="repo")
        result = enumerate_languages(valid_repo_path, config)

        assert "javascript" in result
        assert result["javascript"].files >= 1  # helper.js

    def test_counts_lines(self, valid_repo_path: Path):
        """Should count lines in files."""
        config = RepoConfig(scope="test", name="repo")
        result = enumerate_languages(valid_repo_path, config)

        assert result["python"].lines > 0
        assert result["javascript"].lines > 0

    def test_respects_directory_exclusions(self, tmp_path: Path):
        """Should not count files in excluded directories."""
        # Create structure with excluded dir
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "app.py").write_text("print('hello')\n")

        excluded_dir = tmp_path / "node_modules"
        excluded_dir.mkdir()
        (excluded_dir / "lib.js").write_text("console.log('hi');\n")

        config = RepoConfig(
            scope="test",
            name="repo",
            exclude_directories=["node_modules"],
        )
        result = enumerate_languages(tmp_path, config)

        assert "python" in result
        assert "javascript" not in result  # Excluded

    def test_respects_file_exclusions(self, tmp_path: Path):
        """Should not count excluded file patterns."""
        (tmp_path / "app.js").write_text("console.log('app');\n")
        (tmp_path / "app.min.js").write_text("console.log('minified');\n")

        config = RepoConfig(
            scope="test",
            name="repo",
            exclude_files=["*.min.js"],
        )
        result = enumerate_languages(tmp_path, config)

        # Should only count 1 JS file
        assert result["javascript"].files == 1

    def test_handles_empty_directory(self, tmp_path: Path):
        """Should handle directory with no supported files."""
        config = RepoConfig(scope="test", name="repo")
        result = enumerate_languages(tmp_path, config)

        assert result == {}

    def test_ignores_unsupported_extensions(self, tmp_path: Path):
        """Should ignore files with unsupported extensions."""
        (tmp_path / "readme.md").write_text("# README\n")
        (tmp_path / "data.json").write_text("{}\n")
        (tmp_path / "app.py").write_text("pass\n")

        config = RepoConfig(scope="test", name="repo")
        result = enumerate_languages(tmp_path, config)

        assert "python" in result
        assert "markdown" in result  # .md is now a supported extension
        assert "json" not in result


class TestLanguageStats:
    """Tests for LanguageStats dataclass."""

    def test_language_stats_fields(self):
        """LanguageStats should have files and lines."""
        stats = LanguageStats(files=10, lines=500)
        assert stats.files == 10
        assert stats.lines == 500


class TestSupportedExtensions:
    """Tests for supported file extensions."""

    def test_python_extensions(self):
        """Python extensions should be supported."""
        assert SUPPORTED_EXTENSIONS[".py"] == "python"

    def test_javascript_extensions(self):
        """JavaScript extensions should be supported."""
        assert SUPPORTED_EXTENSIONS[".js"] == "javascript"
        assert SUPPORTED_EXTENSIONS[".jsx"] == "javascript"
        assert SUPPORTED_EXTENSIONS[".mjs"] == "javascript"
        assert SUPPORTED_EXTENSIONS[".cjs"] == "javascript"

    def test_typescript_extensions(self):
        """TypeScript extensions should be supported."""
        assert SUPPORTED_EXTENSIONS[".ts"] == "typescript"
        assert SUPPORTED_EXTENSIONS[".tsx"] == "tsx"

    def test_php_extension(self):
        """PHP extension should be supported."""
        assert SUPPORTED_EXTENSIONS[".php"] == "php"

    def test_rust_extension(self):
        """Rust extension should be supported."""
        assert SUPPORTED_EXTENSIONS[".rs"] == "rust"


class TestDefaultExclusions:
    """Tests for default exclusion lists."""

    def test_default_exclude_directories(self):
        """Should have common excluded directories."""
        assert "node_modules" in DEFAULT_EXCLUDE_DIRECTORIES
        assert ".git" in DEFAULT_EXCLUDE_DIRECTORIES
        assert "__pycache__" in DEFAULT_EXCLUDE_DIRECTORIES
        assert ".venv" in DEFAULT_EXCLUDE_DIRECTORIES
        assert "venv" in DEFAULT_EXCLUDE_DIRECTORIES

    def test_default_exclude_files(self):
        """Should have common excluded file patterns."""
        assert "*.min.js" in DEFAULT_EXCLUDE_FILES
        assert "*.min.css" in DEFAULT_EXCLUDE_FILES
        assert "*.lock" in DEFAULT_EXCLUDE_FILES
        assert "package-lock.json" in DEFAULT_EXCLUDE_FILES


# =============================================================================
# MAIN DISCOVER FUNCTION
# =============================================================================


class TestDiscover:
    """Tests for the main discover() function."""

    def test_discover_valid_repo(self, valid_repo_path: Path, clean_user_env):
        """Should successfully discover a valid repository."""
        result = discover(valid_repo_path, username="test-user")

        assert isinstance(result, DiscoveryResult)
        assert result.scope == "test-scope"
        assert result.repository == "valid-repo"
        assert result.username == "test-user"
        assert result.repo_path == valid_repo_path.resolve()

    def test_discover_includes_languages(self, valid_repo_path: Path, clean_user_env):
        """Should include language statistics."""
        result = discover(valid_repo_path, username="test-user")

        assert "python" in result.languages
        assert result.total_files > 0
        assert result.total_lines > 0

    def test_discover_includes_config_data(self, valid_repo_path: Path, clean_user_env):
        """Should include config data in result."""
        result = discover(valid_repo_path, username="test-user")

        assert result.description == "A valid test repository"
        assert "test" in result.tags

    def test_discover_includes_exclusions(self, valid_repo_path: Path, clean_user_env):
        """Should include exclusion lists in result."""
        result = discover(valid_repo_path, username="test-user")

        assert len(result.exclude_directories) > 0
        assert len(result.exclude_files) > 0

    def test_discover_without_config_raises_error(self, no_config_repo_path: Path, clean_user_env):
        """Should raise error if no magaldi.yaml."""
        with pytest.raises(DiscoveryError):
            discover(no_config_repo_path, username="test-user")

    def test_discover_without_username_raises_error(self, valid_repo_path: Path, clean_user_env):
        """Should raise error if no username provided."""
        with pytest.raises(DiscoveryError):
            discover(valid_repo_path, username=None)


class TestDiscoveryResult:
    """Tests for DiscoveryResult dataclass."""

    def test_discovery_result_fields(self):
        """DiscoveryResult should have all expected fields."""
        result = DiscoveryResult(
            scope="test-scope",
            repository="test-repo",
            username="test-user",
            repo_path=Path("/test/path"),
            description="desc",
            tags=["tag1"],
            exclude_directories=["dir1"],
            exclude_files=["*.txt"],
            languages={"python": LanguageStats(files=5, lines=100)},
            total_files=5,
            total_lines=100,
        )

        assert result.scope == "test-scope"
        assert result.repository == "test-repo"
        assert result.username == "test-user"
        assert result.total_files == 5
        assert result.total_lines == 100
