"""Tests for the change detection module (Phase 2)."""

from datetime import datetime
from pathlib import Path

import pytest

from magaldi_core.change_detection import (
    ChangeDetectionError,
    ChangeManifest,
    DBFileState,
    FileInfo,
    InMemoryFileStateRepository,
    compare_main_branch,
    compare_user_branch,
    compute_file_hash,
    compute_hashes_parallel,
    detect_changes,
    enumerate_files,
)
from magaldi_core.discovery import DiscoveryResult, LanguageStats


# =============================================================================
# FIXTURES
# =============================================================================

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "repos"


@pytest.fixture
def valid_repo_path() -> Path:
    return FIXTURES_DIR / "valid_repo"


@pytest.fixture
def sample_discovery_result(valid_repo_path: Path) -> DiscoveryResult:
    return DiscoveryResult(
        scope="test-scope",
        repository="valid-repo",
        username="main",
        repo_path=valid_repo_path,
        description="Test repo",
        tags=["test"],
        exclude_directories=["node_modules", "__pycache__", ".git"],
        exclude_files=["*.min.js"],
        languages={"python": LanguageStats(files=2, lines=30)},
        total_files=3,
        total_lines=50,
    )


@pytest.fixture
def in_memory_repo() -> InMemoryFileStateRepository:
    return InMemoryFileStateRepository()


# =============================================================================
# FILE ENUMERATION
# =============================================================================


class TestEnumerateFiles:
    """Tests for file enumeration."""

    def test_enumerates_python_files(self, valid_repo_path: Path):
        files = enumerate_files(valid_repo_path, [], [])
        python_files = [f for f in files if f.language == "python"]
        assert len(python_files) >= 2  # main.py, utils.py

    def test_enumerates_javascript_files(self, valid_repo_path: Path):
        files = enumerate_files(valid_repo_path, [], [])
        js_files = [f for f in files if f.language == "javascript"]
        assert len(js_files) >= 1  # helper.js

    def test_stores_relative_paths(self, valid_repo_path: Path):
        files = enumerate_files(valid_repo_path, [], [])
        for f in files:
            assert not f.relative_path.startswith("/")
            assert not f.relative_path.startswith(str(valid_repo_path))

    def test_stores_absolute_paths(self, valid_repo_path: Path):
        files = enumerate_files(valid_repo_path, [], [])
        for f in files:
            assert f.absolute_path.is_absolute()
            assert f.absolute_path.exists()

    def test_respects_directory_exclusions(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "app.py").write_text("pass")

        excluded = tmp_path / "node_modules"
        excluded.mkdir()
        (excluded / "lib.js").write_text("console.log()")

        files = enumerate_files(tmp_path, ["node_modules"], [])
        paths = [f.relative_path for f in files]

        assert "src/app.py" in paths or "src\\app.py" in paths
        assert all("node_modules" not in p for p in paths)

    def test_respects_file_exclusions(self, tmp_path: Path):
        (tmp_path / "app.js").write_text("code")
        (tmp_path / "app.min.js").write_text("minified")

        files = enumerate_files(tmp_path, [], ["*.min.js"])
        paths = [f.relative_path for f in files]

        assert len([p for p in paths if p.endswith(".js")]) == 1

    def test_skips_hidden_directories(self, tmp_path: Path):
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("pass")

        (tmp_path / "visible.py").write_text("pass")

        files = enumerate_files(tmp_path, [], [])
        paths = [f.relative_path for f in files]

        assert all(".hidden" not in p for p in paths)

    def test_skips_hidden_files(self, tmp_path: Path):
        (tmp_path / ".hidden.py").write_text("pass")
        (tmp_path / "visible.py").write_text("pass")

        files = enumerate_files(tmp_path, [], [])
        paths = [f.relative_path for f in files]

        assert len(paths) == 1
        assert ".hidden" not in paths[0]


# =============================================================================
# HASH COMPUTATION
# =============================================================================


class TestComputeFileHash:
    """Tests for file hash computation."""

    def test_computes_sha256_hash(self, tmp_path: Path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        hash_value = compute_file_hash(test_file)

        assert hash_value is not None
        assert len(hash_value) == 64  # SHA256 hex length

    def test_same_content_same_hash(self, tmp_path: Path):
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("identical content")
        file2.write_text("identical content")

        assert compute_file_hash(file1) == compute_file_hash(file2)

    def test_different_content_different_hash(self, tmp_path: Path):
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content a")
        file2.write_text("content b")

        assert compute_file_hash(file1) != compute_file_hash(file2)

    def test_returns_none_for_missing_file(self, tmp_path: Path):
        missing = tmp_path / "nonexistent.txt"
        assert compute_file_hash(missing) is None

    def test_handles_empty_file(self, tmp_path: Path):
        empty = tmp_path / "empty.txt"
        empty.write_text("")

        hash_value = compute_file_hash(empty)
        assert hash_value is not None
        # Empty file has known SHA256 hash
        assert hash_value == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_handles_binary_content(self, tmp_path: Path):
        binary = tmp_path / "binary.bin"
        binary.write_bytes(b"\x00\x01\x02\x03")

        hash_value = compute_file_hash(binary)
        assert hash_value is not None


class TestComputeHashesParallel:
    """Tests for parallel hash computation."""

    def test_hashes_multiple_files(self, tmp_path: Path):
        files = []
        for i in range(5):
            f = tmp_path / f"file{i}.py"
            f.write_text(f"content {i}")
            files.append(
                FileInfo(
                    relative_path=f"file{i}.py",
                    absolute_path=f,
                    language="python",
                )
            )

        result = compute_hashes_parallel(files, max_workers=2)

        assert len(result) == 5
        for f in result:
            assert f.hash is not None

    def test_handles_empty_list(self):
        result = compute_hashes_parallel([], max_workers=2)
        assert result == []

    def test_populates_file_size(self, tmp_path: Path):
        f = tmp_path / "test.py"
        f.write_text("hello")

        files = [FileInfo(relative_path="test.py", absolute_path=f, language="python")]
        result = compute_hashes_parallel(files, max_workers=1)

        assert result[0].file_size > 0

    def test_populates_line_count(self, tmp_path: Path):
        f = tmp_path / "test.py"
        f.write_text("line1\nline2\nline3")

        files = [FileInfo(relative_path="test.py", absolute_path=f, language="python")]
        result = compute_hashes_parallel(files, max_workers=1)

        assert result[0].line_count == 3


# =============================================================================
# COMPARE LOGIC - MAIN BRANCH
# =============================================================================


class TestCompareMainBranch:
    """Tests for main branch comparison."""

    def test_detects_new_files(self):
        files = [
            FileInfo(
                relative_path="new.py",
                absolute_path=Path("/repo/new.py"),
                language="python",
                hash="abc123",
            )
        ]
        db_states: dict[str, DBFileState] = {}

        new, modified, deleted, unchanged = compare_main_branch(files, db_states)

        assert len(new) == 1
        assert new[0].relative_path == "new.py"
        assert len(modified) == 0
        assert len(deleted) == 0
        assert unchanged == 0

    def test_detects_modified_files(self):
        files = [
            FileInfo(
                relative_path="changed.py",
                absolute_path=Path("/repo/changed.py"),
                language="python",
                hash="new_hash",
            )
        ]
        db_states = {"changed.py": DBFileState(relative_path="changed.py", file_hash="old_hash")}

        new, modified, deleted, unchanged = compare_main_branch(files, db_states)

        assert len(new) == 0
        assert len(modified) == 1
        assert modified[0].previous_hash == "old_hash"
        assert len(deleted) == 0

    def test_detects_unchanged_files(self):
        files = [
            FileInfo(
                relative_path="same.py",
                absolute_path=Path("/repo/same.py"),
                language="python",
                hash="same_hash",
            )
        ]
        db_states = {"same.py": DBFileState(relative_path="same.py", file_hash="same_hash")}

        new, modified, deleted, unchanged = compare_main_branch(files, db_states)

        assert len(new) == 0
        assert len(modified) == 0
        assert len(deleted) == 0
        assert unchanged == 1

    def test_detects_deleted_files(self):
        files: list[FileInfo] = []
        db_states = {"deleted.py": DBFileState(relative_path="deleted.py", file_hash="hash123")}

        new, modified, deleted, unchanged = compare_main_branch(files, db_states)

        assert len(new) == 0
        assert len(modified) == 0
        assert len(deleted) == 1
        assert deleted[0].relative_path == "deleted.py"

    def test_ignores_already_deleted_in_db(self):
        files: list[FileInfo] = []
        db_states = {
            "deleted.py": DBFileState(relative_path="deleted.py", file_hash="hash", is_deleted=True)
        }

        new, modified, deleted, unchanged = compare_main_branch(files, db_states)

        assert len(deleted) == 0  # Already marked deleted


# =============================================================================
# COMPARE LOGIC - USER BRANCH
# =============================================================================


class TestCompareUserBranch:
    """Tests for user branch comparison."""

    def test_skips_files_same_as_main(self):
        files = [
            FileInfo(
                relative_path="same.py",
                absolute_path=Path("/repo/same.py"),
                language="python",
                hash="main_hash",
            )
        ]
        main_states = {"same.py": DBFileState(relative_path="same.py", file_hash="main_hash")}
        user_states: dict[str, DBFileState] = {}

        new, modified, deleted, unchanged, skipped = compare_user_branch(
            files, main_states, user_states
        )

        assert skipped == 1
        assert len(new) == 0

    def test_detects_new_for_user(self):
        files = [
            FileInfo(
                relative_path="user_file.py",
                absolute_path=Path("/repo/user_file.py"),
                language="python",
                hash="user_hash",
            )
        ]
        main_states = {"other.py": DBFileState(relative_path="other.py", file_hash="main_hash")}
        user_states: dict[str, DBFileState] = {}

        new, modified, deleted, unchanged, skipped = compare_user_branch(
            files, main_states, user_states
        )

        assert len(new) == 1
        assert new[0].relative_path == "user_file.py"

    def test_detects_modified_from_main(self):
        files = [
            FileInfo(
                relative_path="changed.py",
                absolute_path=Path("/repo/changed.py"),
                language="python",
                hash="user_hash",
            )
        ]
        main_states = {"changed.py": DBFileState(relative_path="changed.py", file_hash="main_hash")}
        user_states: dict[str, DBFileState] = {}

        new, modified, deleted, unchanged, skipped = compare_user_branch(
            files, main_states, user_states
        )

        assert len(new) == 1  # New for user (differs from main)

    def test_detects_user_deletion(self):
        files: list[FileInfo] = []  # User deleted file from main
        main_states = {"main_file.py": DBFileState(relative_path="main_file.py", file_hash="hash")}
        user_states: dict[str, DBFileState] = {}

        new, modified, deleted, unchanged, skipped = compare_user_branch(
            files, main_states, user_states
        )

        assert len(deleted) == 1
        assert deleted[0].relative_path == "main_file.py"


# =============================================================================
# MAIN DETECT_CHANGES FUNCTION
# =============================================================================


class TestDetectChanges:
    """Tests for the main detect_changes function."""

    def test_detects_changes_for_main(
        self, sample_discovery_result: DiscoveryResult, in_memory_repo: InMemoryFileStateRepository
    ):
        result = detect_changes(sample_discovery_result, in_memory_repo)

        assert isinstance(result, ChangeManifest)
        assert result.scope == "test-scope"
        assert result.repository == "valid-repo"
        assert result.username == "main"
        assert result.total_files_scanned > 0

    def test_requires_main_for_user_branch(
        self, sample_discovery_result: DiscoveryResult, in_memory_repo: InMemoryFileStateRepository
    ):
        sample_discovery_result.username = "alice"

        with pytest.raises(ChangeDetectionError) as exc_info:
            detect_changes(sample_discovery_result, in_memory_repo)

        assert "main" in str(exc_info.value).lower()

    def test_user_branch_works_if_main_exists(
        self, sample_discovery_result: DiscoveryResult, in_memory_repo: InMemoryFileStateRepository
    ):
        # Set up main branch
        in_memory_repo.set_file_states(
            "test-scope",
            "valid-repo",
            "main",
            {"src/main.py": DBFileState(relative_path="src/main.py", file_hash="main_hash")},
        )

        sample_discovery_result.username = "alice"
        result = detect_changes(sample_discovery_result, in_memory_repo)

        assert result.username == "alice"

    def test_manifest_has_timestamp(self, sample_discovery_result: DiscoveryResult):
        result = detect_changes(sample_discovery_result)

        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)

    def test_files_to_parse_property(self, sample_discovery_result: DiscoveryResult):
        result = detect_changes(sample_discovery_result)

        # First parse, all files are new
        assert result.files_to_parse == len(result.new_files) + len(result.modified_files)


# =============================================================================
# DATA CLASSES
# =============================================================================


class TestChangeManifest:
    """Tests for ChangeManifest dataclass."""

    def test_files_to_parse_calculation(self):
        manifest = ChangeManifest(
            scope="test",
            repository="repo",
            username="main",
            timestamp=datetime.now(),
            total_files_scanned=10,
            new_files=[FileInfo("a.py", Path(), "python", "h1")],
            modified_files=[
                FileInfo("b.py", Path(), "python", "h2"),
                FileInfo("c.py", Path(), "python", "h3"),
            ],
        )

        assert manifest.files_to_parse == 3

    def test_files_to_remove_calculation(self):
        manifest = ChangeManifest(
            scope="test",
            repository="repo",
            username="main",
            timestamp=datetime.now(),
            total_files_scanned=10,
            deleted_files=[
                FileInfo("d.py", Path(), "python"),
                FileInfo("e.py", Path(), "python"),
            ],
        )

        assert manifest.files_to_remove == 2


class TestFileInfo:
    """Tests for FileInfo dataclass."""

    def test_file_info_fields(self):
        info = FileInfo(
            relative_path="src/app.py",
            absolute_path=Path("/repo/src/app.py"),
            language="python",
            hash="abc123",
            previous_hash="old123",
            file_size=1024,
            line_count=50,
        )

        assert info.relative_path == "src/app.py"
        assert info.language == "python"
        assert info.hash == "abc123"
        assert info.previous_hash == "old123"
        assert info.file_size == 1024
        assert info.line_count == 50
