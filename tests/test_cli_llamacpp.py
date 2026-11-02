"""Tests for the magaldi llamacpp CLI command (llama-server backend)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from shared.cli.llamacpp import (
    DEFAULT_CTX_SIZE,
    DEFAULT_MODELS_MAX,
    DEFAULT_PARALLEL,
    DEFAULT_PORT,
    LLAMA_SERVER,
    MODELS_DIR,
    PROJECT_ROOT,
    _active_model_refs,
    _format_size,
    _generate_presets,
    _get_llamacpp_models,
    _get_llamacpp_port,
    _logfile,
    _pidfile,
    llamacpp,
)
from shared.config import LLMConfig, ModelConfig

# =============================================================================
# Helper functions
# =============================================================================


class TestHelpers:
    """Test helper functions."""

    def test_pidfile_path(self) -> None:
        pf = _pidfile(8090)
        assert pf.name == "llama-server-8090.pid"

    def test_logfile_path(self) -> None:
        lf = _logfile(8090)
        assert lf.name == "llama-server-8090.log"

    def test_format_size_gb(self) -> None:
        assert _format_size(2_740_000_000) == "2.6 GB"

    def test_format_size_mb(self) -> None:
        assert _format_size(150_000_000) == "143 MB"

    def test_format_size_kb(self) -> None:
        assert _format_size(500_000) == "488 KB"

    def test_defaults(self) -> None:
        assert DEFAULT_PORT == 8090
        assert DEFAULT_PARALLEL == 4
        assert DEFAULT_MODELS_MAX == 3
        assert DEFAULT_CTX_SIZE == 40960


# =============================================================================
# _get_llamacpp_models
# =============================================================================


class TestGetLlamacppModels:
    """Test llamacpp model filtering from config."""

    def test_no_llamacpp_models_returns_empty(self) -> None:
        config = LLMConfig(
            models={
                "qwen3-4b": ModelConfig(
                    name="qwen3:4b-instruct",
                    provider="ollama",
                    url="http://localhost:11434",
                ),
            }
        )
        assert _get_llamacpp_models(config) == []

    def test_filters_llamacpp_only(self) -> None:
        config = LLMConfig(
            models={
                "ollama-model": ModelConfig(
                    name="qwen3:4b-instruct",
                    provider="ollama",
                    url="http://localhost:11434",
                ),
                "llamacpp-model": ModelConfig(
                    name="Qwen3-4B-Instruct-2507-Q4_K_M",
                    provider="llamacpp",
                    url="http://localhost:8090",
                ),
                "cloud-model": ModelConfig(
                    name="gpt-4o-mini",
                    provider="openai",
                    url="",
                ),
            }
        )
        models = _get_llamacpp_models(config)
        assert len(models) == 1
        assert models[0].name == "Qwen3-4B-Instruct-2507-Q4_K_M"

    def test_multiple_llamacpp_models(self) -> None:
        config = LLMConfig(
            models={
                "qwen3-4b": ModelConfig(
                    name="Qwen3-4B-Instruct-2507-Q4_K_M",
                    provider="llamacpp",
                    url="http://localhost:8090",
                ),
                "qwen3-1.7b": ModelConfig(
                    name="Qwen3-1.7B-Q4_K_M",
                    provider="llamacpp",
                    url="http://localhost:8090",
                ),
            }
        )
        models = _get_llamacpp_models(config)
        assert len(models) == 2


# =============================================================================
# _get_llamacpp_port
# =============================================================================


class TestGetLlamacppPort:
    """Test port extraction from config."""

    def test_returns_port_from_config(self) -> None:
        config = LLMConfig(
            models={
                "qwen3-4b": ModelConfig(
                    name="Qwen3-4B-Instruct-2507-Q4_K_M",
                    provider="llamacpp",
                    url="http://localhost:8090",
                ),
            }
        )
        assert _get_llamacpp_port(config) == 8090

    def test_returns_default_when_no_llamacpp(self) -> None:
        config = LLMConfig(
            models={
                "ollama": ModelConfig(
                    name="qwen3:4b",
                    provider="ollama",
                    url="http://localhost:11434",
                ),
            }
        )
        assert _get_llamacpp_port(config) == DEFAULT_PORT

    def test_custom_port(self) -> None:
        config = LLMConfig(
            models={
                "model": ModelConfig(
                    name="Qwen3-4B-Instruct-2507-Q4_K_M",
                    provider="llamacpp",
                    url="http://localhost:9999",
                ),
            }
        )
        assert _get_llamacpp_port(config) == 9999


# =============================================================================
# _generate_presets
# =============================================================================


class TestGeneratePresets:
    """Test INI presets file generation."""

    def test_generates_presets_with_num_ctx(self, tmp_path: Path) -> None:
        config = LLMConfig(
            models={
                "qwen3-4b": ModelConfig(
                    name="Qwen3-4B-Instruct-2507-Q4_K_M",
                    provider="llamacpp",
                    url="http://localhost:8090",
                    num_ctx=32768,
                ),
            }
        )
        with patch("shared.cli.llamacpp._presets_file", return_value=tmp_path / "presets.ini"):
            path = _generate_presets(config)
            content = path.read_text()
            assert "[Qwen3-4B-Instruct-2507-Q4_K_M]" in content
            assert "ctx-size = 32768" in content

    def test_empty_presets_when_no_num_ctx(self, tmp_path: Path) -> None:
        config = LLMConfig(
            models={
                "qwen3-4b": ModelConfig(
                    name="Qwen3-4B-Instruct-2507-Q4_K_M",
                    provider="llamacpp",
                    url="http://localhost:8090",
                ),
            }
        )
        with patch("shared.cli.llamacpp._presets_file", return_value=tmp_path / "presets.ini"):
            path = _generate_presets(config)
            content = path.read_text()
            assert "[Qwen3-4B-Instruct-2507-Q4_K_M]" in content

    def test_generates_presets_with_mmproj(self, tmp_path: Path) -> None:
        """mmproj path is written to preset when file exists in projectors dir."""
        fake_models_dir = tmp_path / "models"
        fake_models_dir.mkdir()
        fake_projectors_dir = fake_models_dir / "projectors"
        fake_projectors_dir.mkdir()
        mmproj_file = fake_projectors_dir / "mmproj-BF16.gguf"
        mmproj_file.touch()

        config = LLMConfig(
            models={
                "gemma4-26b": ModelConfig(
                    name="gemma-4-26B-A4B-it-UD-Q4_K_M",
                    provider="llamacpp",
                    url="http://localhost:8090",
                    num_ctx=40960,
                    no_tiers=True,
                    mmproj="unsloth/gemma-4-26B-A4B-it-GGUF:mmproj-BF16.gguf",
                ),
            }
        )
        with (
            patch("shared.cli.llamacpp._presets_file", return_value=tmp_path / "presets.ini"),
            patch("shared.cli.llamacpp.PROJECTORS_DIR", fake_projectors_dir),
        ):
            path = _generate_presets(config)
            content = path.read_text()
            assert "[gemma-4-26B-A4B-it-UD-Q4_K_M]" in content
            assert f"mmproj = {mmproj_file.resolve()}" in content

    def test_skips_mmproj_when_file_missing(self, tmp_path: Path) -> None:
        """mmproj is NOT written to preset when file doesn't exist."""
        fake_projectors_dir = tmp_path / "projectors"
        fake_projectors_dir.mkdir()
        # Don't create the mmproj file

        config = LLMConfig(
            models={
                "gemma4-26b": ModelConfig(
                    name="gemma-4-26B-A4B-it-UD-Q4_K_M",
                    provider="llamacpp",
                    url="http://localhost:8090",
                    num_ctx=40960,
                    no_tiers=True,
                    mmproj="unsloth/gemma-4-26B-A4B-it-GGUF:mmproj-BF16.gguf",
                ),
            }
        )
        with (
            patch("shared.cli.llamacpp._presets_file", return_value=tmp_path / "presets.ini"),
            patch("shared.cli.llamacpp.PROJECTORS_DIR", fake_projectors_dir),
        ):
            path = _generate_presets(config)
            content = path.read_text()
            assert "mmproj" not in content

    def test_skips_non_llamacpp_models(self, tmp_path: Path) -> None:
        config = LLMConfig(
            models={
                "ollama-model": ModelConfig(
                    name="qwen3:4b",
                    provider="ollama",
                    url="http://localhost:11434",
                    num_ctx=8192,
                ),
            }
        )
        with patch("shared.cli.llamacpp._presets_file", return_value=tmp_path / "presets.ini"):
            path = _generate_presets(config)
            content = path.read_text()
            assert "qwen3:4b" not in content


# =============================================================================
# ModelConfig with llamacpp + gguf
# =============================================================================


class TestModelConfigLlamacpp:
    """Test ModelConfig with llamacpp provider and gguf field."""

    def test_get_litellm_model(self) -> None:
        cfg = ModelConfig(
            name="Qwen3-4B-Instruct-2507-Q4_K_M",
            provider="llamacpp",
            url="http://localhost:8090",
        )
        assert cfg.get_litellm_model() == "openai/Qwen3-4B-Instruct-2507-Q4_K_M"

    def test_get_api_base(self) -> None:
        cfg = ModelConfig(
            name="Qwen3-4B-Instruct-2507-Q4_K_M",
            provider="llamacpp",
            url="http://localhost:8090",
        )
        assert cfg.get_api_base() == "http://localhost:8090/v1"

    def test_get_api_base_strips_trailing_slash(self) -> None:
        cfg = ModelConfig(
            name="Qwen3-4B-Instruct-2507-Q4_K_M",
            provider="llamacpp",
            url="http://localhost:8090/",
        )
        assert cfg.get_api_base() == "http://localhost:8090/v1"

    def test_gguf_field(self) -> None:
        cfg = ModelConfig(
            name="Qwen3-4B-Instruct-2507-Q4_K_M",
            provider="llamacpp",
            url="http://localhost:8090",
            gguf="unsloth/Qwen3-4B-Instruct-2507-GGUF:Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        )
        assert cfg.gguf == "unsloth/Qwen3-4B-Instruct-2507-GGUF:Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

    def test_gguf_default_none(self) -> None:
        cfg = ModelConfig(name="qwen3:4b", provider="ollama")
        assert cfg.gguf is None

    def test_mmproj_field(self) -> None:
        cfg = ModelConfig(
            name="gemma-4-26B-A4B-it-UD-Q4_K_M",
            provider="llamacpp",
            url="http://localhost:8090",
            mmproj="unsloth/gemma-4-26B-A4B-it-GGUF:mmproj-BF16.gguf",
        )
        assert cfg.mmproj == "unsloth/gemma-4-26B-A4B-it-GGUF:mmproj-BF16.gguf"

    def test_mmproj_default_none(self) -> None:
        cfg = ModelConfig(name="qwen3:4b", provider="ollama")
        assert cfg.mmproj is None

    def test_ollama_provider_unchanged(self) -> None:
        cfg = ModelConfig(name="qwen3:4b-instruct", provider="ollama")
        assert cfg.get_litellm_model() == "ollama_chat/qwen3:4b-instruct"
        assert cfg.get_api_base() == "http://localhost:11434"

    def test_is_not_thinking_model_qwen35(self) -> None:
        """Qwen3.5 GGUF model is NOT a thinking model."""
        cfg = ModelConfig(
            name="Qwen3.5-4B-Q4_K_M",
            provider="llamacpp",
            url="http://localhost:8090",
        )
        assert cfg.is_thinking_model() is False

    def test_models_dir_exists(self) -> None:
        """MODELS_DIR points to tools/models/ relative to project root."""
        assert MODELS_DIR.name == "models"
        assert MODELS_DIR.parent.name == "tools"


# =============================================================================
# llamacpp update
# =============================================================================


class TestUpdate:
    """Test the llamacpp update subcommand."""

    def _run_update(self, *args: str) -> MagicMock:
        """Helper that patches subprocess.run and invokes update."""
        runner = CliRunner()
        with patch("shared.cli.llamacpp.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="abc1234\n", stderr=""
            )
            result = runner.invoke(llamacpp, ["update", *args])
        return result  # type: ignore[return-value]

    def test_clones_when_dir_missing(self, tmp_path: Path) -> None:
        """When tools/llama.cpp doesn't exist, clones from GitHub."""
        runner = CliRunner()
        fake_dir = tmp_path / "tools" / "llama.cpp"

        with (
            patch("shared.cli.llamacpp.PROJECT_ROOT", tmp_path),
            patch("shared.cli.llamacpp.LLAMA_SERVER", fake_dir / "build" / "bin" / "llama-server"),
            patch("shared.cli.llamacpp.subprocess.run") as mock_run,
            patch("shared.cli.llamacpp.load_config") as mock_config,
            patch("shared.cli.llamacpp._get_pid", return_value=None),
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            mock_config.return_value = MagicMock()
            mock_config.return_value.llm = LLMConfig(models={})

            result = runner.invoke(llamacpp, ["update"])

            # First call should be git clone
            first_call = mock_run.call_args_list[0]
            assert first_call[0][0][0] == "git"
            assert first_call[0][0][1] == "clone"
            assert "llama.cpp" in first_call[0][0][2]

    def test_pulls_when_dir_exists(self, tmp_path: Path) -> None:
        """When tools/llama.cpp exists, fetches and pulls."""
        runner = CliRunner()
        fake_dir = tmp_path / "tools" / "llama.cpp"
        fake_dir.mkdir(parents=True)

        with (
            patch("shared.cli.llamacpp.PROJECT_ROOT", tmp_path),
            patch("shared.cli.llamacpp.LLAMA_SERVER", fake_dir / "build" / "bin" / "llama-server"),
            patch("shared.cli.llamacpp.subprocess.run") as mock_run,
            patch("shared.cli.llamacpp.load_config") as mock_config,
            patch("shared.cli.llamacpp._get_pid", return_value=None),
        ):
            # Simulate: fetch ok, local != remote, pull ok, cmake ok, build ok
            mock_run.return_value = MagicMock(returncode=0, stdout="abc1234\n", stderr="")
            mock_config.return_value = MagicMock()
            mock_config.return_value.llm = LLMConfig(models={})

            result = runner.invoke(llamacpp, ["update"])

            # First call should be git fetch
            first_call = mock_run.call_args_list[0]
            assert first_call[0][0][0] == "git"
            assert "fetch" in first_call[0][0]

    def test_skips_pull_when_up_to_date_and_binary_exists(self, tmp_path: Path) -> None:
        """When already on latest commit and binary exists, skips build."""
        runner = CliRunner()
        fake_dir = tmp_path / "tools" / "llama.cpp"
        fake_dir.mkdir(parents=True)
        fake_server = fake_dir / "build" / "bin" / "llama-server"
        fake_server.parent.mkdir(parents=True)
        fake_server.touch()

        same_sha = "abc1234567890\n"

        with (
            patch("shared.cli.llamacpp.PROJECT_ROOT", tmp_path),
            patch("shared.cli.llamacpp.LLAMA_SERVER", fake_server),
            patch("shared.cli.llamacpp.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=0, stdout=same_sha, stderr=""
            )

            result = runner.invoke(llamacpp, ["update"])

            assert "Already up to date" in result.output
            # Should NOT have cmake calls (only git fetch + 2x rev-parse = 3 calls)
            assert len(mock_run.call_args_list) == 3

    def test_rebuilds_when_up_to_date_but_binary_missing(self, tmp_path: Path) -> None:
        """When on latest commit but binary is missing, rebuilds."""
        runner = CliRunner()
        fake_dir = tmp_path / "tools" / "llama.cpp"
        fake_dir.mkdir(parents=True)
        fake_server = fake_dir / "build" / "bin" / "llama-server"

        same_sha = "abc1234567890\n"

        with (
            patch("shared.cli.llamacpp.PROJECT_ROOT", tmp_path),
            patch("shared.cli.llamacpp.LLAMA_SERVER", fake_server),
            patch("shared.cli.llamacpp.subprocess.run") as mock_run,
            patch("shared.cli.llamacpp.load_config") as mock_config,
            patch("shared.cli.llamacpp._get_pid", return_value=None),
        ):
            mock_run.return_value = MagicMock(
                returncode=0, stdout=same_sha, stderr=""
            )
            mock_config.return_value = MagicMock()
            mock_config.return_value.llm = LLMConfig(models={})

            result = runner.invoke(llamacpp, ["update"])

            assert "Binary missing" in result.output
            # Should have cmake calls after the git checks
            cmake_calls = [
                c for c in mock_run.call_args_list
                if c[0][0][0] == "cmake"
            ]
            assert len(cmake_calls) == 2  # configure + build

    def test_warns_if_server_running(self, tmp_path: Path) -> None:
        """Shows restart hint when server is running."""
        runner = CliRunner()
        fake_dir = tmp_path / "tools" / "llama.cpp"
        fake_dir.mkdir(parents=True)
        fake_server = fake_dir / "build" / "bin" / "llama-server"

        with (
            patch("shared.cli.llamacpp.PROJECT_ROOT", tmp_path),
            patch("shared.cli.llamacpp.LLAMA_SERVER", fake_server),
            patch("shared.cli.llamacpp.subprocess.run") as mock_run,
            patch("shared.cli.llamacpp.load_config") as mock_config,
            patch("shared.cli.llamacpp._get_pid", return_value=12345),
        ):
            # Make all subprocess calls succeed, and make binary appear after build
            def side_effect(cmd, **kwargs):
                if cmd[0] == "cmake" and "--build" in cmd:
                    fake_server.parent.mkdir(parents=True, exist_ok=True)
                    fake_server.touch()
                return MagicMock(returncode=0, stdout="different_sha\n", stderr="")

            mock_run.side_effect = side_effect
            mock_config.return_value = MagicMock()
            mock_config.return_value.llm = LLMConfig(models={})

            result = runner.invoke(llamacpp, ["update"])

            assert "Restart to use the new binary" in result.output

    def test_clone_failure_stops_early(self, tmp_path: Path) -> None:
        """When git clone fails, command stops without building."""
        runner = CliRunner()

        with (
            patch("shared.cli.llamacpp.PROJECT_ROOT", tmp_path),
            patch("shared.cli.llamacpp.LLAMA_SERVER", tmp_path / "tools" / "llama.cpp" / "build" / "bin" / "llama-server"),
            patch("shared.cli.llamacpp.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="fatal: could not connect"
            )

            result = runner.invoke(llamacpp, ["update"])

            assert "Clone failed" in result.output
            assert len(mock_run.call_args_list) == 1  # Only the clone attempt

    def test_cmake_configure_failure(self, tmp_path: Path) -> None:
        """When cmake configure fails, stops before build."""
        runner = CliRunner()
        fake_dir = tmp_path / "tools" / "llama.cpp"
        fake_dir.mkdir(parents=True)

        call_count = 0

        def side_effect(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if cmd[0] == "cmake":
                return MagicMock(returncode=1, stdout="", stderr="CMake Error: ...")
            return MagicMock(returncode=0, stdout="different_sha\n", stderr="")

        with (
            patch("shared.cli.llamacpp.PROJECT_ROOT", tmp_path),
            patch("shared.cli.llamacpp.LLAMA_SERVER", fake_dir / "build" / "bin" / "llama-server"),
            patch("shared.cli.llamacpp.subprocess.run", side_effect=side_effect),
        ):
            result = runner.invoke(llamacpp, ["update"])

            assert "cmake configure failed" in result.output

    def test_jobs_option(self, tmp_path: Path) -> None:
        """The --jobs flag is passed to cmake build."""
        runner = CliRunner()
        fake_dir = tmp_path / "tools" / "llama.cpp"
        fake_dir.mkdir(parents=True)

        with (
            patch("shared.cli.llamacpp.PROJECT_ROOT", tmp_path),
            patch("shared.cli.llamacpp.LLAMA_SERVER", fake_dir / "build" / "bin" / "llama-server"),
            patch("shared.cli.llamacpp.subprocess.run") as mock_run,
            patch("shared.cli.llamacpp.load_config") as mock_config,
            patch("shared.cli.llamacpp._get_pid", return_value=None),
        ):
            mock_run.return_value = MagicMock(
                returncode=0, stdout="different_sha\n", stderr=""
            )
            mock_config.return_value = MagicMock()
            mock_config.return_value.llm = LLMConfig(models={})

            result = runner.invoke(llamacpp, ["update", "--jobs", "2"])

            # Find the cmake --build call
            build_calls = [
                c for c in mock_run.call_args_list
                if c[0][0][0] == "cmake" and "--build" in c[0][0]
            ]
            assert len(build_calls) == 1
            assert "-j2" in build_calls[0][0][0]


# =============================================================================
# pull
# =============================================================================


def _pull_config() -> LLMConfig:
    def m(name: str, **kw: object) -> ModelConfig:
        return ModelConfig(name=name, provider="llamacpp", url="http://localhost:8090", **kw)  # type: ignore[arg-type]

    return LLMConfig(
        models={
            "small": m("Small", gguf="org/Small-GGUF:Small.gguf"),
            "big": m("Big", gguf="org/Big-GGUF:Big.gguf"),
            "embed": m("Embed", gguf="org/Embed-GGUF:Embed.gguf", embedding=True),
            "drafted": m("Drafted", gguf="org/D-GGUF:Drafted.gguf", draft="org/Dr-GGUF:Draft.gguf"),
            "no-gguf": m("NoGguf"),
            "ollama": ModelConfig(name="x", provider="ollama", url="http://localhost:11434"),
        },
        summarize_model="drafted",
        summarize_model_small="small",
        embed_model="embed",
        variable_scoring_model="drafted",
    )


class TestPull:
    """Test the llamacpp pull subcommand."""

    def _run_pull(self, tmp_path: Path, *args: str) -> tuple[object, list[str]]:
        runner = CliRunner()
        mock_cfg = MagicMock()
        mock_cfg.llm = _pull_config()
        with (
            patch("shared.cli.llamacpp.MODELS_DIR", tmp_path),
            patch("shared.cli.llamacpp.PROJECTORS_DIR", tmp_path / "projectors"),
            patch("shared.cli.llamacpp.load_config", return_value=mock_cfg),
            patch("huggingface_hub.hf_hub_download") as mock_dl,
        ):
            result = runner.invoke(llamacpp, ["pull", *args])
        return result, [c.kwargs["filename"] for c in mock_dl.call_args_list]

    def test_active_model_refs(self) -> None:
        assert _active_model_refs(_pull_config()) == {"drafted", "small", "embed"}

    def test_pull_all_by_default(self, tmp_path: Path) -> None:
        result, files = self._run_pull(tmp_path)
        assert result.exit_code == 0  # type: ignore[attr-defined]
        assert sorted(files) == ["Big.gguf", "Draft.gguf", "Drafted.gguf", "Embed.gguf", "Small.gguf"]

    def test_pull_active_skips_unreferenced(self, tmp_path: Path) -> None:
        result, files = self._run_pull(tmp_path, "--active")
        assert result.exit_code == 0  # type: ignore[attr-defined]
        assert "Big.gguf" not in files
        assert sorted(files) == ["Draft.gguf", "Drafted.gguf", "Embed.gguf", "Small.gguf"]

    def test_pull_active_with_model_outside_roles(self, tmp_path: Path) -> None:
        """--active plus -m for unreferenced model pulls nothing."""
        result, files = self._run_pull(tmp_path, "--active", "-m", "big")
        assert files == []
        assert "not found" in result.output  # type: ignore[attr-defined]

    def test_pull_active_with_model_in_roles(self, tmp_path: Path) -> None:
        _, files = self._run_pull(tmp_path, "--active", "-m", "embed")
        assert files == ["Embed.gguf"]

    def test_pull_skips_existing(self, tmp_path: Path) -> None:
        (tmp_path / "Small.gguf").write_bytes(b"x")
        result, files = self._run_pull(tmp_path, "--active")
        assert "Small.gguf" not in files
        assert "already exists" in result.output  # type: ignore[attr-defined]

    def test_pull_download_failure_continues(self, tmp_path: Path) -> None:
        runner = CliRunner()
        mock_cfg = MagicMock()
        mock_cfg.llm = _pull_config()
        with (
            patch("shared.cli.llamacpp.MODELS_DIR", tmp_path),
            patch("shared.cli.llamacpp.PROJECTORS_DIR", tmp_path / "projectors"),
            patch("shared.cli.llamacpp.load_config", return_value=mock_cfg),
            patch("huggingface_hub.hf_hub_download", side_effect=RuntimeError("boom")) as mock_dl,
        ):
            result = runner.invoke(llamacpp, ["pull", "--active"])
        assert result.exit_code == 0
        assert "Failed: boom" in result.output
        assert mock_dl.call_count == 4
