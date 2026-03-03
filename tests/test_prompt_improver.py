"""Tests for the RALPH prompt improver."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from magaldi_core.parsers.base import Call, CodeElement, Import
from shared.ai.ollama_benchmark import CriteriaScores
from shared.ai.prompt_improver import (
    IMPROVEMENT_PROMPT,
    IterationResult,
    PromptImprover,
    PromptVersion,
    RALPHResult,
    _format_user_template,
    _get_required_placeholders,
    _get_template_key,
    _parse_improvement_response,
    _reconstruct_element,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def function_element() -> CodeElement:
    return CodeElement(
        element_id="myorg:myrepo:main:src/utils.py:function:process_data:10",
        scope="myorg",
        repository="myrepo",
        username="main",
        relative_path="src/utils.py",
        element_type="function",
        name="process_data",
        language="python",
        line_start=10,
        line_end=30,
        raw_code="def process_data(items: list[dict]) -> list[str]:\n    \"\"\"Process data items.\"\"\"\n    result = []\n    for item in items:\n        result.append(item['name'])\n    return result",
        signature="def process_data(items: list[dict]) -> list[str]",
        docstring="Process data items.",
        level=2,
        parent_id="myorg:myrepo:main:src/utils.py:file:utils.py:1",
        exceptions_raised=["KeyError"],
    )


@pytest.fixture
def class_element() -> CodeElement:
    return CodeElement(
        element_id="myorg:myrepo:main:src/models.py:class:UserService:5",
        scope="myorg",
        repository="myrepo",
        username="main",
        relative_path="src/models.py",
        element_type="class",
        name="UserService",
        language="python",
        line_start=5,
        line_end=50,
        raw_code="class UserService:\n    def __init__(self, db):\n        self.db = db",
        docstring="Service for user management.",
        decorators=["dataclass"],
        level=1,
        parent_id="myorg:myrepo:main:src/models.py:file:models.py:1",
        base_classes=["BaseService"],
        class_attributes=[{"name": "db"}, {"name": "cache"}],
    )


@pytest.fixture
def file_element() -> CodeElement:
    return CodeElement(
        element_id="myorg:myrepo:main:src/utils.py:file:utils.py:1",
        scope="myorg",
        repository="myrepo",
        username="main",
        relative_path="src/utils.py",
        element_type="file",
        name="utils.py",
        language="python",
        line_start=1,
        line_end=100,
        raw_code='"""Utility functions for data processing."""\nimport os\nimport json',
        level=0,
        imports=[
            Import(name="os", module="os", alias=None, line=2),
            Import(name="json", module="json", alias=None, line=3),
        ],
    )


@pytest.fixture
def method_element() -> CodeElement:
    return CodeElement(
        element_id="myorg:myrepo:main:src/models.py:method:get_user:20",
        scope="myorg",
        repository="myrepo",
        username="main",
        relative_path="src/models.py",
        element_type="method",
        name="get_user",
        language="python",
        line_start=20,
        line_end=35,
        raw_code="def get_user(self, user_id: int) -> User:\n    return self.db.query(User).get(user_id)",
        signature="def get_user(self, user_id: int) -> User",
        level=2,
        parent_id="myorg:myrepo:main:src/models.py:class:UserService:5",
        attributes_modified=["cache"],
        calls=[Call(name="query", receiver="self.db", line=21)],
    )


@pytest.fixture
def sample_doc() -> dict:
    """Sample OpenSearch document for element reconstruction."""
    return {
        "element_id": "myorg:myrepo:main:src/utils.py:function:process_data:10",
        "scope": "myorg",
        "repository": "myrepo",
        "username": "main",
        "relative_path": "src/utils.py",
        "element_type": "function",
        "name": "process_data",
        "language": "python",
        "line_start": 10,
        "line_end": 30,
        "raw_code": "def process_data(items):\n    return [i['name'] for i in items]",
        "signature": "def process_data(items)",
        "docstring": "Process data items.",
        "level": 2,
        "parent_id": "myorg:myrepo:main:src/utils.py:file:utils.py:1",
        "decorators": ["staticmethod"],
        "visibility": "public",
        "is_async": False,
        "is_test": False,
        "imports": [{"name": "os", "module": "os", "alias": None, "line": 1}],
        "calls": [
            {"name": "append", "receiver": "result", "line": 12, "category": "unknown"},
        ],
        "exceptions_raised": ["KeyError"],
        "context_usages": ["used in main.py to transform input"],
        "summary": "Extracts names from a list of dicts.",
    }


@pytest.fixture
def good_scores() -> CriteriaScores:
    return CriteriaScores(
        scores={
            "operation": 9,
            "interface": 8,
            "usage_scenarios": 8,
            "side_effects": 9,
            "edge_cases": 7,
            "no_context_repeat": 9,
        },
        notes="Good summary overall.",
    )


@pytest.fixture
def low_scores() -> CriteriaScores:
    return CriteriaScores(
        scores={
            "operation": 7,
            "interface": 5,
            "usage_scenarios": 4,
            "side_effects": 6,
            "edge_cases": 3,
            "no_context_repeat": 8,
        },
        notes="Weak on edge cases and usage scenarios.",
    )


@pytest.fixture
def baseline_prompt() -> PromptVersion:
    return PromptVersion(
        system_prompt="You describe functions for AI agents.",
        user_template="Function: {function_name}\nSignature: {signature}\n\nCode:\n{code}",
        iteration=0,
        reasoning="Production baseline prompt",
    )


# =============================================================================
# DATA CLASS TESTS
# =============================================================================


class TestPromptVersion:
    def test_to_dict(self, baseline_prompt: PromptVersion) -> None:
        d = baseline_prompt.to_dict()
        assert d["iteration"] == 0
        assert d["system_prompt"] == "You describe functions for AI agents."
        assert d["reasoning"] == "Production baseline prompt"
        assert "user_template" in d

    def test_to_dict_roundtrip(self) -> None:
        pv = PromptVersion(
            system_prompt="sys",
            user_template="usr {code}",
            iteration=3,
            reasoning="improved edge cases",
        )
        d = pv.to_dict()
        assert d["iteration"] == 3
        assert d["reasoning"] == "improved edge cases"


class TestIterationResult:
    def test_avg_score(self, baseline_prompt: PromptVersion, good_scores: CriteriaScores) -> None:
        ir = IterationResult(
            iteration=0,
            prompt=baseline_prompt,
            summary="Processes data items.",
            scores=good_scores,
        )
        # (9+8+8+9+7+9) / 6 = 50/6 ≈ 8.33
        assert 8.3 <= ir.avg_score <= 8.4

    def test_avg_score_empty(self, baseline_prompt: PromptVersion) -> None:
        ir = IterationResult(
            iteration=0,
            prompt=baseline_prompt,
            summary="",
            scores=CriteriaScores(),
        )
        assert ir.avg_score == 0.0


class TestRALPHResult:
    def test_best_iteration(self, baseline_prompt: PromptVersion, good_scores: CriteriaScores, low_scores: CriteriaScores) -> None:
        result = RALPHResult(
            element_type="function",
            element_name="process_data",
            element_id="test:id",
            iterations=[
                IterationResult(0, baseline_prompt, "low", low_scores),
                IterationResult(1, baseline_prompt, "high", good_scores),
            ],
        )
        assert result.best_iteration == 1
        assert result.best_score > 8.0

    def test_best_iteration_empty(self) -> None:
        result = RALPHResult(
            element_type="function",
            element_name="test",
            element_id="test:id",
        )
        assert result.best_iteration == 0
        assert result.best_result is None
        assert result.best_score == 0.0


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestGetTemplateKey:
    def test_function(self) -> None:
        assert _get_template_key("function") == "function"

    def test_method(self) -> None:
        assert _get_template_key("method") == "method"

    def test_class(self) -> None:
        assert _get_template_key("class") == "class"

    def test_file(self) -> None:
        assert _get_template_key("file") == "file"

    def test_unknown_defaults_to_function(self) -> None:
        assert _get_template_key("unknown_type") == "function"

    def test_constant(self) -> None:
        assert _get_template_key("constant") == "constant"

    def test_variable(self) -> None:
        assert _get_template_key("variable") == "variable"


class TestGetRequiredPlaceholders:
    def test_function_placeholders(self) -> None:
        placeholders = _get_required_placeholders("function")
        assert "code" in placeholders
        assert "function_name" in placeholders
        assert "signature" in placeholders
        assert "file_summary" in placeholders

    def test_class_placeholders(self) -> None:
        placeholders = _get_required_placeholders("class")
        assert "code" in placeholders
        assert "class_name" in placeholders

    def test_file_placeholders(self) -> None:
        placeholders = _get_required_placeholders("file")
        assert "code" in placeholders
        assert "language" in placeholders
        assert "file_path" in placeholders


class TestReconstructElement:
    def test_basic_reconstruction(self, sample_doc: dict) -> None:
        element = _reconstruct_element(sample_doc)
        assert element.element_id == sample_doc["element_id"]
        assert element.name == "process_data"
        assert element.element_type == "function"
        assert element.language == "python"
        assert element.line_start == 10
        assert element.line_end == 30

    def test_imports_reconstruction(self, sample_doc: dict) -> None:
        element = _reconstruct_element(sample_doc)
        assert len(element.imports) == 1
        assert element.imports[0].name == "os"
        assert element.imports[0].module == "os"

    def test_calls_reconstruction(self, sample_doc: dict) -> None:
        element = _reconstruct_element(sample_doc)
        assert len(element.calls) == 1
        assert element.calls[0].name == "append"
        assert element.calls[0].receiver == "result"

    def test_missing_optional_fields(self) -> None:
        minimal_doc = {
            "element_id": "a:b:c:d:function:f:1",
            "element_type": "function",
            "name": "f",
            "line_start": 1,
        }
        element = _reconstruct_element(minimal_doc)
        assert element.name == "f"
        assert element.imports == []
        assert element.calls == []
        assert element.decorators == []
        assert element.context_usages == []

    def test_decorators_list(self, sample_doc: dict) -> None:
        element = _reconstruct_element(sample_doc)
        assert element.decorators == ["staticmethod"]

    def test_context_usages(self, sample_doc: dict) -> None:
        element = _reconstruct_element(sample_doc)
        assert len(element.context_usages) == 1
        assert "main.py" in element.context_usages[0]


class TestParseImprovementResponse:
    def test_valid_json(self) -> None:
        response = '{"reasoning": "improved edge cases", "system_prompt": "new sys", "user_template": "new usr {code}"}'
        result = _parse_improvement_response(response)
        assert result is not None
        assert result["reasoning"] == "improved edge cases"
        assert result["system_prompt"] == "new sys"
        assert result["user_template"] == "new usr {code}"

    def test_json_with_thinking_tags(self) -> None:
        response = '<think>Let me analyze...</think>{"reasoning": "r", "system_prompt": "s", "user_template": "u {code}"}'
        result = _parse_improvement_response(response)
        assert result is not None
        assert result["system_prompt"] == "s"

    def test_json_in_markdown_block(self) -> None:
        response = '```json\n{"reasoning": "r", "system_prompt": "s", "user_template": "u"}\n```'
        result = _parse_improvement_response(response)
        assert result is not None
        assert result["system_prompt"] == "s"

    def test_trailing_comma_fix(self) -> None:
        response = '{"reasoning": "r", "system_prompt": "s", "user_template": "u",}'
        result = _parse_improvement_response(response)
        assert result is not None

    def test_missing_required_key(self) -> None:
        response = '{"reasoning": "r", "system_prompt": "s"}'
        result = _parse_improvement_response(response)
        assert result is None

    def test_no_json(self) -> None:
        response = "This is just text without any JSON."
        result = _parse_improvement_response(response)
        assert result is None

    def test_unclosed_thinking_tag(self) -> None:
        response = '<think>still thinking...{"reasoning": "r", "system_prompt": "s", "user_template": "u"}'
        result = _parse_improvement_response(response)
        # The unclosed tag pattern removes from <think> to end, so no JSON remains
        # unless the JSON is after the tag content
        # In this case the tag removal eats the JSON too
        assert result is None

    def test_json_after_closed_thinking(self) -> None:
        response = '<think>thinking</think>\n{"reasoning": "r", "system_prompt": "s", "user_template": "u"}'
        result = _parse_improvement_response(response)
        assert result is not None
        assert result["reasoning"] == "r"


class TestFormatUserTemplate:
    def test_function_template(self, function_element: CodeElement) -> None:
        template = "Function: {function_name}\nSignature: {signature}\n\nCode:\n{code}"
        parent_summaries = {"file": "Utility functions for data processing."}
        result = _format_user_template(template, function_element, parent_summaries)
        assert "process_data" in result
        assert "def process_data" in result

    def test_invalid_template_returns_empty(self, function_element: CodeElement) -> None:
        template = "Invalid: {nonexistent_placeholder}"
        result = _format_user_template(template, function_element, {})
        assert result == ""

    def test_class_template(self, class_element: CodeElement) -> None:
        from shared.ai.prompts import USER_PROMPTS
        template = USER_PROMPTS["class"]
        parent_summaries = {"file": "Data models for the application."}
        result = _format_user_template(template, class_element, parent_summaries)
        assert "UserService" in result
        assert "Data models" in result


# =============================================================================
# CONVERGENCE LOGIC TESTS
# =============================================================================


class TestShouldStop:
    def _make_iteration(self, iteration: int, avg: float) -> IterationResult:
        """Helper to create an IterationResult with a specific average score."""
        # Create scores that average to `avg`
        scores = CriteriaScores(scores={"criterion": int(avg * 10) / 10})
        # Override average via score count = 1
        scores.scores = {"c1": round(avg)}
        return IterationResult(
            iteration=iteration,
            prompt=PromptVersion("sys", "usr", iteration),
            summary="test",
            scores=scores,
        )

    def test_stop_at_max_iterations(self) -> None:
        iterations = [self._make_iteration(i, 5.0) for i in range(5)]
        stop, reason = PromptImprover._should_stop(iterations, target_score=8.0, max_iterations=5)
        assert stop
        assert "max iterations" in reason.lower()

    def test_stop_at_target_score(self) -> None:
        iterations = [self._make_iteration(0, 9.0)]
        stop, reason = PromptImprover._should_stop(iterations, target_score=8.0, max_iterations=5)
        assert stop
        assert "target" in reason.lower()

    def test_continue_below_target(self) -> None:
        iterations = [self._make_iteration(0, 6.0)]
        stop, _ = PromptImprover._should_stop(iterations, target_score=8.0, max_iterations=5)
        assert not stop

    def test_plateau_detection(self) -> None:
        iterations = [
            self._make_iteration(0, 7.0),
            self._make_iteration(1, 6.0),
            self._make_iteration(2, 5.0),
        ]
        stop, reason = PromptImprover._should_stop(iterations, target_score=9.0, max_iterations=10)
        assert stop
        assert "plateau" in reason.lower()

    def test_no_plateau_when_improving(self) -> None:
        iterations = [
            self._make_iteration(0, 5.0),
            self._make_iteration(1, 6.0),
            self._make_iteration(2, 7.0),
        ]
        stop, _ = PromptImprover._should_stop(iterations, target_score=9.0, max_iterations=10)
        assert not stop

    def test_empty_iterations(self) -> None:
        stop, _ = PromptImprover._should_stop([], target_score=8.0, max_iterations=5)
        assert not stop


# =============================================================================
# IMPROVEMENT PROMPT TEMPLATE TESTS
# =============================================================================


class TestImprovementPrompt:
    def test_template_has_required_placeholders(self) -> None:
        """Ensure the IMPROVEMENT_PROMPT template has all expected placeholders."""
        required = [
            "source_code",
            "current_system_prompt",
            "current_user_template",
            "summary",
            "scores_text",
            "eval_notes",
            "criteria_text",
            "required_placeholders",
            "sentence_range",
        ]
        for placeholder in required:
            assert f"{{{placeholder}}}" in IMPROVEMENT_PROMPT, f"Missing placeholder: {placeholder}"

    def test_template_formats_without_error(self) -> None:
        """Ensure the template can be formatted with sample data."""
        result = IMPROVEMENT_PROMPT.format(
            source_code="def foo(): pass",
            current_system_prompt="Describe this function.",
            current_user_template="Code: {code}",
            summary="Does nothing.",
            scores_text="operation: 5/10",
            eval_notes="Weak summary.",
            criteria_text="- operation: describe what it does",
            required_placeholders="{code}, {function_name}",
            sentence_range="2-3",
        )
        assert "def foo" in result
        assert "operation: 5/10" in result


# =============================================================================
# PROMPT IMPROVER ENGINE TESTS (with mocks)
# =============================================================================


class TestPromptImproverRun:
    """Integration-style tests for the RALPH loop using mocks."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        config = MagicMock()
        config.llm.get_summarize_model.return_value = MagicMock(
            name="qwen3.5:4b",
            provider="ollama",
            url="http://localhost:11434",
            api_key=None,
            get_litellm_model=MagicMock(return_value="ollama_chat/qwen3.5:4b"),
            get_api_base=MagicMock(return_value="http://localhost:11434"),
        )
        config.llm.summarize_temperature = 0.2
        config.llm.summarize_max_tokens = 512
        config.benchmark.eval_models = []
        return config

    @pytest.fixture
    def mock_repo(self, sample_doc: dict) -> MagicMock:
        repo = MagicMock()
        repo.get_document_by_id_or_hash.return_value = sample_doc
        # File summary for parent context
        repo.get_document.return_value = {
            "element_type": "file",
            "summary": "Utility functions for processing.",
        }
        repo.close = MagicMock()
        return repo

    @patch("shared.ai.prompt_improver.BenchmarkClient")
    @patch("shared.ai.prompt_improver.LLMClient", create=True)
    @patch("shared.db.store.Repository")
    def test_single_iteration_converges(
        self,
        mock_repo_cls: MagicMock,
        mock_llm_cls: MagicMock,
        mock_bench_cls: MagicMock,
        mock_config: MagicMock,
        sample_doc: dict,
    ) -> None:
        """Test that loop stops after 1 iteration when score meets target."""
        # Setup repo mock
        repo_instance = MagicMock()
        repo_instance.get_document_by_id_or_hash.return_value = sample_doc
        repo_instance.get_document.return_value = {
            "element_type": "file",
            "summary": "Utility functions.",
        }
        mock_repo_cls.return_value = repo_instance

        # Setup LLM mock - returns a good summary
        mock_llm_instance = MagicMock()
        mock_llm_instance.generate_from_messages.return_value = "Extracts name fields from a list of dicts, raising KeyError for missing keys."

        # Patch LLMClient.from_model_config
        with patch("shared.ai.prompt_improver.LLMClient") as llm_mod:
            llm_mod.from_model_config.return_value = mock_llm_instance

            # Setup eval mock - returns high scores
            bench_instance = MagicMock()
            mock_bench_cls.return_value = bench_instance
            bench_instance.generate.return_value = MagicMock(
                success=True,
                response='{"evaluations": {"A": {"operation": 9, "interface": 9, "usage_scenarios": 8, "side_effects": 9, "edge_cases": 8, "no_context_repeat": 9, "notes": "Good"}}}',
            )

            improver = PromptImprover(config=mock_config)
            result = improver.run(
                element_id="abc123",
                max_iterations=5,
                target_score=8.0,
            )

        assert result.converged
        assert len(result.iterations) == 1
        assert result.best_score >= 8.0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    def test_reconstruct_element_with_empty_imports(self) -> None:
        doc = {
            "element_id": "a:b:c:d:function:f:1",
            "element_type": "function",
            "name": "f",
            "line_start": 1,
            "imports": [],
            "calls": [],
        }
        element = _reconstruct_element(doc)
        assert element.imports == []
        assert element.calls == []

    def test_reconstruct_element_with_none_optional_fields(self) -> None:
        doc = {
            "element_id": "a:b:c:d:function:f:1",
            "element_type": "function",
            "name": "f",
            "line_start": 1,
            "decorators": None,
            "context_usages": None,
            "class_attributes": None,
            "base_classes": None,
        }
        element = _reconstruct_element(doc)
        assert element.decorators == []
        assert element.context_usages == []
        assert element.class_attributes is None
        assert element.base_classes is None

    def test_parse_improvement_response_empty_string(self) -> None:
        assert _parse_improvement_response("") is None

    def test_parse_improvement_response_nested_json(self) -> None:
        """Test that nested JSON in reasoning doesn't break parsing."""
        response = '{"reasoning": "Changed {code} handling", "system_prompt": "new", "user_template": "u {code}"}'
        result = _parse_improvement_response(response)
        assert result is not None

    def test_get_template_key_for_all_types(self) -> None:
        """All standard element types should return a valid template key."""
        for etype in ["file", "class", "function", "method", "constant", "variable", "interface", "trait", "enum", "type_alias", "import"]:
            key = _get_template_key(etype)
            assert key in ("file", "class", "function", "method", "constant", "variable", "interface", "trait", "enum", "type_alias", "import")
