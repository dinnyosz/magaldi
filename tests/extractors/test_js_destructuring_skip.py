"""Tests that destructuring patterns are skipped in variable extraction.

Destructured variables (object_pattern, array_pattern) should not be extracted
as elements — they produce huge multi-line names that can overflow OpenSearch's
512-byte document ID limit.
"""

import pytest
import tree_sitter_javascript as ts_js
import tree_sitter_typescript as ts_ts
from tree_sitter import Language, Parser

from magaldi_core.extractors.javascript import extract_javascript_elements


class TestDestructuringSkip:
    """Verify destructuring patterns are skipped during variable extraction."""

    @pytest.fixture
    def js_parser(self):
        """Create a JavaScript parser."""
        return Parser(Language(ts_js.language()))

    @pytest.fixture
    def ts_parser(self):
        """Create a TypeScript parser."""
        return Parser(Language(ts_ts.language_typescript()))

    def test_object_destructuring_const_skipped(self, ts_parser):
        """Object destructuring in const should not produce a variable element."""
        code = """\
const {
    id = newUuid(),
    name = 'Test User',
    email = 'test@example.com',
    profileImagePath = '',
    shouldChangePassword = false,
    isAdmin = false,
    status = 'active',
} = defaults;
"""
        tree = ts_parser.parse(code.encode())
        elements = extract_javascript_elements(tree, code.split("\n"))

        # No variable/constant element with the full object pattern as name
        var_names = [e.name for e in elements if e.element_type in ("variable", "constant")]
        for name in var_names:
            assert "{" not in name, f"Object pattern leaked as variable name: {name}"

    def test_array_destructuring_const_skipped(self, ts_parser):
        """Array destructuring in const should not produce a variable element."""
        code = """\
const [first, second, ...rest] = items;
"""
        tree = ts_parser.parse(code.encode())
        elements = extract_javascript_elements(tree, code.split("\n"))

        var_names = [e.name for e in elements if e.element_type in ("variable", "constant")]
        for name in var_names:
            assert "[" not in name, f"Array pattern leaked as variable name: {name}"

    def test_object_destructuring_let_skipped(self, js_parser):
        """Object destructuring with let should not produce a variable element."""
        code = """\
let { x, y, z } = point;
"""
        tree = js_parser.parse(code.encode())
        elements = extract_javascript_elements(tree, code.split("\n"))

        var_names = [e.name for e in elements if e.element_type in ("variable", "constant")]
        for name in var_names:
            assert "{" not in name

    def test_object_destructuring_var_skipped(self, js_parser):
        """Object destructuring with var should not produce a variable element."""
        code = """\
var { a, b, c } = config;
"""
        tree = js_parser.parse(code.encode())
        elements = extract_javascript_elements(tree, code.split("\n"))

        var_names = [e.name for e in elements if e.element_type in ("variable", "constant")]
        for name in var_names:
            assert "{" not in name

    def test_normal_variables_still_extracted(self, ts_parser):
        """Non-destructured variables should still be extracted normally."""
        code = """\
const API_URL = 'https://api.example.com';
let counter = 0;
const config = { host: 'localhost', port: 3000 };
"""
        tree = ts_parser.parse(code.encode())
        elements = extract_javascript_elements(tree, code.split("\n"))

        var_names = {e.name for e in elements if e.element_type in ("variable", "constant")}
        assert "API_URL" in var_names
        assert "counter" in var_names
        assert "config" in var_names

    def test_arrow_function_with_destructured_param_still_works(self, ts_parser):
        """Arrow functions using destructured params should still be extracted."""
        code = """\
const handler = ({ req, res }) => {
    return res.json({ ok: true });
};
"""
        tree = ts_parser.parse(code.encode())
        elements = extract_javascript_elements(tree, code.split("\n"))

        func = next((e for e in elements if e.name == "handler"), None)
        assert func is not None
        assert func.element_type == "function"


class TestElementIdTruncation:
    """Verify element IDs are capped at 512 bytes for OpenSearch safety."""

    def test_normal_id_unchanged(self):
        from magaldi_core.parsers.base import generate_element_id

        eid = generate_element_id("scope", "repo", "user", "src/foo.ts", "function", "myFunc", 100)
        assert eid == "scope:repo:user:src/foo.ts:function:myFunc:100"

    def test_long_name_truncated_within_limit(self):
        from magaldi_core.parsers.base import generate_element_id

        long_name = "x" * 600
        eid = generate_element_id("scope", "repo", "user", "src/foo.ts", "variable", long_name, 100)
        assert len(eid.encode("utf-8")) <= 512
        # Should contain a hash suffix
        assert "_" in eid.split(":")[-2]  # name part has underscore before hash

    def test_truncated_id_preserves_uniqueness(self):
        from magaldi_core.parsers.base import generate_element_id

        name_a = "a" * 600
        name_b = "b" * 600
        eid_a = generate_element_id("s", "r", "u", "f.ts", "variable", name_a, 0)
        eid_b = generate_element_id("s", "r", "u", "f.ts", "variable", name_b, 0)
        assert eid_a != eid_b

    def test_multiline_object_pattern_as_name_stays_within_limit(self):
        """Simulate the exact immich failure scenario."""
        from magaldi_core.parsers.base import generate_element_id

        object_pattern_name = (
            "{\n    id = newUuid(),\n    name = 'Test User',\n    "
            "email = 'test@immich.cloud',\n    profileImagePath = '',\n    "
            "profileChangedAt = newDate(),\n    storageLabel = null,\n    "
            "shouldChangePassword = false,\n    isAdmin = false,\n    "
            "avatarColor = null,\n    createdAt = newDate(),\n    "
            "updatedAt = newDate(),\n    deletedAt = null,\n    oauthId = '',\n    "
            "quotaSizeInBytes = null,\n    quotaUsageInBytes = 0,\n    "
            "status = UserStatus.Active,\n    metadata = [],\n  }"
        )

        eid = generate_element_id(
            "test-repo", "immich", "main",
            "server/test/small.factory.ts",
            "variable", object_pattern_name, 4119,
        )
        assert len(eid.encode("utf-8")) <= 512
