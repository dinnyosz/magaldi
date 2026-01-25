"""Tests for extended code intelligence extraction."""

from magaldi_core.tree_sitter_manager import (
    Comment,
    associate_comments,
    extract_comments,
    extract_section_markers,
    extract_todos,
)


class TestTodoExtraction:
    """Tests for TODO comment extraction."""

    def test_extract_simple_todo(self):
        source = "# TODO: fix this bug"
        todos = extract_todos(source)
        assert len(todos) == 1
        assert todos[0].kind == "TODO"
        assert todos[0].text == "fix this bug"
        assert todos[0].line == 1

    def test_extract_todo_with_assignee(self):
        source = "# TODO(alice): review this code"
        todos = extract_todos(source)
        assert len(todos) == 1
        assert todos[0].assignee == "alice"
        assert todos[0].text == "review this code"

    def test_extract_todo_with_issue_ref(self):
        source = "# TODO #123: implement feature"
        todos = extract_todos(source)
        assert len(todos) == 1
        assert todos[0].issue_ref == "#123"

    def test_extract_fixme(self):
        source = "# FIXME: memory leak here"
        todos = extract_todos(source)
        assert len(todos) == 1
        assert todos[0].kind == "FIXME"

    def test_extract_multiple_todos(self):
        source = """# TODO: first thing
def foo():
    # FIXME: second thing
    pass
"""
        todos = extract_todos(source)
        assert len(todos) == 2
        assert todos[0].line == 1
        assert todos[1].line == 3

    def test_extract_todo_with_priority(self):
        source = "# TODO!!: urgent fix needed"
        todos = extract_todos(source)
        assert len(todos) == 1
        assert todos[0].priority == "high"


class TestSectionMarkerExtraction:
    """Tests for section marker extraction."""

    def test_extract_equals_style(self):
        source = "# === HELPERS ==="
        markers = extract_section_markers(source)
        assert len(markers) == 1
        assert markers[0].label == "HELPERS"
        assert markers[0].style == "equals"

    def test_extract_dashes_style(self):
        source = "# --- PRIVATE METHODS ---"
        markers = extract_section_markers(source)
        assert len(markers) == 1
        assert markers[0].label == "PRIVATE METHODS"
        assert markers[0].style == "dashes"

    def test_extract_multiple_markers(self):
        source = """# === IMPORTS ===
import os

# === HELPERS ===
def helper():
    pass
"""
        markers = extract_section_markers(source)
        assert len(markers) == 2
        assert markers[0].label == "IMPORTS"
        assert markers[1].label == "HELPERS"


class TestCommentExtraction:
    """Tests for comment extraction and association."""

    def test_extract_inline_comment(self):
        source = "x = 1  # set x to one"
        comments = extract_comments(source)
        assert len(comments) == 1
        assert comments[0].kind == "inline"
        assert "set x to one" in comments[0].text

    def test_extract_block_comment(self):
        source = "# This is a block comment\ndef foo(): pass"
        comments = extract_comments(source)
        assert len(comments) == 1
        assert comments[0].kind == "block"

    def test_associate_comment_above(self):
        comments = [Comment(text="Helper function", line=5, kind="block", position="above")]
        element_line = 6
        associated = associate_comments(element_line, comments, max_distance=3)
        assert len(associated) == 1

    def test_no_associate_distant_comment(self):
        comments = [Comment(text="Far away", line=1, kind="block", position="above")]
        element_line = 10
        associated = associate_comments(element_line, comments, max_distance=3)
        assert len(associated) == 0
