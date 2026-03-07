"""Tests for multi-user query merging in SearchRepository.

Verifies that MCP-facing search methods query BOTH the current user
AND "main" branches, so non-main users see complete results.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from shared.db.repositories.search import SearchRepository  # local import

# =============================================================================
# _build_username_filter() unit tests
# =============================================================================


class TestBuildUsernameFilter:
    """Unit tests for the static username filter builder."""

    def test_none_returns_none(self):
        """No username -> no filter."""
        assert SearchRepository._build_username_filter(None) is None

    def test_empty_string_returns_none(self):
        """Empty username -> no filter."""
        assert SearchRepository._build_username_filter("") is None

    def test_main_returns_simple_term(self):
        """Username 'main' -> simple term filter (no merge needed)."""
        result = SearchRepository._build_username_filter("main")
        assert result == {"term": {"username": "main"}}

    def test_non_main_returns_bool_should(self):
        """Non-main username -> bool/should merging user + main."""
        result = SearchRepository._build_username_filter("alice")
        assert result == {
            "bool": {
                "should": [
                    {"term": {"username": "alice"}},
                    {"term": {"username": "main"}},
                ],
                "minimum_should_match": 1,
            }
        }

    def test_different_users(self):
        """Various non-main usernames all produce the merge pattern."""
        for user in ["bob", "ci-bot", "feature-branch"]:
            result = SearchRepository._build_username_filter(user)
            assert result is not None
            assert "bool" in result
            should = result["bool"]["should"]
            usernames = [clause["term"]["username"] for clause in should]
            assert user in usernames
            assert "main" in usernames


# =============================================================================
# Integration: verify ES queries include multi-user filter
# =============================================================================


def _make_search_repo() -> tuple[SearchRepository, MagicMock]:
    """Create a SearchRepository with a mocked ES client."""
    base = MagicMock()
    base.backend_type = "opensearch"
    client = MagicMock()
    client.search.return_value = {"hits": {"hits": []}}
    base._get_client.return_value = client
    repo = SearchRepository(base)
    return repo, client


def _extract_query_body(client_mock: MagicMock) -> dict[str, Any]:
    """Extract the query body from the last client.search() call."""
    call_kwargs = client_mock.search.call_args
    return call_kwargs[1]["body"] if "body" in call_kwargs[1] else call_kwargs[0][1]


def _find_username_clauses(query_body: dict) -> list[dict]:
    """Recursively find all username-related clauses in a query body."""
    clauses: list[dict] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            # Direct term match
            if "term" in obj and "username" in obj["term"]:
                clauses.append(obj)
            # Bool/should pattern
            if "bool" in obj and "should" in obj["bool"]:
                for should_clause in obj["bool"]["should"]:
                    if isinstance(should_clause, dict) and "term" in should_clause and "username" in should_clause["term"]:
                            clauses.append(obj)
                            return  # Don't recurse into already-found
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(query_body)
    return clauses


class TestSearchByTextMultiUser:
    """Verify search_by_text builds multi-user queries."""

    def test_main_user_simple_filter(self):
        repo, client = _make_search_repo()
        repo.search_by_text("hello", username="main")
        body = _extract_query_body(client)
        # Should have a simple {"term": {"username": "main"}} somewhere
        username_clauses = _find_username_clauses(body)
        assert len(username_clauses) == 1
        assert username_clauses[0] == {"term": {"username": "main"}}

    def test_non_main_user_merges_branches(self):
        repo, client = _make_search_repo()
        repo.search_by_text("hello", username="alice")
        body = _extract_query_body(client)
        username_clauses = _find_username_clauses(body)
        assert len(username_clauses) == 1
        clause = username_clauses[0]
        assert "bool" in clause
        should = clause["bool"]["should"]
        usernames = {c["term"]["username"] for c in should}
        assert usernames == {"alice", "main"}

    def test_no_username_no_filter(self):
        repo, client = _make_search_repo()
        repo.search_by_text("hello")
        body = _extract_query_body(client)
        username_clauses = _find_username_clauses(body)
        assert len(username_clauses) == 0


class TestSearchByKeywordMultiUser:
    """Verify search_by_keyword builds multi-user queries."""

    def test_non_main_user_merges_branches(self):
        repo, client = _make_search_repo()
        repo.search_by_keyword("hello", username="bob")
        body = _extract_query_body(client)
        username_clauses = _find_username_clauses(body)
        assert len(username_clauses) == 1
        clause = username_clauses[0]
        assert "bool" in clause
        should = clause["bool"]["should"]
        usernames = {c["term"]["username"] for c in should}
        assert usernames == {"bob", "main"}


class TestSearchByRegexpMultiUser:
    """Verify search_by_regexp builds multi-user queries."""

    def test_non_main_user_merges_branches(self):
        repo, client = _make_search_repo()
        repo.search_by_regexp(".*test.*", username="charlie")
        body = _extract_query_body(client)
        username_clauses = _find_username_clauses(body)
        assert len(username_clauses) == 1
        clause = username_clauses[0]
        assert "bool" in clause


class TestSearchByWildcardMultiUser:
    """Verify search_by_wildcard builds multi-user queries."""

    def test_non_main_user_merges_branches(self):
        repo, client = _make_search_repo()
        repo.search_by_wildcard("*test*", username="dave")
        body = _extract_query_body(client)
        username_clauses = _find_username_clauses(body)
        assert len(username_clauses) == 1
        clause = username_clauses[0]
        assert "bool" in clause


class TestSearchByProximityMultiUser:
    """Verify search_by_proximity builds multi-user queries."""

    def test_non_main_user_merges_branches(self):
        repo, client = _make_search_repo()
        repo.search_by_proximity("foo bar", username="eve")
        body = _extract_query_body(client)
        username_clauses = _find_username_clauses(body)
        assert len(username_clauses) == 1
        clause = username_clauses[0]
        assert "bool" in clause


class TestFindElementsCallingMultiUser:
    """Verify find_elements_calling builds multi-user queries."""

    def test_non_main_user_merges_branches(self):
        repo, client = _make_search_repo()
        repo.find_elements_calling("some:element:id", username="frank")
        body = _extract_query_body(client)
        username_clauses = _find_username_clauses(body)
        assert len(username_clauses) == 1
        clause = username_clauses[0]
        assert "bool" in clause


class TestFindElementsImportingMultiUser:
    """Verify find_elements_importing builds multi-user queries."""

    def test_non_main_user_merges_branches(self):
        repo, client = _make_search_repo()
        repo.find_elements_importing("some.module", username="grace")
        body = _extract_query_body(client)
        username_clauses = _find_username_clauses(body)
        assert len(username_clauses) == 1
        clause = username_clauses[0]
        assert "bool" in clause


class TestFindCandidatesByNameMultiUser:
    """Verify find_candidates_by_name builds multi-user queries."""

    def test_non_main_user_merges_branches(self):
        repo, client = _make_search_repo()
        repo.find_candidates_by_name(
            "my_func", scope="org", repository="repo", username="heidi",
        )
        body = _extract_query_body(client)
        username_clauses = _find_username_clauses(body)
        assert len(username_clauses) == 1
        clause = username_clauses[0]
        assert "bool" in clause
        should = clause["bool"]["should"]
        usernames = {c["term"]["username"] for c in should}
        assert usernames == {"heidi", "main"}

    def test_main_user_simple_filter(self):
        repo, client = _make_search_repo()
        repo.find_candidates_by_name(
            "my_func", scope="org", repository="repo", username="main",
        )
        body = _extract_query_body(client)
        username_clauses = _find_username_clauses(body)
        assert len(username_clauses) == 1
        assert username_clauses[0] == {"term": {"username": "main"}}


# =============================================================================
# Internal methods should NOT use multi-user filter
# =============================================================================


class TestInternalMethodsSingleUser:
    """Internal pipeline methods should keep single-user filters."""

    def test_find_all_elements_with_calls_single_user(self):
        repo, client = _make_search_repo()
        repo.find_all_elements_with_calls("org", "repo", username="alice")
        body = _extract_query_body(client)
        username_clauses = _find_username_clauses(body)
        # Should be simple term filter, NOT the bool/should pattern
        assert len(username_clauses) == 1
        assert username_clauses[0] == {"term": {"username": "alice"}}

    def test_find_elements_with_unresolved_calls_single_user(self):
        repo, client = _make_search_repo()
        repo.find_elements_with_unresolved_calls("org", "repo", username="alice")
        body = _extract_query_body(client)
        username_clauses = _find_username_clauses(body)
        assert len(username_clauses) == 1
        assert username_clauses[0] == {"term": {"username": "alice"}}
