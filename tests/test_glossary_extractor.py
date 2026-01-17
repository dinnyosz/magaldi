"""Tests for glossary term extraction."""

from __future__ import annotations

import pytest

from shared.ai.glossary.extractor import (
    extract_terms,
    split_name,
    COMMON_TERMS,
    aggregate_glossary_terms,
    GlossaryEntry,
)


class TestSplitName:
    """Tests for name splitting logic."""

    def test_splits_camel_case(self):
        """Test CamelCase splitting."""
        assert split_name("UserService") == ["user", "service"]

    def test_splits_pascal_case(self):
        """Test PascalCase splitting."""
        assert split_name("GetUserById") == ["get", "user", "by", "id"]

    def test_splits_snake_case(self):
        """Test snake_case splitting."""
        assert split_name("user_service") == ["user", "service"]

    def test_splits_mixed_case(self):
        """Test mixed CamelCase and snake_case."""
        assert split_name("get_UserById") == ["get", "user", "by", "id"]

    def test_handles_acronyms(self):
        """Test handling of acronyms like HTTP, API."""
        assert split_name("HTTPClient") == ["http", "client"]
        assert split_name("parseAPIResponse") == ["parse", "api", "response"]

    def test_handles_single_word(self):
        """Test single word names."""
        assert split_name("user") == ["user"]
        assert split_name("User") == ["user"]

    def test_handles_empty_string(self):
        """Test empty string."""
        assert split_name("") == []


class TestExtractTerms:
    """Tests for term extraction with filtering."""

    def test_filters_common_terms(self):
        """Test that common programming terms are filtered out."""
        # "get" and "service" are common, "user" is domain-specific
        result = extract_terms("getUserService")
        assert "user" in result
        assert "get" not in result
        assert "service" not in result

    def test_filters_single_char_terms(self):
        """Test that single character terms are filtered."""
        result = extract_terms("getA")
        assert "a" not in result

    def test_returns_unique_terms(self):
        """Test that duplicate terms are deduplicated."""
        result = extract_terms("userUserData")
        assert result.count("user") == 1

    def test_extracts_domain_terms(self):
        """Test extraction of domain-specific terms."""
        result = extract_terms("EmailValidationService")
        assert "email" in result
        assert "validation" in result
        assert "service" not in result

    def test_preserves_term_order(self):
        """Test that terms maintain appearance order."""
        result = extract_terms("registration_email_sender")
        assert result == ["registration", "email", "sender"]


class TestCommonTerms:
    """Tests for the common terms set."""

    def test_common_terms_includes_verbs(self):
        """Test that common verbs are in the filter set."""
        verbs = ["get", "set", "add", "remove", "delete", "update", "create"]
        for verb in verbs:
            assert verb in COMMON_TERMS

    def test_common_terms_includes_suffixes(self):
        """Test that architectural suffixes are in the filter set."""
        suffixes = ["service", "controller", "handler", "manager", "factory"]
        for suffix in suffixes:
            assert suffix in COMMON_TERMS


class TestAggregateGlossaryTerms:
    """Tests for glossary term aggregation."""

    def test_aggregates_from_multiple_elements(self):
        """Test aggregation across multiple elements."""
        elements = [
            {"element_id": "id1", "name": "UserService", "relative_path": "user.py"},
            {"element_id": "id2", "name": "UserController", "relative_path": "user.py"},
            {"element_id": "id3", "name": "EmailSender", "relative_path": "email.py"},
        ]

        result = aggregate_glossary_terms(elements)

        assert "user" in result
        assert result["user"].total_count == 2
        assert set(result["user"].element_ids) == {"id1", "id2"}
        assert result["user"].file_paths == ["user.py"]

    def test_tracks_file_paths(self):
        """Test that file paths are tracked correctly."""
        elements = [
            {"element_id": "id1", "name": "UserService", "relative_path": "services/user.py"},
            {"element_id": "id2", "name": "UserModel", "relative_path": "models/user.py"},
        ]

        result = aggregate_glossary_terms(elements)

        assert "user" in result
        assert set(result["user"].file_paths) == {"services/user.py", "models/user.py"}

    def test_returns_empty_for_no_domain_terms(self):
        """Test that elements with only common terms return empty."""
        elements = [
            {"element_id": "id1", "name": "GetService", "relative_path": "a.py"},
        ]

        result = aggregate_glossary_terms(elements)

        assert len(result) == 0

    def test_glossary_entry_dataclass(self):
        """Test GlossaryEntry dataclass structure."""
        entry = GlossaryEntry(
            term="user",
            total_count=5,
            element_ids=["id1", "id2"],
            file_paths=["a.py", "b.py"],
        )

        assert entry.term == "user"
        assert entry.total_count == 5
        assert len(entry.element_ids) == 2
