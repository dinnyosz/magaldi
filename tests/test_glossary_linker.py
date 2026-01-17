# tests/test_glossary_linker.py
"""Tests for glossary-feature linking."""

from __future__ import annotations

import pytest

from shared.ai.glossary.linker import (
    compute_feature_associations,
    link_glossary_to_features,
    FeatureAssociation,
)


class TestComputeFeatureAssociations:
    """Tests for computing feature-glossary associations."""

    def test_computes_frequency_and_percentage(self):
        """Test that frequency and percentage are computed correctly."""
        feature = {
            "feature_id": "feat1",
            "label": "user_auth",
            "member_ids": ["id1", "id2", "id3", "id4", "id5"],
        }

        # Mock element names - 3 out of 5 contain "user"
        element_names = {
            "id1": "UserService",
            "id2": "UserController",
            "id3": "AuthManager",
            "id4": "UserValidator",
            "id5": "SessionHandler",
        }

        glossary_terms = {"user", "auth", "session", "validator"}

        result = compute_feature_associations(feature, element_names, glossary_terms)

        user_assoc = next((a for a in result if a.term == "user"), None)
        assert user_assoc is not None
        assert user_assoc.frequency == 3
        assert user_assoc.total_members == 5
        assert user_assoc.percentage == 60.0

    def test_only_includes_existing_glossary_terms(self):
        """Test that only terms in the glossary are included."""
        feature = {
            "feature_id": "feat1",
            "label": "test",
            "member_ids": ["id1"],
        }

        element_names = {"id1": "FooBarBaz"}
        glossary_terms = {"foo"}  # Only "foo" is in glossary

        result = compute_feature_associations(feature, element_names, glossary_terms)

        terms = [a.term for a in result]
        assert "foo" in terms
        assert "bar" not in terms
        assert "baz" not in terms

    def test_handles_empty_feature(self):
        """Test handling of feature with no members."""
        feature = {
            "feature_id": "feat1",
            "label": "empty",
            "member_ids": [],
        }

        result = compute_feature_associations(feature, {}, {"user"})

        assert result == []

    def test_returns_sorted_by_frequency(self):
        """Test that results are sorted by frequency descending."""
        feature = {
            "feature_id": "feat1",
            "label": "test",
            "member_ids": ["id1", "id2", "id3"],
        }

        element_names = {
            "id1": "EmailUserNotification",
            "id2": "EmailSender",
            "id3": "UserProfile",
        }

        glossary_terms = {"email", "user", "notification", "sender", "profile"}

        result = compute_feature_associations(feature, element_names, glossary_terms)

        # email appears in 2 elements, user in 2, others in 1
        frequencies = [a.frequency for a in result]
        assert frequencies == sorted(frequencies, reverse=True)

    def test_handles_missing_element_names(self):
        """Test handling of member IDs not in element_names."""
        feature = {
            "feature_id": "feat1",
            "label": "test",
            "member_ids": ["id1", "id2", "id3"],  # Only id1 has a name
        }
        element_names = {"id1": "UserService"}
        glossary_terms = {"user"}

        result = compute_feature_associations(feature, element_names, glossary_terms)

        # Should still work, with frequency 1 (only from id1)
        assert len(result) == 1
        assert result[0].frequency == 1
        assert result[0].total_members == 3


class TestLinkGlossaryToFeatures:
    """Tests for linking glossary terms to features."""

    def test_links_terms_to_multiple_features(self):
        """Test that terms are correctly linked across features."""
        glossary_terms = {"user", "auth"}
        features = [
            {"feature_id": "f1", "label": "auth", "member_ids": ["id1"]},
            {"feature_id": "f2", "label": "users", "member_ids": ["id2"]},
        ]
        element_names = {"id1": "UserAuth", "id2": "UserProfile"}

        result = link_glossary_to_features(glossary_terms, features, element_names)

        assert "user" in result
        assert len(result["user"]) == 2  # Appears in both features

    def test_returns_empty_list_for_unused_terms(self):
        """Test terms with no feature associations."""
        glossary_terms = {"user", "payment"}
        features = [{"feature_id": "f1", "label": "auth", "member_ids": ["id1"]}]
        element_names = {"id1": "UserService"}

        result = link_glossary_to_features(glossary_terms, features, element_names)

        assert result["payment"] == []  # No associations
