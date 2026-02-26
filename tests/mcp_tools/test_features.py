"""Tests for MCP tools - features category."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from magaldi_mcp.tools import (
    get_feature_members,
    list_features,
    search_features,
)

# =============================================================================
# SEARCH FEATURES TESTS
# =============================================================================


class TestSearchFeatures:
    """Tests for search_features function."""

    def test_search_features_returns_results(self, mock_repo, mock_embed_client):
        """Test search_features returns formatted results."""
        mock_repo.search_by_vector.return_value = [
            {
                "element_id": "feature1",
                "name": "authentication",
                "element_type": "feature",
                "cluster_label": "authentication",
                "summary": "Auth feature",
                "member_count": 5,
            }
        ]

        result = search_features(
            repo=mock_repo,
            embed_client=mock_embed_client,
            query="authentication",
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["label"] == "authentication"
        assert result[0]["type"] == "feature"

    def test_search_features_falls_back_to_keyword(self, mock_repo):
        """Test search_features falls back to keyword search."""
        mock_repo.search_by_keyword.return_value = [
            {
                "element_id": "feature1",
                "name": "auth",
                "element_type": "feature",
                "member_count": 3,
            }
        ]

        result = search_features(
            repo=mock_repo,
            embed_client=None,
            query="auth",
        )

        assert isinstance(result, list)

    def test_search_features_includes_subfeature_parent_info(self, mock_repo, mock_embed_client):
        """Test search_features includes parent info for subfeatures."""
        mock_repo.search_by_vector.return_value = [
            {
                "element_id": "subfeature1",
                "name": "token_validation",
                "element_type": "subfeature",
                "summary": "Validates JWT tokens",
                "member_count": 2,
                "parent_feature_label": "authentication",
                "parent_feature_summary": "Auth system",
            }
        ]

        result = search_features(
            repo=mock_repo,
            embed_client=mock_embed_client,
            query="token",
        )

        assert len(result) == 1
        assert result[0]["type"] == "subfeature"
        assert result[0]["parent_feature_label"] == "authentication"


# =============================================================================
# FIND SIMILAR EXTENDED TESTS
# =============================================================================

# =============================================================================
# SEARCH FEATURES GLOSSARY FILTER TESTS
# =============================================================================


class TestSearchFeaturesGlossaryFilter:
    """Tests for search_features glossary filtering."""

    @pytest.fixture
    def mock_es(self):
        repo = MagicMock()
        repo.get_document_by_id_or_hash.side_effect = lambda x: repo.get_document(x)
        return repo

    @pytest.fixture
    def mock_embed(self):
        embed = MagicMock()
        embed.embed_single.return_value = [0.1] * 768
        return embed

    def test_filters_by_glossary_term(self, mock_es, mock_embed):
        """Features are filtered to those associated with glossary term."""
        # Setup: glossary term has feature associations
        mock_es.get_glossary_term.return_value = {
            "term": "user",
            "feature_associations": [
                {"feature_id": "f1", "percentage": 50.0},
                {"feature_id": "f2", "percentage": 20.0},
            ],
        }
        # Setup: search returns 3 features
        mock_es.search_by_vector.return_value = [
            {"element_id": "f1", "cluster_label": "Auth", "member_count": 10},
            {"element_id": "f2", "cluster_label": "Users", "member_count": 5},
            {"element_id": "f3", "cluster_label": "Email", "member_count": 8},
        ]

        result = search_features(
            mock_es,
            mock_embed,
            query="test",
            scope="s",
            repository="r",
            glossary_term="user",
        )

        # Only f1 and f2 should be returned (they're in glossary associations)
        assert len(result) == 2
        labels = [r["label"] for r in result]
        assert "Auth" in labels
        assert "Users" in labels
        assert "Email" not in labels

    def test_filters_by_min_percentage(self, mock_es, mock_embed):
        """Features below min_percentage are filtered out."""
        mock_es.get_glossary_term.return_value = {
            "term": "user",
            "feature_associations": [
                {"feature_id": "f1", "percentage": 50.0},
                {"feature_id": "f2", "percentage": 20.0},
            ],
        }
        mock_es.search_by_vector.return_value = [
            {"element_id": "f1", "cluster_label": "Auth", "member_count": 10},
            {"element_id": "f2", "cluster_label": "Users", "member_count": 5},
        ]

        result = search_features(
            mock_es,
            mock_embed,
            query="test",
            scope="s",
            repository="r",
            glossary_term="user",
            min_percentage=30.0,
        )

        # Only f1 (50%) should pass the 30% threshold
        assert len(result) == 1
        assert result[0]["label"] == "Auth"

    def test_no_filter_when_glossary_term_not_provided(self, mock_es, mock_embed):
        """Without glossary_term, all results are returned."""
        mock_es.search_by_vector.return_value = [
            {"element_id": "f1", "cluster_label": "Auth", "member_count": 10},
            {"element_id": "f2", "cluster_label": "Users", "member_count": 5},
        ]

        result = search_features(
            mock_es,
            mock_embed,
            query="test",
            scope="s",
            repository="r",
        )

        assert len(result) == 2
        mock_es.get_glossary_term.assert_not_called()

    def test_returns_empty_when_glossary_term_not_found(self, mock_es, mock_embed):
        """When glossary term doesn't exist, return empty results."""
        mock_es.get_glossary_term.return_value = None
        mock_es.search_by_vector.return_value = [
            {"element_id": "f1", "cluster_label": "Auth", "member_count": 10},
        ]

        result = search_features(
            mock_es,
            mock_embed,
            query="test",
            scope="s",
            repository="r",
            glossary_term="nonexistent",
        )

        assert len(result) == 0


# =============================================================================
# GET FEATURE MEMBERS GLOSSARY TESTS
# =============================================================================

# =============================================================================
# LIST FEATURES TESTS
# =============================================================================


class TestListFeatures:
    """Tests for list_features function."""

    def test_list_features_returns_combined_list(self, mock_repo):
        """Test list_features combines features and subfeatures."""
        mock_repo.get_features.return_value = [
            {"label": "auth", "member_count": 10},
        ]
        mock_repo.get_subfeatures.return_value = [
            {"label": "token", "member_count": 3},
        ]

        result = list_features(
            repo=mock_repo,
            scope="github",
            repository="repo",
        )

        assert len(result) == 2
        # Should be sorted by member_count descending
        assert result[0]["member_count"] >= result[1]["member_count"]
        assert result[0]["type"] == "feature"
        assert result[1]["type"] == "subfeature"


# =============================================================================
# GET FEATURE MEMBERS TESTS
# =============================================================================

# =============================================================================
# GET FEATURE MEMBERS TESTS
# =============================================================================


class TestGetFeatureMembers:
    """Tests for get_feature_members function."""

    def test_get_feature_members_returns_members(self, mock_repo):
        """Test get_feature_members returns member elements."""
        mock_repo.get_document.side_effect = [
            # Feature document
            {
                "element_id": "feature1",
                "name": "authentication",
                "member_ids": ["id1", "id2"],
            },
            # First member
            {
                "element_id": "id1",
                "name": "login",
                "element_type": "function",
                "relative_path": "auth.py",
                "line_start": 10,
                "summary": "Login function",
            },
            # Second member
            {
                "element_id": "id2",
                "name": "logout",
                "element_type": "function",
                "relative_path": "auth.py",
                "line_start": 20,
            },
        ]
        mock_repo.get_glossary_terms.return_value = []

        result = get_feature_members(repo=mock_repo, feature_id="feature1")

        assert "members" in result
        assert "glossary_terms" in result
        assert len(result["members"]) == 2
        assert result["members"][0]["name"] == "login"
        assert result["members"][1]["name"] == "logout"

    def test_get_feature_members_not_found(self, mock_repo):
        """Test get_feature_members raises when feature not found."""
        mock_repo.get_document.return_value = None

        with pytest.raises(ValueError, match="not found"):
            get_feature_members(repo=mock_repo, feature_id="nonexistent")

    def test_get_feature_members_empty(self, mock_repo):
        """Test get_feature_members with no members."""
        mock_repo.get_document.return_value = {
            "element_id": "feature1",
            "name": "empty_feature",
            "member_ids": [],
        }

        result = get_feature_members(repo=mock_repo, feature_id="feature1")

        assert result == {"members": [], "glossary_terms": []}


# =============================================================================
# FIND FILES EXTENDED TESTS
# =============================================================================

# =============================================================================
# GET FEATURE MEMBERS GLOSSARY TESTS
# =============================================================================


class TestGetFeatureMembersGlossary:
    """Tests for get_feature_members glossary_terms field."""

    @pytest.fixture
    def mock_es(self):
        repo = MagicMock()
        repo.get_document_by_id_or_hash.side_effect = lambda x: repo.get_document(x)
        return repo

    def test_returns_glossary_terms_for_feature(self, mock_es):
        """Feature members response includes associated glossary terms."""
        # Setup feature
        mock_es.get_document.side_effect = [
            # Feature document
            {"member_ids": ["elem1", "elem2"]},
            # Member 1
            {"element_id": "elem1", "name": "UserService", "element_type": "class"},
            # Member 2
            {"element_id": "elem2", "name": "createUser", "element_type": "function"},
        ]
        # Setup glossary terms
        mock_es.get_glossary_terms.return_value = [
            {
                "term": "user",
                "feature_associations": [
                    {
                        "feature_id": "scope:repo:main:feature:1",
                        "frequency": 2,
                        "percentage": 100.0,
                    },
                ],
            },
            {
                "term": "email",
                "feature_associations": [
                    {"feature_id": "scope:repo:main:feature:2", "frequency": 1, "percentage": 50.0},
                ],
            },
        ]

        result = get_feature_members(mock_es, "scope:repo:main:feature:1")

        assert "members" in result
        assert "glossary_terms" in result
        assert len(result["glossary_terms"]) == 1
        assert result["glossary_terms"][0]["term"] == "user"
        assert result["glossary_terms"][0]["percentage"] == 100.0

    def test_glossary_terms_sorted_by_percentage(self, mock_es):
        """Glossary terms are sorted by percentage descending."""
        mock_es.get_document.side_effect = [
            {"member_ids": ["elem1"]},
            {"element_id": "elem1", "name": "Test", "element_type": "class"},
        ]
        mock_es.get_glossary_terms.return_value = [
            {
                "term": "low",
                "feature_associations": [
                    {"feature_id": "s:r:main:feature:1", "frequency": 1, "percentage": 20.0},
                ],
            },
            {
                "term": "high",
                "feature_associations": [
                    {"feature_id": "s:r:main:feature:1", "frequency": 5, "percentage": 80.0},
                ],
            },
        ]

        result = get_feature_members(mock_es, "s:r:main:feature:1")

        assert result["glossary_terms"][0]["term"] == "high"
        assert result["glossary_terms"][1]["term"] == "low"

    def test_empty_glossary_terms_when_no_associations(self, mock_es):
        """Returns empty glossary_terms when feature has no associations."""
        mock_es.get_document.side_effect = [
            {"member_ids": ["elem1"]},
            {"element_id": "elem1", "name": "Test", "element_type": "class"},
        ]
        mock_es.get_glossary_terms.return_value = []

        result = get_feature_members(mock_es, "s:r:main:feature:1")

        assert result["glossary_terms"] == []

    def test_members_still_returned(self, mock_es):
        """Members are still returned in the new format."""
        mock_es.get_document.side_effect = [
            {"member_ids": ["elem1"]},
            {
                "element_id": "elem1",
                "name": "MyClass",
                "element_type": "class",
                "relative_path": "src/file.py",
                "line_start": 10,
            },
        ]
        mock_es.get_glossary_terms.return_value = []

        result = get_feature_members(mock_es, "s:r:main:feature:1")

        assert len(result["members"]) == 1
        assert result["members"][0]["name"] == "MyClass"

    def test_empty_members_returns_empty_dict(self, mock_es):
        """Empty feature returns proper structure with empty lists."""
        mock_es.get_document.return_value = {
            "element_id": "s:r:main:feature:1",
            "member_ids": [],
        }
        mock_es.get_glossary_terms.return_value = []

        result = get_feature_members(mock_es, "s:r:main:feature:1")

        assert result == {"members": [], "glossary_terms": []}

    def test_handles_malformed_feature_id(self, mock_es):
        """Handles feature_id with fewer than 3 parts gracefully."""
        mock_es.get_document.side_effect = [
            {"member_ids": ["elem1"]},
            {"element_id": "elem1", "name": "Test", "element_type": "class"},
        ]
        # get_glossary_terms should not be called with invalid parts

        result = get_feature_members(mock_es, "invalid")

        # Should still return members but with empty glossary terms
        assert "members" in result
        assert "glossary_terms" in result
        assert result["glossary_terms"] == []


# =============================================================================
# PATTERN SEARCH TESTS
# =============================================================================

