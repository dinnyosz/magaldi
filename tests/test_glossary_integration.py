"""Integration tests for the glossary feature workflow."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from shared.ai.glossary.extractor import (
    extract_terms,
    split_name,
    aggregate_glossary_terms,
    GlossaryEntry,
    COMMON_TERMS,
)
from shared.ai.glossary.linker import (
    compute_feature_associations,
    link_glossary_to_features,
    FeatureAssociation,
)
from magaldi_mcp.tools import (
    list_glossary,
    get_glossary_term,
    search_glossary,
    get_feature_members,
    search_features,
)


class TestGlossaryWorkflow:
    """Integration tests for the complete glossary workflow."""

    def test_full_extraction_to_aggregation_workflow(self):
        """Test extracting terms from elements and aggregating into glossary entries."""
        # Given: A set of code elements with meaningful names
        elements = [
            {"element_id": "e1", "name": "UserService", "relative_path": "src/user.py"},
            {"element_id": "e2", "name": "createUser", "relative_path": "src/user.py"},
            {"element_id": "e3", "name": "validateEmail", "relative_path": "src/email.py"},
            {"element_id": "e4", "name": "sendUserEmail", "relative_path": "src/email.py"},
        ]

        # When: Extracting and aggregating terms
        all_terms = {}
        for elem in elements:
            terms = extract_terms(elem["name"])
            for term in terms:
                if term not in all_terms:
                    all_terms[term] = {"count": 0, "element_ids": set(), "file_paths": set()}
                all_terms[term]["count"] += 1
                all_terms[term]["element_ids"].add(elem["element_id"])
                all_terms[term]["file_paths"].add(elem["relative_path"])

        # Then: Should have extracted domain terms (not common terms)
        assert "user" in all_terms
        assert "email" in all_terms
        assert "validate" not in all_terms  # Common term filtered
        assert "create" not in all_terms  # Common term filtered
        # Note: "send" is not a common term, so it is extracted

        # Verify counts
        assert all_terms["user"]["count"] == 3  # UserService, createUser, sendUserEmail
        assert all_terms["email"]["count"] == 2  # validateEmail, sendUserEmail

    def test_glossary_entry_creation(self):
        """Test creating GlossaryEntry from aggregated data."""
        # Given: Aggregated term data
        term_data = {
            "term": "user",
            "total_count": 3,
            "element_ids": ["e1", "e2", "e3"],
            "file_paths": ["src/user.py"],
        }

        # When: Creating GlossaryEntry
        entry = GlossaryEntry(
            term=term_data["term"],
            total_count=term_data["total_count"],
            element_ids=term_data["element_ids"],
            file_paths=term_data["file_paths"],
        )

        # Then: Entry should have correct values
        assert entry.term == "user"
        assert entry.total_count == 3
        assert len(entry.element_ids) == 3

    def test_aggregate_glossary_terms_workflow(self):
        """Test the aggregate_glossary_terms function end-to-end."""
        # Given: Elements with meaningful names
        elements = [
            {"element_id": "e1", "name": "UserService", "relative_path": "src/user.py"},
            {"element_id": "e2", "name": "createUser", "relative_path": "src/user.py"},
            {"element_id": "e3", "name": "validateEmail", "relative_path": "src/email.py"},
        ]

        # When: Aggregating terms
        result = aggregate_glossary_terms(elements)

        # Then: Should have correct entries
        assert "user" in result
        assert result["user"].total_count == 2  # UserService, createUser
        assert "email" in result
        assert result["email"].total_count == 1

    def test_feature_association_workflow(self):
        """Test linking glossary terms to features."""
        # Given: A feature with members and glossary terms
        feature = {
            "feature_id": "f1",
            "label": "User Management",
            "member_ids": ["e1", "e2", "e3"],
        }
        element_names = {
            "e1": "UserService",
            "e2": "createUser",
            "e3": "deleteUser",
        }
        glossary_terms = {"user"}

        # When: Computing feature associations
        associations = compute_feature_associations(feature, element_names, glossary_terms)

        # Then: Should have association for "user"
        assert len(associations) == 1
        assert associations[0].term == "user"
        assert associations[0].feature_id == "f1"
        assert associations[0].frequency == 3  # All 3 members have "user"
        assert associations[0].percentage == 100.0

    def test_multiple_terms_feature_linking(self):
        """Test linking multiple glossary terms to features."""
        # Given: Features and terms
        features = [
            {"feature_id": "f1", "label": "Auth", "member_ids": ["e1", "e2"]},
            {"feature_id": "f2", "label": "Email", "member_ids": ["e3", "e4"]},
        ]
        element_names = {
            "e1": "UserAuth",
            "e2": "LoginUser",
            "e3": "EmailSender",
            "e4": "validateEmail",
        }
        glossary_terms = {"user", "email", "auth", "login"}

        # When: Linking all terms to features
        term_associations = link_glossary_to_features(glossary_terms, features, element_names)

        # Then: Each term should have its feature associations
        assert "user" in term_associations
        assert "email" in term_associations
        assert "auth" in term_associations
        assert "login" in term_associations

        # User appears in feature f1 only
        user_assocs = term_associations["user"]
        assert len(user_assocs) == 1
        assert user_assocs[0].feature_id == "f1"

        # Email appears in feature f2 only
        email_assocs = term_associations["email"]
        assert len(email_assocs) == 1
        assert email_assocs[0].feature_id == "f2"


class TestMCPToolsIntegration:
    """Integration tests for MCP tools with glossary data."""

    @pytest.fixture
    def mock_es_with_glossary(self):
        """Mock ES repository with glossary data."""
        es = MagicMock()

        # Glossary terms
        glossary_data = [
            {
                "term": "user",
                "total_count": 5,
                "element_ids": ["e1", "e2", "e3", "e4", "e5"],
                "file_paths": ["src/user.py", "src/auth.py"],
                "feature_associations": [
                    {"feature_id": "f1", "feature_label": "Auth", "frequency": 3, "total_members": 4, "percentage": 75.0},
                    {"feature_id": "f2", "feature_label": "User Management", "frequency": 2, "total_members": 3, "percentage": 66.7},
                ],
            },
            {
                "term": "email",
                "total_count": 2,
                "element_ids": ["e6", "e7"],
                "file_paths": ["src/email.py"],
                "feature_associations": [
                    {"feature_id": "f3", "feature_label": "Notifications", "frequency": 2, "total_members": 5, "percentage": 40.0},
                ],
            },
        ]

        es.get_glossary_terms.return_value = glossary_data

        def get_glossary_term_mock(scope, repository, term, username):
            return next((g for g in glossary_data if g["term"] == term), None)

        def search_glossary_mock(scope, repository, query, username):
            return [g for g in glossary_data if query.lower() in g["term"]]

        es.get_glossary_term.side_effect = get_glossary_term_mock
        es.search_glossary.side_effect = search_glossary_mock

        return es

    def test_list_glossary_returns_all_terms(self, mock_es_with_glossary):
        """list_glossary returns all glossary terms for repository."""
        result = list_glossary(
            mock_es_with_glossary,
            scope="test",
            repository="repo",
            username="main",
        )

        assert len(result) == 2
        terms = [r["term"] for r in result]
        assert "user" in terms
        assert "email" in terms

    def test_get_glossary_term_returns_details(self, mock_es_with_glossary):
        """get_glossary_term returns full term details with associations."""
        result = get_glossary_term(
            mock_es_with_glossary,
            scope="test",
            repository="repo",
            term="user",
            username="main",
        )

        assert result is not None
        assert result["term"] == "user"
        assert result["total_count"] == 5
        assert len(result["feature_associations"]) == 2

    def test_search_glossary_finds_matching_terms(self, mock_es_with_glossary):
        """search_glossary finds terms by partial match."""
        result = search_glossary(
            mock_es_with_glossary,
            scope="test",
            repository="repo",
            query="us",
            username="main",
        )

        assert len(result) == 1
        assert result[0]["term"] == "user"

    def test_get_feature_members_includes_glossary_terms(self, mock_es_with_glossary):
        """get_feature_members includes associated glossary terms."""
        # Setup feature document
        mock_es_with_glossary.get_document.side_effect = [
            # Feature document
            {"member_ids": ["e1", "e2"]},
            # First member
            {
                "element_id": "e1",
                "name": "UserService",
                "element_type": "class",
                "relative_path": "src/user.py",
                "line_start": 10,
            },
            # Second member
            {
                "element_id": "e2",
                "name": "createUser",
                "element_type": "function",
                "relative_path": "src/user.py",
                "line_start": 50,
            },
        ]

        # Setup glossary terms - only terms associated with this feature
        mock_es_with_glossary.get_glossary_terms.return_value = [
            {
                "term": "user",
                "total_count": 5,
                "feature_associations": [
                    {"feature_id": "test:repo:main:feature:1", "frequency": 2, "percentage": 100.0},
                ],
            },
        ]

        result = get_feature_members(
            mock_es_with_glossary,
            feature_id="test:repo:main:feature:1",
        )

        assert "members" in result
        assert "glossary_terms" in result
        assert len(result["members"]) == 2

    def test_search_features_with_glossary_term_filter(self, mock_es_with_glossary):
        """search_features can filter by glossary term."""
        # Setup search results
        mock_es_with_glossary.search_by_vector.return_value = []
        mock_es_with_glossary.search_by_keyword.return_value = [
            {
                "element_id": "f1",
                "cluster_label": "Auth",
                "summary": "Authentication features",
                "member_count": 4,
                "element_type": "feature",
            },
        ]

        # Setup glossary term with feature associations
        mock_es_with_glossary.get_glossary_term.return_value = {
            "term": "user",
            "feature_associations": [
                {"feature_id": "f1", "percentage": 75.0},
            ],
        }

        result = search_features(
            mock_es_with_glossary,
            embed_client=None,
            query="authentication",
            scope="test",
            repository="repo",
            glossary_term="user",
            min_percentage=50.0,
        )

        # Feature f1 should be included (user term has 75% presence)
        assert len(result) == 1
        assert result[0]["label"] == "Auth"


class TestTermExtractionEdgeCases:
    """Test edge cases in term extraction."""

    def test_camelcase_splitting(self):
        """CamelCase names are split correctly."""
        terms = extract_terms("UserServiceHandler")
        assert "handler" not in terms  # Common term
        # "user" should be present
        assert "user" in terms

    def test_snake_case_splitting(self):
        """snake_case names are split correctly."""
        terms = extract_terms("user_service_handler")
        assert "handler" not in terms  # Common term
        assert "user" in terms

    def test_mixed_case_splitting(self):
        """Mixed case names are handled."""
        terms = extract_terms("getUserByEmail")
        assert "get" not in terms  # Common term
        assert "by" not in terms  # Common term
        assert "email" in terms

    def test_single_word_extraction(self):
        """Single words are extracted if not common."""
        terms = extract_terms("Authentication")
        assert "authentication" in terms

    def test_common_terms_filtered(self):
        """Common programming terms are filtered out."""
        terms = extract_terms("getServiceHandler")
        assert "get" not in terms
        assert "service" not in terms
        assert "handler" not in terms
        # Result should be empty since all are common terms
        assert len(terms) == 0


class TestAggregationEdgeCases:
    """Test edge cases in glossary aggregation."""

    def test_aggregate_empty_elements(self):
        """Aggregation handles empty element list."""
        entries = aggregate_glossary_terms(elements=[])
        assert len(entries) == 0

    def test_aggregate_elements_with_no_meaningful_terms(self):
        """Aggregation handles elements with only common terms."""
        elements = [
            {"element_id": "e1", "name": "get", "relative_path": "src/a.py"},
            {"element_id": "e2", "name": "set", "relative_path": "src/a.py"},
        ]
        entries = aggregate_glossary_terms(elements=elements)
        # All terms are common, so no glossary entries
        assert len(entries) == 0


class TestFeatureAssociationEdgeCases:
    """Test edge cases in feature-glossary linking."""

    def test_feature_with_no_matching_glossary_terms(self):
        """Feature with elements that have no glossary terms."""
        feature = {
            "feature_id": "f1",
            "label": "Utils",
            "member_ids": ["e1", "e2"],
        }
        element_names = {
            "e1": "GetService",  # Only common terms
            "e2": "SetHandler",  # Only common terms
        }
        glossary_terms = {"user", "email"}

        associations = compute_feature_associations(feature, element_names, glossary_terms)
        assert associations == []

    def test_multiple_features_same_term(self):
        """Term appearing in multiple features at different percentages."""
        features = [
            {"feature_id": "f1", "label": "Auth", "member_ids": ["e1", "e2"]},
            {"feature_id": "f2", "label": "Profile", "member_ids": ["e3", "e4", "e5"]},
        ]
        element_names = {
            "e1": "UserLogin",   # f1: user appears 2/2 = 100%
            "e2": "UserLogout",
            "e3": "UserProfile",  # f2: user appears 1/3 = 33.3%
            "e4": "Settings",
            "e5": "Preferences",
        }
        glossary_terms = {"user"}

        term_associations = link_glossary_to_features(glossary_terms, features, element_names)

        user_assocs = term_associations["user"]
        assert len(user_assocs) == 2

        # Sort by percentage to get predictable order
        user_assocs.sort(key=lambda a: a.percentage, reverse=True)

        assert user_assocs[0].feature_id == "f1"
        assert user_assocs[0].percentage == 100.0

        assert user_assocs[1].feature_id == "f2"
        assert user_assocs[1].percentage == pytest.approx(33.3, rel=0.1)


class TestEndToEndWorkflow:
    """End-to-end tests simulating complete glossary workflow."""

    def test_complete_workflow_from_elements_to_feature_linking(self):
        """Test the complete workflow from element extraction to feature linking."""
        # Step 1: Start with code elements
        elements = [
            {"element_id": "e1", "name": "UserRegistration", "relative_path": "auth/register.py"},
            {"element_id": "e2", "name": "UserAuthentication", "relative_path": "auth/login.py"},
            {"element_id": "e3", "name": "sendRegistrationEmail", "relative_path": "email/sender.py"},
            {"element_id": "e4", "name": "EmailNotification", "relative_path": "email/notify.py"},
            {"element_id": "e5", "name": "PasswordReset", "relative_path": "auth/reset.py"},
        ]

        # Step 2: Aggregate glossary terms
        glossary = aggregate_glossary_terms(elements)

        # Verify glossary extraction
        assert "user" in glossary
        assert "registration" in glossary
        assert "authentication" in glossary
        assert "email" in glossary
        assert "notification" in glossary
        assert "password" in glossary
        assert "reset" in glossary

        # Step 3: Define features (as would be created by clustering)
        features = [
            {"feature_id": "f1", "label": "Authentication", "member_ids": ["e1", "e2", "e5"]},
            {"feature_id": "f2", "label": "Email", "member_ids": ["e3", "e4"]},
        ]
        element_names = {e["element_id"]: e["name"] for e in elements}

        # Step 4: Link glossary to features
        glossary_terms = set(glossary.keys())
        term_to_features = link_glossary_to_features(glossary_terms, features, element_names)

        # Verify feature associations
        # "user" should be associated with f1 (appears in e1, e2 = 2/3 = 66.7%)
        assert len(term_to_features["user"]) == 1
        assert term_to_features["user"][0].feature_id == "f1"
        assert term_to_features["user"][0].percentage == pytest.approx(66.7, rel=0.1)

        # "email" should be associated with f2 (appears in e3, e4 = 2/2 = 100%)
        assert len(term_to_features["email"]) == 1
        assert term_to_features["email"][0].feature_id == "f2"
        assert term_to_features["email"][0].percentage == 100.0

        # "registration" appears in f1 (e1) and f2 (e3)
        registration_assocs = term_to_features["registration"]
        assert len(registration_assocs) == 2
        feature_ids = {a.feature_id for a in registration_assocs}
        assert feature_ids == {"f1", "f2"}
