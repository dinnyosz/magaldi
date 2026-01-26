"""Tests for glossary Elasticsearch operations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shared.db.elasticsearch import INDEX_NAME, ElasticsearchRepository

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = MagicMock()
    config.elasticsearch.host = "localhost"
    config.elasticsearch.port = 9200
    config.elasticsearch.scheme = "http"
    return config


@pytest.fixture
def mock_es_client():
    """Create a mock Elasticsearch client."""
    client = MagicMock()
    client.indices.exists.return_value = True
    return client


@pytest.fixture
def mock_repo(mock_config, mock_es_client):
    """Create repository with mocked ES client."""
    with patch("shared.db.repositories.base.Elasticsearch", return_value=mock_es_client):
        repo = ElasticsearchRepository(mock_config)
        repo._base._client = mock_es_client
        return repo


# =============================================================================
# INDEX GLOSSARY TESTS
# =============================================================================


class TestIndexGlossary:
    """Tests for index_glossary method."""

    def test_indexes_glossary_entry(self, mock_repo, mock_es_client):
        """Test indexing a glossary entry."""
        result = mock_repo.index_glossary(
            glossary_id="scope:repo:main:glossary:user",
            scope="scope",
            repository="repo",
            username="main",
            term="user",
            total_count=5,
            element_ids=["id1", "id2"],
            file_paths=["a.py", "b.py"],
        )

        assert result is True
        mock_es_client.index.assert_called_once()
        call_kwargs = mock_es_client.index.call_args[1]
        doc = call_kwargs["document"]

        assert doc["term"] == "user"
        assert doc["element_type"] == "glossary"
        assert doc["total_count"] == 5
        assert doc["element_ids"] == ["id1", "id2"]
        assert doc["file_paths"] == ["a.py", "b.py"]
        assert doc["name"] == "user"
        assert doc["scope"] == "scope"
        assert doc["repository"] == "repo"
        assert doc["username"] == "main"
        assert doc["level"] == -3
        assert "indexed_at" in doc
        assert "hash_id" in doc

    def test_indexes_with_empty_lists(self, mock_repo, mock_es_client):
        """Test indexing a glossary entry with empty element_ids and file_paths."""
        result = mock_repo.index_glossary(
            glossary_id="scope:repo:main:glossary:empty_term",
            scope="scope",
            repository="repo",
            username="main",
            term="empty_term",
            total_count=0,
            element_ids=[],
            file_paths=[],
        )

        assert result is True
        call_kwargs = mock_es_client.index.call_args[1]
        doc = call_kwargs["document"]
        assert doc["element_ids"] == []
        assert doc["file_paths"] == []
        assert doc["total_count"] == 0

    def test_indexes_glossary_with_description(self, mock_repo, mock_es_client):
        """Test that description field is indexed."""
        mock_repo.index_glossary(
            glossary_id="test:repo:main:glossary:user",
            scope="test",
            repository="repo",
            username="main",
            term="user",
            total_count=5,
            element_ids=["e1", "e2"],
            file_paths=["path/to/file.py"],
            description="A person who uses the system",
        )

        call_args = mock_es_client.index.call_args
        body = call_args[1]["document"]
        assert body["description"] == "A person who uses the system"

    def test_indexes_glossary_with_empty_description_by_default(self, mock_repo, mock_es_client):
        """Test that description defaults to empty string."""
        mock_repo.index_glossary(
            glossary_id="test:repo:main:glossary:user",
            scope="test",
            repository="repo",
            username="main",
            term="user",
            total_count=5,
            element_ids=["e1"],
            file_paths=["path/to/file.py"],
        )

        call_args = mock_es_client.index.call_args
        body = call_args[1]["document"]
        assert body["description"] == ""


# =============================================================================
# GET GLOSSARY TERMS TESTS
# =============================================================================


class TestGetGlossaryTerms:
    """Tests for get_glossary_terms method."""

    def test_returns_all_glossary_terms(self, mock_repo, mock_es_client):
        """Test getting all glossary terms for a repo."""
        mock_es_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "scope:repo:main:glossary:user",
                        "_source": {
                            "term": "user",
                            "total_count": 10,
                            "element_ids": ["id1"],
                            "file_paths": ["user.py"],
                            "feature_associations": [],
                        },
                    },
                    {
                        "_id": "scope:repo:main:glossary:email",
                        "_source": {
                            "term": "email",
                            "total_count": 5,
                            "element_ids": ["id2"],
                            "file_paths": ["email.py"],
                            "feature_associations": [],
                        },
                    },
                ]
            }
        }

        result = mock_repo.get_glossary_terms(
            scope="scope",
            repository="repo",
            username="main",
        )

        assert len(result) == 2
        assert result[0]["term"] == "user"
        assert result[0]["total_count"] == 10
        assert result[1]["term"] == "email"
        assert result[1]["total_count"] == 5

        # Verify search query
        mock_es_client.search.assert_called_once()
        call_kwargs = mock_es_client.search.call_args[1]
        assert call_kwargs["index"] == INDEX_NAME
        assert call_kwargs["size"] == 1000

    def test_returns_empty_list_when_no_terms(self, mock_repo, mock_es_client):
        """Test returns empty list when no glossary terms exist."""
        mock_es_client.search.return_value = {"hits": {"hits": []}}

        result = mock_repo.get_glossary_terms(
            scope="scope",
            repository="repo",
            username="main",
        )

        assert result == []

    def test_filters_by_min_count(self, mock_repo, mock_es_client):
        """Test filtering by minimum count."""
        mock_es_client.search.return_value = {"hits": {"hits": []}}

        mock_repo.get_glossary_terms(
            scope="scope",
            repository="repo",
            username="main",
            min_count=5,
        )

        call_kwargs = mock_es_client.search.call_args[1]
        query = call_kwargs["query"]
        # Check that min_count filter is in the query
        range_filter = next((m for m in query["bool"]["must"] if "range" in m), None)
        assert range_filter is not None
        assert range_filter["range"]["total_count"]["gte"] == 5

    def test_returns_description_field(self, mock_repo, mock_es_client):
        """Test that description field is returned."""
        mock_es_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "scope:repo:main:glossary:user",
                        "_source": {
                            "term": "user",
                            "total_count": 10,
                            "element_ids": ["id1"],
                            "file_paths": ["user.py"],
                            "description": "A person who uses the system",
                            "feature_associations": [],
                        },
                    },
                ]
            }
        }

        result = mock_repo.get_glossary_terms(
            scope="scope",
            repository="repo",
            username="main",
        )

        assert len(result) == 1
        assert result[0]["description"] == "A person who uses the system"

    def test_returns_empty_description_when_missing(self, mock_repo, mock_es_client):
        """Test that description defaults to empty string when not in source."""
        mock_es_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "scope:repo:main:glossary:user",
                        "_source": {
                            "term": "user",
                            "total_count": 10,
                            "element_ids": ["id1"],
                            "file_paths": ["user.py"],
                            "feature_associations": [],
                        },
                    },
                ]
            }
        }

        result = mock_repo.get_glossary_terms(
            scope="scope",
            repository="repo",
            username="main",
        )

        assert len(result) == 1
        assert result[0]["description"] == ""


# =============================================================================
# GET GLOSSARY TERM TESTS
# =============================================================================


class TestGetGlossaryTerm:
    """Tests for get_glossary_term method."""

    def test_returns_specific_term(self, mock_repo, mock_es_client):
        """Test getting a specific glossary term."""
        mock_es_client.get.return_value = {
            "found": True,
            "_source": {
                "term": "user",
                "total_count": 10,
                "element_ids": ["id1", "id2"],
                "file_paths": ["user.py"],
                "feature_associations": [{"feature_id": "feat1"}],
            },
        }

        result = mock_repo.get_glossary_term(
            scope="scope",
            repository="repo",
            term="user",
            username="main",
        )

        assert result is not None
        assert result["term"] == "user"
        assert result["total_count"] == 10
        assert len(result["element_ids"]) == 2
        assert result["feature_associations"] == [{"feature_id": "feat1"}]

    def test_returns_none_when_term_not_found(self, mock_repo, mock_es_client):
        """Test returns None when term doesn't exist."""
        mock_es_client.get.return_value = {"found": False}

        result = mock_repo.get_glossary_term(
            scope="scope",
            repository="repo",
            term="nonexistent",
            username="main",
        )

        assert result is None

    def test_returns_none_on_exception(self, mock_repo, mock_es_client):
        """Test returns None when exception occurs."""
        mock_es_client.get.side_effect = Exception("ES error")

        result = mock_repo.get_glossary_term(
            scope="scope",
            repository="repo",
            term="user",
            username="main",
        )

        assert result is None

    def test_returns_description_field(self, mock_repo, mock_es_client):
        """Test that description field is returned."""
        mock_es_client.get.return_value = {
            "found": True,
            "_source": {
                "term": "user",
                "total_count": 10,
                "element_ids": ["id1"],
                "file_paths": ["user.py"],
                "description": "A person who uses the system",
                "feature_associations": [],
            },
        }

        result = mock_repo.get_glossary_term(
            scope="scope",
            repository="repo",
            term="user",
            username="main",
        )

        assert result is not None
        assert result["description"] == "A person who uses the system"

    def test_returns_empty_description_when_missing(self, mock_repo, mock_es_client):
        """Test that description defaults to empty string when not in source."""
        mock_es_client.get.return_value = {
            "found": True,
            "_source": {
                "term": "user",
                "total_count": 10,
                "element_ids": ["id1"],
                "file_paths": ["user.py"],
                "feature_associations": [],
            },
        }

        result = mock_repo.get_glossary_term(
            scope="scope",
            repository="repo",
            term="user",
            username="main",
        )

        assert result is not None
        assert result["description"] == ""


# =============================================================================
# SEARCH GLOSSARY TESTS
# =============================================================================


class TestSearchGlossary:
    """Tests for search_glossary method."""

    def test_searches_by_partial_match(self, mock_repo, mock_es_client):
        """Test searching glossary terms by partial match."""
        mock_es_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "scope:repo:main:glossary:user_auth",
                        "_source": {
                            "term": "user_auth",
                            "total_count": 10,
                        },
                    },
                    {
                        "_id": "scope:repo:main:glossary:user_profile",
                        "_source": {
                            "term": "user_profile",
                            "total_count": 5,
                        },
                    },
                ]
            }
        }

        result = mock_repo.search_glossary(
            scope="scope",
            repository="repo",
            query="user",
            username="main",
        )

        assert len(result) == 2
        assert result[0]["term"] == "user_auth"
        assert result[1]["term"] == "user_profile"

        # Verify wildcard query is used
        call_kwargs = mock_es_client.search.call_args[1]
        query = call_kwargs["query"]
        wildcard_filter = next((m for m in query["bool"]["must"] if "wildcard" in m), None)
        assert wildcard_filter is not None
        assert "*user*" in wildcard_filter["wildcard"]["term"]

    def test_returns_description_field(self, mock_repo, mock_es_client):
        """Test that description field is returned in search results."""
        mock_es_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "scope:repo:main:glossary:user",
                        "_source": {
                            "term": "user",
                            "total_count": 10,
                            "description": "A person who uses the system",
                        },
                    },
                ]
            }
        }

        result = mock_repo.search_glossary(
            scope="scope",
            repository="repo",
            query="user",
            username="main",
        )

        assert len(result) == 1
        assert result[0]["description"] == "A person who uses the system"

    def test_returns_empty_description_when_missing(self, mock_repo, mock_es_client):
        """Test that description defaults to empty string when not in source."""
        mock_es_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "scope:repo:main:glossary:user",
                        "_source": {
                            "term": "user",
                            "total_count": 10,
                        },
                    },
                ]
            }
        }

        result = mock_repo.search_glossary(
            scope="scope",
            repository="repo",
            query="user",
            username="main",
        )

        assert len(result) == 1
        assert result[0]["description"] == ""


# =============================================================================
# UPDATE GLOSSARY FEATURE ASSOCIATIONS TESTS
# =============================================================================


class TestUpdateGlossaryFeatureAssociations:
    """Tests for update_glossary_feature_associations method."""

    def test_updates_feature_associations(self, mock_repo, mock_es_client):
        """Test updating feature associations on glossary entry."""
        associations = [
            {
                "feature_id": "feat1",
                "feature_label": "user_auth",
                "frequency": 3,
                "total_members": 5,
                "percentage": 60.0,
            }
        ]

        result = mock_repo.update_glossary_feature_associations(
            glossary_id="scope:repo:main:glossary:user",
            feature_associations=associations,
        )

        assert result is True
        mock_es_client.update.assert_called_once()
        call_kwargs = mock_es_client.update.call_args[1]
        assert call_kwargs["id"] == "scope:repo:main:glossary:user"
        assert call_kwargs["doc"]["feature_associations"] == associations
        assert "updated_at" in call_kwargs["doc"]

    def test_updates_with_empty_associations(self, mock_repo, mock_es_client):
        """Test clearing feature associations."""
        result = mock_repo.update_glossary_feature_associations(
            glossary_id="scope:repo:main:glossary:user",
            feature_associations=[],
        )

        assert result is True
        call_kwargs = mock_es_client.update.call_args[1]
        assert call_kwargs["doc"]["feature_associations"] == []

    def test_updates_with_multiple_associations(self, mock_repo, mock_es_client):
        """Test updating with multiple feature associations."""
        associations = [
            {
                "feature_id": "feat1",
                "feature_label": "user_auth",
                "frequency": 3,
                "total_members": 5,
                "percentage": 60.0,
            },
            {
                "feature_id": "feat2",
                "feature_label": "user_profile",
                "frequency": 2,
                "total_members": 10,
                "percentage": 20.0,
            },
        ]

        result = mock_repo.update_glossary_feature_associations(
            glossary_id="scope:repo:main:glossary:user",
            feature_associations=associations,
        )

        assert result is True
        call_kwargs = mock_es_client.update.call_args[1]
        assert len(call_kwargs["doc"]["feature_associations"]) == 2


# =============================================================================
# DELETE GLOSSARY TESTS
# =============================================================================


class TestDeleteGlossary:
    """Tests for delete_glossary method."""

    def test_deletes_all_glossary_entries(self, mock_repo, mock_es_client):
        """Test deleting all glossary entries for a repo."""
        mock_es_client.delete_by_query.return_value = {"deleted": 10}

        result = mock_repo.delete_glossary(
            scope="scope",
            repository="repo",
            username="main",
        )

        assert result == 10
        mock_es_client.delete_by_query.assert_called_once()
        call_kwargs = mock_es_client.delete_by_query.call_args[1]
        query = call_kwargs["query"]
        assert {"term": {"element_type": "glossary"}} in query["bool"]["must"]

    def test_returns_zero_when_nothing_to_delete(self, mock_repo, mock_es_client):
        """Test returns zero when no glossary entries to delete."""
        mock_es_client.delete_by_query.return_value = {"deleted": 0}

        result = mock_repo.delete_glossary(
            scope="scope",
            repository="repo",
            username="main",
        )

        assert result == 0
