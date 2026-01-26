"""FastAPI dependencies for the Web API."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Generator

from shared.config import MagaldiConfig, get_config, load_config
from shared.db.elasticsearch import ElasticsearchRepository

if TYPE_CHECKING:
    pass


@lru_cache
def get_cached_config() -> MagaldiConfig:
    """Get cached configuration."""
    try:
        return get_config()
    except RuntimeError:
        return load_config()


def get_es_repository() -> Generator[ElasticsearchRepository, None, None]:
    """Get Elasticsearch repository as a dependency."""
    config = get_cached_config()
    es_repo = ElasticsearchRepository(config)
    try:
        yield es_repo
    finally:
        es_repo.close()


async def check_elasticsearch_health(es_repo: ElasticsearchRepository) -> dict:
    """Check Elasticsearch health."""
    try:
        client = es_repo._get_client()
        health = client.cluster.health()
        return {
            "status": "healthy",
            "cluster_status": health.get("status", "unknown"),
            "number_of_nodes": health.get("number_of_nodes", 0),
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def check_llm_health(config: MagaldiConfig) -> dict:
    """Check LLM providers health for all configured models."""
    import requests

    results = {}

    for name, model in config.llm.models.items():
        try:
            if model.provider == "ollama":
                response = requests.get(f"{model.url}/api/tags", timeout=5)
                if response.ok:
                    results[name] = {"status": "healthy", "provider": "ollama"}
                else:
                    results[name] = {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
            elif model.provider == "llamacpp":
                response = requests.get(f"{model.url.rstrip('/')}/v1/models", timeout=5)
                results[name] = {"status": "healthy", "provider": "llamacpp"}
            else:
                results[name] = {"status": "unknown", "provider": model.provider}
        except Exception as e:
            results[name] = {"status": "unhealthy", "error": str(e)}

    # Overall health: healthy if any model is healthy
    any_healthy = any(r.get("status") == "healthy" for r in results.values())
    return {
        "status": "healthy" if any_healthy else "unhealthy",
        "models": results,
    }


async def check_redis_health(config: MagaldiConfig) -> dict:
    """Check Redis health."""
    import redis

    try:
        r = redis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            db=config.redis.db,
            socket_timeout=5,
        )
        r.ping()
        r.close()
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def get_redis_queue_stats(config: MagaldiConfig) -> dict:
    """Get Redis queue statistics by scanning job keys.

    Keys follow pattern: magaldi:{job_type}:{key_type}:{scope}:{repository}:{username}
    """
    import json

    import redis

    result = {
        "summarization": {},
        "embedding": {},
        "labeling": {},
        "feature": {},
        "subfeature": {},
        "subfeature_labeling": {},
        "total_pending": 0,
        "total_running": 0,
    }

    try:
        r = redis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            db=config.redis.db,
            decode_responses=True,
            socket_timeout=5,
        )

        # Scan for summarization job keys
        # Pattern: magaldi:summarization:jobs:{scope}:{repository}:{username}
        for jobs_key in r.scan_iter("magaldi:summarization:jobs:*"):
            parts = jobs_key.split(":")
            if len(parts) >= 6:
                scope, repo, username = parts[3], parts[4], parts[5]
                queue_id = f"{scope}/{repo}/{username}"

                # Count pending jobs in the hash
                pending_count = 0
                for job_data in r.hvals(jobs_key):
                    try:
                        job = json.loads(job_data)
                        if job.get("status") == "pending":
                            pending_count += 1
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Count running jobs
                running_key = f"magaldi:summarization:running:{scope}:{repo}:{username}"
                running_count = r.scard(running_key)

                if pending_count > 0 or running_count > 0:
                    result["summarization"][queue_id] = {
                        "pending": pending_count,
                        "running": running_count,
                    }
                    result["total_pending"] += pending_count
                    result["total_running"] += running_count

        # Scan for embedding job keys
        # Pattern: magaldi:embedding:jobs:{scope}:{repository}:{username}
        for jobs_key in r.scan_iter("magaldi:embedding:jobs:*"):
            parts = jobs_key.split(":")
            if len(parts) >= 6:
                scope, repo, username = parts[3], parts[4], parts[5]
                queue_id = f"{scope}/{repo}/{username}"

                # Count pending jobs in the hash
                pending_count = 0
                for job_data in r.hvals(jobs_key):
                    try:
                        job = json.loads(job_data)
                        if job.get("status") == "pending":
                            pending_count += 1
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Count running jobs
                running_key = f"magaldi:embedding:running:{scope}:{repo}:{username}"
                running_count = r.scard(running_key)

                if pending_count > 0 or running_count > 0:
                    result["embedding"][queue_id] = {
                        "pending": pending_count,
                        "running": running_count,
                    }
                    result["total_pending"] += pending_count
                    result["total_running"] += running_count

        # Scan for labeling job keys
        # Pattern: magaldi:labeling:jobs:{scope}:{repository}:{username}
        for jobs_key in r.scan_iter("magaldi:labeling:jobs:*"):
            parts = jobs_key.split(":")
            if len(parts) >= 6:
                scope, repo, username = parts[3], parts[4], parts[5]
                queue_id = f"{scope}/{repo}/{username}"

                # Count pending jobs in the hash
                pending_count = 0
                for job_data in r.hvals(jobs_key):
                    try:
                        job = json.loads(job_data)
                        if job.get("status") == "pending":
                            pending_count += 1
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Count running jobs
                running_key = f"magaldi:labeling:running:{scope}:{repo}:{username}"
                running_count = r.scard(running_key)

                if pending_count > 0 or running_count > 0:
                    result["labeling"][queue_id] = {
                        "pending": pending_count,
                        "running": running_count,
                    }
                    result["total_pending"] += pending_count
                    result["total_running"] += running_count

        # Scan for feature job keys
        # Pattern: magaldi:feature:jobs:{scope}:{repository}:{username}
        for jobs_key in r.scan_iter("magaldi:feature:jobs:*"):
            parts = jobs_key.split(":")
            if len(parts) >= 6:
                scope, repo, username = parts[3], parts[4], parts[5]
                queue_id = f"{scope}/{repo}/{username}"

                # Count pending jobs in the hash
                pending_count = 0
                for job_data in r.hvals(jobs_key):
                    try:
                        job = json.loads(job_data)
                        if job.get("status") == "pending":
                            pending_count += 1
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Count running jobs
                running_key = f"magaldi:feature:running:{scope}:{repo}:{username}"
                running_count = r.scard(running_key)

                if pending_count > 0 or running_count > 0:
                    result["feature"][queue_id] = {
                        "pending": pending_count,
                        "running": running_count,
                    }
                    result["total_pending"] += pending_count
                    result["total_running"] += running_count

        # Scan for subfeature job keys
        # Pattern: magaldi:subfeature:jobs:{scope}:{repository}:{username}
        for jobs_key in r.scan_iter("magaldi:subfeature:jobs:*"):
            parts = jobs_key.split(":")
            if len(parts) >= 6:
                scope, repo, username = parts[3], parts[4], parts[5]
                queue_id = f"{scope}/{repo}/{username}"

                # Count pending jobs in the hash
                pending_count = 0
                for job_data in r.hvals(jobs_key):
                    try:
                        job = json.loads(job_data)
                        if job.get("status") == "pending":
                            pending_count += 1
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Count running jobs
                running_key = f"magaldi:subfeature:running:{scope}:{repo}:{username}"
                running_count = r.scard(running_key)

                if pending_count > 0 or running_count > 0:
                    result["subfeature"][queue_id] = {
                        "pending": pending_count,
                        "running": running_count,
                    }
                    result["total_pending"] += pending_count
                    result["total_running"] += running_count

        # Scan for subfeature labeling job keys
        # Pattern: magaldi:subfeature_labeling:jobs:{scope}:{repository}:{username}
        for jobs_key in r.scan_iter("magaldi:subfeature_labeling:jobs:*"):
            parts = jobs_key.split(":")
            if len(parts) >= 6:
                scope, repo, username = parts[3], parts[4], parts[5]
                queue_id = f"{scope}/{repo}/{username}"

                # Count pending jobs in the hash
                pending_count = 0
                for job_data in r.hvals(jobs_key):
                    try:
                        job = json.loads(job_data)
                        if job.get("status") == "pending":
                            pending_count += 1
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Count running jobs
                running_key = f"magaldi:subfeature_labeling:running:{scope}:{repo}:{username}"
                running_count = r.scard(running_key)

                if pending_count > 0 or running_count > 0:
                    result["subfeature_labeling"][queue_id] = {
                        "pending": pending_count,
                        "running": running_count,
                    }
                    result["total_pending"] += pending_count
                    result["total_running"] += running_count

        r.close()
    except Exception:
        pass  # Return empty stats on error

    return result
