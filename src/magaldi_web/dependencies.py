"""FastAPI dependencies for the Web API."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Generator

from shared.cli._shared import validate_llm_url
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
            validated_url = validate_llm_url(model.url)
            if model.provider == "ollama":
                response = requests.get(f"{validated_url}/api/tags", timeout=5)
                if response.ok:
                    results[name] = {"status": "healthy", "provider": "ollama"}
                else:
                    results[name] = {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
            elif model.provider == "llamacpp":
                response = requests.get(f"{validated_url}/v1/models", timeout=5)
                results[name] = {"status": "healthy", "provider": "llamacpp"}
            else:
                results[name] = {"status": "unknown", "provider": model.provider}
        except ValueError as e:
            results[name] = {"status": "unhealthy", "error": f"Invalid URL: {e}"}
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


async def get_job_queue_totals(config: MagaldiConfig) -> dict:
    """Get aggregated job queue totals for admin dashboard.

    Returns totals for summarization and embedding job queues with
    pending, running, completed, and failed counts.
    """
    import json

    import redis

    result = {
        "summarization": {"pending": 0, "running": 0, "completed": 0, "failed": 0},
        "embedding": {"pending": 0, "running": 0, "completed": 0, "failed": 0},
    }

    try:
        r = redis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            db=config.redis.db,
            decode_responses=True,
            socket_timeout=5,
        )

        # Count summarization jobs by status
        for jobs_key in r.scan_iter("magaldi:summarization:jobs:*"):
            for job_data in r.hvals(jobs_key):
                try:
                    job = json.loads(job_data)
                    status = job.get("status", "pending")
                    if status in result["summarization"]:
                        result["summarization"][status] += 1
                except (json.JSONDecodeError, TypeError):
                    pass

        # Count summarization running jobs from the running set
        for running_key in r.scan_iter("magaldi:summarization:running:*"):
            result["summarization"]["running"] += r.scard(running_key)

        # Count embedding jobs by status
        for jobs_key in r.scan_iter("magaldi:embedding:jobs:*"):
            for job_data in r.hvals(jobs_key):
                try:
                    job = json.loads(job_data)
                    status = job.get("status", "pending")
                    if status in result["embedding"]:
                        result["embedding"][status] += 1
                except (json.JSONDecodeError, TypeError):
                    pass

        # Count embedding running jobs from the running set
        for running_key in r.scan_iter("magaldi:embedding:running:*"):
            result["embedding"]["running"] += r.scard(running_key)

        r.close()
    except Exception:
        pass  # Return zeros on error

    return result


async def retry_failed_jobs_in_redis(config: MagaldiConfig, job_type: str) -> int:
    """Reset failed jobs back to pending status.

    Args:
        config: Magaldi configuration.
        job_type: Either 'summarization' or 'embedding'.

    Returns:
        Number of jobs reset.
    """
    import json

    import redis

    if job_type not in ("summarization", "embedding"):
        return 0

    reset_count = 0

    try:
        r = redis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            db=config.redis.db,
            decode_responses=True,
            socket_timeout=5,
        )

        pattern = f"magaldi:{job_type}:jobs:*"
        for jobs_key in r.scan_iter(pattern):
            # Get all jobs in this hash
            all_jobs = r.hgetall(jobs_key)
            for element_id, job_data_str in all_jobs.items():
                try:
                    job = json.loads(job_data_str)
                    if job.get("status") == "failed":
                        # Reset to pending
                        job["status"] = "pending"
                        job["error_message"] = None
                        # Keep retry_count for debugging
                        r.hset(jobs_key, element_id, json.dumps(job))
                        reset_count += 1
                except (json.JSONDecodeError, TypeError):
                    pass

        r.close()
    except Exception:
        pass  # Return current count on error

    return reset_count


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
