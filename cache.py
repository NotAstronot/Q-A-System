"""Redis caching layer for RAG queries - Week 1 Optimization"""

import os
import json
import hashlib
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Check if Redis is enabled
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "false").lower() == "true"

if REDIS_ENABLED:
    try:
        import redis

        REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
        REDIS_DB = int(os.getenv("REDIS_DB", "0"))
        REDIS_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "3600"))

        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )

        # Test connection
        redis_client.ping()
        logger.info(f"✅ Redis connected: {REDIS_HOST}:{REDIS_PORT}")

    except Exception as e:
        logger.warning(f"⚠️ Redis connection failed: {e}. Caching disabled.")
        REDIS_ENABLED = False
        redis_client = None
else:
    redis_client = None
    logger.info("Redis caching disabled in config")


def cache_key(question: str, context_hash: str) -> str:
    """Generate cache key from question + context hash"""
    # Normalize question (lowercase, strip whitespace)
    q_normalized = question.lower().strip()
    q_hash = hashlib.sha256(q_normalized.encode()).hexdigest()[:16]
    return f"qa:v1:{context_hash}:{q_hash}"


def get_context_hash(collection_count: int, collection_name: str) -> str:
    """
    Generate hash representing current document collection state.

    When documents change (add/delete), hash changes -> cache invalidates.
    """
    state = f"{collection_name}:{collection_count}"
    return hashlib.sha256(state.encode()).hexdigest()[:16]


def get_cached_answer(question: str, context_hash: str) -> Optional[Dict[str, Any]]:
    """Get cached answer if exists"""
    if not REDIS_ENABLED or not redis_client:
        return None

    try:
        key = cache_key(question, context_hash)
        cached = redis_client.get(key)

        if cached:
            logger.info(f"✅ Cache HIT for question: {question[:50]}...")
            return json.loads(cached)
        else:
            logger.debug(f"Cache MISS for question: {question[:50]}...")
            return None

    except Exception as e:
        logger.warning(f"Cache read error: {e}")
        return None


def cache_answer(
    question: str,
    context_hash: str,
    result: Dict[str, Any],
    ttl: Optional[int] = None,
) -> None:
    """Cache answer for TTL seconds"""
    if not REDIS_ENABLED or not redis_client:
        return

    try:
        key = cache_key(question, context_hash)
        ttl = ttl or REDIS_CACHE_TTL

        # Serialize result (exclude heavy objects if any)
        cache_data = {
            "answer": result.get("answer"),
            "rewritten_query": result.get("rewritten_query"),
            "validation": result.get("validation"),
            "attempts": result.get("attempts"),
            # Store source metadata only, not full Document objects
            "sources_meta": [
                {
                    "filename": doc.metadata.get("filename"),
                    "page": doc.metadata.get("page"),
                    "content_preview": doc.page_content[:200],
                }
                for doc in result.get("sources", [])
            ],
        }

        redis_client.setex(key, ttl, json.dumps(cache_data))
        logger.debug(f"Cached answer for {ttl}s: {question[:50]}...")

    except Exception as e:
        logger.warning(f"Cache write error: {e}")


def clear_cache() -> None:
    """Clear all QA cache entries"""
    if not REDIS_ENABLED or not redis_client:
        return

    try:
        # Delete all keys matching qa:v1:*
        cursor = 0
        count = 0

        while True:
            cursor, keys = redis_client.scan(cursor, match="qa:v1:*", count=100)
            if keys:
                redis_client.delete(*keys)
                count += len(keys)
            if cursor == 0:
                break

        logger.info(f"Cleared {count} cache entries")

    except Exception as e:
        logger.error(f"Cache clear error: {e}")


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    if not REDIS_ENABLED or not redis_client:
        return {"enabled": False}

    try:
        info = redis_client.info("stats")

        # Count QA cache keys
        cursor = 0
        qa_keys = 0
        while True:
            cursor, keys = redis_client.scan(cursor, match="qa:v1:*", count=100)
            qa_keys += len(keys)
            if cursor == 0:
                break

        return {
            "enabled": True,
            "connected": True,
            "qa_cache_keys": qa_keys,
            "total_keys": redis_client.dbsize(),
            "hits": info.get("keyspace_hits", 0),
            "misses": info.get("keyspace_misses", 0),
            "hit_rate": (
                info.get("keyspace_hits", 0)
                / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1)
                * 100
            ),
        }

    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        return {"enabled": True, "connected": False, "error": str(e)}
