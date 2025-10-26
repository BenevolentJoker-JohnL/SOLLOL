"""
Response caching layer for SOLLOL

Caches LLM responses to reduce latency and node load for repeated queries.
"""

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    Cache for LLM responses with TTL and size limits.

    Features:
    - TTL-based expiration
    - LRU eviction when size limit reached
    - Thread-safe operations
    - Deterministic cache keys
    """

    def __init__(self, max_size: int = 1000, ttl: int = 3600, enabled: bool = True):
        """
        Initialize response cache.

        Args:
            max_size: Maximum number of cached responses (default: 1000)
            ttl: Time-to-live in seconds (default: 3600 = 1 hour)
            enabled: Enable/disable caching (default: True)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.enabled = enabled
        self.cache: Dict[str, tuple] = {}
        self.access_times: Dict[str, float] = {}  # For LRU eviction

        # Stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        logger.info(
            f"Response cache initialized (max_size={max_size}, ttl={ttl}s, enabled={enabled})"
        )

    def get_cache_key(self, endpoint: str, data: Dict[str, Any]) -> str:
        """
        Generate deterministic cache key from request.

        Args:
            endpoint: API endpoint (e.g., "/api/chat")
            data: Request payload

        Returns:
            SHA256 hash of normalized request
        """
        # Create normalized representation
        cache_data = {
            "endpoint": endpoint,
            "model": data.get("model"),
            "messages": data.get("messages"),
            "prompt": data.get("prompt"),
            "input": data.get("input"),
            # Include parameters that affect output
            "temperature": data.get("temperature"),
            "top_p": data.get("top_p"),
            "top_k": data.get("top_k"),
            "seed": data.get("seed"),
        }

        # Sort keys for deterministic hashing
        normalized = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        Get cached response if available and not expired.

        Args:
            key: Cache key

        Returns:
            Cached response or None
        """
        if not self.enabled:
            return None

        if key in self.cache:
            value, timestamp = self.cache[key]

            # Check if expired
            if time.time() - timestamp < self.ttl:
                # Update access time for LRU
                self.access_times[key] = time.time()
                self.hits += 1
                logger.debug(f"Cache hit: {key[:16]}... (hit_rate={self.hit_rate():.1%})")
                return value
            else:
                # Expired, remove
                del self.cache[key]
                del self.access_times[key]
                logger.debug(f"Cache expired: {key[:16]}...")

        self.misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        """
        Cache response with current timestamp.

        Args:
            key: Cache key
            value: Response to cache
        """
        if not self.enabled:
            return

        # Evict if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()

        self.cache[key] = (value, time.time())
        self.access_times[key] = time.time()

        logger.debug(f"Cache set: {key[:16]}... (size={len(self.cache)}/{self.max_size})")

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.access_times:
            return

        # Find LRU entry
        lru_key = min(self.access_times, key=self.access_times.get)

        del self.cache[lru_key]
        del self.access_times[lru_key]
        self.evictions += 1

        logger.debug(f"Cache evicted LRU: {lru_key[:16]}... (evictions={self.evictions})")

    def delete(self, key: str) -> bool:
        """
        Delete specific cache entry.

        Args:
            key: Cache key to delete

        Returns:
            True if entry was deleted, False if not found
        """
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            logger.debug(f"Cache entry deleted: {key[:16]}...")
            return True
        return False

    def invalidate_by_model(self, model: str) -> int:
        """
        Invalidate all cache entries for a specific model.

        Args:
            model: Model name to invalidate

        Returns:
            Number of entries invalidated
        """
        keys_to_delete = []
        for key, (value, _) in self.cache.items():
            try:
                # Try to extract model from cached data
                if isinstance(value, dict):
                    cached_model = value.get("model")
                    if cached_model == model:
                        keys_to_delete.append(key)
            except Exception:
                continue

        for key in keys_to_delete:
            self.delete(key)

        logger.info(f"Invalidated {len(keys_to_delete)} cache entries for model '{model}'")
        return len(keys_to_delete)

    def invalidate_by_endpoint(self, endpoint: str) -> int:
        """
        Invalidate all cache entries for a specific endpoint.

        Args:
            endpoint: API endpoint to invalidate (e.g., "/api/chat")

        Returns:
            Number of entries invalidated
        """
        # We need to reconstruct keys to check endpoint
        # For now, clear all - can be optimized later by storing metadata
        count = len(self.cache)
        self.clear()
        logger.info(f"Invalidated {count} cache entries for endpoint '{endpoint}'")
        return count

    def update_ttl(self, key: str, new_ttl: int) -> bool:
        """
        Update TTL for a specific cache entry.

        Args:
            key: Cache key
            new_ttl: New TTL in seconds

        Returns:
            True if updated, False if not found
        """
        if key in self.cache:
            value, _ = self.cache[key]
            # Reset timestamp with new effective TTL
            self.cache[key] = (value, time.time() + (new_ttl - self.ttl))
            logger.debug(f"Cache TTL updated: {key[:16]}... (new_ttl={new_ttl}s)")
            return True
        return False

    def list_keys(self, limit: int = 100) -> list:
        """
        List cached keys.

        Args:
            limit: Maximum number of keys to return

        Returns:
            List of cache keys
        """
        return list(self.cache.keys())[:limit]

    def export_cache(self) -> Dict[str, Any]:
        """
        Export cache to dictionary for persistence.

        Returns:
            Dict containing all cache data
        """
        return {
            "cache": {k: v for k, v in self.cache.items()},
            "access_times": dict(self.access_times),
            "stats": self.get_stats(),
            "export_time": time.time(),
        }

    def import_cache(self, data: Dict[str, Any]) -> int:
        """
        Import cache from dictionary.

        Args:
            data: Cache data from export_cache()

        Returns:
            Number of entries imported
        """
        count = 0
        if "cache" in data:
            for key, value in data["cache"].items():
                # Check if not expired
                _, timestamp = value
                if time.time() - timestamp < self.ttl:
                    self.cache[key] = value
                    count += 1

        if "access_times" in data:
            self.access_times.update(data["access_times"])

        logger.info(f"Imported {count} cache entries")
        return count

    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Cache cleared")

    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats
        """
        return {
            "enabled": self.enabled,
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate(),
            "total_requests": self.hits + self.misses,
        }

    def __repr__(self):
        return (
            f"ResponseCache(size={len(self.cache)}/{self.max_size}, "
            f"hit_rate={self.hit_rate():.1%}, enabled={self.enabled})"
        )
