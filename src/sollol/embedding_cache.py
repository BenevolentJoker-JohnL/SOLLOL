"""
Embedding Cache for SOLLOL - MD5-based deduplication.
Prevents reprocessing of duplicate content.
"""

import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple


class EmbeddingCache:
    """
    MD5-based embedding cache to prevent duplicate work.

    Features:
    - Content-based hashing (MD5 of text)
    - TTL support (configurable expiration)
    - Cache hit/miss metrics
    - Optional Redis backend for distributed instances
    """

    def __init__(self, ttl_seconds: int = 3600, use_redis: bool = False, redis_url: str = None):
        """
        Initialize embedding cache.

        Args:
            ttl_seconds: Time-to-live for cache entries (default: 1 hour)
            use_redis: Use Redis for distributed caching
            redis_url: Redis connection URL
        """
        self.ttl_seconds = ttl_seconds
        self.use_redis = use_redis
        self.redis_url = redis_url

        # In-memory cache (fallback or standalone)
        self.cache: Dict[str, Dict] = {}

        # Metrics
        self.hits = 0
        self.misses = 0

        # Redis client (if enabled)
        self.redis_client = None
        if use_redis:
            try:
                import redis

                self.redis_client = redis.from_url(
                    redis_url or "redis://localhost:6379/0", decode_responses=True
                )
            except Exception:
                # Fall back to in-memory if Redis unavailable
                self.use_redis = False

    def _hash_text(self, text: str) -> str:
        """Generate MD5 hash of text content."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _is_expired(self, entry: Dict) -> bool:
        """Check if cache entry has expired."""
        if "timestamp" not in entry:
            return True
        age = time.time() - entry["timestamp"]
        return age > self.ttl_seconds

    def get(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for text.

        Args:
            text: Text content to lookup

        Returns:
            Cached embedding or None if not found/expired
        """
        text_hash = self._hash_text(text)

        # Try Redis first (if enabled)
        if self.use_redis and self.redis_client:
            try:
                cached = self.redis_client.get(f"sollol:embed:{text_hash}")
                if cached:
                    entry = json.loads(cached)
                    if not self._is_expired(entry):
                        self.hits += 1
                        return entry["embedding"]
            except Exception:
                pass  # Fall back to in-memory

        # Try in-memory cache
        if text_hash in self.cache:
            entry = self.cache[text_hash]
            if not self._is_expired(entry):
                self.hits += 1
                return entry["embedding"]
            else:
                # Remove expired entry
                del self.cache[text_hash]

        self.misses += 1
        return None

    def set(self, text: str, embedding: List[float]) -> None:
        """
        Store embedding in cache.

        Args:
            text: Text content (used for hashing)
            embedding: Embedding vector to cache
        """
        text_hash = self._hash_text(text)
        entry = {
            "embedding": embedding,
            "timestamp": time.time(),
        }

        # Store in Redis (if enabled)
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.setex(
                    f"sollol:embed:{text_hash}", self.ttl_seconds, json.dumps(entry)
                )
            except Exception:
                pass  # Continue to in-memory storage

        # Store in-memory (always, for fast access)
        self.cache[text_hash] = entry

    def get_batch(self, texts: List[str]) -> Tuple[List[Optional[List[float]]], List[str]]:
        """
        Get cached embeddings for batch of texts.

        Args:
            texts: List of text strings

        Returns:
            Tuple of (cached_embeddings, texts_to_compute)
            - cached_embeddings: List with embeddings or None for cache misses
            - texts_to_compute: List of texts that need to be computed
        """
        cached = []
        to_compute = []

        for text in texts:
            embedding = self.get(text)
            cached.append(embedding)
            if embedding is None:
                to_compute.append(text)

        return cached, to_compute

    def set_batch(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """
        Store batch of embeddings.

        Args:
            texts: List of text strings
            embeddings: List of embedding vectors
        """
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding)

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()

        if self.use_redis and self.redis_client:
            try:
                # Delete all SOLLOL embedding keys
                keys = self.redis_client.keys("sollol:embed:*")
                if keys:
                    self.redis_client.delete(*keys)
            except Exception:
                pass

        # Reset metrics
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache metrics
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_size": len(self.cache),
            "backend": "redis" if self.use_redis else "memory",
            "ttl_seconds": self.ttl_seconds,
        }

    def cleanup_expired(self) -> int:
        """
        Remove expired entries from in-memory cache.

        Returns:
            Number of entries removed
        """
        expired_keys = [key for key, entry in self.cache.items() if self._is_expired(entry)]

        for key in expired_keys:
            del self.cache[key]

        return len(expired_keys)
