"""
Rate Limiting for SOLLOL.
Prevents cluster overload with token bucket and sliding window algorithms.
"""

import time
from collections import deque
from typing import Dict, Optional


class TokenBucket:
    """
    Token bucket rate limiter.

    Tokens are added at a fixed rate. Each request consumes a token.
    If no tokens available, request is rejected or queued.
    """

    def __init__(self, rate: float, capacity: int):
        """
        Initialize token bucket.

        Args:
            rate: Tokens added per second
            capacity: Maximum tokens in bucket
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill

        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens consumed, False if not enough tokens
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def get_state(self) -> dict:
        """Get current bucket state."""
        self._refill()
        return {
            "tokens": self.tokens,
            "capacity": self.capacity,
            "rate": self.rate,
            "utilization": 1.0 - (self.tokens / self.capacity),
        }


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter.

    Tracks requests in a time window and limits based on count.
    More accurate than fixed window, prevents bursts at window boundaries.
    """

    def __init__(self, max_requests: int, window_seconds: int):
        """
        Initialize sliding window rate limiter.

        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()  # (timestamp, request_id)

    def _clean_old_requests(self):
        """Remove requests outside current window."""
        now = time.time()
        cutoff = now - self.window_seconds

        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

    def allow_request(self) -> bool:
        """
        Check if request is allowed.

        Returns:
            True if request allowed, False if rate limit exceeded
        """
        self._clean_old_requests()

        if len(self.requests) < self.max_requests:
            self.requests.append(time.time())
            return True
        return False

    def get_state(self) -> dict:
        """Get current limiter state."""
        self._clean_old_requests()
        return {
            "requests_in_window": len(self.requests),
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "utilization": len(self.requests) / self.max_requests,
        }


class RateLimiter:
    """
    Combined rate limiter for SOLLOL.

    Supports both per-node and global rate limiting with multiple algorithms.
    """

    def __init__(
        self,
        global_rate: Optional[float] = None,
        global_capacity: Optional[int] = None,
        per_node_rate: Optional[float] = None,
        per_node_capacity: Optional[int] = None,
    ):
        """
        Initialize rate limiter.

        Args:
            global_rate: Global tokens per second (None = no global limit)
            global_capacity: Global token bucket capacity
            per_node_rate: Per-node tokens per second (None = no per-node limit)
            per_node_capacity: Per-node token bucket capacity
        """
        # Global rate limiter
        self.global_limiter = None
        if global_rate and global_capacity:
            self.global_limiter = TokenBucket(global_rate, global_capacity)

        # Per-node rate limiters
        self.node_limiters: Dict[str, TokenBucket] = {}
        self.per_node_rate = per_node_rate
        self.per_node_capacity = per_node_capacity

    def allow_request(self, node: Optional[str] = None) -> tuple[bool, str]:
        """
        Check if request is allowed.

        Args:
            node: Node identifier (for per-node limiting)

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        # Check global rate limit
        if self.global_limiter:
            if not self.global_limiter.consume():
                return False, "global_rate_limit_exceeded"

        # Check per-node rate limit
        if node and self.per_node_rate and self.per_node_capacity:
            if node not in self.node_limiters:
                self.node_limiters[node] = TokenBucket(self.per_node_rate, self.per_node_capacity)

            if not self.node_limiters[node].consume():
                return False, f"node_rate_limit_exceeded:{node}"

        return True, "allowed"

    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        stats = {
            "global": None,
            "nodes": {},
        }

        if self.global_limiter:
            stats["global"] = self.global_limiter.get_state()

        for node, limiter in self.node_limiters.items():
            stats["nodes"][node] = limiter.get_state()

        return stats


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: float = None):
        """
        Initialize exception.

        Args:
            message: Error message
            retry_after: Seconds to wait before retrying
        """
        super().__init__(message)
        self.retry_after = retry_after
