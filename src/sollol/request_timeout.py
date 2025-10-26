"""
Request Timeout Management for SOLLOL.
Prevents hung requests and limits resource waste.
"""

import asyncio
import logging
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RequestTimeoutError(Exception):
    """Exception raised when request times out."""

    def __init__(self, timeout_seconds: float, operation: str = "request"):
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        super().__init__(f"{operation} timed out after {timeout_seconds}s")


async def with_timeout(
    coro: Callable[..., Any],
    timeout_seconds: float,
    operation: str = "request",
    *args,
    **kwargs,
) -> Any:
    """
    Execute async function with timeout.

    Args:
        coro: Async function to execute
        timeout_seconds: Timeout in seconds
        operation: Description for error message
        *args, **kwargs: Arguments for coro

    Returns:
        Result from coro

    Raises:
        RequestTimeoutError: If timeout exceeded
    """
    try:
        result = await asyncio.wait_for(coro(*args, **kwargs), timeout=timeout_seconds)
        return result
    except asyncio.TimeoutError:
        logger.warning(f"⏱️  {operation} timed out after {timeout_seconds}s")
        raise RequestTimeoutError(timeout_seconds, operation)


class TimeoutConfig:
    """Configuration for request timeouts."""

    def __init__(
        self,
        chat_timeout: float = 300.0,  # 5 minutes for chat (CPU-friendly)
        generate_timeout: float = 300.0,  # 5 minutes for generation (CPU-friendly)
        embed_timeout: float = 60.0,  # 1 minute for embeddings
        model_load_timeout: float = 600.0,  # 10 minutes for model loading
    ):
        """
        Initialize timeout configuration.

        Args:
            chat_timeout: Timeout for chat completion requests
            generate_timeout: Timeout for text generation requests
            embed_timeout: Timeout for embedding requests
            model_load_timeout: Timeout for model loading operations
        """
        self.chat_timeout = chat_timeout
        self.generate_timeout = generate_timeout
        self.embed_timeout = embed_timeout
        self.model_load_timeout = model_load_timeout

    def get_timeout(self, operation_type: str) -> float:
        """
        Get timeout for operation type.

        Args:
            operation_type: Type of operation (chat, generate, embed, model_load)

        Returns:
            Timeout in seconds
        """
        timeout_map = {
            "chat": self.chat_timeout,
            "generate": self.generate_timeout,
            "embed": self.embed_timeout,
            "model_load": self.model_load_timeout,
        }
        return timeout_map.get(operation_type, self.chat_timeout)


class TimeoutManager:
    """
    Manager for request timeouts across SOLLOL gateway.

    Features:
    - Per-operation-type timeouts
    - Configurable via environment variables
    - Timeout statistics tracking
    """

    def __init__(self, config: Optional[TimeoutConfig] = None):
        """
        Initialize timeout manager.

        Args:
            config: TimeoutConfig instance (uses defaults if None)
        """
        self.config = config or TimeoutConfig()

        # Statistics
        self.total_requests = 0
        self.timed_out_requests = 0
        self.timeouts_by_operation = {}

    async def execute_with_timeout(
        self,
        coro: Callable[..., Any],
        operation_type: str = "chat",
        timeout_override: Optional[float] = None,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute async operation with timeout tracking.

        Args:
            coro: Async function to execute
            operation_type: Type of operation (chat, generate, embed)
            timeout_override: Override default timeout
            *args, **kwargs: Arguments for coro

        Returns:
            Result from coro

        Raises:
            RequestTimeoutError: If timeout exceeded
        """
        self.total_requests += 1

        timeout = timeout_override or self.config.get_timeout(operation_type)

        try:
            result = await with_timeout(coro, timeout, operation_type, *args, **kwargs)
            return result
        except RequestTimeoutError as e:
            # Track timeout
            self.timed_out_requests += 1
            self.timeouts_by_operation[operation_type] = (
                self.timeouts_by_operation.get(operation_type, 0) + 1
            )
            raise e

    def get_stats(self) -> dict:
        """
        Get timeout statistics.

        Returns:
            Statistics dictionary
        """
        timeout_rate = (
            (self.timed_out_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        )

        return {
            "enabled": True,
            "total_requests": self.total_requests,
            "timed_out_requests": self.timed_out_requests,
            "timeout_rate_percent": timeout_rate,
            "timeouts_by_operation": self.timeouts_by_operation,
            "timeouts": {
                "chat_timeout_seconds": self.config.chat_timeout,
                "generate_timeout_seconds": self.config.generate_timeout,
                "embed_timeout_seconds": self.config.embed_timeout,
                "model_load_timeout_seconds": self.config.model_load_timeout,
            },
        }
