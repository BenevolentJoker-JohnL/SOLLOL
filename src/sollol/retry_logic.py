"""
Retry Logic with Exponential Backoff for SOLLOL.
Handles transient failures gracefully with configurable retry strategies.
"""

import asyncio
import random
import time
from typing import Callable, Optional, Type


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """
        Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff (2.0 = doubles each time)
            jitter: Add random jitter to prevent thundering herd
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for given retry attempt.

        Args:
            attempt: Current retry attempt (0-indexed)

        Returns:
            Delay in seconds
        """
        # Exponential backoff: base_delay * (exponential_base ^ attempt)
        delay = self.base_delay * (self.exponential_base**attempt)

        # Cap at max_delay
        delay = min(delay, self.max_delay)

        # Add jitter (±25% randomness)
        if self.jitter:
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)  # Ensure non-negative


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        config: RetryConfig instance (uses defaults if None)
        exceptions: Tuple of exceptions to retry on

    Example:
        @retry_with_backoff(RetryConfig(max_retries=5))
        def fetch_data():
            return requests.get(url)
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    # Don't retry on last attempt
                    if attempt == config.max_retries:
                        break

                    delay = config.get_delay(attempt)
                    print(
                        f"⚠️  Retry attempt {attempt + 1}/{config.max_retries} "
                        f"after {delay:.2f}s: {str(e)[:100]}"
                    )
                    time.sleep(delay)

            # All retries exhausted
            raise last_exception

        return wrapper

    return decorator


def async_retry_with_backoff(
    config: Optional[RetryConfig] = None,
    exceptions: tuple = (Exception,),
):
    """
    Async decorator for retrying functions with exponential backoff.

    Args:
        config: RetryConfig instance (uses defaults if None)
        exceptions: Tuple of exceptions to retry on

    Example:
        @async_retry_with_backoff(RetryConfig(max_retries=5))
        async def fetch_data():
            return await client.get(url)
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    # Don't retry on last attempt
                    if attempt == config.max_retries:
                        break

                    delay = config.get_delay(attempt)
                    print(
                        f"⚠️  Retry attempt {attempt + 1}/{config.max_retries} "
                        f"after {delay:.2f}s: {str(e)[:100]}"
                    )
                    await asyncio.sleep(delay)

            # All retries exhausted
            raise last_exception

        return wrapper

    return decorator


class RetryableRequest:
    """Class-based retry logic for more control."""

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.attempt_count = 0
        self.last_exception: Optional[Exception] = None

    async def execute_async(
        self,
        func: Callable,
        *args,
        exceptions: tuple = (Exception,),
        **kwargs,
    ):
        """
        Execute async function with retry logic.

        Args:
            func: Async function to execute
            *args, **kwargs: Arguments for function
            exceptions: Exceptions to retry on

        Returns:
            Function result if successful

        Raises:
            Last exception if all retries exhausted
        """
        self.attempt_count = 0
        self.last_exception = None

        for attempt in range(self.config.max_retries + 1):
            self.attempt_count = attempt + 1

            try:
                result = await func(*args, **kwargs)
                return result
            except exceptions as e:
                self.last_exception = e

                # Don't retry on last attempt
                if attempt == self.config.max_retries:
                    break

                delay = self.config.get_delay(attempt)
                await asyncio.sleep(delay)

        # All retries exhausted
        raise self.last_exception

    def execute(
        self,
        func: Callable,
        *args,
        exceptions: tuple = (Exception,),
        **kwargs,
    ):
        """
        Execute sync function with retry logic.

        Args:
            func: Function to execute
            *args, **kwargs: Arguments for function
            exceptions: Exceptions to retry on

        Returns:
            Function result if successful

        Raises:
            Last exception if all retries exhausted
        """
        self.attempt_count = 0
        self.last_exception = None

        for attempt in range(self.config.max_retries + 1):
            self.attempt_count = attempt + 1

            try:
                result = func(*args, **kwargs)
                return result
            except exceptions as e:
                self.last_exception = e

                # Don't retry on last attempt
                if attempt == self.config.max_retries:
                    break

                delay = self.config.get_delay(attempt)
                time.sleep(delay)

        # All retries exhausted
        raise self.last_exception

    def get_stats(self) -> dict:
        """Get retry statistics."""
        return {
            "attempts": self.attempt_count,
            "last_exception": str(self.last_exception) if self.last_exception else None,
            "max_retries": self.config.max_retries,
        }
