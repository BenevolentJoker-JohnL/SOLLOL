"""
Synchronous API wrappers for SOLLOL's async components.

Enables synchronous applications to use SOLLOL without dealing with async/await.
The wrappers run an event loop in a background thread and provide blocking interfaces.

Example:
    from sollol.sync_wrapper import HybridRouter

    # Use synchronously (no async/await needed)
    router = HybridRouter(
        ollama_pool=...,
        enable_distributed=True
    )

    response = router.route_request(
        model="llama3.2",
        messages=[{"role": "user", "content": "Hello!"}]
    )
"""

import asyncio
import logging
import threading
from functools import wraps
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AsyncEventLoop:
    """Manages a background event loop for running async code synchronously."""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()
        self._start_loop()

    def _run_loop(self):
        """Run the event loop in the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def _start_loop(self):
        """Start the background event loop thread."""
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="sollol-async-loop"
        )
        self._thread.start()
        self._started.wait()  # Wait for loop to be ready
        logger.debug("Background event loop started")

    def run_coroutine(self, coro, timeout: Optional[float] = None):
        """
        Run a coroutine synchronously.

        Args:
            coro: Coroutine to run
            timeout: Optional timeout in seconds

        Returns:
            Result from the coroutine

        Raises:
            TimeoutError: If timeout is exceeded
            Exception: Any exception raised by the coroutine
        """
        if not self._loop or not self._loop.is_running():
            raise RuntimeError("Event loop not running")

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def close(self):
        """Stop the event loop and clean up."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.debug("Background event loop stopped")


# Global event loop instance
_event_loop: Optional[AsyncEventLoop] = None


def get_event_loop() -> AsyncEventLoop:
    """Get or create the global event loop."""
    global _event_loop
    if _event_loop is None:
        _event_loop = AsyncEventLoop()
    return _event_loop


class HybridRouter:
    """
    Synchronous wrapper for SOLLOL's async HybridRouter.

    This wrapper allows synchronous code to use HybridRouter without async/await.
    It runs an event loop in a background thread and provides blocking methods.

    Example:
        router = HybridRouter(
            ollama_pool=OllamaPool.auto_configure(),
            enable_distributed=True,
            num_rpc_backends=3
        )

        # Synchronous call (blocks until complete)
        response = router.route_request(
            model="llama3.2",
            messages=[{"role": "user", "content": "Hello!"}]
        )
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize synchronous HybridRouter.

        Args:
            *args, **kwargs: Same arguments as async HybridRouter
        """
        from sollol.hybrid_router import HybridRouter as AsyncHybridRouter

        self._loop_manager = get_event_loop()
        self._async_router = self._loop_manager.run_coroutine(
            self._create_async_router(*args, **kwargs)
        )
        logger.info("Synchronous HybridRouter initialized")

    async def _create_async_router(self, *args, **kwargs):
        """Create the async router instance."""
        from sollol.hybrid_router import HybridRouter as AsyncHybridRouter

        return AsyncHybridRouter(*args, **kwargs)

    def route_request(
        self, model: str, messages: List[Dict[str, str]], timeout: Optional[float] = 300, **kwargs
    ) -> Dict[str, Any]:
        """
        Route a request synchronously.

        Args:
            model: Model name
            messages: List of message dicts with 'role' and 'content'
            timeout: Request timeout in seconds (default: 300)
            **kwargs: Additional arguments passed to the router

        Returns:
            Response dict from the model

        Raises:
            TimeoutError: If request exceeds timeout
            Exception: Any error from routing or inference
        """
        coro = self._async_router.route_request(model=model, messages=messages, **kwargs)
        return self._loop_manager.run_coroutine(coro, timeout=timeout)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get router statistics synchronously.

        Returns:
            Statistics dict
        """
        coro = self._async_router.get_stats()
        return self._loop_manager.run_coroutine(coro, timeout=10.0)

    def should_use_distributed(self, model: str) -> bool:
        """
        Check if model should use distributed inference.

        Args:
            model: Model name

        Returns:
            True if distributed inference should be used
        """
        return self._async_router.should_use_distributed(model)

    def close(self):
        """Clean up resources."""
        # Note: Don't close the global event loop as other instances might use it
        logger.debug("HybridRouter sync wrapper closed")


class OllamaPool:
    """
    Synchronous wrapper for SOLLOL's async OllamaPool.

    Provides synchronous methods for task distribution across Ollama nodes.

    Example:
        pool = OllamaPool.auto_configure()

        response = pool.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": "Hello!"}],
            priority=5
        )
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize synchronous OllamaPool.

        Args:
            *args, **kwargs: Same arguments as async OllamaPool
        """
        from sollol.pool import OllamaPool as AsyncOllamaPool

        self._loop_manager = get_event_loop()
        self._async_pool = AsyncOllamaPool(*args, **kwargs)
        logger.info("Synchronous OllamaPool initialized")

    @classmethod
    def auto_configure(cls, **kwargs):
        """
        Auto-configure pool with discovered nodes.

        Args:
            **kwargs: Additional configuration options

        Returns:
            Configured OllamaPool instance
        """
        from sollol.pool import OllamaPool as AsyncOllamaPool

        loop_manager = get_event_loop()
        async_pool = loop_manager.run_coroutine(
            AsyncOllamaPool.auto_configure(**kwargs), timeout=30.0
        )

        instance = cls.__new__(cls)
        instance._loop_manager = loop_manager
        instance._async_pool = async_pool
        logger.info("Auto-configured synchronous OllamaPool")
        return instance

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        priority: int = 5,
        timeout: Optional[float] = 300,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a chat completion request synchronously.

        Args:
            model: Model name
            messages: List of message dicts
            priority: Request priority (1-10, 10=highest)
            timeout: Request timeout in seconds
            **kwargs: Additional arguments

        Returns:
            Response dict
        """
        coro = self._async_pool.chat(model=model, messages=messages, priority=priority, **kwargs)
        return self._loop_manager.run_coroutine(coro, timeout=timeout)

    def generate(
        self, model: str, prompt: str, priority: int = 5, timeout: Optional[float] = 300, **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text synchronously.

        Args:
            model: Model name
            prompt: Input prompt
            priority: Request priority (1-10)
            timeout: Request timeout in seconds
            **kwargs: Additional arguments

        Returns:
            Response dict
        """
        coro = self._async_pool.generate(model=model, prompt=prompt, priority=priority, **kwargs)
        return self._loop_manager.run_coroutine(coro, timeout=timeout)

    def embed(
        self, model: str, input: str, priority: int = 5, timeout: Optional[float] = 60, **kwargs
    ) -> Dict[str, Any]:
        """
        Generate embeddings synchronously.

        Args:
            model: Model name
            input: Text to embed
            priority: Request priority (1-10)
            timeout: Request timeout in seconds
            **kwargs: Additional arguments

        Returns:
            Response dict with embeddings
        """
        coro = self._async_pool.embed(model=model, input=input, priority=priority, **kwargs)
        return self._loop_manager.run_coroutine(coro, timeout=timeout)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pool statistics synchronously.

        Returns:
            Statistics dict
        """
        coro = self._async_pool.get_stats()
        return self._loop_manager.run_coroutine(coro, timeout=10.0)

    def add_node(self, host: str, port: int = 11434):
        """
        Add a node to the pool.

        Args:
            host: Node hostname or IP
            port: Node port (default: 11434)
        """
        self._async_pool.add_node(host, port)

    def remove_node(self, host: str, port: int = 11434):
        """
        Remove a node from the pool.

        Args:
            host: Node hostname or IP
            port: Node port
        """
        self._async_pool.remove_node(host, port)


def sync_wrapper(async_func):
    """
    Decorator to convert async functions to sync.

    Example:
        @sync_wrapper
        async def my_async_function():
            await asyncio.sleep(1)
            return "done"

        # Can now call synchronously
        result = my_async_function()
    """

    @wraps(async_func)
    def wrapper(*args, **kwargs):
        loop_manager = get_event_loop()
        coro = async_func(*args, **kwargs)
        timeout = kwargs.pop("timeout", None)
        return loop_manager.run_coroutine(coro, timeout=timeout)

    return wrapper


# Cleanup on module exit
import atexit


def _cleanup():
    """Clean up the event loop on exit."""
    global _event_loop
    if _event_loop:
        _event_loop.close()
        _event_loop = None


atexit.register(_cleanup)
