"""
Graceful Shutdown for SOLLOL.
Handles in-flight requests and drains connections before exit.
"""

import asyncio
import signal
import sys
from typing import Callable, List, Optional


class GracefulShutdown:
    """
    Graceful shutdown handler for SOLLOL gateway.

    Features:
    - Handles SIGTERM and SIGINT signals
    - Drains in-flight requests
    - Stops accepting new requests
    - Waits for active tasks to complete
    - Configurable shutdown timeout
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize graceful shutdown handler.

        Args:
            timeout: Maximum seconds to wait for requests to complete
        """
        self.timeout = timeout
        self.is_shutting_down = False
        self.active_requests = 0
        self.shutdown_callbacks: List[Callable] = []

    def register_callback(self, callback: Callable):
        """
        Register callback to run during shutdown.

        Args:
            callback: Async function to call during shutdown
        """
        self.shutdown_callbacks.append(callback)

    def increment_requests(self):
        """Increment active request counter."""
        if self.is_shutting_down:
            raise ShutdownInProgress("Server is shutting down, not accepting new requests")
        self.active_requests += 1

    def decrement_requests(self):
        """Decrement active request counter."""
        self.active_requests = max(0, self.active_requests - 1)

    async def shutdown(self):
        """Execute graceful shutdown sequence."""
        if self.is_shutting_down:
            return

        print("\n🛑 Graceful shutdown initiated...")
        self.is_shutting_down = True

        # Step 1: Stop accepting new requests
        print("   ⏸️  Stopped accepting new requests")

        # Step 2: Wait for in-flight requests to complete
        if self.active_requests > 0:
            print(f"   ⏳ Waiting for {self.active_requests} in-flight requests to complete...")

            start_time = asyncio.get_event_loop().time()
            while self.active_requests > 0:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > self.timeout:
                    print(
                        f"   ⚠️  Shutdown timeout ({self.timeout}s) reached. "
                        f"{self.active_requests} requests still active."
                    )
                    break
                await asyncio.sleep(0.1)

        # Step 3: Run shutdown callbacks
        if self.shutdown_callbacks:
            print(f"   🔄 Running {len(self.shutdown_callbacks)} shutdown callbacks...")
            for callback in self.shutdown_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    print(f"   ⚠️  Shutdown callback error: {e}")

        print("   ✅ Graceful shutdown complete")

    def setup_signal_handlers(self, app=None):
        """
        Setup signal handlers for SIGTERM and SIGINT.

        Args:
            app: FastAPI app instance (optional)
        """

        def handle_signal(signum, frame):
            """Signal handler function."""
            signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
            print(f"\n📡 Received {signal_name} signal")

            # Create shutdown task
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.shutdown())
            else:
                loop.run_until_complete(self.shutdown())

            # Exit
            sys.exit(0)

        # Register signal handlers
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        print("📡 Signal handlers registered (SIGTERM, SIGINT)")


class ShutdownInProgress(Exception):
    """Exception raised when shutdown is in progress."""

    pass


# Middleware for FastAPI
class GracefulShutdownMiddleware:
    """FastAPI middleware for graceful shutdown."""

    def __init__(self, app, shutdown_handler: GracefulShutdown):
        self.app = app
        self.shutdown = shutdown_handler

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Check if shutting down
        if self.shutdown.is_shutting_down:
            await send(
                {
                    "type": "http.response.start",
                    "status": 503,
                    "headers": [[b"content-type", b"application/json"]],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"error": "Server is shutting down"}',
                }
            )
            return

        # Track request
        self.shutdown.increment_requests()
        try:
            await self.app(scope, receive, send)
        finally:
            self.shutdown.decrement_requests()
