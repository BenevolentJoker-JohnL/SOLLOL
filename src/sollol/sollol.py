"""
SOLLOL Main Orchestration Class - Application-friendly API.

This module provides a programmatic interface for applications to configure
and control SOLLOL entirely from within Python code, without CLI or external configs.
"""

import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from sollol.config import SOLLOLConfig


class SOLLOL:
    """
    Main SOLLOL orchestration class.

    Provides a simple, application-friendly API for managing SOLLOL entirely
    from within your application code.

    Example:
        ```python
        from sollol import SOLLOL

        # Zero-config startup (reads from environment if set)
        sollol = SOLLOL()
        sollol.start()  # Runs gateway in background thread

        # Your app can now use SOLLOL via the gateway
        # http://localhost:11434/api/chat

        # Or with explicit configuration (overrides env vars)
        sollol = SOLLOL(
            port=8000,
            ray_workers=8,
            dask_workers=4,
            ollama_nodes=[{"host": "10.0.0.2", "port": 11434}],
            rpc_backends=[{"host": "10.0.0.5", "port": 50052}]
        )
        sollol.start(blocking=False)

        # Environment variable configuration:
        # export SOLLOL_RAY_WORKERS=16
        # export SOLLOL_PORT=8000
        # from sollol import SOLLOL
        # sollol = SOLLOL()  # Reads env vars
        # sollol.start()

        # Check status
        status = sollol.get_status()
        print(status)

        # Stop when done
        sollol.stop()
        ```
    """

    def __init__(
        self,
        port: int = None,
        ray_workers: int = None,
        dask_workers: int = None,
        enable_batch_processing: bool = None,
        autobatch_interval: int = None,
        ollama_nodes: Optional[List[Dict]] = None,
        rpc_backends: Optional[List[Dict]] = None,
    ):
        """
        Initialize SOLLOL with configuration.

        Reads from environment variables if parameters not provided:
        - SOLLOL_PORT or PORT (default: 11434)
        - SOLLOL_RAY_WORKERS or RAY_WORKERS (default: 4)
        - SOLLOL_DASK_WORKERS or DASK_WORKERS (default: 2)
        - SOLLOL_BATCH_PROCESSING (default: true)
        - SOLLOL_AUTOBATCH_INTERVAL or AUTOBATCH_INTERVAL (default: 60)
        - RPC_BACKENDS (comma-separated)
        - OLLAMA_NODES (comma-separated)

        Args:
            port: Gateway port (default from env or 11434)
            ray_workers: Number of Ray actors (default from env or 4)
            dask_workers: Number of Dask workers (default from env or 2)
            enable_batch_processing: Enable Dask batch processing (default from env or True)
            autobatch_interval: Seconds between autobatch cycles (default from env or 60)
            ollama_nodes: List of Ollama node dicts (auto-discovers if None)
            rpc_backends: List of RPC backend dicts (auto-discovers if None)
        """
        # Configure logging
        import logging
        import os

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # Read from environment variables if not provided
        self.port = (
            port if port is not None else int(os.getenv("SOLLOL_PORT", os.getenv("PORT", "11434")))
        )
        self.ray_workers = (
            ray_workers
            if ray_workers is not None
            else int(os.getenv("SOLLOL_RAY_WORKERS", os.getenv("RAY_WORKERS", "4")))
        )
        self.dask_workers = (
            dask_workers
            if dask_workers is not None
            else int(os.getenv("SOLLOL_DASK_WORKERS", os.getenv("DASK_WORKERS", "2")))
        )
        self.enable_batch_processing = (
            enable_batch_processing
            if enable_batch_processing is not None
            else os.getenv("SOLLOL_BATCH_PROCESSING", "true").lower() in ("true", "1", "yes")
        )
        self.autobatch_interval = (
            autobatch_interval
            if autobatch_interval is not None
            else int(os.getenv("SOLLOL_AUTOBATCH_INTERVAL", os.getenv("AUTOBATCH_INTERVAL", "60")))
        )
        self.ollama_nodes = ollama_nodes
        self.rpc_backends = rpc_backends

        # Internal state
        self._gateway_thread: Optional[threading.Thread] = None
        self._running = False

        self.logger.info("SOLLOL initialized with configuration:")
        self.logger.info(f"  Port: {self.port}")
        self.logger.info(f"  Ray workers: {self.ray_workers}")
        self.logger.info(f"  Dask workers: {self.dask_workers}")
        self.logger.info(
            f"  Batch processing: {'enabled' if self.enable_batch_processing else 'disabled'}"
        )
        self.logger.info(
            f"  Ollama nodes: {len(self.ollama_nodes) if self.ollama_nodes else 'auto-discover'}"
        )
        self.logger.info(
            f"  RPC backends: {len(self.rpc_backends) if self.rpc_backends else 'auto-discover'}"
        )

    def start(self, blocking: bool = False):
        """
        Start SOLLOL gateway.

        Args:
            blocking: If True, blocks until stopped. If False, runs in background thread.

        Example:
            ```python
            # Non-blocking (recommended for applications)
            sollol.start(blocking=False)
            # Your app code continues here...

            # Blocking (recommended for standalone SOLLOL service)
            sollol.start(blocking=True)
            ```
        """
        if self._running:
            self.logger.warning("SOLLOL is already running")
            return

        self.logger.info("Starting SOLLOL gateway...")

        if blocking:
            # Run gateway in current thread (blocks)
            self._start_gateway()
        else:
            # Run gateway in background thread
            self._gateway_thread = threading.Thread(target=self._start_gateway, daemon=True)
            self._gateway_thread.start()
            self._running = True
            self.logger.info(f"✅ Gateway started in background thread")
            self.logger.info(f"   API available at http://localhost:{self.port}")
            self.logger.info(f"   Health check: http://localhost:{self.port}/api/health")
            self.logger.info(f"   API docs: http://localhost:{self.port}/docs")

    def _start_gateway(self):
        """Internal method to start the FastAPI gateway."""
        from sollol.gateway import start_api

        self._running = True

        start_api(
            port=self.port,
            rpc_backends=self.rpc_backends,
            ollama_nodes=self.ollama_nodes,
            ray_workers=self.ray_workers,
            dask_workers=self.dask_workers,
            enable_batch_processing=self.enable_batch_processing,
            autobatch_interval=self.autobatch_interval,
        )

    def stop(self):
        """
        Stop SOLLOL gateway.

        Note: Gateway runs in a daemon thread and will stop when your application exits.
        Ray and Dask processes are initialized inside the gateway and will also stop.
        """
        self.logger.info("Stopping SOLLOL...")
        self._running = False

        if self._gateway_thread and self._gateway_thread.is_alive():
            self.logger.warning("⚠️  Gateway is running in a daemon thread")
            self.logger.warning("    It will stop when your application exits")
            self.logger.warning("    To force stop, restart your application")

        self.logger.info("Stopped")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current SOLLOL status.

        Returns:
            Dictionary containing current status information

        Example:
            ```python
            status = sollol.get_status()
            print(f"Running: {status['running']}")
            print(f"Port: {status['port']}")
            ```
        """
        return {
            "running": self._running,
            "port": self.port,
            "ray_workers": self.ray_workers,
            "dask_workers": self.dask_workers,
            "batch_processing_enabled": self.enable_batch_processing,
            "ollama_nodes": len(self.ollama_nodes) if self.ollama_nodes else "auto-discover",
            "rpc_backends": len(self.rpc_backends) if self.rpc_backends else "auto-discover",
            "timestamp": datetime.now().isoformat(),
            "endpoints": {
                "gateway": f"http://localhost:{self.port}",
                "api_docs": f"http://localhost:{self.port}/docs",
                "health": f"http://localhost:{self.port}/api/health",
                "stats": f"http://localhost:{self.port}/api/stats",
            },
        }

    def get_health(self) -> Dict[str, Any]:
        """
        Get health status from the gateway.

        Returns:
            Health information from /api/health endpoint

        Note: Only works when SOLLOL is running.
        """
        if not self._running:
            return {"error": "SOLLOL is not running"}

        import httpx

        try:
            resp = httpx.get(f"http://localhost:{self.port}/api/health", timeout=5.0)
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics from the gateway.

        Returns:
            Statistics from /api/stats endpoint

        Note: Only works when SOLLOL is running.
        """
        if not self._running:
            return {"error": "SOLLOL is not running"}

        import httpx

        try:
            resp = httpx.get(f"http://localhost:{self.port}/api/stats", timeout=5.0)
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def __repr__(self) -> str:
        """String representation of SOLLOL instance."""
        return (
            f"SOLLOL(running={self._running}, "
            f"port={self.port}, "
            f"ray_workers={self.ray_workers}, "
            f"dask_workers={self.dask_workers})"
        )
