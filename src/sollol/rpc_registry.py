"""
RPC Backend Registry - Health monitoring and metrics for RPC backends.

Similar to OllamaNodeRegistry but for llama.cpp RPC servers used in model sharding.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sollol.network_observer import EventType, get_observer
from sollol.rpc_discovery import check_rpc_server

logger = logging.getLogger(__name__)


@dataclass
class RPCMetrics:
    """Performance metrics for an RPC backend."""

    total_requests: int = 0
    total_failures: int = 0
    total_latency_ms: float = 0.0
    last_health_check: float = 0.0
    last_health_status: bool = False

    @property
    def avg_latency(self) -> float:
        """Average latency in milliseconds."""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    @property
    def success_rate(self) -> float:
        """Success rate (0.0 to 1.0)."""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.total_failures) / self.total_requests


@dataclass
class RPCBackend:
    """Represents a single RPC backend with health and metrics."""

    host: str
    port: int
    metrics: RPCMetrics = field(default_factory=RPCMetrics)

    def __post_init__(self):
        # Perform initial health check
        self.check_health()

    def check_health(self, timeout: float = 1.0) -> bool:
        """
        Check if RPC backend is healthy.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            True if backend is reachable
        """
        start = time.time()
        is_healthy = check_rpc_server(self.host, self.port, timeout=timeout)
        latency = (time.time() - start) * 1000  # Convert to ms

        self.metrics.last_health_check = time.time()
        self.metrics.last_health_status = is_healthy

        if is_healthy:
            # Update metrics only if successful
            self.metrics.total_requests += 1
            self.metrics.total_latency_ms += latency
        else:
            self.metrics.total_failures += 1

        return is_healthy

    @property
    def is_healthy(self) -> bool:
        """Check if backend is currently healthy (with caching)."""
        # Re-check if last check was > 30 seconds ago
        if time.time() - self.metrics.last_health_check > 30:
            return self.check_health()
        return self.metrics.last_health_status

    @property
    def address(self) -> str:
        """Return formatted address."""
        return f"{self.host}:{self.port}"

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "host": self.host,
            "port": self.port,
            "healthy": self.is_healthy,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "total_failures": self.metrics.total_failures,
                "avg_latency_ms": self.metrics.avg_latency,
                "success_rate": self.metrics.success_rate,
                "last_check": self.metrics.last_health_check,
            },
        }


class RPCBackendRegistry:
    """
    Registry for managing RPC backends with health monitoring.

    Similar to OllamaNodeRegistry but for RPC servers.
    """

    def __init__(self, enable_heartbeat: bool = True):
        """
        Initialize RPC backend registry.

        Args:
            enable_heartbeat: Enable periodic health check heartbeat (default: True, env: SOLLOL_RPC_BACKEND_HEARTBEAT)
        """
        self.backends: Dict[str, RPCBackend] = {}

        # Heartbeat configuration - load from environment
        if enable_heartbeat is None:
            enable_heartbeat = os.getenv("SOLLOL_RPC_BACKEND_HEARTBEAT", "true").lower() in (
                "true",
                "1",
                "yes",
            )

        self.enable_heartbeat = enable_heartbeat
        self._heartbeat_interval = int(
            os.getenv("SOLLOL_RPC_BACKEND_HEARTBEAT_INTERVAL", "600")
        )  # seconds (10 minutes)
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_running = False

        if self.enable_heartbeat:
            self._start_heartbeat()

    def _start_heartbeat(self):
        """Start periodic health check heartbeat thread."""
        if self._heartbeat_running:
            return

        self._heartbeat_running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="RPCBackendRegistry-Heartbeat"
        )
        self._heartbeat_thread.start()
        logger.info(f"RPC backend heartbeat started (interval: {self._heartbeat_interval}s)")

    def _heartbeat_loop(self):
        """Periodic health check loop for all backends."""
        while self._heartbeat_running:
            try:
                time.sleep(self._heartbeat_interval)

                if self.backends:
                    # Run health check on all backends
                    self.health_check_all()

            except Exception as e:
                logger.error(f"RPC backend heartbeat error: {e}")

    def stop_heartbeat(self):
        """Stop the heartbeat thread."""
        self._heartbeat_running = False
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2)
        logger.info("RPC backend heartbeat stopped")

    def add_backend(self, host: str, port: int = 50052) -> RPCBackend:
        """
        Add an RPC backend to the registry.

        Args:
            host: Backend host/IP
            port: Backend port (default: 50052)

        Returns:
            RPCBackend instance
        """
        address = f"{host}:{port}"

        if address in self.backends:
            logger.info(f"RPC backend already registered: {address}")
            return self.backends[address]

        backend = RPCBackend(host=host, port=port)
        self.backends[address] = backend

        logger.info(f"Added RPC backend: {address} (healthy={backend.is_healthy})")

        # Publish backend registration event to dashboard
        observer = get_observer()
        observer.log_event(
            EventType.RPC_BACKEND_CONNECT,
            backend=address,
            details={
                "model": "rpc_registry",
                "action": "backend_added",
                "status": "healthy" if backend.is_healthy else "unhealthy",
                "total_backends": len(self.backends),
            },
            severity="info",
        )

        return backend

    def remove_backend(self, host: str, port: int = 50052) -> bool:
        """
        Remove an RPC backend from the registry.

        Args:
            host: Backend host/IP
            port: Backend port

        Returns:
            True if backend was removed
        """
        address = f"{host}:{port}"
        if address in self.backends:
            del self.backends[address]
            logger.info(f"Removed RPC backend: {address}")
            return True
        return False

    def get_healthy_backends(self) -> List[RPCBackend]:
        """Get list of healthy backends."""
        return [b for b in self.backends.values() if b.is_healthy]

    def get_best_backend(self) -> Optional[RPCBackend]:
        """
        Get the best backend based on health and performance.

        Returns:
            Best backend or None if no healthy backends
        """
        healthy = self.get_healthy_backends()
        if not healthy:
            return None

        # Sort by success rate and latency
        return min(
            healthy,
            key=lambda b: (
                -b.metrics.success_rate,  # Higher success rate = better
                b.metrics.avg_latency,  # Lower latency = better
            ),
        )

    def health_check_all(self, timeout: float = 1.0) -> Dict[str, bool]:
        """
        Check health of all backends.

        Args:
            timeout: Connection timeout per backend

        Returns:
            Dict of {address: is_healthy}
        """
        results = {}
        for address, backend in self.backends.items():
            results[address] = backend.check_health(timeout=timeout)

        healthy_count = sum(1 for h in results.values() if h)
        logger.info(f"Health check: {healthy_count}/{len(results)} RPC backends healthy")

        # Publish heartbeat to dashboard with overall status
        if self.backends:
            observer = get_observer()
            healthy_backends = [addr for addr, is_healthy in results.items() if is_healthy]

            observer.log_event(
                EventType.RPC_BACKEND_CONNECT,
                backend="rpc_registry",
                details={
                    "model": "rpc_backends",
                    "rpc_backends": healthy_count,
                    "rpc_addresses": healthy_backends,
                    "status": "healthy" if healthy_count > 0 else "unhealthy",
                    "type": "heartbeat",
                    "total_configured": len(self.backends),
                    "total_active": healthy_count,
                },
                severity="info",
            )

        return results

    def get_stats(self) -> Dict:
        """Get overall statistics."""
        healthy = self.get_healthy_backends()

        stats = {
            "total_backends": len(self.backends),
            "healthy_backends": len(healthy),
            "backends": [],
        }

        for backend in self.backends.values():
            stats["backends"].append(backend.to_dict())

        return stats

    def load_from_config(self, rpc_backends: List[Dict]) -> None:
        """
        Load RPC backends from config.

        Args:
            rpc_backends: List of {"host": "...", "port": ...} dicts
        """
        for config in rpc_backends:
            self.add_backend(host=config["host"], port=config.get("port", 50052))

        logger.info(f"Loaded {len(rpc_backends)} RPC backends from config")
