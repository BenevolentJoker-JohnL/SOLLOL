"""
Network Observability for SOLLOL

Comprehensive monitoring of all Ollama and llama.cpp backend actions.

Features:
- Real-time request/response logging
- Backend performance tracking
- Model lifecycle events (load/unload/inference)
- Network traffic analysis
- Error tracking and alerting
- Activity stream for dashboard
- Configurable sampling to reduce overhead
"""

import json
import logging
import os
import queue
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Network event types."""
    # Ollama events
    OLLAMA_REQUEST = "ollama_request"
    OLLAMA_RESPONSE = "ollama_response"
    OLLAMA_ERROR = "ollama_error"
    OLLAMA_MODEL_LOAD = "ollama_model_load"
    OLLAMA_MODEL_UNLOAD = "ollama_model_unload"

    # llama.cpp RPC events
    RPC_REQUEST = "rpc_request"
    RPC_RESPONSE = "rpc_response"
    RPC_ERROR = "rpc_error"
    RPC_BACKEND_CONNECT = "rpc_backend_connect"
    RPC_BACKEND_DISCONNECT = "rpc_backend_disconnect"

    # Coordinator events
    COORDINATOR_START = "coordinator_start"
    COORDINATOR_STOP = "coordinator_stop"
    COORDINATOR_MODEL_LOAD = "coordinator_model_load"

    # Node health events
    NODE_HEALTH_CHECK = "node_health_check"
    NODE_STATUS_CHANGE = "node_status_change"
    NODE_DISCOVERED = "node_discovered"
    NODE_REMOVED = "node_removed"

    # Performance events
    HIGH_LATENCY = "high_latency"
    VRAM_EXHAUSTED = "vram_exhausted"
    BACKEND_OVERLOADED = "backend_overloaded"


@dataclass
class NetworkEvent:
    """A network observability event."""
    event_type: EventType
    timestamp: float
    backend: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # info, warning, error, critical

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "backend": self.backend,
            "details": self.details,
            "severity": self.severity,
            "iso_timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
        }


class NetworkObserver:
    """
    Centralized network observability system.

    Monitors all Ollama and llama.cpp backend actions in real-time.
    """

    def __init__(
        self,
        max_events: int = None,
        redis_url: str = None,
        enable_sampling: bool = None,
        sample_rate: float = None
    ):
        """
        Initialize network observer.

        Args:
            max_events: Maximum events to keep in memory (default: 10000, env: SOLLOL_OBSERVER_MAX_EVENTS)
            redis_url: Redis connection URL for dashboard pub/sub (default: redis://localhost:6379, env: SOLLOL_REDIS_URL)
            enable_sampling: Enable sampling to reduce overhead (default: True, env: SOLLOL_OBSERVER_SAMPLING)
            sample_rate: Fraction of info events to log (0.0-1.0, default: 0.1 = 10%, env: SOLLOL_OBSERVER_SAMPLE_RATE)
        """
        # Load from environment variables if not provided
        if max_events is None:
            max_events = int(os.getenv("SOLLOL_OBSERVER_MAX_EVENTS", "10000"))
        if redis_url is None:
            redis_url = os.getenv("SOLLOL_REDIS_URL", "redis://localhost:6379")
        if enable_sampling is None:
            enable_sampling = os.getenv("SOLLOL_OBSERVER_SAMPLING", "true").lower() in ("true", "1", "yes")
        if sample_rate is None:
            sample_rate = float(os.getenv("SOLLOL_OBSERVER_SAMPLE_RATE", "0.1"))

        self.max_events = max_events
        self.events: deque[NetworkEvent] = deque(maxlen=max_events)
        self.event_queue: queue.Queue = queue.Queue()

        # Sampling configuration
        self.enable_sampling = enable_sampling
        self.sample_rate = max(0.0, min(1.0, sample_rate))  # Clamp to [0, 1]

        # Statistics tracking
        self.stats = {
            "total_events": 0,
            "sampled_events": 0,  # Events that passed sampling
            "dropped_events": 0,  # Events dropped by sampling
            "events_by_type": {},
            "events_by_backend": {},
            "errors_by_backend": {},
            "current_active_requests": 0,
        }

        # Backend tracking
        self.active_requests: Dict[str, List[Dict[str, Any]]] = {}
        self.backend_metrics: Dict[str, Dict[str, Any]] = {}

        self._lock = threading.Lock()
        self._running = True

        # Initialize Redis for dashboard pub/sub
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info(f"📡 Network Observer connected to Redis at {redis_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e} - dashboard activity logs will be unavailable")
                self.redis_client = None

        # Start event processing thread
        self._event_thread = threading.Thread(
            target=self._process_events_loop,
            daemon=True,
            name="NetworkObserver-EventProcessor"
        )
        self._event_thread.start()

        sampling_status = f"sampling={self.sample_rate:.0%}" if self.enable_sampling else "sampling=disabled"
        logger.info(f"🔍 Network Observer initialized (event-driven monitoring, {sampling_status})")

    def log_event(
        self,
        event_type: EventType,
        backend: str,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "info"
    ):
        """
        Log a network event.

        Args:
            event_type: Type of event
            backend: Backend identifier
            details: Event details
            severity: Severity level (info, warning, error, critical)
        """
        event = NetworkEvent(
            event_type=event_type,
            timestamp=time.time(),
            backend=backend,
            details=details or {},
            severity=severity
        )

        # Add to queue for async processing
        self.event_queue.put(event)

    def _process_events_loop(self):
        """Background thread to process events asynchronously."""
        logger.debug("Network Observer event processing thread started")

        while self._running:
            try:
                # Get event from queue (blocking with timeout)
                event = self.event_queue.get(timeout=1)

                # Process event
                self._process_event(event)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing network event: {e}")

        logger.debug("Network Observer event processing thread stopped")

    def _process_event(self, event: NetworkEvent):
        """Process a single event with sampling."""
        with self._lock:
            # Update total event count (before sampling)
            self.stats["total_events"] += 1

            # Apply sampling for info-level events
            if self.enable_sampling and event.severity == "info":
                if random.random() > self.sample_rate:
                    # Drop this event
                    self.stats["dropped_events"] += 1
                    return

            # Event passed sampling (or sampling disabled)
            self.stats["sampled_events"] += 1

            # Add to event history
            self.events.append(event)

            event_type_key = event.event_type.value
            self.stats["events_by_type"][event_type_key] = \
                self.stats["events_by_type"].get(event_type_key, 0) + 1

            self.stats["events_by_backend"][event.backend] = \
                self.stats["events_by_backend"].get(event.backend, 0) + 1

            # Track errors
            if event.severity in ["error", "critical"]:
                self.stats["errors_by_backend"][event.backend] = \
                    self.stats["errors_by_backend"].get(event.backend, 0) + 1

            # Track active requests
            if event.event_type in [EventType.OLLAMA_REQUEST, EventType.RPC_REQUEST]:
                if event.backend not in self.active_requests:
                    self.active_requests[event.backend] = []

                request_info = {
                    "timestamp": event.timestamp,
                    "details": event.details,
                }
                self.active_requests[event.backend].append(request_info)
                self.stats["current_active_requests"] += 1

            elif event.event_type in [EventType.OLLAMA_RESPONSE, EventType.RPC_RESPONSE,
                                       EventType.OLLAMA_ERROR, EventType.RPC_ERROR]:
                # Remove from active requests
                if event.backend in self.active_requests:
                    if self.active_requests[event.backend]:
                        self.active_requests[event.backend].pop(0)
                        self.stats["current_active_requests"] = max(
                            0, self.stats["current_active_requests"] - 1
                        )

            # Update backend metrics
            self._update_backend_metrics(event)

            # Publish to Redis for dashboard activity logs
            self._publish_to_dashboard(event)

            # Log significant events
            if event.severity in ["warning", "error", "critical"]:
                logger.info(
                    f"⚠️  {event.event_type.value}: {event.backend} - {event.details}"
                )

    def _update_backend_metrics(self, event: NetworkEvent):
        """Update backend-specific metrics."""
        backend = event.backend

        if backend not in self.backend_metrics:
            self.backend_metrics[backend] = {
                "total_requests": 0,
                "total_errors": 0,
                "total_latency_ms": 0,
                "avg_latency_ms": 0,
                "last_activity": event.timestamp,
                "status": "active",
            }

        metrics = self.backend_metrics[backend]
        metrics["last_activity"] = event.timestamp

        # Update request count
        if event.event_type in [EventType.OLLAMA_REQUEST, EventType.RPC_REQUEST]:
            metrics["total_requests"] += 1

        # Update error count
        if event.severity in ["error", "critical"]:
            metrics["total_errors"] += 1

        # Update latency
        if "latency_ms" in event.details:
            latency = event.details["latency_ms"]
            metrics["total_latency_ms"] += latency
            # Avoid division by zero
            if metrics["total_requests"] > 0:
                metrics["avg_latency_ms"] = metrics["total_latency_ms"] / metrics["total_requests"]

    def _publish_to_dashboard(self, event: NetworkEvent):
        """Publish event to Redis for dashboard activity logs."""
        if not self.redis_client:
            return

        try:
            # Determine the Redis channel based on event type
            channel = None
            if event.event_type in [EventType.OLLAMA_REQUEST, EventType.OLLAMA_RESPONSE, EventType.OLLAMA_ERROR]:
                channel = "sollol:dashboard:ollama:activity"
            elif event.event_type in [EventType.RPC_REQUEST, EventType.RPC_RESPONSE, EventType.RPC_ERROR,
                                       EventType.RPC_BACKEND_CONNECT, EventType.RPC_BACKEND_DISCONNECT,
                                       EventType.COORDINATOR_START, EventType.COORDINATOR_STOP,
                                       EventType.COORDINATOR_MODEL_LOAD]:
                channel = "sollol:dashboard:rpc:activity"

            if not channel:
                return

            # Format event for dashboard
            message = {
                "timestamp": event.timestamp,
                "backend": event.backend,
                "event_type": event.event_type.value,
                "severity": event.severity,
                "details": event.details,
            }

            # Publish to Redis channel
            self.redis_client.publish(channel, json.dumps(message))

            # Also publish coordinator events to routing_events channel for dashboard routing decisions
            if event.event_type in [EventType.COORDINATOR_START, EventType.COORDINATOR_STOP,
                                     EventType.COORDINATOR_MODEL_LOAD]:
                routing_message = {
                    "timestamp": event.timestamp,
                    "event": event.event_type.value,
                    "backend": event.backend,
                    "details": event.details,
                    "severity": event.severity
                }
                self.redis_client.publish("sollol:routing_events", json.dumps(routing_message))

        except Exception as e:
            logger.debug(f"Failed to publish event to Redis: {e}")

    def get_recent_events(
        self,
        limit: int = 100,
        event_type: Optional[EventType] = None,
        backend: Optional[str] = None,
        min_severity: str = "info"
    ) -> List[Dict[str, Any]]:
        """
        Get recent network events.

        Args:
            limit: Maximum number of events to return
            event_type: Filter by event type
            backend: Filter by backend
            min_severity: Minimum severity level

        Returns:
            List of event dicts
        """
        severity_levels = {"info": 0, "warning": 1, "error": 2, "critical": 3}
        min_level = severity_levels.get(min_severity, 0)

        with self._lock:
            filtered_events = []

            for event in reversed(self.events):
                # Apply filters
                if event_type and event.event_type != event_type:
                    continue

                if backend and event.backend != backend:
                    continue

                event_level = severity_levels.get(event.severity, 0)
                if event_level < min_level:
                    continue

                filtered_events.append(event.to_dict())

                if len(filtered_events) >= limit:
                    break

            return filtered_events

    def get_backend_metrics(self, backend: Optional[str] = None) -> Dict[str, Any]:
        """
        Get backend performance metrics.

        Args:
            backend: Specific backend (None for all)

        Returns:
            Backend metrics dict
        """
        with self._lock:
            if backend:
                return self.backend_metrics.get(backend, {})
            else:
                return dict(self.backend_metrics)

    def get_active_requests(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get currently active requests by backend."""
        with self._lock:
            return dict(self.active_requests)

    def get_stats(self) -> Dict[str, Any]:
        """Get overall observability statistics."""
        with self._lock:
            total = self.stats["total_events"]
            sampled = self.stats["sampled_events"]
            dropped = self.stats["dropped_events"]

            return {
                **self.stats,
                "backends_tracked": len(self.backend_metrics),
                "events_in_memory": len(self.events),
                "sampling_enabled": self.enable_sampling,
                "sample_rate": self.sample_rate,
                "sampling_efficiency": (dropped / total) if total > 0 else 0.0,
            }

    def shutdown(self):
        """Shutdown the observer."""
        logger.info("🔍 Shutting down Network Observer")
        self._running = False
        if self._event_thread.is_alive():
            self._event_thread.join(timeout=2)


# Global observer instance
_global_observer: Optional[NetworkObserver] = None
_observer_lock = threading.Lock()


def get_observer() -> NetworkObserver:
    """Get or create global network observer instance."""
    global _global_observer

    with _observer_lock:
        if _global_observer is None:
            # Use 100% sampling for dashboard visibility (was 10% by default)
            _global_observer = NetworkObserver(sample_rate=1.0)

        return _global_observer


def log_ollama_request(backend: str, model: str, operation: str, **details):
    """Log an Ollama request."""
    observer = get_observer()
    observer.log_event(
        EventType.OLLAMA_REQUEST,
        backend=backend,
        details={"model": model, "operation": operation, **details},
        severity="info"
    )


def log_ollama_response(backend: str, model: str, latency_ms: float, **details):
    """Log an Ollama response."""
    observer = get_observer()
    severity = "warning" if latency_ms > 10000 else "info"
    observer.log_event(
        EventType.OLLAMA_RESPONSE,
        backend=backend,
        details={"model": model, "latency_ms": latency_ms, **details},
        severity=severity
    )


def log_ollama_error(backend: str, model: str, error: str, **details):
    """Log an Ollama error."""
    observer = get_observer()
    observer.log_event(
        EventType.OLLAMA_ERROR,
        backend=backend,
        details={"model": model, "error": error, **details},
        severity="error"
    )


def log_rpc_request(backend: str, model: str, **details):
    """Log an RPC request."""
    observer = get_observer()
    observer.log_event(
        EventType.RPC_REQUEST,
        backend=backend,
        details={"model": model, **details},
        severity="info"
    )


def log_rpc_response(backend: str, model: str, latency_ms: float, **details):
    """Log an RPC response."""
    observer = get_observer()
    severity = "warning" if latency_ms > 15000 else "info"
    observer.log_event(
        EventType.RPC_RESPONSE,
        backend=backend,
        details={"model": model, "latency_ms": latency_ms, **details},
        severity=severity
    )


def log_rpc_error(backend: str, model: str, error: str, **details):
    """Log an RPC error."""
    observer = get_observer()
    observer.log_event(
        EventType.RPC_ERROR,
        backend=backend,
        details={"model": model, "error": error, **details},
        severity="error"
    )


def log_node_health_check(backend: str, status: str, latency_ms: float, **details):
    """Log a node health check."""
    observer = get_observer()
    severity = "info" if status == "healthy" else "warning"
    observer.log_event(
        EventType.NODE_HEALTH_CHECK,
        backend=backend,
        details={"status": status, "latency_ms": latency_ms, **details},
        severity=severity
    )


def log_node_status_change(backend: str, old_status: str, new_status: str, **details):
    """Log a node status change."""
    observer = get_observer()
    severity = "warning" if new_status != "healthy" else "info"
    observer.log_event(
        EventType.NODE_STATUS_CHANGE,
        backend=backend,
        details={"old_status": old_status, "new_status": new_status, **details},
        severity=severity
    )
