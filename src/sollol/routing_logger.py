"""
Structured routing decision logger for SOLLOL.

Captures all routing decisions, load balancing choices, and performance events
across distributed SOLLOL instances. Publishes to Redis for centralized monitoring.

Events logged:
- ROUTE_DECISION: Which backend was selected and why
- TASK_QUEUED: Task added to queue
- TASK_START: Execution begins
- TASK_COMPLETE: Task finished with timing
- WORKER_LOAD: Current worker resource usage
- FALLBACK_TRIGGERED: Automatic fallback between backends
- MODEL_SWITCH: Coordinator switching models
- CACHE_HIT: Routing decision retrieved from cache

All events include:
- timestamp: ISO 8601 timestamp
- instance_id: Unique SOLLOL instance identifier
- event_type: Event category
- model: Model being processed
- backend: Selected backend (ollama/rpc)
- metadata: Additional context
"""

import json
import logging
import os
import socket
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Global instance ID (unique per SOLLOL process)
_instance_id = None
_hostname = socket.gethostname()


def get_instance_id() -> str:
    """Get or create unique instance ID for this SOLLOL process."""
    global _instance_id
    if _instance_id is None:
        # Format: hostname_pid_uuid-prefix
        _instance_id = f"{_hostname}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    return _instance_id


class RoutingEventLogger:
    """
    Structured logger for routing decisions and performance events.

    Publishes events to Redis for centralized aggregation and monitoring.
    """

    # Event types
    ROUTE_DECISION = "ROUTE_DECISION"
    TASK_QUEUED = "TASK_QUEUED"
    TASK_START = "TASK_START"
    TASK_COMPLETE = "TASK_COMPLETE"
    WORKER_LOAD = "WORKER_LOAD"
    FALLBACK_TRIGGERED = "FALLBACK_TRIGGERED"
    MODEL_SWITCH = "MODEL_SWITCH"
    CACHE_HIT = "CACHE_HIT"
    COORDINATOR_START = "COORDINATOR_START"
    COORDINATOR_STOP = "COORDINATOR_STOP"
    RPC_BACKEND_SELECTED = "RPC_BACKEND_SELECTED"
    OLLAMA_NODE_SELECTED = "OLLAMA_NODE_SELECTED"

    def __init__(
        self,
        redis_client=None,
        channel: str = "sollol:routing_events",
        stream_key: str = "sollol:routing_stream",
        enabled: bool = True,
        console_output: bool = False,
    ):
        """
        Initialize routing event logger.

        Args:
            redis_client: Redis client for publishing events
            channel: Redis pub/sub channel
            stream_key: Redis stream key for persistent log
            enabled: Enable/disable logging (default: True)
            console_output: Also print events to console (default: False)
        """
        self.redis_client = redis_client
        self.channel = channel
        self.stream_key = stream_key
        self.enabled = enabled
        self.console_output = console_output
        self.instance_id = get_instance_id()

        # Check if Redis is available
        if self.redis_client:
            try:
                self.redis_client.ping()
                self.redis_available = True
                logger.debug(f"RoutingEventLogger connected to Redis: {channel}")
            except Exception as e:
                self.redis_available = False
                logger.warning(f"Redis not available for routing logs: {e}")
        else:
            self.redis_available = False

        if not self.enabled:
            logger.debug("RoutingEventLogger disabled")

    def _create_event(
        self,
        event_type: str,
        model: Optional[str] = None,
        backend: Optional[str] = None,
        **metadata,
    ) -> Dict[str, Any]:
        """Create structured event dictionary."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "instance_id": self.instance_id,
            "hostname": _hostname,
            "pid": os.getpid(),
            "event_type": event_type,
            "model": model,
            "backend": backend,
        }

        # Add all metadata fields
        event.update(metadata)

        return event

    def _publish_event(self, event: Dict[str, Any]):
        """Publish event to Redis and optionally console."""
        logger.info(f"[DEBUG] _publish_event called, enabled={self.enabled}")
        if not self.enabled:
            logger.info(f"[DEBUG] Routing logger disabled, skipping")
            return

        event_json = json.dumps(event)
        logger.info(f"[DEBUG] Event serialized to JSON")

        # Console output if enabled
        if self.console_output:
            self._print_event(event)

        # Publish to Redis if available
        logger.info(f"[DEBUG] redis_available={self.redis_available}, redis_client={self.redis_client is not None}")
        if self.redis_available and self.redis_client:
            try:
                # Pub/sub for real-time monitoring
                logger.info(f"[DEBUG] Publishing to channel={self.channel}, event_type={event.get('event_type')}")
                pub_result = self.redis_client.publish(self.channel, event_json)
                logger.info(f"[DEBUG] Publish result: {pub_result} subscribers received")

                # Stream for persistent history (with maxlen for memory management)
                logger.info(f"[DEBUG] Adding to stream={self.stream_key}")
                stream_id = self.redis_client.xadd(
                    self.stream_key, {"event": event_json}, maxlen=10000  # Keep last 10k events
                )
                logger.info(f"[DEBUG] Stream ID: {stream_id}")
            except Exception as e:
                logger.error(f"Failed to publish routing event: {e}")
                import traceback
                logger.error(traceback.format_exc())

    def _print_event(self, event: Dict[str, Any]):
        """Print event to console with color coding."""
        event_type = event["event_type"]
        model = event.get("model", "N/A")
        backend = event.get("backend", "N/A")

        # Color codes (ANSI)
        BLUE = "\033[94m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        CYAN = "\033[96m"
        RESET = "\033[0m"

        # Select color based on event type
        if event_type == self.ROUTE_DECISION:
            color = CYAN
        elif event_type in (self.TASK_START, self.TASK_QUEUED):
            color = BLUE
        elif event_type == self.TASK_COMPLETE:
            color = GREEN
        elif event_type == self.FALLBACK_TRIGGERED:
            color = YELLOW
        else:
            color = RESET

        timestamp = event["timestamp"].split(".")[0].split("T")[1]  # Extract HH:MM:SS

        print(
            f"{color}[{timestamp}] {event_type:20s} | "
            f"model={model:20s} | backend={backend:10s}{RESET}"
        )

        # Print reason/metadata on next line if available
        if "reason" in event:
            print(f"  └─ {event['reason']}")
        elif "duration" in event:
            print(f"  └─ duration={event['duration']:.2f}s")

    def log_route_decision(
        self, model: str, backend: str, reason: str, cached: bool = False, **metadata
    ):
        """
        Log routing decision.

        Args:
            model: Model name
            backend: Selected backend (ollama/rpc/llamacpp)
            reason: Why this backend was chosen
            cached: Whether decision came from cache
            **metadata: Additional context
        """
        event = self._create_event(
            self.ROUTE_DECISION if not cached else self.CACHE_HIT,
            model=model,
            backend=backend,
            reason=reason,
            cached=cached,
            **metadata,
        )
        self._publish_event(event)

    def log_task_queued(self, task_id: str, model: str, worker: str = None, **metadata):
        """Log task added to queue."""
        event = self._create_event(
            self.TASK_QUEUED, model=model, task_id=task_id, worker=worker, **metadata
        )
        self._publish_event(event)

    def log_task_start(self, task_id: str, model: str, worker: str, **metadata):
        """Log task execution started."""
        event = self._create_event(
            self.TASK_START,
            model=model,
            task_id=task_id,
            worker=worker,
            start_time=time.perf_counter(),
            **metadata,
        )
        self._publish_event(event)

    def log_task_complete(
        self,
        task_id: str,
        model: str,
        worker: str,
        duration: float,
        success: bool = True,
        **metadata,
    ):
        """Log task completion with timing."""
        event = self._create_event(
            self.TASK_COMPLETE,
            model=model,
            task_id=task_id,
            worker=worker,
            duration=duration,
            success=success,
            **metadata,
        )
        self._publish_event(event)

    def log_worker_load(
        self,
        worker: str,
        active_tasks: int,
        cpu_percent: float = None,
        memory_mb: float = None,
        **metadata,
    ):
        """Log current worker load."""
        event = self._create_event(
            self.WORKER_LOAD,
            worker=worker,
            active_tasks=active_tasks,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            **metadata,
        )
        self._publish_event(event)

    def log_fallback(self, model: str, from_backend: str, to_backend: str, reason: str, **metadata):
        """Log automatic fallback between backends."""
        event = self._create_event(
            self.FALLBACK_TRIGGERED,
            model=model,
            backend=to_backend,
            from_backend=from_backend,
            reason=reason,
            **metadata,
        )
        self._publish_event(event)

    def log_model_switch(self, from_model: str, to_model: str, backend: str, **metadata):
        """Log coordinator switching models."""
        event = self._create_event(
            self.MODEL_SWITCH, model=to_model, backend=backend, from_model=from_model, **metadata
        )
        self._publish_event(event)

    def log_coordinator_start(self, model: str, rpc_backends: int, **metadata):
        """Log llama.cpp coordinator startup."""
        event = self._create_event(
            self.COORDINATOR_START,
            model=model,
            backend="llamacpp",
            rpc_backends=rpc_backends,
            **metadata,
        )
        self._publish_event(event)

    def log_coordinator_stop(self, model: str, **metadata):
        """Log llama.cpp coordinator shutdown."""
        event = self._create_event(
            self.COORDINATOR_STOP, model=model, backend="llamacpp", **metadata
        )
        self._publish_event(event)

    def log_rpc_backend_selected(
        self, backend_host: str, backend_port: int, reason: str, **metadata
    ):
        """Log RPC backend selection."""
        event = self._create_event(
            self.RPC_BACKEND_SELECTED,
            backend="rpc",
            backend_host=backend_host,
            backend_port=backend_port,
            reason=reason,
            **metadata,
        )
        self._publish_event(event)

    def log_ollama_node_selected(self, node_url: str, reason: str, model: str, **metadata):
        """Log Ollama node selection."""
        logger.info(f"[DEBUG] log_ollama_node_selected called: node_url={node_url}, model={model}")
        event = self._create_event(
            self.OLLAMA_NODE_SELECTED,
            backend="ollama",
            node_url=node_url,
            model=model,
            reason=reason,
            **metadata,
        )
        logger.info(f"[DEBUG] Event created, calling _publish_event...")
        self._publish_event(event)
        logger.info(f"[DEBUG] _publish_event returned")


# Global routing logger instance
_routing_logger: Optional[RoutingEventLogger] = None


def get_routing_logger(
    redis_client=None,
    console_output: bool = None,
) -> RoutingEventLogger:
    """
    Get or create global routing event logger.

    Args:
        redis_client: Optional Redis client for publishing
        console_output: Override console output setting

    Returns:
        RoutingEventLogger instance
    """
    global _routing_logger

    if _routing_logger is None:
        # Check environment variables
        enabled = os.getenv("SOLLOL_ROUTING_LOG", "true").lower() in ("true", "1", "yes")

        if console_output is None:
            console_output = os.getenv("SOLLOL_ROUTING_LOG_CONSOLE", "false").lower() in (
                "true",
                "1",
                "yes",
            )

        # Create Redis client if not provided
        if redis_client is None and enabled:
            try:
                import redis

                redis_url = os.getenv("SOLLOL_REDIS_URL", "redis://localhost:6379")
                redis_client = redis.from_url(redis_url, decode_responses=True)
            except Exception as e:
                logger.debug(f"Could not create Redis client for routing logger: {e}")

        _routing_logger = RoutingEventLogger(
            redis_client=redis_client,
            enabled=enabled,
            console_output=console_output,
        )

    return _routing_logger


def enable_console_routing_log():
    """Enable console output for routing decisions (for debugging)."""
    global _routing_logger
    if _routing_logger:
        _routing_logger.console_output = True
    else:
        get_routing_logger(console_output=True)


__all__ = [
    "RoutingEventLogger",
    "get_routing_logger",
    "enable_console_routing_log",
]
