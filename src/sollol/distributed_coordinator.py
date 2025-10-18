"""
Distributed coordination for multi-instance SOLLOL deployments.

Enables multiple SOLLOL instances to coordinate routing decisions through
shared state in Redis, solving the multi-application concurrency problem.

Features:
- Service discovery (instances find each other)
- Shared cluster state (real-time node metrics)
- Distributed routing coordination
- Aggregated performance metrics

Usage:
    from sollol.distributed_coordinator import RedisCoordinator

    # Enable distributed mode
    coordinator = RedisCoordinator(redis_url="redis://localhost:6379")

    # Routing now coordinates across all instances
    router = IntelligentRouter(coordinator=coordinator)
"""

import json
import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


@dataclass
class InstanceInfo:
    """Information about a SOLLOL instance."""
    instance_id: str
    started_at: float
    last_heartbeat: float
    hostname: str
    version: str


@dataclass
class NodeState:
    """Aggregated state of an Ollama node across all instances."""
    host: str
    active_requests: int  # Total across all instances
    cpu_load: float
    gpu_free_mem: int
    success_rate: float
    avg_latency_ms: float
    last_updated: float


class RedisCoordinator:
    """
    Distributed coordinator using Redis for multi-instance SOLLOL deployments.

    Solves the multi-application concurrency problem by:
    1. Discovering other SOLLOL instances
    2. Sharing real-time cluster state
    3. Coordinating routing decisions
    4. Aggregating metrics across instances

    Architecture:
        Instance 1 → Redis ← Instance 2
                       ↓
                  Shared State
                       ↓
                  Ollama Nodes

    Redis Keys:
        sollol:instances - Set of active instance IDs
        sollol:instance:{id}:info - Instance metadata
        sollol:instance:{id}:alive - Heartbeat (TTL: 10s)
        sollol:instance:{id}:node_state - This instance's view of node state
        sollol:node:{host}:active_requests - Active request count
        sollol:routing_lock - Distributed lock for atomic routing
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        heartbeat_interval: float = 600.0,  # 10 minutes
        state_ttl: int = 30
    ):
        """
        Initialize Redis coordinator.

        Args:
            redis_url: Redis connection URL
            heartbeat_interval: Seconds between heartbeats
            state_ttl: TTL for state keys in seconds

        Raises:
            ImportError: If redis library not installed
            ConnectionError: If can't connect to Redis
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis library not installed. "
                "Install with: pip install redis>=5.0.0"
            )

        self.redis_url = redis_url
        self.heartbeat_interval = heartbeat_interval
        self.state_ttl = state_ttl

        # Connect to Redis
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise ConnectionError(f"Redis connection failed: {e}")

        # Register this instance
        self.instance_id = str(uuid.uuid4())
        self.hostname = self._get_hostname()
        self.version = "0.3.7"  # Update when implementing

        self._register_instance()

        # Start heartbeat thread
        self._heartbeat_running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name=f"sollol-heartbeat-{self.instance_id[:8]}"
        )
        self._heartbeat_thread.start()

        logger.info(f"SOLLOL instance {self.instance_id[:8]} registered with Redis")

    def _get_hostname(self) -> str:
        """Get hostname for this instance."""
        import socket
        try:
            return socket.gethostname()
        except Exception:
            return "unknown"

    def _register_instance(self):
        """Register this instance in Redis."""
        info = InstanceInfo(
            instance_id=self.instance_id,
            started_at=time.time(),
            last_heartbeat=time.time(),
            hostname=self.hostname,
            version=self.version
        )

        # Add to instances set
        self.redis_client.sadd("sollol:instances", self.instance_id)

        # Store instance info
        self.redis_client.set(
            f"sollol:instance:{self.instance_id}:info",
            json.dumps(asdict(info))
        )

        # Set heartbeat
        self.redis_client.setex(
            f"sollol:instance:{self.instance_id}:alive",
            10,  # TTL: 10 seconds
            "1"
        )

        logger.debug(f"Instance {self.instance_id[:8]} registered")

    def _heartbeat_loop(self):
        """Periodically update heartbeat."""
        while self._heartbeat_running:
            try:
                # Update heartbeat
                self.redis_client.setex(
                    f"sollol:instance:{self.instance_id}:alive",
                    10,  # TTL: 10 seconds
                    "1"
                )

                # Update last_heartbeat in info
                info_key = f"sollol:instance:{self.instance_id}:info"
                info_json = self.redis_client.get(info_key)
                if info_json:
                    info = json.loads(info_json)
                    info['last_heartbeat'] = time.time()
                    self.redis_client.set(info_key, json.dumps(info))

                # Clean up dead instances
                self._remove_dead_instances()

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

            time.sleep(self.heartbeat_interval)

    def _remove_dead_instances(self):
        """Remove instances that haven't sent heartbeat."""
        all_instances = self.redis_client.smembers("sollol:instances")

        for instance_id in all_instances:
            is_alive = self.redis_client.get(f"sollol:instance:{instance_id}:alive")
            if not is_alive:
                # Instance is dead, clean up
                self.redis_client.srem("sollol:instances", instance_id)
                self.redis_client.delete(f"sollol:instance:{instance_id}:info")
                self.redis_client.delete(f"sollol:instance:{instance_id}:node_state")
                logger.info(f"Removed dead instance {instance_id[:8]}")

    def get_active_instances(self) -> List[InstanceInfo]:
        """Get list of all active SOLLOL instances."""
        instances = []
        all_instance_ids = self.redis_client.smembers("sollol:instances")

        for instance_id in all_instance_ids:
            info_json = self.redis_client.get(f"sollol:instance:{instance_id}:info")
            if info_json:
                info_dict = json.loads(info_json)
                instances.append(InstanceInfo(**info_dict))

        return instances

    def update_node_state(self, host: str, state: Dict[str, Any]):
        """
        Update this instance's view of a node's state.

        Args:
            host: Node host (e.g., "localhost:11434")
            state: Node metrics (cpu_load, gpu_free_mem, etc.)
        """
        # Store this instance's view
        key = f"sollol:instance:{self.instance_id}:node_state"
        current = self.redis_client.hget(key, host) or "{}"
        current_state = json.loads(current)

        # Merge with new state
        current_state.update(state)
        current_state['updated_at'] = time.time()

        self.redis_client.hset(key, host, json.dumps(current_state))
        self.redis_client.expire(key, self.state_ttl)

    def get_aggregated_node_state(self, host: str) -> Optional[NodeState]:
        """
        Get aggregated state for a node across all instances.

        Args:
            host: Node host

        Returns:
            Aggregated NodeState or None if no data
        """
        all_instances = self.redis_client.smembers("sollol:instances")

        if not all_instances:
            return None

        # Collect state from all instances
        states = []
        for instance_id in all_instances:
            state_json = self.redis_client.hget(
                f"sollol:instance:{instance_id}:node_state",
                host
            )
            if state_json:
                states.append(json.loads(state_json))

        if not states:
            return None

        # Aggregate metrics
        total_active_requests = sum(s.get('active_requests', 0) for s in states)
        avg_cpu_load = sum(s.get('cpu_load', 0) for s in states) / len(states)
        min_gpu_mem = min((s.get('gpu_free_mem', 0) for s in states), default=0)
        avg_success_rate = sum(s.get('success_rate', 1.0) for s in states) / len(states)
        avg_latency = sum(s.get('avg_latency_ms', 0) for s in states) / len(states)
        latest_update = max((s.get('updated_at', 0) for s in states), default=0)

        return NodeState(
            host=host,
            active_requests=total_active_requests,
            cpu_load=avg_cpu_load,
            gpu_free_mem=min_gpu_mem,
            success_rate=avg_success_rate,
            avg_latency_ms=avg_latency,
            last_updated=latest_update
        )

    def increment_active_requests(self, host: str) -> int:
        """
        Atomically increment active request count for a node.

        Args:
            host: Node host

        Returns:
            New active request count
        """
        key = f"sollol:node:{host}:active_requests"
        new_count = self.redis_client.incr(key)
        self.redis_client.expire(key, 300)  # TTL: 5 minutes
        return new_count

    def decrement_active_requests(self, host: str) -> int:
        """
        Atomically decrement active request count for a node.

        Args:
            host: Node host

        Returns:
            New active request count
        """
        key = f"sollol:node:{host}:active_requests"
        new_count = self.redis_client.decr(key)

        # Ensure it doesn't go negative
        if new_count < 0:
            self.redis_client.set(key, 0)
            new_count = 0

        return new_count

    def routing_lock(self, timeout: float = 1.0):
        """
        Distributed lock for atomic routing decisions.

        Usage:
            with coordinator.routing_lock():
                # Only one instance can execute this block at a time
                node = select_best_node()

        Args:
            timeout: Lock timeout in seconds
        """
        return self.redis_client.lock(
            "sollol:routing_lock",
            timeout=timeout,
            blocking_timeout=timeout
        )

    def close(self):
        """Clean up and disconnect."""
        self._heartbeat_running = False
        if self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2)

        # Remove this instance from registry
        self.redis_client.srem("sollol:instances", self.instance_id)
        self.redis_client.delete(f"sollol:instance:{self.instance_id}:info")
        self.redis_client.delete(f"sollol:instance:{self.instance_id}:alive")
        self.redis_client.delete(f"sollol:instance:{self.instance_id}:node_state")

        logger.info(f"Instance {self.instance_id[:8]} unregistered")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class LocalCoordinator:
    """
    Fallback coordinator that works without Redis.

    Provides same interface as RedisCoordinator but with no distributed state.
    Used when Redis is unavailable or disabled.
    """

    def __init__(self):
        self.instance_id = str(uuid.uuid4())
        self.local_state: Dict[str, Dict] = {}
        logger.info("Using local coordinator (no distributed state)")

    def get_active_instances(self) -> List[InstanceInfo]:
        """Only this instance."""
        return [InstanceInfo(
            instance_id=self.instance_id,
            started_at=time.time(),
            last_heartbeat=time.time(),
            hostname=self._get_hostname(),
            version="0.3.7"
        )]

    def _get_hostname(self) -> str:
        import socket
        try:
            return socket.gethostname()
        except Exception:
            return "unknown"

    def update_node_state(self, host: str, state: Dict[str, Any]):
        """Store locally."""
        self.local_state[host] = state

    def get_aggregated_node_state(self, host: str) -> Optional[NodeState]:
        """Return local state only."""
        state = self.local_state.get(host)
        if not state:
            return None

        return NodeState(
            host=host,
            active_requests=state.get('active_requests', 0),
            cpu_load=state.get('cpu_load', 0.5),
            gpu_free_mem=state.get('gpu_free_mem', 0),
            success_rate=state.get('success_rate', 1.0),
            avg_latency_ms=state.get('avg_latency_ms', 200.0),
            last_updated=time.time()
        )

    def increment_active_requests(self, host: str) -> int:
        """Increment locally."""
        if host not in self.local_state:
            self.local_state[host] = {'active_requests': 0}
        self.local_state[host]['active_requests'] += 1
        return self.local_state[host]['active_requests']

    def decrement_active_requests(self, host: str) -> int:
        """Decrement locally."""
        if host not in self.local_state:
            return 0
        self.local_state[host]['active_requests'] = max(
            0,
            self.local_state[host]['active_requests'] - 1
        )
        return self.local_state[host]['active_requests']

    def routing_lock(self, timeout: float = 1.0):
        """No-op lock for local mode."""
        from contextlib import nullcontext
        return nullcontext()

    def close(self):
        """No cleanup needed."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_coordinator(
    redis_url: Optional[str] = None,
    enable_distributed: bool = True
) -> Any:
    """
    Factory function to create appropriate coordinator.

    Args:
        redis_url: Redis connection URL (e.g., "redis://localhost:6379/0")
        enable_distributed: Whether to use distributed coordination

    Returns:
        RedisCoordinator if Redis available and enabled, else LocalCoordinator

    Example:
        # Try Redis, fall back to local
        coordinator = create_coordinator(redis_url="redis://localhost:6379")

        # Force local mode
        coordinator = create_coordinator(enable_distributed=False)
    """
    if not enable_distributed:
        logger.info("Distributed coordination disabled, using local mode")
        return LocalCoordinator()

    if not redis_url:
        redis_url = "redis://localhost:6379/0"

    if not REDIS_AVAILABLE:
        logger.warning(
            "Redis library not installed. Install with: pip install redis>=5.0.0. "
            "Falling back to local coordination."
        )
        return LocalCoordinator()

    try:
        return RedisCoordinator(redis_url=redis_url)
    except (ConnectionError, Exception) as e:
        logger.warning(
            f"Failed to connect to Redis at {redis_url}: {e}. "
            f"Falling back to local coordination."
        )
        return LocalCoordinator()
