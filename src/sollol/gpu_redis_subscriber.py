"""
GPU Stats Redis Subscriber for SOLLOL
Subscribes to GPU stats published by gpu_reporter.py and updates node_performance.
"""

import json
import logging
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class GPURedisSubscriber:
    """Subscribes to GPU stats from Redis and updates SOLLOL node metadata."""

    def __init__(self, pool, redis_host: str = "localhost", redis_port: int = 6379):
        """
        Initialize GPU stats subscriber.

        Args:
            pool: OllamaPool instance to update
            redis_host: Redis server hostname
            redis_port: Redis server port
        """
        self.pool = pool
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        self._stop_event = threading.Event()
        self._thread = None

    def connect(self):
        """Connect to Redis server."""
        try:
            import redis

            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            self.redis_client.ping()
            logger.info(
                f"✅ GPU subscriber connected to Redis at {self.redis_host}:{self.redis_port}"
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to connect GPU subscriber to Redis: {e}")
            return False

    def update_node_gpu_stats(self, node_id: str, gpu_data: Dict):
        """
        Update node_performance with accurate GPU stats from Redis.

        Args:
            node_id: Node identifier (e.g., "10.9.66.90:11434")
            gpu_data: GPU stats from gpu_reporter
        """
        try:
            with self.pool._lock:
                if node_id not in self.pool.stats["node_performance"]:
                    logger.debug(f"Node {node_id} not in node_performance, skipping GPU update")
                    return

                node_perf = self.pool.stats["node_performance"][node_id]

                # Extract GPU info
                gpus = gpu_data.get("gpus", [])

                if not gpus:
                    # CPU-only node
                    node_perf["gpu_free_mem"] = 0
                    node_perf["gpu_vendor"] = "none"
                    logger.debug(f"Updated {node_id}: CPU-only (no GPU)")
                    return

                # For now, use first GPU (most systems have 1 GPU)
                gpu = gpus[0]

                # Update node_performance with accurate GPU data
                node_perf["gpu_free_mem"] = gpu.get("memory_free_mb", 0)
                node_perf["gpu_total_mem"] = gpu.get("memory_total_mb", 0)
                node_perf["gpu_used_mem"] = gpu.get("memory_used_mb", 0)
                node_perf["gpu_utilization"] = gpu.get("utilization_percent", 0)
                node_perf["gpu_temperature"] = gpu.get("temperature_c", 0)
                node_perf["gpu_vendor"] = gpu_data.get("vendor", "unknown")
                node_perf["gpu_model"] = gpu.get("name", "unknown")

                logger.debug(
                    f"Updated {node_id}: {gpu.get('name')} | "
                    f"VRAM: {gpu.get('memory_free_mb', 0)}MB free / {gpu.get('memory_total_mb', 0)}MB total | "
                    f"Util: {gpu.get('utilization_percent', 0)}%"
                )

        except Exception as e:
            logger.error(f"Failed to update node GPU stats for {node_id}: {e}")

    def poll_gpu_stats(self):
        """Poll Redis for latest GPU stats (uses key-based lookup for efficiency)."""
        try:
            # Get all GPU stat keys
            keys = self.redis_client.keys("sollol:gpu:*")

            for key in keys:
                # Skip the stream key (sollol:gpu:stats) - only process node keys
                if key == "sollol:gpu:stats":
                    continue

                # Extract node_id from key
                node_id = key.replace("sollol:gpu:", "")

                # Get GPU stats
                data = self.redis_client.get(key)
                if data:
                    gpu_stats = json.loads(data)
                    self.update_node_gpu_stats(node_id, gpu_stats)

        except Exception as e:
            # Silently skip errors - GPU monitoring status visible in dashboard
            logger.debug(f"GPU stats polling issue (not critical): {e}")

    def run_loop(self, interval: int = 5):
        """
        Run subscriber loop.

        Args:
            interval: Polling interval in seconds (default: 5)
        """
        logger.info(f"🚀 Starting GPU stats subscriber (poll interval: {interval}s)")

        while not self._stop_event.is_set():
            try:
                self.poll_gpu_stats()
                time.sleep(interval)

            except Exception as e:
                logger.error(f"Error in GPU subscriber loop: {e}")
                time.sleep(interval)

        logger.info("GPU subscriber stopped")

    def start(self, interval: int = 5):
        """
        Start subscriber in background thread.

        Args:
            interval: Polling interval in seconds (default: 5)
        """
        if self._thread and self._thread.is_alive():
            logger.warning("GPU subscriber already running")
            return

        if not self.connect():
            logger.warning("GPU subscriber not started - Redis connection failed")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self.run_loop, args=(interval,), daemon=True, name="GPUStatsSubscriber"
        )
        self._thread.start()
        logger.info("GPU stats subscriber started in background")

    def stop(self):
        """Stop the subscriber thread."""
        logger.info("Stopping GPU subscriber...")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
