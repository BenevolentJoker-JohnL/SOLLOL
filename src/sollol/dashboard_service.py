"""
SOLLOL Standalone Dashboard Service

A proper distributed observability dashboard that runs as an independent service.
Uses Redis pub/sub for distributed log aggregation across Ray/Dask workers.

Architecture:
- Runs as separate process (not daemon thread)
- Subscribes to Redis channels for logs and activity events
- Polls Ollama/RPC backends for health/activity data
- WebSocket endpoints stream from Redis, not in-process queues
- Works across distributed Ray workers, Dask workers, and main process

Usage:
    # As standalone service:
    python -m sollol.dashboard_service --port 8080 --redis-url redis://localhost:6379

    # From code:
    from sollol.dashboard_service import DashboardService
    service = DashboardService(redis_url="redis://localhost:6379", port=8080)
    service.run()
"""

import json
import logging
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis
from flask import Flask, Response, jsonify, render_template_string, request
from flask_cors import CORS
from flask_sock import Sock
from gevent import pywsgi

logger = logging.getLogger(__name__)

# Redis channel names for pub/sub (aligned with existing sollol:* key pattern)
REDIS_LOG_CHANNEL = "sollol:dashboard:logs"
REDIS_OLLAMA_ACTIVITY_CHANNEL = "sollol:dashboard:ollama:activity"
REDIS_RPC_ACTIVITY_CHANNEL = "sollol:dashboard:rpc:activity"
REDIS_METRICS_CHANNEL = "sollol:dashboard:metrics"

# Redis stream keys for persistent log storage
REDIS_LOG_STREAM = "sollol:dashboard:log_stream"
REDIS_ACTIVITY_STREAM = "sollol:dashboard:activity_stream"


class RedisLogPublisher(logging.Handler):
    """
    Lightweight logging handler that publishes to Redis.

    This can be added to any process/thread and will work correctly
    because Redis handles the inter-process communication.

    Uses both pub/sub (for real-time streaming) and streams (for persistence).
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        channel: str = REDIS_LOG_CHANNEL,
        stream_key: str = REDIS_LOG_STREAM,
        use_streams: bool = True,
    ):
        super().__init__()
        self.redis_client = redis_client
        self.channel = channel
        self.stream_key = stream_key
        self.use_streams = use_streams

    def emit(self, record: logging.LogRecord):
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "name": record.name,
                "message": self.format(record),
                "process": record.process,
                "thread": record.thread,
                "hostname": os.getenv("HOSTNAME", "unknown"),
            }
            log_json = json.dumps(log_entry)

            # Publish to pub/sub for real-time streaming
            self.redis_client.publish(self.channel, log_json)

            # Also add to stream for persistence (optional)
            if self.use_streams:
                self.redis_client.xadd(
                    self.stream_key,
                    log_entry,
                    maxlen=10000,  # Keep last 10k logs
                )
        except Exception as e:
            # Don't let logging errors crash the application
            # But try to log to stderr as fallback
            try:
                sys.stderr.write(f"RedisLogPublisher error: {e}\n")
            except Exception:
                pass


class ActivityMonitor:
    """
    Monitors Ollama servers and RPC backends by polling their APIs.
    Publishes activity to Redis channels for dashboard consumption.
    """

    def __init__(self, redis_client: redis.Redis, router_getter):
        self.redis_client = redis_client
        self.router_getter = router_getter  # Callable that returns router instance
        self.running = False
        self.monitor_thread = None

    def start(self):
        """Start the activity monitoring thread."""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="ActivityMonitor"
        )
        self.monitor_thread.start()
        logger.info("🔍 Activity monitoring started")

    def stop(self):
        """Stop the activity monitoring thread."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def _monitor_loop(self):
        """Main monitoring loop that polls backends."""
        while self.running:
            try:
                router = self.router_getter()
                if router:
                    self._check_ollama_activity(router)
                    self._check_rpc_activity(router)
            except Exception as e:
                logger.error(f"Activity monitor error: {e}")

            time.sleep(30)  # Poll every 30 seconds

    def _check_ollama_activity(self, router):
        """Check Ollama server activity and publish to Redis."""
        try:
            nodes = getattr(router, "nodes", [])
            for node in nodes:
                activity = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "node": node.get("url", "unknown"),
                    "status": "active" if node.get("available", False) else "inactive",
                    "requests": node.get("request_count", 0),
                    "avg_latency": node.get("avg_latency", 0),
                }
                self.redis_client.publish(REDIS_OLLAMA_ACTIVITY_CHANNEL, json.dumps(activity))
        except Exception as e:
            logger.debug(f"Ollama activity check error: {e}")

    def _check_rpc_activity(self, router):
        """Check RPC backend activity and publish to Redis."""
        try:
            # For RayHybridRouter
            if hasattr(router, "rpc_backends"):
                backends = router.rpc_backends
                for backend in backends:
                    activity = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "backend": f"{backend.get('host', 'unknown')}:{backend.get('port', 0)}",
                        "status": "active" if backend.get("available", False) else "inactive",
                        "requests": backend.get("request_count", 0),
                    }
                    self.redis_client.publish(REDIS_RPC_ACTIVITY_CHANNEL, json.dumps(activity))
        except Exception as e:
            logger.debug(f"RPC activity check error: {e}")


class DashboardService:
    """
    Standalone SOLLOL Observability Dashboard Service.

    Runs independently and aggregates data from Redis pub/sub channels.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        port: int = 8080,
        router_getter: Optional[callable] = None,
        ray_dashboard_port: int = 8265,
        dask_dashboard_port: int = 8787,
        enable_dask: bool = True,
        enable_ray: bool = True,
    ):
        self.redis_url = redis_url
        self.port = port
        self.router_getter = router_getter or (lambda: None)
        self.ray_dashboard_port = ray_dashboard_port
        self.dask_dashboard_port = dask_dashboard_port
        self.enable_dask = enable_dask
        self.enable_ray = enable_ray
        self.dask_client = None
        self.ray_initialized = False

        # Initialize Dask if requested
        if enable_dask:
            try:
                import os

                import dask
                from dask.distributed import Client, LocalCluster

                # Set environment variables before cluster creation
                os.environ["DASK_DISTRIBUTED__LOGGING__DISTRIBUTED"] = "error"
                os.environ["DASK_LOGGING__DISTRIBUTED"] = "error"

                dask.config.set({"distributed.worker.daemon": False})
                dask.config.set({"distributed.logging.distributed": "error"})
                dask.config.set({"logging.distributed": "error"})
                dask.config.set({"distributed.admin.tick.interval": "500ms"})

                # Try to connect to existing Dask scheduler first (multi-app coordination)
                try:
                    logger.info("🔍 Attempting to connect to existing Dask scheduler...")
                    # Try default scheduler address
                    self.dask_client = Client("tcp://127.0.0.1:8786", timeout=2)
                    logger.info(
                        f"✅ Connected to existing Dask scheduler at {self.dask_client.scheduler.address}"
                    )
                except Exception as e:
                    # No existing scheduler, create local cluster
                    logger.info("🚀 Starting new Dask cluster (no existing scheduler found)")

                    cluster = LocalCluster(
                        n_workers=1,
                        threads_per_worker=2,
                        processes=False,  # Use threads, not separate processes
                        dashboard_address=f":{dask_dashboard_port}",
                        silence_logs=logging.CRITICAL,
                    )

                    logger.info("📊 Dask cluster using threaded workers (shared logging context)")
                    self.dask_client = Client(cluster)

                # Add filter to ALL handlers to block "Task queue depth" warnings
                # This catches warnings from threaded workers at the handler level
                class DaskWarningFilter(logging.Filter):
                    def filter(self, record):
                        return "Task queue depth" not in record.getMessage()

                dask_filter = DaskWarningFilter()

                # Apply filter to root logger AND all its handlers
                logging.root.addFilter(dask_filter)
                for handler in logging.root.handlers:
                    handler.addFilter(dask_filter)

                # Apply to all distributed loggers and their handlers
                for logger_name in [
                    "distributed",
                    "distributed.worker",
                    "distributed.scheduler",
                    "distributed.core",
                    "distributed.comm",
                ]:
                    dist_logger = logging.getLogger(logger_name)
                    dist_logger.addFilter(dask_filter)
                    dist_logger.setLevel(logging.CRITICAL)
                    for handler in dist_logger.handlers:
                        handler.addFilter(dask_filter)

                # Get actual dashboard URL from client (may be on different port if 8787 was taken)
                if hasattr(self.dask_client, "dashboard_link"):
                    actual_dashboard_url = self.dask_client.dashboard_link
                    # Extract port from URL like http://127.0.0.1:45423/status
                    import re

                    port_match = re.search(r":(\d+)", actual_dashboard_url)
                    if port_match:
                        self.dask_dashboard_port = int(port_match.group(1))
                        logger.info(
                            f"📊 Dask dashboard available at http://127.0.0.1:{self.dask_dashboard_port}"
                        )

                logger.info(f"✅ Dask initialized: {self.dask_client}")
            except ImportError:
                logger.warning("⚠️  Dask not installed, dashboard will not show Dask metrics")
                self.enable_dask = False
                self.dask_client = None
            except Exception as e:
                logger.error(f"❌ Failed to initialize Dask: {e}")
                self.enable_dask = False
                self.dask_client = None

        # Initialize Ray if requested
        if enable_ray:
            try:
                import ray

                # Try to connect to existing Ray cluster first
                try:
                    logger.info("🔍 Attempting to connect to existing Ray cluster...")
                    ray.init(address="auto", ignore_reinit_error=True, logging_level=logging.ERROR)
                    logger.info(f"✅ Connected to existing Ray cluster at {ray.get_address_info()['address']}")
                    self.ray_initialized = True
                except Exception as e:
                    # No existing cluster, start new one
                    logger.info("🚀 Starting new Ray cluster (no existing cluster found)")
                    ray.init(
                        num_cpus=2,
                        ignore_reinit_error=True,
                        logging_level=logging.ERROR,
                        dashboard_host="0.0.0.0",
                        dashboard_port=ray_dashboard_port,
                        include_dashboard=True,
                    )
                    logger.info(f"✅ Ray initialized with dashboard at http://127.0.0.1:{ray_dashboard_port}")
                    self.ray_initialized = True
            except ImportError:
                logger.warning("⚠️  Ray not installed, dashboard will not show Ray metrics")
                self.enable_ray = False
                self.ray_initialized = False
            except Exception as e:
                logger.error(f"❌ Failed to initialize Ray: {e}")
                self.enable_ray = False
                self.ray_initialized = False

        # Redis clients (one for pub/sub, one for regular ops)
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.redis_pubsub_client = redis.from_url(redis_url, decode_responses=True)

        # In-memory buffers for WebSocket streaming
        self.log_buffer = deque(maxlen=1000)
        self.ollama_activity_buffer = deque(maxlen=500)
        self.rpc_activity_buffer = deque(maxlen=500)

        # Background threads
        self.pubsub = None
        self.pubsub_thread = None
        self.activity_monitor = ActivityMonitor(self.redis_client, self.router_getter)

        # Application registry (track which applications are using SOLLOL)
        self.applications: Dict[str, Dict[str, Any]] = {}
        self.application_timeout = 30  # seconds - consider app inactive if no heartbeat

        # Flask app
        self.app = Flask(__name__)
        CORS(self.app)
        self.sock = Sock(self.app)
        self._setup_routes()
        self._setup_websockets()

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route("/")
        def index():
            return render_template_string(self._get_dashboard_html())

        @self.app.route("/api/metrics")
        def metrics():
            try:
                # Try to get router metadata from Redis first
                metadata_json = self.redis_client.get("sollol:router:metadata")
                raw_metrics = {}
                if metadata_json:
                    metadata = json.loads(metadata_json)
                    raw_metrics = metadata.get("metrics", {})
                else:
                    # Fallback to router_getter
                    router = self.router_getter()
                    if router:
                        raw_metrics = {
                            "status": "operational",
                            "total_requests": getattr(router, "total_requests", 0),
                            "success_rate": getattr(router, "success_rate", 0),
                            "avg_latency": getattr(router, "avg_latency", 0),
                            "ray_workers": (
                                len(getattr(router, "ray_workers", []))
                                if hasattr(router, "ray")
                                else 0
                            ),
                        }

                # Add analytics if not present (for HTML compatibility)
                if "analytics" not in raw_metrics:
                    raw_metrics["analytics"] = {
                        "p50_latency_ms": 0,
                        "p95_latency_ms": 0,
                        "p99_latency_ms": 0,
                        "success_rate": 1.0,
                    }

                # Add total_pools if not present
                if "total_pools" not in raw_metrics:
                    if "ollama_pool" in raw_metrics and raw_metrics["ollama_pool"] is not None:
                        raw_metrics["total_pools"] = raw_metrics["ollama_pool"].get("nodes", 0)
                    else:
                        raw_metrics["total_pools"] = 0

                return jsonify(raw_metrics)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/ollama_nodes")
        def ollama_nodes():
            try:
                # Try to get router metadata from Redis first
                metadata_json = self.redis_client.get("sollol:router:metadata")
                if metadata_json:
                    metadata = json.loads(metadata_json)
                    return jsonify(metadata.get("nodes", []))

                # Fallback to router_getter
                router = self.router_getter()
                if router and hasattr(router, "nodes"):
                    return jsonify(router.nodes)

                # Last resort: auto-discover Ollama nodes on network
                from sollol.discovery import discover_ollama_nodes

                discovered = discover_ollama_nodes(
                    timeout=0.5, discover_all_nodes=True, exclude_localhost=True
                )
                logger.info(f"Auto-discovered {len(discovered)} Ollama nodes")
                return jsonify(discovered)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/network/nodes")
        def network_nodes():
            """Unified endpoint (compatibility with UnifiedDashboard)."""
            try:
                # Fallback to router_getter with full metrics
                router = self.router_getter()
                if router and hasattr(router, "nodes"):
                    nodes_data = []
                    for n in router.nodes:
                        host = n.get("host", "localhost")
                        port = n.get("port", 11434)
                        url = f"http://{host}:{port}"
                        node_key = f"{host}:{port}"

                        # Get performance metrics if available
                        perf_stats = {}
                        if hasattr(router, "stats"):
                            perf_stats = router.stats.get("node_performance", {}).get(node_key, {})

                        latency = perf_stats.get("latency_ms", 0)
                        cpu_load = perf_stats.get("cpu_load", 0)
                        gpu_mem = perf_stats.get("gpu_free_mem", 0)
                        available = perf_stats.get("available", True)

                        nodes_data.append(
                            {
                                "url": url,
                                "status": "healthy" if available else "offline",
                                "latency_ms": int(latency),
                                "load_percent": int(cpu_load * 100),
                                "memory_mb": int(gpu_mem),
                            }
                        )

                    return jsonify({"nodes": nodes_data, "total": len(nodes_data)})

                # Last resort: auto-discover Ollama nodes with health check + GPU data from Redis
                import requests

                from sollol.discovery import discover_ollama_nodes

                discovered = discover_ollama_nodes(
                    timeout=0.5, discover_all_nodes=True, exclude_localhost=True
                )
                logger.info(f"Auto-discovered {len(discovered)} Ollama nodes")

                nodes_data = []
                for n in discovered:
                    host = n.get("host", "localhost")
                    port = n.get("port", 11434)
                    url = f"http://{host}:{port}"
                    node_key = f"{host}:{port}"

                    # Ping node for health and latency
                    try:
                        response = requests.get(f"{url}/api/tags", timeout=2)
                        healthy = response.ok
                        latency = int(response.elapsed.total_seconds() * 1000)
                    except:
                        healthy = False
                        latency = 0

                    # Try to get GPU data from Redis
                    gpu_free_mb = 0
                    gpu_total_mb = 0
                    gpu_vendor = "unknown"
                    try:
                        gpu_key = f"sollol:gpu:{node_key}"
                        logger.info(
                            f"🔍 Looking up GPU data for node {node_key} with key: {gpu_key}"
                        )

                        if not self.redis_client:
                            logger.warning(f"❌ Redis client not initialized for GPU lookup")
                        else:
                            gpu_data_json = self.redis_client.get(gpu_key)
                            logger.info(
                                f"📊 Redis returned: {gpu_data_json[:100] if gpu_data_json else 'None'}"
                            )

                            if gpu_data_json:
                                gpu_data = json.loads(gpu_data_json)
                                gpus = gpu_data.get("gpus", [])
                                logger.info(f"✅ Found {len(gpus)} GPU(s) for {node_key}")

                                if gpus:
                                    # Sum up all GPUs (usually just 1)
                                    gpu_free_mb = sum(g.get("memory_free_mb", 0) for g in gpus)
                                    gpu_total_mb = sum(g.get("memory_total_mb", 0) for g in gpus)
                                    gpu_vendor = gpu_data.get("vendor", "unknown")
                                    logger.info(
                                        f"💾 GPU Stats: {gpu_free_mb}MB free / {gpu_total_mb}MB total ({gpu_vendor})"
                                    )
                            else:
                                logger.warning(f"⚠️  No GPU data found in Redis for {node_key}")
                    except Exception as e:
                        logger.error(
                            f"❌ Error fetching GPU data for {node_key}: {e}", exc_info=True
                        )

                    nodes_data.append(
                        {
                            "url": url,
                            "status": "healthy" if healthy else "offline",
                            "latency_ms": latency,
                            "load_percent": 0,
                            "memory_mb": gpu_free_mb,
                            "total_vram_mb": gpu_total_mb,
                            "free_vram_mb": gpu_free_mb,
                            "gpu_vendor": gpu_vendor,
                        }
                    )

                return jsonify({"nodes": nodes_data, "total": len(nodes_data)})
            except Exception as e:
                logger.error(f"Error in /api/network/nodes: {e}")
                return jsonify({"error": str(e), "nodes": [], "total": 0}), 500

        @self.app.route("/api/rpc_backends")
        def rpc_backends():
            try:
                # PRIORITY 1: Detect from running llama-server coordinator process (most reliable)
                try:
                    import re
                    import subprocess

                    result = subprocess.run(
                        ["ps", "aux"], capture_output=True, text=True, timeout=2
                    )

                    for line in result.stdout.split("\n"):
                        if "llama-server" in line and "--rpc" in line:
                            # Extract --rpc argument
                            rpc_match = re.search(r"--rpc\s+([^\s]+)", line)
                            if rpc_match:
                                rpc_arg = rpc_match.group(1)
                                # Parse comma-separated backends
                                backends = []
                                for addr in rpc_arg.split(","):
                                    addr = addr.strip()
                                    # Skip dummy backends
                                    if "coordinator:0" in addr or addr == "coordinator:0":
                                        continue
                                    parts = addr.split(":")
                                    if len(parts) == 2:
                                        backends.append({"host": parts[0], "port": int(parts[1])})

                                if backends:
                                    logger.info(
                                        f"✅ Detected {len(backends)} RPC backends from running coordinator"
                                    )
                                    return jsonify(backends)
                except Exception as e:
                    logger.debug(f"Coordinator process detection failed: {e}")

                # PRIORITY 1.5: Read from config file if coordinator not running
                try:
                    from pathlib import Path

                    config_file = Path("/home/joker/SOLLOL/rpc_backends.conf")
                    if config_file.exists():
                        backends = []
                        with open(config_file, "r") as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith("#"):
                                    parts = line.split(":")
                                    if len(parts) == 2:
                                        backends.append({"host": parts[0], "port": int(parts[1])})
                        if backends:
                            logger.info(f"📄 Loaded {len(backends)} RPC backends from config file")
                            return jsonify(backends)
                except Exception as e:
                    logger.debug(f"Config file read failed: {e}")

                # PRIORITY 2: Auto-discovery from Redis registry
                try:
                    from sollol.rpc_discovery import auto_discover_rpc_backends

                    discovered_backends = auto_discover_rpc_backends()
                    logger.info(f"Auto-discovered {len(discovered_backends)} RPC backends")
                    if discovered_backends:
                        return jsonify(discovered_backends)
                except Exception as e:
                    logger.debug(f"Auto-discovery failed: {e}")

                # PRIORITY 3: Fallback to router_getter
                router = self.router_getter()
                if router and hasattr(router, "rpc_backends"):
                    return jsonify(router.rpc_backends)

                # PRIORITY 4: Fallback to Redis metadata (last resort, often stale)
                try:
                    metadata_json = self.redis_client.get("sollol:router:metadata")
                    if metadata_json:
                        import json

                        metadata = json.loads(metadata_json)
                        rpc_backends_from_redis = metadata.get("rpc_backends", [])
                        # Filter out dummy backends
                        real_backends = [
                            b
                            for b in rpc_backends_from_redis
                            if not (b.get("host") == "coordinator" and b.get("port") == 0)
                        ]
                        if real_backends:
                            logger.info(
                                f"✅ Loaded {len(real_backends)} RPC backends from Redis metadata"
                            )
                            return jsonify(real_backends)
                except Exception as e:
                    logger.debug(f"Redis metadata lookup failed: {e}")

                return jsonify([])
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/network/backends")
        def network_backends():
            """Unified endpoint (compatibility with UnifiedDashboard)."""
            try:
                backends = rpc_backends().get_json()
                if isinstance(backends, list):
                    # Enrich backends with health status for dashboard
                    enriched = []
                    for backend in backends:
                        host = backend.get("host", "unknown")
                        port = backend.get("port", 50052)
                        url = f"{host}:{port}"

                        # RPC backends use gRPC, can't health check via TCP socket.
                        # Instead, assume healthy if registered in Redis (coordinator found them)
                        # OR check if coordinator is actively using them
                        is_healthy = True  # If in Redis registry, assume active
                        latency_ms = 0  # gRPC latency not measurable via TCP

                        # Get stats from Redis if available
                        request_count = 0
                        failure_count = 0
                        try:
                            stats_key = f"sollol:rpc:stats:{host}:{port}"
                            stats_json = self.redis_client.get(stats_key)
                            if stats_json:
                                stats = json.loads(stats_json)
                                request_count = stats.get("request_count", 0)
                                failure_count = stats.get("failure_count", 0)
                        except Exception:
                            pass

                        enriched.append(
                            {
                                "host": host,
                                "port": port,
                                "url": url,
                                "status": "healthy",  # Assume healthy if in Redis registry
                                "latency_ms": latency_ms,
                                "request_count": request_count,
                                "failure_count": failure_count,
                            }
                        )
                    return jsonify({"backends": enriched, "total": len(enriched)})
                return jsonify(backends)
            except Exception as e:
                logger.error(f"Error in network_backends: {e}")
                return jsonify({"backends": [], "total": 0, "error": str(e)})

        @self.app.route("/api/dashboard")
        def dashboard():
            """Combined dashboard data endpoint for compatibility with old HTML template."""
            try:
                metadata_json = self.redis_client.get("sollol:router:metadata")
                dashboard_data = {
                    "metrics": {},
                    "ollama_nodes": [],
                    "rpc_backends": [],
                    "status": {
                        "healthy": True,
                        "available_hosts": 0,
                        "total_hosts": 0,
                        "ray_workers": 0,
                        "dask_workers": 0,
                    },
                    "performance": {
                        "avg_latency_ms": 0,
                        "avg_success_rate": 1.0,
                        "total_gpu_memory_mb": 0,
                    },
                    "hosts": [],
                    "alerts": [],
                    "routing": {
                        "patterns_available": [],
                        "task_types_learned": 0,
                    },
                }

                if metadata_json:
                    metadata = json.loads(metadata_json)
                    dashboard_data["metrics"] = metadata.get("metrics", {})
                    dashboard_data["ollama_nodes"] = metadata.get("nodes", [])
                    dashboard_data["rpc_backends"] = metadata.get("rpc_backends", [])

                # Get Ray worker count (try Redis first, then ray.nodes())
                try:
                    import ray

                    ray_info = ray.nodes()
                    ray_workers = sum(1 for node in ray_info if node.get("Alive", False))
                    dashboard_data["status"]["ray_workers"] = ray_workers
                except (ImportError, Exception) as e:
                    logger.debug(f"Could not get Ray worker count: {e}")

                # Get Dask worker count
                if self.enable_dask and self.dask_client:
                    try:
                        scheduler_info = self.dask_client.scheduler_info()
                        dask_workers = len(scheduler_info.get("workers", {}))
                        dashboard_data["status"]["dask_workers"] = dask_workers
                    except Exception as e:
                        logger.debug(f"Could not get Dask worker count: {e}")

                return jsonify(dashboard_data)
            except Exception as e:
                logger.error(f"Error in /api/dashboard: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/dashboard/config")
        def dashboard_config():
            """Dashboard configuration for iframe URLs."""
            return jsonify(
                {
                    "ray_dashboard_url": f"http://localhost:{self.ray_dashboard_port}",
                    "dask_dashboard_url": (
                        f"http://localhost:{self.dask_dashboard_port}" if self.enable_dask else None
                    ),
                }
            )

        @self.app.route("/api/traces")
        def traces():
            """Get distributed traces from Redis."""
            try:
                # Scan for all trace keys
                trace_keys = []
                cursor = 0
                while True:
                    cursor, keys = self.redis_client.scan(cursor, match="sollol:trace:*", count=100)
                    trace_keys.extend(keys)
                    if cursor == 0:
                        break

                # Fetch all traces
                traces_list = []
                for key in trace_keys:
                    trace_json = self.redis_client.get(key)
                    if trace_json:
                        try:
                            trace_data = json.loads(trace_json)
                            traces_list.append(trace_data)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid trace JSON for key {key}")
                            continue

                # Sort by start_time (most recent first)
                traces_list.sort(key=lambda t: t.get("start_time", 0), reverse=True)

                # Limit to most recent 100 traces
                traces_list = traces_list[:100]

                return jsonify({"traces": traces_list, "total": len(traces_list)})
            except Exception as e:
                logger.error(f"Error getting traces: {e}")
                return jsonify({"error": str(e), "traces": [], "total": 0}), 500

        @self.app.route("/api/routing_logs")
        def routing_logs():
            """Get routing decision logs from Redis stream."""
            try:
                # Get recent routing events from Redis stream
                limit = int(request.args.get("limit", 100))
                events = self.redis_client.xrevrange("sollol:routing_stream", "+", "-", count=limit)

                logs = []
                for event_id, event_data in events:
                    # Parse the event JSON
                    event_json = event_data.get(b"event", b"{}")
                    if isinstance(event_json, bytes):
                        event_json = event_json.decode("utf-8")
                    event = json.loads(event_json)
                    logs.append(event)

                return jsonify({"logs": logs, "total": len(logs)})
            except Exception as e:
                logger.error(f"Error getting routing logs: {e}")
                return jsonify({"error": str(e), "logs": [], "total": 0}), 500

        @self.app.route("/api/applications")
        def applications():
            """Get registered applications."""
            try:
                # Clean up stale applications first
                self._cleanup_stale_applications()
                # Return current applications
                apps = []
                for app_id, app_info in self.applications.items():
                    uptime_seconds = int(time.time() - app_info["started_at"])
                    apps.append(
                        {
                            "app_id": app_id,
                            "name": app_info["name"],
                            "router_type": app_info.get("router_type", "unknown"),
                            "status": "active",
                            "last_heartbeat": app_info["last_heartbeat"],
                            "started_at": app_info["started_at"],
                            "uptime_seconds": uptime_seconds,
                        }
                    )
                return jsonify({"applications": apps, "total": len(apps)})
            except Exception as e:
                logger.error(f"Error getting applications: {e}")
                return jsonify({"error": str(e), "applications": [], "total": 0}), 500

        @self.app.route("/api/applications/register", methods=["POST"])
        def register_application():
            """Register a new application."""
            try:
                data = request.get_json()
                app_id = data.get("app_id")
                app_name = data.get("name", "Unknown")
                router_type = data.get("router_type", "unknown")

                if not app_id:
                    return jsonify({"error": "app_id required"}), 400

                now = time.time()
                self.applications[app_id] = {
                    "name": app_name,
                    "router_type": router_type,
                    "started_at": now,
                    "last_heartbeat": now,
                    "metadata": data.get("metadata", {}),
                }

                logger.info(f"📱 Application registered: {app_name} ({app_id})")
                return jsonify({"status": "registered", "app_id": app_id})
            except Exception as e:
                logger.error(f"Error registering application: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/applications/heartbeat", methods=["POST"])
        def application_heartbeat():
            """Receive heartbeat from registered application."""
            try:
                data = request.get_json()
                app_id = data.get("app_id")

                if not app_id:
                    return jsonify({"error": "app_id required"}), 400

                # Auto-register unknown app_ids (handles dashboard restarts gracefully)
                if app_id not in self.applications:
                    app_name = data.get("name", "Unknown Application")
                    self.applications[app_id] = {
                        "name": app_name,
                        "router_type": data.get("router_type", "unknown"),
                        "version": data.get("version", "unknown"),
                        "metadata": data.get("metadata", {}),
                        "started_at": time.time(),
                        "last_heartbeat": time.time(),
                    }
                    logger.info(
                        f"📱 Auto-registered application from heartbeat: {app_name} ({app_id})"
                    )
                else:
                    self.applications[app_id]["last_heartbeat"] = time.time()

                return jsonify({"status": "ok"})
            except Exception as e:
                logger.error(f"Error processing heartbeat: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/applications/unregister", methods=["POST"])
        def unregister_application():
            """Unregister an application."""
            try:
                data = request.get_json()
                app_id = data.get("app_id")

                if app_id and app_id in self.applications:
                    app_name = self.applications[app_id].get("name", app_id)
                    del self.applications[app_id]
                    logger.info(f"📱 Application unregistered: {app_name} ({app_id})")
                    return jsonify({"status": "unregistered"})

                return jsonify({"error": "Unknown app_id"}), 404
            except Exception as e:
                logger.error(f"Error unregistering application: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/ray/metrics")
        def ray_metrics():
            """Get Ray metrics (placeholder)."""
            try:
                # TODO: Get actual Ray metrics
                return jsonify({"status": "ok", "workers": 0})
            except Exception as e:
                logger.error(f"Error getting Ray metrics: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/dask/metrics")
        def dask_metrics():
            """Get Dask metrics from live cluster."""
            try:
                if not self.enable_dask or not self.dask_client:
                    return jsonify(
                        {
                            "enabled": False,
                            "dashboard_url": None,
                            "workers": 0,
                            "status": "disabled",
                        }
                    )

                # Get scheduler info
                scheduler_info = self.dask_client.scheduler_info()
                workers = scheduler_info.get("workers", {})

                # Calculate total cores and memory
                total_cores = sum(w.get("nthreads", 0) for w in workers.values())
                total_memory = sum(w.get("memory_limit", 0) for w in workers.values())

                # Get task info
                tasks = scheduler_info.get("tasks", {})
                processing = len([t for t in tasks.values() if t.get("state") == "processing"])

                return jsonify(
                    {
                        "enabled": True,
                        "dashboard_url": f"http://127.0.0.1:{self.dask_dashboard_port}",
                        "workers": len(workers),
                        "total_cores": total_cores,
                        "total_memory_gb": round(total_memory / (1024**3), 2),
                        "tasks_processing": processing,
                        "tasks_total": len(tasks),
                        "scheduler_address": self.dask_client.scheduler.address,
                        "status": "active",
                    }
                )
            except Exception as e:
                logger.error(f"Error getting Dask metrics: {e}")
                return jsonify({"error": str(e), "enabled": False}), 500

    def _setup_websockets(self):
        """Setup WebSocket endpoints using Flask-Sock."""

        @self.sock.route("/ws/logs")
        def ws_logs(ws):
            """WebSocket endpoint for streaming logs from Redis."""
            # Send buffered logs first
            for log_entry in list(self.log_buffer):
                try:
                    ws.send(log_entry)
                except Exception:
                    return

            # Track last sent index to avoid duplicates
            last_sent_idx = len(self.log_buffer) - 1

            # Stream new logs
            while True:
                try:
                    current_idx = len(self.log_buffer) - 1
                    # Only send if new logs arrived
                    if current_idx > last_sent_idx:
                        # Send all new logs since last_sent_idx
                        for idx in range(last_sent_idx + 1, current_idx + 1):
                            if idx < len(self.log_buffer):
                                ws.send(self.log_buffer[idx])
                        last_sent_idx = current_idx
                    time.sleep(0.1)
                except Exception:
                    break

        @self.sock.route("/ws/network/ollama_activity")
        def ws_ollama_activity(ws):
            """WebSocket endpoint for Ollama activity - subscribes to Redis pub/sub."""
            logger.info("🔌 Ollama activity WebSocket connected")

            # Create separate Redis connection for pub/sub
            import redis as redis_module

            pubsub_client = redis_module.from_url(self.redis_url, decode_responses=True)
            pubsub = pubsub_client.pubsub()

            try:
                # Subscribe to Ollama activity channel
                pubsub.subscribe("sollol:dashboard:ollama:activity")

                # Send initial connected message
                ws.send(
                    json.dumps(
                        {
                            "type": "connected",
                            "timestamp": time.time(),
                            "message": "✓ Connected to Ollama activity stream",
                        }
                    )
                )
                logger.info("✅ Sent Ollama activity WebSocket connected message")

                # Non-blocking message polling loop
                logger.info("🔄 Starting Ollama activity message polling loop")
                while True:
                    message = pubsub.get_message(timeout=0.1)
                    if message and message["type"] == "message":
                        # Forward Redis message to WebSocket
                        activity_data = json.loads(message["data"])
                        logger.debug(f"Received Ollama activity message: {activity_data}")

                        # Format for display
                        event_type = activity_data.get("event_type", "unknown")
                        backend = activity_data.get("backend", "unknown")
                        details = activity_data.get("details", {})

                        # Create display message
                        if event_type == "ollama_request":
                            display_msg = f"→ REQUEST to {backend}: {details.get('model', 'unknown')} ({details.get('operation', 'chat')})"
                        elif event_type == "ollama_response":
                            latency = details.get("latency_ms", 0)
                            # Add seconds conversion for readability
                            latency_str = f"{latency:.0f}ms"
                            if latency >= 1000:
                                latency_str += f" / {latency/1000:.2f}s"
                            display_msg = f"← RESPONSE from {backend}: {details.get('model', 'unknown')} ({latency_str})"
                        elif event_type == "ollama_error":
                            display_msg = f"✗ ERROR on {backend}: {details.get('error', 'unknown')}"
                        else:
                            display_msg = f"{event_type} on {backend}"

                        payload = json.dumps(
                            {
                                "type": event_type,
                                "timestamp": activity_data.get("timestamp", time.time()),
                                "backend": backend,
                                "details": details,
                                "message": display_msg,
                            }
                        )
                        logger.info(f"📤 Sending Ollama activity WebSocket message: {display_msg}")
                        ws.send(payload)

                    time.sleep(0.1)  # Small delay to prevent busy-waiting

            except Exception as e:
                logger.error(f"Ollama activity WebSocket error: {e}")
            finally:
                pubsub.close()
                pubsub_client.close()

        @self.sock.route("/ws/network/rpc_activity")
        def ws_rpc_activity(ws):
            """WebSocket endpoint for RPC activity - subscribes to Redis pub/sub."""
            logger.info("🔌 RPC activity WebSocket connected")

            # Create separate Redis connection for pub/sub
            import redis as redis_module

            pubsub_client = redis_module.from_url(self.redis_url, decode_responses=True)
            pubsub = pubsub_client.pubsub()

            try:
                # Subscribe to BOTH RPC activity events AND raw coordinator logs
                pubsub.subscribe("sollol:dashboard:rpc:activity")
                pubsub.subscribe("sollol:logs:llama_cpp")  # Raw coordinator logs
                logger.info(
                    "📡 Subscribed to sollol:dashboard:rpc:activity and sollol:logs:llama_cpp"
                )

                # Send initial connected message
                ws.send(
                    json.dumps(
                        {
                            "type": "connected",
                            "timestamp": time.time(),
                            "message": "✓ Connected to llama.cpp activity stream",
                        }
                    )
                )

                # Non-blocking message polling loop
                while True:
                    message = pubsub.get_message(timeout=0.1)
                    if message and message["type"] == "message":
                        # Check which channel the message came from
                        channel = message["channel"]

                        if channel == "sollol:logs:llama_cpp":
                            # Raw coordinator log line - send directly
                            log_line = message["data"]
                            ws.send(
                                json.dumps(
                                    {
                                        "type": "coordinator_log",
                                        "timestamp": time.time(),
                                        "backend": "coordinator",
                                        "message": log_line,
                                        "details": {},
                                    }
                                )
                            )
                            continue

                        # Otherwise, it's a structured activity event from sollol:dashboard:rpc:activity
                        activity_data = json.loads(message["data"])

                        # Format for display
                        event_type = activity_data.get("event_type", "unknown")
                        backend = activity_data.get("backend", "unknown")
                        details = activity_data.get("details", {})

                        # Create display message
                        if event_type == "rpc_request":
                            display_msg = f"→ RPC REQUEST to {backend}"
                        elif event_type == "rpc_response":
                            latency = details.get("latency_ms", 0)
                            # Add seconds conversion for readability
                            latency_str = f"{latency:.0f}ms"
                            if latency >= 1000:
                                latency_str += f" / {latency/1000:.2f}s"
                            display_msg = f"← RPC RESPONSE from {backend} ({latency_str})"
                        elif event_type == "rpc_error":
                            display_msg = (
                                f"✗ RPC ERROR on {backend}: {details.get('error', 'unknown')}"
                            )
                        elif event_type == "rpc_backend_connect":
                            # Show RPC backend addresses for heartbeat/connect events
                            rpc_addresses = details.get("rpc_addresses", [])
                            if rpc_addresses:
                                addresses_str = ", ".join(rpc_addresses)
                                display_msg = f"rpc_backend_connect on {backend}: {addresses_str}"
                            else:
                                display_msg = f"rpc_backend_connect on {backend}"
                        else:
                            display_msg = f"{event_type} on {backend}"

                        ws.send(
                            json.dumps(
                                {
                                    "type": event_type,
                                    "timestamp": activity_data.get("timestamp", time.time()),
                                    "backend": backend,
                                    "details": details,
                                    "message": display_msg,
                                }
                            )
                        )

                    time.sleep(0.1)  # Small delay to prevent busy-waiting

            except Exception as e:
                logger.error(f"RPC activity WebSocket error: {e}")
            finally:
                pubsub.close()
                pubsub_client.close()

        @self.sock.route("/ws/traces")
        def ws_traces(ws):
            """WebSocket endpoint for distributed traces - subscribes to Redis pub/sub."""
            logger.info("🔌 Distributed traces WebSocket connected")

            # Create separate Redis connection for pub/sub
            import redis as redis_module

            pubsub_client = redis_module.from_url(self.redis_url, decode_responses=True)
            pubsub = pubsub_client.pubsub()

            try:
                # Subscribe to trace events channel
                pubsub.subscribe("sollol:traces")
                logger.info("📡 Subscribed to sollol:traces")

                # Send initial connected message
                ws.send(
                    json.dumps(
                        {
                            "type": "connected",
                            "timestamp": time.time(),
                            "message": "✓ Connected to distributed traces stream",
                        }
                    )
                )

                # Non-blocking message polling loop
                while True:
                    message = pubsub.get_message(timeout=0.1)
                    if message and message["type"] == "message":
                        # Parse trace event
                        trace_event = json.loads(message["data"])
                        event = trace_event.get("event", "unknown")
                        trace = trace_event.get("trace", {})

                        # Forward to WebSocket clients
                        ws.send(
                            json.dumps({"type": event, "timestamp": time.time(), "trace": trace})
                        )

                    time.sleep(0.1)  # Small delay to prevent busy-waiting

            except Exception as e:
                logger.error(f"Distributed traces WebSocket error: {e}")
            finally:
                pubsub.close()
                pubsub_client.close()

        @self.sock.route("/ws/routing_events")
        def ws_routing_events(ws):
            """WebSocket endpoint for SOLLOL routing decisions - subscribes to Redis pub/sub."""
            logger.info("🔌 Routing events WebSocket connected")

            # Create separate Redis connection for pub/sub
            import redis as redis_module

            pubsub_client = redis_module.from_url(self.redis_url, decode_responses=True)
            pubsub = pubsub_client.pubsub()

            try:
                # Subscribe to routing events channel
                pubsub.subscribe("sollol:routing_events")

                # Send initial connected message
                ws.send(
                    json.dumps(
                        {
                            "type": "connected",
                            "timestamp": time.time(),
                            "message": "✓ Connected to SOLLOL routing stream",
                        }
                    )
                )
                logger.info("✅ Sent routing events WebSocket connected message")

                # Non-blocking message polling loop
                logger.info("🔄 Starting routing events message polling loop")
                while True:
                    message = pubsub.get_message(timeout=0.1)
                    if message and message["type"] == "message":
                        # Forward Redis message to WebSocket
                        routing_event = json.loads(message["data"])
                        logger.debug(f"Received routing event: {routing_event}")

                        # Format for display
                        event_type = routing_event.get("event_type", "unknown")
                        model = routing_event.get("model", "N/A")
                        backend = routing_event.get("backend", "N/A")
                        reason = routing_event.get("reason", "")
                        instance_id = routing_event.get("instance_id", "unknown")

                        # Create display message based on event type
                        if event_type == "ROUTE_DECISION":
                            display_msg = f"🎯 {model} → {backend} | {reason}"
                        elif event_type == "CACHE_HIT":
                            display_msg = f"💾 {model} → {backend} (cached)"
                        elif event_type == "FALLBACK_TRIGGERED":
                            from_backend = routing_event.get("from_backend", "?")
                            display_msg = f"⚠️  {model}: {from_backend} → {backend} | {reason}"
                        elif event_type == "COORDINATOR_START":
                            rpc_count = routing_event.get("rpc_backends", 0)
                            display_msg = (
                                f"🚀 Coordinator started: {model} ({rpc_count} RPC backends)"
                            )
                        elif event_type == "COORDINATOR_STOP":
                            display_msg = f"⏹️  Coordinator stopped: {model}"
                        elif event_type == "OLLAMA_NODE_SELECTED":
                            node_url = routing_event.get("node_url", "unknown")
                            display_msg = f"📡 {model} → {node_url} | {reason}"
                        elif event_type == "rpc_coordinator_route":
                            # RPC coordinator routing - show list of backends
                            rpc_backends = routing_event.get("rpc_backends", [])
                            backends_str = ", ".join(rpc_backends) if rpc_backends else "N/A"
                            display_msg = f"🔀 {model} → RPC sharding ({backends_str})"
                        else:
                            display_msg = f"{event_type}: {model} → {backend}"

                        payload = json.dumps(
                            {
                                "type": "routing_event",
                                "event_type": event_type,
                                "timestamp": routing_event.get("timestamp", time.time()),
                                "instance_id": instance_id,
                                "model": model,
                                "backend": backend,
                                "reason": reason,
                                "message": display_msg,
                                "full_event": routing_event,
                            }
                        )
                        logger.debug(f"📤 Sending routing event WebSocket message: {display_msg}")
                        ws.send(payload)

                    time.sleep(0.1)  # Small delay to prevent busy-waiting

            except Exception as e:
                logger.error(f"Routing events WebSocket error: {e}")
            finally:
                pubsub.close()
                pubsub_client.close()

        # Add missing WebSocket endpoints expected by unified_dashboard HTML
        @self.sock.route("/ws/network/nodes")
        def ws_network_nodes(ws):
            """WebSocket endpoint for network nodes (Ollama) - uses Redis discovery."""
            logger.info("🔌 Network nodes WebSocket connected")
            previous_state = {}

            while True:
                try:
                    # Get nodes from Redis metadata
                    nodes = []
                    try:
                        metadata_json = self.redis_client.get("sollol:router:metadata")
                        if metadata_json:
                            metadata = json.loads(metadata_json)
                            # Extract Ollama nodes from metadata
                            if "ollama_pool" in metadata:
                                for node_data in metadata["ollama_pool"].get("nodes", []):
                                    nodes.append(
                                        {
                                            "url": node_data.get("url", "unknown"),
                                            "status": (
                                                "healthy"
                                                if node_data.get("healthy", False)
                                                else "unhealthy"
                                            ),
                                            "latency_ms": node_data.get("last_latency_ms", 0),
                                            "failure_count": node_data.get("failure_count", 0),
                                        }
                                    )
                    except Exception as e:
                        logger.debug(f"Could not load nodes from Redis: {e}")

                    # Event-driven change detection
                    events = []
                    for node in nodes:
                        node_url = node["url"]
                        current_status = node.get("status", "unknown")
                        previous_status = previous_state.get(node_url, {}).get("status")

                        # Detect status change
                        if previous_status and current_status != previous_status:
                            events.append(
                                {
                                    "type": "status_change",
                                    "timestamp": time.time(),
                                    "node": node_url,
                                    "old_status": previous_status,
                                    "new_status": current_status,
                                    "message": f"Node {node_url}: {previous_status} → {current_status}",
                                }
                            )

                        # Detect new node
                        if node_url not in previous_state:
                            events.append(
                                {
                                    "type": "node_discovered",
                                    "timestamp": time.time(),
                                    "node": node_url,
                                    "message": f"✅ New node discovered: {node_url}",
                                }
                            )

                        previous_state[node_url] = node

                    # Detect removed nodes
                    current_urls = {n["url"] for n in nodes}
                    removed = set(previous_state.keys()) - current_urls
                    for node_url in removed:
                        events.append(
                            {
                                "type": "node_removed",
                                "timestamp": time.time(),
                                "node": node_url,
                                "message": f"❌ Node removed: {node_url}",
                            }
                        )
                        del previous_state[node_url]

                    # Send events
                    for event in events:
                        ws.send(json.dumps(event))

                    # Heartbeat if no events (every 10 seconds)
                    if len(events) == 0:
                        if not hasattr(ws, "_last_heartbeat"):
                            ws._last_heartbeat = 0
                        if time.time() - ws._last_heartbeat >= 10:
                            ws.send(
                                json.dumps(
                                    {
                                        "type": "heartbeat",
                                        "timestamp": time.time(),
                                        "nodes_count": len(nodes),
                                        "message": f"✓ Monitoring {len(nodes)} nodes",
                                    }
                                )
                            )
                            ws._last_heartbeat = time.time()

                    time.sleep(30)  # Poll every 30 seconds

                except Exception as e:
                    logger.error(f"Network nodes WebSocket error: {e}")
                    break

        @self.sock.route("/ws/network/backends")
        def ws_network_backends(ws):
            """WebSocket endpoint for RPC backends (llama.cpp) - uses Redis discovery."""
            logger.info("🔌 RPC backends WebSocket connected")
            previous_backends = set()

            while True:
                try:
                    # Get backends from Redis metadata
                    backends = []
                    try:
                        metadata_json = self.redis_client.get("sollol:router:metadata")
                        if metadata_json:
                            metadata = json.loads(metadata_json)
                            # Extract RPC backends from metadata
                            if "rpc_backends" in metadata:
                                for backend_data in metadata["rpc_backends"]:
                                    backend_addr = f"{backend_data.get('host')}:{backend_data.get('port', 50052)}"
                                    backends.append(backend_addr)
                    except Exception as e:
                        logger.debug(f"Could not load backends from Redis: {e}")

                    current_backends = set(backends)

                    # Detect new backends
                    new_backends = current_backends - previous_backends
                    for backend_addr in new_backends:
                        ws.send(
                            json.dumps(
                                {
                                    "type": "backend_connected",
                                    "timestamp": time.time(),
                                    "backend": backend_addr,
                                    "message": f"🔗 RPC backend connected: {backend_addr}",
                                }
                            )
                        )

                    # Detect removed backends
                    removed_backends = previous_backends - current_backends
                    for backend_addr in removed_backends:
                        ws.send(
                            json.dumps(
                                {
                                    "type": "backend_disconnected",
                                    "timestamp": time.time(),
                                    "backend": backend_addr,
                                    "message": f"🔌 RPC backend disconnected: {backend_addr}",
                                }
                            )
                        )

                    previous_backends = current_backends

                    # Heartbeat if no changes
                    if len(new_backends) == 0 and len(removed_backends) == 0:
                        if not hasattr(ws, "_last_heartbeat"):
                            ws._last_heartbeat = 0
                        if time.time() - ws._last_heartbeat >= 10:
                            ws.send(
                                json.dumps(
                                    {
                                        "type": "heartbeat",
                                        "timestamp": time.time(),
                                        "backends_count": len(backends),
                                        "message": f"✓ Monitoring {len(backends)} RPC backends",
                                    }
                                )
                            )
                            ws._last_heartbeat = time.time()

                    time.sleep(30)

                except Exception as e:
                    logger.error(f"RPC backends WebSocket error: {e}")
                    break

        @self.sock.route("/ws/ollama_activity")
        def ws_ollama_activity_unified(ws):
            """WebSocket endpoint for Ollama request/response activity stream via Redis pub/sub."""
            logger.info("🔌 Ollama activity WebSocket connected")

            # Create a separate Redis connection for pub/sub (can't use same client)
            import redis as redis_module

            pubsub_client = redis_module.from_url(self.redis_url, decode_responses=True)
            pubsub = pubsub_client.pubsub()

            try:
                # Subscribe to Ollama activity channel
                pubsub.subscribe("sollol:dashboard:ollama:activity")
                logger.info("📡 Subscribed to Ollama activity channel")

                # Send initial connected message
                ws.send(
                    json.dumps(
                        {
                            "type": "connected",
                            "timestamp": time.time(),
                            "message": "✓ Connected to Ollama activity stream",
                        }
                    )
                )

                # Listen for messages
                for message in pubsub.listen():
                    if message["type"] == "message":
                        # Forward Redis message to WebSocket
                        activity_data = json.loads(message["data"])

                        # Format for display
                        event_type = activity_data.get("event_type", "unknown")
                        backend = activity_data.get("backend", "unknown")
                        details = activity_data.get("details", {})

                        # Create display message
                        if event_type == "ollama_request":
                            display_msg = f"→ REQUEST to {backend}: {details.get('model', 'unknown')} ({details.get('operation', 'chat')})"
                        elif event_type == "ollama_response":
                            latency = details.get("latency_ms", 0)
                            # Add seconds conversion for readability
                            latency_str = f"{latency:.0f}ms"
                            if latency >= 1000:
                                latency_str += f" / {latency/1000:.2f}s"
                            display_msg = f"← RESPONSE from {backend}: {details.get('model', 'unknown')} ({latency_str})"
                        elif event_type == "ollama_error":
                            display_msg = f"✗ ERROR on {backend}: {details.get('error', 'unknown')}"
                        else:
                            display_msg = f"{event_type} on {backend}"

                        ws.send(
                            json.dumps(
                                {
                                    "type": event_type,
                                    "timestamp": activity_data.get("timestamp", time.time()),
                                    "backend": backend,
                                    "details": details,
                                    "message": display_msg,
                                }
                            )
                        )

            except Exception as e:
                logger.error(f"Ollama activity WebSocket error: {e}")
            finally:
                pubsub.close()
                pubsub_client.close()

        @self.sock.route("/ws/applications")
        def ws_applications(ws):
            """WebSocket endpoint for applications - uses Redis discovery."""
            logger.info("🔌 Applications WebSocket connected")
            previous_apps = set()

            while True:
                try:
                    # Get applications from Redis
                    apps = []
                    try:
                        metadata_json = self.redis_client.get("sollol:router:metadata")
                        if metadata_json:
                            metadata = json.loads(metadata_json)
                            # Extract applications if available
                            if "applications" in metadata:
                                for app_data in metadata["applications"]:
                                    apps.append(app_data.get("app_id", "unknown"))
                    except Exception as e:
                        logger.debug(f"Could not load applications from Redis: {e}")

                    current_apps = set(apps)

                    # Detect new applications
                    new_apps = current_apps - previous_apps
                    for app_id in new_apps:
                        ws.send(
                            json.dumps(
                                {
                                    "type": "app_registered",
                                    "timestamp": time.time(),
                                    "app_id": app_id,
                                    "name": app_id,
                                    "message": f"📱 Application started: {app_id}",
                                }
                            )
                        )

                    # Detect removed applications
                    removed_apps = previous_apps - current_apps
                    for app_id in removed_apps:
                        ws.send(
                            json.dumps(
                                {
                                    "type": "app_unregistered",
                                    "timestamp": time.time(),
                                    "app_id": app_id,
                                    "message": f"📱 Application stopped: {app_id}",
                                }
                            )
                        )

                    previous_apps = current_apps

                    # Heartbeat if no changes
                    if len(new_apps) == 0 and len(removed_apps) == 0:
                        if not hasattr(ws, "_last_heartbeat"):
                            ws._last_heartbeat = 0
                        if time.time() - ws._last_heartbeat >= 10:
                            ws.send(
                                json.dumps(
                                    {
                                        "type": "heartbeat",
                                        "timestamp": time.time(),
                                        "apps_count": len(apps),
                                        "message": f"✓ Monitoring {len(apps)} applications",
                                    }
                                )
                            )
                            ws._last_heartbeat = time.time()

                    time.sleep(30)

                except Exception as e:
                    logger.error(f"Applications WebSocket error: {e}")
                    break

    def _load_historical_logs(self):
        """Load recent logs from Redis Stream into buffer."""
        try:
            # Read last 100 log entries from stream
            entries = self.redis_client.xrevrange(REDIS_LOG_STREAM, count=100)
            for stream_id, log_data in reversed(entries):
                # Reconstruct JSON from Redis hash
                log_json = json.dumps(log_data)
                self.log_buffer.append(log_json)
            logger.info(f"📜 Loaded {len(entries)} historical log entries from Redis")
        except Exception as e:
            logger.warning(f"Could not load historical logs: {e}")

    def _start_pubsub_listener(self):
        """Start Redis pub/sub listener in background thread."""
        # Load historical logs first
        self._load_historical_logs()

        self.pubsub = self.redis_pubsub_client.pubsub()
        self.pubsub.subscribe(
            REDIS_LOG_CHANNEL,
            REDIS_OLLAMA_ACTIVITY_CHANNEL,
            REDIS_RPC_ACTIVITY_CHANNEL,
        )

        def pubsub_loop():
            logger.info("📡 Redis pub/sub listener started")
            for message in self.pubsub.listen():
                if message["type"] != "message":
                    continue

                channel = message["channel"]
                data = message["data"]

                try:
                    if channel == REDIS_LOG_CHANNEL:
                        self.log_buffer.append(data)
                    elif channel == REDIS_OLLAMA_ACTIVITY_CHANNEL:
                        self.ollama_activity_buffer.append(data)
                    elif channel == REDIS_RPC_ACTIVITY_CHANNEL:
                        self.rpc_activity_buffer.append(data)
                except Exception as e:
                    logger.error(f"Error processing pub/sub message: {e}")

        self.pubsub_thread = threading.Thread(
            target=pubsub_loop, daemon=True, name="RedisPubSubListener"
        )
        self.pubsub_thread.start()

    def _cleanup_stale_applications(self):
        """Remove applications that haven't sent heartbeat recently."""
        now = time.time()
        stale_apps = [
            app_id
            for app_id, app_info in self.applications.items()
            if now - app_info["last_heartbeat"]
            > self.application_timeout * 2  # 2x timeout = remove
        ]
        for app_id in stale_apps:
            app_name = self.applications[app_id].get("name", app_id)
            logger.info(f"📱 Removing stale application: {app_name} ({app_id})")
            del self.applications[app_id]

    def run(self, host: str = "0.0.0.0", debug: bool = False):
        """Start the dashboard service."""
        # Start Redis pub/sub listener
        self._start_pubsub_listener()

        # Start activity monitor
        self.activity_monitor.start()

        # Start Flask server with WebSocket support
        logger.info(f"🚀 SOLLOL Dashboard Service starting on {host}:{self.port}")
        print(f"✅ Dashboard available at http://{host}:{self.port}")
        print(f"📊 Features: Real-time logs, Activity monitoring, Ray/Dask dashboards")
        print(f"📡 Using Redis at {self.redis_url}")

        # Flask-Sock handles WebSockets automatically with simple_websocket
        # No need for WebSocketHandler
        self.app.run(host=host, port=self.port, debug=debug)

    def _get_dashboard_html(self) -> str:
        """Return the dashboard HTML template."""
        # Import the existing template from unified_dashboard
        try:
            from .unified_dashboard import UNIFIED_DASHBOARD_HTML

            return UNIFIED_DASHBOARD_HTML
        except ImportError:
            return "<html><body><h1>Dashboard HTML not found</h1></body></html>"


def install_redis_log_publisher(redis_url: str = "redis://localhost:6379"):
    """
    Install Redis log publisher to current process.

    Call this in Ray workers, Dask workers, or main process to send logs to dashboard.

    Usage:
        from sollol.dashboard_service import install_redis_log_publisher
        install_redis_log_publisher()  # Now all logs go to Redis
    """
    try:
        redis_client = redis.from_url(redis_url, decode_responses=True)
        handler = RedisLogPublisher(redis_client)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        logger.info(f"📡 Redis log publisher installed (publishing to {redis_url})")
        return handler
    except Exception as e:
        logger.warning(f"Failed to install Redis log publisher: {e}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SOLLOL Standalone Dashboard Service")
    parser.add_argument("--port", type=int, default=8080, help="Dashboard port")
    parser.add_argument("--redis-url", default="redis://localhost:6379", help="Redis URL")
    parser.add_argument("--ray-dashboard-port", type=int, default=8265)
    parser.add_argument("--dask-dashboard-port", type=int, default=8787)
    parser.add_argument("--no-dask", action="store_true", help="Disable Dask dashboard")
    parser.add_argument("--no-ray", action="store_true", help="Disable Ray cluster auto-start")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    service = DashboardService(
        redis_url=args.redis_url,
        port=args.port,
        ray_dashboard_port=args.ray_dashboard_port,
        dask_dashboard_port=args.dask_dashboard_port,
        enable_dask=not args.no_dask,
        enable_ray=not args.no_ray,
    )

    service.run(debug=args.debug)
