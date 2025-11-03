"""
SOLLOL Unified Dashboard - Integrates Ray, Dask, and SOLLOL metrics

Combines:
1. SOLLOL metrics (high-level overview)
2. Ray dashboard (task-level details, distributed tracing)
3. Dask dashboard (performance profiling)
4. Prometheus metrics export

Features:
- Embedded Ray dashboard iframe
- Embedded Dask dashboard iframe
- Real-time WebSocket updates (event-driven, not polling)
- Distributed tracing visualization
- Per-task resource monitoring
- Historical analytics (P50/P95/P99)
- Ollama model lifecycle tracking (load/unload/processing)
- llama.cpp coordinator monitoring
- Centralized logging queue
"""

import asyncio
import json
import logging
import os
import queue
import time
from collections import defaultdict, deque
from datetime import datetime
from logging import Handler
from typing import Any, Dict, List, Optional, Set

import ray
import requests
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from flask_sock import Sock
from prometheus_client import REGISTRY, Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger(__name__)

# Centralized logging queue (SynapticLlamas pattern)
log_queue = queue.Queue()


class QueueLogHandler(Handler):
    """Custom logging handler to push logs into a queue for streaming."""

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        log_entry = self.format(record)
        self.log_queue.put(log_entry)


# Prometheus metrics - use try/except to handle duplicate registration gracefully
try:
    request_counter = Counter(
        "sollol_requests_total", "Total requests processed", ["model", "status", "backend"]
    )
except ValueError:
    # Metric already registered with different labels from metrics.py - use a different name
    request_counter = REGISTRY._names_to_collectors.get("sollol_dashboard_requests_total")
    if request_counter is None:
        request_counter = Counter(
            "sollol_dashboard_requests_total",
            "Total dashboard requests processed",
            ["model", "status", "backend"],
        )

try:
    request_duration = Histogram(
        "sollol_request_duration_seconds",
        "Request duration in seconds",
        ["model", "backend"],
        buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    )
except ValueError:
    request_duration = REGISTRY._names_to_collectors.get("sollol_request_duration_seconds")

try:
    active_pools = Gauge("sollol_active_pools", "Number of active Ray pools")
except ValueError:
    active_pools = REGISTRY._names_to_collectors.get("sollol_active_pools")

try:
    gpu_utilization = Gauge(
        "sollol_gpu_utilization", "GPU utilization per node", ["node", "gpu_id"]
    )
except ValueError:
    gpu_utilization = REGISTRY._names_to_collectors.get("sollol_gpu_utilization")


class UnifiedDashboard:
    """Unified dashboard integrating Ray, Dask, and SOLLOL metrics."""

    def __init__(
        self,
        router=None,
        ray_dashboard_port: int = 8265,
        dask_dashboard_port: int = 8787,
        dashboard_port: int = 8080,
        enable_dask: bool = False,
    ):
        """
        Initialize unified dashboard.

        Args:
            router: SOLLOL router (any type)
            ray_dashboard_port: Ray dashboard port
            dask_dashboard_port: Dask dashboard port
            dashboard_port: Unified dashboard port
            enable_dask: Enable Dask distributed client with dashboard
        """
        self.router = router
        self.ray_dashboard_port = ray_dashboard_port
        self.dask_dashboard_port = dask_dashboard_port
        self.dashboard_port = dashboard_port
        self.dask_client = None

        # Request history for analytics
        self.request_history = deque(maxlen=1000)
        self.trace_history = deque(maxlen=100)

        # Initialize Redis client for reading routing streams
        self.redis_client = None
        try:
            import redis

            redis_url = os.getenv("SOLLOL_REDIS_URL", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            logger.info(f"✅ Connected to Redis at {redis_url}")
        except Exception as e:
            logger.warning(f"⚠️  Redis connection failed (routing streams unavailable): {e}")

        # Application registry (track which applications are using SOLLOL)
        self.applications: Dict[str, Dict[str, Any]] = {}
        self.application_timeout = 30  # seconds - consider app inactive if no heartbeat

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
                            f"📊 Dask client initialized - dashboard at {actual_dashboard_url}"
                        )
                    else:
                        logger.info(
                            f"📊 Dask client initialized - dashboard at http://localhost:{dask_dashboard_port}"
                        )
                else:
                    logger.info(
                        f"📊 Dask client initialized - dashboard at http://localhost:{dask_dashboard_port}"
                    )
            except Exception as e:
                logger.warning(f"⚠️  Could not initialize Dask client: {e}")

        # Check Ray dashboard availability
        if ray.is_initialized():
            logger.info(
                f"📊 Ray is initialized - dashboard should be at http://localhost:{ray_dashboard_port}"
            )
        else:
            logger.warning("⚠️  Ray is not initialized - Ray dashboard will not be available")

        # Flask app with WebSocket support
        self.app = Flask(__name__)
        CORS(self.app)
        self.sock = Sock(self.app)
        self._setup_routes()
        self._setup_websocket_routes()

        # Setup centralized logging
        self._setup_logging()

        logger.info(
            f"📊 Unified Dashboard initialized "
            f"(port {dashboard_port}, Ray: {ray_dashboard_port}, Dask: {dask_dashboard_port})"
        )
        logger.info("✨ WebSocket streaming enabled (event-driven monitoring)")

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route("/")
        def index():
            """Serve unified dashboard HTML."""
            return render_template_string(UNIFIED_DASHBOARD_HTML)

        @self.app.route("/api/metrics")
        def metrics():
            """Get SOLLOL metrics."""
            try:
                if self.router and hasattr(self.router, "get_stats"):
                    # get_stats() is not async - call it directly
                    stats = self.router.get_stats()
                else:
                    stats = {}

                # Add request history analytics
                stats["analytics"] = self._calculate_analytics()

                return jsonify(stats)
            except Exception as e:
                logger.error(f"Error getting metrics: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/network/nodes")
        def network_nodes():
            """Get all Ollama nodes in the network (universal)."""
            try:
                # Try to get from router's pool (OllamaPool or HybridRouter with ollama_pool)
                pool = None
                if self.router:
                    # Direct OllamaPool
                    if hasattr(self.router, "nodes") and isinstance(self.router.nodes, list):
                        pool = self.router
                    # HybridRouter with ollama_pool attribute
                    elif hasattr(self.router, "ollama_pool") and self.router.ollama_pool:
                        pool = self.router.ollama_pool

                if pool:
                    nodes = []
                    for node in pool.nodes:
                        # Handle both OllamaNode objects and dict formats
                        if isinstance(node, dict):
                            host = node.get("host", "localhost")
                            port = node.get("port", 11434)
                            url = f"http://{host}:{port}"
                            node_key = f"{host}:{port}"

                            # Get metrics from pool stats if available
                            perf_stats = pool.stats.get("node_performance", {}).get(node_key, {})
                            latency = perf_stats.get("latency_ms", 0)
                            cpu_load = perf_stats.get("cpu_load", 0)
                            gpu_mem = perf_stats.get("gpu_free_mem", 0)
                            available = perf_stats.get("available", True)

                            nodes.append(
                                {
                                    "url": url,
                                    "status": "healthy" if available else "offline",
                                    "latency_ms": int(latency),
                                    "load_percent": int(cpu_load * 100),
                                    "memory_mb": int(gpu_mem),
                                }
                            )
                        else:
                            # OllamaNode object - normalize keys for frontend
                            nodes.append(
                                {
                                    "url": node.url,
                                    "status": "healthy" if node.healthy else "unhealthy",
                                    "latency_ms": node.last_latency_ms or 0,
                                    "load_percent": getattr(node, "load_percent", 0),
                                    "memory_mb": node.free_vram_mb or 0,
                                    "healthy": node.healthy,
                                    "failure_count": node.failure_count,
                                    "models": node.models,
                                    "total_vram_mb": node.total_vram_mb,
                                    "free_vram_mb": node.free_vram_mb,
                                }
                            )
                    return jsonify({"nodes": nodes, "total": len(nodes)})
                else:
                    # Fallback: auto-discover nodes and transform to expected format
                    import requests

                    from sollol.discovery import discover_ollama_nodes

                    discovered = discover_ollama_nodes(
                        discover_all_nodes=True, exclude_localhost=True
                    )
                    nodes = []
                    for node_data in discovered:
                        host = node_data.get("host", "localhost")
                        port = node_data.get("port", 11434)
                        url = f"http://{host}:{port}"

                        # Try to ping the node for health check
                        try:
                            response = requests.get(f"{url}/api/tags", timeout=2)
                            healthy = response.ok
                            latency = int(response.elapsed.total_seconds() * 1000)
                        except:
                            healthy = False
                            latency = 0

                        nodes.append(
                            {
                                "url": url,
                                "status": "healthy" if healthy else "offline",
                                "latency_ms": latency,
                                "load_percent": 0,  # Unknown without pool
                                "memory_mb": 0,  # Unknown without pool
                            }
                        )
                    return jsonify({"nodes": nodes, "total": len(nodes)})
            except Exception as e:
                logger.error(f"Error getting network nodes: {e}")
                return jsonify({"error": str(e), "nodes": []}), 500

        @self.app.route("/api/network/backends")
        def network_backends():
            """Get all RPC backends in the network (universal)."""
            try:
                backends = []

                # Try to get from router
                if self.router and hasattr(self.router, "rpc_backends"):
                    for backend in self.router.rpc_backends:
                        host = backend.get("host")
                        port = backend.get("port", 50052)
                        backends.append(
                            {
                                "url": f"{host}:{port}",
                                "status": "healthy",
                                "latency_ms": 0,
                                "request_count": backend.get("request_count", 0),
                                "failure_count": backend.get("failure_count", 0),
                            }
                        )

                # Try to get from RPC registry
                try:
                    from sollol.rpc_registry import RPCBackendRegistry

                    registry = RPCBackendRegistry()
                    # registry.backends is Dict[str, RPCBackend], need to iterate values()
                    for backend_obj in registry.backends.values():
                        backend_dict = backend_obj.to_dict()
                        host = backend_dict["host"]
                        port = backend_dict["port"]
                        is_healthy = backend_dict["healthy"]
                        metrics = backend_dict.get("metrics", {})
                        backends.append(
                            {
                                "url": f"{host}:{port}",
                                "status": "healthy" if is_healthy else "offline",
                                "latency_ms": metrics.get("avg_latency_ms", 0),
                                "request_count": metrics.get("total_requests", 0),
                                "failure_count": metrics.get("total_failures", 0),
                            }
                        )
                except Exception as e:
                    logger.debug(f"RPC registry lookup failed: {e}")
                    pass

                # If no backends found, try auto-discovery
                if not backends:
                    try:
                        from sollol.rpc_discovery import auto_discover_rpc_backends

                        discovered = auto_discover_rpc_backends()
                        for backend in discovered:
                            host = backend.get("host")
                            port = backend.get("port", 50052)
                            backends.append(
                                {
                                    "url": f"{host}:{port}",
                                    "status": "healthy",
                                    "latency_ms": 0,
                                    "request_count": 0,
                                    "failure_count": 0,
                                }
                            )
                        logger.info(f"Auto-discovered {len(discovered)} RPC backends")
                    except Exception as e:
                        logger.debug(f"RPC auto-discovery failed: {e}")

                # If still no backends, try Redis metadata (same source as /api/dashboard)
                if not backends:
                    logger.debug(
                        f"No backends found, trying Redis metadata. has redis_client: {self.redis_client is not None}"
                    )
                    if self.redis_client:
                        try:
                            metadata_json = self.redis_client.get("sollol:router:metadata")
                            logger.debug(f"Redis metadata exists: {metadata_json is not None}")
                            if metadata_json:
                                import json

                                metadata = json.loads(metadata_json)
                                rpc_backends_from_redis = metadata.get("rpc_backends", [])
                                logger.debug(
                                    f"RPC backends in metadata: {len(rpc_backends_from_redis)}"
                                )
                                for backend in rpc_backends_from_redis:
                                    host = backend.get("host")
                                    port = backend.get("port", 50052)
                                    backends.append(
                                        {
                                            "url": f"{host}:{port}",
                                            "status": "healthy",
                                            "latency_ms": 0,
                                            "request_count": 0,
                                            "failure_count": 0,
                                        }
                                    )
                                if rpc_backends_from_redis:
                                    logger.info(
                                        f"✅ Loaded {len(rpc_backends_from_redis)} RPC backends from Redis metadata"
                                    )
                        except Exception as e:
                            logger.error(f"❌ Redis metadata lookup failed: {e}", exc_info=True)
                    else:
                        logger.warning("⚠️  No Redis client available for backend lookup")

                return jsonify({"backends": backends, "total": len(backends)})
            except Exception as e:
                logger.error(f"Error getting RPC backends: {e}")
                return jsonify({"error": str(e), "backends": []}), 500

        @self.app.route("/api/network/health")
        def network_health():
            """Get overall network health (universal)."""
            try:
                health = {
                    "timestamp": time.time(),
                    "status": "unknown",
                    "nodes_total": 0,
                    "nodes_healthy": 0,
                    "backends_total": 0,
                    "backends_active": 0,
                }

                # Get node health
                try:
                    if self.router and hasattr(self.router, "ollama_pool"):
                        pool = self.router.ollama_pool
                        health["nodes_total"] = len(pool.nodes)
                        health["nodes_healthy"] = sum(1 for n in pool.nodes if n.healthy)
                except:
                    pass

                # Get backend health
                try:
                    if self.router and hasattr(self.router, "rpc_backends"):
                        health["backends_total"] = len(self.router.rpc_backends)
                        health["backends_active"] = len(self.router.rpc_backends)  # Assume active
                except:
                    pass

                # Determine overall status
                if health["nodes_healthy"] > 0 or health["backends_active"] > 0:
                    health["status"] = "healthy"
                else:
                    health["status"] = "degraded"

                return jsonify(health)
            except Exception as e:
                logger.error(f"Error getting network health: {e}")
                return jsonify({"error": str(e), "status": "error"}), 500

        @self.app.route("/api/traces")
        def traces():
            """Get distributed traces."""
            return jsonify(list(self.trace_history))

        @self.app.route("/api/dashboard/config")
        def dashboard_config():
            """Get dashboard configuration (actual ports, etc)."""
            return jsonify(
                {
                    "ray_dashboard_port": self.ray_dashboard_port,
                    "dask_dashboard_port": self.dask_dashboard_port,
                    "ray_dashboard_url": f"http://localhost:{self.ray_dashboard_port}",
                    "dask_dashboard_url": f"http://localhost:{self.dask_dashboard_port}",
                }
            )

        @self.app.route("/api/ray/metrics")
        def ray_metrics():
            """Get Ray metrics."""
            try:
                if ray.is_initialized():
                    # Get Ray metrics
                    ray_stats = {
                        "dashboard_url": f"http://localhost:{self.ray_dashboard_port}",
                        "nodes": len(ray.nodes()),
                        "available_resources": ray.available_resources(),
                        "cluster_resources": ray.cluster_resources(),
                        "status": "active",
                    }
                    return jsonify(ray_stats)
                else:
                    return jsonify({"error": "Ray not initialized", "status": "inactive"}), 500
            except Exception as e:
                return jsonify({"error": str(e), "status": "error"}), 500

        @self.app.route("/api/dask/metrics")
        def dask_metrics():
            """Get Dask metrics."""
            try:
                if self.dask_client:
                    # Get Dask metrics
                    dask_stats = {
                        "dashboard_url": f"http://localhost:{self.dask_dashboard_port}",
                        "workers": len(self.dask_client.scheduler_info().get("workers", {})),
                        "status": "active",
                    }
                    return jsonify(dask_stats)
                else:
                    return jsonify({"error": "Dask not initialized", "status": "inactive"}), 500
            except Exception as e:
                return jsonify({"error": str(e), "status": "error"}), 500

        @self.app.route("/api/prometheus")
        def prometheus_metrics():
            """Prometheus metrics export."""
            return generate_latest()

        @self.app.route("/api/trace", methods=["POST"])
        def add_trace():
            """Add distributed trace."""
            trace_data = request.json
            trace_data["timestamp"] = datetime.utcnow().isoformat()
            self.trace_history.append(trace_data)
            return jsonify({"status": "ok"})

        @self.app.route("/api/applications")
        def applications():
            """Get all registered applications (universal)."""
            try:
                # Clean up stale applications first
                self._cleanup_stale_applications()

                # Return active applications
                apps = []
                for app_id, app_info in self.applications.items():
                    uptime = time.time() - app_info["start_time"]
                    last_seen = time.time() - app_info["last_heartbeat"]

                    apps.append(
                        {
                            "app_id": app_id,
                            "name": app_info["name"],
                            "router_type": app_info.get("router_type", "unknown"),
                            "version": app_info.get("version", "unknown"),
                            "start_time": app_info["start_time"],
                            "last_heartbeat": app_info["last_heartbeat"],
                            "uptime_seconds": int(uptime),
                            "last_seen_seconds": int(last_seen),
                            "status": "active" if last_seen < self.application_timeout else "stale",
                            "metadata": app_info.get("metadata", {}),
                        }
                    )

                return jsonify(
                    {
                        "applications": apps,
                        "total": len(apps),
                        "active": sum(1 for a in apps if a["status"] == "active"),
                    }
                )
            except Exception as e:
                logger.error(f"Error getting applications: {e}")
                return jsonify({"error": str(e), "applications": []}), 500

        @self.app.route("/api/applications/register", methods=["POST"])
        def register_application():
            """Register a new application with the dashboard."""
            try:
                data = request.json
                app_id = data.get("app_id")

                if not app_id:
                    return jsonify({"error": "app_id required"}), 400

                # Register or update application
                now = time.time()
                if app_id not in self.applications:
                    self.applications[app_id] = {
                        "app_id": app_id,
                        "name": data.get("name", "unknown"),
                        "router_type": data.get("router_type", "unknown"),
                        "version": data.get("version", "unknown"),
                        "start_time": now,
                        "last_heartbeat": now,
                        "metadata": data.get("metadata", {}),
                    }
                    logger.info(f"📱 Application registered: {data.get('name')} ({app_id})")
                else:
                    # Update existing
                    self.applications[app_id]["last_heartbeat"] = now
                    self.applications[app_id].update(
                        {
                            "name": data.get("name", self.applications[app_id]["name"]),
                            "router_type": data.get(
                                "router_type", self.applications[app_id]["router_type"]
                            ),
                            "version": data.get("version", self.applications[app_id]["version"]),
                            "metadata": data.get(
                                "metadata", self.applications[app_id].get("metadata", {})
                            ),
                        }
                    )

                return jsonify({"status": "ok", "app_id": app_id})
            except Exception as e:
                logger.error(f"Error registering application: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/applications/heartbeat", methods=["POST"])
        def application_heartbeat():
            """Receive heartbeat from application to keep it active."""
            try:
                data = request.json
                app_id = data.get("app_id")

                if not app_id or app_id not in self.applications:
                    return jsonify({"error": "app_id not registered"}), 404

                # Update heartbeat
                self.applications[app_id]["last_heartbeat"] = time.time()

                # Update metadata if provided
                if "metadata" in data:
                    self.applications[app_id]["metadata"].update(data["metadata"])

                return jsonify({"status": "ok"})
            except Exception as e:
                logger.error(f"Error processing heartbeat: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/applications/<app_id>/unregister", methods=["POST"])
        def unregister_application(app_id):
            """Explicitly unregister an application."""
            try:
                if app_id in self.applications:
                    app_name = self.applications[app_id].get("name", app_id)
                    del self.applications[app_id]
                    logger.info(f"📱 Application unregistered: {app_name} ({app_id})")
                    return jsonify({"status": "ok"})
                else:
                    return jsonify({"error": "app_id not found"}), 404
            except Exception as e:
                logger.error(f"Error unregistering application: {e}")
                return jsonify({"error": str(e)}), 500

    def _setup_logging(self):
        """Setup centralized logging queue."""
        # Add queue handler to root logger
        log_handler = QueueLogHandler(log_queue)
        log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        log_handler.setFormatter(log_formatter)

        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)

        logger.info("📝 Centralized logging queue initialized")

    def _setup_websocket_routes(self):
        """Setup WebSocket routes for real-time streaming."""

        @self.sock.route("/ws/logs")
        def ws_logs(ws):
            """WebSocket endpoint for streaming logs from queue and Redis."""
            logger.info("🔌 Log streaming WebSocket connected")
            last_log_id = "$"  # Start from latest entries

            while True:
                try:
                    # Try in-memory log queue first (real-time logs from this dashboard)
                    try:
                        log_entry = log_queue.get(timeout=0.5)
                        ws.send(log_entry)
                        continue
                    except queue.Empty:
                        pass

                    # Fallback to Redis stream for distributed logs
                    if self.redis_client:
                        try:
                            streams = self.redis_client.xread(
                                {"sollol:dashboard:log_stream": last_log_id}, count=5, block=1000
                            )

                            if streams:
                                for stream_name, messages in streams:
                                    for message_id, data in messages:
                                        last_log_id = message_id
                                        if "log" in data:
                                            ws.send(data["log"])
                            else:
                                time.sleep(0.5)
                        except Exception as redis_err:
                            logger.debug(f"Redis log stream error: {redis_err}")
                            time.sleep(0.5)
                    else:
                        time.sleep(0.5)

                except Exception as e:
                    logger.warning(f"Log streaming disconnected: {e}")
                    break

        @self.sock.route("/ws/network/nodes")
        def ws_network_nodes(ws):
            """WebSocket endpoint for streaming node state changes (event-driven)."""
            logger.info("🔌 Network nodes WebSocket connected")
            previous_state = {}

            while True:
                try:
                    # Get current nodes (router-agnostic - uses fallback discovery)
                    if self.router and hasattr(self.router, "ollama_pool"):
                        pool = self.router.ollama_pool
                        nodes = []
                        for node in pool.nodes:
                            node_key = node.url
                            nodes.append(
                                {
                                    "url": node.url,
                                    "healthy": node.healthy,
                                    "latency_ms": node.last_latency_ms,
                                    "failure_count": node.failure_count,
                                    "status": "healthy" if node.healthy else "unhealthy",
                                }
                            )
                    else:
                        # Fallback: auto-discover
                        from sollol.discovery import discover_ollama_nodes

                        discovered = discover_ollama_nodes(
                            discover_all_nodes=True, exclude_localhost=True
                        )
                        nodes = [
                            {"url": f"http://{n['host']}:{n['port']}", "status": "discovered"}
                            for n in discovered
                        ]

                    # Event-driven change detection (SynapticLlamas pattern)
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

                    time.sleep(2)  # Poll every 2 seconds

                except Exception as e:
                    logger.error(f"Network nodes WebSocket error: {e}")
                    break

        @self.sock.route("/ws/network/backends")
        def ws_network_backends(ws):
            """WebSocket endpoint for streaming RPC backend events (event-driven)."""
            logger.info("🔌 RPC backends WebSocket connected")
            previous_backends: Set[str] = set()

            while True:
                try:
                    backends = []

                    # Get from router
                    if self.router and hasattr(self.router, "rpc_backends"):
                        for backend in self.router.rpc_backends:
                            backend_addr = f"{backend.get('host')}:{backend.get('port', 50052)}"
                            backends.append(backend_addr)

                    # Try RPC registry fallback
                    try:
                        from sollol.rpc_registry import RPCBackendRegistry

                        registry = RPCBackendRegistry()
                        # registry.backends is Dict[str, RPCBackend], need to iterate values()
                        for backend_obj in registry.backends.values():
                            backend_addr = f"{backend_obj.host}:{backend_obj.port}"
                            if backend_addr not in backends:
                                backends.append(backend_addr)
                    except Exception as e:
                        logger.debug(f"RPC registry WebSocket lookup failed: {e}")
                        pass

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

                    time.sleep(2)

                except Exception as e:
                    logger.error(f"RPC backends WebSocket error: {e}")
                    break

        @self.sock.route("/ws/network/ollama_activity")
        def ws_ollama_activity(ws):
            """WebSocket endpoint for Observer events from Redis pub/sub channel."""
            logger.info("🔌 Ollama activity WebSocket connected - subscribing to Redis")

            if not self.redis_client:
                logger.error("Redis client not available - observer events unavailable")
                ws.send(
                    json.dumps(
                        {
                            "type": "error",
                            "message": "Redis not configured - observer events unavailable",
                        }
                    )
                )
                return

            # Send immediate connection confirmation
            ws.send(
                json.dumps(
                    {
                        "type": "connected",
                        "timestamp": time.time(),
                        "message": "🔌 Connected to Ollama activity stream",
                    }
                )
            )

            # Subscribe to the observer events channel
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("sollol:dashboard:ollama:activity")

            logger.info("✅ Subscribed to sollol:dashboard:ollama:activity")

            try:
                for message in pubsub.listen():
                    if message["type"] == "message":
                        try:
                            # Parse the JSON event from observer
                            event_data = json.loads(message["data"])

                            # Format message based on event type
                            event_type = event_data.get("event_type", "unknown")
                            backend = event_data.get("backend", "unknown")
                            details = event_data.get("details", {})
                            model = details.get("model", "")

                            if event_type == "ollama_request":
                                operation = details.get("operation", "")
                                formatted_msg = f"→ REQUEST to {backend}: {model} ({operation})"
                            elif event_type == "ollama_response":
                                latency = details.get("latency_ms", 0)
                                status = details.get("status_code", 200)
                                formatted_msg = f"← RESPONSE from {backend}: {model} ({int(latency)}ms, {status})"
                            elif event_type == "ollama_error":
                                error = details.get("error", "unknown error")
                                formatted_msg = f"✗ ERROR from {backend}: {model} - {error}"
                            else:
                                formatted_msg = f"{event_type}: {backend}"

                            # Forward to WebSocket client
                            ws.send(
                                json.dumps(
                                    {
                                        "type": event_type,
                                        "timestamp": event_data.get("timestamp", time.time()),
                                        "event": event_data,
                                        "message": formatted_msg,
                                    }
                                )
                            )
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse observer event: {e}")
                    elif message["type"] == "subscribe":
                        logger.debug(f"Subscribed to channel: {message['channel']}")
            except Exception as e:
                logger.error(f"Ollama activity WebSocket error: {e}")
            finally:
                pubsub.unsubscribe()
                pubsub.close()
                logger.info("🔌 Ollama activity WebSocket disconnected")

        @self.sock.route("/ws/network/rpc_activity")
        def ws_rpc_activity(ws):
            """WebSocket endpoint for llama.cpp RPC backend activity."""
            logger.info("🔌 RPC activity WebSocket connected")

            # Send immediate connection confirmation
            ws.send(
                json.dumps(
                    {
                        "type": "connected",
                        "timestamp": time.time(),
                        "message": "🔌 Connected to RPC activity stream",
                    }
                )
            )

            previous_state = {}
            last_heartbeat = 0

            while True:
                try:
                    # Get RPC backends to monitor
                    backends_to_monitor = []

                    # Try getting from router first (fast)
                    if self.router and hasattr(self.router, "rpc_backends"):
                        backends_to_monitor = [
                            (f"{b['host']}:{b['port']}", f"{b['host']}:{b['port']}")
                            for b in self.router.rpc_backends
                        ]

                    # Try RPC registry (fast - cached backends)
                    if not backends_to_monitor:
                        try:
                            from sollol.rpc_registry import RPCBackendRegistry

                            registry = RPCBackendRegistry()
                            # registry.backends is Dict[str, RPCBackend], need to iterate values()
                            backends_to_monitor = [
                                (f"{b.host}:{b.port}", f"{b.host}:{b.port}")
                                for b in registry.backends.values()
                            ]
                        except Exception as e:
                            logger.debug(f"RPC registry lookup failed: {e}")

                    # Fallback: auto-discover (slow - 5+ seconds)
                    if not backends_to_monitor:
                        try:
                            from sollol.rpc_discovery import auto_discover_rpc_backends

                            discovered = auto_discover_rpc_backends()
                            backends_to_monitor = [
                                (f"{b['host']}:{b['port']}", f"{b['host']}:{b['port']}")
                                for b in discovered
                            ]
                        except Exception as e:
                            logger.debug(f"RPC discovery failed: {e}")

                    # Send discovery message for all backends (immediate feedback)
                    for backend_key, display_key in backends_to_monitor:
                        # Only send discovery message if we haven't seen this backend before
                        if backend_key not in previous_state:
                            ws.send(
                                json.dumps(
                                    {
                                        "type": "backend_discovered",
                                        "timestamp": time.time(),
                                        "backend": display_key,
                                        "message": f"🔍 RPC backend: {display_key}",
                                    }
                                )
                            )

                    any_state_change = False

                    # Monitor each backend
                    for backend_key, display_key in backends_to_monitor:
                        try:
                            host, port = backend_key.split(":")
                            # Try to check if backend is alive (simplified - actual health check would be gRPC)
                            import socket

                            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            sock.settimeout(1)
                            result = sock.connect_ex((host, int(port)))
                            sock.close()

                            is_reachable = result == 0
                            was_reachable = previous_state.get(backend_key, {}).get(
                                "reachable", False
                            )

                            # Detect state changes ONLY
                            if is_reachable and not was_reachable:
                                ws.send(
                                    json.dumps(
                                        {
                                            "type": "backend_online",
                                            "timestamp": time.time(),
                                            "backend": display_key,
                                            "message": f"✅ RPC backend online: {display_key}",
                                        }
                                    )
                                )
                                any_state_change = True
                            elif not is_reachable and was_reachable:
                                ws.send(
                                    json.dumps(
                                        {
                                            "type": "backend_offline",
                                            "timestamp": time.time(),
                                            "backend": display_key,
                                            "message": f"❌ RPC backend offline: {display_key}",
                                        }
                                    )
                                )
                                any_state_change = True

                            previous_state[backend_key] = {"reachable": is_reachable}

                        except Exception as e:
                            # Mark as unreachable
                            was_reachable = previous_state.get(backend_key, {}).get(
                                "reachable", True
                            )
                            if was_reachable:
                                ws.send(
                                    json.dumps(
                                        {
                                            "type": "backend_error",
                                            "timestamp": time.time(),
                                            "backend": display_key,
                                            "message": f"⚠️  RPC backend error: {display_key}",
                                        }
                                    )
                                )
                                any_state_change = True
                            previous_state[backend_key] = {"reachable": False}

                    # Only send heartbeat every 30 seconds if no state changes
                    current_time = time.time()
                    if not any_state_change and (current_time - last_heartbeat >= 30):
                        ws.send(
                            json.dumps(
                                {
                                    "type": "heartbeat",
                                    "timestamp": current_time,
                                    "message": f"✓ Monitoring {len(backends_to_monitor)} RPC backends",
                                }
                            )
                        )
                        last_heartbeat = current_time

                    time.sleep(3)  # Poll every 3 seconds

                except Exception as e:
                    logger.error(f"RPC activity WebSocket error: {e}")
                    break

        @self.sock.route("/ws/applications")
        def ws_applications(ws):
            """WebSocket endpoint for application lifecycle events (event-driven)."""
            logger.info("🔌 Applications WebSocket connected")
            previous_apps: Set[str] = set()

            while True:
                try:
                    # Clean up stale applications
                    self._cleanup_stale_applications()

                    # Get current applications
                    current_apps = set(self.applications.keys())

                    # Detect new applications
                    new_apps = current_apps - previous_apps
                    for app_id in new_apps:
                        app_info = self.applications[app_id]
                        ws.send(
                            json.dumps(
                                {
                                    "type": "app_registered",
                                    "timestamp": time.time(),
                                    "app_id": app_id,
                                    "name": app_info["name"],
                                    "router_type": app_info.get("router_type"),
                                    "message": f"📱 Application started: {app_info['name']} ({app_info.get('router_type')})",
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

                    # Detect status changes (active → stale)
                    now = time.time()
                    for app_id, app_info in self.applications.items():
                        last_seen = now - app_info["last_heartbeat"]
                        is_stale = last_seen > self.application_timeout

                        # Store previous status
                        if not hasattr(ws, "_app_status"):
                            ws._app_status = {}
                        prev_stale = ws._app_status.get(app_id, False)

                        if is_stale and not prev_stale:
                            ws.send(
                                json.dumps(
                                    {
                                        "type": "app_stale",
                                        "timestamp": time.time(),
                                        "app_id": app_id,
                                        "name": app_info["name"],
                                        "message": f"⚠️  Application not responding: {app_info['name']} (last seen {int(last_seen)}s ago)",
                                    }
                                )
                            )

                        ws._app_status[app_id] = is_stale

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
                                        "apps_count": len(self.applications),
                                        "message": f"✓ Monitoring {len(self.applications)} applications",
                                    }
                                )
                            )
                            ws._last_heartbeat = time.time()

                    time.sleep(2)

                except Exception as e:
                    logger.error(f"Applications WebSocket error: {e}")
                    break

        @self.sock.route("/ws/routing_events")
        def ws_routing_events(ws):
            """WebSocket endpoint for streaming routing decisions from Redis."""
            logger.info("🔌 Routing events WebSocket connected")

            if not self.redis_client:
                logger.error("Redis client not available - routing events unavailable")
                ws.send(
                    json.dumps(
                        {
                            "type": "error",
                            "message": "Redis not configured - routing events unavailable",
                        }
                    )
                )
                return

            # Track last seen ID for streaming new events
            last_id = "0-0"

            while True:
                try:
                    # Read new entries from Redis stream (blocking with timeout)
                    streams = self.redis_client.xread(
                        {"sollol:routing_stream": last_id},
                        count=10,
                        block=2000,  # Block for 2 seconds
                    )

                    if streams:
                        for stream_name, messages in streams:
                            for message_id, data in messages:
                                # Update last ID
                                last_id = message_id

                                # Parse event data
                                if "event" in data:
                                    event_data = json.loads(data["event"])

                                    # Send to WebSocket client
                                    ws.send(
                                        json.dumps(
                                            {
                                                "type": "routing_decision",
                                                "timestamp": event_data.get("timestamp"),
                                                "event": event_data,
                                            }
                                        )
                                    )
                    else:
                        # No new events - send heartbeat
                        ws.send(json.dumps({"type": "heartbeat", "timestamp": time.time()}))

                except Exception as e:
                    logger.error(f"Routing events WebSocket error: {e}")
                    break

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

    def _calculate_analytics(self) -> Dict[str, Any]:
        """Calculate P50/P95/P99 latencies from history."""
        if not self.request_history:
            return {
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "total_requests": 0,
                "success_rate": 0,
            }

        latencies = sorted([r["latency_ms"] for r in self.request_history])
        total = len(latencies)

        def percentile(p):
            idx = int(total * p / 100)
            return latencies[min(idx, total - 1)]

        successful = sum(1 for r in self.request_history if r["status"] == "success")

        return {
            "p50_latency_ms": percentile(50),
            "p95_latency_ms": percentile(95),
            "p99_latency_ms": percentile(99),
            "total_requests": total,
            "success_rate": successful / total if total > 0 else 0,
        }

    def record_request(self, model: str, backend: str, latency_ms: float, status: str = "success"):
        """Record request for analytics."""
        self.request_history.append(
            {
                "model": model,
                "backend": backend,
                "latency_ms": latency_ms,
                "status": status,
                "timestamp": time.time(),
            }
        )

        # Update Prometheus metrics
        request_counter.labels(model=model, status=status, backend=backend).inc()
        request_duration.labels(model=model, backend=backend).observe(latency_ms / 1000)

    def run(self, host: str = "0.0.0.0", debug: bool = False, allow_fallback: bool = True):
        """
        Run dashboard server (production-ready with gevent for WebSocket support).

        Args:
            host: Host to bind to
            debug: Enable debug mode
            allow_fallback: If True and port is in use, assume another dashboard is running
        """
        # Check if dashboard is already running on this port
        if allow_fallback:
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(("localhost", self.dashboard_port))
            sock.close()

            if result == 0:
                # Port is in use - another dashboard is likely running
                logger.info(f"📊 Dashboard already running on port {self.dashboard_port}")
                logger.info(
                    f"   Connecting to existing dashboard at http://localhost:{self.dashboard_port}"
                )
                logger.info("✅ Application will use shared dashboard for observability")
                return  # Don't start a new server

        logger.info(f"🚀 Starting Unified Dashboard on http://{host}:{self.dashboard_port}")
        logger.info(f"   Ray Dashboard: http://localhost:{self.ray_dashboard_port}")
        logger.info(f"   Dask Dashboard: http://localhost:{self.dask_dashboard_port}")
        logger.info(f"   Prometheus: http://{host}:{self.dashboard_port}/api/prometheus")

        try:
            # Use Flask's development server for WebSocket support (Flask-Sock compatible)
            # NOTE: Flask-Sock requires Flask's dev server or specific ASGI servers
            # gevent-websocket is NOT compatible with Flask-Sock
            if debug:
                logger.warning("⚠️  Running in DEBUG mode")
                self.app.run(host=host, port=self.dashboard_port, debug=True)
            else:
                logger.info("✅ Using Flask development server (WebSocket support via Flask-Sock)")
                self.app.run(host=host, port=self.dashboard_port, debug=False, threaded=True)
        except OSError as e:
            if "Address already in use" in str(e):
                logger.warning(f"⚠️  Port {self.dashboard_port} already in use")
                logger.info(
                    "✅ Assuming another SOLLOL dashboard is running - using shared instance"
                )
            else:
                raise


# Unified Dashboard HTML Template
UNIFIED_DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>SOLLOL Unified Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .header h1 {
            font-size: 1.8rem;
            font-weight: 600;
        }
        .header .subtitle {
            color: rgba(255,255,255,0.9);
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        .container {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            grid-template-rows: auto auto auto auto auto;
            gap: 1rem;
            padding: 1rem;
            max-height: calc(100vh - 120px);
            overflow-y: auto;
            overflow-x: hidden;
        }
        .metrics-bar {
            grid-column: 1 / -1;
            display: flex;
            gap: 1rem;
            background: #1e293b;
            padding: 1.5rem;
            border-radius: 0.5rem;
        }
        .metric-card {
            flex: 1;
            background: #334155;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #667eea;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 600;
            color: #a78bfa;
        }
        .metric-label {
            color: #94a3b8;
            font-size: 0.85rem;
            margin-top: 0.25rem;
        }
        .dashboard-panel {
            background: #1e293b;
            border-radius: 0.5rem;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            min-height: 250px;
            max-height: 400px;
        }
        .dashboard-panel.full-height {
            min-height: 500px;
            max-height: 500px;
            height: 500px;
        }
        .panel-header {
            background: #334155;
            padding: 0.75rem 1rem;
            font-weight: 600;
            border-bottom: 2px solid #667eea;
        }
        .panel-content {
            flex: 1;
            overflow: auto;
            min-height: 200px;
        }
        .full-height .panel-content {
            height: 450px;
            min-height: 450px;
        }
        iframe {
            width: 100%;
            height: 100%;
            border: none;
        }
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }
        .status-active { background: #10b981; }
        .status-inactive { background: #ef4444; }
        .node-card, .backend-card {
            background: #334155;
            border-radius: 0.4rem;
            padding: 0.75rem;
            margin: 0.5rem;
            border-left: 3px solid #10b981;
            min-height: 60px;
        }
        .node-card.offline, .backend-card.offline {
            border-left-color: #ef4444;
            opacity: 0.6;
        }
        .node-url, .backend-url {
            font-weight: 600;
            color: #a78bfa;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
            word-break: break-all;
        }
        .node-stats, .backend-stats {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            font-size: 0.75rem;
            color: #94a3b8;
        }
        .stat-badge {
            background: #1e293b;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            white-space: nowrap;
        }
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.7rem;
            font-weight: 600;
            white-space: nowrap;
        }
        .status-healthy { background: #10b981; color: #fff; }
        .status-offline { background: #ef4444; color: #fff; }
        .compact-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8rem;
        }
        .compact-table th {
            background: #334155;
            padding: 0.5rem;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #667eea;
        }
        .compact-table td {
            padding: 0.5rem;
            border-bottom: 1px solid #334155;
        }
        .compact-table tr:hover {
            background: #334155;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 SOLLOL Unified Dashboard</h1>
        <div class="subtitle">
            Real-time monitoring • Distributed tracing • Performance analytics
        </div>
    </div>

    <div class="container">
        <!-- Metrics Bar -->
        <div class="metrics-bar" id="metrics-bar">
            <div class="metric-card">
                <div class="metric-value" id="p50-latency">--</div>
                <div class="metric-label">P50 Latency (ms)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="p95-latency">--</div>
                <div class="metric-label">P95 Latency (ms)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="p99-latency">--</div>
                <div class="metric-label">P99 Latency (ms)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="success-rate">--</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="total-pools">--</div>
                <div class="metric-label">Active Pools</div>
            </div>
        </div>

        <!-- Network Nodes (Ollama) -->
        <div class="dashboard-panel" style="grid-column: span 1;">
            <div class="panel-header">
                🖥️ Network Nodes (Ollama)
            </div>
            <div class="panel-content">
                <div id="network-nodes" style="padding: 0.5rem; overflow-y: auto; height: 100%;">
                    <div style="color: #94a3b8; text-align: center; padding: 2rem;">
                        Loading nodes...
                    </div>
                </div>
            </div>
        </div>

        <!-- RPC Backends (llama.cpp) -->
        <div class="dashboard-panel" style="grid-column: span 1;">
            <div class="panel-header">
                🔗 RPC Backends (llama.cpp)
            </div>
            <div class="panel-content">
                <div id="rpc-backends" style="padding: 0.5rem; overflow-y: auto; height: 100%;">
                    <div style="color: #94a3b8; text-align: center; padding: 2rem;">
                        Loading backends...
                    </div>
                </div>
            </div>
        </div>

        <!-- SOLLOL Metrics -->
        <div class="dashboard-panel">
            <div class="panel-header">
                <span class="status-indicator status-active"></span>
                SOLLOL Metrics
            </div>
            <div class="panel-content" id="sollol-metrics">
                <pre id="metrics-json" style="padding: 1rem; overflow: auto; height: 100%; color: #a78bfa;"></pre>
            </div>
        </div>

        <!-- Distributed Traces -->
        <div class="dashboard-panel">
            <div class="panel-header">
                <span class="status-indicator status-active"></span>
                Distributed Traces
            </div>
            <div class="panel-content">
                <pre id="traces-json" style="padding: 1rem; overflow: auto; height: 100%; color: #10b981;"></pre>
            </div>
        </div>

        <!-- Applications -->
        <div class="dashboard-panel">
            <div class="panel-header">
                <span class="status-indicator status-active"></span>
                Applications
            </div>
            <div class="panel-content">
                <div id="applications-list" style="padding: 1rem; overflow: auto; height: 100%;">
                    <div style="color: #94a3b8; text-align: center; padding: 2rem;">
                        No applications registered yet
                    </div>
                </div>
            </div>
        </div>

        <!-- Ollama Activity Logs -->
        <div class="dashboard-panel full-height">
            <div class="panel-header">
                <span class="status-indicator status-active"></span>
                🖥️ Ollama Activity Logs
            </div>
            <div class="panel-content">
                <div id="ollama-activity" style="padding: 1rem; overflow: auto; height: 100%; font-family: 'Courier New', monospace; font-size: 0.85rem; line-height: 1.4; background: #0f172a;">
                    <div style="color: #94a3b8; text-align: center; padding: 2rem;">
                        Connecting to Ollama activity stream...
                    </div>
                </div>
            </div>
        </div>

        <!-- llama.cpp Activity Logs -->
        <div class="dashboard-panel full-height">
            <div class="panel-header">
                <span class="status-indicator status-active"></span>
                🔗 llama.cpp Activity Logs
            </div>
            <div class="panel-content">
                <div id="rpc-activity" style="padding: 1rem; overflow: auto; height: 100%; font-family: 'Courier New', monospace; font-size: 0.85rem; line-height: 1.4; background: #0f172a;">
                    <div style="color: #94a3b8; text-align: center; padding: 2rem;">
                        Connecting to llama.cpp activity stream...
                    </div>
                </div>
            </div>
        </div>

        <!-- SOLLOL Routing Events -->
        <div class="dashboard-panel full-height">
            <div class="panel-header">
                <span class="status-indicator status-active"></span>
                🎯 SOLLOL Routing Decisions
            </div>
            <div class="panel-content">
                <div id="routing-events" style="padding: 1rem; overflow-y: auto; min-height: 300px; max-height: 600px; font-family: 'Courier New', monospace; background: #0f172a; display: block; visibility: visible;">
                    <div style="color: #10b981; text-align: center; padding: 2rem; font-size: 1rem;">
                        ⟳ Connecting to SOLLOL routing stream...
                    </div>
                </div>
            </div>
        </div>

        <!-- SOLLOL System Logs -->
        <div class="dashboard-panel full-height">
            <div class="panel-header">
                <span class="status-indicator status-active"></span>
                📋 SOLLOL System Logs
            </div>
            <div class="panel-content">
                <div id="sollol-logs" style="padding: 1rem; overflow: auto; height: 100%; font-family: 'Courier New', monospace; font-size: 0.85rem; line-height: 1.4; background: #0f172a;">
                    <div style="color: #94a3b8; text-align: center; padding: 2rem;">
                        Connecting to SOLLOL logs stream...
                    </div>
                </div>
            </div>
        </div>

        <!-- Ray Dashboard -->
        <div class="dashboard-panel full-height">
            <div class="panel-header">
                <span class="status-indicator" id="ray-status"></span>
                Ray Dashboard
            </div>
            <div class="panel-content" id="ray-content">
                <div style="color: #94a3b8; text-align: center; padding: 2rem;">
                    Checking Ray availability...
                </div>
            </div>
        </div>

        <!-- Dask Dashboard -->
        <div class="dashboard-panel full-height">
            <div class="panel-header">
                <span class="status-indicator" id="dask-status"></span>
                Dask Dashboard
            </div>
            <div class="panel-content" id="dask-content">
                <div style="color: #94a3b8; text-align: center; padding: 2rem;">
                    Checking Dask availability...
                </div>
            </div>
        </div>
    </div>

    <script>
        // Fetch dashboard configuration once to get actual ports
        let dashboardConfig = {
            ray_dashboard_url: 'http://localhost:8265',
            dask_dashboard_url: 'http://localhost:8787'
        };

        (async () => {
            try {
                const configRes = await fetch('/api/dashboard/config');
                if (configRes.ok) {
                    dashboardConfig = await configRes.json();
                }
            } catch (e) {
                console.warn('Could not fetch dashboard config, using defaults');
            }
        })();

        // Update metrics every 30 seconds
        setInterval(async () => {
            try {
                // SOLLOL metrics
                const metricsRes = await fetch('/api/metrics');
                const metrics = await metricsRes.json();

                // Update analytics
                if (metrics.analytics) {
                    document.getElementById('p50-latency').textContent =
                        metrics.analytics.p50_latency_ms.toFixed(0);
                    document.getElementById('p95-latency').textContent =
                        metrics.analytics.p95_latency_ms.toFixed(0);
                    document.getElementById('p99-latency').textContent =
                        metrics.analytics.p99_latency_ms.toFixed(0);
                    document.getElementById('success-rate').textContent =
                        (metrics.analytics.success_rate * 100).toFixed(1) + '%';
                }

                // Update pool count
                if (metrics.total_pools !== undefined) {
                    document.getElementById('total-pools').textContent = metrics.total_pools;
                }

                // Display full metrics
                document.getElementById('metrics-json').textContent =
                    JSON.stringify(metrics, null, 2);

                // Traces
                const tracesRes = await fetch('/api/traces');
                const traces = await tracesRes.json();
                document.getElementById('traces-json').textContent =
                    JSON.stringify(traces, null, 2);

                // Network Nodes (Ollama) - Use table view for >5 nodes
                const nodesRes = await fetch('/api/network/nodes');
                const nodesData = await nodesRes.json();
                const nodesList = document.getElementById('network-nodes');

                if (nodesData.nodes && nodesData.nodes.length > 0) {
                    const useTable = nodesData.nodes.length > 5;
                    let html = '';

                    if (useTable) {
                        // Compact table view for many nodes
                        html = '<table class="compact-table"><thead><tr>';
                        html += '<th>Host</th><th>Status</th><th>Latency</th><th>Load</th><th>Memory</th>';
                        html += '</tr></thead><tbody>';

                        nodesData.nodes.forEach(node => {
                            const isHealthy = node.status === 'healthy';
                            const statusClass = isHealthy ? 'status-healthy' : 'status-offline';
                            const statusIcon = isHealthy ? '✅' : '❌';

                            html += '<tr>';
                            html += `<td style="color: #a78bfa; font-weight: 600;">${node.url}</td>`;
                            html += `<td><span class="status-badge ${statusClass}">${statusIcon} ${node.status}</span></td>`;
                            html += `<td>${node.latency_ms}ms</td>`;
                            html += `<td>${node.load_percent}%</td>`;
                            html += `<td>${node.memory_mb}MB</td>`;
                            html += '</tr>';
                        });

                        html += '</tbody></table>';
                    } else {
                        // Card view for few nodes
                        nodesData.nodes.forEach(node => {
                            const isHealthy = node.status === 'healthy';
                            const statusClass = isHealthy ? 'status-healthy' : 'status-offline';
                            const cardClass = isHealthy ? 'node-card' : 'node-card offline';

                            html += `<div class="${cardClass}">`;
                            html += `  <div class="node-url">${node.url}</div>`;
                            html += `  <div class="node-stats">`;
                            html += `    <span class="status-badge ${statusClass}">${node.status}</span>`;
                            html += `    <span class="stat-badge">⏱️ ${node.latency_ms}ms</span>`;
                            html += `    <span class="stat-badge">📊 Load: ${node.load_percent}%</span>`;
                            html += `    <span class="stat-badge">💾 ${node.memory_mb}MB</span>`;
                            html += `  </div>`;
                            html += `</div>`;
                        });
                    }

                    nodesList.innerHTML = html;
                } else {
                    nodesList.innerHTML = '<div style="color: #94a3b8; text-align: center; padding: 2rem;">No Ollama nodes discovered</div>';
                }

                // RPC Backends (llama.cpp) - Use table view for >5 backends
                const backendsRes = await fetch('/api/network/backends');
                const backendsData = await backendsRes.json();
                const backendsList = document.getElementById('rpc-backends');

                if (backendsData.backends && backendsData.backends.length > 0) {
                    const useTable = backendsData.backends.length > 5;
                    let html = '';

                    if (useTable) {
                        // Compact table view for many backends
                        html = '<table class="compact-table"><thead><tr>';
                        html += '<th>Backend</th><th>Status</th><th>Latency</th><th>Requests</th><th>Failures</th>';
                        html += '</tr></thead><tbody>';

                        backendsData.backends.forEach(backend => {
                            const isHealthy = backend.status === 'healthy';
                            const statusClass = isHealthy ? 'status-healthy' : 'status-offline';
                            const statusIcon = isHealthy ? '✅' : '❌';

                            html += '<tr>';
                            html += `<td style="color: #a78bfa; font-weight: 600;">${backend.url}</td>`;
                            html += `<td><span class="status-badge ${statusClass}">${statusIcon} ${backend.status}</span></td>`;
                            html += `<td>${backend.latency_ms}ms</td>`;
                            html += `<td>${backend.request_count || 0}</td>`;
                            html += `<td style="color: ${backend.failure_count > 0 ? '#ef4444' : '#94a3b8'};">${backend.failure_count || 0}</td>`;
                            html += '</tr>';
                        });

                        html += '</tbody></table>';
                    } else {
                        // Card view for few backends
                        backendsData.backends.forEach(backend => {
                            const isHealthy = backend.status === 'healthy';
                            const statusClass = isHealthy ? 'status-healthy' : 'status-offline';
                            const cardClass = isHealthy ? 'backend-card' : 'backend-card offline';

                            html += `<div class="${cardClass}">`;
                            html += `  <div class="backend-url">${backend.url}</div>`;
                            html += `  <div class="backend-stats">`;
                            html += `    <span class="status-badge ${statusClass}">${backend.status}</span>`;
                            html += `    <span class="stat-badge">⏱️ ${backend.latency_ms}ms</span>`;
                            html += `    <span class="stat-badge">📨 Requests: ${backend.request_count || 0}</span>`;
                            html += `    <span class="stat-badge">❌ Failures: ${backend.failure_count || 0}</span>`;
                            html += `  </div>`;
                            html += `</div>`;
                        });
                    }

                    backendsList.innerHTML = html;
                } else {
                    backendsList.innerHTML = '<div style="color: #94a3b8; text-align: center; padding: 2rem;">No RPC backends discovered</div>';
                }

                // Applications
                const appsRes = await fetch('/api/applications');
                const appsData = await appsRes.json();
                const appsList = document.getElementById('applications-list');

                if (appsData.applications && appsData.applications.length > 0) {
                    let html = '<table style="width: 100%; color: #e2e8f0; font-size: 0.85rem;">';
                    html += '<thead><tr style="border-bottom: 1px solid #334155; text-align: left;">';
                    html += '<th style="padding: 0.5rem;">Name</th>';
                    html += '<th style="padding: 0.5rem;">Router</th>';
                    html += '<th style="padding: 0.5rem;">Status</th>';
                    html += '<th style="padding: 0.5rem;">Uptime</th>';
                    html += '</tr></thead><tbody>';

                    appsData.applications.forEach(app => {
                        const statusColor = app.status === 'active' ? '#10b981' : '#f59e0b';
                        const statusIcon = app.status === 'active' ? '✅' : '⚠️';
                        const uptimeMin = Math.floor(app.uptime_seconds / 60);
                        const uptimeSec = app.uptime_seconds % 60;

                        html += '<tr style="border-bottom: 1px solid #1e293b;">';
                        html += `<td style="padding: 0.5rem;">${app.name}</td>`;
                        html += `<td style="padding: 0.5rem; color: #a78bfa;">${app.router_type}</td>`;
                        html += `<td style="padding: 0.5rem; color: ${statusColor};">${statusIcon} ${app.status}</td>`;
                        html += `<td style="padding: 0.5rem;">${uptimeMin}m ${uptimeSec}s</td>`;
                        html += '</tr>';
                    });

                    html += '</tbody></table>';
                    appsList.innerHTML = html;
                } else {
                    appsList.innerHTML = '<div style="color: #94a3b8; text-align: center; padding: 2rem;">No applications registered yet</div>';
                }

                // Check Ray status and load iframe if available
                try {
                    const rayRes = await fetch('/api/ray/metrics', {timeout: 1000});
                    const rayContent = document.getElementById('ray-content');

                    if (rayRes.ok) {
                        document.getElementById('ray-status').className = 'status-indicator status-active';
                        if (!rayContent.querySelector('iframe')) {
                            rayContent.innerHTML = `<iframe src="${dashboardConfig.ray_dashboard_url}" style="width:100%;height:100%;border:none;"></iframe>`;
                        }
                    } else {
                        document.getElementById('ray-status').className = 'status-indicator status-inactive';
                        rayContent.innerHTML = '<div style="color: #f59e0b; text-align: center; padding: 2rem; background: #1e293b; border-radius: 0.5rem; margin: 1rem;"><div style="font-size: 1.5rem; margin-bottom: 0.5rem;">⚠️</div><div style="font-weight: 600; margin-bottom: 0.5rem;">Ray not initialized</div><small style="color: #94a3b8;">Use RayHybridRouter or RayAdvancedRouter to enable</small></div>';
                    }
                } catch {
                    document.getElementById('ray-status').className = 'status-indicator status-inactive';
                    document.getElementById('ray-content').innerHTML = '<div style="color: #f59e0b; text-align: center; padding: 2rem; background: #1e293b; border-radius: 0.5rem; margin: 1rem;"><div style="font-size: 1.5rem; margin-bottom: 0.5rem;">⚠️</div><div style="font-weight: 600; margin-bottom: 0.5rem;">Ray not initialized</div><small style="color: #94a3b8;">Use RayHybridRouter or RayAdvancedRouter to enable</small></div>';
                }

                // Check Dask status and load iframe if available
                try {
                    const daskRes = await fetch('/api/dask/metrics', {timeout: 1000});
                    const daskContent = document.getElementById('dask-content');

                    if (daskRes.ok) {
                        document.getElementById('dask-status').className = 'status-indicator status-active';
                        if (!daskContent.querySelector('iframe')) {
                            daskContent.innerHTML = `<iframe src="${dashboardConfig.dask_dashboard_url}" style="width:100%;height:100%;border:none;"></iframe>`;
                        }
                    } else {
                        document.getElementById('dask-status').className = 'status-indicator status-inactive';
                        daskContent.innerHTML = '<div style="color: #f59e0b; text-align: center; padding: 2rem; background: #1e293b; border-radius: 0.5rem; margin: 1rem;"><div style="font-size: 1.5rem; margin-bottom: 0.5rem;">⚠️</div><div style="font-weight: 600; margin-bottom: 0.5rem;">Dask not initialized</div><small style="color: #94a3b8;">Enable with enable_dask=True in UnifiedDashboard</small></div>';
                    }
                } catch {
                    document.getElementById('dask-status').className = 'status-indicator status-inactive';
                    document.getElementById('dask-content').innerHTML = '<div style="color: #f59e0b; text-align: center; padding: 2rem; background: #1e293b; border-radius: 0.5rem; margin: 1rem;"><div style="font-size: 1.5rem; margin-bottom: 0.5rem;">⚠️</div><div style="font-weight: 600; margin-bottom: 0.5rem;">Dask not initialized</div><small style="color: #94a3b8;">Enable with enable_dask=True in UnifiedDashboard</small></div>';
                }

            } catch (error) {
                console.error('Error updating metrics:', error);
            }
        }, 30000);

        // WebSocket connections for activity logs - fixed to prevent reconnection storm
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsHost = window.location.host;
        const WS_BASE = `${wsProtocol}//${wsHost}`;

        const connectWebSocket = (url, onMessage, onOpen, onClose, onError) => {
            const socket = new WebSocket(url);
            if (onMessage) socket.onmessage = onMessage;
            if (onOpen) socket.onopen = onOpen;
            if (onClose) socket.onclose = onClose;
            if (onError) socket.onerror = onError;
            return socket;
        };

        // Track WebSocket objects globally to prevent duplicates
        let ollamaWs = null;
        let rpcWs = null;
        let routingWs = null;
        let sollolLogsWs = null;
        let reconnecting = {ollama: false, rpc: false, routing: false, sollolLogs: false};

        function connectOllamaLogs() {
            console.log('[DEBUG] connectOllamaLogs() called');
            if (ollamaWs && ollamaWs.readyState === WebSocket.OPEN) {
                console.log('[DEBUG] Already connected, returning');
                return;
            }
            if (reconnecting.ollama) {
                console.log('[DEBUG] Already reconnecting, returning');
                return;
            }

            const ollamaActivity = document.getElementById('ollama-activity');
            console.log('[DEBUG] ollamaActivity element:', ollamaActivity);

            if (ollamaWs) {
                try { ollamaWs.close(); } catch(e) {}
            }

            const handleMessage = (event) => {
                console.log('[DEBUG] WebSocket message received:', event.data);
                try {
                    const data = JSON.parse(event.data);
                    console.log('[DEBUG] Parsed data:', data);

                    if (ollamaActivity.querySelector('div[style*="text-align: center"]')) {
                        console.log('[DEBUG] Clearing placeholder');
                        ollamaActivity.innerHTML = '';
                    }

                    const entry = document.createElement('div');
                    entry.style.padding = '0.25rem 0';
                    entry.style.borderBottom = '1px solid #1e293b';
                    entry.style.fontSize = '0.875rem';

                    // Color code based on event type
                    if (data.type === 'connected') {
                        entry.style.color = '#10b981';
                        entry.style.padding = '0.5rem';
                        entry.style.borderBottom = 'none';
                        console.log('[DEBUG] Styling as connected message');
                    } else if (data.type === 'ollama_request') {
                        entry.style.color = '#22d3ee';  // Cyan for requests
                        console.log('[DEBUG] Styling as request');
                    } else if (data.type === 'ollama_response') {
                        entry.style.color = '#10b981';  // Green for responses
                        console.log('[DEBUG] Styling as response');
                    } else if (data.type === 'ollama_error') {
                        entry.style.color = '#ef4444';  // Red for errors
                        console.log('[DEBUG] Styling as error');
                    } else {
                        entry.style.color = '#e2e8f0';  // Default gray
                    }

                    entry.textContent = data.message || event.data;
                    console.log('[DEBUG] Entry textContent:', entry.textContent);
                    console.log('[DEBUG] Appending entry to ollamaActivity');
                    ollamaActivity.appendChild(entry);
                    console.log('[DEBUG] ✅ Entry appended, total children:', ollamaActivity.children.length);
                    ollamaActivity.scrollTop = ollamaActivity.scrollHeight;

                    // Keep last 100 events
                    while (ollamaActivity.children.length > 100) {
                        ollamaActivity.removeChild(ollamaActivity.firstChild);
                    }
                } catch(e) {
                    console.error('[DEBUG] ERROR in handleMessage:', e);
                }
            };

            const handleOpen = () => {
                console.log('[DEBUG] ✅ WebSocket OPEN - connected to Ollama activity stream');
                reconnecting.ollama = false;

                // Clear the "Connecting..." message - show only LIVE events (no historical data)
                const ollamaActivity = document.getElementById('ollama-activity');
                ollamaActivity.innerHTML = '<div style="color: #10b981; padding: 0.5rem; text-align: center;">✓ Connected - waiting for live events...</div>';
            };

            const handleClose = () => {
                console.log('[DEBUG] WebSocket CLOSED');
                if (!reconnecting.ollama) {
                    reconnecting.ollama = true;
                    setTimeout(() => {
                        ollamaActivity.innerHTML = '<div style="color: #f59e0b; padding: 0.5rem;">⟳ Reconnecting...</div>';
                        reconnecting.ollama = false;
                        connectOllamaLogs();
                    }, 5000);
                }
            };

            const handleError = (error) => {
                console.error('[DEBUG] WebSocket ERROR:', error);
                ollamaActivity.innerHTML = '<div style="color: #ef4444; padding: 0.5rem;">✗ Connection error</div>';
            };

            const wsUrl = `${WS_BASE}/ws/network/ollama_activity`;
            console.log('[DEBUG] Creating WebSocket with URL:', wsUrl);
            ollamaWs = new WebSocket(wsUrl);
            ollamaWs.onmessage = handleMessage;
            ollamaWs.onopen = handleOpen;
            ollamaWs.onclose = handleClose;
            ollamaWs.onerror = handleError;
            console.log('[DEBUG] WebSocket object created, state:', ollamaWs.readyState);
        }

        function connectRPC() {
            if (rpcWs && rpcWs.readyState === WebSocket.OPEN) return;
            if (reconnecting.rpc) return;

            const rpcActivity = document.getElementById('rpc-activity');

            if (rpcWs) {
                try { rpcWs.close(); } catch(e) {}
            }

            rpcWs = new WebSocket(`${WS_BASE}/ws/network/rpc_activity`);

            rpcWs.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (rpcActivity.querySelector('div[style*="text-align: center"]')) {
                        rpcActivity.innerHTML = '';
                    }
                    const entry = document.createElement('div');
                    entry.style.padding = '0.25rem 0';
                    entry.style.borderBottom = '1px solid #1e293b';
                    entry.style.fontSize = '0.875rem';
                    if (data.type === 'connected') {
                        entry.style.color = '#10b981';
                        entry.style.padding = '0.5rem';
                        entry.style.borderBottom = 'none';
                    }
                    entry.textContent = data.message || event.data;
                    rpcActivity.appendChild(entry);
                    rpcActivity.scrollTop = rpcActivity.scrollHeight;
                } catch(e) {
                    console.error('Failed to parse WebSocket message:', e);
                }
            };

            rpcWs.onerror = () => {
                rpcActivity.innerHTML = '<div style="color: #ef4444; padding: 0.5rem;">✗ Connection error</div>';
            };

            rpcWs.onclose = () => {
                if (!reconnecting.rpc) {
                    reconnecting.rpc = true;
                    setTimeout(() => {
                        rpcActivity.innerHTML = '<div style="color: #f59e0b; padding: 0.5rem;">⟳ Reconnecting...</div>';
                        reconnecting.rpc = false;
                        connectRPC();
                    }, 5000);
                }
            };
        }

        function connectRouting() {
            if (routingWs && routingWs.readyState === WebSocket.OPEN) return;
            if (reconnecting.routing) return;

            const routingEvents = document.getElementById('routing-events');

            if (routingWs) {
                try { routingWs.close(); } catch(e) {}
            }

            console.log('[ROUTING WS] Connecting to /ws/routing_events...');
            routingWs = new WebSocket(`${WS_BASE}/ws/routing_events`);

            routingWs.onopen = () => {
                console.log('[ROUTING WS] Connection opened, loading historical decisions...');
                reconnecting.routing = false;

                // Load historical routing decisions from API
                fetch('/api/routing/decisions')
                    .then(r => r.json())
                    .then(data => {
                        const routingEvents = document.getElementById('routing-events');
                        if (data.decisions && data.decisions.length > 0) {
                            routingEvents.innerHTML = '';  // Clear "Connecting..." message
                            console.log('[ROUTING WS] Loading', data.decisions.length, 'historical decisions');

                            data.decisions.slice(0, 50).forEach(decision => {
                                const entry = document.createElement('div');
                                entry.style.padding = '0.5rem';
                                entry.style.margin = '0.25rem 0';
                                entry.style.borderLeft = '3px solid #22d3ee';
                                entry.style.background = '#1e293b';
                                entry.style.fontSize = '0.9rem';
                                entry.style.lineHeight = '1.6';

                                // Format based on decision type
                                const backend = decision.selected_backend || 'unknown';
                                const strategy = decision.strategy || 'unknown';
                                const model = decision.model || '';

                                entry.style.color = '#22d3ee';
                                entry.textContent = `🎯 ${strategy.toUpperCase()}: ${backend} for ${model}`;

                                routingEvents.appendChild(entry);
                            });
                            routingEvents.scrollTop = routingEvents.scrollHeight;
                            console.log('[ROUTING WS] ✅ Loaded historical decisions');
                        } else {
                            routingEvents.innerHTML = '<div style="color: #10b981; padding: 0.5rem; text-align: center;">✓ Connected - waiting for routing decisions...</div>';
                        }
                    })
                    .catch(e => {
                        console.error('[ROUTING WS] Failed to load historical decisions:', e);
                        const routingEvents = document.getElementById('routing-events');
                        routingEvents.innerHTML = '<div style="color: #10b981; padding: 0.5rem; text-align: center;">✓ Connected - waiting for routing decisions...</div>';
                    });
            };

            routingWs.onmessage = (event) => {
                console.log('[ROUTING WS] Message received:', event.data.substring(0, 200));
                try {
                    const data = JSON.parse(event.data);
                    console.log('[ROUTING WS] Parsed data:', data.type, data.event_type);

                    // Clear "Connecting..." message if present
                    if (routingEvents.querySelector('div[style*="text-align: center"]')) {
                        routingEvents.innerHTML = '';
                    }

                    // Create entry with normal styling
                    const entry = document.createElement('div');
                    entry.style.padding = '0.5rem';
                    entry.style.margin = '0.25rem 0';
                    entry.style.borderLeft = '3px solid #22d3ee';
                    entry.style.background = '#1e293b';
                    entry.style.fontSize = '0.9rem';
                    entry.style.lineHeight = '1.6';

                    const eventType = data.event_type;
                    if (data.type === 'connected') {
                        entry.style.color = '#10b981';
                        entry.style.borderLeft = '3px solid #10b981';
                    } else if (eventType === 'ROUTE_DECISION') {
                        entry.style.color = '#22d3ee';
                        entry.style.borderLeft = '3px solid #22d3ee';
                    } else if (eventType === 'OLLAMA_NODE_SELECTED') {
                        entry.style.color = '#a78bfa';
                        entry.style.borderLeft = '3px solid #a78bfa';
                    } else if (eventType === 'FALLBACK_TRIGGERED') {
                        entry.style.color = '#f59e0b';
                        entry.style.borderLeft = '3px solid #f59e0b';
                    } else if (eventType === 'COORDINATOR_START') {
                        entry.style.color = '#10b981';
                        entry.style.borderLeft = '3px solid #10b981';
                    } else if (eventType === 'COORDINATOR_STOP') {
                        entry.style.color = '#ef4444';
                        entry.style.borderLeft = '3px solid #ef4444';
                    } else if (eventType === 'CACHE_HIT') {
                        entry.style.color = '#60a5fa';
                        entry.style.borderLeft = '3px solid #60a5fa';
                    } else {
                        entry.style.color = '#e2e8f0';
                        entry.style.borderLeft = '3px solid #64748b';
                    }

                    entry.textContent = data.message || event.data;
                    routingEvents.appendChild(entry);
                    routingEvents.scrollTop = routingEvents.scrollHeight;
                    console.log('[ROUTING WS] ✅ Entry added to DOM, total children:', routingEvents.children.length);

                    // Keep last 100 events
                    while (routingEvents.children.length > 100) {
                        routingEvents.removeChild(routingEvents.firstChild);
                    }
                } catch(e) {
                    console.error('[ROUTING WS] ❌ Failed to parse message:', e, event.data);
                }
            };

            routingWs.onerror = (error) => {
                console.error('[ROUTING WS] Error:', error);
                routingEvents.innerHTML = '<div style="color: #ef4444; padding: 0.5rem;">✗ Connection error</div>';
            };

            routingWs.onclose = (event) => {
                console.log('[ROUTING WS] Connection closed, code:', event.code, 'reason:', event.reason);
                if (!reconnecting.routing) {
                    reconnecting.routing = true;
                    setTimeout(() => {
                        routingEvents.innerHTML = '<div style="color: #f59e0b; padding: 0.5rem;">⟳ Reconnecting...</div>';
                        reconnecting.routing = false;
                        connectRouting();
                    }, 5000);
                }
            };
        }

        function connectSOLLOLLogs() {
            if (sollolLogsWs && sollolLogsWs.readyState === WebSocket.OPEN) return;
            if (reconnecting.sollolLogs) return;

            const sollolLogs = document.getElementById('sollol-logs');

            if (sollolLogsWs) {
                try { sollolLogsWs.close(); } catch(e) {}
            }

            sollolLogsWs = new WebSocket(`${WS_BASE}/ws/logs`);

            sollolLogsWs.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (sollolLogs.querySelector('div[style*="text-align: center"]')) {
                        sollolLogs.innerHTML = '';
                    }
                    const entry = document.createElement('div');
                    entry.style.padding = '0.25rem 0';
                    entry.style.borderBottom = '1px solid #1e293b';
                    entry.style.fontSize = '0.875rem';

                    // Color code by log level
                    const level = data.level || 'INFO';
                    if (level === 'ERROR') {
                        entry.style.color = '#ef4444';
                    } else if (level === 'WARNING') {
                        entry.style.color = '#f59e0b';
                    } else if (level === 'INFO') {
                        entry.style.color = '#10b981';
                    } else {
                        entry.style.color = '#94a3b8';
                    }

                    const message = data.message || event.data;
                    entry.textContent = message;
                    sollolLogs.appendChild(entry);
                    sollolLogs.scrollTop = sollolLogs.scrollHeight;

                    // Keep only last 100 entries
                    while (sollolLogs.children.length > 100) {
                        sollolLogs.removeChild(sollolLogs.firstChild);
                    }
                } catch(e) {
                    console.error('Failed to parse SOLLOL logs WebSocket message:', e);
                }
            };

            sollolLogsWs.onerror = () => {
                sollolLogs.innerHTML = '<div style="color: #ef4444; padding: 0.5rem;">✗ Connection error</div>';
            };

            sollolLogsWs.onclose = () => {
                if (!reconnecting.sollolLogs) {
                    reconnecting.sollolLogs = true;
                    setTimeout(() => {
                        sollolLogs.innerHTML = '<div style="color: #f59e0b; padding: 0.5rem;">⟳ Reconnecting...</div>';
                        reconnecting.sollolLogs = false;
                        connectSOLLOLLogs();
                    }, 5000);
                }
            };
        }

        // Connect all WebSockets on page load
        connectOllamaLogs();
        connectRPC();
        connectRouting();
        connectSOLLOLLogs();
    </script>
</body>
</html>
"""


def run_unified_dashboard(
    router=None,
    ray_dashboard_port: int = 8265,
    dask_dashboard_port: int = 8787,
    dashboard_port: int = 8080,
    host: str = "0.0.0.0",
    enable_dask: bool = True,
):
    """
    Run unified dashboard server.

    Args:
        router: SOLLOL router instance
        ray_dashboard_port: Ray dashboard port
        dask_dashboard_port: Dask dashboard port
        dashboard_port: Unified dashboard port
        host: Host to bind to
        enable_dask: Enable Dask distributed client with dashboard

    Returns:
        dict: Status with 'started' (bool) and 'url' (str) keys
              - started=True: This process started the dashboard
              - started=False: Dashboard was already running

    Note:
        If a dashboard is already running on the specified port, this function
        will return immediately without starting a new dashboard. This prevents
        port conflicts and duplicate dashboards when using persistent dashboard_service.
    """
    import fcntl
    import os
    import socket

    dashboard_url = f"http://localhost:{dashboard_port}"

    # LAYER 1: Check if port is already in use (fastest check)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Try to bind to the port
        sock.bind(("127.0.0.1", dashboard_port))
        sock.close()
        # Port is available, continue to startup logic
    except OSError:
        # Port is in use - dashboard already running
        logger.info(f"ℹ️  Dashboard already running on port {dashboard_port}")
        logger.info(f"   → Using existing dashboard at {dashboard_url}")
        return {"started": False, "url": dashboard_url}

    # LAYER 2: HTTP API check (verify it's actually a SOLLOL dashboard)
    try:
        response = requests.get(f"{dashboard_url}/api/applications", timeout=1)
        if response.status_code == 200:
            logger.info(f"ℹ️  Dashboard already running at {dashboard_url}")
            logger.info("   → Using existing dashboard (skipping embedded dashboard)")
            return {"started": False, "url": dashboard_url}
    except requests.exceptions.RequestException:
        pass  # Dashboard not running, continue

    # LAYER 3: File lock to coordinate startup (prevent race conditions)
    lock_file_path = f"/tmp/sollol_dashboard_{dashboard_port}.lock"
    lock_file = None

    try:
        # Try to acquire exclusive lock (non-blocking)
        lock_file = open(lock_file_path, "w")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        # We got the lock! Final check before starting
        try:
            response = requests.get(f"{dashboard_url}/api/applications", timeout=1)
            if response.status_code == 200:
                logger.info(f"ℹ️  Dashboard started by another process")
                logger.info(f"   → Using existing dashboard at {dashboard_url}")
                return {"started": False, "url": dashboard_url}
        except requests.exceptions.RequestException:
            pass  # Dashboard definitely not running, start it

        # Start dashboard (lock will be held until process exits)
        logger.info(f"🚀 Starting unified dashboard on port {dashboard_port}")
        dashboard = UnifiedDashboard(
            router=router,
            ray_dashboard_port=ray_dashboard_port,
            dask_dashboard_port=dask_dashboard_port,
            dashboard_port=dashboard_port,
            enable_dask=enable_dask,
        )

        # dashboard.run() blocks - lock held until process exits
        dashboard.run(host=host)

        # This line only reached if dashboard.run() exits (shouldn't happen normally)
        return {"started": True, "url": dashboard_url}

    except BlockingIOError:
        # Another process already has the lock - dashboard is starting/running
        logger.info(f"ℹ️  Dashboard startup in progress (another process has lock)")
        logger.info(f"   → Using existing dashboard at {dashboard_url}")
        return {"started": False, "url": dashboard_url}

    finally:
        # Only release lock if we acquired it and dashboard startup failed
        # If dashboard.run() is blocking successfully, this won't execute until process exits
        if lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
            except:
                pass
