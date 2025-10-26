"""
SOLLOL Dashboard Server - Real-time monitoring for intelligent load balancing

Provides a web-based dashboard for monitoring SOLLOL's routing decisions,
node health, performance metrics, and adaptive learning progress.

Usage:
    from sollol.dashboard import run_dashboard
    from sollol.integration import SOLLOLLoadBalancer

    # Create your load balancer
    load_balancer = SOLLOLLoadBalancer(registry)

    # Start dashboard in background thread
    import threading
    dashboard_thread = threading.Thread(
        target=run_dashboard,
        kwargs={'node_registry': registry, 'sollol_lb': load_balancer},
        daemon=True
    )
    dashboard_thread.start()

    # Dashboard runs at http://localhost:8080
"""

import collections
import logging
import os
from datetime import datetime
from typing import Optional

from flask import Flask, jsonify, send_file
from flask_cors import CORS
from flask_sockets import Sockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for dashboard
sockets = Sockets(app)


# In-memory log handler
class LogHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        for client in clients:
            if not client.closed:
                try:
                    client.send(log_entry)
                except Exception:
                    clients.remove(client)


# Global references - will be set by caller
registry = None
load_balancer = None


@app.route("/")
def index():
    """Serve the dashboard HTML."""
    # Look for dashboard.html in the same directory as this module
    dashboard_path = os.path.join(os.path.dirname(__file__), "..", "..", "dashboard.html")
    if os.path.exists(dashboard_path):
        return send_file(dashboard_path)
    else:
        return "<h1>SOLLOL Dashboard</h1><p>dashboard.html not found</p>", 404


@app.route("/api/dashboard")
def dashboard_data():
    """
    Get comprehensive dashboard data.

    Returns JSON with system status, performance metrics, hosts, alerts, and routing info.
    """
    if not load_balancer or not registry:
        return jsonify({"error": "Dashboard not initialized"}), 500

    stats = load_balancer.get_stats()
    healthy_nodes = registry.get_healthy_nodes()

    # Get all nodes - registry.nodes might be a dict
    if isinstance(registry.nodes, dict):
        all_nodes = list(registry.nodes.values())
    elif isinstance(registry.nodes, list):
        all_nodes = registry.nodes
    else:
        all_nodes = []

    # Calculate aggregated metrics with safe attribute access
    total_requests = sum(
        getattr(node.metrics, "total_requests", 0) for node in all_nodes if hasattr(node, "metrics")
    )
    successful_requests = sum(
        getattr(node.metrics, "successful_requests", 0)
        for node in all_nodes
        if hasattr(node, "metrics")
    )
    avg_success_rate = successful_requests / total_requests if total_requests > 0 else 1.0

    avg_latency = (
        sum(
            getattr(node.metrics, "avg_latency", 0)
            for node in healthy_nodes
            if hasattr(node, "metrics")
        )
        / len(healthy_nodes)
        if healthy_nodes
        else 0
    )

    total_gpu_memory = sum(
        getattr(node.capabilities, "gpu_memory_mb", 0)
        for node in all_nodes
        if hasattr(node, "capabilities") and getattr(node.capabilities, "has_gpu", False)
    )

    # Build host data
    hosts = []
    for node in all_nodes:
        # Safe attribute access
        url = getattr(node, "url", str(node))
        is_healthy = getattr(node, "is_healthy", True)

        # Get metrics safely
        avg_latency_node = 0.0
        total_reqs = 0
        successful_reqs = 0
        if hasattr(node, "metrics"):
            avg_latency_node = getattr(node.metrics, "avg_latency", 0.0)
            total_reqs = getattr(node.metrics, "total_requests", 0)
            successful_reqs = getattr(node.metrics, "successful_requests", 0)

        success_rate = successful_reqs / total_reqs if total_reqs > 0 else 1.0

        # Get load score
        load_score = 0.5
        if hasattr(node, "calculate_load_score"):
            try:
                load_score = node.calculate_load_score() / 100.0
            except:
                load_score = 0.5

        # Get GPU memory
        gpu_mb = 0
        if hasattr(node, "capabilities") and node.capabilities:
            gpu_mb = getattr(node.capabilities, "gpu_memory_mb", 0)

        host_data = {
            "host": url,
            "status": "healthy" if is_healthy else "offline",
            "latency_ms": avg_latency_node,
            "success_rate": success_rate,
            "load": load_score,
            "gpu_mb": gpu_mb,
        }

        # Mark degraded nodes (high latency or low success rate)
        if is_healthy and (avg_latency_node > 1000 or success_rate < 0.9):
            host_data["status"] = "degraded"

        hosts.append(host_data)

    # Build alerts
    alerts = []
    for node in all_nodes:
        url = getattr(node, "url", str(node))
        is_healthy = getattr(node, "is_healthy", True)

        if not is_healthy:
            last_check = getattr(node, "last_health_check", None)
            timestamp = last_check.isoformat() if last_check else datetime.now().isoformat()
            alerts.append(
                {"severity": "error", "message": f"Node {url} is offline", "timestamp": timestamp}
            )
        elif hasattr(node, "metrics"):
            avg_lat = getattr(node.metrics, "avg_latency", 0)
            if avg_lat > 1000:
                alerts.append(
                    {
                        "severity": "warning",
                        "message": f"High latency on {url}: {avg_lat:.0f}ms",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

    # Get routing patterns from metrics
    routing_patterns = []
    task_types_learned = 0
    if hasattr(load_balancer.metrics, "get_summary"):
        metrics_summary = load_balancer.metrics.get_summary()
        if "task_types" in metrics_summary:
            routing_patterns = list(metrics_summary["task_types"].keys())
            task_types_learned = len(routing_patterns)

    return jsonify(
        {
            "status": {
                "healthy": len(healthy_nodes) > 0,
                "available_hosts": len(healthy_nodes),
                "total_hosts": len(all_nodes),
                "ray_workers": 0,  # Not using Ray in embedded mode
            },
            "performance": {
                "avg_latency_ms": avg_latency,
                "avg_success_rate": avg_success_rate,
                "total_gpu_memory_mb": total_gpu_memory,
            },
            "hosts": hosts,
            "alerts": alerts,
            "routing": {
                "patterns_available": routing_patterns,
                "task_types_learned": task_types_learned,
            },
        }
    )


@app.route("/api/stats")
def detailed_stats():
    """Get detailed SOLLOL statistics."""
    if not load_balancer:
        return jsonify({"error": "Load balancer not initialized"}), 500
    return jsonify(load_balancer.get_stats())


@app.route("/api/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "sollol-dashboard"})


@app.route("/api/routing_decisions")
def routing_decisions():
    """
    Get recent routing decisions from Redis stream.

    Returns last 50 routing decisions with full metadata for dashboard tooltips.
    """
    try:
        import redis

        redis_url = os.getenv("SOLLOL_REDIS_URL", "redis://localhost:6379")
        redis_client = redis.from_url(redis_url, decode_responses=True)

        # Read last 50 routing decisions from stream
        stream_key = "sollol:routing_stream"
        messages = redis_client.xrevrange(stream_key, count=50)

        decisions = []
        for msg_id, msg_data in messages:
            try:
                import json

                event_json = msg_data.get("event", "{}")
                event = json.loads(event_json)

                # Only include routing decision events
                if event.get("event_type") in (
                    "ROUTE_DECISION",
                    "OLLAMA_NODE_SELECTED",
                    "RPC_BACKEND_SELECTED",
                ):
                    decisions.append(
                        {
                            "timestamp": event.get("timestamp"),
                            "model": event.get("model"),
                            "backend": event.get("backend"),
                            "node_url": event.get("node_url", "N/A"),
                            "reason": event.get("reason", "No reason provided"),
                            "task_type": event.get("task_type", "unknown"),
                            "complexity": event.get("complexity", "unknown"),
                            "score": event.get("score", 0),
                            "latency_ms": event.get("latency_ms", 0),
                            "gpu_mem": event.get("gpu_mem", 0),
                            "instance_id": event.get("instance_id"),
                        }
                    )
            except:
                continue

        return jsonify({"decisions": decisions})

    except Exception as e:
        # Redis not available - return empty list
        logger.debug(f"Could not fetch routing decisions: {e}")
        return jsonify({"decisions": []})


clients = []


@sockets.route("/api/logs/ws")
def log_socket(ws):
    """Real-time log streaming websocket."""
    clients.append(ws)
    while not ws.closed:
        # Keep the socket open
        ws.receive()


def run_dashboard(host="0.0.0.0", port=8080, production=True, node_registry=None, sollol_lb=None):
    """
    Run the SOLLOL dashboard server.

    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 8080)
        production: Use production WSGI server (default: True)
        node_registry: NodeRegistry instance to use (required)
        sollol_lb: SOLLOLLoadBalancer instance to use (required)

    Raises:
        ValueError: If node_registry or sollol_lb not provided
    """
    global registry, load_balancer

    if node_registry is None:
        raise ValueError("node_registry is required for dashboard")
    if sollol_lb is None:
        raise ValueError("sollol_lb (SOLLOLLoadBalancer) is required for dashboard")

    registry = node_registry
    load_balancer = sollol_lb

    # Attach log handler to root logger
    log_handler = LogHandler()
    log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(log_handler)

    logger.info(f"🚀 Starting SOLLOL Dashboard on http://{host}:{port}")
    logger.info(f"   Dashboard: http://{host}:{port}/")
    logger.info(f"   API Stats: http://{host}:{port}/api/dashboard")
    logger.info(f"   Tracking {len(registry)} nodes")

    if production:
        try:
            from gevent import pywsgi
            from geventwebsocket.handler import WebSocketHandler

            logger.info("   Using gevent WebSocket server for production")
            server = pywsgi.WSGIServer((host, port), app, handler_class=WebSocketHandler)
            server.serve_forever()
        except ImportError:
            logger.warning(
                "   gevent and gevent-websocket not installed, falling back to Flask dev server"
            )
            logger.warning(
                "   Install gevent and gevent-websocket for production: pip install gevent gevent-websocket"
            )
            app.run(host=host, port=port, debug=False, use_reloader=False)
    else:
        app.run(host=host, port=port, debug=False, use_reloader=False)


# Convenience exports
__all__ = ["run_dashboard", "app"]
