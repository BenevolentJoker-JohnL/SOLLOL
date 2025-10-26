"""
InfluxDB Time-Series Metrics Logger for SOLLOL

Provides a clean interface for logging performance metrics to InfluxDB 2.x.
Gracefully degrades if InfluxDB is unavailable or disabled.

Environment Variables:
    SOLLOL_METRICS_BACKEND: Enable metrics ("influxdb" or "disabled", default: disabled)
    INFLUX_URL: InfluxDB server URL (default: http://localhost:8086)
    INFLUX_TOKEN: InfluxDB authentication token
    INFLUX_ORG: InfluxDB organization (default: sollol)
    INFLUX_BUCKET: InfluxDB bucket for metrics (default: sollol_metrics)

Example:
    from sollol.metrics_logger import log_metric

    log_metric(
        "ollama_latency",
        tags={"node": "10.9.66.48:11434", "model": "llama3.1"},
        fields={"latency_ms": 7.8, "success": 1}
    )
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Global client instance (lazy-initialized)
_client = None
_write_api = None
_query_api = None
_enabled = None
_bucket = None


def _initialize():
    """Initialize InfluxDB client (lazy initialization)."""
    global _client, _write_api, _query_api, _enabled, _bucket

    if _enabled is not None:
        return  # Already initialized

    # Check if metrics backend is enabled (environment variable takes precedence)
    backend_env = os.getenv("SOLLOL_METRICS_BACKEND", "").lower()
    if backend_env == "disabled":
        _enabled = False
        logger.info("📊 Metrics backend explicitly disabled via SOLLOL_METRICS_BACKEND=disabled")
        return
    elif backend_env == "influxdb":
        _enabled = True
    else:
        # Use default (enabled)
        _enabled = True

    if not _enabled:
        return

    try:
        from influxdb_client import InfluxDBClient
        from influxdb_client.client.write_api import ASYNCHRONOUS

        url = os.getenv("INFLUX_URL", "http://localhost:8086")
        token = os.getenv("INFLUX_TOKEN")
        org = os.getenv("INFLUX_ORG", "sollol")
        _bucket = os.getenv("INFLUX_BUCKET", "sollol_metrics")

        if not token:
            logger.debug("InfluxDB metrics disabled (INFLUX_TOKEN not set)")
            _enabled = False
            return

        _client = InfluxDBClient(url=url, token=token, org=org)

        # Use async batching for high-throughput scenarios
        _write_api = _client.write_api(write_options=ASYNCHRONOUS)
        _query_api = _client.query_api()

        logger.info(f"✅ InfluxDB metrics enabled: {url} → bucket={_bucket}")

    except ImportError:
        logger.debug("InfluxDB metrics disabled (influxdb-client not installed)")
        _enabled = False
    except Exception as e:
        logger.error(f"❌ Failed to initialize InfluxDB client: {e}")
        _enabled = False


def is_enabled() -> bool:
    """Check if metrics logging is enabled."""
    _initialize()
    return _enabled or False


def log_metric(
    measurement: str,
    tags: Dict[str, str],
    fields: Dict[str, Any],
    timestamp: Optional[datetime] = None,
) -> bool:
    """
    Log a metric point to InfluxDB.

    Args:
        measurement: Metric name (e.g., "ollama_latency", "rpc_request")
        tags: Dictionary of tags (indexed dimensions, e.g., {"node": "10.9.66.48"})
        fields: Dictionary of field values (e.g., {"latency_ms": 7.8, "success": 1})
        timestamp: Optional timestamp (defaults to now)

    Returns:
        True if logged successfully, False if disabled or failed

    Example:
        log_metric(
            "ollama_request",
            tags={"node": "10.9.66.48", "model": "llama3.1", "operation": "generate"},
            fields={"latency_ms": 12.5, "tokens": 50, "success": 1}
        )
    """
    _initialize()

    if not _enabled or not _write_api:
        return False

    try:
        from influxdb_client import Point

        point = Point(measurement)

        # Add tags (indexed, for filtering)
        for key, value in tags.items():
            if value is not None:
                point = point.tag(key, str(value))

        # Add fields (actual metric values)
        for key, value in fields.items():
            if value is not None:
                # Convert booleans to integers for InfluxDB
                if isinstance(value, bool):
                    value = 1 if value else 0
                point = point.field(key, value)

        # Set timestamp if provided
        if timestamp:
            point = point.time(timestamp)

        _write_api.write(bucket=_bucket, record=point)
        return True

    except Exception as e:
        logger.debug(f"Failed to log metric {measurement}: {e}")
        return False


def log_node_health(
    node_url: str,
    healthy: bool,
    latency_ms: float,
    models_loaded: int = 0,
    vram_free_mb: int = 0,
    vram_total_mb: int = 0,
    failure_count: int = 0,
):
    """
    Log Ollama node health metrics.

    Args:
        node_url: Node URL (e.g., "http://10.9.66.48:11434")
        healthy: Whether node is healthy
        latency_ms: Response latency in milliseconds
        models_loaded: Number of models currently loaded
        vram_free_mb: Free VRAM in MB
        vram_total_mb: Total VRAM in MB
        failure_count: Consecutive failure count
    """
    log_metric(
        "node_health",
        tags={"node": node_url, "service": "ollama"},
        fields={
            "healthy": healthy,
            "latency_ms": latency_ms,
            "models_loaded": models_loaded,
            "vram_free_mb": vram_free_mb,
            "vram_total_mb": vram_total_mb,
            "vram_usage_percent": (
                100 * (vram_total_mb - vram_free_mb) / vram_total_mb if vram_total_mb > 0 else 0
            ),
            "failure_count": failure_count,
        },
    )


def log_rpc_backend_health(backend_url: str, reachable: bool, latency_ms: float = 0):
    """
    Log RPC backend health metrics.

    Args:
        backend_url: Backend URL (e.g., "10.9.66.48:50052")
        reachable: Whether backend is reachable
        latency_ms: Connection latency in milliseconds
    """
    log_metric(
        "rpc_health",
        tags={"backend": backend_url, "service": "llama_cpp_rpc"},
        fields={
            "reachable": reachable,
            "latency_ms": latency_ms,
        },
    )


def log_request(
    service: str,
    node: str,
    model: str,
    operation: str,
    latency_ms: float,
    success: bool,
    tokens: int = 0,
    error: Optional[str] = None,
):
    """
    Log a request/response metric.

    Args:
        service: Service type ("ollama" or "rpc")
        node: Node/backend identifier
        model: Model name
        operation: Operation type (e.g., "generate", "embed", "inference")
        latency_ms: Request latency in milliseconds
        success: Whether request succeeded
        tokens: Number of tokens processed
        error: Error message if failed
    """
    log_metric(
        "request",
        tags={
            "service": service,
            "node": node,
            "model": model,
            "operation": operation,
            "success": "true" if success else "false",
        },
        fields={
            "latency_ms": latency_ms,
            "success": success,
            "tokens": tokens,
            "error": 1 if error else 0,
        },
    )


def log_routing_decision(
    model: str,
    selected_node: str,
    strategy: str,
    candidate_nodes: int,
    selection_latency_ms: float = 0,
):
    """
    Log a routing decision metric.

    Args:
        model: Model name being routed
        selected_node: Selected node URL
        strategy: Routing strategy used
        candidate_nodes: Number of candidate nodes considered
        selection_latency_ms: Time taken to select node
    """
    log_metric(
        "routing_decision",
        tags={"model": model, "selected_node": selected_node, "strategy": strategy},
        fields={"candidate_nodes": candidate_nodes, "selection_latency_ms": selection_latency_ms},
    )


def query_metrics(query: str) -> list:
    """
    Execute a Flux query against InfluxDB.

    Args:
        query: Flux query string

    Returns:
        List of query result tables

    Example:
        query = '''
        from(bucket:"sollol_metrics")
          |> range(start: -1h)
          |> filter(fn: (r) => r._measurement == "node_health")
          |> mean()
        '''
        results = query_metrics(query)
    """
    _initialize()

    if not _enabled or not _query_api:
        return []

    try:
        return _query_api.query(query)
    except Exception as e:
        logger.error(f"Failed to query metrics: {e}")
        return []


def get_node_latency_avg(node_url: str, window: str = "-1h") -> Optional[float]:
    """
    Get average latency for a node over time window.

    Args:
        node_url: Node URL
        window: Time window (e.g., "-1h", "-5m", "-1d")

    Returns:
        Average latency in ms, or None if unavailable
    """
    query = f"""
    from(bucket:"{_bucket}")
      |> range(start: {window})
      |> filter(fn: (r) => r._measurement == "node_health")
      |> filter(fn: (r) => r.node == "{node_url}")
      |> filter(fn: (r) => r._field == "latency_ms")
      |> mean()
    """

    tables = query_metrics(query)
    for table in tables:
        for record in table.records:
            return record.get_value()
    return None


def get_success_rate(node_url: str, window: str = "-1h") -> Optional[float]:
    """
    Get success rate for a node over time window.

    Args:
        node_url: Node URL
        window: Time window (e.g., "-1h", "-5m", "-1d")

    Returns:
        Success rate as percentage (0-100), or None if unavailable
    """
    query = f"""
    from(bucket:"{_bucket}")
      |> range(start: {window})
      |> filter(fn: (r) => r._measurement == "request")
      |> filter(fn: (r) => r.node == "{node_url}")
      |> filter(fn: (r) => r._field == "success")
      |> mean()
      |> map(fn: (r) => ({{r with _value: r._value * 100.0}}))
    """

    tables = query_metrics(query)
    for table in tables:
        for record in table.records:
            return record.get_value()
    return None


def close():
    """Close InfluxDB client connection."""
    global _client, _write_api, _query_api, _enabled

    if _client:
        try:
            _client.close()
        except:
            pass

    _client = None
    _write_api = None
    _query_api = None
    _enabled = None
