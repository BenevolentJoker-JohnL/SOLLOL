"""
SOLLOL - Super Ollama Load Balancer

Intelligent Load Balancing and Distributed Inference for Ollama

Features:
- Intelligent load balancing with adaptive routing
- Auto-discovery of Ollama nodes and RPC backends
- Hybrid routing: Ollama for small models, llama.cpp for large models
- Connection pooling and health monitoring
- Request hedging for lower latency

Example:
    ```python
    from sollol import OllamaPool, HybridRouter
    from sollol.discovery import discover_ollama_nodes
    from sollol.rpc_discovery import auto_discover_rpc_backends

    # Auto-discover Ollama nodes
    pool = OllamaPool.auto_configure()

    # Use hybrid routing for distributed inference
    rpc_backends = auto_discover_rpc_backends()
    router = HybridRouter(
        ollama_pool=pool,
        rpc_backends=rpc_backends,
        enable_distributed=True
    )

    # Make requests
    response = await router.route_request(
        model="llama3.1:405b",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    ```
"""

import logging

# Configure library logging - use NullHandler by default when used as a library
# Applications can configure their own handlers if they want SOLLOL logs
logging.getLogger("sollol").addHandler(logging.NullHandler())
logging.getLogger("sollol").setLevel(logging.WARNING)  # Only show warnings/errors by default

# Adaptive parallelism
from sollol.adaptive_parallelism import AdaptiveParallelismStrategy
from sollol.client import SOLLOLClient
from sollol.config import SOLLOLConfig
from sollol.dashboard_client import DashboardClient

# Discovery
from sollol.discovery import discover_ollama_nodes
from sollol.distributed_tracing import DistributedTracer, get_tracer

# Docker IP Resolution
from sollol.docker_ip_resolver import (
    auto_resolve_ips,
    get_deployment_context,
    get_docker_network_mode,
    is_docker_ip,
    is_running_in_docker,
    resolve_docker_ip,
    resolve_docker_ip_with_alternatives,
)

# Distributed execution
from sollol.execution import AsyncDistributedExecutor, DistributedExecutor
from sollol.gpu_auto_setup import GPUAutoSetup, auto_setup_gpu_monitoring

# GPU monitoring
from sollol.gpu_redis_subscriber import GPURedisSubscriber

# Distributed inference
from sollol.hybrid_router import HybridRouter
from sollol.llama_cpp_coordinator import LlamaCppCoordinator
from sollol.llama_cpp_rpc import LlamaCppDistributedCluster, LlamaCppRPCClient

# Time-Series Metrics (InfluxDB)
from sollol.metrics_logger import is_enabled as metrics_enabled
from sollol.metrics_logger import (
    log_metric,
    log_node_health,
    log_request,
    log_routing_decision,
    log_rpc_backend_health,
    query_metrics,
)
from sollol.network_observer import NetworkObserver, get_observer

# Core load balancing
from sollol.pool import OllamaPool
from sollol.ray_advanced_router import RayAdvancedRouter
from sollol.ray_hybrid_router import RayHybridRouter
from sollol.rpc_auto_setup import RPCAutoSetup, auto_setup_rpc_backends
from sollol.rpc_discovery import auto_discover_rpc_backends, check_rpc_server
from sollol.rpc_registry import RPCBackendRegistry

# Legacy support
from sollol.sollol import SOLLOL
from sollol.tasks import DistributedTask, ExecutionResult, TaskResult

# Monitoring & Dashboards
from sollol.unified_dashboard import UnifiedDashboard, run_unified_dashboard

__version__ = "0.9.66"
__all__ = [
    # Core
    "OllamaPool",
    "SOLLOLClient",
    # Distributed inference
    "HybridRouter",
    "RayHybridRouter",
    "RayAdvancedRouter",
    "LlamaCppRPCClient",
    "LlamaCppDistributedCluster",
    "LlamaCppCoordinator",
    # Monitoring & Dashboards
    "UnifiedDashboard",
    "run_unified_dashboard",
    "DistributedTracer",
    "get_tracer",
    "DashboardClient",
    "NetworkObserver",
    "get_observer",
    # Time-Series Metrics
    "metrics_enabled",
    "log_metric",
    "log_node_health",
    "log_rpc_backend_health",
    "log_request",
    "log_routing_decision",
    "query_metrics",
    # Distributed execution
    "DistributedExecutor",
    "AsyncDistributedExecutor",
    "DistributedTask",
    "TaskResult",
    "ExecutionResult",
    # Discovery & Auto-Setup
    "discover_ollama_nodes",
    "auto_discover_rpc_backends",
    "auto_setup_rpc_backends",
    "check_rpc_server",
    "RPCAutoSetup",
    "RPCBackendRegistry",
    "GPUAutoSetup",
    "auto_setup_gpu_monitoring",
    # Docker IP Resolution
    "is_docker_ip",
    "resolve_docker_ip",
    "resolve_docker_ip_with_alternatives",
    "auto_resolve_ips",
    "is_running_in_docker",
    "get_docker_network_mode",
    "get_deployment_context",
    # Adaptive Parallelism
    "AdaptiveParallelismStrategy",
    # GPU Monitoring
    "GPURedisSubscriber",
    # Legacy
    "SOLLOL",
    "SOLLOLConfig",
]
