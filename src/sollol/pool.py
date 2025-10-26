"""
Zero-config Ollama connection pool with intelligent load balancing.

Auto-discovers nodes, manages connections, routes requests intelligently.
Thread-safe and ready to use immediately.

Features full SynapticLlamas observability:
- Intelligent routing with task analysis
- Performance tracking and learning
- Detailed logging of routing decisions
- Real-time VRAM monitoring for GPU-aware routing
"""

import asyncio
import json
import logging
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

# Force unbuffered stderr for real-time progress display
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)
elif hasattr(sys.stderr, "buffer"):
    # Python 3.6 compatibility
    class Unbuffered:
        def __init__(self, stream):
            self.stream = stream

        def write(self, data):
            self.stream.write(data)
            self.stream.flush()

        def __getattr__(self, attr):
            return getattr(self.stream, attr)

    sys.stderr = Unbuffered(sys.stderr)

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    import requests

    HTTPX_AVAILABLE = False

try:
    from distributed import Client as DaskClient

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

from .intelligence import IntelligentRouter, get_router
from .metrics_logger import log_node_health, log_request
from .network_observer import (
    log_node_health_check,
    log_node_status_change,
    log_ollama_error,
    log_ollama_request,
    log_ollama_response,
)
from .node_health import NodeHealthMonitor, normalize_model_name
from .routing_logger import get_routing_logger
from .routing_strategy import RoutingStrategy
from .vram_monitor import VRAMMonitor

logger = logging.getLogger(__name__)


# Worker-local pool cache (reused across Dask tasks on same worker)
_worker_pool_cache = {}


# Module-level helper for Dask (must be picklable)
def _dask_embed_task(pool_nodes, model, text_data, priority, **kwargs):
    """
    Standalone embedding task for Dask distributed processing.

    This function must be at module level to be picklable by Dask.
    Creates a worker-local pool that's reused across tasks.
    """
    idx, text = text_data
    try:
        # Get or create worker-local pool (cached per worker process)
        import os

        worker_id = os.getpid()

        if worker_id not in _worker_pool_cache:
            # Create a lightweight pool instance for this worker
            from .pool import OllamaPool

            _worker_pool_cache[worker_id] = OllamaPool(
                nodes=pool_nodes,
                enable_cache=False,
                enable_dask=False,
                register_with_dashboard=False,
            )

        pool = _worker_pool_cache[worker_id]
        result = pool.embed(model, text, priority=priority, **kwargs)
        return idx, result, None
    except Exception as e:
        return idx, None, str(e)


class OllamaPool:
    """
    Connection pool that automatically discovers and load balances across Ollama nodes.

    Usage:
        pool = OllamaPool.auto_configure()
        response = pool.chat("llama3.2", [{"role": "user", "content": "Hi"}])
    """

    # VRAM safety buffer (MB) - subtracted from reported free VRAM to prevent OOM
    VRAM_BUFFER_MB = 200  # 0.2GB cushion for system processes and safety margin

    def __init__(
        self,
        nodes: Optional[List[Dict[str, str]]] = None,
        enable_intelligent_routing: bool = True,
        routing_strategy: RoutingStrategy = RoutingStrategy.INTELLIGENT,
        exclude_localhost: bool = False,
        discover_all_nodes: bool = False,
        app_name: Optional[str] = None,
        register_with_dashboard: bool = True,
        enable_ray: bool = False,
        enable_dask: bool = True,
        dask_address: Optional[str] = None,
        enable_gpu_redis: bool = True,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        enable_cache: bool = True,
        cache_max_size: int = 1000,
        cache_ttl: int = 3600,
    ):
        """
        Initialize connection pool with full observability.

        Args:
            nodes: List of node dicts. If None, auto-discovers.
            enable_intelligent_routing: Use intelligent routing (default: True, deprecated - use routing_strategy instead)
            routing_strategy: Routing strategy to use (default: INTELLIGENT)
            exclude_localhost: Skip localhost during discovery (for SOLLOL gateway)
            discover_all_nodes: Scan full network for ALL nodes (slower but comprehensive)
            app_name: Custom application name for dashboard registration (e.g., "FlockParser")
            register_with_dashboard: Whether to auto-register with dashboard (default: True)
            enable_ray: Initialize Ray cluster for multi-app coordination (default: False)
            enable_dask: Use Dask for distributed batch processing (default: True)
            dask_address: Dask scheduler address (None = auto-connect or start local)
            enable_gpu_redis: Subscribe to accurate GPU stats from Redis via gpustat (default: True, requires gpu_reporter.py on nodes)
            redis_host: Redis server hostname (default: localhost)
            redis_port: Redis server port (default: 6379)
            enable_cache: Enable response caching (default: True)
            cache_max_size: Maximum number of cached responses (default: 1000)
            cache_ttl: Cache TTL in seconds (default: 3600 = 1 hour)
        """
        # Normalize nodes: convert string URLs to dicts
        if nodes:
            self.nodes = [self._normalize_node(n) for n in nodes]
        else:
            self.nodes = []
        self.exclude_localhost = exclude_localhost
        self.discover_all_nodes = discover_all_nodes
        self.app_name = app_name  # Store custom app name for dashboard registration
        self.register_with_dashboard = register_with_dashboard  # Control dashboard registration
        self._lock = threading.Lock()
        self._current_index = 0

        # Persistent HTTP session for connection reuse with HTTP/2 support
        if HTTPX_AVAILABLE:
            # Use httpx with HTTP/2 multiplexing for better performance
            limits = httpx.Limits(
                max_keepalive_connections=max(20, len(self.nodes or []) * 20),
                max_connections=max(50, len(self.nodes or []) * 50),
                keepalive_expiry=30.0,
            )

            # Configure retry transport
            transport = httpx.HTTPTransport(retries=3, limits=limits, http2=True)  # Enable HTTP/2

            self.session = httpx.Client(
                transport=transport,
                timeout=httpx.Timeout(300.0, connect=10.0),
                follow_redirects=True,
                headers={"Connection": "keep-alive"},
            )
            logger.info("🚀 HTTP/2 multiplexing enabled via httpx")
        else:
            # Fallback to requests (HTTP/1.1 only)
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            self.session = requests.Session()
            self.session.headers.update({"Connection": "keep-alive"})

            retry_strategy = Retry(
                total=3,
                backoff_factor=0.1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["POST", "GET"],
            )

            adapter = HTTPAdapter(
                pool_connections=max(10, len(self.nodes or []) * 10),
                pool_maxsize=max(20, len(self.nodes or []) * 20),
                max_retries=retry_strategy,
            )

            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
            logger.warning("⚠️  httpx not available, using requests (HTTP/1.1 only)")

        # Initialize async session for async I/O support (httpx only)
        self.async_session = None
        if HTTPX_AVAILABLE:
            limits = httpx.Limits(
                max_keepalive_connections=max(20, len(self.nodes or []) * 20),
                max_connections=max(50, len(self.nodes or []) * 50),
                keepalive_expiry=30.0,
            )

            transport = httpx.AsyncHTTPTransport(retries=3, limits=limits, http2=True)

            self.async_session = httpx.AsyncClient(
                transport=transport,
                timeout=httpx.Timeout(300.0, connect=10.0),
                follow_redirects=True,
                headers={"Connection": "keep-alive"},
            )
            logger.info("⚡ Async I/O support enabled with httpx AsyncClient")

        # Auto-discover if no nodes provided
        if not self.nodes:
            self._auto_discover()

        # Initialize response cache
        from .response_cache import ResponseCache

        self.cache = ResponseCache(max_size=cache_max_size, ttl=cache_ttl, enabled=enable_cache)

        # Initialize routing strategy
        # Handle backwards compatibility: enable_intelligent_routing overrides routing_strategy
        if not enable_intelligent_routing and routing_strategy == RoutingStrategy.INTELLIGENT:
            # User explicitly disabled intelligent routing but didn't specify strategy
            self.routing_strategy = RoutingStrategy.ROUND_ROBIN
        else:
            self.routing_strategy = routing_strategy

        # Keep enable_intelligent_routing for backwards compatibility
        self.enable_intelligent_routing = self.routing_strategy == RoutingStrategy.INTELLIGENT
        self.router = get_router() if self.enable_intelligent_routing else None

        # Initialize routing event logger (separate channel from regular logs)
        self.routing_logger = get_routing_logger()

        # Initialize health monitoring (FlockParser pattern)
        self.health_monitor = NodeHealthMonitor()

        # Initialize VRAM monitoring for GPU-aware routing
        self.vram_monitor = VRAMMonitor()
        self._vram_refresh_interval = 30  # seconds
        self._vram_refresh_enabled = True

        # Health check configuration
        self._health_check_base_interval = 30  # seconds - base interval for healthy nodes
        self._health_check_enabled = True
        self._adaptive_health_checks = True  # Enable adaptive intervals based on node stability

        # Enhanced stats tracking with performance metrics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "nodes_used": {},
            "node_performance": {},  # Track performance per node
        }

        # Latency buffer for percentile calculation (rolling window of last 1000 requests)
        self._latency_buffer = deque(maxlen=1000)

        # Redis client for publishing metrics to dashboard
        self._metrics_redis_client = None
        if enable_gpu_redis:  # Reuse Redis connection settings
            try:
                import redis

                self._metrics_redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=False,  # We'll handle JSON encoding
                    socket_connect_timeout=2,
                    socket_timeout=2,
                )
                # Test connection
                self._metrics_redis_client.ping()
                logger.debug(f"📊 Metrics Redis connected at {redis_host}:{redis_port}")
            except Exception as e:
                logger.debug(f"Metrics Redis unavailable (metrics won't be published): {e}")
                self._metrics_redis_client = None

        # Deduplicate nodes (removes localhost if real IP exists)
        self._deduplicate_nodes()

        # Initialize node metadata for intelligent routing
        self._init_node_metadata()

        # Initialize Ray cluster if requested (for multi-app coordination)
        if enable_ray:
            self._init_ray_cluster()

        # Initialize Dask client for distributed batch processing
        self.dask_client = None
        if enable_dask:
            self._init_dask_client(dask_address)

        # Start background VRAM monitoring thread
        self._vram_refresh_thread = threading.Thread(
            target=self._refresh_vram_loop, daemon=True, name="OllamaPool-VRAM-Monitor"
        )
        self._vram_refresh_thread.start()

        # Start background health check thread for live dashboard updates
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop, daemon=True, name="OllamaPool-Health-Monitor"
        )
        self._health_check_thread.start()

        # Start GPU Redis subscriber if enabled (accurate GPU stats from nodes)
        self.gpu_subscriber = None
        if enable_gpu_redis:
            try:
                from .gpu_redis_subscriber import GPURedisSubscriber

                self.gpu_subscriber = GPURedisSubscriber(
                    pool=self, redis_host=redis_host, redis_port=redis_port
                )
                self.gpu_subscriber.start(interval=5)  # Poll every 5 seconds
                logger.info(f"✅ GPU Redis subscriber enabled (Redis: {redis_host}:{redis_port})")
            except Exception as e:
                logger.warning(f"Failed to start GPU Redis subscriber: {e}")

        # Start metrics publishing thread (publishes p50/p95/p99 latency to Redis)
        if self._metrics_redis_client:
            self._metrics_thread = threading.Thread(
                target=self._publish_metrics_loop, daemon=True, name="OllamaPool-Metrics-Publisher"
            )
            self._metrics_thread.start()
            logger.debug("📊 Metrics publisher thread started")

        logger.info(
            f"OllamaPool initialized with {len(self.nodes)} nodes "
            f"(intelligent_routing={'enabled' if enable_intelligent_routing else 'disabled'}, "
            f"vram_monitoring=enabled, health_checks=enabled, gpu_type={self.vram_monitor.gpu_type}, "
            f"gpu_redis={'enabled' if enable_gpu_redis else 'disabled'}, "
            f"dask={'enabled' if self.dask_client else 'disabled'})"
        )

        # Auto-register with SOLLOL dashboard if available
        if self.register_with_dashboard:
            self._auto_register_with_dashboard()

    @classmethod
    def auto_configure(
        cls, discover_all_nodes: bool = False, setup_local_gpu_monitoring: bool = False, **kwargs
    ) -> "OllamaPool":
        """
        Create pool with automatic discovery.

        Args:
            discover_all_nodes: If True, scan full network for ALL nodes (default: False for speed)
            setup_local_gpu_monitoring: If True, auto-setup GPU monitoring on localhost (default: False)
            **kwargs: Additional arguments passed to __init__ (e.g., enable_cache, cache_max_size, cache_ttl)

        Returns:
            OllamaPool instance ready to use

        Example:
            >>> # Auto-discover nodes and setup local GPU monitoring
            >>> pool = OllamaPool.auto_configure(setup_local_gpu_monitoring=True)
        """
        # Auto-setup GPU monitoring on localhost if requested
        if setup_local_gpu_monitoring:
            try:
                from .gpu_auto_setup import auto_setup_gpu_monitoring

                logger.info("🔧 Setting up local GPU monitoring...")

                redis_host = kwargs.get("redis_host", "localhost")
                redis_port = kwargs.get("redis_port", 6379)

                success = auto_setup_gpu_monitoring(
                    redis_host=redis_host,
                    redis_port=redis_port,
                    auto_install=True,
                    auto_start=True,
                )

                if success:
                    logger.info("✅ Local GPU monitoring ready")
                else:
                    logger.warning("⚠️  GPU monitoring setup failed (continuing anyway)")

            except Exception as e:
                logger.warning(f"⚠️  GPU monitoring setup failed: {e} (continuing anyway)")

        return cls(nodes=None, discover_all_nodes=discover_all_nodes, **kwargs)

    def _normalize_node(self, node):
        """
        Normalize a node to dict format.

        Args:
            node: Either a string URL like "http://localhost:11434" or a dict {"host": "...", "port": "..."}

        Returns:
            Dict with "host" and "port" keys
        """
        if isinstance(node, dict):
            # Already a dict, ensure it has the required keys
            if "host" in node and "port" in node:
                return node
            elif "url" in node:
                # Convert from URL-based dict
                url = node["url"]
                if "://" in url:
                    url = url.split("://", 1)[1]
                if ":" in url:
                    host, port = url.rsplit(":", 1)
                else:
                    host, port = url, "11434"
                return {"host": host, "port": port}
            else:
                raise ValueError(f"Invalid node dict format: {node}")
        elif isinstance(node, str):
            # String URL like "http://localhost:11434" or "localhost:11434"
            url = node
            if "://" in url:
                url = url.split("://", 1)[1]  # Remove http:// or https://
            if ":" in url:
                host, port = url.rsplit(":", 1)
            else:
                host, port = url, "11434"  # Default Ollama port
            return {"host": host, "port": port}
        else:
            raise ValueError(f"Node must be string URL or dict, got {type(node)}: {node}")

    def _auto_discover(self):
        """Discover Ollama nodes automatically."""
        import socket

        from .discovery import discover_ollama_nodes

        if self.discover_all_nodes:
            logger.info("Auto-discovering ALL Ollama nodes on network (full subnet scan)...")
        elif self.exclude_localhost:
            logger.debug("Auto-discovering Ollama nodes (excluding localhost)...")
        else:
            logger.debug("Auto-discovering Ollama nodes...")

        nodes = discover_ollama_nodes(
            timeout=2.0,  # Increased for high-latency nodes (was 0.5s)
            exclude_localhost=self.exclude_localhost,
            discover_all_nodes=self.discover_all_nodes,
        )

        with self._lock:
            self.nodes = nodes
            if self.exclude_localhost and len(nodes) == 0:
                logger.info("No remote Ollama nodes found (localhost excluded)")
            else:
                logger.info(f"Auto-discovered {len(nodes)} nodes: {nodes}")

    def _deduplicate_nodes(self):
        """
        Remove duplicate nodes where localhost/127.0.0.1 refers to the same machine as a real IP.

        This handles cases where nodes are loaded from config files or manually added,
        ensuring localhost and the machine's real IP aren't both shown.
        """
        if not self.nodes:
            return

        import socket

        # Get this machine's actual IP
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("10.255.255.255", 1))  # Doesn't actually connect, just determines route
            local_ip = s.getsockname()[0]
            s.close()
        except:
            return  # Can't determine local IP, skip deduplication

        with self._lock:
            # Check if we have both localhost and the real IP
            # NOTE: All 127.x.x.x addresses are loopback aliases!
            has_localhost = any(
                node["host"] in ("localhost", "127.0.0.1") or node["host"].startswith("127.")
                for node in self.nodes
            )
            has_real_ip = any(node["host"] == local_ip for node in self.nodes)

            # If we have both, filter out ALL localhost entries (entire 127.0.0.0/8 subnet)
            if has_localhost and has_real_ip:
                original_count = len(self.nodes)
                self.nodes = [
                    node
                    for node in self.nodes
                    if not (
                        node["host"] in ("localhost", "127.0.0.1")
                        or node["host"].startswith("127.")
                    )
                ]
                logger.info(
                    f"🔍 Deduplicated nodes: removed localhost aliases (same as {local_ip})"
                )
                logger.debug(f"   Reduced from {original_count} to {len(self.nodes)} nodes")

    def count_unique_physical_hosts(self) -> int:
        """
        Count unique physical machines in the node pool.

        This is critical for parallel execution decisions - running parallel tasks
        on the same physical machine is often slower due to resource contention.

        Returns:
            Number of unique physical machines

        Examples:
            - localhost:11434 + localhost:11435 = 1 unique host
            - localhost:11434 + 192.168.1.20:11434 = 2 unique hosts
            - 127.0.0.1:11434 + 0.0.0.0:11434 = 1 unique host
        """
        if not self.nodes:
            return 0

        import socket

        unique_hosts = set()

        for node in self.nodes:
            hostname = node.get("host", "")

            # Normalize all 127.x.x.x addresses to 127.0.0.1 (entire 127.0.0.0/8 is loopback)
            if hostname.startswith("127."):
                hostname = "127.0.0.1"

            # Resolve to IP to catch localhost aliases
            try:
                ip = socket.gethostbyname(hostname)
                # Normalize loopback IPs
                if ip.startswith("127."):
                    ip = "127.0.0.1"
                unique_hosts.add(ip)
            except:
                # If resolution fails, use hostname as-is
                unique_hosts.add(hostname)

        return len(unique_hosts)

    def should_use_parallel_execution(self, num_tasks: int) -> bool:
        """
        Intelligent decision: Should we use parallel execution?

        Considers:
        - Number of unique physical machines
        - Number of tasks to execute
        - Resource contention on same machine

        Args:
            num_tasks: Number of parallel tasks to execute

        Returns:
            True if parallel execution will be beneficial

        Note:
            Running parallel CPU inference on the same physical machine is typically
            50-100% SLOWER than sequential due to:
            - CPU context switching overhead
            - Memory bandwidth saturation
            - Cache thrashing
            - BLAS library serialization
        """
        if num_tasks < 2:
            return False  # Nothing to parallelize

        if not self.nodes:
            return False  # No nodes available

        unique_hosts = self.count_unique_physical_hosts()

        if unique_hosts < 2:
            logger.warning(
                f"⚠️  Parallel execution NOT recommended: {len(self.nodes)} nodes available "
                f"but all on same physical machine.\n"
                f"   Running {num_tasks} parallel CPU tasks on same machine is typically "
                f"50-100% SLOWER than sequential due to resource contention.\n"
                f"   💡 Recommendation: Use sequential execution or add nodes on different machines."
            )
            return False

        # We have multiple physical machines - parallel makes sense
        logger.info(
            f"✅ Parallel execution enabled: {unique_hosts} physical machines available, "
            f"{len(self.nodes)} total nodes"
        )
        return True

    def _init_node_metadata(self):
        """Initialize metadata for each node with REAL VRAM data."""
        with self._lock:
            for node in self.nodes:
                node_key = f"{node['host']}:{node['port']}"
                if node_key not in self.stats["node_performance"]:
                    # Query VRAM for this node
                    gpu_free_mem = self._query_node_vram(node)

                    self.stats["node_performance"][node_key] = {
                        "host": node_key,
                        "latency_ms": 0.0,
                        "success_rate": 1.0,
                        "total_requests": 0,
                        "failed_requests": 0,
                        "available": True,
                        "active_requests": 0,  # Real-time concurrent load
                        "cpu_load": 0.5,  # Default assumption
                        "gpu_free_mem": gpu_free_mem,  # REAL VRAM DATA
                        "priority": 999,  # Default priority
                    }

                    logger.debug(f"Initialized {node_key}: gpu_free_mem={gpu_free_mem}MB")

    def _init_ray_cluster(self):
        """Initialize Ray cluster for multi-app coordination."""
        try:
            import json
            import os

            import ray

            if ray.is_initialized():
                logger.info("✅ Ray already initialized (shared cluster)")
                return

            # Disable Ray memory monitor
            os.environ["RAY_memory_monitor_refresh_ms"] = "0"

            # Try to connect to existing Ray cluster first (multi-app coordination)
            try:
                logger.info("🔍 Attempting to connect to existing Ray cluster...")
                ray.init(address="auto", ignore_reinit_error=True)
                logger.info("✅ Connected to existing Ray cluster")
            except (ConnectionError, Exception) as e:
                # No existing cluster, start a new one
                logger.info("🚀 Starting new Ray cluster for multi-app coordination")

                # Conservative memory settings
                ray.init(
                    ignore_reinit_error=True,
                    dashboard_host="0.0.0.0",
                    dashboard_port=8265,
                    include_dashboard=True,
                    num_cpus=1,  # Minimal workers
                    object_store_memory=256 * 1024 * 1024,  # 256MB for object store
                    _system_config={
                        "automatic_object_spilling_enabled": True,
                        "object_spilling_config": json.dumps(
                            {"type": "filesystem", "params": {"directory_path": "/tmp/ray_spill"}}
                        ),
                    },
                )
                logger.info("📊 Ray dashboard available at http://localhost:8265")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize Ray cluster: {e}")
            logger.info("   Continuing without Ray (not required for OllamaPool)")

    def _init_dask_client(self, dask_address: Optional[str] = None):
        """Initialize Dask client for distributed batch processing."""
        if not DASK_AVAILABLE:
            logger.info("⚠️  Dask not available - install with: pip install dask distributed")
            logger.info("   Falling back to ThreadPoolExecutor for batch processing")
            return

        try:
            # Try to connect to existing Dask scheduler first
            if dask_address:
                logger.info(f"🔍 Connecting to Dask scheduler at {dask_address}...")
                self.dask_client = DaskClient(dask_address)
                logger.info(f"✅ Connected to Dask scheduler at {dask_address}")
                logger.info(f"   Dask workers: {len(self.dask_client.scheduler_info()['workers'])}")
            else:
                # Try to connect to running scheduler
                try:
                    logger.info("🔍 Looking for existing Dask scheduler...")
                    self.dask_client = DaskClient(timeout="2s")
                    logger.info(f"✅ Connected to existing Dask scheduler")
                    logger.info(
                        f"   Dask workers: {len(self.dask_client.scheduler_info()['workers'])}"
                    )
                except (OSError, Exception):
                    # No existing scheduler, start local cluster
                    logger.info("🚀 Starting local Dask cluster for distributed batch processing")
                    from distributed import LocalCluster

                    # Start with conservative settings
                    cluster = LocalCluster(
                        n_workers=len(self.nodes) if self.nodes else 2,
                        threads_per_worker=2,
                        processes=True,
                        memory_limit="auto",
                        silence_logs=logging.WARNING,
                    )
                    self.dask_client = DaskClient(cluster)
                    logger.info(
                        f"✅ Dask cluster started with {len(self.nodes) if self.nodes else 2} workers"
                    )
                    logger.info(f"   Dashboard: {self.dask_client.dashboard_link}")

        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize Dask: {e}")
            logger.info("   Falling back to ThreadPoolExecutor for batch processing")
            self.dask_client = None

    def _select_node(
        self, payload: Optional[Dict[str, Any]] = None, priority: int = 5
    ) -> tuple[Dict[str, str], Optional[Dict[str, Any]]]:
        """
        Select best node for request using configured routing strategy.

        Args:
            payload: Request payload for task analysis
            priority: Request priority (1-10)

        Returns:
            (selected_node, routing_decision) tuple
        """
        with self._lock:
            if not self.nodes:
                raise RuntimeError("No Ollama nodes available")

            # Route based on configured strategy
            if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
                return self._select_round_robin(), None

            elif self.routing_strategy == RoutingStrategy.LATENCY_FIRST:
                return self._select_latency_first(), None

            elif self.routing_strategy == RoutingStrategy.LEAST_LOADED:
                return self._select_least_loaded(), None

            elif self.routing_strategy == RoutingStrategy.FAIRNESS:
                return self._select_fairness(), None

            elif self.routing_strategy == RoutingStrategy.INTELLIGENT:
                return self._select_intelligent(payload, priority)

            else:
                # Unknown strategy - fallback to round-robin
                logger.warning(
                    f"Unknown routing strategy: {self.routing_strategy}, using round-robin"
                )
                return self._select_round_robin(), None

    def _select_round_robin(self) -> Dict[str, str]:
        """
        Simple round-robin node selection.

        Rotates through nodes in order, providing predictable distribution.
        No intelligence, no overhead - just simple rotation.

        Returns:
            Selected node
        """
        node = self.nodes[self._current_index % len(self.nodes)]
        self._current_index += 1
        logger.debug(f"Round-robin selected: {node['host']}:{node['port']}")
        return node

    def _select_latency_first(self) -> Dict[str, str]:
        """
        Select node with lowest average latency.

        Prioritizes fastest responding nodes to minimize response time.
        Good for latency-sensitive applications.

        Returns:
            Node with lowest average latency
        """
        # Build list of (node, performance_data) tuples
        perf_nodes = [
            (node, self.stats["node_performance"].get(f"{node['host']}:{node['port']}", {}))
            for node in self.nodes
        ]

        # Filter out unavailable nodes
        available_nodes = [(node, perf) for node, perf in perf_nodes if perf.get("available", True)]

        if not available_nodes:
            # All nodes unavailable - fallback to first node
            logger.warning("All nodes unavailable for latency-first routing, using fallback")
            return self.nodes[0]

        # Select node with minimum average latency
        best_node, best_perf = min(
            available_nodes, key=lambda x: x[1].get("latency_ms", float("inf"))
        )

        latency = best_perf.get("latency_ms", 0)
        logger.debug(
            f"Latency-first selected: {best_node['host']}:{best_node['port']} ({latency:.1f}ms avg)"
        )
        return best_node

    def _select_least_loaded(self) -> Dict[str, str]:
        """
        Select node with fewest active requests.

        Maximizes parallelism by distributing load evenly across nodes.
        Good for high-throughput batch processing.

        Returns:
            Node with fewest active requests
        """
        # Build list of (node, performance_data) tuples
        perf_nodes = [
            (node, self.stats["node_performance"].get(f"{node['host']}:{node['port']}", {}))
            for node in self.nodes
        ]

        # Filter out unavailable nodes
        available_nodes = [(node, perf) for node, perf in perf_nodes if perf.get("available", True)]

        if not available_nodes:
            # All nodes unavailable - fallback to first node
            logger.warning("All nodes unavailable for least-loaded routing, using fallback")
            return self.nodes[0]

        # Select node with minimum active requests
        best_node, best_perf = min(available_nodes, key=lambda x: x[1].get("active_requests", 0))

        active = best_perf.get("active_requests", 0)
        logger.debug(
            f"Least-loaded selected: {best_node['host']}:{best_node['port']} ({active} active requests)"
        )
        return best_node

    def _select_fairness(self) -> Dict[str, str]:
        """
        Distribute requests evenly based on total request count.

        Ensures all nodes get equal share of requests over time.
        Good for fair resource utilization across heterogeneous hardware.

        Returns:
            Node with fewest total requests
        """
        # Build list of (node, performance_data) tuples
        perf_nodes = [
            (node, self.stats["node_performance"].get(f"{node['host']}:{node['port']}", {}))
            for node in self.nodes
        ]

        # Filter out unavailable nodes
        available_nodes = [(node, perf) for node, perf in perf_nodes if perf.get("available", True)]

        if not available_nodes:
            # All nodes unavailable - fallback to first node
            logger.warning("All nodes unavailable for fairness routing, using fallback")
            return self.nodes[0]

        # Select node with minimum total requests
        best_node, best_perf = min(available_nodes, key=lambda x: x[1].get("total_requests", 0))

        total = best_perf.get("total_requests", 0)
        logger.debug(
            f"Fairness selected: {best_node['host']}:{best_node['port']} ({total} total requests)"
        )
        return best_node

    def _select_intelligent(
        self, payload: Optional[Dict[str, Any]], priority: int
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        """
        Intelligent task-aware routing with performance learning.

        Analyzes request characteristics and selects optimal node based on:
        - Task type (generation, embedding, chat)
        - Model requirements
        - Historical performance
        - Node capabilities

        Args:
            payload: Request payload for task analysis
            priority: Request priority

        Returns:
            (selected_node, routing_decision) tuple
        """
        if not payload or not self.router:
            # No payload or router unavailable - fallback to round-robin
            logger.debug("Intelligent routing unavailable, falling back to round-robin")
            return self._select_round_robin(), None

        try:
            # Analyze request
            context = self.router.analyze_request(payload, priority=priority)

            # Get available hosts metadata
            available_hosts = list(self.stats["node_performance"].values())

            # Select optimal node
            selected_host, decision = self.router.select_optimal_node(context, available_hosts)

            # Find matching node dict
            for node in self.nodes:
                node_key = f"{node['host']}:{node['port']}"
                if node_key == selected_host:
                    # Log the routing decision
                    logger.info(f"🎯 Intelligent routing: {decision['reasoning']}")

                    # Log to SOLLOL routing stream (separate from regular logs)
                    model = payload.get("model", "unknown") if payload else "unknown"
                    node_url = f"{node['host']}:{node['port']}"
                    self.routing_logger.log_ollama_node_selected(
                        node_url=node_url,
                        model=model,
                        reason=decision["reasoning"],
                        confidence=decision.get("confidence", 0),
                        priority=priority,
                    )
                    return node, decision

            # Fallback if not found
            logger.warning(
                f"Selected host {selected_host} not in nodes, using round-robin fallback"
            )
            return self._select_round_robin(), None

        except Exception as e:
            logger.warning(f"Intelligent routing failed: {e}, falling back to round-robin")
            return self._select_round_robin(), None

    def _make_request(
        self, endpoint: str, data: Dict[str, Any], priority: int = 5, timeout: float = 300.0
    ) -> Any:
        """
        Make HTTP request to selected node with intelligent routing and performance tracking.

        Args:
            endpoint: API endpoint (e.g., '/api/chat')
            data: Request payload
            priority: Request priority (1-10)
            timeout: Request timeout

        Returns:
            Response data

        Raises:
            RuntimeError: If all nodes fail
        """
        # Check cache first (only for non-streaming, deterministic requests)
        cache_key = None
        if self.cache.enabled and not data.get("stream", False):
            cache_key = self.cache.get_cache_key(endpoint, data)
            cached_response = self.cache.get(cache_key)
            if cached_response is not None:
                logger.debug(f"Cache hit for {endpoint} (key={cache_key[:16]}...)")
                return cached_response

        # Track request
        with self._lock:
            self.stats["total_requests"] += 1

        # Try nodes until one succeeds
        errors = []
        routing_decision = None

        for attempt in range(len(self.nodes)):
            # Select node with intelligent routing
            node, decision = self._select_node(payload=data, priority=priority)
            if decision:
                routing_decision = decision

            node_key = f"{node['host']}:{node['port']}"
            url = f"http://{node['host']}:{node['port']}{endpoint}"

            # Track active request (for load balancing)
            with self._lock:
                if node_key in self.stats["node_performance"]:
                    self.stats["node_performance"][node_key]["active_requests"] = (
                        self.stats["node_performance"][node_key].get("active_requests", 0) + 1
                    )

            # Track request start time
            start_time = time.time()

            # Log request to observer
            model = data.get("model", "unknown")
            operation = endpoint.split("/")[-1]  # "chat", "generate", etc.
            log_ollama_request(
                backend=node_key, model=model, operation=operation, priority=priority
            )

            try:
                logger.debug(f"Request to {url}")

                # Use persistent session for connection reuse
                response = self.session.post(url, json=data, timeout=timeout)

                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000

                # Record latency for percentile calculation
                self._latency_buffer.append(latency_ms)

                # Detect VRAM exhaustion (FlockParser pattern)
                vram_exhausted = self.health_monitor.detect_vram_exhaustion(node_key, latency_ms)
                if vram_exhausted:
                    # Mark node as degraded in performance tracking
                    with self._lock:
                        if node_key in self.stats["node_performance"]:
                            self.stats["node_performance"][node_key]["vram_exhausted"] = True

                # Update health baseline
                self.health_monitor.update_baseline(node_key, latency_ms)

                if response.status_code == 200:
                    # Success! Update metrics
                    with self._lock:
                        self.stats["successful_requests"] += 1
                        self.stats["nodes_used"][node_key] = (
                            self.stats["nodes_used"].get(node_key, 0) + 1
                        )

                        # Update node performance metrics
                        perf = self.stats["node_performance"][node_key]
                        perf["total_requests"] += 1

                        # Update running average latency
                        if perf["total_requests"] == 1:
                            perf["latency_ms"] = latency_ms
                        else:
                            perf["latency_ms"] = (
                                perf["latency_ms"] * (perf["total_requests"] - 1) + latency_ms
                            ) / perf["total_requests"]

                        # Update success rate
                        perf["success_rate"] = (
                            perf["total_requests"] - perf["failed_requests"]
                        ) / perf["total_requests"]

                    # Log performance
                    logger.info(
                        f"✅ Request succeeded: {node_key} "
                        f"(latency: {latency_ms:.1f}ms, "
                        f"avg: {self.stats['node_performance'][node_key]['latency_ms']:.1f}ms)"
                    )

                    # Log response to observer
                    log_ollama_response(
                        backend=node_key,
                        model=model,
                        latency_ms=latency_ms,
                        status_code=response.status_code,
                    )

                    # Record performance for router learning
                    if self.router and "model" in data:
                        task_type = (
                            routing_decision.get("task_type", "generation")
                            if routing_decision
                            else "generation"
                        )
                        self.router.record_performance(
                            task_type=task_type, model=data["model"], actual_duration_ms=latency_ms
                        )

                    # Cache successful response
                    response_data = response.json()
                    if cache_key is not None:
                        self.cache.set(cache_key, response_data)

                    return response_data
                else:
                    errors.append(f"{url}: HTTP {response.status_code}")
                    self._record_failure(node_key, latency_ms)

                    # Log error to observer
                    log_ollama_error(
                        backend=node_key,
                        model=model,
                        error=f"HTTP {response.status_code}",
                        latency_ms=latency_ms,
                    )

            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                errors.append(f"{url}: {str(e)}")
                logger.debug(f"Request failed: {e}")
                self._record_failure(node_key, latency_ms)

                # Log error to observer
                log_ollama_error(backend=node_key, model=model, error=str(e), latency_ms=latency_ms)

            finally:
                # Decrement active request counter (always runs)
                with self._lock:
                    if node_key in self.stats["node_performance"]:
                        self.stats["node_performance"][node_key]["active_requests"] = max(
                            0,
                            self.stats["node_performance"][node_key].get("active_requests", 1) - 1,
                        )

        # All nodes failed
        with self._lock:
            self.stats["failed_requests"] += 1

        raise RuntimeError(f"All Ollama nodes failed. Errors: {'; '.join(errors)}")

    def _make_streaming_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        priority: int = 5,
        timeout: float = 300.0,
        node: Optional[Dict[str, Any]] = None,
    ):
        """
        Make streaming HTTP request that yields response chunks.

        Args:
            endpoint: API endpoint (e.g., '/api/chat')
            data: Request payload
            priority: Request priority (1-10)
            timeout: Request timeout
            node: Specific node to use (None = auto-select)

        Yields:
            Response chunks as they arrive

        Raises:
            RuntimeError: If streaming fails
        """
        # Ensure stream is enabled in request
        data["stream"] = True

        # Track request
        with self._lock:
            self.stats["total_requests"] += 1

        # Select or use specified node
        if node is None:
            node, routing_decision = self._select_node(payload=data, priority=priority)
        else:
            routing_decision = None

        node_key = f"{node['host']}:{node['port']}"
        url = f"http://{node['host']}:{node['port']}{endpoint}"

        # Track active request
        with self._lock:
            if node_key in self.stats["node_performance"]:
                self.stats["node_performance"][node_key]["active_requests"] = (
                    self.stats["node_performance"][node_key].get("active_requests", 0) + 1
                )

        # Track request start time
        start_time = time.time()

        # Log request to observer
        model = data.get("model", "unknown")
        operation = endpoint.split("/")[-1]
        log_ollama_request(
            backend=node_key, model=model, operation=operation, priority=priority, streaming=True
        )

        try:
            logger.debug(f"Streaming request to {url}")

            # Handle streaming for httpx vs requests
            if HTTPX_AVAILABLE and isinstance(self.session, httpx.Client):
                # httpx streaming
                with self.session.stream("POST", url, json=data, timeout=timeout) as response:
                    if response.status_code == 200:
                        # Track success
                        with self._lock:
                            self.stats["successful_requests"] += 1
                            self.stats["nodes_used"][node_key] = (
                                self.stats["nodes_used"].get(node_key, 0) + 1
                            )

                        logger.info(f"🌊 Streaming from {node_key}...")

                        # Yield chunks as they arrive
                        for line in response.iter_lines():
                            if line:
                                try:
                                    chunk = json.loads(line)
                                    yield chunk
                                except json.JSONDecodeError as e:
                                    logger.warning(f"Failed to decode streaming chunk: {e}")
                                    continue

                        # Calculate total latency
                        latency_ms = (time.time() - start_time) * 1000

                        # Record latency for percentile calculation
                        self._latency_buffer.append(latency_ms)

                        # Update metrics
                        with self._lock:
                            perf = self.stats["node_performance"][node_key]
                            perf["total_requests"] += 1

                            # Update running average latency
                            if perf["total_requests"] == 1:
                                perf["latency_ms"] = latency_ms
                            else:
                                perf["latency_ms"] = (
                                    perf["latency_ms"] * (perf["total_requests"] - 1) + latency_ms
                                ) / perf["total_requests"]

                            # Update success rate
                            perf["success_rate"] = (
                                perf["total_requests"] - perf["failed_requests"]
                            ) / perf["total_requests"]

                        # Log completion
                        log_ollama_response(
                            backend=node_key,
                            model=model,
                            latency_ms=latency_ms,
                            status_code=response.status_code,
                            streaming=True,
                        )

                        logger.info(f"✅ Stream complete from {node_key} ({latency_ms:.1f}ms)")

                    else:
                        latency_ms = (time.time() - start_time) * 1000
                        self._record_failure(node_key, latency_ms)

                        # Log error
                        log_ollama_error(
                            backend=node_key,
                            model=model,
                            error=f"HTTP {response.status_code}",
                            latency_ms=latency_ms,
                        )

                        raise RuntimeError(f"Streaming failed: HTTP {response.status_code}")
            else:
                # requests (fallback) streaming
                with self.session.post(url, json=data, timeout=timeout, stream=True) as response:
                    if response.status_code == 200:
                        # Track success
                        with self._lock:
                            self.stats["successful_requests"] += 1
                            self.stats["nodes_used"][node_key] = (
                                self.stats["nodes_used"].get(node_key, 0) + 1
                            )

                        logger.info(f"🌊 Streaming from {node_key}...")

                        # Yield chunks as they arrive
                        for line in response.iter_lines():
                            if line:
                                try:
                                    chunk = json.loads(line)
                                    yield chunk
                                except json.JSONDecodeError as e:
                                    logger.warning(f"Failed to decode streaming chunk: {e}")
                                    continue

                        # Calculate total latency
                        latency_ms = (time.time() - start_time) * 1000

                        # Record latency for percentile calculation
                        self._latency_buffer.append(latency_ms)

                        # Update metrics
                        with self._lock:
                            perf = self.stats["node_performance"][node_key]
                            perf["total_requests"] += 1

                            # Update running average latency
                            if perf["total_requests"] == 1:
                                perf["latency_ms"] = latency_ms
                            else:
                                perf["latency_ms"] = (
                                    perf["latency_ms"] * (perf["total_requests"] - 1) + latency_ms
                                ) / perf["total_requests"]

                            # Update success rate
                            perf["success_rate"] = (
                                perf["total_requests"] - perf["failed_requests"]
                            ) / perf["total_requests"]

                        # Log completion
                        log_ollama_response(
                            backend=node_key,
                            model=model,
                            latency_ms=latency_ms,
                            status_code=response.status_code,
                            streaming=True,
                        )

                        logger.info(f"✅ Stream complete from {node_key} ({latency_ms:.1f}ms)")

                    else:
                        latency_ms = (time.time() - start_time) * 1000
                        self._record_failure(node_key, latency_ms)

                        # Log error
                        log_ollama_error(
                            backend=node_key,
                            model=model,
                            error=f"HTTP {response.status_code}",
                            latency_ms=latency_ms,
                        )

                        raise RuntimeError(f"Streaming failed: HTTP {response.status_code}")

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._record_failure(node_key, latency_ms)

            # Log error
            log_ollama_error(backend=node_key, model=model, error=str(e), latency_ms=latency_ms)

            raise RuntimeError(f"Streaming request failed: {e}")

        finally:
            # Decrement active request counter
            with self._lock:
                if node_key in self.stats["node_performance"]:
                    self.stats["node_performance"][node_key]["active_requests"] = max(
                        0, self.stats["node_performance"][node_key].get("active_requests", 1) - 1
                    )

    def _query_node_vram(self, node: Dict[str, str]) -> int:
        """
        Query VRAM for a specific node.

        For localhost: Use nvidia-smi/rocm-smi directly
        For remote: Query via Ollama /api/ps endpoint

        Args:
            node: Node dict with host and port

        Returns:
            Free VRAM in MB, or 0 if unknown
        """
        host = node["host"]
        port = node.get("port", "11434")

        # Check if this is localhost
        if host in ("localhost", "127.0.0.1"):
            # Try local VRAM monitoring (nvidia-smi, rocm-smi, etc.)
            local_vram = self.vram_monitor.get_local_vram_info()
            if local_vram and local_vram.get("free_vram_mb", 0) > 0:
                free_mb = local_vram.get("free_vram_mb", 0)
                # Apply VRAM buffer for safety margin
                free_mb_with_buffer = max(0, free_mb - self.VRAM_BUFFER_MB)
                logger.debug(
                    f"Local VRAM: {free_mb}MB free, {free_mb_with_buffer}MB after {self.VRAM_BUFFER_MB}MB buffer ({local_vram['vendor']})"
                )
                return free_mb_with_buffer
            else:
                # nvidia-smi failed or not installed (old GPUs, incompatible drivers)
                # Fall back to querying Ollama /api/ps for localhost
                logger.debug("nvidia-smi unavailable for localhost, falling back to Ollama /api/ps")
                # Fall through to remote query logic below (works for localhost too)

        # Remote node - query via Ollama API
        try:
            url = f"http://{host}:{port}/api/ps"
            response = self.session.get(url, timeout=2.0)

            if response.status_code == 200:
                ps_data = response.json()
                # Parse GPU info from /api/ps response
                free_mb = self._parse_vram_from_ps(ps_data)
                logger.debug(f"Remote {host}:{port} VRAM: {free_mb}MB free")
                return free_mb
            else:
                logger.debug(f"Remote {host}:{port} /api/ps returned {response.status_code}")
                return 0

        except Exception as e:
            logger.debug(f"Failed to query VRAM for {host}:{port}: {e}")
            return 0

    def _estimate_gpu_vram_from_model(self) -> int:
        """
        Estimate GPU VRAM capacity from GPU model name (for GPUs without nvidia-smi).

        Uses lspci detection from VRAMMonitor to get GPU model, then looks up
        known VRAM capacity. Critical for old GPUs where nvidia-smi doesn't work.

        Returns:
            Estimated VRAM in MB, or 0 if unknown
        """
        # GPU Model → VRAM (MB) lookup table for common GPUs without nvidia-smi support
        GPU_VRAM_TABLE = {
            # NVIDIA GTX 10 series (mostly have nvidia-smi, but some old drivers don't)
            "GTX 1050": 2048,  # 2GB
            "GTX 1050 Ti": 4096,  # 4GB
            "GTX 1060": 6144,  # 6GB (also 3GB variant exists)
            "GTX 1070": 8192,  # 8GB
            "GTX 1080": 8192,  # 8GB
            "GTX 1080 Ti": 11264,  # 11GB
            # NVIDIA GTX 900 series
            "GTX 950": 2048,  # 2GB
            "GTX 960": 2048,  # 2GB (also 4GB variant)
            "GTX 970": 4096,  # 4GB
            "GTX 980": 4096,  # 4GB
            "GTX 980 Ti": 6144,  # 6GB
            # NVIDIA GTX 700 series
            "GTX 750": 1024,  # 1GB (also 2GB variant)
            "GTX 750 Ti": 2048,  # 2GB
            "GTX 760": 2048,  # 2GB
            "GTX 770": 2048,  # 2GB
            "GTX 780": 3072,  # 3GB
            "GTX 780 Ti": 3072,  # 3GB
            # NVIDIA GTX 600 series
            "GTX 650": 1024,  # 1GB
            "GTX 660": 2048,  # 2GB
            "GTX 670": 2048,  # 2GB
            "GTX 680": 2048,  # 2GB
            "GTX 690": 4096,  # 4GB (2GB per GPU, dual GPU)
            # NVIDIA GTX 500 series and older (nvidia-smi often unavailable)
            "GTX 580": 1536,  # 1.5GB
            "GTX 570": 1280,  # 1.25GB
            "GTX 560": 1024,  # 1GB
            "GTX 550 Ti": 1024,  # 1GB
            # Quadro cards (older ones)
            "Quadro K620": 2048,
            "Quadro K1200": 4096,
            "Quadro K2200": 4096,
            "Quadro P400": 2048,
            "Quadro P1000": 4096,
            "Quadro P2000": 5120,
        }

        # Try to get GPU info from VRAMMonitor
        local_vram = self.vram_monitor.get_local_vram_info()
        if not local_vram or local_vram.get("total_vram_mb", 0) == 0:
            # nvidia-smi failed, check if we have GPU names from lspci fallback
            if local_vram and "gpus" in local_vram:
                for gpu in local_vram["gpus"]:
                    gpu_name = gpu.get("name", "")

                    # Try to match GPU model in lookup table
                    for model_key, vram_mb in GPU_VRAM_TABLE.items():
                        if model_key.upper() in gpu_name.upper():
                            logger.info(
                                f"Estimated VRAM for {gpu_name}: {vram_mb}MB (from model lookup)"
                            )
                            return vram_mb

        # Unknown GPU model, return 0 (will use conservative fallback)
        return 0

    def _parse_vram_from_ps(self, ps_data: Dict) -> int:
        """
        Parse VRAM info from Ollama /api/ps response.

        For GPUs where nvidia-smi doesn't work (old drivers, incompatible versions),
        this calculates VRAM based on loaded models + estimated GPU capacity.

        Args:
            ps_data: Response from /api/ps

        Returns:
            Estimated free VRAM in MB
        """
        try:
            # Ollama /api/ps format:
            # {
            #   "models": [
            #     {
            #       "name": "llama3.1:8b",
            #       "size": 4661211648,  # bytes
            #       "size_vram": 4661211648
            #     }
            #   ]
            # }

            models = ps_data.get("models", [])

            # Sum up VRAM used by loaded models
            # Also detect if GPU is actually being used (not just available)
            total_vram_used_mb = 0
            has_gpu_loaded_models = False
            for model in models:
                size_vram_bytes = model.get("size_vram", 0)
                total_vram_used_mb += size_vram_bytes / (1024 * 1024)
                if size_vram_bytes > 0:
                    has_gpu_loaded_models = True

            # Determine total VRAM capacity (priority order):
            # 1. From nvidia-smi (if available)
            # 2. From GPU model lookup table (for old GPUs)
            # 3. Conservative fallback based on loaded models

            local_vram = self.vram_monitor.get_local_vram_info()
            if local_vram and local_vram.get("total_vram_mb", 0) > 0:
                # nvidia-smi worked - use accurate total
                total_vram_mb = local_vram.get("total_vram_mb", 0)
                has_actual_gpu = True
            else:
                # nvidia-smi failed - try model-based estimation
                estimated_vram = self._estimate_gpu_vram_from_model()
                if estimated_vram > 0:
                    total_vram_mb = estimated_vram
                    has_actual_gpu = True
                elif total_vram_used_mb > 0:
                    # Model is loaded but we don't know GPU capacity (remote node without nvidia-smi)
                    # Assume at least 8GB for modern GPUs (most common: 3060/3070/4060/4070)
                    # If model uses more than 8GB, assume 2GB headroom above that
                    min_reasonable_vram = 8192  # 8GB minimum
                    total_vram_mb = max(min_reasonable_vram, total_vram_used_mb + 2048)
                    has_actual_gpu = True
                    logger.debug(
                        f"Unknown GPU capacity, assuming {total_vram_mb:.0f}MB (used: {total_vram_used_mb:.0f}MB)"
                    )
                else:
                    # Nothing loaded and unknown GPU
                    has_actual_gpu = False
                    total_vram_mb = 0

            # IMPORTANT: Only return 0 if we're certain there's no GPU hardware
            # If GPU hardware exists but no generation models loaded (only embeddings),
            # still return available VRAM so node can be considered for routing
            if not has_actual_gpu and not has_gpu_loaded_models:
                logger.debug("No GPU hardware detected, treating as CPU-only")
                return 0

            # If GPU hardware exists but no models loaded, assume all VRAM is free
            if has_actual_gpu and not has_gpu_loaded_models:
                if total_vram_mb == 0:
                    # GPU exists but capacity unknown - assume 6GB (common for entry-level GPUs)
                    total_vram_mb = 6144
                    logger.debug(
                        f"GPU detected with no loaded models, assuming {total_vram_mb}MB capacity"
                    )
                free_vram_mb_with_buffer = total_vram_mb - self.VRAM_BUFFER_MB
                logger.debug(
                    f"GPU hardware present, {free_vram_mb_with_buffer}MB available (no generation models loaded)"
                )
                return int(max(0, free_vram_mb_with_buffer))

            # Calculate free VRAM with safety buffer
            free_vram_mb = max(0, total_vram_mb - total_vram_used_mb)
            # Apply VRAM buffer for safety margin to prevent OOM
            free_vram_mb_with_buffer = max(0, free_vram_mb - self.VRAM_BUFFER_MB)

            return int(free_vram_mb_with_buffer)

        except Exception as e:
            logger.debug(f"Failed to parse VRAM from /api/ps: {e}")
            return 0

    def _query_loaded_models(self, node: Dict[str, Any]) -> List[str]:
        """
        Query which models are currently loaded on a node via /api/ps.

        This helps avoid cold model load times by preferring nodes that already
        have the target model loaded in VRAM.

        Args:
            node: Node dictionary with 'host' and 'port'

        Returns:
            List of loaded model names (e.g., ['llama3.1:8b', 'mxbai-embed-large'])
        """
        try:
            url = f"http://{node['host']}:{node['port']}/api/ps"
            response = self.session.get(url, timeout=2)

            if response.status_code != 200:
                return []

            data = response.json()
            models = data.get("models", [])

            # Extract model names from running models
            loaded_model_names = []
            for model in models:
                model_name = model.get("name", "")
                if model_name:
                    # Normalize model name (e.g., "llama3.1:8b" or "mxbai-embed-large:latest")
                    loaded_model_names.append(model_name)

            return loaded_model_names

        except Exception as e:
            logger.debug(f"Failed to query loaded models from {node['host']}:{node['port']}: {e}")
            return []

    def _auto_register_with_dashboard(self):
        """
        Auto-register with SOLLOL dashboard if one is running.

        This provides automatic observability without requiring manual DashboardClient setup.
        Checks for dashboard on default port (8080) and registers silently if found.
        """
        try:
            import socket

            from .dashboard_client import DashboardClient

            # Check if dashboard is running (quick timeout to avoid blocking startup)
            dashboard_url = "http://localhost:8080"
            test_response = self.session.get(f"{dashboard_url}/api/applications", timeout=0.5)

            if test_response.status_code == 200:
                # Dashboard is running, auto-register
                hostname = socket.gethostname()
                # Use custom app_name if provided, otherwise default to "OllamaPool (hostname)"
                app_name = self.app_name or f"OllamaPool ({hostname})"

                # Auto-discover RPC backends for metadata
                try:
                    from .rpc_discovery import auto_discover_rpc_backends

                    rpc_backends = auto_discover_rpc_backends()
                    self._rpc_backends = rpc_backends
                except Exception:
                    self._rpc_backends = []

                self._dashboard_client = DashboardClient(
                    app_name=app_name,
                    router_type="OllamaPool",
                    version="0.9.46",
                    dashboard_url=dashboard_url,
                    metadata={
                        "nodes": len(self.nodes),
                        "intelligent_routing": self.enable_intelligent_routing,
                        "vram_monitoring": True,
                        "health_checks": True,
                        "gpu_type": self.vram_monitor.gpu_type,
                        "model_sharding": len(self._rpc_backends) > 0,  # Simple boolean indicator
                        "rpc_backends": len(self._rpc_backends) if self._rpc_backends else None,
                    },
                    auto_register=True,
                )
                logger.info(f"✅ Auto-registered with SOLLOL dashboard at {dashboard_url}")
        except (ImportError, Exception) as e:
            # Dashboard not running or not available - silent failure is fine
            logger.debug(f"Dashboard auto-registration skipped: {e}")
            self._dashboard_client = None
            self._rpc_backends = []

    def _refresh_vram_loop(self):
        """Background thread to periodically refresh VRAM data."""
        logger.debug("VRAM monitoring thread started")

        while self._vram_refresh_enabled:
            try:
                time.sleep(self._vram_refresh_interval)

                # Refresh VRAM for all nodes
                with self._lock:
                    for node in self.nodes:
                        node_key = f"{node['host']}:{node['port']}"

                        if node_key in self.stats["node_performance"]:
                            # Query current VRAM
                            gpu_free_mem = self._query_node_vram(node)

                            # Query loaded models from /api/ps to avoid cold loads
                            loaded_models = self._query_loaded_models(node)

                            # Update metadata
                            old_vram = self.stats["node_performance"][node_key].get(
                                "gpu_free_mem", 0
                            )
                            self.stats["node_performance"][node_key]["gpu_free_mem"] = gpu_free_mem
                            self.stats["node_performance"][node_key][
                                "loaded_models"
                            ] = loaded_models

                            # Log significant changes
                            if abs(gpu_free_mem - old_vram) > 1000:  # >1GB change
                                logger.info(
                                    f"🔄 VRAM changed on {node_key}: "
                                    f"{old_vram}MB → {gpu_free_mem}MB"
                                )

            except Exception as e:
                logger.error(f"VRAM refresh loop error: {e}")

        logger.debug("VRAM monitoring thread stopped")

    def _get_adaptive_health_interval(self, node_key: str) -> int:
        """
        Calculate adaptive health check interval based on node stability.

        Args:
            node_key: Node identifier

        Returns:
            Health check interval in seconds
        """
        if not self._adaptive_health_checks:
            return self._health_check_base_interval

        perf_data = self.stats["node_performance"].get(node_key, {})
        total_requests = perf_data.get("total_requests", 0)
        failed_requests = perf_data.get("failed_requests", 0)

        if total_requests == 0:
            return self._health_check_base_interval

        failure_rate = failed_requests / total_requests

        # Adaptive intervals based on stability
        if failure_rate < 0.01:  # Very stable (<1% failures)
            return 60  # Check every 60s
        elif failure_rate < 0.05:  # Stable (<5% failures)
            return self._health_check_base_interval  # Check every 30s
        elif failure_rate < 0.15:  # Degraded (5-15% failures)
            return 15  # Check every 15s
        else:  # Unstable (>15% failures)
            return 5  # Check every 5s

    def _health_check_loop(self):
        """Background thread to periodically check node health for live dashboard updates."""
        logger.debug("Health check monitoring thread started")

        # Track last check request count per node (check every N requests instead of time-based)
        last_check_request_counts = {}
        health_check_interval_requests = 100  # Check every 100 requests per node
        interval = 5  # Base loop interval in seconds

        while self._health_check_enabled:
            try:
                time.sleep(interval)  # Base loop interval

                # Check health of all nodes based on request count
                with self._lock:
                    for node in self.nodes:
                        node_key = f"{node['host']}:{node['port']}"

                        if node_key not in self.stats["node_performance"]:
                            continue

                        # Get current request count for this node
                        current_requests = self.stats["node_performance"][node_key].get(
                            "total_requests", 0
                        )
                        last_check_requests = last_check_request_counts.get(node_key, 0)

                        # Skip if not enough requests have happened (check every 100 requests)
                        if current_requests - last_check_requests < health_check_interval_requests:
                            continue

                        last_check_request_counts[node_key] = current_requests

                        # Get current performance data
                        perf_data = self.stats["node_performance"][node_key]

                        # Ping node with lightweight /api/tags request
                        start_time = time.time()
                        try:
                            url = f"http://{node['host']}:{node['port']}/api/tags"
                            response = self.session.get(url, timeout=2)
                            latency_ms = (time.time() - start_time) * 1000

                            if response.status_code == 200:
                                # Check for status change
                                old_status = (
                                    "offline" if not perf_data.get("available", True) else "healthy"
                                )
                                new_status = "healthy"

                                # Update health status
                                self.stats["node_performance"][node_key]["available"] = True
                                self.stats["node_performance"][node_key]["latency_ms"] = latency_ms

                                # Update health monitor baseline
                                self.health_monitor.update_baseline(
                                    node_key, latency_ms, is_gpu=True
                                )

                                # Log health check to observer
                                log_node_health_check(
                                    backend=node_key, status=new_status, latency_ms=latency_ms
                                )

                                # Log to InfluxDB time-series metrics
                                log_node_health(
                                    node_url=f"http://{node['host']}:{node['port']}",
                                    healthy=True,
                                    latency_ms=latency_ms,
                                    models_loaded=len(node.get("models", [])),
                                    vram_free_mb=perf_data.get("gpu_free_mem", 0),
                                    vram_total_mb=perf_data.get("gpu_total_mem", 0),
                                    failure_count=0,
                                )

                                # Log status change if needed
                                if old_status != new_status:
                                    log_node_status_change(
                                        backend=node_key,
                                        old_status=old_status,
                                        new_status=new_status,
                                    )

                                logger.debug(
                                    f"✓ Health check {node_key}: {latency_ms:.0f}ms (interval={interval}s)"
                                )
                            else:
                                # Check for status change
                                old_status = (
                                    "healthy" if perf_data.get("available", True) else "offline"
                                )
                                new_status = "offline"

                                # Node returned error
                                self.stats["node_performance"][node_key]["available"] = False

                                # Log health check failure to observer
                                log_node_health_check(
                                    backend=node_key,
                                    status=new_status,
                                    latency_ms=latency_ms,
                                    error=f"HTTP {response.status_code}",
                                )

                                # Log to InfluxDB time-series metrics
                                log_node_health(
                                    node_url=f"http://{node['host']}:{node['port']}",
                                    healthy=False,
                                    latency_ms=latency_ms,
                                    failure_count=perf_data.get("failure_count", 0),
                                )

                                # Log status change if needed
                                if old_status != new_status:
                                    log_node_status_change(
                                        backend=node_key,
                                        old_status=old_status,
                                        new_status=new_status,
                                    )

                                logger.warning(
                                    f"⚠️  Health check {node_key}: HTTP {response.status_code}"
                                )

                        except (TimeoutError, Exception) as e:
                            # Node timed out or unreachable
                            self.stats["node_performance"][node_key]["available"] = False
                            error_msg = (
                                "timeout"
                                if isinstance(e, TimeoutError) or "timeout" in str(e).lower()
                                else f"unreachable ({e})"
                            )
                            logger.warning(f"⚠️  Health check {node_key}: {error_msg}")

            except Exception as e:
                logger.error(f"Health check loop error: {e}")

        logger.debug("Health check monitoring thread stopped")

    def _publish_metrics_loop(self):
        """Background thread to publish aggregated metrics to Redis every 5 seconds."""
        logger.debug("Metrics publisher thread started")

        while True:
            try:
                if self._metrics_redis_client and len(self._latency_buffer) > 0:
                    import numpy as np

                    # Get latency snapshot
                    latencies = list(self._latency_buffer)

                    # Calculate percentiles
                    p50 = float(np.percentile(latencies, 50))
                    p95 = float(np.percentile(latencies, 95))
                    p99 = float(np.percentile(latencies, 99))

                    # Calculate success rate
                    with self._lock:
                        success_rate = self.stats["successful_requests"] / max(
                            1, self.stats["total_requests"]
                        )

                    # Build metrics payload (wrapped in "metrics" key for dashboard compatibility)
                    payload = {
                        "metrics": {
                            "analytics": {
                                "p50_latency_ms": p50,
                                "p95_latency_ms": p95,
                                "p99_latency_ms": p99,
                                "success_rate": success_rate,
                            },
                            "total_pools": len(self.nodes),
                        }
                    }

                    # Publish to Redis (30s TTL)
                    self._metrics_redis_client.setex(
                        "sollol:router:metadata", 30, json.dumps(payload)
                    )

                    logger.debug(
                        f"📊 Published metrics: p50={p50:.1f}ms, p95={p95:.1f}ms, "
                        f"p99={p99:.1f}ms, success={success_rate:.2%}"
                    )

                time.sleep(5)

            except Exception as e:
                logger.debug(f"Metrics publishing error: {e}")
                time.sleep(5)

        logger.debug("Metrics publisher thread stopped")

    def _record_failure(self, node_key: str, latency_ms: float):
        """Record a failed request for a node."""
        with self._lock:
            if node_key in self.stats["node_performance"]:
                perf = self.stats["node_performance"][node_key]
                perf["failed_requests"] += 1
                perf["total_requests"] += 1

                # Update success rate
                if perf["total_requests"] > 0:
                    perf["success_rate"] = (
                        perf["total_requests"] - perf["failed_requests"]
                    ) / perf["total_requests"]

                logger.warning(
                    f"❌ Request failed: {node_key} " f"(success_rate: {perf['success_rate']:.1%})"
                )

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        priority: int = 5,
        **kwargs,
    ):
        """
        Chat completion with intelligent routing and observability.

        Args:
            model: Model name (e.g., "llama3.2")
            messages: Chat messages
            stream: Stream response token-by-token (default: False)
            priority: Request priority 1-10 (default: 5)
            **kwargs: Additional Ollama parameters

        Returns:
            If stream=False: Chat response dict
            If stream=True: Generator yielding response chunks

        Example (streaming):
            for chunk in pool.chat("llama2", messages, stream=True):
                print(chunk.get("message", {}).get("content", ""), end="")
        """
        data = {"model": model, "messages": messages, "stream": stream, **kwargs}

        if stream:
            return self._make_streaming_request("/api/chat", data, priority=priority)
        else:
            return self._make_request("/api/chat", data, priority=priority)

    def generate(self, model: str, prompt: str, stream: bool = False, priority: int = 5, **kwargs):
        """
        Generate text with intelligent routing and observability.

        Args:
            model: Model name
            prompt: Text prompt
            stream: Stream response token-by-token (default: False)
            priority: Request priority 1-10 (default: 5)
            **kwargs: Additional Ollama parameters

        Returns:
            If stream=False: Generation response dict
            If stream=True: Generator yielding response chunks

        Example (streaming):
            for chunk in pool.generate("llama2", "Tell me a story", stream=True):
                print(chunk.get("response", ""), end="")
        """
        data = {"model": model, "prompt": prompt, "stream": stream, **kwargs}

        if stream:
            return self._make_streaming_request("/api/generate", data, priority=priority)
        else:
            return self._make_request("/api/generate", data, priority=priority)

    def embed(self, model: str, input: str, priority: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Generate embeddings with intelligent routing and observability.

        Args:
            model: Embedding model name
            input: Text to embed
            priority: Request priority 1-10 (default: 5)
            **kwargs: Additional Ollama parameters

        Returns:
            Embedding response dict
        """
        data = {"model": model, "input": input, **kwargs}

        return self._make_request("/api/embed", data, priority=priority)

    def _embed_batch_sequential(
        self, model: str, inputs: List[str], node_key: str, priority: int = 5, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        FlockParser Legacy's high-performance sequential batch embedding with connection reuse.

        Uses a single persistent Ollama client for ~12x speedup over individual requests:
        - Zero connection overhead (client reused for all embeddings)
        - No lock contention during embedding loop
        - Minimal HTTP overhead

        This is the pattern that made FlockParser Legacy fast for distributed CPU loads.

        Args:
            model: Embedding model name
            inputs: List of texts to embed
            node_key: Node to use (format: "host:port")
            priority: Request priority (not used in sequential mode)
            **kwargs: Additional Ollama parameters

        Returns:
            List of embedding responses in same order as inputs
        """
        import ollama

        batch_size = len(inputs)
        if batch_size == 0:
            return []

        # Parse node_key
        if ":" in node_key:
            host, port = node_key.rsplit(":", 1)
        else:
            logger.error(f"Invalid node_key format: {node_key}, expected 'host:port'")
            return [None] * batch_size

        node_url = f"http://{host}:{port}"

        logger.info(
            f"➡️  Sequential mode: Processing {batch_size} embeddings on {node_key} with connection reuse"
        )

        # Create Ollama client ONCE for connection reuse (FlockParser Legacy pattern!)
        client = ollama.Client(host=node_url)

        # Process batch with NO LOCKS during loop (maximum performance)
        results = []
        start_time = time.time()
        completed = 0
        errors = 0

        for i, text in enumerate(inputs):
            try:
                # Use pre-created client for efficiency - connection is reused!
                result = client.embed(model=model, input=text, **kwargs)
                results.append(result)
                completed += 1

                # Progress logging every 50 items
                if (i + 1) % 50 == 0 or (i + 1) == batch_size:
                    progress_pct = ((i + 1) * 100) // batch_size
                    logger.info(f"   Progress: {i + 1}/{batch_size} embeddings ({progress_pct}%)")
            except Exception as e:
                logger.error(f"Error embedding text {i}: {e}")
                results.append(None)
                errors += 1

        # Update stats ONCE at end (minimal lock usage - FlockParser Legacy pattern)
        total_time = time.time() - start_time
        with self._lock:
            self.stats["successful_requests"] += completed
            self.stats["failed_requests"] += errors
            self.stats["nodes_used"][node_key] = (
                self.stats["nodes_used"].get(node_key, 0) + batch_size
            )

            # Update node performance
            if node_key in self.stats["node_performance"]:
                perf = self.stats["node_performance"][node_key]
                perf["total_requests"] += batch_size
                perf["failed_requests"] += errors

        avg_time_per_embedding = (total_time / batch_size * 1000) if batch_size > 0 else 0
        logger.info(
            f"✅ Sequential batch complete: {completed}/{batch_size} embeddings successful "
            f"in {total_time:.2f}s ({avg_time_per_embedding:.1f}ms/embedding) on {node_key}"
        )

        return results

    def embed_batch(
        self,
        model: str,
        inputs: List[str],
        max_workers: Optional[int] = None,
        priority: int = 5,
        use_adaptive: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for multiple texts in parallel across nodes.

        This method provides 10-12x speedup over serial processing by:
        - Parallelizing requests across multiple worker threads
        - Distributing load across available nodes
        - Adaptive parallelism strategy (sequential vs parallel based on cluster state)

        Args:
            model: Embedding model name
            inputs: List of texts to embed
            max_workers: Number of parallel workers (auto-calculated if None)
            priority: Request priority 1-10 (default: 5)
            use_adaptive: Use adaptive parallelism strategy (default: True)
            **kwargs: Additional Ollama parameters

        Returns:
            List of embedding responses in same order as inputs

        Example:
            texts = ["chunk 1", "chunk 2", "chunk 3"]
            embeddings = pool.embed_batch("mxbai-embed-large", texts)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from .adaptive_parallelism import AdaptiveParallelismStrategy

        batch_size = len(inputs)
        if batch_size == 0:
            return []

        # Use adaptive parallelism strategy if enabled
        should_parallel = True  # Default: use parallel
        if use_adaptive:
            strategy = AdaptiveParallelismStrategy(pool=self)
            should_parallel, reasoning = strategy.should_parallelize(batch_size, model)

            # Log decision
            logger.info(f"🔀 Adaptive strategy: {reasoning['reason']} - {reasoning['detail']}")

            # If sequential mode recommended, use FlockParser Legacy's connection reuse pattern
            if not should_parallel:
                recommended_node = reasoning.get("recommended_node")
                if recommended_node:
                    return self._embed_batch_sequential(
                        model, inputs, recommended_node, priority, **kwargs
                    )

            # Auto-calculate optimal workers if not specified (parallel mode)
            if max_workers is None:
                max_workers = strategy.get_optimal_workers(batch_size)
                logger.info(
                    f"🔧 Adaptive parallelism chose {max_workers} workers for batch_size={batch_size}, nodes={len(self.nodes)}"
                )
        else:
            # Default behavior without adaptive strategy
            if max_workers is None:
                # Use 2 workers per node, capped at batch size
                max_workers = min(len(self.nodes) * 2, batch_size)
                max_workers = max(1, max_workers)  # At least 1 worker

            # IMPORTANT: If max_workers=1, use sequential mode with connection reuse
            # This provides ~5-12x speedup over parallel with 1 worker due to:
            # - Reused ollama.Client() (no connection overhead)
            # - Single lock acquisition (minimal contention)
            # - No ThreadPoolExecutor overhead
            #
            # Note: Sequential on one node is ALWAYS better than parallel with 1 worker,
            # even if multiple nodes are available. If you want multi-node parallelism,
            # set max_workers > 1 or use adaptive mode.
            if max_workers == 1:
                # Use first available node for sequential processing
                node_key = f"{self.nodes[0]['host']}:{self.nodes[0]['port']}"
                return self._embed_batch_sequential(model, inputs, node_key, priority, **kwargs)

        # Adaptive routing: Use Dask for VERY large batches (>10000), ThreadPoolExecutor otherwise
        # Dask has significant overhead (worker spawn, serialization) that only pays off for massive batches
        # For medium batches (100-10000), ThreadPoolExecutor with connection reuse is much faster
        use_dask = self.dask_client and batch_size > 10000

        if use_dask:
            logger.info(
                f"🚀 Batch embedding {batch_size} texts with Dask (distributed) across cluster"
            )
        else:
            logger.info(
                f"🚀 Batch embedding {batch_size} texts with {max_workers} workers (local threads) across {len(self.nodes)} nodes"
            )

        results = [None] * batch_size
        completed = 0

        def embed_single(index: int, text: str):
            """Embed single text with error handling."""
            try:
                result = self.embed(model, text, priority=priority, **kwargs)
                return index, result, None
            except Exception as e:
                return index, None, e

        # Use Dask for distributed processing if available
        if use_dask:
            logger.info(f"⚡ Using Dask distributed processing across cluster")

            # Prepare task data
            tasks = [(i, text) for i, text in enumerate(inputs)]

            # Use functools.partial to bind parameters for the module-level function
            from functools import partial

            task_func = partial(_dask_embed_task, self.nodes, model, priority=priority, **kwargs)

            # Use client.map for better performance
            futures = self.dask_client.map(task_func, tasks)

            # Collect results as they complete
            from distributed import as_completed as dask_as_completed

            for future in dask_as_completed(futures):
                try:
                    index, result, error = future.result()
                    completed += 1

                    # Show progress: every 5 for small batches (<50), every 50 for large batches
                    progress_interval = 5 if batch_size < 50 else 50
                    if completed % progress_interval == 0 or completed == batch_size:
                        progress_pct = (completed * 100) // batch_size
                        logger.info(
                            f"   Progress: {completed}/{batch_size} embeddings ({progress_pct}%)"
                        )

                    if error:
                        logger.error(f"⚠️  Error embedding text {index}: {error}")
                    else:
                        results[index] = result
                except Exception as e:
                    logger.error(f"⚠️  Dask task error: {e}")
                    completed += 1

            # Count successful embeddings
            success_count = sum(1 for r in results if r is not None)
            logger.info(
                f"✅ Dask batch complete: {success_count}/{batch_size} embeddings successful"
            )

            return results

        # Use optimized batching: split chunks across nodes, use connection reuse per node
        # This provides ~10-12x speedup over individual requests by eliminating per-request overhead
        import sys

        num_nodes = len(self.nodes)

        # Calculate performance-weighted distribution (not simple 50/50!)
        # Faster nodes get more work so all nodes finish at same time
        node_weights = []
        node_throughputs = []
        for node in self.nodes:
            node_key = f"{node['host']}:{node['port']}"
            perf = self.stats.get("node_performance", {}).get(node_key, {})

            # Use actual measured batch throughput as weight (chunks/sec)
            throughput = perf.get("batch_throughput", 0.5)  # Default 0.5 chunks/sec
            weight = max(throughput, 0.1)  # Ensure minimum weight
            node_weights.append(weight)
            node_throughputs.append(throughput)

        # DEBUG: Show current performance stats
        node_keys = [f"{n['host']}:{n['port']}" for n in self.nodes]
        print(
            f"   📊 DEBUG: Current stats: {self.stats.get('node_performance', {})}", file=sys.stderr
        )
        print(
            f"   📊 DEBUG: Node throughputs: {list(zip(node_keys, node_throughputs))}",
            file=sys.stderr,
        )

        # ADAPTIVE PARALLELISM DECISION (from FlockParser-legacy)
        # If one node is 5x+ faster, route ALL work to it (sequential mode)
        # This avoids slow node bottlenecks and achieves better total throughput
        fastest_throughput = max(node_throughputs)
        slowest_throughput = min(node_throughputs)
        speed_ratio = fastest_throughput / max(slowest_throughput, 0.01)

        # CASE 1: Dominant node (10x+ faster) - Use sequential on fastest
        # Lowered from 5.0 to 10.0 to allow work sharing with moderate speed differences
        # At 5-10x difference, weighted distribution is often better (both nodes finish together)
        if speed_ratio >= 10.0 and num_nodes > 1:
            fastest_idx = node_throughputs.index(fastest_throughput)
            fastest_node_key = (
                f"{self.nodes[fastest_idx]['host']}:{self.nodes[fastest_idx]['port']}"
            )
            print(
                f"   🎯 ADAPTIVE MODE: Sequential (speed ratio {speed_ratio:.1f}x)",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"   🚀 Routing ALL {batch_size} chunks to fastest node: {fastest_node_key}",
                file=sys.stderr,
                flush=True,
            )

            # Route everything to fastest node
            node_weights = [0.0] * num_nodes
            node_weights[fastest_idx] = 1.0
        # CASE 2: Small batch (<20 items) - Sequential on fastest
        elif batch_size < 20 and num_nodes > 1:
            fastest_idx = node_throughputs.index(fastest_throughput)
            fastest_node_key = (
                f"{self.nodes[fastest_idx]['host']}:{self.nodes[fastest_idx]['port']}"
            )
            print(
                f"   🎯 ADAPTIVE MODE: Sequential (batch too small: {batch_size} < 20)",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"   🚀 Routing ALL {batch_size} chunks to fastest node: {fastest_node_key}",
                file=sys.stderr,
                flush=True,
            )

            # Route everything to fastest node
            node_weights = [0.0] * num_nodes
            node_weights[fastest_idx] = 1.0
        else:
            # CASE 3: Balanced cluster or medium difference - Use parallel
            print(
                f"   🔀 ADAPTIVE MODE: Parallel (speed ratio {speed_ratio:.1f}x, batch {batch_size})",
                file=sys.stderr,
                flush=True,
            )

        # Normalize weights to percentages
        total_weight = sum(node_weights)
        if total_weight > 0:
            node_percentages = [w / total_weight for w in node_weights]
        else:
            # No performance data yet, use equal distribution
            node_percentages = [1.0 / num_nodes for _ in range(num_nodes)]

        # PER-NODE QUEUES with cross-node work stealing when done
        # Each node gets its own queue, workers share their node's queue
        # When a node finishes, its workers help other nodes
        import queue as queue_module
        import threading

        print(
            f"   ⚡ Per-node queues with cross-node stealing: {batch_size} chunks",
            file=sys.stderr,
            flush=True,
        )

        # Distribute chunks to per-node queues based on weights
        chunks_per_node = [[] for _ in range(num_nodes)]
        indices_per_node = [[] for _ in range(num_nodes)]

        for idx, text in enumerate(inputs):
            ratio = (idx / len(inputs)) if len(inputs) > 0 else 0
            cumulative = 0.0
            node_idx = 0
            for i, pct in enumerate(node_percentages):
                cumulative += pct
                if ratio < cumulative:
                    node_idx = i
                    break
            chunks_per_node[node_idx].append((idx, text))
            indices_per_node[node_idx].append(idx)

        # Create per-node queues
        node_queues = []
        for i, node in enumerate(self.nodes):
            node_queue = queue_module.Queue()
            for idx, text in chunks_per_node[i]:
                node_queue.put((idx, text))
            node_queues.append(node_queue)

        # Log ESTIMATED distribution
        import sys

        for i, node in enumerate(self.nodes):
            node_key = f"{node['host']}:{node['port']}"
            node_perf = self.stats.get("node_performance", {}).get(node_key, {})
            throughput = node_perf.get("batch_throughput", 0.5)
            avg_time = node_perf.get("avg_response_time", 1.0)
            workers_for_node = 2 if avg_time < 2.5 else 1
            print(
                f"   ⚖️  {node_key}: ~{len(chunks_per_node[i])} chunks target ({len(chunks_per_node[i])*100//batch_size}% - {throughput:.2f} chunks/s, {workers_for_node} workers)",
                file=sys.stderr,
                flush=True,
            )
        sys.stderr.flush()

        # Shared results array and progress tracking
        results = [None] * batch_size
        results_lock = threading.Lock()
        completed_count = [0]

        # Calculate workers per node based on performance
        workers_per_node_list = []
        node_speeds = []  # Track relative speed for each node

        for node in self.nodes:
            node_key = f"{node['host']}:{node['port']}"
            node_perf = self.stats.get("node_performance", {}).get(node_key, {})
            avg_time = node_perf.get("avg_response_time", None)
            throughput = node_perf.get("batch_throughput", 0.5)

            # Determine workers for this node
            if avg_time is None or avg_time < 2.5:
                workers = 2  # Fast node
            else:
                workers = 1  # Slow node

            workers_per_node_list.append(workers)
            node_speeds.append(throughput)

        # (No longer spawning workers - using async instead)

        def async_batch_submit_worker(node_idx: int):
            """Submit ALL chunks for this node async, collect responses as they complete"""
            import asyncio
            import time as time_module

            import httpx

            node = self.nodes[node_idx]
            node_key = f"{node['host']}:{node['port']}"
            node_url = f"http://{node['host']}:{node['port']}"

            my_chunks = chunks_per_node[node_idx]
            if not my_chunks:
                return

            print(
                f"      🚀 {node_key} submitting {len(my_chunks)} chunks async",
                file=sys.stderr,
                flush=True,
            )

            async def submit_all_async():
                """Fire off ALL requests immediately, collect as they complete"""
                async with httpx.AsyncClient(timeout=300.0) as client:

                    async def embed_one(idx, text):
                        """Single async embed request"""
                        try:
                            response = await client.post(
                                f"{node_url}/api/embed",
                                json={"model": model, "input": text, **kwargs},
                            )
                            response.raise_for_status()
                            return idx, response.json(), None
                        except Exception as e:
                            return idx, None, e

                    # Fire off ALL requests at once!
                    tasks = [embed_one(idx, text) for idx, text in my_chunks]

                    # Collect responses as they complete (any order)
                    completed = 0
                    for coro in asyncio.as_completed(tasks):
                        idx, result, error = await coro

                        with results_lock:
                            if error:
                                print(
                                    f"      ❌ {node_key} chunk {idx} error: {error}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                            else:
                                results[idx] = result

                            completed_count[0] += 1
                            completed += 1
                            current_progress = completed_count[0]

                        # Progress every 50 chunks
                        if current_progress % 50 == 0 or current_progress == batch_size:
                            progress_pct = (current_progress * 100) // batch_size
                            print(
                                f"      📊 Progress: {current_progress}/{batch_size} ({progress_pct}%) - {node_key}",
                                file=sys.stderr,
                                flush=True,
                            )

                print(
                    f"      ✅ {node_key} completed {len(my_chunks)} chunks",
                    file=sys.stderr,
                    flush=True,
                )

            # Run the async function
            asyncio.run(submit_all_async())

        # Spawn one thread per node to submit all chunks async
        import concurrent.futures

        worker_futures = []

        print(
            f"   🔥 Firing {batch_size} async requests across {num_nodes} nodes...",
            file=sys.stderr,
            flush=True,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_nodes) as executor:
            for node_idx in range(num_nodes):
                if chunks_per_node[node_idx]:  # Only if node has work
                    future = executor.submit(async_batch_submit_worker, node_idx)
                    worker_futures.append(future)

            concurrent.futures.wait(worker_futures)

        print(
            f"   ✅ Async batch complete: {completed_count[0]}/{batch_size} chunks",
            file=sys.stderr,
            flush=True,
        )

        return results

        def process_node_batch(node_idx: int, node_texts: List[str], node_indices: List[int]):
            """Process a batch on a single node with concurrent workers (legacy pattern)"""
            if not node_texts:
                return []

            node = self.nodes[node_idx]
            node_key = f"{node['host']}:{node['port']}"
            batch_size_node = len(node_texts)

            # Track batch start time for throughput measurement
            import time as time_mod

            batch_start_time = time_mod.time()

            # Adaptive workers: fewer workers on slower nodes to reduce CPU contention
            # Fast nodes (< 2.5s/embed) → 2 workers for saturation
            # Slow nodes (≥ 2.5s/embed) → 1 worker to avoid overwhelming CPU
            # Unknown nodes (no data) → 1 worker to start conservatively
            node_perf = self.stats.get("node_performance", {}).get(node_key, {})
            avg_time = node_perf.get("avg_response_time", None)

            if avg_time is None:
                # No performance data - start conservatively with 1 worker
                workers_per_node = 1
                print(
                    f"      ⚙️  {node_key} using 1 worker (unknown performance - probing)",
                    file=sys.stderr,
                    flush=True,
                )
            elif avg_time < 2.5:
                # Fast node - use 2 workers for saturation
                workers_per_node = 2
                print(
                    f"      ⚙️  {node_key} using 2 workers (fast node: {avg_time:.1f}s avg)",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                # Slow node - single worker to avoid CPU contention
                workers_per_node = 1
                print(
                    f"      ⚙️  {node_key} using 1 worker (slow node: {avg_time:.1f}s avg)",
                    file=sys.stderr,
                    flush=True,
                )
            import queue as queue_module
            import threading

            import ollama

            node_url = f"http://{node['host']}:{node['port']}"
            results_node = [None] * batch_size_node
            work_queue = queue_module.Queue()
            completed_count = [0]  # Mutable counter for progress
            lock = threading.Lock()

            # Queue all work
            for i, text in enumerate(node_texts):
                work_queue.put((i, text))

            def worker(worker_id):
                """Worker: process chunks from queue with direct Ollama client"""
                import sys

                print(
                    f"      🔧 {node_key} worker {worker_id} STARTED", file=sys.stderr, flush=True
                )
                sys.stderr.flush()
                try:
                    # Each worker creates its own client for connection reuse
                    client = ollama.Client(host=node_url)
                    print(
                        f"      ✅ {node_key} worker {worker_id} client created",
                        file=sys.stderr,
                        flush=True,
                    )
                    sys.stderr.flush()
                except Exception as e:
                    print(
                        f"      ❌ {node_key} worker {worker_id} failed to create client: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    sys.stderr.flush()
                    return

                chunks_done = 0
                while not work_queue.empty():
                    try:
                        idx, text = work_queue.get(timeout=0.1)
                    except queue_module.Empty:
                        break

                    import time as time_module

                    start_embed = time_module.time()
                    try:
                        # Embed with timeout detection (warn if >10s)
                        result = client.embed(model=model, input=text, **kwargs)
                        embed_time = time_module.time() - start_embed

                        if embed_time > 10:
                            print(
                                f"      ⚠️  {node_key} worker {worker_id} chunk {idx} took {embed_time:.1f}s (slow!)",
                                file=sys.stderr,
                                flush=True,
                            )

                        results_node[idx] = result
                        chunks_done += 1

                        # Update progress
                        with lock:
                            completed_count[0] += 1
                            current = completed_count[0]

                        # Show progress every 50 chunks OR every 10s if slow
                        if current % 50 == 0 or current == batch_size_node or embed_time > 5:
                            pct = (current * 100) // batch_size_node
                            print(
                                f"      {node_key}: {current}/{batch_size_node} ({pct}%) - last chunk: {embed_time:.1f}s",
                                file=sys.stderr,
                                flush=True,
                            )
                            sys.stderr.flush()  # Force flush the stream itself

                        work_queue.task_done()
                    except Exception as e:
                        embed_time = time_module.time() - start_embed
                        print(
                            f"      ❌ {node_key} worker {worker_id} chunk {idx} FAILED after {embed_time:.1f}s: {e}",
                            file=sys.stderr,
                            flush=True,
                        )
                        work_queue.task_done()

                print(
                    f"      ✅ {node_key} worker {worker_id} FINISHED - processed {chunks_done} chunks",
                    file=sys.stderr,
                    flush=True,
                )
                sys.stderr.flush()

            # Launch workers for this node
            worker_threads = []
            for worker_id in range(workers_per_node):
                t = threading.Thread(target=worker, args=(worker_id,))
                t.start()
                worker_threads.append(t)

            # Wait for all workers to complete
            for t in worker_threads:
                t.join()

            # Calculate actual batch throughput
            batch_elapsed = time_mod.time() - batch_start_time
            throughput = batch_size_node / batch_elapsed if batch_elapsed > 0 else 0

            # Update node performance with REAL batch throughput
            with self._lock:
                # Ensure node_performance dict exists
                if "node_performance" not in self.stats:
                    self.stats["node_performance"] = {}

                # Get or create the perf dict (should already exist from __init__)
                if node_key not in self.stats["node_performance"]:
                    # Node doesn't exist - initialize with basic structure
                    self.stats["node_performance"][node_key] = {
                        "host": node_key,
                        "latency_ms": 0.0,
                        "success_rate": 1.0,
                        "total_requests": 0,
                        "failed_requests": 0,
                        "available": True,
                        "active_requests": 0,
                        "cpu_load": 0.5,
                        "gpu_free_mem": 0,
                        "priority": 999,
                    }

                perf = self.stats["node_performance"][node_key]

                # Update batch throughput (running average)
                old_throughput = perf.get("batch_throughput", throughput)
                perf["batch_throughput"] = (
                    old_throughput * 0.7 + throughput * 0.3
                )  # Weighted average

                # Update average response time based on throughput
                perf["avg_response_time"] = 1.0 / throughput if throughput > 0 else 5.0

                # DEBUG: Confirm stats were saved
                print(
                    f"   💾 Saved {node_key} stats: batch_throughput={perf['batch_throughput']:.2f}, avg_response_time={perf['avg_response_time']:.2f}s",
                    file=sys.stderr,
                    flush=True,
                )

            print(
                f"   ✅ {node_key}: {batch_size_node} chunks in {batch_elapsed:.1f}s ({throughput:.2f} chunks/sec)",
                file=sys.stderr,
                flush=True,
            )

            # Return results with original indices
            return list(zip(node_indices, results_node))

        # Process all nodes in parallel with timeout
        import sys

        with ThreadPoolExecutor(max_workers=num_nodes) as executor:
            # Submit one task per node
            futures = []
            for node_idx in range(num_nodes):
                if chunks_per_node[node_idx]:  # Only submit if node has work
                    node = self.nodes[node_idx]
                    node_key = f"{node['host']}:{node['port']}"
                    print(
                        f"   🚀 Submitting {len(chunks_per_node[node_idx])} chunks to {node_key}",
                        file=sys.stderr,
                        flush=True,
                    )
                    sys.stderr.flush()

                    future = executor.submit(
                        process_node_batch,
                        node_idx,
                        chunks_per_node[node_idx],
                        indices_per_node[node_idx],
                    )
                    futures.append((node_key, future))

            # Collect results as nodes complete (truly parallel)
            timeout_per_node = max(120, batch_size * 2)  # 2s per chunk minimum, 2min minimum
            futures_dict = {future: node_key for node_key, future in futures}

            for future in as_completed(futures_dict.keys(), timeout=timeout_per_node):
                node_key = futures_dict[future]
                try:
                    node_results = future.result()

                    for idx, result in node_results:
                        results[idx] = result
                        completed += 1

                    # Show progress after each node completes
                    progress_pct = (completed * 100) // batch_size
                    print(
                        f"   ✅ {node_key} completed: {len(node_results)} chunks ({progress_pct}% total)",
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception as e:
                    print(f"   ❌ {node_key} error: {e}", file=sys.stderr, flush=True)

        # Count successful embeddings
        success_count = sum(1 for r in results if r is not None)
        logger.info(f"✅ Batch complete: {success_count}/{batch_size} embeddings successful")

        return results

    def _old_embed_batch_individual_requests(
        self, model: str, inputs: List[str], max_workers: int, priority: int, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        OLD METHOD: Individual requests per chunk (kept for reference/fallback).

        This method makes separate HTTP requests for each chunk, which adds significant overhead.
        Use the new batched approach above for 10-12x better performance.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        batch_size = len(inputs)
        results = [None] * batch_size
        completed = 0

        def embed_single(index: int, text: str):
            """Embed single text with error handling."""
            try:
                result = self.embed(model, text, priority=priority, **kwargs)
                return index, result, None
            except Exception as e:
                return index, None, e

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(embed_single, i, text): i for i, text in enumerate(inputs)}

            # Collect results as they complete
            for future in as_completed(futures):
                index, result, error = future.result()
                completed += 1

                # Show progress: every 5 for small batches (<50), every 50 for large batches
                progress_interval = 5 if batch_size < 50 else 50
                if completed % progress_interval == 0 or completed == batch_size:
                    progress_pct = (completed * 100) // batch_size
                    logger.info(
                        f"   Progress: {completed}/{batch_size} embeddings ({progress_pct}%)"
                    )

                if error:
                    logger.error(f"⚠️  Error embedding text {index}: {error}")
                else:
                    results[index] = result

        # Count successful embeddings
        success_count = sum(1 for r in results if r is not None)
        logger.info(f"✅ Batch complete: {success_count}/{batch_size} embeddings successful")

        return results

    # Async I/O methods (requires httpx)

    async def chat_async(
        self,
        model: str,
        messages: List[Dict[str, str]],
        priority: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Async chat completion with non-blocking I/O.

        Args:
            model: Model name
            messages: Chat messages
            priority: Request priority 1-10
            **kwargs: Additional Ollama parameters

        Returns:
            Chat response dict

        Example:
            response = await pool.chat_async("llama2", messages)
        """
        if not self.async_session:
            raise RuntimeError("Async I/O requires httpx - install with: pip install httpx")

        data = {"model": model, "messages": messages, "stream": False, **kwargs}
        return await self._make_request_async("/api/chat", data, priority=priority)

    async def generate_async(
        self, model: str, prompt: str, priority: int = 5, **kwargs
    ) -> Dict[str, Any]:
        """
        Async text generation with non-blocking I/O.

        Args:
            model: Model name
            prompt: Text prompt
            priority: Request priority 1-10
            **kwargs: Additional Ollama parameters

        Returns:
            Generation response dict

        Example:
            response = await pool.generate_async("llama2", "Tell me a story")
        """
        if not self.async_session:
            raise RuntimeError("Async I/O requires httpx - install with: pip install httpx")

        data = {"model": model, "prompt": prompt, "stream": False, **kwargs}
        return await self._make_request_async("/api/generate", data, priority=priority)

    async def embed_async(
        self, model: str, input: str, priority: int = 5, **kwargs
    ) -> Dict[str, Any]:
        """
        Async embedding generation with non-blocking I/O.

        Args:
            model: Embedding model name
            input: Text to embed
            priority: Request priority 1-10
            **kwargs: Additional Ollama parameters

        Returns:
            Embedding response dict

        Example:
            response = await pool.embed_async("mxbai-embed-large", "Hello world")
        """
        if not self.async_session:
            raise RuntimeError("Async I/O requires httpx - install with: pip install httpx")

        data = {"model": model, "input": input, **kwargs}
        return await self._make_request_async("/api/embed", data, priority=priority)

    async def _make_request_async(
        self, endpoint: str, data: Dict[str, Any], priority: int = 5, timeout: float = 300.0
    ) -> Dict[str, Any]:
        """
        Make async HTTP request with intelligent routing (simplified version).

        Args:
            endpoint: API endpoint
            data: Request payload
            priority: Request priority
            timeout: Request timeout

        Returns:
            Response data
        """
        # Check cache first
        cache_key = None
        if self.cache.enabled and not data.get("stream", False):
            cache_key = self.cache.get_cache_key(endpoint, data)
            cached_response = self.cache.get(cache_key)
            if cached_response is not None:
                logger.debug(f"Cache hit for {endpoint} (async)")
                return cached_response

        # Track request
        with self._lock:
            self.stats["total_requests"] += 1

        # Select node
        node, routing_decision = self._select_node(payload=data, priority=priority)
        node_key = f"{node['host']}:{node['port']}"
        url = f"http://{node['host']}:{node['port']}{endpoint}"

        # Track active request
        with self._lock:
            if node_key in self.stats["node_performance"]:
                self.stats["node_performance"][node_key]["active_requests"] = (
                    self.stats["node_performance"][node_key].get("active_requests", 0) + 1
                )

        start_time = time.time()

        # Log request
        model = data.get("model", "unknown")
        operation = endpoint.split("/")[-1]
        log_ollama_request(
            backend=node_key, model=model, operation=operation, priority=priority, async_io=True
        )

        try:
            # Make async request
            response = await self.async_session.post(url, json=data, timeout=timeout)
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                # Success - update metrics
                with self._lock:
                    self.stats["successful_requests"] += 1
                    self.stats["nodes_used"][node_key] = (
                        self.stats["nodes_used"].get(node_key, 0) + 1
                    )

                    perf = self.stats["node_performance"][node_key]
                    perf["total_requests"] += 1

                    if perf["total_requests"] == 1:
                        perf["latency_ms"] = latency_ms
                    else:
                        perf["latency_ms"] = (
                            perf["latency_ms"] * (perf["total_requests"] - 1) + latency_ms
                        ) / perf["total_requests"]

                    perf["success_rate"] = (
                        perf["total_requests"] - perf["failed_requests"]
                    ) / perf["total_requests"]

                logger.info(f"✅ Async request succeeded: {node_key} ({latency_ms:.1f}ms)")

                # Log response
                log_ollama_response(
                    backend=node_key,
                    model=model,
                    latency_ms=latency_ms,
                    status_code=response.status_code,
                    async_io=True,
                )

                # Cache and return
                response_data = response.json()
                if cache_key is not None:
                    self.cache.set(cache_key, response_data)

                return response_data
            else:
                self._record_failure(node_key, latency_ms)
                log_ollama_error(
                    backend=node_key,
                    model=model,
                    error=f"HTTP {response.status_code}",
                    latency_ms=latency_ms,
                )
                raise RuntimeError(f"Request failed: HTTP {response.status_code}")

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._record_failure(node_key, latency_ms)
            log_ollama_error(backend=node_key, model=model, error=str(e), latency_ms=latency_ms)
            raise RuntimeError(f"Async request failed: {e}")

        finally:
            # Decrement active request counter
            with self._lock:
                if node_key in self.stats["node_performance"]:
                    self.stats["node_performance"][node_key]["active_requests"] = max(
                        0, self.stats["node_performance"][node_key].get("active_requests", 1) - 1
                    )

    def warm_model(self, model: str, node: Optional[Dict[str, Any]] = None) -> bool:
        """
        Warm up a model by pre-loading it on a node.

        This sends a minimal inference request to load the model into VRAM,
        reducing first-request latency by 1-5 seconds.

        Args:
            model: Model name to warm up
            node: Specific node to warm (None = all nodes)

        Returns:
            True if successful, False otherwise
        """
        try:
            target_nodes = [node] if node else self.nodes

            for n in target_nodes:
                node_key = f"{n['host']}:{n['port']}"
                logger.info(f"🔥 Warming model '{model}' on {node_key}...")

                # Send minimal generation request to load model
                # Use direct HTTP POST with extended timeout (600s = 10min) for large model loading
                # Bypasses pool's routing logic to avoid overhead
                # Long timeout is critical for CPU-only generation and resource-constrained systems
                try:
                    url = f"http://{n['host']}:{n['port']}/api/generate"
                    payload = {
                        "model": model,
                        "prompt": "",  # Empty prompt
                        "options": {"num_predict": 1},  # Generate only 1 token
                        "stream": False,
                    }

                    # Use persistent session with extended timeout for model loading
                    # 600s allows large models to load even on CPU-only or slow systems
                    response = self.session.post(url, json=payload, timeout=600)

                    if response.status_code == 200:
                        logger.info(f"✅ Model '{model}' warmed on {node_key}")
                    else:
                        logger.warning(
                            f"⚠️  Failed to warm model '{model}' on {node_key}: HTTP {response.status_code}"
                        )
                        return False

                except Exception as e:
                    logger.warning(f"⚠️  Error warming model '{model}' on {node_key}: {e}")
                    return False

            return True

        except Exception as e:
            logger.error(f"❌ Error in warm_model: {e}")
            return False

    def warm_models(self, models: List[str], parallel: bool = True) -> Dict[str, bool]:
        """
        Warm up multiple models across all nodes.

        This pre-loads models to reduce cold-start latency on first use.

        Args:
            models: List of model names to warm
            parallel: Warm models in parallel (default: True)

        Returns:
            Dict mapping model names to success status

        Example:
            pool.warm_models(["llama2", "codellama", "mistral"])
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}

        if not models:
            return results

        logger.info(f"🔥 Warming {len(models)} models across {len(self.nodes)} nodes...")

        if parallel:
            # Warm models in parallel
            with ThreadPoolExecutor(max_workers=min(len(models), 5)) as executor:
                futures = {executor.submit(self.warm_model, model): model for model in models}

                for future in as_completed(futures):
                    model = futures[future]
                    try:
                        success = future.result()
                        results[model] = success
                    except Exception as e:
                        logger.error(f"Error warming model {model}: {e}")
                        results[model] = False
        else:
            # Warm models sequentially
            for model in models:
                results[model] = self.warm_model(model)

        success_count = sum(1 for v in results.values() if v)
        logger.info(f"✅ Model warming complete: {success_count}/{len(models)} successful")

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics with performance metrics."""
        with self._lock:
            # Get real-time VRAM data
            local_vram = self.vram_monitor.get_local_vram_info()

            # Get Dask cluster info if available
            dask_info = {}
            if self.dask_client:
                try:
                    scheduler_info = self.dask_client.scheduler_info()
                    dask_info = {
                        "enabled": True,
                        "workers": len(scheduler_info.get("workers", {})),
                        "dashboard": self.dask_client.dashboard_link,
                    }
                except Exception:
                    dask_info = {"enabled": True, "status": "unknown"}
            else:
                dask_info = {"enabled": False}

            stats_data = {
                **self.stats,
                "nodes_configured": len(self.nodes),
                "nodes": [f"{n['host']}:{n['port']}" for n in self.nodes],
                "routing_strategy": self.routing_strategy.value,
                "intelligent_routing_enabled": self.enable_intelligent_routing,
                "http2_enabled": HTTPX_AVAILABLE,
                "async_io_enabled": self.async_session is not None,
                "dask": dask_info,
                "cache": self.cache.get_stats(),
                "vram_monitoring": {
                    "enabled": True,
                    "gpu_type": self.vram_monitor.gpu_type,
                    "local_gpu": local_vram if local_vram else {},
                    "refresh_interval_seconds": self._vram_refresh_interval,
                    "health_monitoring": self.health_monitor.get_stats(),
                },
            }
            return stats_data

    def add_node(self, host: str, port: int = 11434):
        """
        Add a node to the pool.

        Args:
            host: Node hostname/IP
            port: Node port
        """
        with self._lock:
            node = {"host": host, "port": str(port)}
            if node not in self.nodes:
                self.nodes.append(node)
                logger.info(f"Added node: {host}:{port}")

    def remove_node(self, host: str, port: int = 11434):
        """
        Remove a node from the pool.

        Args:
            host: Node hostname/IP
            port: Node port
        """
        with self._lock:
            node = {"host": host, "port": str(port)}
            if node in self.nodes:
                self.nodes.remove(node)
                logger.info(f"Removed node: {host}:{port}")

    # Cache management methods
    def clear_cache(self) -> None:
        """Clear all cached responses."""
        self.cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()

    def invalidate_cache_by_model(self, model: str) -> int:
        """
        Invalidate all cached responses for a specific model.

        Args:
            model: Model name

        Returns:
            Number of entries invalidated
        """
        return self.cache.invalidate_by_model(model)

    def export_cache(self) -> Dict[str, Any]:
        """
        Export cache for persistence.

        Returns:
            Cache data dictionary
        """
        return self.cache.export_cache()

    def import_cache(self, data: Dict[str, Any]) -> int:
        """
        Import cache from exported data.

        Args:
            data: Cache data from export_cache()

        Returns:
            Number of entries imported
        """
        return self.cache.import_cache(data)

    def stop(self):
        """
        Stop the pool and cleanup background threads.

        This method stops the VRAM refresh thread and performs cleanup.
        Call this when shutting down the pool to ensure proper resource cleanup.
        """
        logger.info("Stopping OllamaPool and cleaning up background threads...")

        # Stop VRAM refresh thread
        self._vram_refresh_enabled = False
        if self._vram_refresh_thread and self._vram_refresh_thread.is_alive():
            self._vram_refresh_thread.join(timeout=5.0)
            if self._vram_refresh_thread.is_alive():
                logger.warning("VRAM refresh thread did not stop within timeout")
            else:
                logger.info("VRAM refresh thread stopped successfully")

        # Stop health check thread
        self._health_check_enabled = False
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5.0)
            if self._health_check_thread.is_alive():
                logger.warning("Health check thread did not stop within timeout")
            else:
                logger.info("Health check thread stopped successfully")

        # Stop GPU subscriber if enabled
        if self.gpu_subscriber:
            try:
                self.gpu_subscriber.stop()
                logger.info("GPU subscriber stopped successfully")
            except Exception as e:
                logger.warning(f"Error stopping GPU subscriber: {e}")

        # Close persistent HTTP sessions
        if hasattr(self, "session"):
            try:
                self.session.close()
                logger.info("HTTP session closed successfully")
            except Exception as e:
                logger.warning(f"Error closing HTTP session: {e}")

        if hasattr(self, "async_session") and self.async_session:
            try:
                # Close async session (httpx AsyncClient)
                import asyncio

                loop = None
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                if loop.is_running():
                    # Schedule close task
                    asyncio.create_task(self.async_session.aclose())
                else:
                    # Run close in event loop
                    loop.run_until_complete(self.async_session.aclose())

                logger.info("Async HTTP session closed successfully")
            except Exception as e:
                logger.warning(f"Error closing async HTTP session: {e}")

        logger.info("OllamaPool stopped")

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            # Close session if it exists
            if hasattr(self, "session"):
                self.session.close()
        except Exception:
            pass  # Suppress errors during cleanup

    def __repr__(self):
        return f"OllamaPool(nodes={len(self.nodes)}, requests={self.stats['total_requests']})"


# Global pool instance (lazy-initialized)
_global_pool: Optional[OllamaPool] = None
_pool_lock = threading.Lock()


def get_pool() -> OllamaPool:
    """
    Get or create the global Ollama connection pool.

    This is thread-safe and lazy-initializes the pool on first access.

    Returns:
        Global OllamaPool instance
    """
    global _global_pool

    if _global_pool is None:
        with _pool_lock:
            # Double-check locking
            if _global_pool is None:
                _global_pool = OllamaPool.auto_configure()

    return _global_pool
