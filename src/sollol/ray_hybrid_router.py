"""
Ray-based Hybrid Router for parallel RPC sharding.

Uses Ray actors to manage multiple sharded model pools in parallel,
enabling better load balancing and fault tolerance for distributed inference.

Architecture:
- Each ShardedModelPool is a Ray actor managing N RPC backends
- Multiple pools can run in parallel
- Ray handles load balancing, queuing, and fault tolerance automatically
"""

# Set Bokeh session token expiration BEFORE any imports
# This prevents "Token is expired" errors on Dask dashboard
import os

os.environ["BOKEH_SESSION_TOKEN_EXPIRATION"] = "2147483647"  # Max int32 (~24 days)
os.environ["BOKEH_ALLOW_WS_ORIGIN"] = "*"

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import ray

from sollol.llama_cpp_coordinator import LlamaCppCoordinator, RPCBackend
from sollol.ollama_gguf_resolver import resolve_ollama_model
from sollol.pool import OllamaPool
from sollol.rpc_registry import RPCBackendRegistry

logger = logging.getLogger(__name__)


@ray.remote
class ShardedModelPool:
    """
    Ray actor managing one sharded model instance across N RPC backends.

    This runs as an independent process, allowing multiple pools to serve
    the same model in parallel for higher throughput.

    Now supports remote coordinator execution using Ray placement strategies.
    """

    def __init__(
        self,
        rpc_backends: List[Dict[str, Any]],
        coordinator_host: str = "127.0.0.1",
        coordinator_port: int = 18080,
        pool_id: int = 0,
        enable_remote_coordinator: bool = True,
    ):
        """
        Initialize sharded model pool.

        Args:
            rpc_backends: List of RPC backend configs for this pool
            coordinator_host: Host for llama-server coordinator (used if remote disabled)
            coordinator_port: Base port (actual port = base + pool_id)
            pool_id: Unique pool identifier
            enable_remote_coordinator: Enable remote coordinator execution on best node
        """
        self.pool_id = pool_id
        self.rpc_backends = rpc_backends
        self.coordinator_host = coordinator_host
        # Each pool gets unique port to avoid conflicts
        self.coordinator_port = coordinator_port + pool_id
        self.coordinator: Optional[LlamaCppCoordinator] = None
        self.current_model: Optional[str] = None
        self.enable_remote_coordinator = enable_remote_coordinator

        # Track which node the coordinator is actually running on
        self.coordinator_node: Optional[str] = None

        logger.info(
            f"ShardedModelPool {pool_id} initialized with {len(rpc_backends)} backends "
            f"(port {self.coordinator_port}, remote={enable_remote_coordinator})"
        )

    def _select_best_coordinator_node(self, model: str, gguf_path: str) -> Optional[str]:
        """
        Select the best node to run the coordinator based on available resources.

        Strategy:
        1. Query all RPC backends for resource availability
        2. Calculate score based on:
           - Available RAM (higher = better)
           - Available GPU VRAM (higher = better)
           - Current load (lower = better)
        3. Prefer nodes with more resources for large models

        Args:
            model: Model name for size estimation
            gguf_path: GGUF file path (could check file size)

        Returns:
            IP address of best node, or None to use local node
        """
        if not self.enable_remote_coordinator or not self.rpc_backends:
            return None

        try:
            import os

            # Estimate model requirements based on name
            import re

            from sollol.rpc_discovery import detect_node_resources

            size_match = re.search(r"(\d+)b", model.lower())
            if size_match:
                size_billions = int(size_match.group(1))
                # Estimate RAM needed: ~2GB per billion parameters for fp16
                estimated_ram_mb = size_billions * 2 * 1024
            else:
                estimated_ram_mb = 16384  # Default 16GB

            logger.info(
                f"Pool {self.pool_id}: Selecting coordinator node for {model} "
                f"(estimated {estimated_ram_mb}MB needed)"
            )

            # Score each backend node
            best_node = None
            best_score = -1

            for backend in self.rpc_backends:
                host = backend["host"]

                # Query resources (from Redis or direct detection)
                resources = detect_node_resources(host)

                # Calculate score
                cpu_ram = resources.get("cpu_ram_mb", 0)
                gpu_vram = sum(resources.get("gpu_vram_mb", []))
                total_ram = cpu_ram + gpu_vram

                # Score: available RAM minus estimated requirements
                # Positive score = node can handle it
                score = total_ram - estimated_ram_mb

                # Bonus for GPU nodes (better for coordinator if available)
                if resources.get("has_gpu", False):
                    score += 5000  # 5GB bonus

                logger.debug(
                    f"  {host}: RAM={cpu_ram}MB, GPU_VRAM={gpu_vram}MB, " f"score={score:.0f}"
                )

                if score > best_score:
                    best_score = score
                    best_node = host

            if best_node and best_score > 0:
                logger.info(
                    f"Pool {self.pool_id}: Selected {best_node} for coordinator "
                    f"(score={best_score:.0f})"
                )
                return best_node
            else:
                logger.warning(
                    f"Pool {self.pool_id}: No node has sufficient resources "
                    f"(best_score={best_score}), using local node"
                )
                return None

        except Exception as e:
            logger.warning(
                f"Pool {self.pool_id}: Error selecting coordinator node: {e}, " "using local node"
            )
            return None

    async def load_model(self, model: str, gguf_path: str) -> Dict[str, Any]:
        """
        Load model into this pool's coordinator.

        Uses intelligent node selection to run coordinator on best available node.

        Args:
            model: Model name (e.g., "llama3.1:405b")
            gguf_path: Path to GGUF file

        Returns:
            Status dict with coordinator info
        """
        if self.coordinator and self.current_model == model:
            logger.debug(f"Pool {self.pool_id}: Model {model} already loaded")
            return {
                "status": "already_loaded",
                "model": model,
                "pool_id": self.pool_id,
                "coordinator": f"{self.coordinator_node or self.coordinator_host}:{self.coordinator_port}",
            }

        # Convert dict configs to RPCBackend objects
        backends = [
            RPCBackend(host=backend["host"], port=backend.get("port", 50052))
            for backend in self.rpc_backends
        ]

        # Select best node for coordinator
        selected_node = self._select_best_coordinator_node(model, gguf_path)

        if selected_node:
            # Remote coordinator execution
            coordinator_host = selected_node
            self.coordinator_node = selected_node
            logger.info(
                f"Pool {self.pool_id}: Loading {model} across {len(backends)} RPC backends "
                f"(coordinator on {selected_node})"
            )
        else:
            # Local coordinator execution (current behavior)
            coordinator_host = self.coordinator_host
            self.coordinator_node = None
            logger.info(
                f"Pool {self.pool_id}: Loading {model} across {len(backends)} RPC backends "
                f"(coordinator local)"
            )

        self.coordinator = LlamaCppCoordinator(
            model_path=gguf_path,
            rpc_backends=backends,
            host=coordinator_host,
            port=self.coordinator_port,
        )

        await self.coordinator.start()
        self.current_model = model

        return {
            "status": "loaded",
            "model": model,
            "pool_id": self.pool_id,
            "coordinator": f"{coordinator_host}:{self.coordinator_port}",
            "coordinator_node": self.coordinator_node,
            "rpc_backends": len(backends),
        }

    async def chat(
        self, messages: List[Dict[str, str]], stream: bool = False, **kwargs
    ) -> Dict[str, Any]:
        """
        Run chat inference on this pool.

        When running on a remote node, Ray automatically streams results back
        to the requesting node through its distributed object store.

        Args:
            messages: Chat messages
            stream: Whether to stream response (not yet supported)
            **kwargs: Additional parameters

        Returns:
            Chat completion response (streamed back from remote node via Ray)
        """
        if not self.coordinator:
            raise RuntimeError(f"Pool {self.pool_id}: No model loaded")

        if stream:
            raise NotImplementedError("Streaming not yet supported in Ray pools")

        # Log which node is executing
        node_info = f" on {self.coordinator_node}" if self.coordinator_node else " locally"
        logger.debug(f"Pool {self.pool_id}: Running inference for {self.current_model}{node_info}")

        # Coordinator executes on selected node, Ray streams result back automatically
        response = await self.coordinator.chat(messages, stream=False, **kwargs)

        return response

    async def shutdown(self):
        """Shutdown this pool's coordinator."""
        if self.coordinator:
            logger.info(f"Pool {self.pool_id}: Shutting down coordinator")
            await self.coordinator.stop()
            self.coordinator = None
            self.current_model = None


class RayHybridRouter:
    """
    Ray-based hybrid router with parallel RPC sharding.

    Routes small models to Ollama pool, large models to Ray-managed sharded pools.
    Automatically load balances across pools and handles failures.

    Benefits over basic HybridRouter:
    - Multiple pools serve same model in parallel (higher throughput)
    - Automatic load balancing by Ray
    - Fault tolerance with automatic pool restarts
    - Better GPU utilization
    """

    def __init__(
        self,
        ollama_pool: Optional[OllamaPool] = None,
        rpc_backends: Optional[List[Dict[str, Any]]] = None,
        coordinator_host: str = "127.0.0.1",
        coordinator_base_port: int = 18080,
        backends_per_pool: int = 2,
        num_pools: int = None,
        enable_distributed: bool = True,
        auto_discover_rpc: bool = True,
        model_vram_threshold_mb: int = 16384,
        auto_fallback: bool = True,
        enable_remote_coordinator: bool = True,
        ctx_size: Optional[int] = None,
    ):
        """
        Initialize Ray-based hybrid router.

        Args:
            ollama_pool: OllamaPool for small models
            rpc_backends: List of ALL RPC backends (will be divided into pools)
            coordinator_host: Host for coordinators
            coordinator_base_port: Base port (each pool gets base + pool_id)
            backends_per_pool: Number of RPC backends per pool (default: 2)
            num_pools: Number of pools to create (auto-calculated if None)
            enable_distributed: Enable RPC sharding
            auto_discover_rpc: Auto-discover RPC backends if none provided
            model_vram_threshold_mb: VRAM threshold for Ollama vs RPC routing (16GB default)
            auto_fallback: Fallback to RPC if Ollama fails
            enable_remote_coordinator: Enable remote coordinator execution on best node
            ctx_size: Context window size (default: 8192, or env SOLLOL_CTX_SIZE)
        """
        # Store parameters first
        self.enable_distributed = enable_distributed
        self.auto_fallback = auto_fallback
        self.model_vram_threshold_mb = model_vram_threshold_mb

        # Support remote coordinator via environment variables
        import os

        self.coordinator_host = os.getenv(
            "SOLLOL_COORDINATOR_HOST", coordinator_host or "127.0.0.1"
        )
        self.coordinator_base_port = int(
            os.getenv("SOLLOL_COORDINATOR_PORT", str(coordinator_base_port or 18080))
        )
        # Context size with environment variable override
        self.ctx_size = int(
            os.getenv(
                "SOLLOL_CTX_SIZE", str(ctx_size or 8192)  # Default 8192 for long-form generation
            )
        )

        self.backends_per_pool = backends_per_pool
        self.enable_remote_coordinator = enable_remote_coordinator

        # Auto-discover RPC backends if needed (BEFORE ollama_pool initialization)
        if rpc_backends is None and enable_distributed and auto_discover_rpc:
            logger.info("🔍 Auto-discovering RPC backends...")
            from sollol.rpc_discovery import auto_discover_rpc_backends

            rpc_backends = auto_discover_rpc_backends()
            if rpc_backends:
                logger.info(f"✅ Auto-discovered {len(rpc_backends)} RPC backends")

        self.rpc_backends = rpc_backends or []
        self.has_rpc_backends = len(self.rpc_backends) > 0

        # Initialize ollama_pool AFTER we know about RPC backends
        # Only auto-configure if distributed enabled AND no RPC backends
        # (If RPC backends exist, we use them for large models instead)
        if ollama_pool is None:
            self.ollama_pool = (
                OllamaPool.auto_configure()
                if (enable_distributed and not self.has_rpc_backends)
                else None
            )
            if self.ollama_pool:
                logger.info("✅ Auto-configured Ollama pool (no RPC backends found)")
            else:
                logger.info("⏭️  Ollama pool disabled (using RPC backends for inference)")
        else:
            self.ollama_pool = ollama_pool

        # Log SOLLOL version at initialization
        from sollol import __version__

        logger.info(f"📦 SOLLOL v{__version__} - RayHybridRouter initializing")

        # Initialize Ray with dashboard enabled (for Ollama pool parallelization even without RPC)
        if self.enable_distributed:
            if not ray.is_initialized():
                import os

                # Disable Ray memory monitor
                os.environ["RAY_memory_monitor_refresh_ms"] = "0"

                # Try to connect to existing Ray cluster first (multi-app coordination)
                try:
                    logger.info("🔍 Attempting to connect to existing Ray cluster...")
                    ray.init(address="auto", ignore_reinit_error=True)
                    logger.info("✅ Connected to existing Ray cluster")
                except (ConnectionError, Exception) as e:
                    # No existing cluster, start a new one
                    logger.info(
                        f"🚀 Starting new Ray cluster for distributed coordination (no existing cluster found)"
                    )

                    import json

                    # Conservative memory settings to avoid "insufficient memory" errors
                    ray.init(
                        ignore_reinit_error=True,
                        dashboard_host="0.0.0.0",
                        dashboard_port=8265,
                        include_dashboard=True,
                        num_cpus=1,  # Single CPU to minimize workers
                        object_store_memory=256 * 1024 * 1024,  # 256MB for object store
                        _system_config={
                            "automatic_object_spilling_enabled": True,
                            "object_spilling_config": json.dumps(
                                {
                                    "type": "filesystem",
                                    "params": {"directory_path": "/tmp/ray_spill"},
                                }
                            ),
                        },
                    )
                    logger.info("📊 Ray dashboard available at http://localhost:8265")

            # RPC backends configuration (no Ray pools needed - route directly to coordinator)
            if self.has_rpc_backends:
                # Create RPC backend registry for health monitoring
                self.rpc_registry = RPCBackendRegistry()
                self.rpc_registry.load_from_config(self.rpc_backends)

                # No pools needed - we route directly to llama.cpp coordinator
                self.num_pools = 0
                self.pools: List[ray.actor.ActorHandle] = []
                self.current_model: Optional[str] = None

                # Check if using remote coordinator
                is_remote = self.coordinator_host not in ("127.0.0.1", "localhost")
                coordinator_info = f"{self.coordinator_host}:{self.coordinator_base_port}"
                if is_remote:
                    logger.info(
                        f"✅ RayHybridRouter initialized: "
                        f"Remote coordinator at {coordinator_info} 🌐, "
                        f"{len(self.rpc_backends)} RPC backends registered"
                    )
                else:
                    logger.info(
                        f"✅ RayHybridRouter initialized: "
                        f"Local coordinator at {coordinator_info}, "
                        f"{len(self.rpc_backends)} RPC backends registered"
                    )

                # Check if coordinator is available (non-blocking health check)
                try:
                    import httpx

                    with httpx.Client(timeout=2.0) as client:
                        coordinator_url = (
                            f"http://{coordinator_host}:{coordinator_base_port}/health"
                        )
                        response = client.get(coordinator_url)
                        if response.status_code == 200:
                            logger.info(f"✅ llama.cpp coordinator health check passed")
                        else:
                            logger.warning(
                                f"⚠️  llama.cpp coordinator health check returned {response.status_code}"
                            )
                except Exception as e:
                    if self.ollama_pool:
                        logger.warning(
                            f"⚠️  llama.cpp coordinator not responding at {coordinator_host}:{coordinator_base_port}. "
                            f"Will fall back to Ollama pool for large models if needed. "
                            f"Error: {e}"
                        )
                    else:
                        logger.error(
                            f"❌ llama.cpp coordinator not responding at {coordinator_host}:{coordinator_base_port}. "
                            f"No Ollama pool configured for fallback. "
                            f"Large model requests will fail until coordinator is started. "
                            f"Error: {e}"
                        )
            else:
                # No RPC backends - Ray still used for parallel Ollama pool execution
                self.rpc_registry = None
                self.num_pools = 0
                self.pools: List[ray.actor.ActorHandle] = []
                self.current_model: Optional[str] = None
                logger.info(
                    "✅ RayHybridRouter initialized (Ray enabled for Ollama parallelization, no RPC backends)"
                )
        else:
            logger.info("ℹ️  RayHybridRouter initialized without distributed support")
            self.pools = []
            self.num_pools = 0

        # Auto-start observability dashboard (configurable via ENV)
        import os

        self.dashboard = None
        self.dashboard_enabled = os.getenv("SOLLOL_DASHBOARD", "true").lower() in (
            "true",
            "1",
            "yes",
            "on",
        )
        self.dashboard_port = int(os.getenv("SOLLOL_DASHBOARD_PORT", "8080"))
        self.dashboard_enable_dask = os.getenv("SOLLOL_DASHBOARD_DASK", "true").lower() in (
            "true",
            "1",
            "yes",
            "on",
        )

        # Initialize Dask client for dashboard if enabled
        self.dask_client = None
        if self.dashboard_enable_dask:
            try:
                from dask.distributed import Client

                # Try to connect to existing Dask scheduler first (multi-app coordination)
                try:
                    logger.info("🔍 Attempting to connect to existing Dask scheduler...")
                    self.dask_client = Client("tcp://127.0.0.1:8786", timeout=2)
                    logger.info(f"✅ Connected to existing Dask scheduler")
                except Exception as e:
                    # No existing scheduler, create local cluster
                    logger.info(
                        "🚀 Starting Dask client with local cluster (no existing scheduler found)"
                    )
                    self.dask_client = Client(
                        processes=False, dashboard_address=":8787", silence_logs=logging.WARNING
                    )
                    logger.info(
                        f"✅ Dask client initialized for dashboard observability at {self.dask_client.dashboard_link}"
                    )
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize Dask client for dashboard: {e}")
                self.dashboard_enable_dask = False

        if self.dashboard_enabled:
            self._start_dashboard()
        else:
            msg = "📊 Dashboard disabled via SOLLOL_DASHBOARD=false"
            logger.info(msg)
            print(msg)

    async def route_request(
        self, model: str, messages: List[Dict[str, str]], stream: bool = False, **kwargs
    ) -> Dict[str, Any]:
        """
        Route request to appropriate backend.

        Small models → Ollama pool (task distribution)
        Large models → llama.cpp coordinator (RPC sharding)

        Args:
            model: Model name
            messages: Chat messages
            stream: Whether to stream (only supported on Ollama)
            **kwargs: Additional parameters

        Returns:
            Chat completion response
        """
        # Determine routing
        route_to_rpc = self._should_use_rpc(model)

        if route_to_rpc and self.enable_distributed and self.has_rpc_backends:
            # Large model → Route directly to llama.cpp coordinator (RPC sharding)
            logger.info(
                f"Routing {model} to llama.cpp coordinator for RPC sharding (estimated large model)"
            )
            return await self._route_to_llama_cpp_coordinator(model, messages, stream, **kwargs)
        elif self.ollama_pool:
            # Small model → Use Ollama pool for task distribution
            logger.info(f"Routing {model} to Ollama pool (estimated small model)")
            try:
                return await self.ollama_pool.chat_async(
                    model=model, messages=messages, stream=stream, **kwargs
                )
            except Exception as e:
                # DISABLED: Don't fallback to RPC for small models - it's slower and experimental
                # Only use RPC if explicitly configured for large models (70B+)
                # if self.auto_fallback and self.enable_distributed and self.has_rpc_backends:
                #     logger.warning(
                #         f"Ollama failed for {model}, falling back to RPC sharding: {e}"
                #     )
                #     return await self._route_to_llama_cpp_coordinator(model, messages, stream, **kwargs)
                logger.error(f"Ollama request failed for {model}: {e}")
                raise
        elif self.enable_distributed and self.has_rpc_backends:
            # No Ollama pool but have RPC → Force RPC routing
            logger.info(
                f"Routing {model} to llama.cpp coordinator for RPC sharding (no Ollama pool available)"
            )
            return await self._route_to_llama_cpp_coordinator(model, messages, stream, **kwargs)
        else:
            raise RuntimeError(
                f"Cannot route request for {model}: No Ollama pool and no RPC backends available. "
                "Configure either Ollama nodes or RPC backends."
            )

    async def _route_to_llama_cpp_coordinator(
        self, model: str, messages: List[Dict[str, str]], stream: bool, **kwargs
    ) -> Dict[str, Any]:
        """
        Route request directly to llama.cpp coordinator for RPC sharding.

        **AUTO-START:** If coordinator is not running, it will be automatically started
        with the model and RPC backends. The coordinator manages distributed inference
        across RPC workers.

        Fallback behavior:
        - If coordinator fails AND Ollama pool exists → fall back to Ollama
        - Otherwise → raise exception

        Args:
            model: Model name
            messages: Chat messages
            stream: Whether to stream (not supported for RPC)
            **kwargs: Additional parameters

        Returns:
            Chat completion response from llama.cpp coordinator or Ollama fallback
        """
        if stream and not self.ollama_pool:
            raise NotImplementedError(
                "Streaming not supported for RPC sharding (no Ollama pool available)"
            )

        # Use the coordinator HTTP client (created during init)
        if not hasattr(self, "coordinator_client"):
            # Import here to avoid circular dependency
            import httpx

            # Separate timeouts: connect=30s, read=1200s (20min for distributed model loading), write=30s
            timeout_config = httpx.Timeout(connect=30.0, read=1200.0, write=30.0, pool=None)
            self.coordinator_client = httpx.AsyncClient(timeout=timeout_config)

        # Use coordinator_host and coordinator_base_port from init
        coordinator_url = (
            f"http://{self.coordinator_host}:{self.coordinator_base_port}/v1/chat/completions"
        )

        # Auto-start coordinator if not running
        if not hasattr(self, "_coordinator_instance") or self._coordinator_instance is None:
            try:
                # Health check: try to ping coordinator
                health_url = f"http://{self.coordinator_host}:{self.coordinator_base_port}/health"
                health_response = await self.coordinator_client.get(health_url, timeout=2.0)
                logger.debug(f"✅ Coordinator health check passed: {health_response.status_code}")
            except Exception as health_err:
                # Coordinator not running - auto-start it!
                logger.info(
                    f"⚙️  Coordinator not running, auto-starting on {self.coordinator_host}:{self.coordinator_base_port}..."
                )

                # Resolve model path from Ollama if needed
                from sollol.ollama_gguf_resolver import resolve_ollama_model

                gguf_path = resolve_ollama_model(model)

                # Convert dict configs to RPCBackend objects
                backends = [
                    RPCBackend(host=backend["host"], port=backend.get("port", 50052))
                    for backend in self.rpc_backends
                ]

                # Create and start coordinator
                self._coordinator_instance = LlamaCppCoordinator(
                    model_path=gguf_path,
                    rpc_backends=backends,
                    host=self.coordinator_host,
                    port=self.coordinator_base_port,
                    ctx_size=self.ctx_size,
                )

                logger.info(f"🚀 Starting coordinator with {len(backends)} RPC backends...")
                await self._coordinator_instance.start()
                logger.info(f"✅ Coordinator auto-started successfully!")

        payload = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": False,
        }

        logger.debug(f"Sending request to llama.cpp coordinator at {coordinator_url}")
        logger.debug(f"Payload: {payload}")

        # Generate trace ID and record trace start
        import json as json_module
        import time
        import uuid

        import redis

        trace_id = str(uuid.uuid4())
        start_time = time.time()
        backend_list = (
            [f"{b['host']}:{b['port']}" for b in self.rpc_backends] if self.rpc_backends else []
        )

        # Publish routing decision event
        try:
            r = redis.Redis(
                host="localhost", port=6379, decode_responses=True, socket_connect_timeout=0.5
            )

            routing_event = {
                "timestamp": start_time,
                "event_type": "rpc_coordinator_route",
                "model": model,
                "coordinator_url": coordinator_url,
                "rpc_backends": backend_list,
                "routing_reason": "Model sharding via llama.cpp coordinator",
                "trace_id": trace_id,
            }
            r.publish("sollol:routing_events", json_module.dumps(routing_event))

            # Record trace start in Redis (TTL: 1 hour)
            trace_data = {
                "trace_id": trace_id,
                "model": model,
                "start_time": start_time,
                "status": "in_progress",
                "rpc_backends": backend_list,
                "num_backends": len(backend_list),
                "coordinator_url": coordinator_url,
                "prompt_tokens": sum(
                    len(m.get("content", "").split()) for m in messages
                ),  # Rough estimate
            }
            r.setex(f"sollol:trace:{trace_id}", 3600, json_module.dumps(trace_data))

            # Publish trace start event for real-time dashboard updates
            r.publish(
                "sollol:traces",
                json_module.dumps(
                    {
                        "event": "trace_start",
                        "trace": trace_data,
                    }
                ),
            )

        except Exception as e:
            logger.error(f"Failed to publish routing event or record trace: {e}", exc_info=True)

        try:
            response = await self.coordinator_client.post(coordinator_url, json=payload)
            if response.status_code != 200:
                logger.error(
                    f"❌ Coordinator returned {response.status_code}: {response.text[:500]}"
                )
            response.raise_for_status()
            result = response.json()

            # Calculate latency and extract token counts
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            # Extract token counts from response
            usage = result.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            # Update trace as completed
            try:
                r = redis.Redis(
                    host="localhost", port=6379, decode_responses=True, socket_connect_timeout=0.5
                )

                # Get existing trace data
                trace_json = r.get(f"sollol:trace:{trace_id}")
                if trace_json:
                    trace_data = json_module.loads(trace_json)
                    trace_data.update(
                        {
                            "status": "completed",
                            "end_time": end_time,
                            "latency_ms": latency_ms,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                        }
                    )
                    r.setex(f"sollol:trace:{trace_id}", 3600, json_module.dumps(trace_data))

                    # Publish trace completion event
                    r.publish(
                        "sollol:traces",
                        json_module.dumps(
                            {
                                "event": "trace_complete",
                                "trace": trace_data,
                            }
                        ),
                    )

                # Publish RPC activity event
                activity_event = {
                    "timestamp": end_time,
                    "event_type": "rpc_response",
                    "backend": coordinator_url,
                    "trace_id": trace_id,
                    "details": {
                        "model": model,
                        "rpc_backends_used": len(self.rpc_backends) if self.rpc_backends else 0,
                        "status": "success",
                        "latency_ms": latency_ms,
                        "completion_tokens": completion_tokens,
                    },
                }
                r.publish("sollol:dashboard:rpc:activity", json_module.dumps(activity_event))

            except Exception as e:
                logger.debug(f"Failed to publish RPC activity or update trace: {e}")

            return result
        except Exception as e:
            logger.error(f"llama.cpp coordinator request failed: {e}")

            # Mark trace as failed
            try:
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                r = redis.Redis(
                    host="localhost", port=6379, decode_responses=True, socket_connect_timeout=0.5
                )

                trace_json = r.get(f"sollol:trace:{trace_id}")
                if trace_json:
                    trace_data = json_module.loads(trace_json)
                    trace_data.update(
                        {
                            "status": "failed",
                            "end_time": end_time,
                            "latency_ms": latency_ms,
                            "error": str(e),
                        }
                    )
                    r.setex(f"sollol:trace:{trace_id}", 3600, json_module.dumps(trace_data))

                    # Publish trace failure event
                    r.publish(
                        "sollol:traces",
                        json_module.dumps(
                            {
                                "event": "trace_failed",
                                "trace": trace_data,
                            }
                        ),
                    )
            except Exception as trace_err:
                logger.debug(f"Failed to update trace on error: {trace_err}")

            # Graceful fallback to Ollama pool if available
            if self.ollama_pool:
                logger.warning(
                    f"⚠️  llama.cpp coordinator unavailable, falling back to Ollama pool for {model}"
                )
                try:
                    return await self.ollama_pool.chat_async(
                        model=model, messages=messages, stream=stream, **kwargs
                    )
                except Exception as ollama_error:
                    logger.error(f"Ollama fallback also failed: {ollama_error}")
                    raise RuntimeError(
                        f"Both RPC coordinator and Ollama fallback failed. "
                        f"Coordinator: {e}, Ollama: {ollama_error}"
                    )
            else:
                logger.error("No Ollama pool available for fallback")
                raise RuntimeError(
                    f"llama.cpp coordinator unavailable and no Ollama pool configured. "
                    f"Please ensure coordinator is running at {coordinator_url} or configure Ollama nodes. "
                    f"Error: {e}"
                )

    async def _route_to_ray_pool(
        self, model: str, messages: List[Dict[str, str]], stream: bool, **kwargs
    ) -> Dict[str, Any]:
        """
        Route request to Ray-managed sharded pool.

        DEPRECATED: This method spawns llama-server coordinators inside Ray actors,
        which causes OOM errors. Use _route_to_llama_cpp_coordinator instead.

        Ray automatically handles:
        - Load balancing (picks least busy pool)
        - Queuing (if all pools busy)
        - Fault tolerance (restarts failed pools)
        """
        # Load model into all pools if not already loaded
        if self.current_model != model:
            gguf_path = resolve_ollama_model(model)
            if not gguf_path:
                raise ValueError(f"Could not resolve {model} to GGUF file")

            logger.info(f"🔄 Loading {model} into {len(self.pools)} Ray pools...")

            # Load model into all pools in parallel
            load_tasks = [pool.load_model.remote(model, gguf_path) for pool in self.pools]
            # Use ray.get directly in gather (it's async-compatible)
            # Increased timeout to 300s for large model sharding (70B+ models take time to distribute)
            results = await asyncio.gather(
                *[asyncio.to_thread(ray.get, task, timeout=300) for task in load_tasks]
            )

            for result in results:
                logger.info(
                    f"  Pool {result['pool_id']}: {result['status']} "
                    f"({result.get('rpc_backends', 0)} backends)"
                )

            self.current_model = model

        # Ray automatically picks the least busy pool
        # We use round-robin for simplicity, but Ray's scheduler is smarter
        pool = self.pools[hash(str(messages)) % len(self.pools)]

        # Execute inference (Ray handles queuing if pool is busy)
        response_ref = pool.chat.remote(messages, stream=stream, **kwargs)
        # Use asyncio.to_thread for ray.get to avoid blocking
        response = await asyncio.to_thread(ray.get, response_ref)

        return response

    def _should_use_rpc(self, model: str) -> bool:
        """
        Determine if model should use RPC sharding.

        Small models → Ollama (task distribution across nodes)
        Large models → RPC sharding (model layers across GPUs)
        """
        # Extract size from model name
        import re

        size_match = re.search(r"(\d+)b", model.lower())
        if size_match:
            size_billions = int(size_match.group(1))
            # Estimate VRAM: ~2GB per billion parameters for fp16
            estimated_vram_mb = size_billions * 2 * 1024

            return estimated_vram_mb > self.model_vram_threshold_mb

        # Default: use RPC for unknown large models
        return False

    async def shutdown(self):
        """Shutdown all Ray pools."""
        if self.pools:
            logger.info(f"🛑 Shutting down {len(self.pools)} Ray pools...")
            shutdown_tasks = [pool.shutdown.remote() for pool in self.pools]
            await asyncio.gather(*[asyncio.wrap_future(ray.get(task)) for task in shutdown_tasks])
            self.pools = []
            logger.info("✅ All Ray pools shut down")

    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        stats = {
            "router_type": "ray_hybrid",
            "ollama_pool": {
                "nodes": len(self.ollama_pool.nodes) if self.ollama_pool else 0,
                "requests": self.ollama_pool.stats["total_requests"] if self.ollama_pool else 0,
            }
            if self.ollama_pool
            else None,
            "ray_pools": {
                "num_pools": len(self.pools) if hasattr(self, "pools") else 0,
                "backends_per_pool": self.backends_per_pool,
                "total_backends": len(self.rpc_backends),
                "current_model": self.current_model if hasattr(self, "current_model") else None,
            },
        }

        return stats

    def _start_dashboard(self):
        """
        Start the standalone observability dashboard as a subprocess.

        The dashboard provides real-time monitoring via WebSockets:
        - System metrics (hosts, latency, success rate, GPU memory, Ray workers)
        - Real-time logs streaming (via Redis pub/sub)
        - Ollama server activity monitoring
        - llama.cpp RPC activity monitoring
        - Embedded Ray dashboard iframe
        - Embedded Dask dashboard iframe

        Configured via environment variables:
        - SOLLOL_DASHBOARD=true|false (default: true)
        - SOLLOL_DASHBOARD_PORT=8080 (default: 8080)
        - SOLLOL_DASHBOARD_DASK=true|false (default: true)
        - SOLLOL_REDIS_URL=redis://localhost:6379 (default)
        """
        try:
            from .dashboard_launcher import launch_dashboard_subprocess
            from .dashboard_log_hooks import auto_install_hooks, install_log_hook_main

            # Get Redis URL from environment or coordinator
            redis_url = os.getenv("SOLLOL_REDIS_URL", "redis://localhost:6379")
            if hasattr(self, "coordinator") and self.coordinator:
                # Use same Redis as coordinator
                redis_url = getattr(self.coordinator, "redis_url", redis_url)

            msg = f"🚀 Launching SOLLOL Dashboard subprocess on port {self.dashboard_port}..."
            logger.info(msg)
            print(msg)

            # Launch dashboard as subprocess
            self.dashboard = launch_dashboard_subprocess(
                redis_url=redis_url,
                port=self.dashboard_port,
                ray_dashboard_port=8265,
                dask_dashboard_port=8787,
                enable_dask=self.dashboard_enable_dask,
                debug=False,
            )

            # Install log hooks in main process
            try:
                hook_result = install_log_hook_main(redis_url)
                if hook_result:
                    logger.info(f"✅ Main process log hook installed -> {redis_url}")
                    print(f"✅ Main process log hook installed -> {redis_url}")
                else:
                    logger.warning("⚠️  Main process log hook installation returned False")
                    print("⚠️  Main process log hook installation returned False")
            except Exception as e:
                logger.error(f"❌ Failed to install main process log hook: {e}")
                print(f"❌ Failed to install main process log hook: {e}")

            # Auto-install hooks on Ray/Dask workers
            try:
                ray_ref = self.ray if hasattr(self, "ray") else None
                dask_client = self.dask_client if hasattr(self, "dask_client") else None
                hook_results = auto_install_hooks(
                    redis_url=redis_url,
                    ray_ref=ray_ref,
                    dask_client=dask_client,
                )
                logger.info(f"📡 Log hooks installed: {hook_results}")
            except Exception as e:
                logger.warning(f"Failed to auto-install worker hooks: {e}")

            # Publish router metadata to Redis for dashboard
            try:
                import json

                import redis

                r = redis.from_url(redis_url, decode_responses=True)

                # Get nodes from ollama_pool
                nodes_list = []
                if hasattr(self, "ollama_pool") and hasattr(self.ollama_pool, "nodes"):
                    nodes_list = list(self.ollama_pool.nodes)

                # Get RPC backends
                rpc_list = []
                if hasattr(self, "rpc_backends") and self.rpc_backends:
                    for b in self.rpc_backends:
                        if isinstance(b, dict):
                            rpc_list.append({"host": b.get("host"), "port": b.get("port")})
                        else:
                            rpc_list.append(
                                {"host": getattr(b, "host", ""), "port": getattr(b, "port", "")}
                            )

                router_metadata = {
                    "nodes": nodes_list,
                    "rpc_backends": rpc_list,
                    "metrics": self.get_stats() if hasattr(self, "get_stats") else {},
                }
                r.set(
                    "sollol:router:metadata", json.dumps(router_metadata), ex=3600
                )  # Expire after 1 hour
                logger.info(
                    f"📡 Published router metadata to Redis: {len(nodes_list)} nodes, {len(rpc_list)} RPC backends"
                )
            except Exception as e:
                logger.warning(f"Failed to publish router metadata to Redis: {e}")

            msg1 = "✅ SOLLOL Dashboard launched successfully"
            msg2 = f"   📊 Access at http://localhost:{self.dashboard_port}"
            msg3 = f"   📡 Using Redis at {redis_url}"
            msg4 = f"   🔧 Disable with: SOLLOL_DASHBOARD=false"

            logger.info(msg1)
            logger.info(msg2)
            logger.info(msg3)
            logger.info(msg4)
            print(msg1)
            print(msg2)
            print(msg3)
            print(msg4)

        except Exception as e:
            err_msg = f"⚠️  Failed to start dashboard: {e}"
            info_msg = "   Dashboard can be started manually: python -m sollol.dashboard_service"
            logger.warning(err_msg)
            logger.info(info_msg)
            print(err_msg)
            print(info_msg)
