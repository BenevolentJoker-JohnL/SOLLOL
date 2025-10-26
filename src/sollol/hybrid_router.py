"""
Hybrid Router: Intelligent Selection Between Task Distribution and Model Sharding

TWO INDEPENDENT ROUTING MODES (can be used together):
1. Task Distribution (Ollama Pool):
   - Load balance agent requests across Ollama nodes (parallel execution)
   - For small/medium models that fit on single GPU (≤13B)
   - Multiple agents run in parallel on different nodes

2. Model Sharding (llama.cpp RPC):
   - Distribute a single large model across multiple RPC backends
   - For large models requiring multiple nodes (70B+)
   - Single model split across nodes via llama.cpp

💡 Both modes can be enabled simultaneously for optimal performance!

This enables seamless support for models of ANY size while maintaining
Ollama's simple API.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .llama_cpp_coordinator import LlamaCppCoordinator, RPCBackend
from .ollama_gguf_resolver import OllamaGGUFResolver
from .pool import OllamaPool
from .routing_logger import get_routing_logger
from .rpc_registry import RPCBackendRegistry

logger = logging.getLogger(__name__)


@dataclass
class ModelProfile:
    """Profile of a model's resource requirements."""

    name: str
    parameter_count: int  # Billion parameters
    estimated_memory_gb: float
    requires_distributed: bool
    num_layers: int = 0


# Model profiles for routing decisions
MODEL_PROFILES = {
    # Small models (fit on single GPU)
    "llama3.2": ModelProfile("llama3.2", 3, 2.5, False, 32),
    "llama3.2:3b": ModelProfile("llama3.2:3b", 3, 2.5, False, 32),
    "phi": ModelProfile("phi", 3, 1.5, False, 32),
    "phi3": ModelProfile("phi3", 4, 2.0, False, 32),
    "gemma:7b": ModelProfile("gemma:7b", 7, 5.0, False, 28),
    "llama3:8b": ModelProfile("llama3:8b", 8, 6.0, False, 32),
    "llama3.1:8b": ModelProfile("llama3.1:8b", 8, 6.0, False, 32),
    "mistral:7b": ModelProfile("mistral:7b", 7, 5.0, False, 32),
    "llama2:7b": ModelProfile("llama2:7b", 7, 5.0, False, 32),
    "llama2:13b": ModelProfile("llama2:13b", 13, 9.0, False, 40),
    "codellama:13b": ModelProfile(
        "codellama:13b", 13, 7.4, True, 40
    ),  # Force distributed for testing
    # Medium models (might fit on large single GPU)
    "llama2:70b": ModelProfile("llama2:70b", 70, 40.0, True, 80),
    "llama3:70b": ModelProfile("llama3:70b", 70, 40.0, True, 80),
    "llama3.1:70b": ModelProfile("llama3.1:70b", 70, 40.0, True, 80),
    "mixtral:8x7b": ModelProfile("mixtral:8x7b", 47, 26.0, True, 32),
    "qwen2.5:72b": ModelProfile("qwen2.5:72b", 72, 42.0, True, 80),
    # Large models (REQUIRE distributed)
    "llama3.1:405b": ModelProfile("llama3.1:405b", 405, 230.0, True, 126),
    "mixtral:8x22b": ModelProfile("mixtral:8x22b", 141, 80.0, True, 56),
}


class HybridRouter:
    """
    Routes requests between Task Distribution (Ollama pool) and Model Sharding (llama.cpp).

    TWO INDEPENDENT ROUTING MODES (can both be enabled):
    - Task Distribution: Ollama pool load balances agent requests across nodes (parallel execution)
    - Model Sharding: llama.cpp distributes model layers across RPC backends (single model split)

    💡 When both are enabled, routing is automatic based on model size!

    Decision logic:
    1. Small models (<= 13B) → Ollama pool (task distribution / load balancing)
    2. Medium models (14B-70B) → Ollama if available, else llama.cpp model sharding
    3. Large models (> 70B) → llama.cpp model sharding across RPC backends
    4. Unknown models → Estimate from name, default to Ollama pool
    """

    def __init__(
        self,
        ollama_pool: Optional[OllamaPool] = None,
        rpc_backends: Optional[List[Dict[str, Any]]] = None,
        coordinator_host: str = "127.0.0.1",
        coordinator_port: int = 18080,
        enable_distributed: bool = True,
        auto_discover_rpc: bool = True,
        auto_setup_rpc: bool = False,
        num_rpc_backends: int = 1,
        auto_fallback: bool = True,
        debug_coordinator_recovery: bool = False,
    ):
        """
        Initialize hybrid router with automatic GGUF resolution from Ollama.

        Args:
            ollama_pool: OllamaPool for standard requests
            rpc_backends: List of RPC backend configs [{"host": "ip", "port": 50052}]
                         If None and auto_discover_rpc=True, will auto-discover/setup
            coordinator_host: Host for llama-server coordinator
            coordinator_port: Port for llama-server coordinator
            enable_distributed: Enable llama.cpp distributed routing
            auto_discover_rpc: Auto-discover RPC backends if none provided
            auto_setup_rpc: Auto-setup RPC backends if none found (requires llama.cpp)
            num_rpc_backends: Number of RPC backends to start if auto-setup is enabled
            auto_fallback: Automatically fallback to RPC if Ollama fails (default: True)
            debug_coordinator_recovery: Enable verbose logging for coordinator crash recovery (default: False)
                                       Can also be enabled via SOLLOL_DEBUG_COORDINATOR_RECOVERY=true env var
        """
        self.ollama_pool = ollama_pool
        self.auto_fallback = auto_fallback
        # Allow environment variable to override debug setting
        self.debug_coordinator_recovery = debug_coordinator_recovery or os.getenv(
            "SOLLOL_DEBUG_COORDINATOR_RECOVERY", ""
        ).lower() in ("true", "1", "yes", "on")

        # Cache for routing decisions (model -> use_rpc)
        self.routing_cache = {}

        # Auto-discover RPC backends if none provided
        if rpc_backends is None and enable_distributed and auto_discover_rpc:
            logger.info("🔍 Auto-discovering RPC backends...")
            from .rpc_discovery import auto_discover_rpc_backends

            discovered = auto_discover_rpc_backends()
            if discovered:
                rpc_backends = discovered
                logger.info(f"✅ Auto-discovered {len(discovered)} RPC backends")
            else:
                logger.info("ℹ️  No RPC backends found via auto-discovery")

                # Try auto-setup if enabled
                if auto_setup_rpc:
                    logger.info("🚀 Attempting to auto-setup RPC backends...")
                    from .rpc_auto_setup import auto_setup_rpc_backends

                    try:
                        setup_backends = auto_setup_rpc_backends(
                            num_backends=num_rpc_backends, auto_build=True, discover_network=False
                        )
                        if setup_backends:
                            rpc_backends = setup_backends
                            logger.info(f"✅ Auto-setup created {len(setup_backends)} RPC backends")
                        else:
                            logger.warning("⚠️  Auto-setup failed to create RPC backends")
                    except Exception as e:
                        logger.error(f"❌ Auto-setup failed: {e}")

        self.enable_distributed = enable_distributed and rpc_backends is not None

        # Store RPC backend configs for on-demand coordinator creation
        self.rpc_backends = rpc_backends
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port

        # Create RPC backend registry for intelligent backend selection and monitoring
        self.rpc_registry = RPCBackendRegistry()
        if rpc_backends:
            self.rpc_registry.load_from_config(rpc_backends)
            logger.info(f"Loaded {len(rpc_backends)} RPC backends into registry")

        # Coordinator created on-demand when first large model request arrives
        self.coordinator: Optional[LlamaCppCoordinator] = None
        self.coordinator_model: Optional[str] = None  # Track which model is loaded

        # GGUF resolver for extracting models from Ollama storage
        self.gguf_resolver = OllamaGGUFResolver()

        # Initialize routing event logger
        self.routing_logger = get_routing_logger()

        # Log initialization status
        rpc_info = ""
        if self.enable_distributed:
            rpc_info = f", RPC backends={len(self.rpc_backends)}"

        logger.info(
            f"HybridRouter initialized: "
            f"Task Distribution (Ollama pool)={'enabled' if ollama_pool else 'disabled'}, "
            f"Model Sharding (llama.cpp)={'enabled' if self.enable_distributed else 'disabled'}"
            f"{rpc_info}"
        )

        # Auto-start observability dashboard (configurable via ENV)
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

        if self.dashboard_enabled:
            self._start_dashboard()
        else:
            msg = "📊 Dashboard disabled via SOLLOL_DASHBOARD=false"
            logger.info(msg)
            print(msg)

    async def _ensure_coordinator_for_model(self, model: str):
        """
        Ensure coordinator is started with the correct model.

        This method:
        1. Resolves GGUF path from Ollama storage (automatic!)
        2. Creates coordinator if not exists
        3. Restarts coordinator if different model requested
        4. Starts coordinator if not already running

        Args:
            model: Ollama model name (e.g., "llama3.1:405b")
        """
        # Quick check without lock (optimization for common case)
        if self.coordinator and self.coordinator_model == model:
            # Check if coordinator process is still alive
            if self.coordinator.process and self.coordinator.process.poll() is None:
                logger.info(f"✅ Coordinator already running for {model}, reusing")
                return
            else:
                logger.warning(f"⚠️  Coordinator process died! Will recreate.")
                self.coordinator = None

        # Resolve GGUF path from Ollama storage
        logger.info(f"🔍 Resolving GGUF path for Ollama model: {model}")
        gguf_path = self.gguf_resolver.resolve(model)

        if not gguf_path:
            raise FileNotFoundError(
                f"Could not find GGUF for '{model}' in Ollama storage. "
                f"Please ensure model is pulled: ollama pull {model}"
            )

        logger.info(f"✅ Found GGUF: {gguf_path}")

        # Stop existing coordinator if serving different model
        if self.coordinator and self.coordinator_model != model:
            logger.info(
                f"Stopping coordinator (switching from {self.coordinator_model} to {model})"
            )
            self.routing_logger.log_model_switch(
                from_model=self.coordinator_model, to_model=model, backend="llamacpp"
            )
            self.routing_logger.log_coordinator_stop(model=self.coordinator_model)
            await self.coordinator.stop()
            self.coordinator = None

        # Create coordinator if needed
        if not self.coordinator:
            # Convert dict configs to RPCBackend objects
            backends = [
                RPCBackend(host=backend["host"], port=backend.get("port", 50052))
                for backend in self.rpc_backends
            ]

            # Create coordinator
            self.coordinator = LlamaCppCoordinator(
                model_path=gguf_path,
                rpc_backends=backends,
                host=self.coordinator_host,
                port=self.coordinator_port,
            )

            # Start coordinator
            logger.info(f"🚀 Starting llama.cpp coordinator for {model}...")
            try:
                await self.coordinator.start()
                # Track which model is loaded
                self.coordinator_model = model
                logger.info(
                    f"✅ Coordinator started with {len(backends)} RPC backends "
                    f"on {self.coordinator_host}:{self.coordinator_port}"
                )
                self.routing_logger.log_coordinator_start(
                    model=model,
                    rpc_backends=len(backends),
                    coordinator_host=self.coordinator_host,
                    coordinator_port=self.coordinator_port,
                )
            except Exception as e:
                # Startup failed - clean up the failed coordinator
                logger.error(f"Coordinator startup failed: {e}")
                if self.coordinator and self.coordinator.process:
                    self.coordinator.process.kill()
                self.coordinator = None
                raise

    def should_use_distributed(self, model: str) -> bool:
        """
        Determine if model should use model sharding (llama.cpp RPC distribution).

        Args:
            model: Model name

        Returns:
            True if should use llama.cpp model sharding (False = use Ollama task distribution)
        """
        if not self.enable_distributed:
            return False

        # Get model profile
        profile = self._get_model_profile(model)

        # Decision rules for routing
        if profile.parameter_count <= 13:
            # Small models: use Ollama pool (task distribution / load balancing)
            return False
        elif profile.parameter_count <= 70:
            # Medium models: prefer Ollama pool, use model sharding if marked required
            return profile.requires_distributed
        else:
            # Large models: must use model sharding (llama.cpp RPC distribution)
            return True

    def _get_model_profile(self, model: str) -> ModelProfile:
        """Get or estimate model profile."""
        # Normalize model name
        model_key = model.lower().strip()

        # Direct lookup
        if model_key in MODEL_PROFILES:
            return MODEL_PROFILES[model_key]

        # Try without tag
        base_model = model_key.split(":")[0]
        if base_model in MODEL_PROFILES:
            return MODEL_PROFILES[base_model]

        # Estimate from name
        return self._estimate_model_profile(model)

    def _estimate_model_profile(self, model: str) -> ModelProfile:
        """Estimate model requirements from name."""
        model_lower = model.lower()

        # Extract parameter count from name
        param_count = 8  # Default assumption

        # Common patterns
        if "405b" in model_lower:
            param_count = 405
        elif "70b" in model_lower:
            param_count = 70
        elif "34b" in model_lower:
            param_count = 34
        elif "13b" in model_lower:
            param_count = 13
        elif "8b" in model_lower:
            param_count = 8
        elif "7b" in model_lower:
            param_count = 7
        elif "3b" in model_lower:
            param_count = 3
        elif "1b" in model_lower:
            param_count = 1

        # Estimate memory (rough: ~600MB per billion parameters)
        estimated_memory = param_count * 0.6

        # Requires model sharding if > 70B
        requires_distributed = param_count > 70

        logger.info(
            f"Estimated profile for '{model}': {param_count}B params, "
            f"~{estimated_memory:.1f}GB, model_sharding={requires_distributed}"
        )

        return ModelProfile(
            name=model,
            parameter_count=param_count,
            estimated_memory_gb=estimated_memory,
            requires_distributed=requires_distributed,
            num_layers=max(32, param_count),  # Rough estimate
        )

    async def route_request(
        self, model: str, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """
        Route request to appropriate backend with intelligent fallback.

        Routing logic:
        1. Check cached routing decision for this model
        2. If no cache:
           a. Try Ollama pool first
           b. If Ollama fails (OOM/timeout/error) AND auto_fallback enabled → try RPC
           c. Cache successful routing decision

        Args:
            model: Model name
            messages: Chat messages
            **kwargs: Additional parameters

        Returns:
            Response from either Ollama or llama.cpp
        """
        # Check if we have a cached routing decision
        if model in self.routing_cache:
            use_rpc = self.routing_cache[model]
            if use_rpc:
                logger.info(f"🔗 Routing '{model}' to RPC (cached decision)")
                self.routing_logger.log_route_decision(
                    model=model, backend="rpc", reason="cached_routing_decision", cached=True
                )
                return await self._route_to_llamacpp(model, messages, **kwargs)
            else:
                logger.info(f"📡 Routing '{model}' to Ollama (cached decision)")
                self.routing_logger.log_route_decision(
                    model=model, backend="ollama", reason="cached_routing_decision", cached=True
                )
                return await self._route_to_ollama(model, messages, **kwargs)

        # No cached decision - check resources proactively
        if self.ollama_pool:
            # Proactive resource check to avoid OOM
            can_fit_on_ollama = await self._check_if_model_fits_ollama(model)

            if can_fit_on_ollama:
                # Resources sufficient - try Ollama
                try:
                    logger.info(f"📡 Routing '{model}' to Ollama pool (sufficient resources)")
                    profile = self._get_model_profile(model)
                    self.routing_logger.log_route_decision(
                        model=model,
                        backend="ollama",
                        reason=f"sufficient_resources (estimated {profile.estimated_memory_gb:.1f}GB)",
                        parameter_count=profile.parameter_count,
                    )
                    result = await self._route_to_ollama(model, messages, **kwargs)
                    # Success! Cache this decision
                    self.routing_cache[model] = False
                    return result
                except Exception as e:
                    # Unexpected error - try RPC fallback if enabled
                    if self.auto_fallback and self.enable_distributed:
                        logger.warning(
                            f"⚠️  Ollama failed unexpectedly for '{model}': {str(e)[:100]}"
                        )
                        logger.info(f"🔗 Falling back to RPC model sharding...")
                        self.routing_logger.log_fallback(
                            model=model,
                            from_backend="ollama",
                            to_backend="rpc",
                            reason=f"ollama_error: {str(e)[:100]}",
                        )
                        result = await self._route_to_llamacpp(model, messages, **kwargs)
                        self.routing_cache[model] = True
                        return result
                    else:
                        raise
            else:
                # Insufficient resources - use RPC if available
                if self.enable_distributed:
                    logger.info(f"🔗 Routing '{model}' to RPC (insufficient Ollama resources)")
                    profile = self._get_model_profile(model)
                    self.routing_logger.log_route_decision(
                        model=model,
                        backend="rpc",
                        reason=f"insufficient_ollama_resources (requires {profile.estimated_memory_gb:.1f}GB)",
                        parameter_count=profile.parameter_count,
                    )
                    result = await self._route_to_llamacpp(model, messages, **kwargs)
                    self.routing_cache[model] = True
                    return result
                else:
                    # No RPC available - try Ollama anyway (may fail)
                    logger.warning(
                        f"⚠️  '{model}' may not fit on Ollama nodes, but no RPC available"
                    )
                    result = await self._route_to_ollama(model, messages, **kwargs)
                    self.routing_cache[model] = False
                    return result

        # No Ollama pool available - use RPC directly
        if self.enable_distributed:
            logger.info(f"🔗 Routing '{model}' to RPC (no Ollama pool)")
            self.routing_logger.log_route_decision(
                model=model, backend="rpc", reason="no_ollama_pool_available"
            )
            result = await self._route_to_llamacpp(model, messages, **kwargs)
            self.routing_cache[model] = True
            return result

        raise RuntimeError("No backends available for routing")

    async def _route_to_ollama(
        self, model: str, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """Route to Ollama pool for task distribution (load balancing)."""
        if not self.ollama_pool:
            raise RuntimeError("Ollama pool not available")

        # Convert messages to prompt if needed
        if isinstance(messages, list) and len(messages) > 0:
            # Use pool's async chat method to avoid blocking the event loop
            priority = kwargs.pop("priority", 5)
            result = await self.ollama_pool.chat_async(
                model=model, messages=messages, priority=priority, **kwargs
            )
            return result
        else:
            raise ValueError("Invalid messages format")

    async def _route_to_llamacpp(
        self, model: str, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """
        Route to llama.cpp for model sharding (distribute layers across RPC backends).

        This method automatically:
        1. Resolves the GGUF from Ollama's blob storage
        2. Starts the coordinator with the correct model
        3. Distributes model layers across RPC backends
        4. Makes the inference request
        5. Handles coordinator crashes with automatic retry
        """
        # Ensure coordinator is started with correct model (auto-resolves GGUF!)
        await self._ensure_coordinator_for_model(model)

        # Use the coordinator's chat method with crash recovery
        try:
            result = await self.coordinator.chat(
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.7),
            )
        except Exception as e:
            # Coordinator crashed or failed - attempt recovery once
            logger.warning(f"⚠️  Coordinator failed for '{model}': {str(e)[:100]}")

            if self.debug_coordinator_recovery:
                logger.info("🔄 Attempting coordinator recovery (restarting)...")
            else:
                logger.debug("Attempting coordinator recovery (restarting)...")

            try:
                # Stop and clear the failed coordinator
                if self.coordinator:
                    if self.debug_coordinator_recovery:
                        logger.debug("Stopping failed coordinator...")
                    try:
                        await self.coordinator.stop()
                    except:
                        pass  # Ignore errors stopping crashed coordinator
                    self.coordinator = None
                    self.coordinator_model = None

                # Restart coordinator with fresh instance
                if self.debug_coordinator_recovery:
                    logger.debug(f"Restarting coordinator for model '{model}'...")
                await self._ensure_coordinator_for_model(model)

                # Retry the request once
                if self.debug_coordinator_recovery:
                    logger.info("✅ Coordinator restarted, retrying request...")
                else:
                    logger.debug("Coordinator restarted, retrying request...")

                result = await self.coordinator.chat(
                    messages=messages,
                    max_tokens=kwargs.get("max_tokens", 512),
                    temperature=kwargs.get("temperature", 0.7),
                )

                if self.debug_coordinator_recovery:
                    logger.info("✅ Request succeeded after coordinator restart")
                else:
                    logger.debug("Request succeeded after coordinator restart")

            except Exception as retry_error:
                # Recovery failed - log and re-raise
                logger.error(f"❌ Coordinator recovery failed: {retry_error}")
                raise RuntimeError(
                    f"Coordinator failed and recovery attempt unsuccessful: {retry_error}"
                ) from retry_error

        # Convert to Ollama-style response
        return self._convert_llamacpp_to_ollama(result, model)

    def _convert_llamacpp_to_ollama(self, llamacpp_result: Dict, model: str) -> Dict[str, Any]:
        """Convert llama.cpp response to Ollama format."""
        # llama.cpp /v1/chat/completions returns OpenAI-compatible format
        # Extract the message content
        choices = llamacpp_result.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
        else:
            content = ""

        # Build Ollama-style response
        return {
            "model": model,
            "message": {"role": "assistant", "content": content},
            "done": True,
            "_routing": {
                "mode": "model_sharding",
                "backend": "llama.cpp-rpc",
                "coordinator": f"{self.coordinator.host}:{self.coordinator.port}",
                "rpc_backends": len(self.coordinator.rpc_backends),
                "description": "Model layers distributed across RPC backends",
            },
        }

    async def _check_if_model_fits_ollama(self, model: str) -> bool:
        """
        Check if model can fit on Ollama nodes by checking:
        1. Current VRAM availability across all nodes
        2. Estimated model size vs available memory

        Args:
            model: Model name

        Returns:
            True if model likely fits on at least one Ollama node
        """
        if not self.ollama_pool:
            return False

        # CRITICAL FIX: Check ACTUAL current VRAM on all nodes
        # Only route to RPC if model is LARGE and all GPUs are overwhelmed
        # For small models, it's better to wait for GPU than use RPC
        stats = self.ollama_pool.get_stats()
        node_performance = stats.get("node_performance", {})

        if node_performance:
            # Get VRAM for all nodes with GPU
            gpu_nodes = [
                node for node in node_performance.values() if node.get("gpu_free_mem", 0) > 0
            ]

            if gpu_nodes:
                all_gpus_overwhelmed = all(node.get("gpu_free_mem", 0) < 2000 for node in gpu_nodes)

                if all_gpus_overwhelmed:
                    # Only route large models to RPC when GPUs are overwhelmed
                    # Small models should wait for GPU (RPC would be slower)
                    profile = self._get_model_profile(model)
                    if profile.estimated_memory_gb > 30:  # Large models (70B+)
                        logger.warning(
                            f"⚠️  All GPUs overwhelmed + large model ({profile.estimated_memory_gb:.1f}GB) - routing to RPC sharding"
                        )
                        return False
                    else:
                        logger.info(
                            f"ℹ️  All GPUs overwhelmed but model is small ({profile.estimated_memory_gb:.1f}GB) - will wait for GPU"
                        )

        # Get model size estimate
        profile = self._get_model_profile(model)
        estimated_gb = profile.estimated_memory_gb

        # Add safety margin (models often use more memory than parameter count suggests)
        # Account for context, KV cache, etc.
        required_gb = estimated_gb * 1.5  # 50% safety margin

        logger.info(
            f"📊 Model '{model}' estimated size: {estimated_gb:.1f}GB (with margin: {required_gb:.1f}GB)"
        )

        # Check if any Ollama node has sufficient memory
        # Heuristic for routing:
        # - Models < 8GB: Small enough for Ollama task distribution
        # - Models >= 8GB: Use RPC sharding for better resource utilization

        if required_gb >= 8:
            logger.info(f"🔗 Model size ({required_gb:.1f}GB) - using RPC sharding")
            return False
        else:
            logger.info(f"✅ Model small enough ({required_gb:.1f}GB) - using Ollama")
            return True

    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics for both distribution modes."""
        stats = {
            "task_distribution": {
                "enabled": self.ollama_pool is not None,
                "description": "Load balance agent requests across Ollama nodes",
            },
            "model_sharding": {
                "enabled": self.enable_distributed,
                "description": "Distribute large models across llama.cpp RPC backends",
            },
            "routing_cache": {
                "cached_models": len(self.routing_cache),
                "ollama_models": [m for m, use_rpc in self.routing_cache.items() if not use_rpc],
                "rpc_models": [m for m, use_rpc in self.routing_cache.items() if use_rpc],
            },
        }

        if self.ollama_pool:
            stats["task_distribution"]["ollama_stats"] = self.ollama_pool.get_stats()

        if self.coordinator:
            stats["model_sharding"]["coordinator_active"] = True
            stats["model_sharding"]["model_loaded"] = self.coordinator_model
            stats["model_sharding"]["rpc_backends"] = len(self.coordinator.rpc_backends)

        return stats

    def _start_dashboard(self):
        """
        Start the unified observability dashboard in a background thread.

        The dashboard provides real-time monitoring via WebSockets:
        - System metrics (hosts, latency, success rate, GPU memory, Ray workers)
        - Real-time logs streaming
        - Ollama server activity monitoring
        - llama.cpp RPC activity monitoring
        - Embedded Ray dashboard iframe
        - Embedded Dask dashboard iframe

        Configured via environment variables:
        - SOLLOL_DASHBOARD=true|false (default: true)
        - SOLLOL_DASHBOARD_PORT=8080 (default: 8080)
        - SOLLOL_DASHBOARD_DASK=true|false (default: true)
        """
        try:
            from .dashboard_launcher import launch_dashboard_subprocess
            from .dashboard_log_hooks import install_log_hook_main

            # Get Redis URL from environment
            redis_url = os.getenv("SOLLOL_REDIS_URL", "redis://localhost:6379")

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
            install_log_hook_main(redis_url)

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
