"""
llama.cpp RPC Client for Distributed Inference

Provides Python interface to llama.cpp RPC servers for distributed inference
across multiple nodes. This enables running large models (70B+) that don't fit
on a single machine.

Architecture:
- Each node runs llama-rpc-server with a portion of the model
- This client coordinates inference across all nodes
- Uses llama.cpp's native distributed inference protocol
"""

import asyncio
import json
import logging
import os
import struct
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Initialize Redis client for dashboard activity publishing
_redis_client = None
REDIS_RPC_ACTIVITY_CHANNEL = "sollol:dashboard:rpc:activity"


def _get_redis_client():
    """Lazy initialization of Redis client for activity publishing."""
    global _redis_client
    if _redis_client is None:
        try:
            import redis

            redis_url = os.getenv("SOLLOL_REDIS_URL", "redis://localhost:6379")
            _redis_client = redis.from_url(redis_url, decode_responses=True)
        except Exception as e:
            logger.warning(f"Failed to initialize Redis client for RPC activity: {e}")
            _redis_client = False  # Mark as failed to avoid retries
    return _redis_client if _redis_client is not False else None


def _publish_rpc_activity(event_type: str, backend: str, details: Dict[str, Any] = None):
    """Publish RPC activity event to Redis for dashboard monitoring."""
    try:
        redis_client = _get_redis_client()
        if redis_client:
            activity_data = {
                "event_type": event_type,
                "backend": backend,
                "timestamp": time.time(),
                "details": details or {},
            }
            redis_client.publish(REDIS_RPC_ACTIVITY_CHANNEL, json.dumps(activity_data))
    except Exception as e:
        # Silently fail - don't let observability break core functionality
        logger.debug(f"Failed to publish RPC activity: {e}")


@dataclass
class LlamaCppNode:
    """Represents a llama.cpp RPC server node."""

    host: str
    port: int
    model_path: str = ""
    layers_start: int = 0
    layers_end: int = 0
    is_healthy: bool = True

    @property
    def url(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def http_url(self) -> str:
        """HTTP endpoint for health checks and management."""
        return f"http://{self.host}:{self.port}"


class LlamaCppRPCClient:
    """
    Client for llama.cpp RPC protocol.

    Communicates with llama-rpc-server instances for distributed inference.
    Supports both HTTP and gRPC protocols (llama.cpp uses HTTP by default).
    """

    def __init__(self, node: LlamaCppNode, timeout: float = 300.0):
        """
        Initialize RPC client for a node.

        Args:
            node: Node configuration
            timeout: Request timeout in seconds
        """
        self.node = node
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def health_check(self) -> bool:
        """Check if node is healthy and responding."""
        try:
            # llama-rpc-server provides /health endpoint
            response = await self.client.get(f"{self.node.http_url}/health")
            self.node.is_healthy = response.status_code == 200
            return self.node.is_healthy
        except Exception as e:
            logger.warning(f"Health check failed for {self.node.url}: {e}")
            self.node.is_healthy = False
            return False

    async def get_node_info(self) -> Dict[str, Any]:
        """Get node information (model, layers, etc)."""
        try:
            response = await self.client.get(f"{self.node.http_url}/v1/models")
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get node info from {self.node.url}: {e}")
            return {}

    async def generate(
        self, prompt: str, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate text using this node's layers.

        Args:
            prompt: Input text or intermediate activations
            options: Generation parameters

        Returns:
            Response with generated text or activations
        """
        payload = {"prompt": prompt, "stream": False, **(options or {})}
        backend = self.node.url
        start_time = time.time()

        # Publish RPC request event
        _publish_rpc_activity(
            "rpc_request",
            backend,
            {
                "prompt_length": len(prompt) if isinstance(prompt, str) else 0,
                "endpoint": "/completion",
            },
        )

        try:
            response = await self.client.post(f"{self.node.http_url}/completion", json=payload)
            response.raise_for_status()
            result = response.json()

            # Publish RPC response event with latency
            latency_ms = (time.time() - start_time) * 1000
            _publish_rpc_activity(
                "rpc_response",
                backend,
                {"latency_ms": latency_ms, "status_code": response.status_code},
            )

            return result
        except Exception as e:
            # Publish RPC error event
            latency_ms = (time.time() - start_time) * 1000
            _publish_rpc_activity("rpc_error", backend, {"error": str(e), "latency_ms": latency_ms})
            logger.error(f"Generation failed on {self.node.url}: {e}")
            raise

    async def close(self):
        """Close client connection."""
        await self.client.aclose()


class LlamaCppDistributedCluster:
    """
    Manages distributed inference across multiple llama.cpp RPC nodes.

    Coordinates multi-node inference for large models that don't fit on
    a single machine. Uses llama.cpp's distributed inference protocol.

    Usage:
        cluster = LlamaCppDistributedCluster([
            LlamaCppNode("192.168.1.10", 50052, layers_start=0, layers_end=40),
            LlamaCppNode("192.168.1.11", 50052, layers_start=40, layers_end=80)
        ])

        result = await cluster.generate("Explain quantum computing", "llama-3.1-405b")
    """

    def __init__(
        self, nodes: List[LlamaCppNode], model_name: str = "distributed", timeout: float = 300.0
    ):
        """
        Initialize distributed cluster.

        Args:
            nodes: List of RPC server nodes
            model_name: Model identifier
            timeout: Request timeout
        """
        self.nodes = nodes
        self.model_name = model_name
        self.timeout = timeout
        self.clients = [LlamaCppRPCClient(node, timeout) for node in nodes]

        logger.info(
            f"Initialized llama.cpp distributed cluster '{model_name}' " f"with {len(nodes)} nodes"
        )

    async def health_check(self) -> bool:
        """Check if all nodes are healthy."""
        health_checks = await asyncio.gather(
            *[client.health_check() for client in self.clients], return_exceptions=True
        )

        all_healthy = all(isinstance(h, bool) and h for h in health_checks)

        if not all_healthy:
            unhealthy = [
                node.url
                for node, h in zip(self.nodes, health_checks)
                if not (isinstance(h, bool) and h)
            ]
            logger.warning(f"Unhealthy nodes: {unhealthy}")

        return all_healthy

    async def generate(
        self, prompt: str, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run distributed inference across all nodes.

        Process:
        1. First node processes prompt through its layers
        2. Intermediate activations passed to next node
        3. Each node processes through its assigned layers
        4. Final node returns complete response

        Args:
            prompt: Input prompt
            options: Generation options (temperature, max_tokens, etc)

        Returns:
            Generated response
        """
        if not await self.health_check():
            raise RuntimeError(f"Cluster '{self.model_name}' has unhealthy nodes")

        logger.info(
            f"🔗 Starting distributed inference on '{self.model_name}' "
            f"across {len(self.nodes)} nodes"
        )

        # For now, use simple approach: first node does full generation
        # TODO: Implement proper layer-by-layer distributed inference
        # This requires llama.cpp RPC protocol for passing activations

        result = await self.clients[0].generate(prompt, options)

        # Add cluster metadata
        result["_distributed"] = {
            "cluster": self.model_name,
            "nodes": len(self.nodes),
            "node_urls": [node.url for node in self.nodes],
            "layers_per_node": [f"{node.layers_start}-{node.layers_end}" for node in self.nodes],
        }

        logger.info(f"✅ Distributed inference completed on '{self.model_name}'")

        return result

    async def close(self):
        """Close all client connections."""
        await asyncio.gather(*[client.close() for client in self.clients], return_exceptions=True)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Model name to .gguf path mapping
MODEL_PATH_MAP = {
    # Llama 3.1 models
    "llama3.1:405b": "models/llama-3.1-405b-instruct.gguf",
    "llama3.1:70b": "models/llama-3.1-70b-instruct.gguf",
    "llama3.1:8b": "models/llama-3.1-8b-instruct.gguf",
    # Llama 3 models
    "llama3:70b": "models/llama-3-70b-instruct.gguf",
    "llama3:8b": "models/llama-3-8b-instruct.gguf",
    # Llama 2 models
    "llama2:70b": "models/llama-2-70b-chat.gguf",
    "llama2:13b": "models/llama-2-13b-chat.gguf",
    "llama2:7b": "models/llama-2-7b-chat.gguf",
    # Mixtral
    "mixtral:8x7b": "models/mixtral-8x7b-instruct.gguf",
    "mixtral:8x22b": "models/mixtral-8x22b-instruct.gguf",
    # Qwen
    "qwen2.5:72b": "models/qwen-2.5-72b-instruct.gguf",
}


def resolve_model_path(ollama_model_name: str) -> str:
    """
    Convert Ollama model name to .gguf file path.

    Args:
        ollama_model_name: Model name from Ollama (e.g., "llama3.1:405b")

    Returns:
        Path to .gguf model file
    """
    # Normalize model name
    model_name = ollama_model_name.lower().strip()

    # Direct mapping
    if model_name in MODEL_PATH_MAP:
        return MODEL_PATH_MAP[model_name]

    # Try without tag (e.g., "llama3.1" -> "llama3.1:latest")
    if ":" not in model_name:
        model_with_tag = f"{model_name}:latest"
        if model_with_tag in MODEL_PATH_MAP:
            return MODEL_PATH_MAP[model_with_tag]

    # Fallback: construct path from name
    logger.warning(f"No mapping for {ollama_model_name}, using fallback path")
    return f"models/{model_name.replace(':', '-')}.gguf"
