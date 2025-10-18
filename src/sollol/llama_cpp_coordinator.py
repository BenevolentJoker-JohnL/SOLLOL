"""
llama.cpp Coordinator for Distributed Inference

Manages llama-server instances configured with --rpc flag to coordinate
distributed inference across multiple RPC backend nodes.

Architecture:
    Python Client → llama-server (coordinator) → RPC servers (workers)

The coordinator (llama-server) handles:
- Automatic layer slicing across RPC backends
- Inter-node communication via RPC protocol
- Standard HTTP API (Ollama-compatible)
- Intelligent load balancing across CPU/GPU resources
- Parallel inference with full resource utilization

We manage starting the coordinator and intelligently selecting healthy RPC backends.
"""

import asyncio
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx

if TYPE_CHECKING:
    from sollol.rpc_registry import RPCBackendRegistry

from .network_observer import (
    log_rpc_request,
    log_rpc_response,
    log_rpc_error,
    EventType,
    get_observer,
)

logger = logging.getLogger(__name__)


@dataclass
class RPCBackend:
    """Configuration for an RPC backend node."""

    host: str
    port: int = 50052

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


class LlamaCppCoordinator:
    """
    Manages llama-server as coordinator for distributed inference.

    The coordinator automatically distributes model layers across RPC
    backends and handles all inter-node communication.

    Usage:
        coordinator = LlamaCppCoordinator(
            model_path="/path/to/model.gguf",
            rpc_backends=[
                RPCBackend("192.168.1.10", 50052),
                RPCBackend("192.168.1.11", 50052)
            ]
        )

        await coordinator.start()

        # Use standard HTTP API
        response = await coordinator.generate("Hello world")
    """

    def __init__(
        self,
        model_path: str,
        rpc_backends: List[RPCBackend],
        host: str = "127.0.0.1",
        port: int = 8080,
        n_gpu_layers: int = 0,  # Use 0 for RPC - distributes across CPU nodes
        ctx_size: int = 2048,
        rpc_registry: Optional["RPCBackendRegistry"] = None,
    ):
        """
        Initialize coordinator.

        Args:
            model_path: Path to .gguf model file
            rpc_backends: List of RPC backend nodes
            host: Host to bind llama-server to
            port: Port for llama-server HTTP API
            n_gpu_layers: Number of layers to attempt GPU offload
            ctx_size: Context window size
            rpc_registry: Optional registry for intelligent backend selection
        """
        self.model_path = model_path
        self.rpc_backends = rpc_backends
        self.host = host
        self.port = port
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self.rpc_registry = rpc_registry

        self.process: Optional[subprocess.Popen] = None
        self.http_client = httpx.AsyncClient(timeout=300.0)

        # Heartbeat for live monitoring - configurable via environment variable
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_interval = int(os.getenv("SOLLOL_RPC_HEARTBEAT_INTERVAL", "600"))  # seconds (10 minutes)

    def _get_healthy_backends(self) -> List[RPCBackend]:
        """
        Get list of healthy RPC backends using registry if available.

        Returns:
            List of healthy backends, or all backends if no registry
        """
        if not self.rpc_registry:
            return self.rpc_backends

        # Get healthy backends from registry
        healthy = self.rpc_registry.get_healthy_backends()

        if not healthy:
            logger.warning("No healthy RPC backends found, using all configured backends")
            return self.rpc_backends

        logger.info(f"Using {len(healthy)}/{len(self.rpc_backends)} healthy RPC backends")

        # Convert registry backends to RPCBackend objects
        return [RPCBackend(host=b.host, port=b.port) for b in healthy]

    async def start(self):
        """
        Start llama-server coordinator with healthy RPC backends.

        Uses RPCBackendRegistry if available to filter to only healthy backends.

        Command format:
            llama-server \\
              --model model.gguf \\
              --host 0.0.0.0 \\
              --port 8080 \\
              --rpc node1:50052,node2:50052,node3:50052 \\
              --gpu-layers 0 \\
              --ctx-size 8192
        """
        # Get healthy backends (uses registry if available)
        healthy_backends = self._get_healthy_backends()

        if not healthy_backends:
            raise RuntimeError("No healthy RPC backends available")

        # Build RPC backend address list
        rpc_addresses = ",".join([backend.address for backend in healthy_backends])

        # Build llama-server command
        # For RPC: use --gpu-layers 0 to distribute across CPU nodes
        # llama.cpp automatically splits layers across RPC backends
        cmd = [
            "llama-server",
            "--model",
            self.model_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--rpc",
            rpc_addresses,
            "--gpu-layers",
            "0",  # CPU-only for RPC distribution
            "--ctx-size",
            str(self.ctx_size),
            "--no-warmup",  # Skip slow warmup - inference will be fast enough
        ]

        logger.info(f"Starting llama-server coordinator: {' '.join(cmd)}")

        # Log coordinator startup to routing decisions for dashboard visibility
        observer = get_observer()
        observer.log_event(
            EventType.COORDINATOR_START,
            backend=f"{self.host}:{self.port}",
            details={
                "model": os.path.basename(self.model_path),
                "coordinator": f"{self.host}:{self.port}",
                "rpc_backends": [b.address for b in healthy_backends],
                "num_backends": len(healthy_backends),
                "model_path": self.model_path,
                "ctx_size": self.ctx_size,
                "command": ' '.join(cmd)
            },
            severity="info"
        )

        try:
            # Use subprocess.PIPE to capture output for parsing layer distribution
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )

            # Start a thread to parse llama-server output for layer distribution
            self._output_parser_task = threading.Thread(
                target=self._parse_llama_output,
                daemon=True,
                name="LlamaServerOutputParser"
            )
            self._output_parser_task.start()

            # Wait for server to be ready
            try:
                await self._wait_for_ready()
            except TimeoutError as e:
                # Coordinator failed to start - check if process crashed
                if self.process.poll() is not None:
                    # Process exited
                    exit_code = self.process.returncode
                    observer.log_event(
                        EventType.COORDINATOR_STOP,
                        backend=f"{self.host}:{self.port}",
                        details={
                            "model": os.path.basename(self.model_path),
                            "coordinator": f"{self.host}:{self.port}",
                            "reason": f"Crashed with exit code {exit_code}",
                            "error": "Process terminated during startup",
                            "log_file": "/tmp/llama-server.log"
                        },
                        severity="error"
                    )
                    raise RuntimeError(f"llama-server crashed with exit code {exit_code}. Check /tmp/llama-server.log for details")
                else:
                    # Still running but not responding
                    observer.log_event(
                        EventType.COORDINATOR_STOP,
                        backend=f"{self.host}:{self.port}",
                        details={
                            "model": os.path.basename(self.model_path),
                            "coordinator": f"{self.host}:{self.port}",
                            "reason": "Timeout waiting for health endpoint",
                            "error": str(e),
                            "log_file": "/tmp/llama-server.log"
                        },
                        severity="error"
                    )
                    raise

            logger.info(
                f"✅ llama-server coordinator started on {self.host}:{self.port} "
                f"with {len(healthy_backends)} RPC backends"
            )

            # Log successful startup with backend details
            observer.log_event(
                EventType.RPC_BACKEND_CONNECT,
                backend=f"{self.host}:{self.port}",
                details={
                    "status": "ready",
                    "rpc_backends": len(healthy_backends),
                    "rpc_addresses": [b.address for b in healthy_backends],
                    "model": os.path.basename(self.model_path),
                    "type": "coordinator_ready"
                },
                severity="info"
            )

            # Start heartbeat loop for dashboard visibility
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            logger.debug("RPC heartbeat monitoring started")

        except Exception as e:
            logger.error(f"Failed to start llama-server: {e}")
            # Log failure to routing decisions
            observer.log_event(
                EventType.COORDINATOR_STOP,
                backend=f"{self.host}:{self.port}",
                details={
                    "model": os.path.basename(self.model_path),
                    "coordinator": f"{self.host}:{self.port}",
                    "reason": "Startup failed",
                    "error": str(e),
                    "log_file": "/tmp/llama-server.log"
                },
                severity="error"
            )
            raise

    def _parse_llama_output(self):
        """
        Parse llama-server output to extract layer distribution and memory allocation.

        Publishes COORDINATOR_MODEL_LOAD events when model sharding is detected.
        Also writes all output to /tmp/llama-server.log for debugging.
        """
        log_file = open("/tmp/llama-server.log", "w")
        observer = get_observer()

        # Regex patterns for layer distribution and memory info
        # Example: "llm_load_tensors: offloading 32 repeating layers to GPU"
        # Example: "llm_load_tensors: using CUDA for GPU acceleration"
        # Example: "llama_model_load: total VRAM used: 4096 MB"
        layer_pattern = re.compile(r"offloading (\d+) .*?layers? to (GPU|CPU|RPC)")
        memory_pattern = re.compile(r"total (VRAM|RAM|memory) used:?\s*(\d+\.?\d*)\s*(MB|GB|KiB)", re.IGNORECASE)
        backend_pattern = re.compile(r"RPC backend: ([\w\.\-]+:\d+)")

        model_load_details = {
            "layers_offloaded": {},
            "memory_allocated": {},
            "rpc_backends_used": []
        }

        try:
            for line in iter(self.process.stdout.readline, ''):
                if not line:
                    break

                # Write to log file for debugging
                log_file.write(line)
                log_file.flush()

                # Parse layer distribution
                layer_match = layer_pattern.search(line)
                if layer_match:
                    num_layers = int(layer_match.group(1))
                    target = layer_match.group(2)
                    model_load_details["layers_offloaded"][target] = num_layers

                    logger.info(f"🧠 Offloading {num_layers} layers to {target}")

                # Parse memory allocation
                memory_match = memory_pattern.search(line)
                if memory_match:
                    mem_type = memory_match.group(1).upper()
                    mem_value = float(memory_match.group(2))
                    mem_unit = memory_match.group(3)

                    # Convert to MB
                    if mem_unit == "GB":
                        mem_value *= 1024
                    elif mem_unit == "KiB":
                        mem_value /= 1024

                    model_load_details["memory_allocated"][mem_type] = {
                        "value_mb": mem_value,
                        "display": f"{mem_value:.2f} MB"
                    }

                    logger.info(f"💾 {mem_type} allocated: {mem_value:.2f} MB")

                # Parse RPC backend connections
                backend_match = backend_pattern.search(line)
                if backend_match:
                    backend_addr = backend_match.group(1)
                    if backend_addr not in model_load_details["rpc_backends_used"]:
                        model_load_details["rpc_backends_used"].append(backend_addr)
                        logger.info(f"🔗 Connected to RPC backend: {backend_addr}")

                # Detect model load completion
                if "model loaded" in line.lower() or "ready" in line.lower():
                    # Publish model load event with sharding details
                    if model_load_details["layers_offloaded"] or model_load_details["memory_allocated"]:
                        observer.log_event(
                            EventType.COORDINATOR_MODEL_LOAD,
                            backend=f"{self.host}:{self.port}",
                            details={
                                "model": os.path.basename(self.model_path),
                                "layers_offloaded": model_load_details["layers_offloaded"],
                                "memory_allocated": model_load_details["memory_allocated"],
                                "rpc_backends": len(self.rpc_backends),
                                "rpc_addresses": [b.address for b in self.rpc_backends],
                                "status": "loaded",
                                "type": "model_sharding"
                            },
                            severity="info"
                        )

                        logger.info(
                            f"✅ Model sharded across {len(self.rpc_backends)} RPC backends: "
                            f"{model_load_details}"
                        )

                        # Reset for next model load
                        model_load_details = {
                            "layers_offloaded": {},
                            "memory_allocated": {},
                            "rpc_backends_used": []
                        }

        except Exception as e:
            logger.error(f"Error parsing llama-server output: {e}")
        finally:
            log_file.close()

    async def _wait_for_ready(self, timeout: float = 1200.0):  # 20 minutes for large models
        """Wait for llama-server to be ready."""
        start_time = asyncio.get_event_loop().time()

        while True:
            try:
                response = await self.http_client.get(f"http://{self.host}:{self.port}/health")
                if response.status_code == 200:
                    return
            except:
                pass

            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError("llama-server did not start in time")

            await asyncio.sleep(0.5)

    async def _heartbeat_loop(self):
        """Periodically log RPC coordinator heartbeat for dashboard visibility."""
        logger.debug("RPC heartbeat loop started")

        while True:
            try:
                await asyncio.sleep(self._heartbeat_interval)

                # Log heartbeat event
                observer = get_observer()
                observer.log_event(
                    EventType.RPC_BACKEND_CONNECT,
                    backend=f"{self.host}:{self.port}",
                    details={
                        "model": "coordinator",
                        "rpc_backends": len(self.rpc_backends),
                        "rpc_addresses": [b.address for b in self.rpc_backends],
                        "status": "healthy",
                        "type": "heartbeat"
                    },
                    severity="info"
                )

                logger.debug(f"RPC heartbeat: {len(self.rpc_backends)} backends connected")

            except asyncio.CancelledError:
                logger.debug("RPC heartbeat loop cancelled")
                break
            except Exception as e:
                logger.error(f"RPC heartbeat error: {e}")

    async def generate(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using distributed inference.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Response from llama-server
        """
        # Log request to observer
        backend_key = f"{self.host}:{self.port}"
        model = kwargs.get("model", "distributed")

        log_rpc_request(
            backend=backend_key,
            model=model,
            rpc_backends=len(self.rpc_backends),
            operation="generate"
        )

        start_time = time.time()

        try:
            payload = {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "stream": False,
                **kwargs,
            }

            response = await self.http_client.post(
                f"http://{self.host}:{self.port}/completion", json=payload
            )
            response.raise_for_status()

            latency_ms = (time.time() - start_time) * 1000

            # Log successful response
            log_rpc_response(
                backend=backend_key,
                model=model,
                latency_ms=latency_ms,
                rpc_backends=len(self.rpc_backends)
            )

            return response.json()

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            # Log error
            log_rpc_error(
                backend=backend_key,
                model=model,
                error=str(e),
                latency_ms=latency_ms
            )
            raise

    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Chat completion using distributed inference.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Response from llama-server
        """
        # Log request to observer
        backend_key = f"{self.host}:{self.port}"
        model = kwargs.get("model", "distributed")

        log_rpc_request(
            backend=backend_key,
            model=model,
            rpc_backends=len(self.rpc_backends),
            operation="chat"
        )

        start_time = time.time()

        try:
            payload = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
                **kwargs,
            }

            response = await self.http_client.post(
                f"http://{self.host}:{self.port}/v1/chat/completions", json=payload
            )
            response.raise_for_status()

            latency_ms = (time.time() - start_time) * 1000

            # Log successful response
            log_rpc_response(
                backend=backend_key,
                model=model,
                latency_ms=latency_ms,
                rpc_backends=len(self.rpc_backends)
            )

            return response.json()

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            # Log error
            log_rpc_error(
                backend=backend_key,
                model=model,
                error=str(e),
                latency_ms=latency_ms
            )
            raise

    async def stop(self):
        """Stop the llama-server coordinator."""
        # Stop heartbeat task
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            logger.debug("RPC heartbeat monitoring stopped")

        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

            logger.info("llama-server coordinator stopped")

        await self.http_client.aclose()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


async def start_rpc_backend(
    host: str = "0.0.0.0", port: int = 50052, mem_mb: int = 2048
) -> subprocess.Popen:
    """
    Start an RPC backend server on a node.

    This should be run on each worker node.

    Command:
        rpc-server --host 0.0.0.0 --port 50052 --mem 2048

    Args:
        host: Host to bind to
        port: Port for RPC server
        mem_mb: Memory limit in MB

    Returns:
        Process handle
    """
    cmd = ["rpc-server", "--host", host, "--port", str(port), "--mem", str(mem_mb)]

    logger.info(f"Starting RPC backend: {' '.join(cmd)}")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Give it a moment to start
    await asyncio.sleep(1)

    logger.info(f"✅ RPC backend started on {host}:{port}")

    return process
