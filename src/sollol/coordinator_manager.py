"""
SOLLOL Coordinator Manager
Handles automatic coordinator startup, health monitoring, and integration with observability.
"""

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class CoordinatorConfig:
    """Configuration for llama.cpp coordinator"""

    host: str = "127.0.0.1"
    port: int = 18080
    model_path: Optional[str] = None
    rpc_backends: List[str] = None
    ctx_size: int = None  # Will use env SOLLOL_CTX_SIZE or default 8192
    parallel: int = 1
    auto_start: bool = True

    def __post_init__(self):
        """Apply environment variable defaults"""
        if self.ctx_size is None:
            self.ctx_size = int(os.getenv("SOLLOL_CTX_SIZE", "8192"))


class CoordinatorManager:
    """
    Manages llama.cpp coordinator lifecycle and health monitoring.

    Features:
    - Auto-detect GGUF models from Ollama blob storage
    - Auto-start coordinator with RPC backends
    - Health monitoring and metrics
    - Integration with SOLLOL dashboard
    """

    def __init__(self, config: CoordinatorConfig, redis_url: Optional[str] = None):
        self.config = config
        self.redis_url = redis_url
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self._health_check_interval = 30  # seconds
        self._last_health_check = 0
        self._metrics: Dict[str, Any] = {}

    async def ensure_running(self) -> bool:
        """
        Ensure coordinator is running, start if needed.

        Returns:
            True if coordinator is running and healthy
        """
        # Check if already running
        if await self.check_health():
            logger.info(f"✅ Coordinator already running at {self.config.host}:{self.config.port}")
            self.is_running = True

            # Try to detect RPC backends from running process if not configured
            # Check if backends are dummy/empty (coordinator:0 or None)
            needs_detection = (
                not self.config.rpc_backends
                or len(self.config.rpc_backends) == 0
                or (
                    len(self.config.rpc_backends) == 1
                    and "coordinator" in self.config.rpc_backends[0]
                )
            )

            if needs_detection:
                detected_backends = self._detect_running_backends()
                if detected_backends:
                    self.config.rpc_backends = detected_backends
                    logger.info(
                        f"🔍 Detected {len(detected_backends)} RPC backend(s) from running coordinator:"
                    )
                    for backend in detected_backends:
                        logger.info(f"   • {backend}")

                    # Publish to Redis for dashboard
                    self._publish_backends_to_redis()
                else:
                    logger.warning(
                        "⚠️  Could not detect RPC backends from running coordinator process"
                    )

            return True

        if not self.config.auto_start:
            logger.warning(f"⚠️  Coordinator not running and auto_start=False")
            return False

        # Auto-detect model if not specified
        if not self.config.model_path:
            self.config.model_path = self._detect_ollama_model()

        if not self.config.model_path:
            logger.error("❌ No model path specified and auto-detection failed")
            return False

        # ALWAYS read RPC backends from rpc_backends.conf (source of truth)
        # This overrides any backends passed from config files like .synapticllamas.json
        self.config.rpc_backends = await self._discover_rpc_backends()

        # Start coordinator
        return await self.start()

    def _publish_backends_to_redis(self):
        """Publish RPC backends to Redis for dashboard visibility."""
        if not self.redis_url or not self.config.rpc_backends:
            return

        # Don't publish dummy backends (coordinator:0)
        if (
            len(self.config.rpc_backends) == 1
            and "coordinator" in self.config.rpc_backends[0]
            and ":0" in self.config.rpc_backends[0]
        ):
            logger.debug("Skipping Redis publish for dummy backend (coordinator:0)")
            return

        try:
            import json

            import redis

            redis_client = redis.from_url(self.redis_url)

            # Get existing metadata
            metadata_json = redis_client.get("sollol:router:metadata")
            if metadata_json:
                metadata = json.loads(metadata_json)
            else:
                metadata = {"nodes": [], "rpc_backends": [], "metrics": {}}

            # Update RPC backends with real addresses
            metadata["rpc_backends"] = [
                {"host": addr.split(":")[0], "port": int(addr.split(":")[1])}
                for addr in self.config.rpc_backends
            ]

            # Publish back to Redis
            redis_client.set("sollol:router:metadata", json.dumps(metadata))
            logger.info(f"📊 Published {len(self.config.rpc_backends)} RPC backends to dashboard")

        except Exception as e:
            logger.debug(f"Could not publish backends to Redis: {e}")

    def _detect_running_backends(self) -> Optional[List[str]]:
        """
        Detect RPC backends from running coordinator process.

        Returns:
            List of RPC backend addresses (host:port) or None
        """
        try:
            import re
            import subprocess

            # Find llama-server process on the configured port
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=5)

            # Look for llama-server with our port and extract --rpc argument
            for line in result.stdout.split("\n"):
                if f"llama-server" in line and f"--port {self.config.port}" in line:
                    # Extract --rpc argument
                    rpc_match = re.search(r"--rpc\s+([^\s]+)", line)
                    if rpc_match:
                        rpc_arg = rpc_match.group(1)
                        # Split by comma to get individual backends
                        backends = [addr.strip() for addr in rpc_arg.split(",")]
                        return backends

            return None

        except Exception as e:
            logger.debug(f"Could not detect running backends: {e}")
            return None

    def _detect_ollama_model(self) -> Optional[str]:
        """
        Auto-detect GGUF model from Ollama blob storage.

        Priority:
        1. Environment variable SOLLOL_MODEL_PATH
        2. codellama:13b from Ollama blobs
        3. First GGUF file found in Ollama blobs
        """
        # Check environment variable
        env_model = os.getenv("SOLLOL_MODEL_PATH")
        if env_model and Path(env_model).exists():
            logger.info(f"📍 Using model from SOLLOL_MODEL_PATH: {env_model}")
            return env_model

        # Check Ollama blob storage
        ollama_blob_dir = Path("/usr/share/ollama/.ollama/models/blobs")
        if not ollama_blob_dir.exists():
            ollama_blob_dir = Path.home() / ".ollama/models/blobs"

        if not ollama_blob_dir.exists():
            logger.warning("⚠️  Ollama blob directory not found")
            return None

        # Look for codellama:13b specifically (known hash)
        codellama_13b_hash = (
            "sha256-e73cc17c718156e5ad34b119eb363e2c10389a503673f9c36144c42dfde8334c"
        )
        codellama_path = ollama_blob_dir / codellama_13b_hash

        if codellama_path.exists():
            logger.info(f"🎯 Found codellama:13b at {codellama_path}")
            return str(codellama_path)

        # Fallback: find any large GGUF-like file
        blobs = list(ollama_blob_dir.glob("sha256-*"))
        if blobs:
            # Sort by size, take largest (likely a model)
            largest = max(blobs, key=lambda p: p.stat().st_size)
            size_gb = largest.stat().st_size / (1024**3)

            if size_gb > 1.0:  # Only use if > 1GB (likely a model)
                logger.info(f"📦 Auto-detected model: {largest.name} ({size_gb:.1f}GB)")
                return str(largest)

        logger.warning("⚠️  No suitable model found in Ollama blobs")
        return None

    async def _discover_rpc_backends(self) -> List[str]:
        """
        Discover RPC backends with priority order:
        1. Read from rpc_backends.conf file (highest priority)
        2. Auto-discover from Redis registry (fallback)

        Returns:
            List of RPC backend addresses (host:port)
        """
        # PRIORITY 1: Read from config file (source of truth)
        try:
            from pathlib import Path

            config_file = Path("/home/joker/SOLLOL/rpc_backends.conf")
            if config_file.exists():
                backends = []
                with open(config_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        # Skip comments and empty lines
                        if line and not line.startswith("#"):
                            backends.append(line)
                if backends:
                    logger.info(f"📄 Loaded {len(backends)} RPC backend(s) from {config_file}:")
                    for addr in backends:
                        logger.info(f"   • {addr}")
                    return backends
        except Exception as e:
            logger.debug(f"Could not read RPC config file: {e}")

        # PRIORITY 2: Auto-discover from Redis registry
        if not self.redis_url:
            logger.info("ℹ️  No Redis URL configured, skipping RPC backend discovery")
            return []

        try:
            from sollol.rpc_discovery import auto_discover_rpc_backends

            backends = auto_discover_rpc_backends()
            if backends:
                addresses = [f"{b['host']}:{b['port']}" for b in backends]
                logger.info(f"🔍 Discovered {len(addresses)} RPC backend(s):")
                for addr in addresses:
                    logger.info(f"   • {addr}")
                return addresses
            else:
                logger.info("ℹ️  No RPC backends discovered")
                return []

        except Exception as e:
            logger.warning(f"⚠️  RPC backend discovery failed: {e}")
            return []

    async def start(self) -> bool:
        """
        Start the coordinator process.

        Returns:
            True if started successfully
        """
        if self.is_running:
            logger.warning("⚠️  Coordinator already running")
            return True

        # Build command
        cmd = [
            "llama-server",
            "--model",
            self.config.model_path,
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "--ctx-size",
            str(self.config.ctx_size),
            "--parallel",
            str(self.config.parallel),
        ]

        # Add RPC backends
        if self.config.rpc_backends:
            rpc_str = ",".join(self.config.rpc_backends)
            cmd.extend(["--rpc", rpc_str])

        logger.info(f"🚀 Starting coordinator: {' '.join(cmd)}")

        try:
            # Start process
            log_path = f"/tmp/coordinator-{self.config.port}.log"
            log_file = open(log_path, "w")

            self.process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # Detach from parent
            )

            logger.info(f"📝 Coordinator logs: {log_path}")
            logger.info(f"🔧 Coordinator PID: {self.process.pid}")

            # Wait for startup (model loading)
            logger.info("⏳ Waiting for coordinator to be ready (model loading ~40s)...")

            for i in range(12):  # 60 seconds max (5s * 12)
                await asyncio.sleep(5)
                if await self.check_health():
                    self.is_running = True
                    logger.info(
                        f"✅ Coordinator started successfully on {self.config.host}:{self.config.port}"
                    )
                    return True
                logger.info(f"   ⏳ Attempt {i+1}/12 - still loading...")

            logger.error("❌ Coordinator failed to start within 60 seconds")
            return False

        except Exception as e:
            logger.error(f"❌ Failed to start coordinator: {e}")
            return False

    async def check_health(self) -> bool:
        """
        Check if coordinator is healthy.

        Returns:
            True if coordinator responds to health check
        """
        try:
            url = f"http://{self.config.host}:{self.config.port}/health"
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)

                if response.status_code == 200:
                    self._last_health_check = time.time()
                    self._metrics = response.json()
                    return True

        except Exception:
            pass

        return False

    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get coordinator metrics.

        Returns:
            Dictionary of metrics (requests, latency, backends, etc.)
        """
        # Update metrics if stale
        if time.time() - self._last_health_check > self._health_check_interval:
            await self.check_health()

        return {
            "coordinator": {
                "host": self.config.host,
                "port": self.config.port,
                "healthy": self.is_running,
                "uptime": time.time() - self._last_health_check if self.is_running else 0,
            },
            "rpc_backends": {
                "configured": len(self.config.rpc_backends) if self.config.rpc_backends else 0,
                "addresses": self.config.rpc_backends or [],
            },
            "model": {
                "path": self.config.model_path,
                "ctx_size": self.config.ctx_size,
            },
            **self._metrics,
        }

    async def shutdown(self):
        """Stop the coordinator process."""
        if self.process and self.process.poll() is None:
            logger.info(f"🛑 Stopping coordinator (PID: {self.process.pid})")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
                logger.info("✅ Coordinator stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("⚠️  Coordinator didn't stop gracefully, killing...")
                self.process.kill()

        self.is_running = False

    async def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status for CLI display.

        Returns:
            Status dictionary with coordinator and RPC backend info
        """
        is_healthy = await self.check_health()

        status = {
            "coordinator": {
                "url": f"http://{self.config.host}:{self.config.port}",
                "healthy": is_healthy,
                "pid": self.process.pid if self.process else None,
            },
            "rpc_backends": [],
            "model": {
                "path": self.config.model_path,
                "name": Path(self.config.model_path).name if self.config.model_path else None,
            },
        }

        # Check each RPC backend
        if self.config.rpc_backends:
            for backend_addr in self.config.rpc_backends:
                host, port = backend_addr.split(":")
                # TODO: Add RPC backend health check via gRPC
                status["rpc_backends"].append(
                    {
                        "address": backend_addr,
                        "host": host,
                        "port": int(port),
                        "healthy": None,  # Will be populated by RPC health check
                    }
                )

        return status
