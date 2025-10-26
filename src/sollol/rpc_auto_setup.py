"""
RPC Auto-Setup - Automatically configure llama.cpp RPC backends

This module automatically:
1. Discovers running RPC servers
2. If none found, checks if llama.cpp is built
3. If not built, automatically builds it
4. Starts RPC servers automatically
"""

import logging
import multiprocessing
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from sollol.rpc_discovery import check_rpc_server, discover_rpc_backends

logger = logging.getLogger(__name__)


class RPCAutoSetup:
    """Automatically setup and manage llama.cpp RPC backends."""

    def __init__(
        self,
        llama_dir: Optional[str] = None,
        auto_build: bool = True,
        auto_start: bool = True,
        default_port: int = 50052,
        default_host: str = "127.0.0.1",
    ):
        """
        Initialize RPC auto-setup.

        Args:
            llama_dir: Directory where llama.cpp is/will be installed
            auto_build: Automatically build llama.cpp if not found
            auto_start: Automatically start RPC servers if none running
            default_port: Default RPC server port
            default_host: Default RPC server host
        """
        self.llama_dir = Path(llama_dir or os.path.expanduser("~/llama.cpp"))
        self.auto_build = auto_build
        self.auto_start = auto_start
        self.default_port = default_port
        self.default_host = default_host
        self.rpc_processes = []

    def get_or_create_backends(
        self, num_backends: int = 1, discover_network: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get RPC backends, automatically setting them up if needed.

        Args:
            num_backends: Number of local backends to start if none found
            discover_network: Also discover backends on the network

        Returns:
            List of RPC backend configurations
        """
        # First, try to discover existing backends
        logger.info("🔍 Discovering RPC backends...")
        backends = discover_rpc_backends() if discover_network else []

        # Check localhost specifically
        if check_rpc_server(self.default_host, self.default_port):
            if not any(b["host"] == self.default_host for b in backends):
                backends.append({"host": self.default_host, "port": self.default_port})

        if backends:
            logger.info(f"✅ Found {len(backends)} existing RPC backends")
            return backends

        # No backends found - set them up automatically
        logger.info("❌ No RPC backends found")

        if not self.auto_start:
            logger.warning("Auto-start disabled. Please start RPC servers manually.")
            return []

        # Check if llama.cpp exists
        if not self._check_llama_cpp_exists():
            logger.info("📥 llama.cpp not found")

            if not self.auto_build:
                logger.warning("Auto-build disabled. Please build llama.cpp manually.")
                return []

            # Clone and build llama.cpp
            if not self._setup_llama_cpp():
                logger.error("❌ Failed to setup llama.cpp")
                return []

        # Start RPC servers
        logger.info(f"🚀 Starting {num_backends} RPC server(s)...")
        for i in range(num_backends):
            port = self.default_port + i
            if self._start_rpc_server(port=port):
                backends.append({"host": self.default_host, "port": port})

        if backends:
            logger.info(f"✅ Started {len(backends)} RPC backend(s)")

        return backends

    def _check_llama_cpp_exists(self) -> bool:
        """Check if llama.cpp is installed and built."""
        rpc_server = self.llama_dir / "build" / "bin" / "rpc-server"
        return rpc_server.exists()

    def _setup_llama_cpp(self) -> bool:
        """Clone and build llama.cpp with RPC support."""
        try:
            # Clone if needed
            if not self.llama_dir.exists():
                logger.info(f"📥 Cloning llama.cpp to {self.llama_dir}...")
                result = subprocess.run(
                    ["git", "clone", "https://github.com/ggerganov/llama.cpp", str(self.llama_dir)],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    logger.error(f"Failed to clone: {result.stderr}")
                    return False
                logger.info("✅ Cloned llama.cpp")

            # Build with RPC support
            logger.info("🔨 Building llama.cpp with RPC support...")

            # Configure
            result = subprocess.run(
                ["cmake", "-B", "build", "-DGGML_RPC=ON", "-DLLAMA_CURL=OFF"],
                cwd=str(self.llama_dir),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error(f"CMake configuration failed: {result.stderr}")
                return False

            # Build
            nproc = multiprocessing.cpu_count()
            result = subprocess.run(
                [
                    "cmake",
                    "--build",
                    "build",
                    "--config",
                    "Release",
                    "--target",
                    "rpc-server",
                    f"-j{nproc}",
                ],
                cwd=str(self.llama_dir),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error(f"Build failed: {result.stderr}")
                return False

            logger.info("✅ Built llama.cpp with RPC support")
            return True

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False

    def _start_rpc_server(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        mem_gb: Optional[int] = None,
        background: bool = True,
    ) -> bool:
        """
        Start an RPC server with automatic hybrid GPU+CPU parallelization.

        Detects local GPU and CPU resources, then starts rpc-server with:
        - Hybrid device config (cpu,cuda:0,...) if GPU(s) detected
        - Safe memory allocations (80% of available with 20% reserve)
        - Multiple parallel workers per physical node

        Args:
            host: Host to bind to (default: self.default_host)
            port: Port to bind to (default: self.default_port)
            mem_gb: Memory limit in GB (deprecated - auto-detected)
            background: Run in background

        Returns:
            True if started successfully
        """
        rpc_server = self.llama_dir / "build" / "bin" / "rpc-server"

        if not rpc_server.exists():
            logger.error(f"RPC server not found at {rpc_server}")
            return False

        host = host or self.default_host
        port = port or self.default_port

        # Check if already running
        if check_rpc_server(host, port):
            logger.info(f"RPC server already running on {host}:{port}")
            return True

        # Auto-detect hybrid GPU+CPU resources
        from sollol.rpc_discovery import detect_node_resources

        logger.info("🔍 Auto-detecting hybrid GPU+CPU resources for RPC server...")
        resources = detect_node_resources("localhost")

        # Log detected configuration
        if resources["has_gpu"]:
            logger.info(f"✅ Detected {resources['total_parallel_workers']} parallel workers:")
            logger.info(f"   • CPU device: {resources['cpu_ram_mb']} MB RAM")
            for device, vram in zip(resources["gpu_devices"], resources["gpu_vram_mb"]):
                logger.info(f"   • {device}: {vram} MB VRAM")
        else:
            logger.info(f"✅ Detected CPU-only configuration: {resources['cpu_ram_mb']} MB RAM")

        # Build command with hybrid device config
        cmd = [str(rpc_server), "--host", host, "--port", str(port)]
        cmd.extend(["--device", resources["device_config"]])
        cmd.extend(["--mem", resources["memory_config"]])

        try:
            if background:
                # Start in background
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    cwd=str(self.llama_dir),
                )
                self.rpc_processes.append(process)

                # Give it a moment to start
                import time

                time.sleep(1)

                # Verify it's running
                if check_rpc_server(host, port):
                    if resources["has_gpu"]:
                        logger.info(
                            f"✅ RPC server started on {host}:{port} (PID: {process.pid}) "
                            f"with {resources['total_parallel_workers']} parallel workers (HYBRID mode)"
                        )
                    else:
                        logger.info(
                            f"✅ RPC server started on {host}:{port} (PID: {process.pid}) (CPU-only mode)"
                        )
                    return True
                else:
                    logger.error(f"Failed to verify RPC server on {host}:{port}")
                    return False
            else:
                # Run in foreground (blocking)
                subprocess.run(cmd, cwd=str(self.llama_dir))
                return True

        except Exception as e:
            logger.error(f"Failed to start RPC server: {e}")
            return False

    def stop_all_servers(self):
        """Stop all RPC servers started by this instance."""
        logger.info(f"Stopping {len(self.rpc_processes)} RPC server(s)...")
        for process in self.rpc_processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception as e:
                logger.error(f"Failed to stop process {process.pid}: {e}")
        self.rpc_processes.clear()


def auto_setup_rpc_backends(
    num_backends: int = 1,
    llama_dir: Optional[str] = None,
    auto_build: bool = True,
    discover_network: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convenience function to auto-setup RPC backends.

    Args:
        num_backends: Number of local backends to start if none found
        llama_dir: Directory where llama.cpp is/will be installed
        auto_build: Automatically build llama.cpp if needed
        discover_network: Also discover backends on the network

    Returns:
        List of RPC backend configurations

    Example:
        >>> from sollol.rpc_auto_setup import auto_setup_rpc_backends
        >>> backends = auto_setup_rpc_backends(num_backends=2)
        >>> print(backends)
        [{'host': '127.0.0.1', 'port': 50052}, {'host': '127.0.0.1', 'port': 50053}]
    """
    setup = RPCAutoSetup(llama_dir=llama_dir, auto_build=auto_build, auto_start=True)
    return setup.get_or_create_backends(
        num_backends=num_backends, discover_network=discover_network
    )


if __name__ == "__main__":
    # Test auto-setup
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("Testing RPC auto-setup...")
    backends = auto_setup_rpc_backends(num_backends=2)

    if backends:
        print(f"\n✅ RPC backends ready:")
        for backend in backends:
            print(f"   → {backend['host']}:{backend['port']}")
    else:
        print("\n❌ Failed to setup RPC backends")
