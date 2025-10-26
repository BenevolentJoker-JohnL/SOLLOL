"""
SOLLOL CLI - Intelligent load balancer for Ollama clusters.
Runs on Ollama's port (11434) and routes requests to backend Ollama nodes.
"""

import logging
from typing import Optional

import typer

from .gateway import start_api

app = typer.Typer(
    name="sollol",
    help="SOLLOL - Intelligent load balancer for Ollama clusters. Runs on Ollama's port, routes to backend nodes.",
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@app.command()
def up(
    port: int = typer.Option(11434, help="Port for SOLLOL gateway (default: 11434, Ollama's port)"),
    ray_workers: int = typer.Option(4, help="Number of Ray actors for parallel execution"),
    dask_workers: int = typer.Option(2, help="Number of Dask workers for batch processing"),
    batch_processing: bool = typer.Option(
        True, "--batch-processing/--no-batch-processing", help="Enable Dask batch processing"
    ),
    autobatch_interval: int = typer.Option(60, help="Seconds between autobatch cycles"),
    rpc_backends: Optional[str] = typer.Option(
        None,
        help="Comma-separated RPC backends for model sharding (e.g., '192.168.1.10:50052,192.168.1.11:50052')",
    ),
    ollama_nodes: Optional[str] = typer.Option(
        None,
        help="Comma-separated Ollama nodes for task distribution (e.g., '192.168.1.20:11434,192.168.1.21:11434'). Auto-discovers if not set.",
    ),
    setup_gpu_monitoring: bool = typer.Option(
        True,
        "--setup-gpu-monitoring/--no-setup-gpu-monitoring",
        help="Auto-setup GPU monitoring if not running",
    ),
    redis_host: str = typer.Option("localhost", help="Redis host for GPU monitoring"),
    redis_port: int = typer.Option(6379, help="Redis port for GPU monitoring"),
):
    """
    Start SOLLOL gateway - Intelligent load balancer for Ollama clusters.

    SOLLOL runs on Ollama's port (11434) and routes requests to backend Ollama nodes.

    THREE DISTRIBUTION MODES:
    1. Intelligent Task Distribution - 7-factor routing + Ray parallel execution
    2. Batch Processing - Dask distributed batch operations (embeddings, bulk inference)
    3. Model Sharding - Distribute large models via llama.cpp RPC backends

    All modes work together for maximum performance!

    Features:
    - 7-factor intelligent routing engine
    - Ray actors for parallel request execution
    - Dask for distributed batch processing
    - Model sharding for 70B+ models via llama.cpp
    - Auto-discovers Ollama nodes and RPC backends
    - Automatic GGUF extraction from Ollama storage
    - Zero-config setup

    Examples:
        # Zero-config (auto-discovers everything):
        sollol up

        # Custom workers:
        sollol up --ray-workers 8 --dask-workers 4

        # With RPC backends for model sharding:
        sollol up --rpc-backends "192.168.1.10:50052,192.168.1.11:50052"

        # Disable batch processing:
        sollol up --no-batch-processing

        # Full configuration:
        sollol up --port 8000 --ray-workers 8 --rpc-backends "10.0.0.1:50052"
    """
    logger.info("=" * 70)
    logger.info("🚀 Starting SOLLOL Gateway")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Distribution Modes:")
    logger.info("  🎯 Intelligent Routing - 7-factor scoring engine")
    logger.info("  ⚡ Ray Parallel - Concurrent request execution")
    logger.info("  🔄 Dask Batch - Distributed bulk operations")
    logger.info("  🔗 Model Sharding - llama.cpp distributed inference")
    logger.info("")
    logger.info(f"Configuration:")
    logger.info(f"  Port: {port}")
    logger.info(f"  Ray workers: {ray_workers}")
    logger.info(f"  Dask workers: {dask_workers}")
    logger.info(f"  Batch processing: {'enabled' if batch_processing else 'disabled'}")

    # Parse RPC backends
    parsed_rpc_backends = None
    if rpc_backends:
        parsed_rpc_backends = []
        for backend_str in rpc_backends.split(","):
            backend_str = backend_str.strip()
            if ":" in backend_str:
                host, port_str = backend_str.rsplit(":", 1)
                parsed_rpc_backends.append({"host": host, "port": int(port_str)})
            else:
                parsed_rpc_backends.append({"host": backend_str, "port": 50052})
        logger.info(f"  RPC Backends: {len(parsed_rpc_backends)} configured")
        logger.info("  → Model Sharding ENABLED")
    else:
        logger.info("  RPC Backends: Auto-discovery mode")

    # Parse Ollama nodes
    parsed_ollama_nodes = None
    if ollama_nodes:
        parsed_ollama_nodes = []
        for node_str in ollama_nodes.split(","):
            node_str = node_str.strip()
            if ":" in node_str:
                host, node_port = node_str.rsplit(":", 1)
                parsed_ollama_nodes.append({"host": host, "port": int(node_port)})
            else:
                parsed_ollama_nodes.append({"host": node_str, "port": 11434})
        logger.info(f"  Ollama Nodes: {len(parsed_ollama_nodes)} configured")
        logger.info("  → Task Distribution ENABLED")
    else:
        logger.info("  Ollama Nodes: Auto-discovery mode")

    logger.info("")
    logger.info("=" * 70)
    logger.info("")

    # Auto-setup GPU monitoring if enabled
    if setup_gpu_monitoring:
        logger.info("🔧 Setting up GPU monitoring...")
        try:
            from .gpu_auto_setup import auto_setup_gpu_monitoring

            gpu_setup_success = auto_setup_gpu_monitoring(
                redis_host=redis_host,
                redis_port=redis_port,
                auto_install=True,
                auto_start=True,
            )

            if gpu_setup_success:
                logger.info("✅ GPU monitoring ready")
            else:
                logger.warning("⚠️  GPU monitoring setup failed (non-critical)")
                logger.warning("   You can set it up manually: sollol install-gpu-reporter")
        except Exception as e:
            logger.warning(f"⚠️  GPU monitoring setup failed: {e}")
            logger.warning("   You can set it up manually: sollol install-gpu-reporter")

        logger.info("")

    # Start gateway (blocking call)
    start_api(
        port=port,
        rpc_backends=parsed_rpc_backends,
        ollama_nodes=parsed_ollama_nodes,
        ray_workers=ray_workers,
        dask_workers=dask_workers,
        enable_batch_processing=batch_processing,
        autobatch_interval=autobatch_interval,
    )


@app.command()
def down():
    """
    Stop SOLLOL service.

    Note: For MVP, manually kill Ray/Dask processes:
        pkill -f "ray::"
        pkill -f "dask"
    """
    logger.info("🛑 SOLLOL shutdown")
    logger.info("   To stop Ray: pkill -f 'ray::'")
    logger.info("   To stop Dask: pkill -f 'dask'")


@app.command()
def status():
    """
    Check SOLLOL service status.
    """
    logger.info("📊 SOLLOL Status")
    logger.info("   Gateway: http://localhost:8000/api/health")
    logger.info("   Metrics: http://localhost:9090/metrics")
    logger.info("   Stats: http://localhost:8000/api/stats")


@app.command()
def install_gpu_reporter(
    redis_host: str = typer.Option("localhost", help="Redis server hostname"),
    redis_port: int = typer.Option(6379, help="Redis server port"),
    node_id: Optional[str] = typer.Option(
        None, help="Node ID (e.g., 10.9.66.90:11434). Auto-detected if not specified."
    ),
    interval: int = typer.Option(5, help="Reporting interval in seconds"),
):
    """
    Install GPU reporter service for real-time VRAM monitoring.

    This installs a systemd user service that publishes GPU stats to Redis
    using gpustat (vendor-agnostic: NVIDIA, AMD, Intel).

    Examples:
        # Auto-detect node ID, use localhost Redis:
        sollol install-gpu-reporter

        # Specify Redis host:
        sollol install-gpu-reporter --redis-host 10.9.66.154

        # Full configuration:
        sollol install-gpu-reporter --redis-host 10.9.66.154 --node-id 10.9.66.90:11434 --interval 5
    """
    import socket
    import subprocess
    from pathlib import Path

    # Auto-detect node ID if not specified
    if not node_id:
        try:
            # Get primary network interface IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            node_id = f"{local_ip}:11434"
        except Exception:
            node_id = "127.0.0.1:11434"
        logger.info(f"Auto-detected Node ID: {node_id}")

    logger.info("=" * 70)
    logger.info("🔧 Installing SOLLOL GPU Reporter Service")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"Configuration:")
    logger.info(f"  Redis Host: {redis_host}")
    logger.info(f"  Redis Port: {redis_port}")
    logger.info(f"  Node ID: {node_id}")
    logger.info(f"  Interval: {interval}s")
    logger.info("")

    # Find the installer script
    sollol_path = Path(__file__).parent.parent.parent
    installer_script = sollol_path / "scripts" / "install-gpu-reporter-service.sh"

    if not installer_script.exists():
        logger.error(f"❌ Installer script not found at: {installer_script}")
        logger.error("   Run from SOLLOL repository directory")
        raise typer.Exit(1)

    # Run installer with auto-configuration (non-interactive)
    env = {
        "REDIS_HOST": redis_host,
        "REDIS_PORT": str(redis_port),
        "NODE_ID": node_id,
        "REPORT_INTERVAL": str(interval),
    }

    try:
        # Make script executable
        installer_script.chmod(0o755)

        # Run with environment variables set (for non-interactive mode)
        result = subprocess.run(
            [str(installer_script)],
            env={**subprocess.os.environ, **env},
            check=True,
            capture_output=False,
        )

        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ GPU Reporter service installed!")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Useful commands:")
        logger.info("  systemctl --user status sollol-gpu-reporter")
        logger.info("  systemctl --user restart sollol-gpu-reporter")
        logger.info("  journalctl --user -u sollol-gpu-reporter -f")
        logger.info("")

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Installation failed: {e}")
        raise typer.Exit(1)


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
