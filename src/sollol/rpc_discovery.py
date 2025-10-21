"""
RPC Backend Discovery - Auto-detect llama.cpp RPC servers on the network

Similar to Ollama discovery, this module scans the network for running
RPC servers (default port: 50052).

Features:
- Automatic Docker IP resolution (172.17.x.x → localhost)
- Multi-threaded network scanning
- Health checking
- GPU detection and VRAM/RAM calculation
- Hybrid device configuration (CPU + GPU per node)
"""

import asyncio
import logging
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import httpx

from sollol.docker_ip_resolver import auto_resolve_ips, is_docker_ip
from sollol.vram_monitor import VRAMMonitor

logger = logging.getLogger(__name__)


def detect_node_resources(host: str) -> Dict[str, Any]:
    """
    Detect GPU and RAM resources for a node via SSH or local detection.

    Returns dict with:
        - has_gpu: bool
        - gpu_devices: List[str] (e.g., ["cuda:0", "cuda:1"])
        - gpu_vram_mb: List[int] (VRAM per GPU)
        - cpu_ram_mb: int (available system RAM)
        - device_config: str (for rpc-server --device flag)
        - memory_config: str (for rpc-server --mem flag)
    """
    monitor = VRAMMonitor()

    # For localhost, detect directly
    if host in ["127.0.0.1", "localhost"]:
        vram_info = monitor.get_local_vram_info()

        # Get system RAM
        try:
            with open("/proc/meminfo", "r") as f:
                mem_total_kb = int(f.readline().split()[1])
                cpu_ram_mb = mem_total_kb // 1024
                # Reserve 20% for OS
                available_ram_mb = int(cpu_ram_mb * 0.8)
        except:
            available_ram_mb = 8000  # Conservative default

        if vram_info and vram_info.get("gpus"):
            # Has GPU(s)
            gpus = vram_info["gpus"]
            gpu_devices = []
            gpu_vram_mb = []

            for gpu in gpus:
                idx = gpu["index"]
                vendor = gpu["vendor"].lower()

                if vendor == "nvidia":
                    gpu_devices.append(f"cuda:{idx}")
                elif vendor == "amd":
                    gpu_devices.append(f"rocm:{idx}")
                else:
                    continue

                # Reserve 20% VRAM for safety
                total_vram = gpu.get("total_mb", 0)
                if total_vram > 0:
                    safe_vram = int(total_vram * 0.8)
                    gpu_vram_mb.append(safe_vram)
                else:
                    gpu_vram_mb.append(8000)  # Conservative default

            # Build hybrid device config: cpu + gpu(s)
            devices = ["cpu"] + gpu_devices
            memory = [available_ram_mb] + gpu_vram_mb

            return {
                "has_gpu": True,
                "gpu_devices": gpu_devices,
                "gpu_vram_mb": gpu_vram_mb,
                "cpu_ram_mb": available_ram_mb,
                "device_config": ",".join(devices),
                "memory_config": ",".join(str(m) for m in memory),
                "total_parallel_workers": len(devices),  # CPU + each GPU = parallel workers!
            }
        else:
            # CPU only
            return {
                "has_gpu": False,
                "gpu_devices": [],
                "gpu_vram_mb": [],
                "cpu_ram_mb": available_ram_mb,
                "device_config": "cpu",
                "memory_config": str(available_ram_mb),
                "total_parallel_workers": 1,
            }

    # For remote nodes, try Redis first
    try:
        import redis
        import json as json_module
        import os

        redis_url = os.getenv("SOLLOL_REDIS_URL", "redis://localhost:6379")
        r = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=1)

        # Try to get node config from Redis
        key = f"sollol:rpc:node:{host}:50052"
        config_json = r.get(key)

        if config_json:
            logger.info(f"✅ Found GPU config for {host} in Redis")
            return json_module.loads(config_json)
    except Exception as e:
        logger.debug(f"Could not fetch config from Redis for {host}: {e}")

    # Fallback: conservative CPU-only config
    logger.debug(f"Using CPU-only fallback config for {host}")
    return {
        "has_gpu": False,
        "gpu_devices": [],
        "gpu_vram_mb": [],
        "cpu_ram_mb": 8000,
        "device_config": "cpu",
        "memory_config": "8000",
        "total_parallel_workers": 1,
    }


def check_rpc_server(host: str, port: int = 50052, timeout: float = 1.0) -> bool:
    """
    Check if an RPC server is running at host:port.

    Args:
        host: IP address to check
        port: RPC port (default: 50052)
        timeout: Connection timeout in seconds

    Returns:
        True if RPC server is reachable
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def discover_rpc_backends(
    cidr: str = None, port: int = 50052, timeout: float = 1.0, auto_resolve_docker: bool = True
) -> List[Dict[str, Any]]:
    """
    Discover RPC backends on the network.

    Args:
        cidr: Network CIDR (e.g., "192.168.1.0/24"). Auto-detects if None.
        port: RPC port to scan (default: 50052)
        timeout: Connection timeout per host
        auto_resolve_docker: If True, automatically resolve Docker IPs to accessible IPs

    Returns:
        List of discovered backends: [{"host": "ip", "port": 50052}, ...]

    Features:
        - Parallel network scanning
        - Automatic Docker IP resolution (172.17.x.x → localhost)
        - CIDR auto-detection
    """
    backends = []

    # Skip localhost for distributed RPC (coordinator runs on same machine)
    # localhost RPC backend adds no distribution benefit, only overhead

    # Check network
    if cidr is None:
        # Auto-detect local network
        cidr = _detect_local_network()
        if not cidr:
            logger.warning("Could not auto-detect network. Skipping network scan.")
            if backends:
                logger.info(f"✅ Discovered {len(backends)} RPC backends")
            return backends

    logger.info(f"🔍 Scanning {cidr} for RPC servers on port {port}...")

    # Parse CIDR to get IP range
    ips = _cidr_to_ips(cidr)

    # Scan in parallel
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(check_rpc_server, ip, port, timeout): ip for ip in ips}

        for future in futures:
            ip = futures[future]
            try:
                if future.result():
                    # Skip only localhost/127.0.0.1 (network IPs are valid for distribution)
                    if ip in ["127.0.0.1", "localhost"]:
                        logger.debug(f"   ⏭️  Skipping localhost: {ip}:{port}")
                        continue
                    logger.info(f"   ✅ Found RPC server: {ip}:{port}")
                    backends.append({"host": ip, "port": port})
            except Exception as e:
                logger.debug(f"Error checking {ip}: {e}")

    # Auto-resolve Docker IPs if enabled
    if auto_resolve_docker and backends:
        logger.debug("Checking for Docker IPs...")
        backends = auto_resolve_ips(backends, timeout, verify_func=check_rpc_server)

    logger.info(f"✅ Discovered {len(backends)} RPC backends")
    return backends


def _detect_local_network() -> str:
    """
    Auto-detect local network CIDR.

    Returns:
        CIDR string (e.g., "192.168.1.0/24") or None
    """
    try:
        # Get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()

        # Assume /24 network
        network = ".".join(local_ip.split(".")[:-1]) + ".0/24"
        return network
    except Exception as e:
        logger.debug(f"Could not auto-detect network: {e}")
        return None


def _cidr_to_ips(cidr: str) -> List[str]:
    """
    Convert CIDR notation to list of IPs.

    Args:
        cidr: CIDR notation (e.g., "192.168.1.0/24")

    Returns:
        List of IP addresses in the range
    """
    import ipaddress

    return [str(ip) for ip in ipaddress.IPv4Network(cidr, strict=False)]


# Convenience function
def auto_discover_rpc_backends(port: int = 50052, auto_resolve_docker: bool = True) -> List[Dict[str, Any]]:
    """
    Auto-discover RPC backends on the local network with GPU metadata.

    Args:
        port: RPC port to scan (default: 50052)
        auto_resolve_docker: If True, automatically resolve Docker IPs

    Returns:
        List of discovered backends with GPU metadata enrichment

    Features:
        - PRIORITY 1: Check Redis registry first (sollol:router:metadata)
        - PRIORITY 2: Read from rpc_backends.conf file
        - FALLBACK: Network scanning
        - GPU metadata enrichment from Redis
    """
    backends = []

    # PRIORITY 1: Try Redis registry first
    try:
        import redis as redis_client
        import json
        r = redis_client.Redis(host='localhost', port=6379, decode_responses=True)
        metadata_json = r.get("sollol:router:metadata")
        if metadata_json:
            metadata = json.loads(metadata_json)
            rpc_backends_from_redis = metadata.get("rpc_backends", [])
            if rpc_backends_from_redis:
                logger.info(f"✅ Found {len(rpc_backends_from_redis)} RPC backend(s) in Redis registry")
                backends = rpc_backends_from_redis
    except Exception as e:
        logger.debug(f"Could not read RPC backends from Redis: {e}")

    # PRIORITY 2: Try config file if Redis didn't work
    if not backends:
        try:
            from pathlib import Path
            import os

            # Check multiple locations for rpc_backends.conf
            possible_paths = [
                Path("rpc_backends.conf"),  # Current directory
                Path("/home/joker/SOLLOL/rpc_backends.conf"),  # SOLLOL directory
                Path(os.path.expanduser("~/SOLLOL/rpc_backends.conf")),  # Home/SOLLOL
                Path("/home/joker/SynapticLlamas/rpc_backends.conf"),  # SynapticLlamas directory
            ]

            config_file = None
            for path in possible_paths:
                if path.exists():
                    config_file = path
                    break

            if config_file:
                with open(config_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            host, port_str = line.split(":")
                            backends.append({"host": host, "port": int(port_str)})
                if backends:
                    logger.info(f"✅ Found {len(backends)} RPC backend(s) in config file: {config_file}")
        except Exception as e:
            logger.debug(f"Could not read RPC config file: {e}")

    # FALLBACK: Network scanning if nothing found
    if not backends:
        logger.info("🔍 No RPC backends in Redis or config - falling back to network scan")
        backends = discover_rpc_backends(port=port, auto_resolve_docker=auto_resolve_docker)

    # Enrich each backend with GPU metadata from Redis
    enriched_backends = []
    for backend in backends:
        host = backend["host"]
        port_num = backend["port"]

        # Get GPU metadata for this node
        resources = detect_node_resources(host)

        # Merge with backend info
        enriched_backend = {
            "host": host,
            "port": port_num,
            "has_gpu": resources.get("has_gpu", False),
            "gpu_devices": resources.get("gpu_devices", []),
            "gpu_vram_mb": resources.get("gpu_vram_mb", []),
            "gpu_names": resources.get("gpu_names", []),
            "cpu_ram_mb": resources.get("cpu_ram_mb", 0),
            "device_config": resources.get("device_config", "cpu"),
            "memory_config": resources.get("memory_config", "8000"),
            "num_workers": resources.get("total_parallel_workers", 1),
        }
        enriched_backends.append(enriched_backend)

    return enriched_backends


if __name__ == "__main__":
    # Test discovery
    logging.basicConfig(level=logging.INFO)

    print("Testing RPC backend discovery...")
    backends = auto_discover_rpc_backends()

    if backends:
        print(f"\n✅ Found {len(backends)} RPC backends:")
        for backend in backends:
            print(f"   → {backend['host']}:{backend['port']}")
    else:
        print("\n❌ No RPC backends found")
        print("   Make sure RPC servers are running:")
        print("   rpc-server --host 0.0.0.0 --port 50052 --mem 2048")
