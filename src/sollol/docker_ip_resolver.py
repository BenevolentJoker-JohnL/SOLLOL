"""
Docker IP Resolution for SOLLOL

Detects Docker container IPs and resolves them to accessible host IPs.

Problem:
    Ollama/RPC servers running in Docker containers report internal IPs
    (e.g., 172.17.0.5) that are inaccessible from the host network.

Solution:
    Automatically detect deployment mode (bare metal vs Docker) and use
    appropriate resolution strategy:

    Bare Metal → Docker:
        1. localhost (if on same machine)
        2. Host's actual IP
        3. Docker host gateway

    Docker → Docker:
        1. Try Docker IP directly (same network)
        2. Try service name resolution
        3. Fall back to host gateway
"""

import ipaddress
import logging
import os
import socket
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Common Docker network ranges
DOCKER_IP_RANGES = [
    "172.17.0.0/16",  # Default bridge network
    "172.18.0.0/16",  # User-defined bridges
    "172.19.0.0/16",
    "172.20.0.0/16",
    "172.21.0.0/16",
    "172.22.0.0/16",
    "172.23.0.0/16",
    "172.24.0.0/16",
    "172.25.0.0/16",
    "172.26.0.0/16",
    "172.27.0.0/16",
    "172.28.0.0/16",
    "172.29.0.0/16",
    "172.30.0.0/16",
    "172.31.0.0/16",
    "10.0.0.0/8",  # Docker swarm overlay networks
]


# Cache for deployment detection (avoid repeated filesystem checks)
_deployment_mode_cache = None


def is_running_in_docker() -> bool:
    """
    Detect if SOLLOL is running inside a Docker container.

    Multiple detection methods:
    1. Check /.dockerenv file (most reliable)
    2. Check /proc/1/cgroup for docker/containerd
    3. Check if hostname is container ID format

    Returns:
        True if running inside Docker container

    Examples:
        >>> is_running_in_docker()
        False  # On bare metal
        True   # Inside Docker container
    """
    global _deployment_mode_cache

    # Use cache if available
    if _deployment_mode_cache is not None:
        return _deployment_mode_cache

    # Method 1: Check for /.dockerenv file
    if Path("/.dockerenv").exists():
        logger.debug("Detected Docker deployment: /.dockerenv exists")
        _deployment_mode_cache = True
        return True

    # Method 2: Check /proc/1/cgroup for docker/containerd
    try:
        with open("/proc/1/cgroup", "r") as f:
            content = f.read()
            if "docker" in content or "containerd" in content or "kubepods" in content:
                logger.debug(
                    "Detected Docker deployment: /proc/1/cgroup contains docker/containerd"
                )
                _deployment_mode_cache = True
                return True
    except (FileNotFoundError, PermissionError):
        pass

    # Method 3: Check environment variable (set by Docker)
    if os.getenv("DOCKER_CONTAINER") == "true":
        logger.debug("Detected Docker deployment: DOCKER_CONTAINER env var")
        _deployment_mode_cache = True
        return True

    # Not in Docker
    logger.debug("Detected bare metal deployment")
    _deployment_mode_cache = False
    return False


def get_docker_network_mode() -> str:
    """
    Detect Docker network mode if running in Docker.

    Returns:
        "host" - Container uses host network
        "bridge" - Container uses bridge network (default)
        "overlay" - Container uses overlay network (swarm)
        "none" - No networking
        "unknown" - Can't determine or not in Docker

    Examples:
        >>> get_docker_network_mode()
        "bridge"  # Most common
    """
    if not is_running_in_docker():
        return "unknown"

    # Method 1: Check if we're using host network
    # In host mode, container sees host's network interfaces
    try:
        hostname = socket.gethostname()
        host_ip = socket.gethostbyname(hostname)

        # If we can resolve host's hostname to non-Docker IP, likely host mode
        if not is_docker_ip(host_ip):
            return "host"
    except:
        pass

    # Method 2: Check network interfaces
    try:
        import netifaces

        interfaces = netifaces.interfaces()

        # Host mode usually has eth0, wlan0, etc.
        if "docker0" in interfaces:
            return "bridge"

        # Bridge mode usually has eth0 with Docker IP
        if "eth0" in interfaces:
            addrs = netifaces.ifaddresses("eth0")
            if netifaces.AF_INET in addrs:
                ip = addrs[netifaces.AF_INET][0]["addr"]
                if is_docker_ip(ip):
                    return "bridge"
    except ImportError:
        # netifaces not available - skip this check
        pass
    except:
        pass

    # Method 3: Check /proc/net/route for default gateway
    try:
        with open("/proc/net/route", "r") as f:
            for line in f:
                fields = line.strip().split()
                if fields[0] == "eth0" and fields[1] == "00000000":  # Default route
                    # Has default route on eth0 - likely bridge mode
                    return "bridge"
    except:
        pass

    # Default assumption for containers
    return "bridge"


def get_deployment_context() -> Dict[str, any]:
    """
    Get comprehensive deployment context.

    Returns:
        Dict with deployment information:
        {
            "mode": "docker" | "bare_metal",
            "network_mode": "host" | "bridge" | "overlay" | "unknown",
            "is_docker": bool,
            "container_id": str | None,
        }

    Examples:
        >>> get_deployment_context()
        {"mode": "bare_metal", "network_mode": "unknown", "is_docker": False, "container_id": None}
    """
    is_docker = is_running_in_docker()

    context = {
        "mode": "docker" if is_docker else "bare_metal",
        "is_docker": is_docker,
        "network_mode": get_docker_network_mode() if is_docker else "unknown",
        "container_id": None,
    }

    # Try to get container ID
    if is_docker:
        try:
            # Read from /proc/self/cgroup
            with open("/proc/self/cgroup", "r") as f:
                for line in f:
                    if "docker" in line:
                        # Extract container ID from path
                        parts = line.strip().split("/")
                        if len(parts) > 2:
                            container_id = parts[-1]
                            if len(container_id) == 64:  # Full container ID
                                context["container_id"] = container_id[:12]  # Short form
                            break
        except:
            pass

    return context


def is_docker_ip(ip: str) -> bool:
    """
    Check if IP address is in Docker's internal IP ranges.

    Args:
        ip: IP address to check

    Returns:
        True if IP is likely a Docker internal IP

    Examples:
        >>> is_docker_ip("172.17.0.5")
        True
        >>> is_docker_ip("192.168.1.100")
        False
    """
    try:
        ip_obj = ipaddress.ip_address(ip)

        for cidr in DOCKER_IP_RANGES:
            network = ipaddress.ip_network(cidr)
            if ip_obj in network:
                return True

        return False
    except ValueError:
        return False


def resolve_docker_ip(
    docker_ip: str,
    port: int,
    timeout: float = 1.0,
    verify_func=None,
    deployment_context: Optional[Dict] = None,
) -> Optional[str]:
    """
    Resolve Docker internal IP to accessible host IP.

    Strategy depends on deployment mode:

    Bare Metal → Docker:
        1. Try localhost (container published ports)
        2. Try host's actual IP (if different from localhost)
        3. Try Docker host gateway (host.docker.internal)
        4. Try subnet gateway (usually x.x.x.1)

    Docker → Docker:
        1. Try Docker IP directly (same network)
        2. Try Docker host gateway
        3. Try localhost (for host network mode)

    Args:
        docker_ip: Docker internal IP (e.g., "172.17.0.5")
        port: Port to check
        timeout: Connection timeout per attempt
        verify_func: Optional function to verify service (e.g., _is_ollama_running)
        deployment_context: Optional deployment context (auto-detected if None)

    Returns:
        Accessible IP address, or None if resolution failed

    Examples:
        >>> resolve_docker_ip("172.17.0.5", 11434)
        "127.0.0.1"  # or host IP if accessible
    """
    # Get deployment context
    if deployment_context is None:
        deployment_context = get_deployment_context()

    is_docker = deployment_context["is_docker"]
    network_mode = deployment_context.get("network_mode", "unknown")

    logger.info(
        f"🐳 Detected Docker IP: {docker_ip}:{port} "
        f"(Running in: {deployment_context['mode']}, Network: {network_mode})"
    )

    # Candidates to try (in order of likelihood)
    candidates = []

    # STRATEGY: Docker → Docker (same network, might be directly accessible)
    if is_docker:
        logger.debug("   Using Docker→Docker resolution strategy")

        # 1. Try Docker IP directly (same Docker network)
        candidates.append((docker_ip, "direct Docker IP (same network)"))

        # 2. Try Docker host gateway
        try:
            host_docker_internal = socket.gethostbyname("host.docker.internal")
            candidates.append((host_docker_internal, "host.docker.internal"))
        except socket.gaierror:
            pass

        # 3. If host network mode, try localhost
        if network_mode == "host":
            candidates.append(("127.0.0.1", "localhost (host network mode)"))

        # 4. Try subnet gateway
        try:
            parts = docker_ip.split(".")
            gateway = f"{parts[0]}.{parts[1]}.{parts[2]}.1"
            candidates.append((gateway, "Docker network gateway"))
        except:
            pass

    # STRATEGY: Bare Metal → Docker
    else:
        logger.debug("   Using Bare Metal→Docker resolution strategy")

        # 1. localhost (most common - published ports)
        candidates.append(("127.0.0.1", "published port mapping"))
        candidates.append(("localhost", "published port mapping (hostname)"))

        # 2. Host's actual IP
        host_ip = _get_host_ip()
        if host_ip and host_ip not in ("127.0.0.1", "localhost"):
            candidates.append((host_ip, "host network IP"))

        # 3. Docker host gateway (newer Docker versions)
        try:
            host_docker_internal = socket.gethostbyname("host.docker.internal")
            candidates.append((host_docker_internal, "host.docker.internal"))
        except socket.gaierror:
            pass

        # 4. Gateway IP (usually x.x.x.1)
        try:
            parts = docker_ip.split(".")
            gateway = f"{parts[0]}.{parts[1]}.{parts[2]}.1"
            candidates.append((gateway, "Docker network gateway"))
        except:
            pass

    # Try each candidate
    for candidate_ip, source in candidates:
        logger.debug(f"   Trying {candidate_ip}:{port} ({source})...")

        # First quick port check
        if _is_port_open(candidate_ip, port, timeout):
            # If verify function provided, use it
            if verify_func:
                if verify_func(candidate_ip, port, timeout):
                    logger.info(
                        f"   ✅ Resolved {docker_ip}:{port} → {candidate_ip}:{port} ({source})"
                    )
                    return candidate_ip
            else:
                # No verify function - assume port open = success
                logger.info(f"   ✅ Resolved {docker_ip}:{port} → {candidate_ip}:{port} ({source})")
                return candidate_ip

    logger.warning(f"   ❌ Failed to resolve Docker IP {docker_ip}:{port} to accessible address")
    return None


def resolve_docker_ip_with_alternatives(
    docker_ip: str,
    port: int,
    timeout: float = 1.0,
    verify_func=None,
    deployment_context: Optional[Dict] = None,
) -> List[Tuple[str, int]]:
    """
    Resolve Docker IP to all accessible alternatives.

    Unlike resolve_docker_ip which returns first match, this returns
    all working alternatives for redundancy.

    Args:
        docker_ip: Docker internal IP
        port: Port to check
        timeout: Connection timeout per attempt
        verify_func: Optional verification function
        deployment_context: Optional deployment context (auto-detected if None)

    Returns:
        List of (ip, port) tuples that are accessible

    Examples:
        >>> resolve_docker_ip_with_alternatives("172.17.0.5", 11434)
        [("127.0.0.1", 11434), ("192.168.1.50", 11434)]
    """
    # Get deployment context
    if deployment_context is None:
        deployment_context = get_deployment_context()

    is_docker = deployment_context["is_docker"]
    network_mode = deployment_context.get("network_mode", "unknown")

    logger.debug(
        f"Finding all accessible alternatives for {docker_ip}:{port} "
        f"(mode: {deployment_context['mode']}, network: {network_mode})"
    )

    alternatives = []
    candidates = []

    # Different candidates based on deployment mode
    if is_docker:
        # Docker → Docker: Try direct IP first
        candidates.append((docker_ip, "direct Docker IP"))

        # Try host gateway
        try:
            host_docker_internal = socket.gethostbyname("host.docker.internal")
            candidates.append((host_docker_internal, "host.docker.internal"))
        except:
            pass

        # If host mode, try localhost
        if network_mode == "host":
            candidates.append(("127.0.0.1", "localhost (host mode)"))
    else:
        # Bare Metal → Docker
        candidates.append(("127.0.0.1", "localhost"))
        candidates.append(("localhost", "localhost hostname"))

        host_ip = _get_host_ip()
        if host_ip and host_ip not in ("127.0.0.1", "localhost"):
            candidates.append((host_ip, "host IP"))

        try:
            host_docker_internal = socket.gethostbyname("host.docker.internal")
            candidates.append((host_docker_internal, "host.docker.internal"))
        except:
            pass

    # Try all candidates
    for candidate_ip, source in candidates:
        if _is_port_open(candidate_ip, port, timeout):
            if verify_func:
                if verify_func(candidate_ip, port, timeout):
                    alternatives.append((candidate_ip, port))
                    logger.debug(f"   ✅ Alternative found: {candidate_ip}:{port} ({source})")
            else:
                alternatives.append((candidate_ip, port))
                logger.debug(f"   ✅ Alternative found: {candidate_ip}:{port} ({source})")

    return alternatives


def auto_resolve_ips(
    nodes: List[Dict[str, str]],
    timeout: float = 1.0,
    verify_func=None,
    deployment_context: Optional[Dict] = None,
) -> List[Dict[str, str]]:
    """
    Auto-resolve Docker IPs in a list of nodes.

    If a node has a Docker IP that's unresponsive, try to resolve it
    to an accessible IP. Resolution strategy depends on whether we're
    running bare metal or in Docker.

    Args:
        nodes: List of node dicts with "host" and "port" keys
        timeout: Connection timeout per check
        verify_func: Optional verification function
        deployment_context: Optional deployment context (auto-detected if None)

    Returns:
        Updated list with Docker IPs resolved

    Examples:
        >>> nodes = [{"host": "172.17.0.5", "port": "11434"}]
        >>> auto_resolve_ips(nodes)
        [{"host": "127.0.0.1", "port": "11434"}]  # Bare metal
        [{"host": "172.17.0.5", "port": "11434"}]  # Docker (direct access)
    """
    # Get deployment context once for all nodes
    if deployment_context is None:
        deployment_context = get_deployment_context()

    logger.debug(
        f"Auto-resolving Docker IPs (mode: {deployment_context['mode']}, "
        f"network: {deployment_context.get('network_mode', 'unknown')})"
    )

    resolved_nodes = []

    for node in nodes:
        host = node["host"]
        port = int(node.get("port", 11434))

        # Check if it's a Docker IP
        if is_docker_ip(host):
            # First check if Docker IP is directly accessible
            # (common in Docker→Docker same network)
            is_accessible = False
            if _is_port_open(host, port, timeout):
                if verify_func:
                    is_accessible = verify_func(host, port, timeout)
                else:
                    is_accessible = True

            if is_accessible:
                # Docker IP is directly accessible - keep it
                logger.debug(f"   ✅ Docker IP {host}:{port} is directly accessible")
                resolved_nodes.append(node)
            else:
                # Docker IP not accessible - try to resolve
                resolved_ip = resolve_docker_ip(
                    host, port, timeout, verify_func, deployment_context
                )

                if resolved_ip:
                    # Successfully resolved
                    resolved_nodes.append({"host": resolved_ip, "port": str(port)})
                else:
                    # Failed to resolve - skip this node
                    logger.warning(f"⚠️  Skipping unresolvable Docker IP: {host}:{port}")
        else:
            # Not a Docker IP - keep as is
            resolved_nodes.append(node)

    return resolved_nodes


def _is_port_open(host: str, port: int, timeout: float) -> bool:
    """Quick TCP port check."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False


def _get_host_ip() -> Optional[str]:
    """
    Get the host's actual IP address (not localhost).

    Returns:
        Host IP or None if detection fails
    """
    try:
        # Connect to external IP to find our local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return None


# Convenience function for backward compatibility
def resolve_docker_nodes(
    nodes: List[Dict[str, str]], timeout: float = 1.0, verify_func=None
) -> List[Dict[str, str]]:
    """Alias for auto_resolve_ips (backward compatibility)."""
    return auto_resolve_ips(nodes, timeout, verify_func)


if __name__ == "__main__":
    # Test Docker IP detection
    logging.basicConfig(level=logging.INFO)

    print("Testing Docker IP detection:")
    print(f"  172.17.0.5 is Docker IP: {is_docker_ip('172.17.0.5')}")
    print(f"  192.168.1.100 is Docker IP: {is_docker_ip('192.168.1.100')}")
    print(f"  10.0.0.5 is Docker IP: {is_docker_ip('10.0.0.5')}")

    print("\nTesting Docker IP resolution:")
    # Example: resolve a Docker IP
    test_nodes = [
        {"host": "172.17.0.5", "port": "11434"},
        {"host": "192.168.1.100", "port": "11434"},
    ]

    resolved = auto_resolve_ips(test_nodes)
    print(f"\nOriginal nodes: {test_nodes}")
    print(f"Resolved nodes: {resolved}")
