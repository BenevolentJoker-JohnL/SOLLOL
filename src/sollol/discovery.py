"""
Fast Ollama node discovery - returns in <1 second.

Tries multiple strategies in order:
1. Environment variable (instant)
2. Known locations (instant)
3. Network scan (parallel, ~500ms)

Features:
- Automatic Docker IP resolution (172.17.x.x → localhost)
- Multi-strategy fallback
- Fast parallel scanning
"""

import logging
import os
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from sollol.docker_ip_resolver import auto_resolve_ips, is_docker_ip

logger = logging.getLogger(__name__)


def discover_ollama_nodes(
    timeout: float = 0.5,
    exclude_localhost: bool = False,
    auto_resolve_docker: bool = True,
    discover_all_nodes: bool = False,
) -> List[Dict[str, str]]:
    """
    Discover Ollama nodes using multiple strategies.

    Args:
        timeout: Connection timeout per node
        exclude_localhost: If True, skip localhost (useful when SOLLOL runs on 11434)
        auto_resolve_docker: If True, automatically resolve Docker IPs to accessible IPs
        discover_all_nodes: If True, scan full network and return ALL nodes (slower but comprehensive)

    Returns:
        List of node dicts: [{"host": "192.168.1.10", "port": "11434"}, ...]

    Features:
        - Multi-strategy discovery (env → known → network scan)
        - Automatic Docker IP resolution (172.17.x.x → localhost)
        - Fast parallel network scanning
        - Full network scan mode for comprehensive discovery
    """
    # If full network scan requested, skip fast strategies and scan entire subnet
    if discover_all_nodes:
        logger.info("Full network discovery mode - scanning entire subnet for all Ollama nodes")
        nodes = _from_network_scan(timeout, exclude_localhost)

        if auto_resolve_docker:
            nodes = auto_resolve_ips(nodes, timeout, _is_ollama_running)

        # Deduplicate localhost vs real IP
        nodes = _deduplicate_nodes(nodes)

        if nodes:
            return nodes
        elif not exclude_localhost:
            logger.debug("No nodes discovered on network, falling back to localhost")
            return [{"host": "localhost", "port": "11434"}]
        else:
            return []

    # Fast discovery mode (original behavior)
    strategies = [
        lambda t: _from_environment(t, exclude_localhost),
        lambda t: _from_known_locations(t, exclude_localhost),
        lambda t: _from_network_scan(t, exclude_localhost),
    ]

    for strategy in strategies:
        nodes = strategy(timeout)
        if nodes:
            logger.debug(f"Discovered {len(nodes)} nodes via {strategy.__name__}")

            # Auto-resolve Docker IPs if enabled
            if auto_resolve_docker:
                nodes = auto_resolve_ips(nodes, timeout, _is_ollama_running)

            # Deduplicate localhost vs real IP
            nodes = _deduplicate_nodes(nodes)

            return nodes

    # Fallback: only use localhost if not excluded
    if not exclude_localhost:
        logger.debug("No nodes discovered, falling back to localhost")
        return [{"host": "localhost", "port": "11434"}]
    else:
        logger.debug("No remote Ollama nodes discovered (localhost excluded)")
        return []


def _from_environment(timeout: float, exclude_localhost: bool = False) -> List[Dict[str, str]]:
    """Check OLLAMA_HOST environment variable."""
    host = os.getenv("OLLAMA_HOST", "").strip()
    if host:
        parsed = _parse_host(host)
        # Skip if localhost and excluded (check entire 127.0.0.0/8 loopback range)
        if exclude_localhost and (
            parsed["host"] in ("localhost", "127.0.0.1") or parsed["host"].startswith("127.")
        ):
            return []
        if _is_ollama_running(parsed["host"], int(parsed["port"]), timeout):
            return [parsed]
    return []


def _from_known_locations(timeout: float, exclude_localhost: bool = False) -> List[Dict[str, str]]:
    """Check common Ollama locations."""
    locations = [
        ("localhost", 11434),
        ("127.0.0.1", 11434),
    ]

    # Skip localhost checks if excluded
    if exclude_localhost:
        return []

    results = []
    for host, port in locations:
        if _is_ollama_running(host, port, timeout):
            results.append({"host": host, "port": str(port)})

    return results


def _from_network_scan(timeout: float, exclude_localhost: bool = False) -> List[Dict[str, str]]:
    """
    Fast parallel network scan of local subnet.

    Scans the FULL subnet and returns ALL discovered nodes.
    """
    try:
        subnet = _get_local_subnet()
    except:
        return []

    # IMPORTANT: If subnet detection fails and returns loopback range, abort scan
    # The entire 127.0.0.0/8 subnet is loopback - we'd create 254 duplicate localhost nodes!
    if subnet == "127.0.0" and exclude_localhost:
        logger.info("Skipping network scan: detected loopback subnet (no real network found)")
        return []

    def check_host(ip: str) -> Optional[Dict[str, str]]:
        """Check if Ollama is running on this IP."""
        # Skip ALL localhost IPs if excluded (entire 127.0.0.0/8 is loopback)
        if exclude_localhost and (ip.startswith("127.") or ip == "localhost"):
            return None
        # Use reasonable timeout for port check (0.1s min to allow TCP handshake)
        port_timeout = max(0.1, timeout / 5)  # At least 100ms for TCP connection
        if _is_port_open(ip, 11434, port_timeout):
            if _is_ollama_running(ip, 11434, timeout):
                return {"host": ip, "port": "11434"}
        return None

    # Scan all IPs in subnet (1-254)
    all_ips = [f"{subnet}.{i}" for i in range(1, 255)]
    discovered_nodes = []

    # Parallel scan with 100 workers for speed
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = {executor.submit(check_host, ip): ip for ip in all_ips}

        for future in as_completed(futures):
            result = future.result()
            if result:
                discovered_nodes.append(result)
                logger.info(f"Discovered Ollama node: {result['host']}:{result['port']}")

    logger.info(f"Network scan complete: found {len(discovered_nodes)} nodes on {subnet}.0/24")
    return discovered_nodes


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


def _is_ollama_running(host: str, port: int, timeout: float) -> bool:
    """Verify Ollama API is actually running."""
    import requests

    try:
        resp = requests.get(f"http://{host}:{port}/api/tags", timeout=timeout)
        return resp.status_code == 200
    except:
        return False


def _get_local_subnet() -> str:
    """
    Get local subnet (e.g., '192.168.1').

    Uses trick: connect to external IP to find our local IP.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't actually connect, just determines route
        s.connect(("10.255.255.255", 1))
        local_ip = s.getsockname()[0]
        return ".".join(local_ip.split(".")[:-1])
    finally:
        s.close()


def _parse_host(host_string: str) -> Dict[str, str]:
    """
    Parse host string into dict.

    Examples:
        "localhost" -> {"host": "localhost", "port": "11434"}
        "192.168.1.100:11434" -> {"host": "192.168.1.100", "port": "11434"}
        "http://example.com:11434" -> {"host": "example.com", "port": "11434"}
    """
    # Remove http:// or https://
    host_string = host_string.replace("http://", "").replace("https://", "")

    # Split host:port
    if ":" in host_string:
        host, port = host_string.rsplit(":", 1)
        return {"host": host, "port": port}
    else:
        return {"host": host_string, "port": "11434"}


def _get_local_ip() -> Optional[str]:
    """Get the machine's primary local IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't actually connect, just determines route
        s.connect(("10.255.255.255", 1))
        local_ip = s.getsockname()[0]
        return local_ip
    except:
        return None
    finally:
        s.close()


def _deduplicate_nodes(nodes: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Remove duplicate nodes where localhost/127.0.0.1 refers to the same machine as a real IP.

    Strategy:
    - If we have both localhost/127.0.0.1 AND the machine's actual IP, keep only the real IP
    - This prevents showing the same machine twice in the node list
    """
    if not nodes:
        return nodes

    # Get this machine's actual IP
    local_ip = _get_local_ip()
    if not local_ip:
        return nodes  # Can't determine local IP, return as-is

    # Check if we have both localhost and the real IP
    # NOTE: Entire 127.0.0.0/8 subnet is loopback!
    has_localhost = any(
        node["host"] in ("localhost", "127.0.0.1") or node["host"].startswith("127.")
        for node in nodes
    )
    has_real_ip = any(node["host"] == local_ip for node in nodes)

    # If we have both, filter out ALL localhost entries (entire 127.0.0.0/8 subnet)
    if has_localhost and has_real_ip:
        logger.debug(
            f"Deduplicating: localhost aliases and {local_ip} both found, keeping only {local_ip}"
        )
        return [
            node
            for node in nodes
            if not (node["host"] in ("localhost", "127.0.0.1") or node["host"].startswith("127."))
        ]

    return nodes
