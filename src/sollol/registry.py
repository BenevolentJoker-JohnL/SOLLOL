"""
Node Registry for SOLLOL

Manages Ollama nodes with:
- Add/remove nodes
- Health checking
- Network discovery
- Configuration persistence
- GPU detection
- Capabilities detection
- Node clustering for layer partitioning

This makes SOLLOL a complete replacement for distributed Ollama deployments.
"""

import concurrent.futures
import json
import logging
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Union

import requests

from sollol.node_cluster import NodeCluster, needs_partitioning

# Import compatible OllamaNode implementation
from sollol.ollama_node import NodeCapabilities, NodeMetrics, OllamaNode

logger = logging.getLogger(__name__)


class NodeRegistry:
    """
    Central registry for managing Ollama nodes.

    Features:
    - Add/remove nodes manually
    - Discover nodes on network automatically
    - Health checking with configurable intervals
    - Node capability detection (GPU, memory, etc.)
    - Configuration persistence (save/load)
    - Filtering (healthy, GPU, by priority)
    """

    def __init__(self):
        self.nodes: Dict[str, OllamaNode] = {}
        self.clusters: Dict[str, NodeCluster] = {}  # name -> cluster
        self._health_check_interval = 30  # seconds

    def add_node(
        self, url: str, name: Optional[str] = None, priority: int = 5, check_health: bool = True
    ) -> OllamaNode:
        """
        Add a node to the registry.

        Args:
            url: Node URL (e.g., "http://192.168.1.10:11434")
            name: Optional friendly name (auto-generated if None)
            priority: Node priority (0-10, lower = preferred)
            check_health: Run health check immediately (default: True)

        Returns:
            OllamaNode instance

        Raises:
            ValueError: If node already exists or is unhealthy
        """
        # Normalize URL
        if not url.startswith("http"):
            url = f"http://{url}"
        if ":" not in url.split("//")[-1]:
            url = f"{url}:11434"

        # Check if already exists
        if url in self.nodes:
            logger.warning(f"Node {url} already exists in registry")
            return self.nodes[url]

        # Check for IP duplicates (localhost, 127.0.0.1, or actual IP)
        host = url.split("//")[1].split(":")[0]
        port = url.split(":")[-1]

        try:
            # Resolve hostname to IP
            new_ip = socket.gethostbyname(host)

            # Get local IPs for duplicate detection
            local_ips = {"127.0.0.1", "localhost"}
            try:
                local_hostname = socket.gethostname()
                local_ips.add(socket.gethostbyname(local_hostname))
            except:
                pass

            # Check if this IP already exists in registry
            for existing_url, existing_node in self.nodes.items():
                existing_host = existing_url.split("//")[1].split(":")[0]
                existing_port = existing_url.split(":")[-1]

                # Same port required for duplicate
                if port != existing_port:
                    continue

                try:
                    existing_ip = socket.gethostbyname(existing_host)

                    # Direct IP match
                    if new_ip == existing_ip:
                        logger.warning(
                            f"⚠️  Node {url} is a duplicate of {existing_url} (same IP: {new_ip}). Using existing node."
                        )
                        return existing_node

                    # Both are localhost
                    if new_ip in local_ips and existing_ip in local_ips:
                        logger.warning(
                            f"⚠️  Node {url} is a duplicate of {existing_url} (both localhost). Using existing node."
                        )
                        return existing_node
                except:
                    pass
        except:
            pass  # If resolution fails, continue with original logic

        # Generate name if not provided
        if not name:
            host = url.split("//")[1].split(":")[0]
            name = f"ollama-{host}"

        # Create node (class-based OllamaNode)
        node = OllamaNode(url=url, name=name, priority=priority)
        # Note: capabilities and metrics are auto-initialized in __init__

        # Health check and capability detection
        if check_health:
            self._check_node_health(node)
            self._detect_capabilities(node)

        if not node.is_healthy:
            raise ValueError(f"Node {url} is unhealthy, not adding to registry")

        self.nodes[url] = node
        logger.info(f"✅ Added node: {name} ({url}) - Priority: {priority}")
        return node

    def remove_node(self, url: str) -> bool:
        """
        Remove a node from the registry.

        Args:
            url: Node URL to remove

        Returns:
            True if removed, False if not found
        """
        if url in self.nodes:
            node = self.nodes.pop(url)
            logger.info(f"🗑️  Removed node: {node.name} ({url})")
            return True

        logger.warning(f"Node {url} not found in registry")
        return False

    def discover_nodes(
        self,
        cidr: str = "192.168.1.0/24",
        port: int = 11434,
        timeout: float = 2.0,
        max_workers: int = 50,
    ) -> List[OllamaNode]:
        """
        Discover Ollama nodes on the network.

        Args:
            cidr: Network CIDR to scan (e.g., "192.168.1.0/24")
            port: Port to check (default: 11434)
            timeout: Connection timeout in seconds (default: 2.0)
            max_workers: Max parallel workers for scanning (default: 50)

        Returns:
            List of discovered OllamaNode instances
        """
        logger.info(f"🔍 Discovering Ollama nodes on {cidr}:{port}")

        # Generate IP list from CIDR
        ips = self._generate_ips_from_cidr(cidr)
        discovered = []

        # Parallel scan
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._check_ollama_endpoint, ip, port, timeout): ip for ip in ips
            }

            for future in concurrent.futures.as_completed(futures):
                ip = futures[future]
                try:
                    is_ollama, info = future.result()
                    if is_ollama:
                        url = f"http://{ip}:{port}"

                        # Add if not already in registry
                        if url not in self.nodes:
                            try:
                                node = self.add_node(url, check_health=True)
                                discovered.append(node)
                                logger.info(
                                    f"🔍 Discovered: ✓ {'🎮' if node.capabilities.has_gpu else '💻'} "
                                    f"{node.name} ({url}) - Load: {node.calculate_load_score():.2f}"
                                )
                            except ValueError:
                                pass  # Node unhealthy, skip
                except Exception as e:
                    logger.debug(f"Error checking {ip}: {e}")

        logger.info(f"✅ Discovered {len(discovered)} nodes")
        return discovered

    def health_check_all(self) -> Dict[str, bool]:
        """
        Run health checks on all nodes.

        Returns:
            Dict mapping node URL to health status
        """
        logger.info("🏥 Running health checks on all nodes")

        results = {}
        for url, node in self.nodes.items():
            self._check_node_health(node)
            results[url] = node.is_healthy

        healthy_count = sum(1 for h in results.values() if h)
        logger.info(f"✅ Health check complete: {healthy_count}/{len(results)} healthy")

        return results

    def get_healthy_nodes(self) -> List[OllamaNode]:
        """Get all healthy nodes."""
        return [node for node in self.nodes.values() if node.is_healthy]

    def get_gpu_nodes(self) -> List[OllamaNode]:
        """Get all nodes with GPU capabilities."""
        return [
            node for node in self.nodes.values() if node.is_healthy and node.capabilities.has_gpu
        ]

    def get_node_by_url(self, url: str) -> Optional[OllamaNode]:
        """Get node by URL."""
        return self.nodes.get(url)

    def create_cluster(
        self, name: str, node_urls: List[str], model: str, partitioning_strategy: str = "even"
    ) -> NodeCluster:
        """
        Create a node cluster for distributed model inference.

        Args:
            name: Cluster identifier
            node_urls: List of node URLs to include in cluster
            model: Model to partition (e.g., "llama2:70b")
            partitioning_strategy: How to distribute layers

        Returns:
            NodeCluster instance

        Raises:
            ValueError: If nodes don't exist or insufficient
        """
        # Validate nodes exist and are healthy
        cluster_nodes = []
        for url in node_urls:
            node = self.nodes.get(url)
            if not node:
                raise ValueError(f"Node {url} not found in registry")
            if not node.is_healthy:
                raise ValueError(f"Node {url} is unhealthy, cannot add to cluster")
            cluster_nodes.append(node)

        if len(cluster_nodes) < 2:
            raise ValueError("Cluster requires at least 2 nodes")

        # Create cluster
        cluster = NodeCluster(
            name=name, nodes=cluster_nodes, model=model, partitioning_strategy=partitioning_strategy
        )

        self.clusters[name] = cluster
        logger.info(f"✅ Created cluster: {name} with {len(cluster_nodes)} nodes")
        return cluster

    def remove_cluster(self, name: str) -> bool:
        """
        Remove a cluster.

        Args:
            name: Cluster name

        Returns:
            True if removed, False if not found
        """
        if name in self.clusters:
            cluster = self.clusters.pop(name)
            logger.info(f"🗑️  Removed cluster: {name}")
            return True

        logger.warning(f"Cluster {name} not found")
        return False

    def get_cluster(self, name: str) -> Optional[NodeCluster]:
        """Get cluster by name."""
        return self.clusters.get(name)

    def get_all_clusters(self) -> List[NodeCluster]:
        """Get all clusters."""
        return list(self.clusters.values())

    def get_healthy_clusters(self) -> List[NodeCluster]:
        """Get all healthy clusters."""
        return [c for c in self.clusters.values() if c.is_healthy]

    async def health_check_clusters(self) -> Dict[str, bool]:
        """
        Run health checks on all clusters.

        Returns:
            Dict mapping cluster name to health status
        """
        results = {}
        for name, cluster in self.clusters.items():
            is_healthy = await cluster.health_check()
            results[name] = is_healthy

        return results

    def get_worker_for_model(
        self, model: str, prefer_cluster: bool = True
    ) -> Union[OllamaNode, NodeCluster, None]:
        """
        Get best worker (node or cluster) for a model.

        For large models (70B+), returns cluster if available.
        For small models, returns individual node.

        Args:
            model: Model name
            prefer_cluster: Prefer cluster for large models

        Returns:
            OllamaNode or NodeCluster, or None if unavailable
        """
        requires_partition = needs_partitioning(model)

        if requires_partition and prefer_cluster:
            # Look for cluster with this model
            for cluster in self.clusters.values():
                if cluster.model == model and cluster.is_healthy:
                    logger.info(f"🔗 Routing {model} to cluster: {cluster.name}")
                    return cluster

            # No cluster available - check if we can create one
            healthy_nodes = self.get_healthy_nodes()
            if len(healthy_nodes) >= 2:
                logger.info(
                    f"⚠️  No cluster for {model}, but {len(healthy_nodes)} nodes available. "
                    "Consider creating cluster with registry.create_cluster()"
                )

        # Fall back to individual node
        healthy_nodes = self.get_healthy_nodes()
        if healthy_nodes:
            # Use node with lowest load
            best_node = min(healthy_nodes, key=lambda n: n.calculate_load_score())
            logger.info(f"📍 Routing {model} to node: {best_node.name}")
            return best_node

        return None

    def save_config(self, filepath: str):
        """
        Save node configuration to JSON file.

        Args:
            filepath: Path to save configuration
        """
        config = {
            "version": "1.0",
            "nodes": [
                {"url": node.url, "name": node.name, "priority": node.priority}
                for node in self.nodes.values()
            ],
        }

        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"💾 Saved {len(self.nodes)} nodes to {filepath}")

    def load_config(self, filepath: str) -> int:
        """
        Load node configuration from JSON file.

        Args:
            filepath: Path to load configuration from

        Returns:
            Number of nodes loaded
        """
        try:
            with open(filepath, "r") as f:
                config = json.load(f)

            loaded_count = 0
            for node_config in config.get("nodes", []):
                try:
                    self.add_node(
                        url=node_config["url"],
                        name=node_config.get("name"),
                        priority=node_config.get("priority", 5),
                        check_health=True,
                    )
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to load node {node_config['url']}: {e}")

            logger.info(f"📂 Loaded {loaded_count} nodes from {filepath}")
            return loaded_count

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return 0

    def _check_node_health(self, node: OllamaNode):
        """Check node health and update status."""
        try:
            response = requests.get(f"{node.url}/api/tags", timeout=5)
            # Update metrics (class-based OllamaNode)
            node.metrics.is_healthy = response.status_code == 200
            node.metrics.last_health_check = datetime.now()

            if not node.is_healthy:
                logger.warning(f"⚠️  Node {node.url} unhealthy (status: {response.status_code})")
        except Exception as e:
            # Update metrics (class-based OllamaNode)
            node.metrics.is_healthy = False
            node.metrics.last_health_check = datetime.now()
            # last_error is a property, skip for now
            logger.warning(f"⚠️  Node {node.url} unreachable: {e}")

    def _detect_capabilities(self, node: OllamaNode):
        """Detect node capabilities (GPU, memory, etc.)."""
        try:
            # Try to get version info (may include GPU info in some Ollama versions)
            response = requests.get(f"{node.url}/api/version", timeout=5)
            if response.ok:
                # For now, basic detection
                # TODO: Add proper GPU detection when Ollama exposes this via API
                node.capabilities.has_gpu = False  # Default
                node.capabilities.cpu_cores = 4  # Default (matches OllamaNode default)
        except Exception as e:
            logger.debug(f"Could not detect capabilities for {node.url}: {e}")

    def _check_ollama_endpoint(self, ip: str, port: int, timeout: float) -> tuple:
        """
        Check if an IP:port is running Ollama.

        Returns:
            (is_ollama, info_dict)
        """
        try:
            url = f"http://{ip}:{port}/api/tags"
            response = requests.get(url, timeout=timeout)

            if response.status_code == 200:
                return True, {"url": f"http://{ip}:{port}"}
        except:
            pass

        return False, {}

    def _generate_ips_from_cidr(self, cidr: str) -> List[str]:
        """Generate list of IPs from CIDR notation."""
        import ipaddress

        network = ipaddress.ip_network(cidr, strict=False)
        return [str(ip) for ip in network.hosts()]

    def __len__(self):
        return len(self.nodes)

    def __repr__(self):
        healthy = len(self.get_healthy_nodes())
        gpu = len(self.get_gpu_nodes())
        clusters = len(self.clusters)
        healthy_clusters = len([c for c in self.clusters.values() if c.is_healthy])
        return (
            f"NodeRegistry(nodes={len(self)}, healthy={healthy}, gpu={gpu}, "
            f"clusters={clusters}, healthy_clusters={healthy_clusters})"
        )


# Convenience exports
__all__ = ["NodeRegistry", "OllamaNode", "NodeCapabilities", "NodeMetrics"]
