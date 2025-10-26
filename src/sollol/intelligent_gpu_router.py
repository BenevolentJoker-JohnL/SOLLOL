"""
Intelligent GPU Router for SOLLOL
Makes smart VRAM-aware routing decisions based on:
- Actual VRAM capacity
- Model sizes (from model_sizes.py database)
- Current VRAM usage
- Performance requirements

Ported from FlockParser and integrated with SOLLOL's architecture.
"""

import logging
from typing import Dict, List, Optional, Tuple

from sollol.model_sizes import can_fit_in_vram, estimate_model_size, get_model_size

logger = logging.getLogger(__name__)


class IntelligentGPURouter:
    """
    Smart GPU routing that makes decisions based on actual hardware capabilities.

    Prevents VRAM exhaustion by checking model sizes before routing.

    Example:
        Node with 4GB VRAM:
        - mxbai-embed-large (705MB) → GPU ✓
        - llama3.1:8b (4.7GB) → CPU ✗ (too large)
        - llama3.2:3b (1.9GB) → GPU ✓
    """

    def __init__(self, registry=None, vram_monitor=None, safety_margin: float = 0.8):
        """
        Initialize Intelligent GPU Router.

        Args:
            registry: NodeRegistry instance
            vram_monitor: VRAMMonitor instance (optional)
            safety_margin: Use only this fraction of VRAM (default: 0.8 = 80%)
        """
        self.registry = registry
        self.vram_monitor = vram_monitor
        self.safety_margin = safety_margin

        # Node capabilities cache
        self.node_capabilities = {}

    def set_registry(self, registry):
        """Set the NodeRegistry after initialization."""
        self.registry = registry

    def set_vram_monitor(self, vram_monitor):
        """Set the VRAMMonitor after initialization."""
        self.vram_monitor = vram_monitor

    def _get_node_vram_info(self, node) -> Dict:
        """
        Get VRAM information for a node.

        Args:
            node: Node object from registry

        Returns:
            Dict with VRAM info
        """
        # Try to get from VRAM monitor if available
        if self.vram_monitor and hasattr(node, "url"):
            try:
                vram_info = self.vram_monitor.get_vram_stats(node.url)
                if vram_info:
                    return {
                        "total_vram_mb": vram_info.get("total_mb", 0),
                        "used_vram_mb": vram_info.get("used_mb", 0),
                        "free_vram_mb": vram_info.get("free_mb", 0),
                        "has_gpu": vram_info.get("has_gpu", False),
                    }
            except Exception as e:
                logger.debug(f"Could not get VRAM info from monitor: {e}")

        # Fallback to node capabilities
        if hasattr(node, "capabilities"):
            has_gpu = getattr(node.capabilities, "has_gpu", False)
            vram_total = getattr(node.capabilities, "vram_total_mb", 0)

            return {
                "total_vram_mb": vram_total,
                "used_vram_mb": 0,  # Unknown
                "free_vram_mb": vram_total,  # Assume all free
                "has_gpu": has_gpu,
            }

        # No info available
        return {"total_vram_mb": 0, "used_vram_mb": 0, "free_vram_mb": 0, "has_gpu": False}

    def can_fit_on_node(self, node, model_name: str) -> Tuple[bool, str]:
        """
        Check if a model can fit on a specific node's GPU.

        Uses node-specific model size lookup to handle different quantizations.

        Args:
            node: Node object
            model_name: Model name

        Returns:
            (can_fit: bool, reason: str)
        """
        vram_info = self._get_node_vram_info(node)

        # Check if node has GPU
        if not vram_info["has_gpu"]:
            return False, "Node has no GPU"

        # Get model size (node-specific to handle different quantizations)
        node_url = f"http://{node.host}:{node.port}" if hasattr(node, "host") else None
        model_size_mb = estimate_model_size(model_name, ollama_url=node_url)

        # Get usable VRAM (with safety margin)
        total_vram_mb = vram_info["total_vram_mb"]
        free_vram_mb = vram_info["free_vram_mb"]
        usable_vram_mb = int(total_vram_mb * self.safety_margin)

        # Check if model fits in total usable VRAM
        if model_size_mb > usable_vram_mb:
            return False, f"Model too large ({model_size_mb}MB > {usable_vram_mb}MB usable VRAM)"

        # Check if enough free VRAM currently available
        if model_size_mb > free_vram_mb:
            return False, f"Not enough free VRAM ({model_size_mb}MB needed, {free_vram_mb}MB free)"

        return True, f"Fits in {usable_vram_mb}MB VRAM ({free_vram_mb}MB free)"

    def find_suitable_nodes(self, model_name: str) -> Tuple[bool, List]:
        """
        Find all nodes that can accommodate a model.

        Args:
            model_name: Model name

        Returns:
            (has_suitable_nodes: bool, suitable_nodes: List)
        """
        if not self.registry:
            logger.warning("No registry available for GPU routing")
            return False, []

        nodes = list(self.registry.nodes.values())
        suitable_nodes = []

        for node in nodes:
            can_fit, reason = self.can_fit_on_node(node, model_name)
            if can_fit:
                suitable_nodes.append(
                    {"node": node, "reason": reason, "vram_info": self._get_node_vram_info(node)}
                )

        return len(suitable_nodes) > 0, suitable_nodes

    def route_model(self, model_name: str) -> Dict:
        """
        Intelligently route a model to the best node with VRAM pre-check.

        Args:
            model_name: Model name

        Returns:
            Routing decision dict:
            {
                'model': str,
                'node': Node or None,
                'target': 'GPU' or 'CPU',
                'reason': str,
                'can_use_gpu': bool
            }
        """
        if not self.registry:
            return {
                "model": model_name,
                "node": None,
                "target": "CPU",
                "reason": "No registry available",
                "can_use_gpu": False,
            }

        model_size_mb = estimate_model_size(model_name)
        logger.info(f"🎯 Intelligent routing for {model_name} ({model_size_mb}MB)")

        # Find suitable GPU nodes
        has_suitable, suitable_nodes = self.find_suitable_nodes(model_name)

        if has_suitable:
            # Pick node with most free VRAM
            best = max(suitable_nodes, key=lambda x: x["vram_info"]["free_vram_mb"])

            logger.info(
                f"   ✅ GPU routing: {best['node'].url} "
                f"({best['vram_info']['free_vram_mb']}MB free)"
            )

            return {
                "model": model_name,
                "node": best["node"],
                "target": "GPU",
                "reason": best["reason"],
                "can_use_gpu": True,
                "all_suitable": suitable_nodes,
            }
        else:
            # No suitable GPU nodes - route to CPU or fail
            logger.warning(f"   ⚠️ No suitable GPU nodes for {model_name} ({model_size_mb}MB)")

            # Find any healthy node for CPU fallback
            cpu_nodes = [n for n in self.registry.nodes.values() if n.is_healthy]

            if cpu_nodes:
                # Pick node with lowest load
                best_cpu = min(cpu_nodes, key=lambda n: n.metrics.get_avg_latency())

                logger.info(f"   ℹ️ CPU fallback: {best_cpu.url}")

                return {
                    "model": model_name,
                    "node": best_cpu,
                    "target": "CPU",
                    "reason": f"Model too large for available GPU VRAM ({model_size_mb}MB)",
                    "can_use_gpu": False,
                }
            else:
                logger.error(f"   ❌ No healthy nodes available")

                return {
                    "model": model_name,
                    "node": None,
                    "target": "CPU",
                    "reason": "No healthy nodes available",
                    "can_use_gpu": False,
                }

    def get_cluster_capacity(self) -> Dict:
        """
        Get overall cluster VRAM capacity info.

        Returns:
            Dict with cluster capacity statistics
        """
        if not self.registry:
            return {"error": "No registry available"}

        nodes = list(self.registry.nodes.values())
        gpu_nodes = []

        for node in nodes:
            vram_info = self._get_node_vram_info(node)
            if vram_info["has_gpu"]:
                gpu_nodes.append(
                    {
                        "url": node.url,
                        "total_mb": vram_info["total_vram_mb"],
                        "free_mb": vram_info["free_vram_mb"],
                        "usable_mb": int(vram_info["total_vram_mb"] * self.safety_margin),
                    }
                )

        if not gpu_nodes:
            return {
                "total_nodes": len(nodes),
                "gpu_nodes": 0,
                "total_vram_mb": 0,
                "free_vram_mb": 0,
                "usable_vram_mb": 0,
            }

        total_vram = sum(n["total_mb"] for n in gpu_nodes)
        free_vram = sum(n["free_mb"] for n in gpu_nodes)
        usable_vram = sum(n["usable_mb"] for n in gpu_nodes)

        return {
            "total_nodes": len(nodes),
            "gpu_nodes": len(gpu_nodes),
            "total_vram_mb": total_vram,
            "free_vram_mb": free_vram,
            "usable_vram_mb": usable_vram,
            "nodes": gpu_nodes,
        }

    def print_cluster_report(self):
        """Print comprehensive cluster VRAM capacity report."""
        capacity = self.get_cluster_capacity()

        print("\n" + "=" * 80)
        print("🧠 SOLLOL INTELLIGENT GPU ROUTER - CLUSTER REPORT")
        print("=" * 80)

        if capacity.get("error"):
            print(f"\n❌ {capacity['error']}")
            print("=" * 80 + "\n")
            return

        print(f"\nTotal Nodes: {capacity['total_nodes']}")
        print(f"GPU Nodes: {capacity['gpu_nodes']}")

        if capacity["gpu_nodes"] > 0:
            print(f"\nCluster VRAM:")
            print(
                f"  Total: {capacity['total_vram_mb']}MB ({capacity['total_vram_mb']/1024:.1f}GB)"
            )
            print(
                f"  Usable: {capacity['usable_vram_mb']}MB ({capacity['usable_vram_mb']/1024:.1f}GB)"
            )
            print(f"  Free: {capacity['free_vram_mb']}MB ({capacity['free_vram_mb']/1024:.1f}GB)")

            print("\nPer-Node Breakdown:")
            for node_info in capacity.get("nodes", []):
                print(f"\n  🚀 {node_info['url']}")
                print(f"     Total: {node_info['total_mb']}MB")
                print(f"     Usable: {node_info['usable_mb']}MB (80% safety)")
                print(f"     Free: {node_info['free_mb']}MB")
        else:
            print("\n⚠️ No GPU nodes detected")

        print("\n" + "=" * 80 + "\n")


__all__ = ["IntelligentGPURouter"]
