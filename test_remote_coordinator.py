#!/usr/bin/env python3
"""
Test script for remote coordinator execution.

Tests the intelligent node selection logic without requiring a full Ray cluster.
"""

import asyncio
import logging
import sys
from typing import Any, Dict, List

# Add src to path
sys.path.insert(0, "src")

from sollol.ray_hybrid_router import ShardedModelPool
from sollol.rpc_discovery import detect_node_resources

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mock_detect_node_resources(host: str) -> Dict[str, Any]:
    """
    Mock resource detection for testing.

    Simulates the cluster:
    - 10.9.66.154: 16GB RAM (low resources)
    - 10.9.66.90: 128GB RAM + GPU (high resources)
    - 10.9.66.48: 32GB RAM (medium resources)
    """
    resources = {
        "10.9.66.154": {
            "has_gpu": False,
            "gpu_devices": [],
            "gpu_vram_mb": [],
            "cpu_ram_mb": 16000,  # 16GB
            "device_config": "cpu",
            "memory_config": "16000",
            "total_parallel_workers": 1,
        },
        "10.9.66.90": {
            "has_gpu": True,
            "gpu_devices": ["cuda:0"],
            "gpu_vram_mb": [24000],  # 24GB VRAM
            "cpu_ram_mb": 128000,  # 128GB RAM
            "device_config": "cpu,cuda:0",
            "memory_config": "128000,24000",
            "total_parallel_workers": 2,
        },
        "10.9.66.48": {
            "has_gpu": False,
            "gpu_devices": [],
            "gpu_vram_mb": [],
            "cpu_ram_mb": 32000,  # 32GB
            "device_config": "cpu",
            "memory_config": "32000",
            "total_parallel_workers": 1,
        },
    }

    return resources.get(host, {
        "has_gpu": False,
        "gpu_devices": [],
        "gpu_vram_mb": [],
        "cpu_ram_mb": 8000,
        "device_config": "cpu",
        "memory_config": "8000",
        "total_parallel_workers": 1,
    })


def select_best_coordinator_node(
    model: str,
    rpc_backends: List[Dict[str, Any]],
    enable_remote: bool = True
) -> str:
    """
    Standalone version of node selection logic for testing.

    This is extracted from ShardedModelPool._select_best_coordinator_node()
    """
    if not enable_remote or not rpc_backends:
        return None

    import re

    # Estimate model requirements
    size_match = re.search(r"(\d+)b", model.lower())
    if size_match:
        size_billions = int(size_match.group(1))
        estimated_ram_mb = size_billions * 2 * 1024  # ~2GB per billion
    else:
        estimated_ram_mb = 16384  # Default 16GB

    logger.info(
        f"Selecting coordinator node for {model} (estimated {estimated_ram_mb}MB needed)"
    )

    # Score each backend node
    best_node = None
    best_score = -1

    for backend in rpc_backends:
        host = backend["host"]

        # Query resources (will use mocked version when patched)
        from sollol.rpc_discovery import detect_node_resources as get_resources
        resources = get_resources(host)

        # Calculate score
        cpu_ram = resources.get("cpu_ram_mb", 0)
        gpu_vram = sum(resources.get("gpu_vram_mb", []))
        total_ram = cpu_ram + gpu_vram

        score = total_ram - estimated_ram_mb

        # GPU bonus
        if resources.get("has_gpu", False):
            score += 5000

        logger.info(
            f"  {host}: RAM={cpu_ram}MB, GPU_VRAM={gpu_vram}MB, score={score:.0f}"
        )

        if score > best_score:
            best_score = score
            best_node = host

    if best_node and best_score > 0:
        logger.info(f"Selected {best_node} (score={best_score:.0f})")
        return best_node
    else:
        logger.warning(f"No suitable node (best_score={best_score}), using local")
        return None


def test_node_selection():
    """Test the node selection logic for various model sizes."""

    # Mock RPC backends
    rpc_backends = [
        {"host": "10.9.66.154", "port": 50052},
        {"host": "10.9.66.90", "port": 50052},
        {"host": "10.9.66.48", "port": 50052},
    ]

    # Patch detect_node_resources
    import sollol.rpc_discovery
    original_detect = sollol.rpc_discovery.detect_node_resources
    sollol.rpc_discovery.detect_node_resources = mock_detect_node_resources

    try:
        # Test cases
        test_cases = [
            ("llama3.1:8b", 8, "Small model (8B)"),
            ("llama3.1:70b", 70, "Large model (70B)"),
            ("llama3.1:405b", 405, "Huge model (405B)"),
        ]

        print("\n" + "="*70)
        print("Remote Coordinator Node Selection Test")
        print("="*70 + "\n")

        for model_name, size_b, description in test_cases:
            print(f"Test: {description}")
            print(f"Model: {model_name} (~{size_b * 2}GB estimated)")
            print("-" * 70)

            # Call the selection function
            selected_node = select_best_coordinator_node(
                model=model_name,
                rpc_backends=rpc_backends
            )

            if selected_node:
                print(f"✅ Selected node: {selected_node}")
                resources = mock_detect_node_resources(selected_node)
                print(f"   RAM: {resources['cpu_ram_mb']}MB")
                print(f"   GPU: {resources['has_gpu']} ({resources.get('gpu_vram_mb', [])})")
            else:
                print(f"⚠️  No suitable node found - will use local execution")

            print()

        print("="*70)
        print("Test completed successfully!")
        print("="*70)

    finally:
        # Restore original function
        sollol.rpc_discovery.detect_node_resources = original_detect


if __name__ == "__main__":
    test_node_selection()
