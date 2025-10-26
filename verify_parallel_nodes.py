#!/usr/bin/env python3
"""Verify that both Ollama nodes are loaded and parallel mode will be enabled."""

import sys
import os

# Add SynapticLlamas to path
sys.path.insert(0, '/home/joker/SynapticLlamas')

from node_registry import NodeRegistry

def main():
    print("=" * 60)
    print("VERIFYING PARALLEL NODE CONFIGURATION")
    print("=" * 60)
    print()

    # Create registry
    registry = NodeRegistry()

    # Load nodes from config
    nodes_config = os.path.expanduser("~/.synapticllamas_nodes.json")
    print(f"📂 Loading nodes from: {nodes_config}")

    if not os.path.exists(nodes_config):
        print(f"❌ Config file not found: {nodes_config}")
        return 1

    registry.load_config(nodes_config)
    print()

    # Check loaded nodes
    total_nodes = len(registry.nodes)
    healthy_nodes = registry.get_healthy_nodes()

    print(f"📊 Node Statistics:")
    print(f"   Total nodes: {total_nodes}")
    print(f"   Healthy nodes: {len(healthy_nodes)}")
    print()

    if total_nodes == 0:
        print("❌ No nodes loaded! Check the config file.")
        return 1

    # List nodes
    print(f"📋 Loaded Nodes:")
    for url, node in registry.nodes.items():
        status = "✅ Healthy" if node.metrics.is_healthy else "❌ Unhealthy"
        print(f"   {status} - {node.name} ({url})")
        if node.capabilities.models:
            print(f"      Models: {len(node.capabilities.models)}")
    print()

    # Check if parallel mode will be enabled
    chunks_needed = 3  # Typical for long-form generation
    use_parallel = len(healthy_nodes) >= 2 and chunks_needed > 1

    print(f"🔀 Parallel Generation Check:")
    print(f"   Healthy nodes: {len(healthy_nodes)}")
    print(f"   Chunks needed: {chunks_needed}")
    print(f"   Parallel mode enabled: {'✅ YES' if use_parallel else '❌ NO'}")
    print()

    if use_parallel:
        print("🎉 SUCCESS! Parallel distributed generation is ready!")
        print()
        print("Next steps:")
        print("   1. Start SynapticLlamas: cd ~/SynapticLlamas && python main.py")
        print("   2. Test with a long-form query")
        print("   3. Look for 'PARALLEL MULTI-NODE MODE' in the logs")
        return 0
    else:
        print("⚠️  WARNING: Parallel mode will NOT be enabled")
        if len(healthy_nodes) < 2:
            print(f"   Need at least 2 healthy nodes (have {len(healthy_nodes)})")
        return 1

if __name__ == "__main__":
    sys.exit(main())
