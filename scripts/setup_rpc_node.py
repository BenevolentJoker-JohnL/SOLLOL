#!/usr/bin/env python3
"""
RPC Node Setup Helper - Generate optimal rpc-server command with hybrid GPU+CPU parallelization

This script:
1. Detects GPU(s) and their VRAM
2. Detects available system RAM
3. Calculates safe allocations (80% of each to avoid crashes)
4. Generates rpc-server command with hybrid device config

Example output for GPU node:
  rpc-server --host 0.0.0.0 --port 50052 --device cpu,cuda:0 --mem 12000,9600

This creates 2 parallel workers on 1 physical machine:
  - CPU worker: 12GB RAM
  - GPU worker: 9.6GB VRAM

For 3 physical nodes (2 CPU + 1 GPU), you get 4 total parallel workers!
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sollol.rpc_discovery import detect_node_resources
import json


def main():
    print("=" * 70)
    print("RPC NODE SETUP - Hybrid GPU+CPU Parallelization")
    print("=" * 70)
    print()

    print("🔍 Detecting local resources...")
    resources = detect_node_resources('localhost')

    print()
    print("=" * 70)
    print("DETECTED RESOURCES")
    print("=" * 70)

    if resources["has_gpu"]:
        print(f"✅ GPU(s) Found: {len(resources['gpu_devices'])}")
        for i, (device, vram) in enumerate(zip(resources['gpu_devices'], resources['gpu_vram_mb'])):
            print(f"   GPU {i}: {device} - {vram} MB VRAM (safe allocation)")
        print()

    print(f"💾 CPU RAM: {resources['cpu_ram_mb']} MB (safe allocation)")
    print()
    print(f"⚡ Total Parallel Workers: {resources['total_parallel_workers']}")
    if resources['total_parallel_workers'] > 1:
        print(f"   (1 CPU worker + {resources['total_parallel_workers']-1} GPU worker(s))")
    print()

    print("=" * 70)
    print("GENERATED RPC-SERVER COMMAND")
    print("=" * 70)
    cmd = f"rpc-server --host 0.0.0.0 --port 50052 --device {resources['device_config']} --mem {resources['memory_config']}"
    print(cmd)
    print()

    if resources["has_gpu"]:
        print("💡 This command creates HYBRID parallelization:")
        print(f"   • CPU device processes layers using {resources['cpu_ram_mb']} MB RAM")
        for i, (device, vram) in enumerate(zip(resources['gpu_devices'], resources['gpu_vram_mb'])):
            print(f"   • {device} processes layers using {vram} MB VRAM")
        print()
        print(f"   ALL {resources['total_parallel_workers']} devices work IN PARALLEL on this single node!")
    else:
        print("💡 This is a CPU-only node - contributes 1 parallel worker")

    print()
    print("=" * 70)
    print("CLUSTER SETUP EXAMPLE")
    print("=" * 70)
    print()
    print("With 3 physical nodes:")
    print("  • CPU Node 1: 1 worker")
    print("  • CPU Node 2: 1 worker")
    print("  • GPU Node (hybrid): 2 workers (CPU + GPU)")
    print()
    print("Total: 4 parallel workers across 3 machines! 🚀")
    print()

    # Save to JSON for programmatic access
    output_file = "/tmp/rpc_node_config.json"
    with open(output_file, "w") as f:
        json.dump({
            "resources": resources,
            "command": cmd
        }, f, indent=2)

    print(f"📄 Config saved to: {output_file}")

    # Publish to Redis for remote discovery
    try:
        import redis
        import socket

        redis_url = os.getenv("SOLLOL_REDIS_URL", "redis://localhost:6379")
        r = redis.from_url(redis_url, decode_responses=True)

        # Get this node's IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        node_ip = s.getsockname()[0]
        s.close()

        # Publish node config to Redis with 1 hour expiration
        key = f"sollol:rpc:node:{node_ip}:50052"
        r.set(key, json.dumps(resources), ex=3600)

        print(f"📡 Published config to Redis: {redis_url}")
        print(f"   Key: {key}")
        print()
    except Exception as e:
        print(f"⚠️  Could not publish to Redis: {e}")
        print(f"   (This is optional - coordinator can still use manual RPC backend list)")
        print()


if __name__ == "__main__":
    main()
