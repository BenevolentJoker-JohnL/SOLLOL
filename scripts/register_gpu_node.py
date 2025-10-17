#!/usr/bin/env python3
"""
Standalone GPU Node Registration - Detect GPU and publish to Redis

This script can be run on remote RPC nodes to detect their GPU
and publish the configuration to the central Redis server.

Usage:
    python register_gpu_node.py --redis-host 10.9.66.154

The script will:
1. Detect local GPU(s) using nvidia-smi
2. Calculate safe VRAM and RAM allocations
3. Publish config to Redis for coordinator discovery
4. Display the optimal rpc-server command to run

Requirements:
    - python3
    - redis-py: pip install redis
    - nvidia-smi (for GPU nodes)
"""

import argparse
import json
import socket
import subprocess
import sys
from typing import Any, Dict, List, Optional


def get_nvidia_gpus() -> List[Dict[str, Any]]:
    """
    Detect NVIDIA GPUs using nvidia-smi.

    Returns:
        List of dicts with GPU info: [{"index": 0, "name": "RTX 3090", "memory_mb": 24576}, ...]
    """
    try:
        # Query nvidia-smi for GPU info
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            check=True
        )

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_mb": int(parts[2]),
                        "vendor": "nvidia"
                    })

        return gpus
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def get_system_ram_mb() -> int:
    """Get available system RAM in MB."""
    try:
        with open("/proc/meminfo", "r") as f:
            mem_total_kb = int(f.readline().split()[1])
            return mem_total_kb // 1024
    except:
        return 16000  # Conservative default


def detect_resources() -> Dict[str, Any]:
    """
    Detect GPU and RAM resources on this node.

    Returns:
        Resource configuration dict
    """
    gpus = get_nvidia_gpus()
    total_ram = get_system_ram_mb()

    # Reserve 20% for OS
    available_ram_mb = int(total_ram * 0.8)

    if gpus:
        # GPU node - configure hybrid CPU + GPU
        gpu_devices = []
        gpu_vram_mb = []

        for gpu in gpus:
            idx = gpu["index"]
            gpu_devices.append(f"cuda:{idx}")

            # Reserve 20% VRAM for safety
            total_vram = gpu["memory_mb"]
            safe_vram = int(total_vram * 0.8)
            gpu_vram_mb.append(safe_vram)

        # Hybrid config: cpu + gpu(s)
        devices = ["cpu"] + gpu_devices
        memory = [available_ram_mb] + gpu_vram_mb

        return {
            "has_gpu": True,
            "gpu_devices": gpu_devices,
            "gpu_vram_mb": gpu_vram_mb,
            "gpu_names": [g["name"] for g in gpus],
            "cpu_ram_mb": available_ram_mb,
            "device_config": ",".join(devices),
            "memory_config": ",".join(str(m) for m in memory),
            "total_parallel_workers": len(devices),
        }
    else:
        # CPU-only node
        return {
            "has_gpu": False,
            "gpu_devices": [],
            "gpu_vram_mb": [],
            "gpu_names": [],
            "cpu_ram_mb": available_ram_mb,
            "device_config": "cpu",
            "memory_config": str(available_ram_mb),
            "total_parallel_workers": 1,
        }


def get_node_ip() -> str:
    """Get this node's IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "unknown"


def publish_to_redis(redis_host: str, redis_port: int, resources: Dict[str, Any], node_ip: str, rpc_port: int = 50052):
    """Publish node config to Redis."""
    try:
        import redis

        redis_url = f"redis://{redis_host}:{redis_port}"
        r = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=2)

        # Test connection
        r.ping()

        # Publish with 1 hour expiration
        key = f"sollol:rpc:node:{node_ip}:{rpc_port}"
        r.set(key, json.dumps(resources), ex=3600)

        print(f"✅ Published to Redis: {redis_url}")
        print(f"   Key: {key}")
        print(f"   TTL: 1 hour")
        return True
    except Exception as e:
        print(f"❌ Failed to publish to Redis: {e}")
        print(f"   Make sure Redis is accessible at {redis_host}:{redis_port}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Register GPU node with SOLLOL coordinator via Redis"
    )
    parser.add_argument(
        "--redis-host",
        default="localhost",
        help="Redis server hostname/IP (default: localhost)"
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis server port (default: 6379)"
    )
    parser.add_argument(
        "--rpc-port",
        type=int,
        default=50052,
        help="RPC server port on this node (default: 50052)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GPU NODE REGISTRATION - SOLLOL")
    print("=" * 70)
    print()

    # Detect node IP
    node_ip = get_node_ip()
    print(f"📍 Node IP: {node_ip}")
    print()

    # Detect resources
    print("🔍 Detecting resources...")
    resources = detect_resources()
    print()

    # Display results
    print("=" * 70)
    print("DETECTED RESOURCES")
    print("=" * 70)

    if resources["has_gpu"]:
        print(f"✅ GPU(s) Found: {len(resources['gpu_devices'])}")
        for i, (name, device, vram) in enumerate(zip(
            resources['gpu_names'],
            resources['gpu_devices'],
            resources['gpu_vram_mb']
        )):
            print(f"   GPU {i}: {name} ({device}) - {vram} MB VRAM")
        print()
    else:
        print("ℹ️  No GPU detected (CPU-only node)")
        print()

    print(f"💾 CPU RAM: {resources['cpu_ram_mb']} MB")
    print(f"⚡ Parallel Workers: {resources['total_parallel_workers']}")
    print()

    # Generate RPC command
    print("=" * 70)
    print("RPC-SERVER COMMAND")
    print("=" * 70)
    cmd = f"rpc-server --host 0.0.0.0 --port {args.rpc_port} --device {resources['device_config']} --mem {resources['memory_config']}"
    print(cmd)
    print()

    # Publish to Redis
    print("=" * 70)
    print("REDIS REGISTRATION")
    print("=" * 70)

    success = publish_to_redis(
        args.redis_host,
        args.redis_port,
        resources,
        node_ip,
        args.rpc_port
    )
    print()

    if success:
        print("=" * 70)
        print("✅ REGISTRATION COMPLETE")
        print("=" * 70)
        print()
        print("Next steps:")
        print("1. Start RPC server with the command above")
        print("2. Coordinator will automatically discover this node")
        print(f"3. Re-run this script every hour (or use cron) to keep registration fresh")
        print()
    else:
        print("=" * 70)
        print("⚠️  REGISTRATION FAILED")
        print("=" * 70)
        print()
        print("Troubleshooting:")
        print(f"1. Check Redis is accessible: redis-cli -h {args.redis_host} ping")
        print(f"2. Check network connectivity to {args.redis_host}")
        print(f"3. Check Redis is listening on network interface (not just localhost)")
        print()
        print("You can still run the RPC server manually,")
        print("but coordinator won't know about GPU capabilities.")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
