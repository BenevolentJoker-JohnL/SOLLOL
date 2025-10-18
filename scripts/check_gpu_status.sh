#!/bin/bash
# SOLLOL - Check GPU detection and RPC node status
# Run this to verify GPU detection is working

echo "=========================================="
echo "SOLLOL: GPU Detection Status Check"
echo "=========================================="
echo ""

echo "📍 1. Redis Network Status"
echo "────────────────────────────────────────"
echo "Current bind configuration:"
redis-cli CONFIG GET bind | tail -1
echo ""
echo "Network listening status:"
netstat -tuln 2>/dev/null | grep 6379 || ss -tuln | grep 6379
echo ""

echo "📍 2. Redis GPU Registration Keys"
echo "────────────────────────────────────────"
redis-cli KEYS "sollol:rpc:node:*" | head -20
if [ $(redis-cli KEYS "sollol:rpc:node:*" | wc -l) -eq 0 ]; then
    echo "❌ No GPU nodes registered in Redis"
    echo "   GPU nodes need to run: python3 scripts/register_gpu_node.py --redis-host <coordinator-ip>"
else
    echo ""
    echo "Registered nodes:"
    for key in $(redis-cli KEYS "sollol:rpc:node:*"); do
        echo ""
        echo "Key: $key"
        redis-cli GET "$key" | python3 -m json.tool 2>/dev/null || redis-cli GET "$key"
    done
fi
echo ""

echo "📍 3. RPC Discovery Results"
echo "────────────────────────────────────────"
PYTHONPATH=src python3 -c "
from sollol.rpc_discovery import auto_discover_rpc_backends
import json

backends = auto_discover_rpc_backends()
print(f'Discovered {len(backends)} RPC backends:\n')

for backend in backends:
    has_gpu = backend.get('has_gpu', False)
    gpu_emoji = '🎮' if has_gpu else '💻'
    print(f'{gpu_emoji} {backend[\"host\"]}:{backend[\"port\"]}')
    print(f'   Has GPU: {has_gpu}')
    print(f'   GPU devices: {backend.get(\"gpu_devices\", [])}')
    print(f'   GPU VRAM: {backend.get(\"gpu_vram_mb\", [])} MB')
    print(f'   CPU RAM: {backend.get(\"cpu_ram_mb\", 0)} MB')
    print(f'   Workers: {backend.get(\"num_workers\", 1)}')
    print()
"
echo ""

echo "📍 4. RPC Server Processes"
echo "────────────────────────────────────────"
echo "Local RPC servers:"
ps aux | grep "rpc-server" | grep -v grep || echo "No RPC servers running locally"
echo ""

echo "📍 5. Network Connectivity Tests"
echo "────────────────────────────────────────"
echo "Testing RPC node connectivity:"
for host in 10.9.66.45 10.9.66.48 10.9.66.90; do
    echo -n "$host:50052 - "
    timeout 2 bash -c "echo -n '' > /dev/tcp/$host/50052 2>&1" && echo "✅ Reachable" || echo "❌ Unreachable"
done
echo ""

echo "=========================================="
echo "Summary:"
echo "=========================================="

# Check if Redis is network-accessible
if netstat -tuln 2>/dev/null | grep -q "10.9.66.154:6379" || ss -tuln 2>/dev/null | grep -q "10.9.66.154:6379"; then
    echo "✅ Redis is network-accessible"
else
    echo "❌ Redis is NOT network-accessible (only localhost)"
    echo "   Run: sudo bash scripts/configure_redis_network.sh"
fi

# Check if GPU nodes are registered
gpu_count=$(redis-cli KEYS "sollol:rpc:node:*" | grep -c "gpu_devices" || echo "0")
if [ "$gpu_count" -gt 0 ]; then
    echo "✅ GPU nodes registered in Redis"
else
    echo "❌ No GPU nodes registered in Redis"
    echo "   On GPU nodes, run: python3 scripts/register_gpu_node.py --redis-host 10.9.66.154"
fi

echo ""
