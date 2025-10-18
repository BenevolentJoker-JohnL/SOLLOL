# Ray Cluster Setup for SOLLOL Remote Coordinator Execution

## Overview

This guide shows how to set up a Ray cluster to enable **remote coordinator execution**. This allows SOLLOL to intelligently spawn coordinators on high-RAM nodes even when requests arrive on low-RAM nodes.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Ray Cluster (for distributed inference coordination only)   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Head Node (192.168.1.10 - 16GB RAM)                         │
│    - Receives HTTP requests                                  │
│    - Ray scheduler decides WHERE to run coordinator          │
│    - Streams results back to client                          │
│                                                              │
│  Worker Node (192.168.1.20 - 128GB RAM + GPU)                 │
│    - ShardedModelPool actor runs here                        │
│    - LlamaCppCoordinator starts here                         │
│    - Distributes inference to RPC backends                   │
│                                                              │
│  Worker Nodes (192.168.1.21, 192.168.1.22)                      │
│    - Available for coordinator placement                     │
│    - Ray can schedule actors here if needed                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Important**: Ray is ONLY used for:
- Scheduling where coordinators run (intelligent placement)
- Streaming results back from remote coordinators
- **NOT** for running inference itself (that's done by llama.cpp RPC)

## Prerequisites

1. **Redis running** (for GPU metadata storage)
   ```bash
   # Should already be running
   redis-cli ping  # Should return PONG
   ```

2. **RPC backends registered** with GPU metadata
   ```bash
   # On each GPU node, run:
   python3 register_rpc_gpu_node.py
   ```

3. **Ray installed** on all nodes
   ```bash
   pip install ray
   ```

## Step 1: Start Ray Head Node

**On 192.168.1.10 (current node - request receiver):**

```bash
# Start Ray head node
ray start --head \
  --port=6380 \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --num-cpus=2 \
  --object-store-memory=500000000

# Verify it's running
ray status
```

**Output should show:**
```
Ray runtime started.
---
Local node IP: 192.168.1.10
Dashboard: http://192.168.1.10:8265
---
To add worker nodes:
  ray start --address='192.168.1.10:6380'
```

## Step 2: Join Worker Nodes

**On each worker node (192.168.1.20, 192.168.1.21, 192.168.1.22):**

```bash
# Join the Ray cluster
ray start --address='192.168.1.10:6380'

# Verify connection
ray status
```

**Expected output:**
```
Ray runtime started.
Connected to Ray cluster.
```

### Optional: Custom Resources

For advanced placement, you can register custom resources:

```bash
# On high-RAM node (192.168.1.20)
ray start --address='192.168.1.10:6380' \
  --resources='{"high_memory": 1, "gpu_node": 1}'

# On medium-RAM nodes
ray start --address='192.168.1.10:6380' \
  --resources='{"medium_memory": 1}'
```

## Step 3: Verify Cluster

**On head node:**

```bash
# Check cluster status
ray status

# Should show all nodes
# Example:
# Node status
# Active:
#   4 nodes (1 head, 3 workers)
```

**Check Ray dashboard:**
```
http://192.168.1.10:8265
```

## Step 4: Test Remote Coordinator

**Start SOLLOL with remote coordinator enabled (default):**

```bash
PYTHONPATH=src python3 -c "
from sollol.ray_hybrid_router import RayHybridRouter

# Initialize with remote coordinator enabled
router = RayHybridRouter(
    enable_distributed=True,
    auto_discover_rpc=True,
    enable_remote_coordinator=True  # Default: True
)

print('✅ SOLLOL initialized with remote coordinator support')
print(f'   Ray pools: {len(router.pools)}')
print(f'   RPC backends: {len(router.rpc_backends)}')
"
```

**Expected log output:**
```
📦 Creating N sharded model pools
  Pool 0: 3 backends (port 18080, remote coordinator enabled)
  Pool 1: 3 backends (port 18081, remote coordinator enabled)

Pool 0: Selecting coordinator node for llama3.1:70b (estimated 143360MB needed)
  192.168.1.10: RAM=16000MB, GPU_VRAM=0MB, score=-127360
  192.168.1.20: RAM=128000MB, GPU_VRAM=24000MB, score=13640 ✅ BEST
  192.168.1.21: RAM=32000MB, GPU_VRAM=0MB, score=-111360

Pool 0: Selected 192.168.1.20 for coordinator (score=13640)
Pool 0: Loading llama3.1:70b across 3 RPC backends (coordinator on 192.168.1.20)
```

## How It Works

1. **Request arrives** on 192.168.1.10 (16GB RAM, low resources)

2. **SOLLOL intelligence** analyzes all RPC backend hosts:
   - Queries Redis for GPU/RAM metadata
   - Calculates score: `total_ram - estimated_model_size`
   - GPU nodes get 5GB bonus

3. **Best node selected**: 192.168.1.20 (128GB RAM + 24GB GPU)

4. **Ray places actor** on selected node:
   - `ShardedModelPool` actor runs on 192.168.1.20
   - `LlamaCppCoordinator` starts on 192.168.1.20:18080

5. **Coordinator distributes** to RPC backends:
   ```
   192.168.1.20:18080 (coordinator)
     ├─> 192.168.1.21:50052 (layers 0-20)
     ├─> 192.168.1.22:50052 (layers 21-40)
     └─> 192.168.1.20:50052 (layers 41-60)
   ```

6. **Results stream back** to 192.168.1.10 via Ray's object store

## Monitoring

### Ray Dashboard
```
http://192.168.1.10:8265
```
Shows:
- Active nodes
- Resource usage
- Actor placement
- Task execution

### SOLLOL Logs
Watch for coordinator placement decisions:
```bash
tail -f /var/log/sollol.log | grep "coordinator"
```

### Redis Metadata
Check stored GPU information:
```bash
redis-cli keys "sollol:rpc:node:*"
redis-cli get "sollol:rpc:node:192.168.1.20:50052"
```

## Troubleshooting

### Problem: "No suitable node found"
**Cause**: No node has enough RAM for the model

**Solution**:
1. Check GPU metadata is registered:
   ```bash
   redis-cli keys "sollol:rpc:node:*"
   ```
2. Re-register GPU nodes:
   ```bash
   python3 register_rpc_gpu_node.py
   ```
3. Check logs for resource scores

### Problem: "Ray cluster not connected"
**Cause**: Worker nodes can't reach head node

**Solution**:
1. Check firewall on port 6380:
   ```bash
   ss -tunlp | grep 6380
   ```
2. Test connectivity:
   ```bash
   telnet 192.168.1.10 6380
   ```
3. Restart worker nodes with correct address

### Problem: Port 6379 already in use
**Cause**: Redis is using port 6379

**Solution**: Use port 6380 for Ray (as shown above)

## Stopping the Cluster

### Stop Worker Nodes
**On each worker node:**
```bash
ray stop
```

### Stop Head Node
**On head node:**
```bash
ray stop
```

## Configuration Options

### Disable Remote Coordinator
If you want to force local execution:

```python
router = RayHybridRouter(
    enable_remote_coordinator=False  # Force local execution
)
```

### Custom Placement Strategy
```python
# In ray_hybrid_router.py, change line 435:
pool = ShardedModelPool.options(
    scheduling_strategy="SPREAD",  # Current: spread across nodes
    # scheduling_strategy="DEFAULT",  # Ray default scheduler
    # num_cpus=4,  # Require 4 CPUs
    # memory=30_000_000_000,  # Require 30GB RAM
).remote(...)
```

## Performance Notes

- **Network overhead**: ~50-100ms for result streaming (negligible for long inference)
- **Coordination overhead**: <10ms for actor placement
- **Memory savings**: Prevents OOM on low-RAM nodes
- **Throughput**: Same as RPC (not affected by Ray)

## Next Steps

1. ✅ Ray cluster running
2. Test with actual model request
3. Monitor coordinator placement in logs
4. Benchmark latency vs local execution
