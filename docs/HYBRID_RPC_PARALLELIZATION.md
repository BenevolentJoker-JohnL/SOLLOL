# Hybrid GPU+CPU RPC Parallelization
**llama.cpp RPC backend only**

> **Important:** This feature requires llama.cpp's `rpc-server` and is NOT available with Ollama. Ollama does not support multi-device parallelization via `--device cpu,cuda:0` flags.

## The Problem

Traditional RPC setups treat each node as a single worker:
- CPU nodes: 1 worker (RAM only)
- GPU nodes: 1 worker (VRAM only)

This wastes resources! A GPU node has BOTH VRAM (for GPU) AND RAM (for CPU).

## The Solution: Hybrid Parallelization

Configure GPU nodes to contribute **multiple workers**:
- 1 CPU worker (using RAM)
- 1+ GPU workers (using VRAM)

All devices work **in parallel** on the same physical machine!

## Example: 3 Physical Nodes → 4 Parallel Workers

### Traditional Setup (3 workers):
```
CPU Node 1  → 1 worker (8GB RAM)
CPU Node 2  → 1 worker (8GB RAM)
GPU Node    → 1 worker (12GB VRAM)
Total: 3 parallel workers
```

### Hybrid Setup (4 workers):
```
CPU Node 1  → 1 worker (8GB RAM)
CPU Node 2  → 1 worker (8GB RAM)
GPU Node    → 2 workers:
              ├─ CPU device (10GB RAM)
              └─ GPU device (9.6GB VRAM)
Total: 4 parallel workers (+33% throughput!)
```

## How It Works

llama.cpp's `rpc-server` supports multiple devices via `--device`:

```bash
# CPU-only node
rpc-server --host 0.0.0.0 --port 50052 --device cpu --mem 8000

# GPU node with HYBRID parallelization
rpc-server --host 0.0.0.0 --port 50052 --device cpu,cuda:0 --mem 10000,9600
                                                  ^^^  ^^^^^^       ^^^^^  ^^^^
                                                  CPU   GPU         CPU    GPU
                                                        device      RAM    VRAM
```

When the coordinator distributes model layers:
1. Some layers go to CPU workers (uses RAM)
2. Other layers go to GPU workers (uses VRAM)
3. All workers process **in parallel**

## Layer Distribution Example (40 layers)

With 2 CPU nodes + 1 hybrid GPU node:

```
CPU Node 1:      Layers 0-9   (10 layers, 8GB RAM)
CPU Node 2:      Layers 10-19 (10 layers, 8GB RAM)
GPU Node CPU:    Layers 20-29 (10 layers, 10GB RAM)
GPU Node GPU:    Layers 30-39 (10 layers, 9.6GB VRAM) ⚡
```

All 4 workers compute simultaneously!

## Automatic Configuration

**SOLLOL automatically configures hybrid GPU+CPU parallelization!**

When you start SOLLOL with auto-discovery, it:
1. Detects all local resources (GPUs, CPUs, RAM, VRAM)
2. Calculates safe allocations (80% with 20% reserve)
3. Starts RPC servers with optimal hybrid device configs

**No manual configuration needed!**

### Manual Configuration (Optional)

If you want to see what SOLLOL would configure, run the detection script:

```bash
python scripts/setup_rpc_node.py
```

Output:
```
======================================================================
RPC NODE SETUP - Hybrid GPU+CPU Parallelization
======================================================================

🔍 Detecting local resources...

======================================================================
DETECTED RESOURCES
======================================================================
✅ GPU(s) Found: 1
   GPU 0: cuda:0 - 9600 MB VRAM (safe allocation)

💾 CPU RAM: 10240 MB (safe allocation)

⚡ Total Parallel Workers: 2
   (1 CPU worker + 1 GPU worker(s))

======================================================================
GENERATED RPC-SERVER COMMAND
======================================================================
rpc-server --host 0.0.0.0 --port 50052 --device cpu,cuda:0 --mem 10240,9600

💡 This command creates HYBRID parallelization:
   • CPU device processes layers using 10240 MB RAM
   • cuda:0 processes layers using 9600 MB VRAM

   ALL 2 devices work IN PARALLEL on this single node!
```

## Safety Features

- **Auto 20% reserve**: Leaves headroom to prevent OOM crashes
- **Per-device limits**: Each worker has its own safe memory allocation
- **Vendor detection**: Supports NVIDIA (cuda), AMD (rocm), Intel
- **Fallback**: Gracefully falls back to CPU-only if no GPU detected

## Benefits

1. **More throughput**: Extra parallel workers without extra hardware
2. **Better utilization**: Use ALL resources (RAM + VRAM)
3. **No coordinator bottleneck**: Computation is distributed
4. **Safe allocations**: 80% limits prevent crashes
5. **Auto-detection**: No manual config needed

## Comparison

| Metric | Traditional | Hybrid | Improvement |
|--------|-------------|--------|-------------|
| Physical nodes | 3 | 3 | Same |
| Parallel workers | 3 | 4 | +33% |
| GPU utilization | 100% | 100% | Same |
| CPU utilization | 100% | 100% | Same |
| RAM waste | High | None | Maximized |

With hybrid parallelization, you get more workers using the same hardware!

## Future: Multi-GPU Nodes

For nodes with 2+ GPUs:

```bash
# 4 workers on 1 machine!
rpc-server --device cpu,cuda:0,cuda:1,cuda:2 --mem 10000,9600,9600,9600
```

Physical setup:
- 2 CPU nodes: 2 workers
- 1 quad-GPU node: 4 workers (1 CPU + 3 GPUs)

**Total: 6 parallel workers across 3 physical machines!**
