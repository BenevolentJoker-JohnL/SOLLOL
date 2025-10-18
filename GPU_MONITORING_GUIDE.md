# GPU Monitoring Guide for SOLLOL

SOLLOL uses **real-time VRAM monitoring** via `gpustat` and Redis to make intelligent routing decisions. This guide explains how GPU monitoring works and how to set it up.

## Why GPU Monitoring Matters

SOLLOL routes requests based on actual VRAM availability to:
- **Prevent OOM errors** - Don't send requests to nodes with insufficient VRAM
- **Optimize performance** - Route large models to nodes with more VRAM
- **Balance load** - Distribute requests based on actual GPU capacity

Without real monitoring, SOLLOL falls back to estimates which can be inaccurate.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SOLLOL Orchestrator                      │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │   OllamaPool (Intelligent Routing)                 │    │
│  │   - Subscribes to Redis GPU stats                  │    │
│  │   - Updates node_performance with real VRAM data   │    │
│  │   - Routes based on gpu_free_mem                   │    │
│  └────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│                    [Redis Server]                            │
│                           │                                  │
└───────────────────────────┼──────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
┌───────────────────┐                  ┌───────────────────┐
│   Node 1          │                  │   Node 2          │
│                   │                  │                   │
│ gpu_reporter.py   │                  │ gpu_reporter.py   │
│  - gpustat        │                  │  - gpustat        │
│  - Publishes to   │                  │  - Publishes to   │
│    Redis every 5s │                  │    Redis every 5s │
│                   │                  │                   │
│ [GPU: 8GB total]  │                  │ [GPU: 24GB total] │
│ [Free: 2.5GB]     │                  │ [Free: 18GB]      │
└───────────────────┘                  └───────────────────┘
```

## How It Works

### 1. GPU Reporter (per node)

Each Ollama node runs `gpu_reporter.py` as a systemd service:

- **Monitors GPU** using `gpustat` (vendor-agnostic: NVIDIA, AMD, Intel)
- **Publishes to Redis** every 5 seconds:
  - Total VRAM
  - Used VRAM
  - Free VRAM
  - GPU utilization %
  - Temperature
- **Checks Ollama usage** - only reports GPU if Ollama is actually using it

### 2. SOLLOL Pool (orchestrator)

The `OllamaPool` subscribes to Redis and:

- **Reads GPU stats** for all registered nodes
- **Updates `node_performance`** with real VRAM data:
  ```python
  node_perf["gpu_free_mem"] = 18432  # Real data from gpustat
  node_perf["gpu_total_mem"] = 24576
  node_perf["gpu_utilization"] = 45
  ```
- **Routes intelligently** based on actual capacity:
  - Nodes with <2GB free → marked as overwhelmed
  - Nodes with >4GB free → 1.5× score boost
  - Load balancing considers VRAM capacity

### 3. Fallback Mode

If Redis monitoring isn't available, SOLLOL falls back to:
- `nvidia-smi` / `rocm-smi` subprocess calls (localhost only)
- `/api/ps` endpoint queries (remote nodes)
- Model-based estimates if above fail

**Note:** Fallbacks are less accurate and may cause routing issues.

## Setup

### Prerequisites

- Redis server (can be on any node)
- `gpustat` Python package
- `redis-py` Python package

Both are now **automatically installed** with SOLLOL v0.9.49+:
```bash
pip install sollol  # Includes gpustat and redis
```

### Quick Setup (Recommended)

Use the built-in CLI command:

```bash
# On each Ollama node, run:
sollol install-gpu-reporter --redis-host <redis-server-ip>

# Example:
sollol install-gpu-reporter --redis-host 192.168.1.10
```

This will:
1. Auto-detect your node ID (IP:11434)
2. Install gpustat and redis-py if needed
3. Create systemd user service
4. Start monitoring immediately

### Manual Setup

If you need more control:

```bash
# 1. Install dependencies
pip install gpustat redis

# 2. Run installer script
cd /path/to/SOLLOL
bash scripts/install-gpu-reporter-service.sh

# Follow the prompts to configure:
# - Redis host
# - Redis port
# - Node ID
# - Report interval
```

### Verify Setup

Check if GPU reporter is running:

```bash
# Check service status
systemctl --user status sollol-gpu-reporter

# View live logs
journalctl --user -u sollol-gpu-reporter -f

# Expected output:
# GPU 0: NVIDIA GeForce RTX 3070 | VRAM: 4123/8192MB | Util: 55% | Temp: 62°C
```

Check Redis has data:

```bash
redis-cli
> KEYS sollol:gpu:*
1) "sollol:gpu:192.168.1.20:11434"
2) "sollol:gpu:192.168.1.10:11434"

> GET sollol:gpu:192.168.1.20:11434
{"vendor":"nvidia","gpus":[{"index":0,"name":"NVIDIA GeForce RTX 3070","memory_total_mb":8192,"memory_free_mb":4123, ...}]}
```

### SOLLOL Configuration

GPU Redis monitoring is **enabled by default** in SOLLOL v0.9.49+.

To customize:

```python
from sollol import OllamaPool

# Default (Redis monitoring enabled)
pool = OllamaPool.auto_configure()

# Custom Redis location
pool = OllamaPool.auto_configure(
    enable_gpu_redis=True,
    redis_host="192.168.1.10",
    redis_port=6379
)

# Disable Redis monitoring (not recommended)
pool = OllamaPool.auto_configure(enable_gpu_redis=False)
```

## Verification

Check if SOLLOL is receiving GPU data:

```python
from sollol import OllamaPool

pool = OllamaPool.auto_configure()
stats = pool.get_stats()

print(stats['vram_monitoring'])
# {
#   "enabled": True,
#   "gpu_type": "nvidia",
#   "local_gpu": {...},
#   "refresh_interval_seconds": 30
# }

# Check node performance data
for node_id, perf in stats['pool']['node_performance'].items():
    print(f"{node_id}: {perf['gpu_free_mem']}MB free")
    # 192.168.1.20:11434: 4123MB free
    # 192.168.1.10:11434: 18432MB free
```

## Troubleshooting

### GPU Reporter Not Starting

**Check logs:**
```bash
journalctl --user -u sollol-gpu-reporter -n 50
```

**Common issues:**

1. **"No module named 'gpustat'"**
   ```bash
   pip install --user gpustat
   systemctl --user restart sollol-gpu-reporter
   ```

2. **"Failed to connect to Redis"**
   - Check Redis is running: `redis-cli ping`
   - Check Redis host in service config
   - Verify firewall allows port 6379

3. **"No GPU detected"**
   - For NVIDIA: Install `nvidia-smi`
   - For AMD: Install `rocm-smi`
   - For Intel: Install `xpu-smi` or `intel_gpu_top`
   - Check: `gpustat` (should show your GPU)

### SOLLOL Not Receiving Data

**Check if GPU subscriber connected:**
```python
pool = OllamaPool.auto_configure()
# Look for log line:
# ✅ GPU subscriber connected to Redis at localhost:6379
```

**If not connected:**
- Verify Redis is accessible: `redis-cli -h <redis-host> ping`
- Check `enable_gpu_redis=True` in pool config
- Check Redis credentials if auth is enabled

### Inaccurate VRAM Data

**Verify gpustat works:**
```bash
gpustat
# Should show accurate VRAM usage
```

**Check Ollama is using GPU:**
```bash
curl http://localhost:11434/api/ps
# Look for "size_vram" > 0 in models
```

**If Ollama shows size_vram=0:**
- Ollama is in CPU-only mode
- GPU reporter will correctly report no GPU available
- SOLLOL will route to this node as CPU-only

## Performance Impact

GPU monitoring has minimal overhead:
- **gpustat query:** ~5-10ms
- **Redis publish:** ~1-2ms
- **Total per node:** <50ms every 5 seconds
- **Network traffic:** ~500 bytes per node per interval

For a 10-node cluster:
- **Bandwidth:** ~1 KB/s
- **Redis memory:** ~10KB (with 1000-entry stream limit)

## Vendor Support

### NVIDIA
- **Tool:** `gpustat` → `nvidia-smi`
- **Support:** All modern GPUs (Kepler 2012+)
- **Metrics:** Total/Free/Used VRAM, Utilization, Temperature, Power

### AMD
- **Tool:** `gpustat` → `rocm-smi`
- **Support:** All ROCm-compatible GPUs
- **Metrics:** Total/Free/Used VRAM, Utilization, Temperature

### Intel
- **Tool:** `gpustat` → `xpu-smi` (Arc) or `intel_gpu_top`
- **Support:** Arc GPUs and integrated graphics
- **Metrics:** Limited (depends on driver version)

## Best Practices

1. **Run Redis on a stable node** - Don't run on a node that might go down frequently

2. **Use consistent Redis host** - All nodes should report to same Redis instance

3. **Monitor Redis health** - Set up alerts for Redis downtime

4. **Set reasonable intervals** - 5 seconds is a good balance (faster = more overhead, slower = stale data)

5. **Enable lingering** - Ensures GPU reporter runs even when not logged in:
   ```bash
   loginctl enable-linger $USER
   ```

6. **Check logs regularly** - Watch for GPU reporter errors:
   ```bash
   journalctl --user -u sollol-gpu-reporter -f
   ```

## Advanced: Multi-Redis Setup

For large deployments, you can use multiple Redis instances:

```python
# Data center 1
pool1 = OllamaPool(
    nodes=dc1_nodes,
    enable_gpu_redis=True,
    redis_host="redis1.internal"
)

# Data center 2
pool2 = OllamaPool(
    nodes=dc2_nodes,
    enable_gpu_redis=True,
    redis_host="redis2.internal"
)
```

## See Also

- [GPU Monitoring Setup (Quick Start)](/GPU_MONITORING_SETUP.md)
- [VRAM Monitor Implementation](/src/sollol/vram_monitor.py)
- [GPU Redis Subscriber](/src/sollol/gpu_redis_subscriber.py)
- [GPU Reporter Script](/gpu_reporter.py)
