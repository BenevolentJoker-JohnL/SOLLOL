# Accurate GPU Monitoring Setup for SOLLOL

## Overview

SOLLOL can use **accurate GPU stats from your Ollama nodes** instead of making assumptions. Each node runs a lightweight GPU reporter that publishes real-time stats to Redis.

## Architecture

```
┌─────────────────┐          ┌──────────────┐          ┌─────────────────┐
│ Ollama Node     │          │              │          │ SOLLOL Client   │
│ (192.168.1.20)    │──publish─▶│    Redis     │◀─subscribe─│ (FlockParser)   │
│                 │          │              │          │                 │
│ gpu_reporter.py │          │ GPU Stats    │          │ GPURedisSubscr  │
│ + nvidia-smi    │          │ Stream       │          │                 │
└─────────────────┘          └──────────────┘          └─────────────────┘
```

## Features

✅ **Unified GPU Detection**: Uses `gpustat` for automatic vendor detection
✅ **Multi-Vendor Support**: NVIDIA, AMD, Intel
✅ **Ollama Integration**: Detects if GPU is actually being used (not just present)
✅ **Automatic CPU-Only Detection**: Handles nodes with GPU hardware but CPU-only Ollama

## Supported GPUs

- **NVIDIA**: Tesla, RTX, GTX series (automatic via gpustat or nvidia-smi)
- **AMD**: Radeon (automatic via gpustat or rocm-smi)
- **Intel**: Arc, Integrated Graphics (automatic via gpustat or xpu-smi)

## Setup

### Step 1: Install Dependencies on Each Node

```bash
# On each Ollama node
pip3 install gpustat redis requests

# gpustat automatically works with NVIDIA, AMD, and Intel GPUs!
```

**Why gpustat?**
- Single command works across NVIDIA, AMD, Intel
- Automatic vendor detection
- Cleaner API than parsing nvidia-smi output
- Falls back to vendor tools if gpustat fails

### Step 2: Copy GPU Reporter to Each Node

Copy `gpu_reporter.py` to each Ollama node:

```bash
# On each Ollama node (192.168.1.20, 192.168.1.21, etc.)
scp gpu_reporter.py user@192.168.1.20:/opt/sollol/

# Or if SSH isn't available, use a shared drive/USB
```

### Step 2: Install Dependencies

```bash
# On each node
pip3 install redis
```

### Step 3: Start GPU Reporter on Each Node

```bash
# On 192.168.1.20 (GPU node)
python3 /opt/sollol/gpu_reporter.py \
  --redis-host 192.168.1.10 \
  --node-id 192.168.1.20:11434 \
  --interval 5 \
  &

# On 192.168.1.21 (CPU node)
python3 /opt/sollol/gpu_reporter.py \
  --redis-host 192.168.1.10 \
  --node-id 192.168.1.21:11434 \
  --interval 5 \
  &

# On 192.168.1.10 (local node with Redis)
python3 /opt/sollol/gpu_reporter.py \
  --redis-host localhost \
  --node-id 192.168.1.10:11434 \
  --interval 5 \
  &
```

**Create systemd service (recommended):**

```bash
sudo tee /etc/systemd/system/sollol-gpu-reporter.service <<EOF
[Unit]
Description=SOLLOL GPU Stats Reporter
After=network.target redis.service

[Service]
Type=simple
User=ollama
ExecStart=/usr/bin/python3 /opt/sollol/gpu_reporter.py \\
  --redis-host 192.168.1.10 \\
  --node-id $(hostname -I | awk '{print $1}'):11434 \\
  --interval 5
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable sollol-gpu-reporter
sudo systemctl start sollol-gpu-reporter
```

### Step 4: Enable GPU Redis in SOLLOL Client

```python
from sollol import OllamaPool

pool = OllamaPool(
    nodes=None,  # Auto-discover
    enable_intelligent_routing=True,
    enable_gpu_redis=True,  # ← Enable accurate GPU stats
    redis_host="192.168.1.10",  # Redis server
    redis_port=6379
)
```

**For FlockParser:**

```python
# In flockparsecli.py
load_balancer = OllamaPool(
    nodes=None,
    enable_intelligent_routing=True,
    exclude_localhost=True,
    discover_all_nodes=True,
    app_name="FlockParser",
    enable_ray=True,
    register_with_dashboard=False,
    enable_gpu_redis=True,  # ← Add this
    redis_host="192.168.1.10",  # ← Add this
)
```

## Verify It's Working

### Check GPU Reporter Output

```bash
# On a GPU node
journalctl -u sollol-gpu-reporter -f
```

You should see:
```
INFO GPU 0: NVIDIA GeForce RTX 4090 | VRAM: 8192/24576MB | Util: 45% | Temp: 62°C
```

### Check Redis

```bash
# On Redis server
redis-cli KEYS "sollol:gpu:*"
# Output: sollol:gpu:192.168.1.20:11434

redis-cli GET "sollol:gpu:192.168.1.20:11434"
# Output: {"vendor":"nvidia","gpus":[{...}],"timestamp":1696845234.5}
```

### Check SOLLOL Logs

```bash
# In FlockParser or your SOLLOL client
# You should see:
INFO ✅ GPU subscriber connected to Redis at 192.168.1.10:6379
INFO Updated 192.168.1.20:11434: NVIDIA GeForce RTX 4090 | VRAM: 16384MB free / 24576MB total | Util: 45%
```

## Intelligent Ollama GPU Detection

The reporter **automatically detects if Ollama is actually using the GPU**:

```python
# Checks Ollama /api/ps for size_vram > 0
if any(model["size_vram"] > 0 for model in models):
    # GPU is being used ✅
else:
    # GPU present but Ollama running CPU-only ❌
    # Report as CPU-only to prevent mis-routing
```

**Example:**
```
Node has RTX 4090 (24GB)
But Ollama is running: ollama serve --gpu=false

Reporter detects: ⚠️ GPU detected (RTX 4090) but Ollama is running in CPU-only mode
Reports to SOLLOL: CPU-only (0MB VRAM)
```

This prevents SOLLOL from routing GPU tasks to nodes with GPU hardware that Ollama isn't using!

## Benefits

### Before (Assumptions)
```
192.168.1.21: Assumes 8GB GPU (but actually CPU-only)
192.168.1.20: Assumes 1674MB total (actually 24GB!)
192.168.1.10: Assumes 2GB GPU (no GPU)
```

### After (Accurate)
```
192.168.1.21: CPU-only (0MB VRAM) ✅
192.168.1.20: RTX 4090 - 16384MB free / 24576MB total ✅
192.168.1.10: CPU-only (0MB VRAM) ✅
```

## Routing Impact

With accurate GPU stats:
- **Extraction tasks** route to .90 (GPU) instead of .48 (CPU)
- **Large models** won't try to load on nodes without sufficient VRAM
- **Load balancing** distributes based on actual GPU utilization

## Troubleshooting

### GPU Reporter Not Detecting GPU

```bash
# Test nvidia-smi manually
nvidia-smi

# If that works, check permissions
sudo usermod -aG video $USER
```

### Redis Connection Failed

```bash
# Test Redis connectivity
redis-cli -h 192.168.1.10 ping
# Should return: PONG

# Check firewall
sudo ufw allow 6379/tcp
```

### SOLLOL Not Receiving Stats

```bash
# Check Redis has data
redis-cli -h 192.168.1.10 KEYS "sollol:gpu:*"

# Check SOLLOL logs for subscriber errors
```

## Performance

- **Overhead**: ~0.1% CPU per node (5-second polling)
- **Network**: ~1KB/s per node to Redis
- **Latency**: Stats updated every 5 seconds

## Notes

- GPU reporter works on **any GPU** (NVIDIA, AMD, Intel)
- Falls back to CPU-only reporting if no GPU detected
- Stats expire after 60 seconds (handles node failures gracefully)
- Compatible with multi-GPU systems (reports all GPUs)
