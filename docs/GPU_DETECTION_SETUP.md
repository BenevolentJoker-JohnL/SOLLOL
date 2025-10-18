# GPU Detection and Reporting Setup Guide

This guide walks through configuring SOLLOL for automatic GPU detection across your distributed cluster.

## Overview

SOLLOL's GPU detection system enables intelligent routing by detecting GPU capabilities on remote RPC nodes. The system uses Redis as a central registry where nodes publish their GPU specifications.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Coordinator Node (192.168.1.10)                         │
│  ┌────────────────────────────────────────────────┐    │
│  │  Redis (0.0.0.0:6379)                          │    │
│  │  - Central GPU registry                        │    │
│  │  - Stores node capabilities                    │    │
│  └────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────┐    │
│  │  SOLLOL Discovery                              │    │
│  │  - Reads GPU info from Redis                   │    │
│  │  - Routes requests to GPU nodes                │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                       ▲
                       │ Publish GPU specs via Redis
                       │
         ┌─────────────┼─────────────┬─────────────┐
         │             │             │             │
    ┌────▼────┐   ┌───▼─────┐  ┌───▼─────┐  ┌───▼─────┐
    │GPU Node │   │GPU Node │  │CPU Node │  │GPU Node │
    │.90:50052│   │.45:50052│  │.48:50052│  │.X:50052 │
    │RTX 3090 │   │RTX 3080 │  │CPU-only │  │  ...    │
    │24GB VRAM│   │10GB VRAM│  │16GB RAM │  │         │
    └─────────┘   └─────────┘  └─────────┘  └─────────┘
         │             │             │             │
         └─────────────┴─────────────┴─────────────┘
                Run register_gpu_node.py on startup
```

## Prerequisites

- SOLLOL installed on coordinator node
- Redis installed and running on coordinator
- Python 3.8+ on all nodes
- Network connectivity between nodes (ports 6379, 50052)
- NVIDIA drivers on GPU nodes (for CUDA detection)

### IMPORTANT: CUDA-Specific RPC Backend Binaries

**You MAY need to compile CUDA-specific `rpc-server` binaries for GPU nodes.**

The `rpc-server` binary must be built with CUDA support to utilize GPUs. There are two deployment strategies:

#### Strategy 1: CUDA Binary on GPU Nodes (Recommended)

Build `rpc-server` with CUDA enabled and deploy ONLY to GPU nodes:

```bash
# On build machine (with CUDA toolkit installed)
cd ~/llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DGGML_RPC=ON

cmake --build build --config Release --target rpc-server -j $(nproc)

# Deploy to GPU nodes
scp build/bin/rpc-server 192.168.1.20:~/.local/bin/
```

**Requirements**:
- CUDA Toolkit 12.6+ installed on build machine
- NVIDIA drivers (version 535+) on GPU nodes at runtime
- Binary size: ~689MB (includes CUDA libraries)
- Will NOT run on CPU-only coordinator (missing `libcuda.so.1`)

#### Strategy 2: Separate CPU and CUDA Binaries

Build separate binaries for CPU-only coordinator and GPU nodes:

```bash
# CPU-only binary for coordinator (lightweight ~200MB)
cmake -B build-cpu \
  -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DGGML_RPC=ON

cmake --build build-cpu --target rpc-server -j $(nproc)

# CUDA binary for GPU nodes (heavy ~689MB)
cmake -B build-cuda \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_RPC=ON

cmake --build build-cuda --target rpc-server -j $(nproc)
```

**When to use**:
- Coordinator needs RPC backend locally (e.g., for testing)
- Avoiding CUDA dependency errors on CPU-only machines

#### Troubleshooting CUDA Binary Issues

**Error**: `libcuda.so.1: cannot open shared object file`

```bash
# This means CUDA binary is trying to run without NVIDIA drivers
# Solution: Use CPU binary on coordinator, CUDA binary on GPU nodes only
```

**Error**: `Unsupported GPU architecture`

```bash
# Your GPU compute capability isn't in CUDA_ARCHITECTURES list
# Solution: Check your GPU's compute capability and add to build:
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Add to CMAKE_CUDA_ARCHITECTURES (e.g., "75" for Turing, "89" for Ada)
```

**Automated Build Script**: Use `scripts/install_cuda_llama.sh` for guided setup.

---

## Part 1: Configure Redis for Network Access

By default, Redis only listens on `localhost` (127.0.0.1). Remote GPU nodes need network access to register their capabilities.

### Step 1.1: Edit Redis Configuration

On the **coordinator node** (192.168.1.10):

```bash
# Open Redis config
sudo nano /etc/redis/redis.conf

# Find the line:
bind 127.0.0.1 ::1

# Change it to (replace with your coordinator IP):
bind 127.0.0.1 ::1 192.168.1.10
```

**What this does**: Allows Redis to accept connections from the network while still listening on localhost.

### Step 1.2: Restart Redis

```bash
sudo systemctl restart redis
```

### Step 1.3: Verify Network Listening

```bash
# Check Redis is listening on network interface
netstat -tuln | grep 6379

# Expected output:
# tcp  0  0 127.0.0.1:6379      0.0.0.0:*  LISTEN  (localhost)
# tcp  0  0 192.168.1.10:6379    0.0.0.0:*  LISTEN  (network)
```

### Step 1.4: Test Remote Connection

From a **remote node** (e.g., 192.168.1.20):

```bash
# Test Redis connectivity
redis-cli -h 192.168.1.10 ping

# Expected output:
# PONG
```

If you get "Connection refused", check:
- Firewall rules (see Security section below)
- Redis bind configuration
- Network connectivity (`ping 192.168.1.10`)

---

## Part 2: Register GPU Nodes

Each GPU node needs to publish its capabilities to Redis on startup.

### Step 2.1: Copy Registration Script to GPU Nodes

From the **coordinator**:

```bash
# Copy registration script to each GPU node
scp /home/joker/SOLLOL/scripts/register_gpu_node.py 192.168.1.20:~/
scp /home/joker/SOLLOL/scripts/register_gpu_node.py 192.168.1.22:~/
# ... repeat for all GPU nodes
```

### Step 2.2: Install Dependencies on GPU Nodes

On **each GPU node**:

```bash
# Install Python Redis client
pip install redis

# Verify nvidia-smi is available (for GPU detection)
nvidia-smi
```

### Step 2.3: Run Registration Script

On **each GPU node** (e.g., 192.168.1.20):

```bash
# Register GPU with coordinator
python3 register_gpu_node.py --redis-host 192.168.1.10

# Expected output:
# ======================================================================
# GPU NODE REGISTRATION - SOLLOL
# ======================================================================
#
# 📍 Node IP: 192.168.1.20
#
# 🔍 Detecting resources...
#
# ======================================================================
# DETECTED RESOURCES
# ======================================================================
# ✅ GPU(s) Found: 1
#    GPU 0: NVIDIA GeForce RTX 3090 (cuda:0) - 19200 MB VRAM
#
# 💾 CPU RAM: 12000 MB
# ⚡ Parallel Workers: 2
#
# ======================================================================
# RPC-SERVER COMMAND
# ======================================================================
# rpc-server --host 0.0.0.0 --port 50052 --device cpu,cuda:0 --mem 12000,19200
#
# ======================================================================
# REDIS REGISTRATION
# ======================================================================
# ✅ Published to Redis: redis://192.168.1.10:6379
#    Key: sollol:rpc:node:192.168.1.20:50052
#    TTL: 1 hour
#
# ======================================================================
# ✅ REGISTRATION COMPLETE
# ======================================================================
```

### Step 2.4: Start RPC Server with Detected Configuration

Use the command shown by the registration script:

```bash
# Example from script output:
nohup rpc-server --host 0.0.0.0 --port 50052 --device cpu,cuda:0 --mem 12000,19200 > /tmp/rpc-server.log 2>&1 &
```

**What this does**:
- Starts RPC server with hybrid CPU + GPU workers
- CPU worker: 12GB RAM
- GPU worker: 19.2GB VRAM (80% of total, 20% reserved)
- 2 parallel workers on this node

---

## Part 3: Verify GPU Detection on Coordinator

On the **coordinator node** (192.168.1.10):

### Step 3.1: Check Redis Registration

```bash
# List all registered nodes
redis-cli KEYS "sollol:rpc:node:*"

# Expected output:
# 1) "sollol:rpc:node:192.168.1.20:50052"
# 2) "sollol:rpc:node:192.168.1.22:50052"
# 3) "sollol:rpc:node:192.168.1.21:50052"

# View specific node info
redis-cli GET "sollol:rpc:node:192.168.1.20:50052"

# Expected output (JSON):
# {"has_gpu":true,"gpu_devices":["cuda:0"],"gpu_vram_mb":[19200],"gpu_names":["NVIDIA GeForce RTX 3090"],"cpu_ram_mb":12000,"device_config":"cpu,cuda:0","memory_config":"12000,19200","total_parallel_workers":2}
```

### Step 3.2: Test SOLLOL Discovery

```bash
cd /home/joker/SOLLOL

PYTHONPATH=src python3 -c "
from sollol.rpc_discovery import auto_discover_rpc_backends, detect_node_resources
import json

print('=== RPC Node Discovery ===')
backends = auto_discover_rpc_backends()
print(f'Found {len(backends)} RPC backends:')

for backend in backends:
    host = backend['host']
    port = backend.get('port', 50052)
    print(f'\n📍 {host}:{port}')

    resources = detect_node_resources(host)
    print(f'   Has GPU: {resources[\"has_gpu\"]}')
    print(f'   GPU devices: {resources.get(\"gpu_devices\", [])}')
    print(f'   GPU VRAM: {resources.get(\"gpu_vram_mb\", [])} MB')
    print(f'   CPU RAM: {resources.get(\"cpu_ram_mb\", 0)} MB')
    print(f'   Workers: {resources[\"total_parallel_workers\"]}')
"
```

**Expected output**:

```
=== RPC Node Discovery ===
Found 3 RPC backends:

📍 192.168.1.20:50052
   Has GPU: True
   GPU devices: ['cuda:0']
   GPU VRAM: [19200] MB
   CPU RAM: 12000 MB
   Workers: 2

📍 192.168.1.22:50052
   Has GPU: True
   GPU devices: ['cuda:0']
   GPU VRAM: [10240] MB
   CPU RAM: 8000 MB
   Workers: 2

📍 192.168.1.21:50052
   Has GPU: False
   GPU devices: []
   GPU VRAM: [] MB
   CPU RAM: 16000 MB
   Workers: 1
```

---

## Part 4: Automate Registration on Startup

GPU registrations expire after 1 hour. Set up automatic re-registration.

### Option A: Systemd Service (Recommended)

Create `/etc/systemd/system/sollol-gpu-reporter.service` on **each GPU node**:

```ini
[Unit]
Description=SOLLOL GPU Registration Service
After=network.target redis.service

[Service]
Type=simple
User=your-username
WorkingDirectory=/home/your-username
ExecStart=/usr/bin/python3 /home/your-username/register_gpu_node.py --redis-host 192.168.1.10
Restart=always
RestartSec=3600

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable sollol-gpu-reporter
sudo systemctl start sollol-gpu-reporter
```

### Option B: Cron Job

Add to crontab on **each GPU node**:

```bash
crontab -e

# Add this line (runs every hour):
0 * * * * cd /home/your-username && python3 register_gpu_node.py --redis-host 192.168.1.10 > /tmp/gpu-registration.log 2>&1
```

---

## Security Considerations

### Firewall Configuration

On the **coordinator node**:

```bash
# Allow Redis from trusted subnet only
sudo ufw allow from 10.9.66.0/24 to any port 6379 comment "Redis from cluster nodes"

# Reload firewall
sudo ufw reload
```

### Redis Authentication (Optional but Recommended)

Add password protection to Redis:

```bash
# On coordinator, edit Redis config
sudo nano /etc/redis/redis.conf

# Add this line:
requirepass your_strong_password_here

# Restart Redis
sudo systemctl restart redis
```

Update registration script usage:

```bash
# On GPU nodes, set password
export REDIS_PASSWORD="your_strong_password_here"

# Or pass via URL
python3 register_gpu_node.py --redis-host "redis://:your_strong_password_here@192.168.1.10:6379"
```

---

## Troubleshooting

### Issue: Redis connection refused from remote nodes

**Symptoms**:
```
❌ Failed to publish to Redis: Error 111 connecting to 192.168.1.10:6379. Connection refused.
```

**Solutions**:

1. **Check Redis is listening on network**:
   ```bash
   netstat -tuln | grep 6379
   # Should show: 192.168.1.10:6379
   ```

2. **Verify bind configuration**:
   ```bash
   redis-cli CONFIG GET bind
   # Should include your coordinator IP
   ```

3. **Check firewall**:
   ```bash
   sudo ufw status
   # Should allow port 6379 from cluster subnet
   ```

4. **Test connectivity**:
   ```bash
   # From remote node
   telnet 192.168.1.10 6379
   # Should connect (press Ctrl+] then 'quit' to exit)
   ```

### Issue: GPU not detected on node

**Symptoms**:
```
ℹ️  No GPU detected (CPU-only node)
```

**Solutions**:

1. **Check NVIDIA drivers**:
   ```bash
   nvidia-smi
   # Should show GPU info
   ```

2. **Verify nvidia-smi in PATH**:
   ```bash
   which nvidia-smi
   # Should return: /usr/bin/nvidia-smi (or similar)
   ```

3. **Run registration with verbose output**:
   ```bash
   python3 -u register_gpu_node.py --redis-host 192.168.1.10
   ```

### Issue: Registration expires too quickly

**Symptoms**: GPU info disappears after 1 hour

**Solutions**:

1. **Set up systemd service** (see Part 4, Option A)
   - Automatically re-registers every hour

2. **Increase TTL** (modify `register_gpu_node.py`):
   ```python
   # Line 158: Change ex=3600 to ex=86400 (24 hours)
   r.set(key, json.dumps(resources), ex=86400)
   ```

### Issue: Coordinator shows "Has GPU: False" after registration

**Symptoms**:
```bash
redis-cli GET "sollol:rpc:node:192.168.1.20:50052"
# Returns valid JSON with has_gpu:true

# But SOLLOL discovery shows:
# Has GPU: False
```

**Solutions**:

1. **Check Redis connection in SOLLOL**:
   ```bash
   # Verify SOLLOL can reach Redis
   redis-cli -h localhost ping
   ```

2. **Verify key format**:
   ```bash
   # Keys must match pattern: sollol:rpc:node:<ip>:<port>
   redis-cli KEYS "sollol:rpc:node:*"
   ```

3. **Check SOLLOL discovery code**:
   ```python
   # In src/sollol/rpc_discovery.py
   # Ensure SOLLOL_REDIS_URL env var is set correctly
   export SOLLOL_REDIS_URL="redis://localhost:6379"
   ```

---

## Advanced Configuration

### Multi-GPU Nodes

For nodes with multiple GPUs:

```bash
# register_gpu_node.py automatically detects all GPUs
python3 register_gpu_node.py --redis-host 192.168.1.10

# Example output for 2-GPU node:
# ✅ GPU(s) Found: 2
#    GPU 0: NVIDIA RTX 3090 (cuda:0) - 19200 MB VRAM
#    GPU 1: NVIDIA RTX 3080 (cuda:1) - 10240 MB VRAM
#
# ⚡ Parallel Workers: 3 (1 CPU + 2 GPU)
#
# RPC-SERVER COMMAND:
# rpc-server --host 0.0.0.0 --port 50052 --device cpu,cuda:0,cuda:1 --mem 12000,19200,10240
```

### Custom Memory Allocation

Override automatic VRAM detection:

```bash
# Modify register_gpu_node.py before running:
# Line 102: Change 0.8 (80%) to your desired percentage
safe_vram = int(total_vram * 0.7)  # Use 70% instead of 80%
```

### AMD/Intel GPU Support

For non-NVIDIA GPUs, you'll need to modify the detection logic:

```python
# In register_gpu_node.py, add AMD GPU detection
def get_amd_gpus():
    """Detect AMD GPUs using rocm-smi"""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse rocm-smi output...
        return gpus
    except:
        return []
```

---

## Architecture Notes

### Why Redis?

1. **Centralized discovery**: Single source of truth for cluster capabilities
2. **Automatic expiration**: Stale nodes auto-removed (TTL-based)
3. **Fast lookups**: O(1) key-value retrieval
4. **Network-accessible**: Supports distributed clusters
5. **Atomic updates**: Race-condition free registration

### Why Not Direct GPU Queries?

Direct SSH or RPC-based GPU queries have drawbacks:

- ❌ Slower (network round-trip per query)
- ❌ Requires credentials (SSH keys, auth)
- ❌ Tight coupling (coordinator needs node access)
- ❌ No caching (repeated queries for same info)

Redis registration is:

- ✅ Fast (local Redis lookup)
- ✅ Decoupled (nodes self-register)
- ✅ Cached (1-hour TTL)
- ✅ Scalable (add nodes without config changes)

---

## Related Documentation

- [CUDA RPC Build Guide](../README.md#building-llamacpp-with-gpu-support) - Building CUDA-enabled binaries
- [Bare Metal Deployment](../README.md#production-deployment-bare-metal) - Production setup with systemd
- [Hybrid Parallelization](HYBRID_RPC_PARALLELIZATION.md) - CPU + GPU worker configuration

---

## Quick Reference

### Essential Commands

```bash
# Configure Redis for network access
sudo nano /etc/redis/redis.conf  # Add coordinator IP to bind line
sudo systemctl restart redis

# Register GPU node
python3 register_gpu_node.py --redis-host 192.168.1.10

# Verify registration
redis-cli KEYS "sollol:rpc:node:*"
redis-cli GET "sollol:rpc:node:192.168.1.20:50052"

# Test SOLLOL discovery
PYTHONPATH=src python3 -c "from sollol.rpc_discovery import auto_discover_rpc_backends; print(auto_discover_rpc_backends())"
```

### File Locations

- **Registration script**: `/home/joker/SOLLOL/scripts/register_gpu_node.py`
- **Redis config**: `/etc/redis/redis.conf`
- **Systemd service**: `/etc/systemd/system/sollol-gpu-reporter.service`
- **Discovery code**: `/home/joker/SOLLOL/src/sollol/rpc_discovery.py`

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/BenevolentJoker-JohnL/SOLLOL/issues
- Documentation: https://github.com/BenevolentJoker-JohnL/SOLLOL/tree/main/docs
