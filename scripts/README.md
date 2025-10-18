# SOLLOL Installation Scripts

Automated setup scripts for SOLLOL RPC backends, GPU monitoring, and Ray cluster.

> **Note**: These scripts are designed for **bare-metal deployment**, which provides optimal performance and reduces abstraction layers. Docker deployments are not fully battle-tested.

## Quick Start

### Complete Setup (Recommended)

Install everything (RPC server + GPU monitoring):

```bash
./scripts/install-all-services.sh
```

This will:
1. Build llama.cpp with RPC support (build 6743+ recommended)
2. Install RPC server as systemd service
3. Install GPU monitoring service (if Redis available)
4. Configure for bare-metal deployment

### Individual Components

#### RPC Server Only

```bash
# Build and install RPC server
python3 src/sollol/setup_llama_cpp.py --all

# Or use the script directly
./scripts/install-rpc-service.sh
```

#### GPU Monitoring Only

```bash
# Requires: Redis, gpustat, sollol
./scripts/install-gpu-reporter-service.sh
```

## Scripts Overview

### `install-all-services.sh`

Complete setup script that installs both RPC server and GPU monitoring.

**Features:**
- Checks prerequisites (Python, cmake, Redis)
- Clones and builds llama.cpp with latest stable version
- Installs both systemd services
- Handles optional GPU monitoring gracefully

**Requirements:**
- Python 3.8+
- cmake and build-essential
- Redis (optional, for GPU monitoring)

### `install-rpc-service.sh`

Installs llama.cpp RPC server as a systemd user service.

**Service Configuration:**
- Host: `0.0.0.0` (accessible from network)
- Port: `50052` (default RPC port)
- Auto-restart on failure
- Runs on system boot (with loginctl enable-linger)

**Service Management:**
```bash
systemctl --user status sollol-rpc-server
systemctl --user restart sollol-rpc-server
systemctl --user stop sollol-rpc-server
journalctl --user -u sollol-rpc-server -f
```

### `install-gpu-reporter-service.sh`

Installs GPU monitoring reporter as a systemd user service.

**Service Configuration:**
- Reports GPU stats to Redis every 5 seconds
- Detects NVIDIA, AMD, and Intel GPUs
- Verifies Ollama GPU usage
- Publishes to `gpu:stats` Redis stream

**Environment Variables:**
- `REDIS_HOST`: Redis server (default: localhost)
- `REDIS_PORT`: Redis port (default: 6379)
- `OLLAMA_PORT`: Ollama API port (default: 11434)
- `REPORT_INTERVAL`: Report interval in seconds (default: 5)

**Service Management:**
```bash
systemctl --user status sollol-gpu-reporter
systemctl --user restart sollol-gpu-reporter
systemctl --user stop sollol-gpu-reporter
journalctl --user -u sollol-gpu-reporter -f
```

## Advanced Usage

### Python Setup Script

The `setup_llama_cpp.py` script offers more options:

```bash
# Full automated setup
python3 src/sollol/setup_llama_cpp.py --all

# Full setup with GPU monitoring
python3 src/sollol/setup_llama_cpp.py --all --gpu-monitoring

# Custom build location
python3 src/sollol/setup_llama_cpp.py --all --install-dir ~/custom/llama.cpp

# Interactive start (no systemd)
python3 src/sollol/setup_llama_cpp.py --start --port 50052

# Update existing installation
cd ~/llama.cpp && git pull
python3 src/sollol/setup_llama_cpp.py --build
```

### Multi-Node Setup

#### GPU Node (with monitoring)

```bash
# On GPU node
./scripts/install-all-services.sh
```

This installs:
- RPC server on port 50052
- GPU monitoring service

#### CPU-Only Node

```bash
# On CPU node
python3 src/sollol/setup_llama_cpp.py --all
```

This installs:
- RPC server on port 50052
- No GPU monitoring

### Verifying Installation

```bash
# Check RPC server
timeout 1 bash -c "cat < /dev/null > /dev/tcp/localhost/50052" && echo "✅ RPC running" || echo "❌ Down"

# Check GPU reporter (if installed)
redis-cli XREAD COUNT 1 STREAMS gpu:stats 0

# Test with SOLLOL
sollol discover
```

## Systemd Services

### Service Files Location

- User services: `~/.config/systemd/user/`
- Source files: `systemd/`

### Enable/Disable Services

```bash
# Enable (start on boot)
systemctl --user enable sollol-rpc-server
systemctl --user enable sollol-gpu-reporter

# Disable
systemctl --user disable sollol-rpc-server
systemctl --user disable sollol-gpu-reporter

# Enable lingering (keep services running when logged out)
loginctl enable-linger $USER
```

### Customizing Services

Edit the service file:
```bash
nano ~/.config/systemd/user/sollol-rpc-server.service
```

Reload after changes:
```bash
systemctl --user daemon-reload
systemctl --user restart sollol-rpc-server
```

## Troubleshooting

### RPC Server Not Starting

**Check logs:**
```bash
journalctl --user -u sollol-rpc-server -n 50
```

**Common issues:**
- Port already in use: Change port in service file
- Binary not found: Check `~/.local/bin/rpc-server` exists
- Permission denied: `chmod +x ~/.local/bin/rpc-server`

### GPU Monitoring Not Working

**Check logs:**
```bash
journalctl --user -u sollol-gpu-reporter -n 50
```

**Common issues:**
- Redis not running: `sudo systemctl start redis-server`
- gpustat not installed: `pip install gpustat`
- No GPU detected: Check `gpustat` output
- Ollama not using GPU: Check `/api/ps` shows size_vram > 0

### Build Failures

**CMake configuration failed:**
```bash
sudo apt-get install cmake build-essential
```

**Build 6689 crashes:**
Update to build 6743+:
```bash
cd ~/llama.cpp
git pull origin master
python3 ~/SOLLOL/src/sollol/setup_llama_cpp.py --build
```

## Build Version Requirements

**Recommended:** llama.cpp build 6743 or later

**Why?**
- Build 6689 has crashes during response handling
- Build 6743+ includes 54 commits with stability fixes
- Critical for distributed inference with 70B+ models

**Check your version:**
```bash
cd ~/llama.cpp
git log --oneline -1
```

**Update:**
```bash
cd ~/llama.cpp
git pull
python3 ~/SOLLOL/src/sollol/setup_llama_cpp.py --build
systemctl --user restart sollol-rpc-server
```

## Ray Cluster Setup (for Remote Coordinator Execution)

For intelligent coordinator placement on multi-node setups:

### Head Node Setup

```bash
# Start Ray head node
ray start --head \
  --port=6380 \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --num-cpus=2 \
  --object-store-memory=500000000

# Verify
ray status
```

### Worker Node Setup

```bash
# Join Ray cluster (on each worker node)
ray start --address='HEAD_NODE_IP:6380'

# Example:
ray start --address='10.9.66.154:6380'
```

### Ray Systemd Service

For automatic startup, see the systemd service templates in [INSTALLATION.md](../INSTALLATION.md#production-deployment).

## Documentation

- [INSTALLATION.md](../INSTALLATION.md) - Complete installation guide (bare-metal focus)
- [CONFIGURATION.md](../CONFIGURATION.md) - Configuration options
- [RAY_CLUSTER_SETUP.md](../RAY_CLUSTER_SETUP.md) - Ray cluster setup for remote coordinators
- [REMOTE_COORDINATOR_DESIGN.md](../REMOTE_COORDINATOR_DESIGN.md) - Remote coordinator architecture
- [DISTRIBUTED_INFERENCE_STATUS.md](../DISTRIBUTED_INFERENCE_STATUS.md) - Testing results
- [GPU_MONITORING_SETUP.md](../GPU_MONITORING_SETUP.md) - GPU monitoring architecture
- [REDIS_SETUP.md](../REDIS_SETUP.md) - Redis configuration

## Support

Report issues: https://github.com/BenevolentJoker-JohnL/SOLLOL/issues
