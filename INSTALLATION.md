# SOLLOL Installation Guide

**Complete installation and setup guide for SOLLOL - Super Ollama Load Balancer**

> **Note**: This guide focuses on **bare-metal deployment**, which provides optimal performance and reduces abstraction layers. Docker deployments are not fully battle-tested and may have performance implications.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start (Single Node)](#quick-start-single-node)
3. [Multi-Node Bare-Metal Setup](#multi-node-bare-metal-setup)
4. [Distributed Inference Setup](#distributed-inference-setup)
5. [Production Deployment](#production-deployment)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

**Minimum (Single Node):**
- Ubuntu 20.04+ / Debian 11+ / RHEL 8+
- Python 3.9+
- 8GB RAM
- 2 CPU cores

**Recommended (Production):**
- Ubuntu 22.04 LTS
- Python 3.11+
- 16GB+ RAM
- 4+ CPU cores
- SSD storage

**Multi-Node Cluster:**
- 3+ nodes on same network
- Network bandwidth: 1Gbps+
- Low latency (<1ms between nodes)

###Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev redis-server

# RHEL/CentOS
sudo yum install -y python3-pip python3-devel redis

# Verify Python version
python3 --version  # Should be 3.9+
```

---

## Quick Start (Single Node)

### 1. Install SOLLOL

```bash
# From PyPI (recommended)
pip install sollol

# OR from source
git clone https://github.com/B-A-M-N/SOLLOL.git
cd SOLLOL
pip install -e .
```

### 2. Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
sudo systemctl start ollama
sudo systemctl enable ollama

# Pull a model
ollama pull llama3.2
```

### 3. Start Redis

```bash
# Start Redis
sudo systemctl start redis
sudo systemctl enable redis

# Verify Redis is running
redis-cli ping  # Should return: PONG
```

### 4. Start SOLLOL

```bash
# Zero-config start (auto-discovers local Ollama)
sollol up

# Custom port
sollol up --port 8000

# With more workers
sollol up --ray-workers 16
```

### 5. Test

```bash
# Check health
curl http://localhost:11434/api/health | jq

# Chat request
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}'
```

**✅ You're done!** SOLLOL is running with intelligent routing.

---

## Multi-Node Bare-Metal Setup

This setup creates a distributed SOLLOL cluster across multiple nodes.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Multi-Node SOLLOL Cluster                                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Gateway Node (192.168.1.10)                                 │
│    - SOLLOL Gateway :11434                                   │
│    - Redis :6379 (shared state)                              │
│    - Ray Head :6380                                          │
│                                                              │
│  Ollama Nodes (192.168.1.21, 192.168.1.22)                      │
│    - Ollama Server :11434                                    │
│    - Small/medium model inference                            │
│                                                              │
│  GPU Node (192.168.1.20)                                       │
│    - Ollama Server :11434 (optional)                         │
│    - RPC Backend :50052 (for large models)                   │
│    - 128GB RAM + 24GB GPU                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Step 1: Setup Gateway Node

**On 192.168.1.10 (main node):**

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y python3-pip redis-server

# Install SOLLOL
pip install sollol

# Configure Redis for remote access
sudo tee -a /etc/redis/redis.conf <<EOF
bind 0.0.0.0
protected-mode no
EOF

# Restart Redis
sudo systemctl restart redis

# Install Ollama (optional, for small models)
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl start ollama
```

### Step 2: Setup Ollama Nodes

**On each Ollama node (192.168.1.21, 192.168.1.22):**

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Configure Ollama to listen on all interfaces
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf <<EOF
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
EOF

# Restart Ollama
sudo systemctl daemon-reload
sudo systemctl restart ollama

# Pull models
ollama pull llama3.2
ollama pull llama3.1:8b
```

### Step 3: Setup GPU Node (for large models)

**On GPU node (192.168.1.20):**

See [Distributed Inference Setup](#distributed-inference-setup) below for RPC backend installation.

### Step 4: Start Ray Cluster

**On gateway node (192.168.1.10):**

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

**On worker nodes (192.168.1.20, 192.168.1.21, 192.168.1.22):**

```bash
# Join Ray cluster
ray start --address='192.168.1.10:6380'

# Verify
ray status
```

### Step 5: Start SOLLOL Gateway

**On gateway node:**

```bash
# Set environment variables
export SOLLOL_PORT=11434
export SOLLOL_REDIS_URL="redis://192.168.1.10:6379"
export SOLLOL_RAY_WORKERS=16
export SOLLOL_DASHBOARD=true

# Start SOLLOL with auto-discovery
sollol up
```

SOLLOL will automatically discover:
- Ollama nodes on the network
- RPC backends registered in Redis
- Ray cluster nodes

---

## Distributed Inference Setup

For large models (70B+) that don't fit on a single node, use llama.cpp RPC sharding.

### Prerequisites

- GPU node with CUDA/ROCm
- llama.cpp compiled with CUDA/ROCm support
- At least 2 nodes for distribution

### Step 1: Build llama.cpp with RPC Support

**On GPU node (192.168.1.20):**

```bash
# Install dependencies
sudo apt-get install -y build-essential cmake git

# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with CUDA (NVIDIA)
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DGGML_RPC=ON
cmake --build . --config Release -j$(nproc)

# OR build with ROCm (AMD)
mkdir build && cd build
cmake .. -DGGML_ROCM=ON -DGGML_RPC=ON
cmake --build . --config Release -j$(nproc)

# OR build CPU-only
mkdir build && cd build
cmake .. -DGGML_CUDA=OFF -DGGML_BLAS=OFF -DGGML_METAL=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build . --config Release -j$(nproc)

# Verify binaries
./bin/rpc-server --help
./bin/llama-server --help
```

### Step 2: Register GPU Node Metadata in Redis

**On GPU node:**

```bash
# Download registration script
cd /path/to/SOLLOL
python3 src/sollol/register_rpc_gpu_node.py \
  --redis-url redis://192.168.1.10:6379 \
  --rpc-host 192.168.1.20 \
  --rpc-port 50052
```

This stores GPU/RAM metadata in Redis for intelligent coordinator placement.

### Step 3: Start RPC Backend

**On GPU node:**

```bash
# Start rpc-server
~/llama.cpp/build/bin/rpc-server \
  --host 0.0.0.0 \
  --port 50052 \
  --device cuda:0 \
  --mem 24000

# For multi-GPU:
~/llama.cpp/build/bin/rpc-server \
  --host 0.0.0.0 \
  --port 50052 \
  --device cpu,cuda:0,cuda:1 \
  --mem 128000,24000,24000
```

**On CPU nodes (for more distribution):**

```bash
~/llama.cpp/build/bin/rpc-server \
  --host 0.0.0.0 \
  --port 50052 \
  --device cpu \
  --mem 32000
```

### Step 4: Verify RPC Backends

**From gateway node:**

```bash
# Check Redis registration
redis-cli keys "sollol:rpc:node:*"
redis-cli get "sollol:rpc:node:192.168.1.20:50052"

# Test RPC connectivity
python3 -c "
from sollol.rpc_discovery import check_rpc_server
print(check_rpc_server('192.168.1.20', 50052))
"
```

### Step 5: Test Distributed Inference

```bash
# Request a large model
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.1:70b",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}'
```

**What happens:**
1. SOLLOL detects 70B model (large)
2. Queries RPC backends for resources
3. Selects best node (192.168.1.20 with 128GB RAM)
4. Spawns coordinator on that node via Ray
5. Coordinator distributes layers across RPC backends
6. Results stream back to client

---

## Production Deployment

### Using systemd Services

Create systemd service files for automatic startup and management.

#### 1. Redis Service

```bash
# Redis is usually installed with systemd service
sudo systemctl enable redis
sudo systemctl start redis
```

#### 2. Ray Cluster Service

**On head node (/etc/systemd/system/ray-head.service):**

```ini
[Unit]
Description=Ray Head Node
After=network.target

[Service]
Type=forking
User=sollol
WorkingDirectory=/home/sollol
ExecStart=/home/sollol/.local/bin/ray start --head --port=6380 --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-cpus=2 --object-store-memory=500000000
ExecStop=/home/sollol/.local/bin/ray stop
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**On worker nodes (/etc/systemd/system/ray-worker.service):**

```ini
[Unit]
Description=Ray Worker Node
After=network.target

[Service]
Type=forking
User=sollol
WorkingDirectory=/home/sollol
ExecStart=/home/sollol/.local/bin/ray start --address='192.168.1.10:6380'
ExecStop=/home/sollol/.local/bin/ray stop
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 3. RPC Backend Service

**/etc/systemd/system/rpc-backend.service:**

```ini
[Unit]
Description=llama.cpp RPC Backend
After=network.target

[Service]
Type=simple
User=sollol
WorkingDirectory=/home/sollol
ExecStart=/home/sollol/llama.cpp/build/bin/rpc-server --host 0.0.0.0 --port 50052 --device cuda:0 --mem 24000
Restart=on-failure
RestartSec=10
Environment="CUDA_VISIBLE_DEVICES=0"

[Install]
WantedBy=multi-user.target
```

#### 4. SOLLOL Gateway Service

**/etc/systemd/system/sollol.service:**

```ini
[Unit]
Description=SOLLOL Gateway
After=network.target redis.service ray-head.service
Requires=redis.service

[Service]
Type=simple
User=sollol
WorkingDirectory=/home/sollol
Environment="SOLLOL_PORT=11434"
Environment="SOLLOL_REDIS_URL=redis://192.168.1.10:6379"
Environment="SOLLOL_RAY_WORKERS=16"
Environment="SOLLOL_DASHBOARD=true"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/home/sollol/.local/bin/sollol up
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

#### Enable and Start Services

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable services (auto-start on boot)
sudo systemctl enable redis
sudo systemctl enable ray-head  # On head node only
sudo systemctl enable ray-worker  # On worker nodes only
sudo systemctl enable rpc-backend  # On GPU nodes
sudo systemctl enable sollol  # On gateway node

# Start services
sudo systemctl start redis
sudo systemctl start ray-head
sudo systemctl start rpc-backend
sudo systemctl start sollol

# Check status
sudo systemctl status sollol
journalctl -u sollol -f  # Follow logs
```

### Environment Configuration File

Create `/etc/sollol/config.env`:

```bash
# SOLLOL Configuration
SOLLOL_PORT=11434
SOLLOL_REDIS_URL=redis://192.168.1.10:6379
SOLLOL_RAY_WORKERS=16
SOLLOL_DASHBOARD=true
SOLLOL_DASHBOARD_PORT=8080

# Optional: Manual node specification
#OLLAMA_NODES=192.168.1.21:11434,192.168.1.22:11434
#RPC_BACKENDS=192.168.1.20:50052
```

Update service file to use it:

```ini
[Service]
EnvironmentFile=/etc/sollol/config.env
...
```

### Monitoring and Logs

```bash
# SOLLOL logs
journalctl -u sollol -f

# Ray logs
journalctl -u ray-head -f

# RPC backend logs
journalctl -u rpc-backend -f

# Redis logs
journalctl -u redis -f

# System resources
htop
nvidia-smi  # GPU monitoring

# SOLLOL Dashboard
http://192.168.1.10:8080

# Ray Dashboard
http://192.168.1.10:8265
```

---

## Verification

### 1. Check All Services

```bash
# Health check
curl http://localhost:11434/api/health | jq

# Expected output:
{
  "status": "healthy",
  "service": "SOLLOL",
  "version": "0.5.0",
  "ray_parallel_execution": {
    "enabled": true,
    "actors": 16
  },
  "intelligent_routing": {
    "enabled": true
  }
}
```

### 2. Check Cluster Components

```bash
# Ray cluster
ray status

# Redis connectivity
redis-cli -h 192.168.1.10 ping

# RPC backends
redis-cli keys "sollol:rpc:node:*"

# Ollama nodes
curl http://192.168.1.21:11434/api/tags
```

### 3. Test Inference

```bash
# Test small model (Ollama pool)
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}'

# Test large model (RPC sharding)
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.1:70b",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}'
```

### 4. Check Logs

Watch for intelligent routing decisions:

```bash
journalctl -u sollol -f | grep "coordinator"
# Should see:
# Pool 0: Selecting coordinator node for llama3.1:70b
# Pool 0: Selected 192.168.1.20 for coordinator (score=140616)
```

---

## Troubleshooting

### SOLLOL won't start

```bash
# Check if port is in use
sudo ss -tunlp | grep 11434

# Check Redis connectivity
redis-cli ping

# Check logs
journalctl -u sollol -n 50
```

### Ray cluster not connecting

```bash
# Check firewall
sudo ufw allow 6380/tcp
sudo ufw allow 8265/tcp

# Test connectivity
telnet 192.168.1.10 6380

# Restart Ray
ray stop
ray start --head --port=6380 --dashboard-host=0.0.0.0
```

### RPC backends not found

```bash
# Verify registration in Redis
redis-cli keys "sollol:rpc:node:*"

# Re-register
python3 register_rpc_gpu_node.py

# Check RPC server is running
ss -tunlp | grep 50052
```

### OOM errors on large models

This is what remote coordinator execution solves! Check:

```bash
# Verify Ray cluster is running
ray status

# Verify GPU metadata in Redis
redis-cli get "sollol:rpc:node:192.168.1.20:50052"

# Check logs for node selection
journalctl -u sollol | grep "Selected.*for coordinator"
```

### Performance issues

```bash
# Check system resources
htop
nvidia-smi

# Increase workers
export SOLLOL_RAY_WORKERS=32
sudo systemctl restart sollol

# Check network latency
ping 192.168.1.20
```

---

## Next Steps

1. **Configure Monitoring**: Set up Prometheus/Grafana for metrics
2. **Tune Performance**: Adjust worker counts based on load
3. **Scale Horizontally**: Add more Ollama/RPC nodes
4. **Secure Setup**: Add authentication, TLS, firewall rules
5. **Backup Configuration**: Document your custom settings

For detailed configuration options, see [CONFIGURATION.md](CONFIGURATION.md).

For distributed inference details, see [REMOTE_COORDINATOR_DESIGN.md](REMOTE_COORDINATOR_DESIGN.md).

For Ray cluster setup, see [RAY_CLUSTER_SETUP.md](RAY_CLUSTER_SETUP.md).
