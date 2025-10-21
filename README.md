# SOLLOL - Orchestration Framework for Local LLM Clusters

<div align="center">

[![Development Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/BenevolentJoker-JohnL/SOLLOL)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/BenevolentJoker-JohnL/SOLLOL/actions/workflows/tests.yml/badge.svg)](https://github.com/BenevolentJoker-JohnL/SOLLOL/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/BenevolentJoker-JohnL/SOLLOL/branch/main/graph/badge.svg)](https://codecov.io/gh/BenevolentJoker-JohnL/SOLLOL)
[![Ollama](https://img.shields.io/badge/Ollama-Compatible-green.svg)](https://ollama.ai/)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-Integrated-orange.svg)](https://github.com/ggerganov/llama.cpp)

**Open-source orchestration framework that combines intelligent task routing with distributed model inference for local LLM clusters — designed for production scalability.**

[Quick Start](#-quick-start) • [Features](#-why-sollol) • [Architecture](#-architecture) • [Documentation](#-documentation) • [Examples](#-integration-examples)

</div>

---

## ⚠️ Development Status

**SOLLOL is currently in alpha development.**

### ✅ Production-Stable Features:
- **Task distribution** across Ollama nodes (mature, tested)
- **Intelligent routing** with adaptive learning (stable)
- **Observability dashboard** (fully functional)
- **Auto-discovery** and failover (reliable)

### 🔬 Experimental Features:
- **Distributed inference** (llama.cpp RPC integration)
  - ⚠️ **Not production-ready** - manual setup, version-sensitive, not optimized
  - Basic validation only (13B models, simple cases)
  - Known issues: slow performance, complex troubleshooting, frequent version conflicts
  - **Funding required for production optimization**

**Our Recommendation:** Use SOLLOL for task distribution and intelligent routing (proven, stable). Distributed inference is available for experimentation but not recommended for production use without dedicated engineering resources.

---

## 🎯 What is SOLLOL?

SOLLOL (Super Ollama Load balancer & Orchestration Layer) transforms your collection of Ollama nodes into an **intelligent AI cluster** with adaptive routing and automatic failover—all running on your own hardware.

### The Problem

You have multiple machines with GPUs running Ollama, but:
- ❌ Manual node selection for each request
- ❌ No way to run models larger than your biggest GPU
- ❌ Can't distribute multi-agent workloads efficiently
- ❌ No automatic failover or load balancing
- ❌ Zero visibility into cluster performance

### The SOLLOL Solution

**Production-Ready Features:**
- ✅ **Intelligent routing** that learns which nodes work best for each task
- ✅ **Parallel agent execution** for multi-agent frameworks
- ✅ **Auto-discovery** of Ollama nodes across your network
- ✅ **Built-in observability** with real-time metrics and dashboard
- ✅ **Automatic failover** and health monitoring

**Experimental Features (Not Recommended for Production):**
- 🔬 **Distributed inference** via llama.cpp RPC
  - ⚠️ **WARNING:** Experimental only - 5x slower than local, version-sensitive, manual setup
  - Works for testing only (13B models, 2-3 nodes)
  - **Funding required for production optimization**
  - See [EXPERIMENTAL_FEATURES.md](EXPERIMENTAL_FEATURES.md) for details and realistic expectations

---

## ⚡ Quickstart (3 Commands)

```bash
# 1. Install SOLLOL
pip install sollol

# 2. Start the dashboard (optional but recommended)
python3 -m sollol.dashboard_service &

# 3. Run your first query
python3 -c "from sollol import OllamaPool; pool = OllamaPool.auto_configure(); print(pool.chat(model='llama3.2', messages=[{'role': 'user', 'content': 'Hello!'}])['message']['content'])"
```

**What just happened?**
- ✅ SOLLOL auto-discovered all Ollama nodes on your network
- ✅ Intelligently routed your request to the best available node
- ✅ Dashboard live at `http://localhost:8080` (shows routing decisions, metrics, logs)

**Expected output:**
```
Discovering Ollama nodes...
Found 3 nodes: 192.168.1.22:11434, 192.168.1.10:11434, localhost:11434
Selected node: 192.168.1.22:11434 (GPU, 12ms latency)
Hello! How can I help you today?
```

**Next steps:**
- Visit `http://localhost:8080` to see the dashboard
- Check [Full Quick Start](#-full-quick-start) for production setup
- Read [Examples](#-integration-examples) for multi-agent, batch, and distributed inference patterns

---

## 🚀 Full Quick Start

### Installation

```bash
pip install sollol
```

### Basic Usage

```python
from sollol import OllamaPool

# Auto-discover nodes and start routing
pool = OllamaPool.auto_configure()

# Make requests - SOLLOL routes intelligently
response = pool.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Enable Real-Time GPU Monitoring

For accurate VRAM-aware routing, install the GPU reporter on each node:

```bash
# On each Ollama node, run:
sollol install-gpu-reporter --redis-host <redis-server-ip>

# Example:
sollol install-gpu-reporter --redis-host 192.168.1.10
```

**What this does:**
- Installs vendor-agnostic GPU monitoring (NVIDIA/AMD/Intel via `gpustat`)
- Publishes real-time VRAM stats to Redis every 5 seconds
- SOLLOL uses this data for intelligent routing decisions
- See [GPU Monitoring Guide](GPU_MONITORING_GUIDE.md) for details

**Without GPU monitoring:** SOLLOL falls back to estimates which may be inaccurate.

---

## 📸 Screenshots

### Dashboard Overview
![SOLLOL Unified Dashboard](docs/screenshots/dashboard-overview.png)
*Real-time monitoring with P50/P95/P99 latency metrics, network nodes, RPC backends, and active applications*

### Ray & Dask Integration
![Ray and Dask Dashboards](docs/screenshots/dashboard-metrics.png)
*Embedded Ray and Dask dashboards for distributed task monitoring*

### Activity Monitoring
![Real-time Activity Logs](docs/screenshots/dashboard-activity.png)
*Live request/response activity streams from Ollama nodes and RPC backends*

### Applications & Traces
![Performance Analytics](docs/screenshots/dashboard-ray-dask.png)
*Applications, distributed traces, and Ollama activity logs with real-time request/response tracking*

---

## 🔥 Why SOLLOL?

### 1. **Two Distribution Modes in One System**

SOLLOL combines both task distribution and distributed inference:

#### 📊 Task Distribution (Horizontal Scaling)
Distribute **multiple requests** across your cluster in parallel:
```python
# Run 10 agents simultaneously across 5 nodes
pool = OllamaPool.auto_configure()
responses = await asyncio.gather(*[
    pool.chat(model="llama3.2", messages=[...])
    for _ in range(10)
])
# Parallel execution across available nodes
```

#### 🔬 Distributed Inference (Experimental - Not Production Ready)

**⚠️ WARNING: This feature is experimental and not recommended for production use.**

Distribute **model layers** across RPC backends for parallel inference computation:
```python
# EXPERIMENTAL: Distribute inference layers across multiple RPC worker nodes
# ⚠️ Known Limitations:
#   - Slow startup (2-5 min vs 20s local)
#   - Slow inference (~5 tok/s vs ~20 tok/s local)
#   - Version-sensitive (rpc-server must exactly match coordinator)
#   - Manual configuration required
#   - Coordinator still needs full model in memory
# ✅ Works for: Simple 13B model testing across 2-3 compatible nodes
# ❌ Not suitable for: Production workloads, complex setups, large models

router = HybridRouter(
    enable_distributed=True,  # Only enable if you understand the limitations
    num_rpc_backends=2
)
# Use at your own risk - task distribution is recommended instead
```

**Our Recommendation:** Use task distribution (proven, stable) instead of distributed inference (experimental, slow) for production workloads.

---

### 2. **Intelligent, Not Just Balanced**

SOLLOL doesn't just distribute requests randomly—it **learns** and **optimizes**:

| Feature | Simple Load Balancer | SOLLOL |
|---------|---------------------|---------|
| **Routing** | Round-robin | Context-aware scoring |
| **Learning** | None | Adapts from performance history |
| **Resource Awareness** | None | GPU/CPU/memory-aware |
| **Task Optimization** | None | Routes by task type complexity |
| **Failover** | Manual | Automatic with health checks |
| **Priority** | FIFO | Priority queue with fairness |

**Example**: SOLLOL automatically routes:
- Heavy generation tasks → GPU nodes with 24GB VRAM
- Fast embeddings → CPU nodes or smaller GPUs
- Critical requests → Fastest, most reliable nodes
- Batch processing → Lower priority, distributed load

---

### 3. **Production-Ready from Day One**

```python
from sollol import SOLLOL, SOLLOLConfig

# Literally 3 lines to production
config = SOLLOLConfig.auto_discover()
sollol = SOLLOL(config)
sollol.start()  # ✅ Gateway running on :8000
```

**Out of the box**:
- Auto-discovery of Ollama nodes
- Health monitoring and failover
- Prometheus metrics
- Web dashboard
- Connection pooling
- Request hedging
- Priority queuing

---

### 4. **Unified Observability for Your Entire AI Network**

SOLLOL provides a **single pane of glass** to monitor every application and every node in your distributed AI network.

- ✅ **Centralized Dashboard**: One web interface shows all applications, nodes, and RPC backends.
- ✅ **Multi-App Tracking**: See which applications (e.g., SynapticLlamas, custom agents) are using the cluster in real-time.
- ✅ **Network-Wide Visibility**: The dashboard runs as a persistent service, discovering and monitoring all components even if no applications are running.
- ✅ **Zero-Config**: Applications automatically appear in the dashboard with no extra code required.

This moves beyond per-application monitoring to provide true, centralized observability for your entire infrastructure.

---

## 🏗️ Architecture

### High-Level Overview

```
┌────────────────────────────────────────────────────────┐
│                  Your Application                       │
│         (SynapticLlamas, custom agents, etc.)          │
└──────────────────────┬─────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│                 SOLLOL Gateway (:8000)                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Intelligent Routing Engine               │  │
│  │  • Analyzes: task type, complexity, resources    │  │
│  │  • Scores: all nodes based on context            │  │
│  │  • Learns: from performance history              │  │
│  │  • Routes: to optimal node                       │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │          Priority Queue + Failover               │  │
│  └──────────────────────────────────────────────────┘  │
└────────┬─────────────────────────┬─────────────────────┘
         │                         │
         ▼                         ▼
  ┌─────────────┐          ┌──────────────┐
  │ Task Mode   │          │  Shard Mode  │
  │ Ray Cluster │          │  llama.cpp   │
  └──────┬──────┘          └──────┬───────┘
         │                         │
         ▼                         ▼
┌────────────────────────────────────────────────────────┐
│              Your Heterogeneous Cluster                 │
│  GPU (24GB) │ GPU (16GB) │ CPU (64c) │ GPU (8GB) │...  │
└────────────────────────────────────────────────────────┘
```

### How Routing Works

```python
# 1. Request arrives
POST /api/chat {
  "model": "llama3.2",
  "messages": [{"role": "user", "content": "Complex analysis task..."}],
  "priority": 8
}

# 2. SOLLOL analyzes
task_type = "generation"       # Auto-detected
complexity = "high"             # Token count analysis
requires_gpu = True             # Based on task
estimated_duration = 3.2s       # From history

# 3. SOLLOL scores all nodes
Node A (GPU 24GB, load: 0.2, latency: 120ms) → Score: 185.3 ✓ WINNER
Node B (GPU 8GB,  load: 0.6, latency: 200ms) → Score: 92.1
Node C (CPU only, load: 0.1, latency: 80ms)  → Score: 41.2

# 4. Routes to Node A, monitors execution, learns for next time
```

**Scoring Algorithm**:
```
Score = 100.0 (baseline)
      × success_rate (0.0-1.0)
      ÷ (1 + latency_penalty)
      × gpu_bonus (1.5x if GPU available & needed)
      ÷ (1 + load_penalty)
      × priority_alignment
      × task_specialization
```

---

## 📦 Installation

### Quick Install (PyPI)
```bash
pip install sollol
```

### From Source
```bash
git clone https://github.com/BenevolentJoker-JohnL/SOLLOL.git
cd SOLLOL
pip install -e .
```

---

## ⚡ Quick Start

### 1. Synchronous API (No async/await needed!)

**New in v0.3.6:** SOLLOL now provides a synchronous API for easier integration with non-async applications.

```python
from sollol.sync_wrapper import OllamaPool
from sollol.priority_helpers import Priority

# Auto-discover and connect to all Ollama nodes
pool = OllamaPool.auto_configure()

# Make requests - SOLLOL routes intelligently
# No async/await needed!
response = pool.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Hello!"}],
    priority=Priority.HIGH,  # Semantic priority levels
    timeout=60  # Request timeout in seconds
)

print(response['message']['content'])
print(f"Routed to: {response.get('_sollol_routing', {}).get('host', 'unknown')}")
```

**Key features of synchronous API:**
- ✅ No async/await syntax required
- ✅ Works with synchronous agent frameworks
- ✅ Same intelligent routing and features
- ✅ Runs async code in background thread automatically

---

### 2. Async API (Original)

For async applications, use the original async API:

```python
from sollol import OllamaPool

# Auto-discover and connect to all Ollama nodes
pool = await OllamaPool.auto_configure()

# Make requests - SOLLOL routes intelligently
response = await pool.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response['message']['content'])
print(f"Routed to: {response['_sollol_routing']['host']}")
print(f"Task type: {response['_sollol_routing']['task_type']}")
```

---

### 3. Priority-Based Multi-Agent Execution

**New in v0.3.6:** Use semantic priority levels and role-based mapping.

```python
from sollol.sync_wrapper import OllamaPool
from sollol.priority_helpers import Priority, get_priority_for_role

pool = OllamaPool.auto_configure()

# Define agents with different priorities
agents = [
    {"name": "Researcher", "role": "researcher"},  # Priority 8
    {"name": "Editor", "role": "editor"},          # Priority 6
    {"name": "Summarizer", "role": "summarizer"},  # Priority 5
]

for agent in agents:
    priority = get_priority_for_role(agent["role"])

    response = pool.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": f"Task for {agent['name']}"}],
        priority=priority
    )
    # User-facing agents get priority, background tasks wait
```

**Priority levels available:**
- `Priority.CRITICAL` (10) - Mission-critical
- `Priority.URGENT` (9) - Fast response needed
- `Priority.HIGH` (7) - Important tasks
- `Priority.NORMAL` (5) - Default
- `Priority.LOW` (3) - Background tasks
- `Priority.BATCH` (1) - Can wait

---

### 4. Distributed Inference with llama.cpp RPC (Large Models)

**Distribute inference computation** for large models across multiple machines using llama.cpp RPC backends.

#### When to Use Distributed Inference

Use distributed inference when:
- ✅ Model doesn't fit on your largest GPU (e.g., 70B models on 16GB GPUs)
- ✅ You have multiple machines with network connectivity
- ✅ You can tolerate slower inference for capability
- ✅ Coordinator node has sufficient RAM for the full model

Don't use distributed inference when:
- ❌ Model fits on a single GPU (use task distribution instead)
- ❌ You need maximum inference speed
- ❌ Network latency is high (>10ms between machines)
- ❌ No single node has RAM for the full model (see known limitations)

#### Quick Start: Auto-Setup (Easiest)

```python
from sollol.sync_wrapper import HybridRouter, OllamaPool

# SOLLOL handles all setup automatically
router = HybridRouter(
    ollama_pool=OllamaPool.auto_configure(),
    enable_distributed=True,  # Enable distributed inference via llama.cpp RPC
    auto_setup_rpc=True,      # Auto-configure RPC backends
    num_rpc_backends=3        # Distribute across 3 machines
)

# Use large model that doesn't fit on one machine
response = router.route_request(
    model="llama3.1:70b",  # Automatically sharded across backends
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)

print(response['message']['content'])
```

**What happens automatically:**
1. SOLLOL discovers available RPC backends on your network
2. Extracts the GGUF model from Ollama storage
3. Starts llama-server coordinator with optimal settings
4. Distributes model layers across backends
5. Routes your request to the coordinator

#### RPC Server Auto-Installation

**SOLLOL can automatically clone, build, and start llama.cpp RPC servers for you!**

**One-line installation:**

```python
from sollol.rpc_auto_setup import auto_setup_rpc_backends

# Automatically: clone → build → start RPC servers
backends = auto_setup_rpc_backends(num_backends=2)
# Output: [{'host': '127.0.0.1', 'port': 50052}, {'host': '127.0.0.1', 'port': 50053}]
```

**What this does:**
1. ✅ Scans network for existing RPC servers
2. ✅ If none found: clones llama.cpp to `~/llama.cpp`
3. ✅ Builds llama.cpp with RPC support (`cmake -DGGML_RPC=ON`)
4. ✅ Starts RPC servers on ports 50052-50053
5. ✅ Returns ready-to-use backend list

**CLI installation:**

```bash
# Full automated setup (clone + build + install systemd service)
python3 -m sollol.setup_llama_cpp --all

# Or step by step
python3 -m sollol.setup_llama_cpp --clone  # Clone llama.cpp
python3 -m sollol.setup_llama_cpp --build  # Build with RPC support
python3 -m sollol.setup_llama_cpp --start  # Start RPC server
```

#### Building llama.cpp with GPU Support 🚀

**For GPU-accelerated distributed inference**, rebuild llama.cpp with CUDA support. The GPU-enabled binaries work on ALL nodes (CPU-only nodes will just use CPU).

**Quick install script (automated CUDA + build):**

```bash
# Download and run installation script
curl -fsSL https://github.com/BenevolentJoker-JohnL/SOLLOL/raw/main/scripts/install_cuda_llama.sh | bash

# Or manual installation (see below)
```

**Manual Installation Steps:**

```bash
# 1. Install CUDA toolkit (Ubuntu/Debian)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6

# 2. Set CUDA environment
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# 3. Clone and build llama.cpp with CUDA
cd ~
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build RPC server with CUDA support
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DGGML_RPC=ON

cmake --build build --config Release --target rpc-server -j $(nproc)

# 4. Install binary
mkdir -p ~/.local/bin
cp build/bin/rpc-server ~/.local/bin/

# 5. Deploy and start RPC servers

# For GPU nodes (192.168.1.20, etc.):
# Copy binary to GPU node
scp ~/.local/bin/rpc-server <gpu-node>:~/.local/bin/

# On GPU node, start with CUDA support
ssh <gpu-node>
nohup ~/.local/bin/rpc-server --host 0.0.0.0 --port 50052 > /tmp/rpc-server.log 2>&1 &

# For CPU-only nodes:
# Build without CUDA (faster, smaller binary)
cd ~/llama.cpp
cmake -B build-cpu \
  -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DGGML_RPC=ON

cmake --build build-cpu --config Release --target rpc-server -j $(nproc)
cp build-cpu/bin/rpc-server ~/.local/bin/rpc-server-cpu

# Start CPU-only RPC server
nohup ~/.local/bin/rpc-server-cpu --host 0.0.0.0 --port 50052 > /tmp/rpc-server.log 2>&1 &
```

**Why build separate binaries?**
- ✅ **CUDA binary (689MB)**: Runs on GPU nodes, uses CUDA automatically
- ✅ **CPU-only binary (~200MB)**: Smaller, runs on coordinator/CPU-only nodes
- ✅ **Deployment**: CUDA binary requires NVIDIA drivers on target machine
- ✅ **Hybrid parallelization**: GPU nodes can use CPU + GPU workers

**Note**: The CUDA-enabled binary **will not run** on machines without NVIDIA drivers (like CPU-only coordinator). Build CPU-only version for coordinator nodes.

**For other GPU vendors:**
- **AMD GPUs**: Use `-DGGML_HIPBLAS=ON` instead of `-DGGML_CUDA=ON`
- **Intel GPUs**: Use `-DGGML_VULKAN=ON` for Intel/AMD/NVIDIA compatibility

**Enabling GPU Detection and Reporting:**

For SOLLOL to intelligently route requests to GPU nodes, you need to configure Redis and GPU registration:

📖 **See [GPU Detection Setup Guide](docs/GPU_DETECTION_SETUP.md)** for complete instructions on:
- Configuring Redis for network access
- Registering GPU nodes with Redis
- Automatic GPU capability detection
- Troubleshooting GPU detection issues

**Quick summary:**
1. Configure Redis to listen on network interface (not just localhost)
2. Run `scripts/register_gpu_node.py` on each GPU node
3. SOLLOL automatically detects GPU capabilities and routes accordingly

**Docker IP Resolution:**

SOLLOL automatically resolves Docker container IPs to accessible host IPs:

```python
# If Docker container reports IP 172.17.0.5:11434
# SOLLOL automatically resolves to:
# → 127.0.0.1:11434 (published port mapping)
# → host IP (if accessible)
# → Docker host gateway

from sollol import is_docker_ip, resolve_docker_ip

# Check if IP is Docker internal
is_docker = is_docker_ip("172.17.0.5")  # True

# Resolve Docker IP to accessible IP
accessible_ip = resolve_docker_ip("172.17.0.5", port=11434)
# Returns: "127.0.0.1" or host IP
```

**Network Discovery with Docker Support:**

```python
from sollol import OllamaPool

# Auto-discover nodes (automatically resolves Docker IPs)
pool = OllamaPool.auto_configure()

# Manual control
from sollol.discovery import discover_ollama_nodes
nodes = discover_ollama_nodes(auto_resolve_docker=True)
```

**Multi-Node Production Setup:**

For distributed clusters, use systemd services on each node:

```bash
# On each RPC node
sudo systemctl enable llama-rpc@50052.service
sudo systemctl start llama-rpc@50052.service
```

See [SOLLOL_RPC_SETUP.md](https://github.com/BenevolentJoker-JohnL/FlockParser/blob/main/SOLLOL_RPC_SETUP.md) for complete installation guide.

#### Architecture: Hybrid GPU+CPU Parallelization 🚀
**llama.cpp RPC backend only**

**NEW:** llama.cpp RPC backends support hybrid device parallelization - GPU nodes contribute BOTH their GPU (VRAM) AND CPU (RAM) as separate parallel workers!

> **Note:** This feature is specific to llama.cpp's `rpc-server` architecture and is NOT available with Ollama. Ollama nodes can only contribute one worker per instance.

**Example: 3 Physical Nodes → 4 Parallel Workers (+33% throughput!)**

```
Traditional Setup (3 workers):
CPU Node 1  → 1 worker (8GB RAM)
CPU Node 2  → 1 worker (8GB RAM)
GPU Node    → 1 worker (12GB VRAM)
Total: 3 parallel workers

Hybrid Setup (4 workers):
CPU Node 1  → 1 worker (8GB RAM)
CPU Node 2  → 1 worker (8GB RAM)
GPU Node    → 2 workers:
              ├─ CPU device (10GB RAM)  ← Both run in parallel!
              └─ GPU device (9.6GB VRAM) ←
Total: 4 parallel workers  (+33% more!)
```

**Fully Automatic** - SOLLOL detects and configures hybrid parallelization when starting llama.cpp RPC servers:
- Auto-detects GPU(s) and calculates safe VRAM allocations (80% with 20% reserve)
- Auto-detects system RAM and reserves 20% for OS stability
- Generates optimal `--device` and `--mem` flags for llama.cpp's `rpc-server` automatically
- No manual configuration needed!

**Why llama.cpp only?** Ollama doesn't support the `--device cpu,cuda:0` multi-device flag that enables this hybrid parallelization.

**Manual inspection** (optional) - See what SOLLOL would configure:
```bash
python scripts/setup_rpc_node.py
```

**Sample output (GPU node):**
```
✅ GPU(s) Found: 1
   GPU 0: cuda:0 - 9600 MB VRAM (safe allocation)
💾 CPU RAM: 10240 MB (safe allocation)
⚡ Total Parallel Workers: 2 (1 CPU worker + 1 GPU worker)

RPC-SERVER COMMAND:
rpc-server --host 0.0.0.0 --port 50052 --device cpu,cuda:0 --mem 10240,9600
```

**Key Benefits:**
- ✅ Maximizes hardware utilization (use ALL resources)
- ✅ No extra hardware needed (same 3 machines)
- ✅ Safe memory allocations (80% with 20% reserve)
- ✅ Automatic GPU detection (NVIDIA, AMD, Intel)

See [docs/HYBRID_RPC_PARALLELIZATION.md](docs/HYBRID_RPC_PARALLELIZATION.md) for complete guide.

#### Layer Distribution Example

```
┌────────────────────────────────────────────┐
│    Llama 3.1 70B Model (40GB total)        │
│     Hybrid Distributed Parallelization     │
└────────────────────────────────────────────┘
                    │
       ┌────────────┼────────────┬────────────┐
       │            │            │            │
       ▼            ▼            ▼            ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  CPU Node 1  │ │  CPU Node 2  │ │  GPU Node    │ │  GPU Node    │
│  Layers 0-9  │ │ Layers 10-19 │ │ CPU Device   │ │ GPU Device   │
│   (8GB RAM)  │ │   (8GB RAM)  │ │ Layers 20-29 │ │ Layers 30-39 │
│ RPC Backend  │ │ RPC Backend  │ │ (10GB RAM)   │ │ (9.6GB VRAM) │
│              │ │              │ │ RPC Backend  │ │ RPC Backend  │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
       ▲            ▲                    ▲                ▲
       └────────────┴────────────────────┴────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │ llama-server        │
                    │ Coordinator         │
                    │ (Port 18080)        │
                    └─────────────────────┘

4 parallel workers across 3 physical machines!
```

#### Manual Setup (Advanced)

For explicit control over RPC backends:

```python
from sollol.llama_cpp_coordinator import LlamaCppCoordinator
from sollol.rpc_registry import RPCBackendRegistry

# 1. Register RPC backends explicitly
registry = RPCBackendRegistry()
registry.add_backend("rpc_1", "grpc://192.168.1.22:50052")
registry.add_backend("rpc_2", "grpc://10.9.66.46:50052")
registry.add_backend("rpc_3", "grpc://10.9.66.47:50052")

# 2. Create coordinator
coordinator = LlamaCppCoordinator(
    coordinator_port=18080,
    rpc_backends=registry.get_all_backends(),
    context_size=4096,
    gpu_layers=-1  # Use all available GPU layers
)

# 3. Start and use
await coordinator.start(model_name="llama3.1:70b")
response = await coordinator.generate(
    prompt="Explain the theory of relativity",
    max_tokens=500
)
```

#### Performance Expectations

| Model Size | Single GPU | Sharded (3 nodes) | Trade-off |
|------------|-----------|-------------------|-----------|
| **13B** | ✅ 20 tok/s | ✅ 5 tok/s | -75% speed, works on 3×smaller GPUs |
| **70B** | ❌ OOM | ⚠️ 3-5 tok/s (est.) | Enables model that won't run otherwise |

**Trade-offs:**
- 🐌 **Startup**: 2-5 minutes (model distribution + loading)
- 🐌 **Inference**: ~4x slower than local (network overhead)
- ✅ **Capability**: Run models that won't fit on single GPU

**Learn More:**
- 📖 [Complete llama.cpp Guide](docs/llama_cpp_guide.md) - Setup, optimization, troubleshooting
- 💻 [Working Examples](examples/llama_cpp_distributed.py) - 5 complete examples including conversation, batch processing, error handling

#### ⚠️ Known Limitation: llama.cpp Coordinator Bottleneck

**The Problem:** llama.cpp's `--rpc` flag enables distributed *computation* but NOT distributed *storage*. The coordinator node must load the entire model into its own RAM before distributing layer computation to RPC workers.

**Impact:**
- ❌ 70B model (40GB GGUF) on coordinator with 8GB RAM → **Crashes**
- ❌ Even with 48GB total across 4 workers, coordinator still needs 40GB locally
- ✅ Works if coordinator node has sufficient RAM (e.g., 32GB+ for 70B models)

**Workaround:** Run the llama-server coordinator on your machine with the most RAM (e.g., your GPU node with 32GB). The RPC workers can still be CPU-only nodes with less RAM.

**Research Track (WIP):** SOLLOL includes experimental code (`src/sollol/distributed_pipeline.py`) exploring true distributed model weight sharding where NO single node needs the full model. This uses Ray-based pipeline parallelism inspired by prima.cpp's architecture. See module documentation for technical details and path forward.

#### 🔬 Future Work: Production-Ready Distributed Inference & Model Weight Sharding

**Current Status:** SOLLOL's distributed inference is **experimental only**:
- Basic functionality validated (13B models, 2-3 nodes)
- Not optimized for performance (5x slower than local)
- Requires manual setup and exact version matching
- **Not recommended for production use**

**What's Needed for Production:**
1. **Optimization of current RPC integration:**
   - Performance tuning (reduce 2-5min startup to <30s)
   - Automated version management
   - Better error handling and recovery
   - Comprehensive testing across model sizes

2. **True model weight sharding** (future):
   - NO single node needs full model in RAM
   - Requires significant R&D (see `distributed_pipeline.py`)

**What's Needed:**
- ✅ GGUF layer analysis (completed)
- ✅ Ray-based architecture design (completed)
- ⚠️ Quantization-aware GGUF writing (complex, blocked)
- ⚠️ Tensor streaming via Ray object store (unimplemented)
- ⚠️ Multi-node validation with 70B+ models (requires cluster access)

**Proposed Extension:**
- **Ollama ⇄ GGUF conversion** for tensor-level distribution across nodes
- **Weight-sharded model loading** with Ray actors (no single-node memory bottleneck)
- **Pipeline parallelism** with activation passing through Ray object store
- **Production validation** on real multi-node clusters with frontier models (70B-405B)

**Important Distinction:**
- **Current:** Distributed **inference** - layers spread across nodes, coordinator still needs full model
- **Future:** Distributed **model weights** - model parameters split across nodes, no single node needs full model

> 💡 **This feature is gated behind funding or partnership opportunities.**
>
> It addresses a genuine limitation in local LLM infrastructure: running frontier models on consumer hardware clusters without requiring any single machine to have enterprise-grade RAM (40GB+).
>
> **Why this matters:** Enables sovereign AI deployment at scale without cloud dependencies or specialized hardware.

**Status:** Research track with working foundation (570+ lines), documented blockers, and clear implementation path. See `src/sollol/distributed_pipeline.py` for technical details.

**Interested in funding or collaboration?** Open an issue or contact via GitHub.

---

### 5. Batch Processing API

**New in v0.7.0:** RESTful API for asynchronous batch job management.

Submit large-scale batch operations (thousands of embeddings, bulk inference) and track progress via job IDs:

```python
import requests

# Submit batch embedding job (up to 10,000 documents)
response = requests.post("http://localhost:11434/api/batch/embed", json={
    "model": "nomic-embed-text",
    "documents": ["Document 1", "Document 2", ...],  # Can be thousands
    "metadata": {"source": "knowledge_base"}  # Optional metadata
})

job_id = response.json()["job_id"]
print(f"Job submitted: {job_id}")

# Poll for job status
import time
while True:
    status = requests.get(f"http://localhost:11434/api/batch/jobs/{job_id}").json()

    progress = status["progress"]["percent"]
    print(f"Progress: {progress}%")

    if status["status"] == "completed":
        break
    time.sleep(1)

# Get results
results = requests.get(f"http://localhost:11434/api/batch/results/{job_id}").json()
embeddings = results["results"]  # List of embedding vectors
print(f"Processed {len(embeddings)} documents in {status['duration_seconds']}s")
```

**Available Batch Endpoints:**
- `POST /api/batch/embed` - Submit batch embedding job
- `GET /api/batch/jobs/{job_id}` - Get job status
- `GET /api/batch/results/{job_id}` - Get job results
- `GET /api/batch/jobs?limit=100` - List recent jobs
- `DELETE /api/batch/jobs/{job_id}` - Cancel job

**Use cases:**
- Embedding large document collections (thousands of documents)
- Bulk inference for batch predictions
- Background processing without blocking
- Long-running operations with progress tracking

---

### 6. SOLLOL Detection

**New in v0.3.6:** Detect if SOLLOL is running vs native Ollama.

```python
import requests

def is_sollol(url="http://localhost:11434"):
    """Check if SOLLOL is running at the given URL."""

    # Method 1: Check X-Powered-By header
    response = requests.get(url)
    if response.headers.get("X-Powered-By") == "SOLLOL":
        return True

    # Method 2: Check health endpoint
    response = requests.get(f"{url}/api/health")
    data = response.json()
    if data.get("service") == "SOLLOL":
        return True

    return False

# Use it
if is_sollol("http://localhost:11434"):
    print("✓ SOLLOL detected - using intelligent routing")
else:
    print("Native Ollama detected")
```

**Why this matters:**
- Enables graceful fallback in client applications
- Makes SOLLOL a true drop-in replacement
- Clients can auto-detect and use SOLLOL features when available

---

### 7. Production Gateway

```python
from sollol import SOLLOL, SOLLOLConfig

# Full production setup
config = SOLLOLConfig(
    ray_workers=4,
    dask_workers=2,
    hosts=["gpu-1:11434", "gpu-2:11434", "cpu-1:11434"],
    gateway_port=8000,
    metrics_port=9090
)

sollol = SOLLOL(config)
sollol.start()  # Blocks and runs gateway

# Access via HTTP:
# curl http://localhost:8000/api/chat -d '{...}'
# curl http://localhost:8000/api/stats
# curl http://localhost:8000/api/dashboard
```

---

## 🎓 Use Cases

### 1. Multi-Agent AI Systems (SynapticLlamas, CrewAI, AutoGPT)

**Problem**: Running 10 agents sequentially takes 10x longer than necessary.

**Solution**: SOLLOL distributes agents across nodes in parallel.

```python
# Before: Sequential execution on one node
# After: Parallel execution with SOLLOL
pool = OllamaPool.auto_configure()
agents = await asyncio.gather(*[
    pool.chat(model="llama3.2", messages=agent_prompts[i])
    for i in range(10)
])
# Speedup depends on number of available nodes and their capacity
```

### 2. Large Model Inference

**Problem**: Your model doesn't fit in available VRAM.

**Solution**: SOLLOL distributes inference computation across multiple machines via llama.cpp RPC.

```python
# Distribute inference across multiple RPC backends
# Note: Coordinator node must have RAM for full model
# Verified with 13B models; larger models not extensively tested
router = HybridRouter(
    enable_distributed=True,
    num_rpc_backends=4
)
# Trade-off: Slower startup/inference but enables running larger models
# Distributes computation, not model storage (see known limitations)
```

### 3. Mixed Workloads

**Problem**: Different tasks need different resources.

**Solution**: SOLLOL routes each task to the optimal node.

```python
pool = OllamaPool.auto_configure()

# Heavy generation → GPU node
chat = pool.chat(model="llama3.2:70b", messages=[...])

# Fast embeddings → CPU node
embeddings = pool.embed(model="nomic-embed-text", input=[...])

# SOLLOL automatically routes each to the best available node
```

### 4. High Availability Production

**Problem**: Node failures break your service.

**Solution**: SOLLOL auto-fails over and recovers.

```python
# Node A fails mid-request
# ✅ SOLLOL automatically:
# 1. Detects failure
# 2. Retries on Node B
# 3. Marks Node A as degraded
# 4. Periodically re-checks Node A
# 5. Restores Node A when healthy
```

#### **Simulate Failure & Recovery**

Want to see SOLLOL's automatic failover in action? Run the included simulation:

```bash
python test_failure_recovery.py
```

**What the simulation does:**
1. Starts 3 mock Ollama nodes
2. Sends baseline requests (all nodes healthy)
3. **Kills node #1 mid-execution**
4. Continues sending requests (SOLLOL routes around failed node)
5. Restores node #1
6. Resumes sending requests (traffic returns to recovered node)

**Expected output:**
```
STEP 1: Starting Mock Nodes
✅ Started 3 mock nodes

BASELINE: Requests with all nodes healthy
  Request 1: ✓ Routed to localhost:21434
  Request 2: ✓ Routed to localhost:21435
  ...

STEP 3: Simulating Node Failure (killing node 0)
Killing node on port 21434...
✅ Node 21434 terminated

STEP 4: Requests after node failure (observe failover)
  Request 1: ✓ Routed to localhost:21435  ← Automatically avoided dead node
  Request 2: ✓ Routed to localhost:21436
  ...

STEP 5: Simulating Node Recovery
✅ Node 21434 recovered successfully

✅ Key Observations:
  1. Requests succeeded even after node failure
  2. SOLLOL automatically routed around the dead node
  3. Node recovered and rejoined the pool
  4. Traffic resumed to recovered node
```

This demonstrates SOLLOL's production-grade resilience without needing real infrastructure.

---

## 📊 Performance & Benchmarks

### Validation Status

**What's Been Validated ✅**
- Single-node baseline performance measured
- Code exists and is reviewable (75+ modules)
- Tests pass in CI (57 tests, coverage tracked)
- Architecture implements intelligent routing

**What Needs Validation ⚠️**
- Comparative benchmarks (SOLLOL vs round-robin)
- Multi-node performance improvements
- Real-world latency/throughput gains

📖 **See [BENCHMARKING.md](BENCHMARKING.md) for complete validation roadmap and how to run comparative tests.**

---

### Measured Baseline Performance

**Single Ollama Node** (llama3.2-3B, 50 requests, concurrency=5):
- ✅ **Success Rate:** 100%
- ⚡ **Throughput:** 0.51 req/s
- 📈 **Average Latency:** 5,659 ms
- 📈 **P95 Latency:** 11,299 ms
- 📈 **P99 Latency:** 12,259 ms

**Hardware:** Single Ollama instance with 75+ models loaded
**Data:** See [`benchmarks/results/`](benchmarks/results/) for raw JSON

**Run Your Own:**
```bash
# Baseline test (no cluster needed)
python benchmarks/simple_ollama_benchmark.py llama3.2 50

# Comparative test (requires docker-compose)
docker-compose up -d
python benchmarks/run_benchmarks.py --sollol-url http://localhost:8000 --duration 60
```

---

### Projected Performance (Unvalidated)

**Note:** These are architectural projections, not measured results. Requires multi-node cluster setup for validation.

**Theory:** With N nodes and parallelizable workload:
- Task distribution can approach N× parallelization (limited by request rate)
- Intelligent routing should reduce tail latencies vs random selection
- Resource-aware placement reduces contention and failures

**Reality:** Requires multi-node cluster validation. See [BENCHMARKING.md](BENCHMARKING.md) for test procedure and [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) for implementation details.

### Distributed Inference Performance

| Model | Single 24GB GPU | SOLLOL (3×16GB via RPC) | Status |
|-------|----------------|-------------------------|-----------|
| **13B** | ✅ ~20 tok/s | ✅ ~5 tok/s | ✅ Verified working |
| **70B** | ❌ OOM | ⚠️ Estimated ~3-5 tok/s | ⚠️ Not extensively tested |

**When to use distributed inference**: When model doesn't fit on your largest GPU. You trade speed for capability.

**Performance trade-offs**: Distributed inference is 2-5 minutes slower to start and ~4x slower for inference compared to local. Use only when necessary.

**Note**: This distributes *computation* (inference), not model storage. Coordinator node must still have sufficient RAM for the full model.

### Overhead

- **Routing decision**: ~5-10ms (tested with 5-10 nodes)
- **Network overhead**: Varies by network (typically 5-20ms)
- **Total added latency**: ~20-50ms
- **Benefit**: Better resource utilization + automatic failover

---

## 🛠️ Advanced Configuration

### Custom Routing Strategy

```python
from sollol import OllamaPool

pool = OllamaPool(
    nodes=[
        {"host": "gpu-1.local", "port": 11434, "priority": 10},  # Prefer this
        {"host": "gpu-2.local", "port": 11434, "priority": 5},
        {"host": "cpu-1.local", "port": 11434, "priority": 1},   # Last resort
    ],
    enable_intelligent_routing=True,
    enable_hedging=True,  # Duplicate critical requests
    max_queue_size=100
)
```

### Priority-Based Scheduling

```python
# Critical user-facing request
response = pool.chat(
    model="llama3.2",
    messages=[...],
    priority=10  # Highest priority
)

# Background batch job
response = pool.chat(
    model="llama3.2",
    messages=[...],
    priority=1  # Lowest priority
)

# SOLLOL ensures high-priority requests jump the queue
```

### Observability & Monitoring

#### **Zero-Config Auto-Registration** 🎯

SOLLOL provides **automatic observability** with zero configuration required. All applications automatically register with the dashboard when they create an `OllamaPool`:

```python
from sollol import OllamaPool

# Creates pool AND auto-registers with dashboard (if running)
pool = OllamaPool.auto_configure()
# ✅ Application automatically appears in dashboard at http://localhost:8080
```

**How it works:**
1. `OllamaPool` automatically detects if a dashboard is running on port 8080
2. Auto-discovers RPC backends and Ollama nodes
3. Registers application with metadata (node count, GPU info, etc.)
4. Sends periodic heartbeats to maintain "alive" status
5. No manual `DashboardClient` setup needed!

**Architecture:**
- **ONE persistent dashboard service** runs independently
- **Multiple applications** (SynapticLlamas, FlockParser, etc.) auto-register
- **Dashboard survives** application exits
- **Zero-config** auto-discovery of nodes and RPC backends

#### **Custom Application Names** 🏷️

By default, applications register as "OllamaPool (hostname)". To give your application a custom name in the dashboard:

```python
from sollol import OllamaPool

# Register with custom application name
pool = OllamaPool(
    nodes=[{"host": "localhost", "port": 11434}],
    enable_intelligent_routing=True,
    app_name="MyApplication"  # Shows as "MyApplication" in dashboard
)
```

**Example - Multi-application setup:**

```python
# Application 1: FlockParser
from sollol import OllamaPool

pool = OllamaPool.auto_configure(app_name="FlockParser")
# Dashboard shows: "FlockParser"

# Application 2: SynapticLlamas
from sollol.dashboard_client import DashboardClient

dashboard_client = DashboardClient(
    app_name="SynapticLlamas",
    router_type="IntelligentRouter",
    version="1.0.0",
    dashboard_url="http://localhost:8080",
    metadata={"agents": 3, "distributed": True},
    auto_register=True
)
# Dashboard shows: "SynapticLlamas"
```

**Why use custom names?**
- Distinguish between multiple applications using SOLLOL
- Better visibility in multi-tenant environments
- Easier debugging and monitoring
- Professional dashboard presentation

#### **Manual/Programmatic Registration** 🔧

For applications that don't use `OllamaPool` or need custom registration logic, use `DashboardClient` directly:

```python
from sollol.dashboard_client import DashboardClient

# Create dashboard client with custom metadata
dashboard_client = DashboardClient(
    app_name="CustomApplication",
    router_type="CustomRouter",  # Or "OllamaPool", "HybridRouter", etc.
    version="1.0.0",
    dashboard_url="http://localhost:8080",
    metadata={
        # Custom metadata shown in dashboard
        "nodes": 5,
        "distributed": True,
        "custom_field": "value"
    },
    auto_register=True  # Registers immediately
)

# Dashboard client automatically sends heartbeats every 5 seconds
# to keep application status as "active"

# When application exits, clean up:
dashboard_client.close()  # Stops heartbeat thread
```

**Advanced: Custom Heartbeat Logic**

```python
from sollol.dashboard_client import DashboardClient
import time

# Create client without auto-registration
dashboard_client = DashboardClient(
    app_name="BackgroundWorker",
    router_type="WorkerPool",
    version="2.0.0",
    dashboard_url="http://localhost:8080",
    metadata={"worker_count": 10},
    auto_register=False  # Don't register yet
)

# Register when ready
dashboard_client.register()

# Update metadata dynamically
dashboard_client.update_metadata({"worker_count": 15, "status": "processing"})

# Send manual heartbeat
dashboard_client.heartbeat()

# Application logic here...
time.sleep(60)

# Deregister when done
dashboard_client.deregister()
dashboard_client.close()
```

**Use cases for manual registration:**
- Custom routers or load balancers
- Background workers or daemons
- Applications that need dynamic metadata updates
- Testing and debugging
- Applications without OllamaPool

#### **Registration Methods Comparison** 📊

| Method | Use Case | Complexity | Customization |
|--------|----------|------------|---------------|
| **Auto-registration** | Standard SOLLOL applications | ✅ Zero config | Limited (app_name only) |
| **Custom app_name** | Multiple apps, better naming | ✅ One parameter | App name |
| **Manual DashboardClient** | Custom applications | ⚠️ More code | Full control |

**Quick decision guide:**
- Using `OllamaPool`? → Use `app_name` parameter
- Need custom metadata? → Use `DashboardClient` directly
- Need dynamic updates? → Use `DashboardClient` with manual heartbeats
- Just want it to work? → Use auto-registration (default)

#### **Persistent Dashboard Service**

Start the persistent dashboard once (survives application exits):

```bash
# Start dashboard service (runs until stopped)
python3 -m sollol.dashboard_service --port 8080 --redis-url redis://localhost:6379

# Or run in background
nohup python3 -m sollol.dashboard_service --port 8080 --redis-url redis://localhost:6379 > /tmp/dashboard_service.log 2>&1 &
```

**Features:**
- 📊 **Real-time metrics**: System status, latency, success rate, GPU memory, Ray workers
- 📜 **Live log streaming**: WebSocket-based log tailing (via Redis pub/sub)
- 🌐 **Activity monitoring**: Ollama server and llama.cpp RPC activity
- 🔷 **Embedded Ray dashboard**: Task-level distributed tracing
- 📈 **Embedded Dask dashboard**: Performance profiling and task graphs
- 🔍 **Auto-discovery**: Automatically discovers Ollama nodes and RPC backends when no router context

#### **Embedded Dashboard (Alternative)**

Applications can also start their own embedded dashboards:

```python
from sollol import run_unified_dashboard
import threading

# Start embedded dashboard with router context
dashboard_thread = threading.Thread(
    target=run_unified_dashboard,
    kwargs={
        "router": pool,  # Provides node/backend context
        "dashboard_port": 8080,
        "host": "0.0.0.0",
        "enable_dask": False
    },
    daemon=True
)
dashboard_thread.start()
```

**Environment Variables** (configure before initializing):

```bash
# Disable dashboard (default: true)
export SOLLOL_DASHBOARD=false

# Change dashboard port (default: 8080)
export SOLLOL_DASHBOARD_PORT=9090

# Disable Dask dashboard integration (default: true)
export SOLLOL_DASHBOARD_DASK=false
```

#### **Multi-Application Pattern** ✨

The persistent dashboard service enables multiple applications to share observability:

```bash
# Terminal 1: Start persistent dashboard
python3 -m sollol.dashboard_service --port 8080 --redis-url redis://localhost:6379

# Terminal 2: Start application 1
python my_app1.py  # Auto-registers with dashboard

# Terminal 3: Start application 2
python my_app2.py  # Also auto-registers

# Visit http://localhost:8080 to see both applications!
```

**Benefits:**
- Single dashboard for all SOLLOL-based applications
- Dashboard stays running when applications exit
- Aggregated logs from all applications (via Redis pub/sub)
- Centralized observability for distributed systems

#### **Programmatic Stats Access**

```python
# Get detailed stats
stats = pool.get_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Average latency: {stats['avg_latency_ms']}ms")
print(f"Success rate: {stats['success_rate']:.2%}")

# Per-node breakdown
for host, metrics in stats['hosts'].items():
    print(f"{host}: {metrics['latency_ms']}ms, {metrics['success_rate']:.2%}")
```

#### **Prometheus Metrics**

```bash
# Prometheus metrics endpoint
curl http://localhost:9090/metrics

# sollol_requests_total{host="gpu-1:11434",model="llama3.2"} 1234
# sollol_latency_seconds{host="gpu-1:11434"} 0.234
# sollol_success_rate{host="gpu-1:11434"} 0.98
```

---

## 🔌 Integration Examples

### 🔗 Integration with SynapticLlamas & FlockParser

SOLLOL is the **distributed inference platform** for the complete AI ecosystem, powering both **[SynapticLlamas](https://github.com/BenevolentJoker-JohnL/SynapticLlamas)** (multi-agent orchestration) and **[FlockParser](https://github.com/BenevolentJoker-JohnL/FlockParser)** (document RAG).

### **The Complete Stack**

```
┌─────────────────────────────────────────────────────────────┐
│              SynapticLlamas (v0.1.0+)                       │
│          Multi-Agent System & Orchestration                 │
│  • Research agents  • Editor agents  • Storyteller agents  │
└───────────┬────────────────────────────────────┬───────────┘
            │                                    │
            │ RAG Queries                        │ Distributed
            │ (with pre-computed embeddings)     │ Inference
            │                                    │
     ┌──────▼──────────┐              ┌─────────▼────────────┐
     │  FlockParser    │              │      SOLLOL          │
     │  API (v1.0.4+)  │              │  Load Balancer       │
     │  Port: 8000     │              │  (v0.9.31+)          │
     └─────────────────┘              └──────────────────────┘
            │                                    │
            │ ChromaDB                          │ Intelligent
            │ Vector Store                      │ GPU/CPU Routing
            │                                    │
     ┌──────▼──────────┐              ┌─────────▼────────────┐
     │  Knowledge Base │              │  Ollama Nodes        │
     │  41 Documents   │              │  (Distributed)       │
     │  6,141 Chunks   │              │  GPU + CPU           │
     └─────────────────┘              └──────────────────────┘
```

### **Why This Integration Matters**

| Component | Role | Key Feature |
|-----------|------|-------------|
| **SOLLOL** | Distributed Inference | Intelligent GPU/CPU routing with load balancing |
| **SynapticLlamas** | Multi-Agent Orchestration | Research, Editor, Storyteller agents |
| **FlockParser** | Document RAG & Knowledge Base | ChromaDB vector store with 6,141+ chunks |

### **Quick Start: Complete Ecosystem**

```bash
# Install all three packages (auto-installs dependencies)
pip install synaptic-llamas  # Pulls in flockparser>=1.0.4 and sollol>=0.9.31

# Start FlockParser API
flockparse

# Run SynapticLlamas with SOLLOL + FlockParser integration
synaptic-llamas --interactive --distributed
```

### **Integration Example: Load Balanced RAG**

```python
from sollol import OllamaPool
from flockparser_adapter import FlockParserAdapter

# Initialize SOLLOL for distributed inference
sollol = OllamaPool.auto_configure()

# Initialize FlockParser adapter
flockparser = FlockParserAdapter("http://localhost:8000", remote_mode=True)

# Step 1: Generate embedding using SOLLOL (load balanced!)
user_query = "What does research say about quantum entanglement?"
embedding = sollol.embed(
    model="mxbai-embed-large",
    input=user_query
)
# SOLLOL routes to fastest GPU automatically

# Step 2: Query FlockParser with pre-computed embedding
rag_results = flockparser.query_remote(
    query=user_query,
    embedding=embedding,  # Skip FlockParser's embedding generation
    n_results=5
)
# FlockParser returns relevant chunks from 41 documents

# Performance gain: 2-5x faster when SOLLOL has faster nodes!
```

### **Production Integrations**

**SOLLOL is actively used in production by:**

- **[FlockParser](https://github.com/BenevolentJoker-JohnL/FlockParser)** - Document RAG Intelligence with distributed processing. FlockParser's legacy load balancing code was refactored and became core SOLLOL logic. FlockParser now uses SOLLOL directly via `OllamaPool` for intelligent routing across document embeddings and LLM queries.

- **[SynapticLlamas](https://github.com/BenevolentJoker-JohnL/SynapticLlamas)** - Multi-agent collaborative research framework. Uses SOLLOL's `HybridRouter` for distributed agent execution with RAG-enhanced research capabilities via FlockParser integration.

**Related Projects:**
- **[SynapticLlamas](https://github.com/BenevolentJoker-JohnL/SynapticLlamas)** - Multi-Agent Orchestration
- **[FlockParser](https://github.com/BenevolentJoker-JohnL/FlockParser)** - Document RAG Intelligence

**📹 Want to see some of the original core logic?**

FlockParser was developed *before* SOLLOL and formed the early load balancing logic. You can find the legacy repository with a condensed 71-second demo here: **[FlockParser-legacy](https://github.com/BenevolentJoker-JohnL/FlockParser-legacy)**

---

### SynapticLlamas Integration

```python
from sollol import SOLLOL, SOLLOLConfig
from synaptic_llamas import AgentOrchestrator

# Setup SOLLOL for multi-agent orchestration
config = SOLLOLConfig.auto_discover()
sollol = SOLLOL(config)
sollol.start(blocking=False)

# SynapticLlamas now uses SOLLOL for intelligent routing
orchestrator = AgentOrchestrator(
    llm_endpoint="http://localhost:8000/api/chat"
)

# All agents automatically distributed and optimized
orchestrator.run_parallel_agents([...])
```

### FlockParser Integration

```python
from sollol import OllamaPool

# FlockParser uses SOLLOL's OllamaPool directly
pool = OllamaPool(
    nodes=None,  # Auto-discover all Ollama nodes
    enable_intelligent_routing=True,
    exclude_localhost=True,
    discover_all_nodes=True,
    app_name="FlockParser",
    enable_ray=True
)

# All FlockParser document embeddings and queries route through SOLLOL
embeddings = pool.embed(model="mxbai-embed-large", input="document text")
response = pool.chat(model="llama3.2", messages=[{"role": "user", "content": "query"}])
```

### LangChain Integration

```python
from langchain.llms import Ollama
from sollol import OllamaPool

# Use SOLLOL as LangChain backend
pool = OllamaPool.auto_configure()

llm = Ollama(
    base_url="http://localhost:8000",
    model="llama3.2"
)

# LangChain requests now go through SOLLOL
response = llm("What is quantum computing?")
```

---

## 🏭 Production Deployment (Bare Metal)

For teams preferring bare metal infrastructure over containers, SOLLOL provides systemd-based deployment for production environments.

### **Multi-Node Bare Metal Setup**

This setup assumes you have 3+ physical machines with Ollama installed. We'll configure SOLLOL as a centralized routing layer.

#### **Architecture:**
```
┌─────────────────────────────────────────┐
│   Central Router Machine (Control Plane│
│   - SOLLOL Dashboard (port 8080)       │
│   - Redis (port 6379)                  │
│   - Optional: GPU reporter             │
└────────────┬────────────────────────────┘
             │ Auto-discovery via network
             │ scan (ports 11434)
     ┌───────┼──────────┬─────────────┐
     ▼       ▼          ▼             ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ Node 1  │ │ Node 2  │ │ Node 3  │ │ Node N  │
│ Ollama  │ │ Ollama  │ │ Ollama  │ │ Ollama  │
│ :11434  │ │ :11434  │ │ :11434  │ │ :11434  │
│ GPU 24GB│ │ GPU 16GB│ │ CPU 64c │ │ ...     │
└─────────┘ └─────────┘ └─────────┘ └─────────┘
```

#### **Step 1: Install Ollama on each node**

On each worker node (Node 1, 2, 3, ...):

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
sudo systemctl enable ollama
sudo systemctl start ollama

# Verify it's running
curl http://localhost:11434/api/tags
```

#### **Step 2: Install SOLLOL on control plane machine**

On your central router machine:

```bash
# Install SOLLOL and dependencies
pip install sollol redis

# Install Redis
sudo apt-get install redis-server  # Ubuntu/Debian
# OR
sudo yum install redis              # RHEL/CentOS

# Start Redis
sudo systemctl enable redis
sudo systemctl start redis
```

#### **Step 3: Create systemd service for SOLLOL Dashboard**

Create `/etc/systemd/system/sollol-dashboard.service`:

```ini
[Unit]
Description=SOLLOL Dashboard Service
After=network.target redis.service
Requires=redis.service

[Service]
Type=simple
User=sollol  # Create dedicated user for security
Group=sollol
WorkingDirectory=/opt/sollol
Environment="SOLLOL_DASHBOARD=true"
Environment="SOLLOL_DASHBOARD_PORT=8080"
Environment="REDIS_URL=redis://localhost:6379"
ExecStart=/usr/bin/python3 -m sollol.dashboard_service --port 8080 --redis-url redis://localhost:6379
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo useradd -r -s /bin/false sollol  # Create dedicated user
sudo mkdir -p /opt/sollol
sudo chown sollol:sollol /opt/sollol

sudo systemctl daemon-reload
sudo systemctl enable sollol-dashboard
sudo systemctl start sollol-dashboard

# Verify
sudo systemctl status sollol-dashboard
curl http://localhost:8080/health
```

#### **Step 4: Install GPU reporters on nodes (optional but recommended)**

On each GPU node for accurate VRAM monitoring:

```bash
# Install on each node with GPUs
pip install sollol gpustat

# Run GPU reporter (publishes to central Redis)
sollol install-gpu-reporter --redis-host <control-plane-ip>

# Example for node at 192.168.1.22
sollol install-gpu-reporter --redis-host 192.168.1.10
```

Create `/etc/systemd/system/sollol-gpu-reporter.service` on each GPU node:

```ini
[Unit]
Description=SOLLOL GPU Reporter
After=network.target

[Service]
Type=simple
User=sollol
ExecStart=/usr/local/bin/sollol-gpu-reporter --redis-host <control-plane-ip> --interval 5
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

#### **Step 5: Configure firewall rules**

On all nodes:

```bash
# Allow Ollama traffic (port 11434)
sudo ufw allow 11434/tcp comment "Ollama API"

# On control plane only: allow dashboard access
sudo ufw allow 8080/tcp comment "SOLLOL Dashboard"
sudo ufw allow 6379/tcp comment "Redis"  # Only from trusted nodes

# Reload firewall
sudo ufw reload
```

#### **Step 6: Test the deployment**

From any machine with network access:

```python
from sollol import OllamaPool

# SOLLOL auto-discovers all nodes via network scan
pool = OllamaPool.auto_configure()

# Verify nodes discovered
stats = pool.get_stats()
print(f"Discovered {stats['active_nodes']} nodes")

# Make a test request
response = pool.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response['message']['content'])
print(f"Routed to: {response['_sollol_routing']['host']}")
```

#### **Step 7: Monitor with systemd**

```bash
# Check dashboard status
sudo systemctl status sollol-dashboard

# View live logs
sudo journalctl -u sollol-dashboard -f

# Check GPU reporters
sudo systemctl status sollol-gpu-reporter

# View metrics
curl http://localhost:8080/api/stats | jq
```

### **Production Hardening**

#### **Security:**
```bash
# 1. Run SOLLOL as dedicated unprivileged user
sudo useradd -r -s /bin/false sollol

# 2. Configure Redis authentication
sudo vi /etc/redis/redis.conf
# Add: requirepass <strong-password>

# 3. Use firewall to restrict access
sudo ufw allow from 10.9.66.0/24 to any port 6379  # Redis from trusted subnet only
sudo ufw allow from 10.9.66.0/24 to any port 8080  # Dashboard from trusted subnet
```

#### **High Availability:**
```bash
# Use systemd watchdog for automatic restart on crashes
[Service]
WatchdogSec=30
Restart=always
RestartSec=10
```

#### **Monitoring:**
```bash
# Integrate with Prometheus
curl http://localhost:9090/metrics

# Or use systemd monitoring
systemctl status sollol-dashboard | grep "Active:"
```

### **Troubleshooting**

**Nodes not discovered:**
```bash
# Check network connectivity
for ip in 10.9.66.{1..255}; do
    timeout 0.5 bash -c "cat < /dev/null > /dev/tcp/$ip/11434 2>/dev/null" && echo "$ip:11434 reachable"
done

# Check Ollama is listening on all interfaces (not just localhost)
curl http://<node-ip>:11434/api/tags
```

**Dashboard not starting:**
```bash
# Check Redis is running
systemctl status redis
redis-cli ping  # Should return "PONG"

# Check port not in use
sudo lsof -i :8080

# View detailed logs
journalctl -u sollol-dashboard --since "10 minutes ago"
```

**Performance issues:**
```bash
# Check node health
curl http://localhost:8080/api/stats | jq '.node_performance'

# Monitor resource usage
htop
nvidia-smi  # On GPU nodes

# Check network latency between nodes
ping <node-ip>
```

---

## 📚 Documentation

### Getting Started

- **[Installation Guide](INSTALLATION.md)** - Complete setup for bare-metal deployment
- **[Quick Start](QUICK_START.md)** - Get up and running in 3 commands
- **[Configuration](CONFIGURATION.md)** - All configuration options
- **[Architecture](ARCHITECTURE.md)** - System architecture overview
- **[Contributing](CONTRIBUTING.md)** - Contribution guidelines
- **[Security](SECURITY.md)** - Security policies

### Complete Documentation

See **[docs/](docs/)** for detailed documentation organized by category:

- **[Setup Guides](docs/setup/)** - Ray, Redis, GPU monitoring, Grafana
- **[Features](docs/features/)** - Backends, batch processing, routing, dashboard
- **[Architecture](docs/architecture/)** - Remote coordinator, multi-app coordination
- **[Integration](docs/integration/)** - Code examples and walkthroughs
- **[Benchmarks](docs/benchmarks/)** - Performance testing and results
- **[Troubleshooting](docs/troubleshooting/)** - Known issues and fixes

### Key Guides

- **[Remote Coordinator Execution](docs/architecture/remote-coordinator.md)** - Intelligent coordinator placement
- **[Ray Cluster Setup](docs/setup/ray-cluster.md)** - Configure distributed execution
- **[GPU Monitoring](docs/setup/gpu-monitoring-setup.md)** - GPU metrics and monitoring
- **[Batch Processing API](docs/features/batch-processing.md)** - Batch job management
- **[Routing Strategies](docs/features/routing.md)** - Intelligent routing algorithms

---

## 🆕 What's New in v0.7.0

### 📦 Batch Processing API
Complete RESTful API for asynchronous batch job management. Submit large-scale batch operations (embeddings, bulk inference) and track progress via job IDs.

```python
import requests

# Submit batch embedding job (up to 10,000 documents)
response = requests.post("http://localhost:11434/api/batch/embed", json={
    "model": "nomic-embed-text",
    "documents": ["doc1", "doc2", ...],  # Thousands of documents
})
job_id = response.json()["job_id"]

# Check status
status = requests.get(f"http://localhost:11434/api/batch/jobs/{job_id}")
print(status.json()["progress"]["percent"])  # 100.0

# Get results
results = requests.get(f"http://localhost:11434/api/batch/results/{job_id}")
embeddings = results.json()["results"]
```

**Batch API Endpoints:**
- `POST /api/batch/embed` - Submit batch embedding job
- `GET /api/batch/jobs/{job_id}` - Get job status with progress tracking
- `GET /api/batch/results/{job_id}` - Retrieve job results and errors
- `DELETE /api/batch/jobs/{job_id}` - Cancel running jobs
- `GET /api/batch/jobs?limit=100` - List recent jobs

**Features:**
- UUID-based job tracking with 5 states (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)
- Automatic TTL-based cleanup (1 hour default)
- Progress tracking: completed_items, failed_items, percentage
- Duration calculation and metadata storage
- Async job execution via Dask distributed processing

---

## ⚡ Performance Optimizations (v0.9.18+)

SOLLOL now includes **8 production-grade performance optimizations** designed to improve throughput and latency:

**⚠️ Transparency Note:** These features are implemented and functional, but claimed performance improvements are projections based on architecture, NOT independently validated benchmarks. See [Performance Impact](#-performance-impact) section below for details.

### 🚀 Response Caching Layer
**Expected Impact:** Reduces latency for repeated queries (cache hit/miss tracking validated)

Intelligent LRU cache with TTL expiration:
```python
from sollol import OllamaPool

# Enable response caching (enabled by default)
pool = OllamaPool.auto_configure(
    enable_cache=True,
    cache_max_size=1000,  # Cache up to 1000 responses
    cache_ttl=3600        # 1 hour TTL
)

# First request: normal latency
response1 = pool.embed(model="mxbai-embed-large", input="Hello world")

# Cached request: faster
response2 = pool.embed(model="mxbai-embed-large", input="Hello world")  # Cache hit

# Programmatic cache management
pool.clear_cache()                              # Clear all
pool.invalidate_cache_by_model("llama3.2")     # Invalidate by model
cache_data = pool.export_cache()                # Export for persistence
pool.import_cache(cache_data)                   # Restore from export

# Get cache stats
stats = pool.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")    # 85.2%
print(f"Cache size: {stats['size']}")           # 234/1000
```

### 🌊 Streaming Support
**Expected Impact:** Better UX, reduced perceived latency (streaming functionality validated)

Token-by-token streaming for `chat()` and `generate()`:
```python
# Stream chat responses
for chunk in pool.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
):
    content = chunk.get("message", {}).get("content", "")
    print(content, end="", flush=True)

# Stream text generation
for chunk in pool.generate(
    model="llama3.2",
    prompt="Explain quantum computing",
    stream=True
):
    print(chunk.get("response", ""), end="", flush=True)
```

### 🔥 Smart Model Prefetching
**Expected Impact:** 1-5 seconds reduced first-request latency (projection, not measured)

Pre-load models into VRAM before first use:
```python
# Warm a single model
pool.warm_model("llama3.2")

# Warm multiple models in parallel
results = pool.warm_models(
    models=["llama3.2", "codellama", "mistral"],
    parallel=True
)
print(f"Warmed {sum(results.values())} models")
```

### ⚡ Async I/O Support
**Expected Impact:** 2-3x throughput for concurrent requests (projection, not measured)

True non-blocking I/O with httpx AsyncClient:
```python
import asyncio

# Async methods for concurrent requests
async def process_batch():
    responses = await asyncio.gather(
        pool.chat_async("llama3.2", messages=[...]),
        pool.generate_async("llama3.2", prompt="..."),
        pool.embed_async("mxbai-embed-large", input="...")
    )
    return responses

# Run async batch
results = asyncio.run(process_batch())
```

### 🔗 HTTP/2 Multiplexing
**Expected Impact:** 30-50% latency reduction for concurrent requests (projection, not measured)

Automatic HTTP/2 support when `httpx` is installed:
```python
# Automatically uses HTTP/2 if available
pool = OllamaPool.auto_configure()

# Check if HTTP/2 is enabled
stats = pool.get_stats()
print(f"HTTP/2 enabled: {stats['http2_enabled']}")  # True
```

### 📊 Additional Optimizations

**Connection Pool Tuning** (10-20% better concurrency):
- Optimized pool sizes: 10-20 connections per node
- Automatic retry with exponential backoff
- Connection reuse with keep-alive

**Adaptive Health Checks** (5-10% overhead reduction):
- Dynamic intervals based on node stability:
  - Very stable (<1% failures): 60s interval
  - Stable (<5% failures): 30s interval
  - Degraded (5-15% failures): 15s interval
  - Unstable (>15% failures): 5s interval

**Telemetry Sampling** (~90% overhead reduction):
- Configurable sampling for info-level events (default: 10%)
- Always logs errors and critical events
- Reduces dashboard logging overhead

### 📈 Performance Impact

**⚠️ IMPORTANT: These are architectural projections, NOT measured results**

These optimizations are implemented and functional, but multi-node performance gains have not been independently validated:

**Projected improvements (unvalidated):**
- **Throughput:** +150-300% for concurrent workloads (theory: parallel request handling)
- **Latency:** -40-70% for typical requests (theory: caching + HTTP/2)
- **Cache hits:** Significant latency reduction for repeated queries (validated in single-node tests)

**What's actually measured:**
- ✅ Response caching works (cache hit/miss rates tracked)
- ✅ Streaming works (token-by-token delivery confirmed)
- ✅ HTTP/2 enabled (httpx connection verified)
- ⚠️ Multi-node throughput gains: Not independently benchmarked

**To validate these claims yourself:**
```bash
# Run comparative benchmarks
cd benchmarks
python run_benchmarks.py --sollol-url http://localhost:8000 --duration 120
```

See [BENCHMARKING.md](BENCHMARKING.md) for methodology.

---

### Previous Features (v0.3.6+)

**Synchronous API** - No async/await required:
```python
from sollol.sync_wrapper import OllamaPool
pool = OllamaPool.auto_configure()
response = pool.chat(...)  # Synchronous call
```

**Priority Helpers** - Semantic priority levels:
```python
from sollol.priority_helpers import Priority
priority = Priority.HIGH  # 7
```

**SOLLOL Detection:**
- `X-Powered-By: SOLLOL` header on all responses
- `/api/health` endpoint returns `{"service": "SOLLOL", "version": "0.7.0"}`

---

## 🆚 Comparison

### SOLLOL vs. Simple Load Balancers

| Feature | nginx/HAProxy | SOLLOL |
|---------|--------------|---------|
| Routing | Round-robin/random | Context-aware, adapts from history |
| Resource awareness | None | GPU/CPU/memory-aware |
| Failover | Manual config | Automatic detection & recovery |
| Distributed Inference | ❌ | ✅ llama.cpp RPC integration |
| Task prioritization | ❌ | ✅ Priority queue |
| Observability | Basic | Rich metrics + dashboard |
| Setup | Complex config | Auto-discover |

### SOLLOL vs. Kubernetes

| Feature | Kubernetes | SOLLOL |
|---------|-----------|---------|
| **Complexity** | High - requires cluster setup | Low - pip install |
| **AI-specific** | Generic container orchestration | Purpose-built for LLMs |
| **Intelligence** | None | Task-aware routing |
| **Distributed Inference** | Manual | Automatic via llama.cpp RPC |
| **Best for** | Large-scale production | AI-focused teams |

**Use both!** Deploy SOLLOL on Kubernetes for ultimate scalability.

---

## 🤝 Contributing

We welcome contributions! Areas we'd love help with:

- ML-based routing predictions
- Additional monitoring integrations
- Cloud provider integrations
- Performance optimizations
- Documentation improvements

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Credits

Created by [BenevolentJoker-JohnL](https://github.com/BenevolentJoker-JohnL)

**Part of the Complete AI Ecosystem:**
- **[SynapticLlamas](https://github.com/BenevolentJoker-JohnL/SynapticLlamas)** - Multi-Agent Orchestration
- **[FlockParser](https://github.com/BenevolentJoker-JohnL/FlockParser)** - Document RAG Intelligence
- **[SOLLOL](https://github.com/BenevolentJoker-JohnL/SOLLOL)** - Distributed Inference Platform (this project)

**Special Thanks:**
- **[Dallan Loomis](https://github.com/DallanL)** - For always providing invaluable support, feedback, and guidance throughout development. Your insights and encouragement have been instrumental in shaping this project.

Built with: Ray, Dask, FastAPI, llama.cpp, Ollama

---

## 🎯 What Makes SOLLOL Different?

1. **Combines task distribution AND distributed inference** in one system
2. **Context-aware routing** that adapts based on performance metrics
3. **Auto-discovery** of nodes with minimal configuration
4. **Built-in failover** and priority queuing
5. **Purpose-built for Ollama clusters** (understands GPU requirements, task types)

**Limitations to know**:
- Distributed inference verified with 13B models; larger models not extensively tested
- Performance benefits depend on network latency and workload patterns
- Not a drop-in replacement for single-node setups in all scenarios

---

<div align="center">

**Stop manually managing your LLM cluster. Let SOLLOL optimize it for you.**

[Get Started](#quick-start) • [View on GitHub](https://github.com/BenevolentJoker-JohnL/SOLLOL) • [Report Issue](https://github.com/BenevolentJoker-JohnL/SOLLOL/issues)

</div>
