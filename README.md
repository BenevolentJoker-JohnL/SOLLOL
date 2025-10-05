# SOLLOL - Hybrid Cluster Orchestrator for Local LLMs

**The first open-source orchestration layer that unifies task routing and distributed model inference for local LLM clusters.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SOLLOL (Super Ollama Load balancer & Orchestration Layer) is a production-ready cluster orchestrator designed specifically for local LLM deployments. It intelligently manages both **task-level parallelism** (distributing agent tasks across nodes) and **model-level parallelism** (sharding large models across RPC backends).

## ✅ What's Actually Tested

**Verified working:**
- ✅ 13B models across 2-3 RPC backends
- ✅ GGUF extraction from Ollama blob storage
- ✅ Automatic layer distribution visible in coordinator logs
- ✅ Real-time dashboard monitoring
- ✅ Auto-discovery of RPC backends

**Should work (not extensively tested):**
- ⚠️ 70B+ models across 4+ backends
- ⚠️ Larger models with sufficient nodes

**Performance characteristics:**
- ⚠️ Startup time: 2-5 minutes for 13B (vs ~20s local)
- ⚠️ Inference speed: ~5 tok/s distributed vs ~20 tok/s local
- ⚠️ Worth it when model doesn't fit on single machine

## Features

### 🚀 Core Features
- **Intelligent Load Balancing**: Adaptive routing based on node performance, GPU availability, and task complexity
- **Auto-Discovery**: Automatic detection of Ollama nodes and RPC backends on your network
- **Connection Pooling**: Efficient connection management with health monitoring
- **Request Hedging**: Duplicate requests to multiple nodes for lower latency
- **Task Prioritization**: Priority-based request queuing

### 🔗 Distribution Modes

SOLLOL supports **two independent distribution modes** that can be used together or separately:

#### 1. Task Distribution (Multi-Agent Parallel Execution)
- **Load Balancing**: Distribute multiple agent requests across Ollama nodes in parallel
- **Connection Pooling**: Efficient connection management across nodes
- **Performance Learning**: Adapts routing based on historical performance
- **Use Case**: Speed up queries by running multiple agents simultaneously on different nodes

#### 2. Model Sharding (Layer-Level Distribution for Larger Models)
- **Hybrid Routing**: Routes to llama.cpp when distributed mode enabled
- **RPC Backend Support**: Connect to llama.cpp RPC servers for layer-level model sharding
- **GGUF Auto-Resolution**: Automatically extracts GGUFs from Ollama blob storage
- **Auto-Discovery**: Discovers RPC backends on your network
- **Use Case**: Run models that don't fit on a single machine by distributing layers across nodes (verified with 13B across 2-3 nodes)

**💡 Enable BOTH modes** to get task distribution for small models AND model sharding for large models!

### 📊 Monitoring & Observability
- **Real-time Metrics**: Track performance, latency, and node health
- **Web Dashboard**: Monitor routing decisions and backend status
- **Performance Learning**: Adapts routing based on historical performance

## Installation

### From PyPI (when published)
```bash
pip install sollol
```

### From Source
```bash
git clone https://github.com/BenevolentJoker-JohnL/SynapticLlamas.git
cd SynapticLlamas/sollol
pip install -e .
```

## Quick Start

### Basic Usage

```python
from sollol import OllamaPool

# Auto-discover Ollama nodes and create pool
pool = OllamaPool.auto_configure()

# Make a chat request
response = pool.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response)
```

### With Model Sharding (Large Model Distribution)

```python
from sollol import HybridRouter, OllamaPool
from sollol.rpc_discovery import auto_discover_rpc_backends

# Discover RPC backends for model sharding
rpc_backends = auto_discover_rpc_backends()

# Create hybrid router with model sharding enabled
router = HybridRouter(
    ollama_pool=OllamaPool.auto_configure(),
    rpc_backends=rpc_backends,
    enable_distributed=True  # Enables model sharding via llama.cpp
)

# Routes based on configuration: when distributed mode enabled → llama.cpp sharding
response = await router.route_request(
    model="codellama:13b",  # Uses model sharding across RPC backends
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
```

### Auto-Discovery

```python
from sollol.discovery import discover_ollama_nodes
from sollol.rpc_discovery import auto_discover_rpc_backends

# Discover Ollama nodes (for task distribution / load balancing)
ollama_nodes = discover_ollama_nodes()
print(f"Found {len(ollama_nodes)} Ollama nodes for task distribution")

# Discover RPC backends (for model sharding of large models)
rpc_backends = auto_discover_rpc_backends()
print(f"Found {len(rpc_backends)} RPC backends for model sharding")
```

## Configuration

### OllamaPool Options

```python
from sollol import OllamaPool

pool = OllamaPool(
    nodes=[
        {"host": "10.9.66.154", "port": "11434"},
        {"host": "10.9.66.157", "port": "11434"}
    ],
    enable_intelligent_routing=True,  # Use smart routing
    exclude_localhost=False  # Include localhost in discovery
)
```

### HybridRouter Options

```python
from sollol import HybridRouter

router = HybridRouter(
    ollama_pool=pool,
    rpc_backends=[
        {"host": "192.168.1.10", "port": 50052},
        {"host": "192.168.1.11", "port": 50052}
    ],
    coordinator_host="127.0.0.1",
    coordinator_port=8080,
    enable_distributed=True,
    auto_discover_rpc=True  # Auto-discover RPC backends
)
```

## Model Sharding Setup (Distributed Inference for Large Models)

**Note**: This section is about **Model Sharding** - distributing a single large model across multiple RPC backends. For **Task Distribution** (load balancing multiple agent requests across Ollama nodes), simply use OllamaPool with multiple nodes.

**💡 You can enable BOTH modes simultaneously** - task distribution for small models (Ollama pool) AND model sharding for large models (llama.cpp RPC)!

### Option 1: Zero-Config Auto-Setup (Easiest!)

SOLLOL can automatically setup llama.cpp RPC backends for you:

```python
from sollol import HybridRouter, OllamaPool

# Everything auto-configures AND auto-setups model sharding!
router = HybridRouter(
    ollama_pool=OllamaPool.auto_configure(),
    enable_distributed=True,  # Enable model sharding for large models
    auto_discover_rpc=True,   # Discover existing RPC servers
    auto_setup_rpc=True,      # Auto-build & start RPC servers if none found
    num_rpc_backends=2        # Number of RPC backends to start
)

# SOLLOL will automatically:
# 1. Check for running RPC servers on network
# 2. If none found, clone llama.cpp
# 3. Build with RPC support
# 4. Start RPC server processes
# 5. Configure hybrid routing (small → Ollama, large → llama.cpp sharding)

# Use it immediately! Model shards across RPC backends when distributed mode enabled
response = await router.route_request(
    model="codellama:13b",  # Model distributed across RPC backends
    messages=[{"role": "user", "content": "Hello!"}]
)
```

Or use the standalone auto-setup:

```python
from sollol import auto_setup_rpc_backends

# Automatically setup RPC backends
backends = auto_setup_rpc_backends(
    num_backends=2,      # Start 2 RPC servers
    auto_build=True      # Build llama.cpp if needed
)
print(f"RPC backends ready: {backends}")
# Output: [{'host': '127.0.0.1', 'port': 50052}, {'host': '127.0.0.1', 'port': 50053}]
```

### Option 2: Manual Setup (Full Control)

#### 1. Start RPC Servers (Worker Nodes)

**Option A: Production (Systemd Service - Recommended)**
```bash
# One command setup: clone + build + install as systemd service
pip install sollol
python3 -m sollol.setup_llama_cpp --all

# Service runs automatically on boot and restarts on failure
# Manage with systemctl:
systemctl --user status sollol-rpc-server
systemctl --user restart sollol-rpc-server
systemctl --user stop sollol-rpc-server
```

**Option B: Manual/Development**
```bash
# Build llama.cpp with RPC support
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_RPC=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j$(nproc)

# Start RPC server (blocks terminal)
./build/bin/rpc-server --host 0.0.0.0 --port 50052
```

#### 2. Use SOLLOL with Auto-Discovery

```python
from sollol import HybridRouter, OllamaPool

# Everything auto-configures!
router = HybridRouter(
    ollama_pool=OllamaPool.auto_configure(),
    enable_distributed=True,  # Enable model sharding
    auto_discover_rpc=True    # Finds RPC servers automatically
)

# Use it - models shard across RPC backends when distributed mode enabled
response = await router.route_request(
    model="codellama:13b",  # Model sharded across network
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## API Reference

### OllamaPool

**Methods:**
- `chat(model, messages, priority=5, **kwargs)` - Chat completion
- `generate(model, prompt, priority=5, **kwargs)` - Text generation
- `embed(model, input, priority=5, **kwargs)` - Generate embeddings
- `get_stats()` - Get pool statistics
- `add_node(host, port)` - Add a node to the pool
- `remove_node(host, port)` - Remove a node

### HybridRouter

**Methods:**
- `route_request(model, messages, **kwargs)` - Route request to appropriate backend
- `should_use_distributed(model)` - Check if model should use distributed inference
- `get_stats()` - Get routing statistics

### Discovery & Auto-Setup

**Functions:**
- `discover_ollama_nodes(timeout=0.5)` - Discover Ollama nodes on the network
- `auto_discover_rpc_backends(port=50052)` - Discover existing llama.cpp RPC backends
- `auto_setup_rpc_backends(num_backends=1, auto_build=True)` - Auto-setup RPC backends (clone, build, start)
- `check_rpc_server(host, port, timeout=1.0)` - Check if RPC server is running

## Environment Variables

- `OLLAMA_HOST` - Default Ollama host (e.g., `http://localhost:11434`)
- `LLAMA_RPC_BACKENDS` - Comma-separated RPC backends (e.g., `192.168.1.10:50052,192.168.1.11:50052`)

## Performance & Distribution Modes

SOLLOL provides **two independent distribution modes** that can be used together or separately:

### Task Distribution (Load Balancing)
Distributes **multiple agent requests** in parallel across Ollama nodes:
- **Node Performance**: Routes requests to faster nodes
- **GPU Availability**: Prefers nodes with available GPU memory
- **Task Complexity**: Routes complex tasks to more capable nodes
- **Historical Performance**: Learns from past routing decisions
- **Use Case**: Speed up multi-agent queries by running agents in parallel

### Model Sharding (Layer-Level Distribution)
Distributes **a single model's layers** across multiple RPC backends via llama.cpp:
- **Configuration-Based Routing**: Routes to llama.cpp when distributed mode enabled
- **Layer Distribution**: Model layers split across RPC backends (e.g., 40 layers → ~13 per backend)
- **GGUF Auto-Extraction**: Automatically finds models in Ollama storage
- **Verified Testing**: 13B models across 2-3 nodes (should work with larger models but not extensively tested)
- **Use Case**: Run models that don't fit on one machine (trade-off: slower startup and inference)

## Integration with SynapticLlamas

SOLLOL is the load balancing engine that powers [SynapticLlamas](https://github.com/BenevolentJoker-JohnL/SynapticLlamas), a distributed multi-agent AI orchestration platform. While SOLLOL can be used standalone, SynapticLlamas adds:

- Multi-agent orchestration
- Collaborative workflows
- AST-based quality voting
- Interactive CLI
- Web dashboard

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Credits

Part of the [SynapticLlamas](https://github.com/BenevolentJoker-JohnL/SynapticLlamas) project by BenevolentJoker-JohnL.
