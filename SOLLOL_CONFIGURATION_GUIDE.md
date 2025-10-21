# SOLLOL Configuration Guide

**Complete guide to configuring SOLLOL for distributed task execution and performance optimization**

---

## Table of Contents

1. [Overview](#overview)
2. [Task Distribution vs Performance](#task-distribution-vs-performance)
3. [Auto-Discovery Configuration](#auto-discovery-configuration)
4. [Locality Awareness](#locality-awareness)
5. [Manual Configuration](#manual-configuration)
6. [Performance Tuning](#performance-tuning)
7. [Integration Examples](#integration-examples)
8. [Troubleshooting](#troubleshooting)

---

## Overview

SOLLOL provides **intelligent load balancing** with two key capabilities:

1. **Task Distribution**: Distribute work across multiple Ollama nodes
2. **Performance Optimization**: Avoid resource contention via locality awareness

### Key Features

✅ **Automatic Network Discovery** - Scans subnet for all Ollama nodes
✅ **Locality Awareness** - Detects same-machine nodes (prevents false parallelism)
✅ **Intelligent Routing** - Chooses parallel vs sequential based on topology
✅ **Zero Configuration** - Works out-of-the-box in distributed mode

---

## Task Distribution vs Performance

### When to Use Task Distribution

**Goal**: Parallelize work across multiple machines for faster completion

**Use Cases**:
- Multi-chunk document generation
- Batch inference across many prompts
- RAG search with multiple retrieval queries
- Parallel research tasks

**Requirements**:
- ✅ Multiple physical machines on network
- ✅ Each machine running Ollama
- ✅ Tasks can be parallelized (independent chunks)

**Expected Speedup**: ~1.5-2x with 2 machines, ~2-3x with 3+ machines

### When to Prioritize Performance (Avoid False Parallelism)

**Goal**: Prevent resource contention that slows down execution

**Scenarios to Avoid**:
- ❌ Multiple Ollama instances on **same machine, same port** (waste of resources)
- ❌ Multiple Ollama instances on **same machine, different ports** (CPU/memory contention)
- ❌ Parallel execution with **same GPU** (VRAM/compute contention)

**Performance Impact of False Parallelism**:
- **Same machine, CPU inference**: 50-100% SLOWER (context switching, memory bandwidth)
- **Same GPU**: 30-50% SLOWER (VRAM contention, compute serialization)

**SOLLOL automatically prevents this** by detecting locality and disabling parallel mode.

---

## Auto-Discovery Configuration

### Default Behavior (Distributed Mode)

When using SOLLOL in distributed mode, auto-discovery runs automatically:

```python
from node_registry import NodeRegistry

# Automatic discovery on initialization
registry = NodeRegistry(auto_discover=True)

# Or trigger manually
registry = NodeRegistry()
registry.discover_and_add_nodes()
```

**What happens**:
1. Scans entire local subnet (e.g., 10.9.66.0/24)
2. Tests port 11434 on all IPs (parallel scan with 100 workers)
3. Verifies Ollama API is running (`GET /api/tags`)
4. Returns deduplicated list (localhost → real IP)

**Discovery time**: ~500ms for /24 subnet (254 IPs)

### Configuration Options

```python
# Full network scan (default in distributed mode)
discovered = registry.discover_and_add_nodes(
    timeout=0.5  # Connection timeout per node (default: 0.5s)
)
```

**From SOLLOL directly**:

```python
from sollol.discovery import discover_ollama_nodes

# Full subnet scan
nodes = discover_ollama_nodes(
    timeout=0.5,              # Connection timeout
    exclude_localhost=False,  # Include localhost nodes
    auto_resolve_docker=True, # Resolve Docker IPs
    discover_all_nodes=True   # Full scan (not just fast mode)
)

# Result format:
# [{"host": "10.9.66.154", "port": "11434"},
#  {"host": "10.9.66.194", "port": "11434"}]
```

### Environment Variable Override

Set `OLLAMA_HOST` to add a specific node first:

```bash
export OLLAMA_HOST="http://specific-node:11434"
```

This node will be discovered FIRST (before network scan).

---

## Locality Awareness

### What Is Locality Awareness?

SOLLOL detects when multiple nodes are on the **same physical machine** and automatically disables parallel execution to prevent resource contention.

### How It Works

```python
from sollol.pool import OllamaPool

# Create pool with nodes
pool = OllamaPool(nodes=[
    {'host': '10.9.66.154', 'port': '11434'},
    {'host': '10.9.66.194', 'port': '11434'}
])

# Check locality
unique_hosts = pool.count_unique_physical_hosts()
should_parallel = pool.should_use_parallel_execution(num_tasks=3)

print(f"Unique physical machines: {unique_hosts}")
print(f"Parallel mode enabled: {should_parallel}")
```

**Locality detection logic**:

1. **Resolve hostnames to IPs**:
   - `localhost` → `127.0.0.1` → `10.9.66.154` (actual machine IP)
   - `10.9.66.154` → `10.9.66.154`

2. **Count unique IPs**:
   - `localhost:11434` + `10.9.66.154:11434` = **1 unique host** (same machine)
   - `10.9.66.154:11434` + `10.9.66.194:11434` = **2 unique hosts** (different machines)

3. **Enable parallel only if beneficial**:
   - 1 unique host: **Disable parallel** (resource contention)
   - 2+ unique hosts: **Enable parallel** (true distributed execution)

### Example Scenarios

#### Scenario 1: Same Machine (False Parallelism)

**Config**:
```json
{
  "nodes": [
    {"url": "http://localhost:11434"},
    {"url": "http://localhost:11435"}
  ]
}
```

**SOLLOL detection**:
```
Unique physical machines: 1
Parallel mode: DISABLED ❌
Reason: All nodes on same machine - resource contention will make it slower
```

**Performance**: Sequential execution is 50-100% FASTER

#### Scenario 2: Different Machines (True Parallelism)

**Config**:
```json
{
  "nodes": [
    {"url": "http://10.9.66.154:11434"},
    {"url": "http://10.9.66.194:11434"}
  ]
}
```

**SOLLOL detection**:
```
Unique physical machines: 2
Parallel mode: ENABLED ✅
Reason: Multiple machines available - true parallelism benefit
```

**Performance**: Parallel execution is ~1.8x FASTER

#### Scenario 3: localhost + Real IP (Duplicate Detection)

**Config**:
```json
{
  "nodes": [
    {"url": "http://localhost:11434"},
    {"url": "http://10.9.66.154:11434"}
  ]
}
```

**SOLLOL detection**:
```
🔍 Duplicate detected: localhost is same machine as 10.9.66.154
Unique physical machines: 1 (after deduplication)
Parallel mode: DISABLED ❌
```

**NodeRegistry behavior**: Automatically removes duplicate, keeps only real IP

---

## Manual Configuration

### When to Use Manual Configuration

Auto-discovery may not work in these scenarios:
- Ollama nodes on different subnets
- Nodes behind firewall/NAT
- VPN-connected nodes
- Custom ports (not 11434)

### Configuration File

**Location**: `~/.synapticllamas_nodes.json`

**Format**:
```json
{
  "nodes": [
    {
      "url": "http://10.9.66.154:11434",
      "name": "machine-1",
      "priority": 0
    },
    {
      "url": "http://10.9.66.194:11434",
      "name": "machine-2",
      "priority": 0
    },
    {
      "url": "http://remote.example.com:8080",
      "name": "cloud-node",
      "priority": 1
    }
  ]
}
```

**Fields**:
- `url` (required): Full URL including port
- `name` (optional): Friendly name for logging
- `priority` (optional): Higher priority nodes selected first (default: 0)

### Load Manual Configuration

```python
from node_registry import NodeRegistry

# Standard mode: Load from config file
registry = NodeRegistry()
registry.load_config("~/.synapticllamas_nodes.json")

# Distributed mode: Auto-discovery is PRIMARY
# Config file only used as fallback if discovery finds 0 nodes
```

### Add Nodes Programmatically

```python
from node_registry import NodeRegistry

registry = NodeRegistry()

# Add single node
registry.add_node(
    url="http://10.9.66.200:11434",
    name="additional-node",
    priority=0,
    auto_probe=True  # Automatically detect capabilities
)

# Save updated config
registry.save_config("~/.synapticllamas_nodes.json")
```

---

## Performance Tuning

### 1. Optimize Discovery Timeout

**Default**: 0.5s per node

**Tune for your network**:

```python
# Fast local network (data center)
registry.discover_and_add_nodes(timeout=0.2)  # 200ms

# Slow network (WiFi, cloud)
registry.discover_and_add_nodes(timeout=1.0)  # 1 second

# Very slow network (VPN, intercontinental)
registry.discover_and_add_nodes(timeout=2.0)  # 2 seconds
```

**Trade-off**: Lower timeout = faster scan, but may miss slow-responding nodes

### 2. Control Parallel Execution Threshold

```python
from sollol.pool import OllamaPool

pool = OllamaPool(nodes=discovered_nodes)

# Check if parallel makes sense for your task
num_tasks = 5  # How many chunks/prompts to process

should_parallel = pool.should_use_parallel_execution(num_tasks)

if should_parallel:
    # Use parallel execution (Ray, Dask, ThreadPool)
    execute_parallel(tasks, pool)
else:
    # Use sequential execution (faster for same-machine)
    execute_sequential(tasks, pool)
```

**SOLLOL's decision logic**:
- `num_tasks < 2`: Sequential (nothing to parallelize)
- `unique_hosts < 2`: Sequential (resource contention)
- `unique_hosts >= 2`: Parallel (true distributed benefit)

### 3. Node Priority for Load Balancing

Use priority to prefer certain nodes:

```python
# High-performance GPU node (priority 0 = highest)
registry.add_node("http://gpu-node:11434", priority=0)

# Medium-performance nodes
registry.add_node("http://node-1:11434", priority=1)
registry.add_node("http://node-2:11434", priority=1)

# Low-performance fallback (priority 2 = lowest)
registry.add_node("http://slow-node:11434", priority=2)
```

**Routing behavior**: SOLLOL selects lowest priority number first (when available)

### 4. Disable Auto-Discovery (Performance Critical)

If you need predictable startup time and have a stable config:

```python
# Skip auto-discovery, use config file only
registry = NodeRegistry(auto_discover=False)
registry.load_config("~/.synapticllamas_nodes.json")
```

**Use case**: Production deployments with fixed infrastructure

---

## Integration Examples

### Example 1: SynapticLlamas (Automatic)

**File**: `/home/joker/SynapticLlamas/main.py`

```python
# Distributed mode: Auto-discovery PRIMARY
if current_mode == "distributed":
    # Auto-discover all nodes on network
    discovered_count = global_registry.discover_and_add_nodes()

    if discovered_count > 0:
        # Save discovered nodes
        global_registry.save_config(NODES_CONFIG_PATH)
    else:
        # Fallback to config file
        if os.path.exists(NODES_CONFIG_PATH):
            global_registry.load_config(NODES_CONFIG_PATH)

# Check locality and enable parallel mode
if len(global_registry.nodes) > 1:
    unique_ips = count_unique_ips(global_registry.nodes)

    if unique_ips >= 2:
        logger.info("✅ Parallel mode ENABLED (multi-machine)")
    else:
        logger.info("ℹ️  Parallel mode DISABLED (same machine)")
```

### Example 2: Custom Application (Manual)

```python
from sollol.pool import OllamaPool
from sollol.discovery import discover_ollama_nodes

# Discover nodes
nodes = discover_ollama_nodes(discover_all_nodes=True)

# Create pool
pool = OllamaPool(
    nodes=nodes,
    app_name="My Application",
    register_with_dashboard=True  # Enable monitoring
)

# Check if parallel execution makes sense
if pool.should_use_parallel_execution(num_tasks=10):
    print("Using parallel execution across multiple machines")
    # Your parallel execution logic here
else:
    print("Using sequential execution (same machine or few tasks)")
    # Your sequential execution logic here
```

### Example 3: Hybrid Routing (RPC + Ollama)

```python
from sollol.ray_hybrid_router import RayHybridRouter
from sollol.pool import OllamaPool

# Discover Ollama nodes for task distribution
ollama_nodes = discover_ollama_nodes(discover_all_nodes=True)
ollama_pool = OllamaPool(nodes=ollama_nodes)

# Configure RPC backends for model sharding
rpc_backends = [
    {"host": "10.9.66.154", "port": 50052},
    {"host": "10.9.66.194", "port": 50052}
]

# Create hybrid router
router = RayHybridRouter(
    ollama_pool=ollama_pool,      # Task distribution pool
    rpc_backends=rpc_backends,    # Model sharding backends
    coordinator_host="127.0.0.1",
    coordinator_port=18080
)

# Router automatically chooses:
# - Ollama parallelization for multi-task workloads
# - RPC backends for large models that need sharding
```

---

## Troubleshooting

### Issue 1: No Nodes Discovered

**Symptoms**:
```
⚠️  No nodes discovered on network
✅ Total Ollama nodes available: 0
```

**Solutions**:

1. **Check Ollama is running**:
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. **Check firewall**:
   ```bash
   sudo ufw status
   sudo ufw allow 11434/tcp
   ```

3. **Verify subnet detection**:
   ```python
   import socket
   s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
   s.connect(("10.255.255.255", 1))
   print(s.getsockname()[0])  # Should show your local IP
   ```

4. **Try manual configuration**:
   ```bash
   export OLLAMA_HOST="http://localhost:11434"
   ```

### Issue 2: Duplicate Nodes (3 nodes, only 2 machines)

**Symptoms**:
```
⚡ PARALLEL MULTI-TURN MODE: 3 nodes available
(But you only have 2 physical machines)
```

**Cause**: Stale config file with `localhost` entry

**Solution**: Delete config and re-discover
```bash
rm ~/.synapticllamas_nodes.json
# Restart in distributed mode - will auto-discover fresh
```

**Prevention**: Now fixed - auto-discovery runs FIRST, config is fallback only

### Issue 3: Parallel Mode Not Enabled (Multiple Machines)

**Symptoms**:
```
✅ 2 nodes available
ℹ️  Parallel mode DISABLED (same machine)
```

**Cause**: Nodes have localhost hostname instead of real IPs

**Check node IPs**:
```python
from node_registry import NodeRegistry
import os

registry = NodeRegistry()
registry.load_config(os.path.expanduser("~/.synapticllamas_nodes.json"))

for url, node in registry.nodes.items():
    print(f"{url} - {node.name}")
```

**Solution**: Ensure real IPs in config:
```json
{
  "nodes": [
    {"url": "http://10.9.66.154:11434"},  // ✅ Real IP
    {"url": "http://10.9.66.194:11434"}   // ✅ Real IP
  ]
}
```

**NOT**:
```json
{
  "nodes": [
    {"url": "http://localhost:11434"},    // ❌ Localhost
    {"url": "http://127.0.0.1:11434"}     // ❌ Localhost
  ]
}
```

### Issue 4: Slow Auto-Discovery

**Symptoms**: Startup takes 10+ seconds

**Cause**: Network timeout too high or slow network

**Solution 1**: Reduce timeout
```python
registry.discover_and_add_nodes(timeout=0.2)  # 200ms instead of 500ms
```

**Solution 2**: Disable auto-discovery, use config
```python
registry = NodeRegistry(auto_discover=False)
registry.load_config("~/.synapticllamas_nodes.json")
```

### Issue 5: Parallel Mode Enabled But Slower

**Symptoms**: Parallel execution takes longer than sequential

**Possible causes**:

1. **Same machine** (locality detection failed):
   - Check: Are all IPs resolving to same machine?
   - Fix: Verify node IPs are actually different machines

2. **Small tasks** (overhead exceeds benefit):
   - Check: Are tasks too small? (< 30s each)
   - Fix: Increase chunk size or use sequential mode

3. **Network latency** (remote nodes):
   - Check: Ping time between machines
   - Fix: Use local nodes only or increase timeout

4. **Resource limits** (CPU/memory bottleneck):
   - Check: htop during execution
   - Fix: Reduce parallelism level or upgrade hardware

---

## Configuration Quick Reference

### Auto-Discovery (Recommended)

```python
# Automatic - just start in distributed mode
python main.py
> mode distributed

# Manual trigger
from node_registry import NodeRegistry
registry = NodeRegistry(auto_discover=True)
```

### Manual Configuration

```bash
# Edit config file
nano ~/.synapticllamas_nodes.json

# Format:
{
  "nodes": [
    {"url": "http://IP:PORT", "name": "name", "priority": 0}
  ]
}
```

### Check Locality

```python
from sollol.pool import OllamaPool

pool = OllamaPool(nodes=discovered_nodes)
unique_hosts = pool.count_unique_physical_hosts()
should_parallel = pool.should_use_parallel_execution(num_tasks=3)

print(f"Machines: {unique_hosts}, Parallel: {should_parallel}")
```

### Environment Variables

```bash
# Override primary node
export OLLAMA_HOST="http://specific-node:11434"

# Disable auto-discovery (use config only)
export SOLLOL_AUTO_DISCOVER=false
```

---

## Summary

### Task Distribution Configuration

**Goal**: Enable parallel execution across multiple machines

**Config**:
1. ✅ Auto-discovery (distributed mode) - finds all nodes automatically
2. ✅ Manual config (advanced) - specify exact nodes

**Verify**: Check node count and unique machines

### Performance Configuration

**Goal**: Avoid false parallelism and resource contention

**Config**:
1. ✅ Locality awareness (automatic) - detects same-machine nodes
2. ✅ Intelligent mode selection - parallel only when beneficial

**Verify**: Check parallel mode status and reasoning

### Best Practices

1. **Use auto-discovery** in distributed mode (primary source)
2. **Config file as backup** only (for edge cases/overrides)
3. **Trust SOLLOL's intelligence** - it knows when parallel helps
4. **Monitor performance** - compare parallel vs sequential for your workload
5. **Update config rarely** - auto-discovery keeps it fresh

---

## Related Documentation

- `INTELLIGENT_NODE_DISCOVERY.md` - Technical details on discovery algorithm
- `SOLLOL_LOCALITY_AWARENESS_ISSUE.md` - Problem analysis and solution
- `SOLLOL_DISCOVERY_PRIORITY_FIX.md` - Config file vs auto-discovery priority
- `HYBRID_RPC_PARALLELIZATION.md` - RPC backend configuration

---

**Last Updated**: 2025-10-21
**SOLLOL Version**: 0.2.0+
