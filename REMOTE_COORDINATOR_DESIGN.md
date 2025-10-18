# Remote Coordinator Execution Design

## Problem Statement

**Current Issue:**
When a request arrives on a resource-constrained node (e.g., `10.9.66.154` with 16GB RAM), the `llama-server` coordinator process starts **locally** on that node, even though other nodes in the cluster have significantly more resources (e.g., `10.9.66.90` with 128GB RAM).

**Why This Matters:**
- The coordinator process loads the entire GGUF model into memory before distributing layers
- This causes OOM (Out of Memory) crashes on low-RAM nodes
- Wastes the cluster's available resources
- Defeats the purpose of distributed inference

## Current Architecture

```
Request on 10.9.66.154 (16GB RAM, low resources)
  ↓
  Ray spawns ShardedModelPool actor LOCALLY
  ↓
  LlamaCppCoordinator starts on 127.0.0.1:18080 (local to 10.9.66.154)
  ↓
  llama-server loads model.gguf into RAM (requires 20GB+) → OOM CRASH
  ↓
  Would distribute to RPC backends (never gets here)
```

## Desired Architecture

```
Request on 10.9.66.154 (16GB RAM, low resources)
  ↓
  SOLLOL Intelligence analyzes cluster:
    - 10.9.66.154: 16GB RAM, 80% CPU load → SCORE: 2.5/10
    - 10.9.66.90:  128GB RAM, 20% CPU load → SCORE: 9.8/10 ✅
    - 10.9.66.48:  32GB RAM, 40% CPU load → SCORE: 7.2/10
  ↓
  Ray placement strategy: spawn actor on 10.9.66.90
  ↓
  ShardedModelPool actor runs on 10.9.66.90
  ↓
  LlamaCppCoordinator starts on 10.9.66.90:18080
  ↓
  llama-server loads model.gguf into RAM (128GB available) → SUCCESS
  ↓
  Distributes layers to RPC backends:
    ├─> 10.9.66.48:50052 (layers 0-20)
    ├─> 10.9.66.45:50052 (layers 21-40)
    └─> 10.9.66.90:50052 (layers 41-60)
  ↓
  Results stream back to 10.9.66.154
```

## Solution Components

### 1. Resource Discovery
Query each potential coordinator host for:
- Available RAM
- CPU load
- GPU availability (if needed for coordinator process)
- Network bandwidth
- Active coordinator count

### 2. SOLLOL Intelligence Integration
Use existing `intelligence.py::select_optimal_node()` to score hosts based on:
- Memory availability (critical for coordinator)
- CPU load (lightweight scoring)
- Task complexity
- Historical performance

### 3. Ray Placement Constraints
Use Ray's `@ray.remote` decorator with `resources` parameter:

```python
# Option 1: Node affinity by hostname
@ray.remote(resources={"node:10.9.66.90": 1})
class ShardedModelPool:
    ...

# Option 2: Custom resource requirements
@ray.remote(num_cpus=2, memory=30_000_000_000)  # 30GB
class ShardedModelPool:
    ...
```

### 4. Dynamic Coordinator Host Selection

#### Current Code (ray_hybrid_router.py:176)
```python
coordinator_host: str = "127.0.0.1",  # HARDCODED LOCAL
```

#### New Code
```python
def select_coordinator_host(
    self,
    model_size_gb: float,
    rpc_backends: List[Dict[str, Any]]
) -> str:
    """
    Select the best host to run the coordinator based on:
    - Available RAM (must exceed model size)
    - CPU load
    - Existing coordinator count
    - Network proximity to RPC backends
    """
    # Query resource stats from all RPC backend hosts
    candidate_hosts = []
    for backend in rpc_backends:
        stats = self._query_host_resources(backend["host"])
        if stats["free_memory_gb"] >= model_size_gb * 1.2:  # 20% buffer
            candidate_hosts.append({
                "host": backend["host"],
                "free_memory_gb": stats["free_memory_gb"],
                "cpu_load": stats["cpu_load"],
                "coordinator_count": stats["coordinator_count"],
            })

    if not candidate_hosts:
        raise RuntimeError(
            f"No hosts have sufficient RAM ({model_size_gb}GB) "
            "to run coordinator"
        )

    # Use SOLLOL intelligence to score and select
    best_host = self.intelligence_router.select_optimal_node(
        context=TaskContext(
            task_type="coordinator",
            complexity="high",
            estimated_tokens=0,
            model_preference=None,
            priority=10,
            requires_gpu=False,
            estimated_duration_ms=999999,  # Long-running
            metadata={"model_size_gb": model_size_gb}
        ),
        available_hosts=candidate_hosts
    )

    return best_host[0]  # Returns hostname
```

### 5. Ray Multi-Node Cluster Setup

For Ray to spawn actors on remote nodes, we need a Ray cluster:

#### Head Node (10.9.66.154)
```bash
ray start --head --port=6379 --dashboard-host=0.0.0.0
```

#### Worker Nodes (10.9.66.90, 10.9.66.48, 10.9.66.45)
```bash
ray start --address='10.9.66.154:6379'
```

#### Register Custom Resources
```bash
# On 10.9.66.90 (128GB RAM node)
ray start --address='10.9.66.154:6379' \
  --resources='{"high_memory": 1, "coordinator_slots": 4}'

# On 10.9.66.48 (32GB RAM node)
ray start --address='10.9.66.154:6379' \
  --resources='{"medium_memory": 1, "coordinator_slots": 2}'
```

### 6. Modified ShardedModelPool Actor

```python
@ray.remote(num_cpus=2, memory=30_000_000_000)  # 30GB requirement
class ShardedModelPool:
    """
    Ray actor that can be placed on ANY node in the Ray cluster.
    """

    def __init__(
        self,
        rpc_backends: List[Dict[str, Any]],
        coordinator_host: str,  # NOW DYNAMIC (not hardcoded 127.0.0.1)
        coordinator_port: int = 18080,
        pool_id: int = 0,
    ):
        # coordinator_host will be the actual IP of the selected node
        # e.g., "10.9.66.90" instead of "127.0.0.1"
        self.coordinator_host = coordinator_host
        ...
```

## Implementation Steps

### Step 1: Resource Query Service
Create `sollol/resource_query.py` to query remote hosts:

```python
import httpx

async def query_host_resources(host: str) -> Dict[str, Any]:
    """
    Query a remote host for resource availability.

    Requires a lightweight agent running on each host that exposes:
    - Free RAM
    - CPU load
    - Active coordinators
    - GPU stats (if applicable)
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://{host}:9090/stats")
        return response.json()
```

### Step 2: Coordinator Host Selection
Modify `RayHybridRouter.__init__()` to determine coordinator host:

```python
def __init__(self, ...):
    ...
    # NEW: Determine optimal coordinator host
    self.coordinator_host = self._select_coordinator_host()
```

### Step 3: Ray Placement Strategy
Create pools with placement constraints:

```python
# Before creating actors, determine which node should run them
coordinator_host = self.select_coordinator_host(model_size_gb=30)

# Create actor with node affinity
pool = ShardedModelPool.options(
    resources={f"node:{coordinator_host}": 0.01}
).remote(
    rpc_backends=self.rpc_backends,
    coordinator_host=coordinator_host,  # Pass the selected host
    coordinator_port=self.coordinator_base_port + i,
    pool_id=i
)
```

### Step 4: Network Accessibility
Ensure the coordinator can be reached from the originating node:

```python
# In LlamaCppCoordinator.start()
# Change host binding from 127.0.0.1 to 0.0.0.0
cmd = [
    "llama-server",
    "--model", self.model_path,
    "--host", "0.0.0.0",  # CHANGED from 127.0.0.1
    "--port", str(self.port),
    ...
]
```

## Resource Query Agent

Each node should run a lightweight resource reporting service:

### sollol/resource_agent.py
```python
from flask import Flask, jsonify
import psutil
import socket

app = Flask(__name__)

@app.route("/stats")
def get_stats():
    return jsonify({
        "hostname": socket.gethostname(),
        "free_memory_gb": psutil.virtual_memory().available / (1024**3),
        "total_memory_gb": psutil.virtual_memory().total / (1024**3),
        "cpu_load": psutil.cpu_percent(interval=1),
        "cpu_count": psutil.cpu_count(),
        "coordinator_count": len([p for p in psutil.process_iter(['name'])
                                 if 'llama-server' in p.info['name']]),
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090)
```

Run on each node:
```bash
python -m sollol.resource_agent &
```

## Benefits

1. **Automatic Resource Awareness**: Coordinators start on nodes with sufficient RAM
2. **OOM Prevention**: No more crashes from oversized models
3. **Optimal Resource Utilization**: Uses the cluster's full capacity
4. **Transparent to User**: Same API, smarter routing
5. **Fault Tolerance**: If best node unavailable, falls back to next-best

## Tradeoffs

1. **Additional Complexity**: Requires Ray cluster setup across nodes
2. **Network Dependency**: Coordinator results stream over network
3. **Resource Agent**: Requires lightweight service on each node
4. **Latency**: Small increase for remote coordinator (negligible for long inference)

## Implementation Status

### ✅ Completed

1. **Design complete** - Architecture documented
2. **Resource query system** - Leverages existing `rpc_discovery.py::detect_node_resources()`
   - Queries Redis for GPU/RAM metadata from remote nodes
   - Fallback to conservative CPU-only estimates
   - No separate agent needed - uses existing RPC backend registration

3. **Intelligent node selection** - `ShardedModelPool._select_best_coordinator_node()` (line 80-174)
   - Queries all RPC backends for available resources
   - Calculates score: `total_ram - estimated_ram_needed`
   - GPU nodes get 5GB bonus
   - Selects best node or falls back to local
   - Integrates with existing `detect_node_resources()` from `rpc_discovery.py`

4. **Dynamic coordinator execution** - `ShardedModelPool.load_model()` (line 176-241)
   - Coordinator runs on selected node (not hardcoded 127.0.0.1)
   - `coordinator_host` set dynamically based on node selection
   - Tracks which node is running coordinator (`self.coordinator_node`)
   - Returns metadata about coordinator placement

5. **Ray placement strategies** - `RayHybridRouter.__init__()` (line 416-461)
   - Uses `ShardedModelPool.options(scheduling_strategy="SPREAD")`
   - Spreads pools across Ray cluster nodes
   - `enable_remote_coordinator` parameter to toggle feature
   - Graceful fallback to local execution if disabled

6. **Result streaming** - Automatic via Ray
   - Ray's distributed object store handles result streaming
   - No code changes needed - works transparently
   - Results flow from remote node back to requesting node

### Implementation Details

**Modified Files:**
- `src/sollol/ray_hybrid_router.py`
  - Added `enable_remote_coordinator` parameter to `ShardedModelPool.__init__()` (line 51)
  - Added `_select_best_coordinator_node()` method (line 80-174)
  - Modified `load_model()` to use intelligent node selection (line 176-241)
  - Updated `chat()` to log coordinator node (line 243-278)
  - Added `enable_remote_coordinator` parameter to `RayHybridRouter.__init__()` (line 306)
  - Updated pool creation to use Ray placement strategies (line 416-461)

**Resource Detection:**
Uses existing infrastructure from `rpc_discovery.py`:
- `detect_node_resources(host)` - Gets RAM/GPU info from Redis
- Redis keys: `sollol:rpc:node:{host}:{port}`
- Registered by GPU nodes running `register_rpc_gpu_node.py`

**How It Works:**

```python
# 1. Request arrives on 10.9.66.154 (16GB RAM)
# 2. ShardedModelPool._select_best_coordinator_node() queries resources:
#    - 10.9.66.154: 16GB RAM → score = 16GB - 40GB = -24GB (rejected)
#    - 10.9.66.90: 128GB RAM → score = 128GB - 40GB = 88GB ✅ BEST
#    - 10.9.66.48: 32GB RAM → score = 32GB - 40GB = -8GB (rejected)
#
# 3. Selected node: 10.9.66.90
# 4. LlamaCppCoordinator starts on 10.9.66.90:18080
# 5. Ray streams results back to 10.9.66.154 automatically
```

### ⬜ Remaining Tasks

1. **Set up Ray cluster across nodes** (manual setup required)
   ```bash
   # Head node (10.9.66.154)
   ray start --head --port=6379 --dashboard-host=0.0.0.0

   # Worker nodes (10.9.66.90, 10.9.66.48, 10.9.66.45)
   ray start --address='10.9.66.154:6379'
   ```

2. **Test with large model** (e.g., llama3.1:70b)
   - Request from low-RAM node (10.9.66.154)
   - Verify coordinator spawns on high-RAM node (10.9.66.90)
   - Verify no OOM crashes
   - Measure inference latency

3. **Benchmark latency impact**
   - Compare local vs remote coordinator execution
   - Measure network overhead for result streaming
   - Document performance characteristics

4. **Optional: Resource agent** (not needed for MVP)
   - Current implementation uses Redis + existing RPC registration
   - Could add dedicated agent for real-time metrics if needed
