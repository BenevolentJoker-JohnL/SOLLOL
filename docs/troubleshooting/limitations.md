# Known Limitations and Future Work

## Critical Limitations

### 1. No Multi-Instance Coordination (SIGNIFICANT)

**The Problem:**

When multiple applications run independent SOLLOL instances, they **do not coordinate**:

```python
# App 1
pool1 = OllamaPool.auto_configure()
pool1.chat(...)  # Thinks: "Node 1 has 10% CPU load, route there"

# App 2 (same moment)
pool2 = OllamaPool.auto_configure()
pool2.chat(...)  # Thinks: "Node 1 has 10% CPU load, route there"

# Reality: Node 1 now has 50% CPU load, but neither instance knows!
```

**What SHOULD happen:**
1. Both instances detect each other
2. Share real-time cluster state (CPU load, active requests, queue depth)
3. Coordinate routing decisions
4. Aggregate metrics across all instances

**What ACTUALLY happens:**
- Each instance maintains local state only
- No inter-process communication
- No distributed state management
- Routing decisions based on stale information

**Impact:**
- ❌ Resource contention (multiple instances route to same node)
- ❌ Suboptimal load distribution
- ❌ Tail latencies higher than necessary
- ❌ No global priority queue

**Current Workaround:**
Run single SOLLOL gateway, all apps connect via HTTP:
```bash
# One gateway
sollol up --port 8000

# All apps use it
curl http://localhost:8000/api/chat
```

**Why This Is a Real Problem:**
This forces a specific deployment architecture (centralized gateway) rather than allowing distributed operation. It's not a "feature", it's a **limitation**.

---

## What Would Real Coordination Look Like?

### Distributed State Architecture

**Components needed:**

1. **Service Discovery**
   - SOLLOL instances find each other on network
   - Maintain membership list
   - Detect when instances join/leave

2. **Shared Cluster State**
   - Real-time node metrics (CPU, GPU, queue depth)
   - Active request count per node
   - Recent routing decisions
   - Failure state

3. **Coordination Protocol**
   - Distributed lock for routing decisions
   - Eventually-consistent state propagation
   - Conflict resolution

4. **Aggregated Metrics**
   - Global request rate
   - Cluster-wide latency distribution
   - Per-node load across all instances

**Implementation Options:**

#### Option A: Redis-Based Coordination (Lightweight)

```python
class DistributedRouter:
    def __init__(self):
        # Connect to shared Redis
        self.redis = Redis(host='localhost', port=6379)

        # Register this instance
        self.instance_id = str(uuid.uuid4())
        self.redis.sadd('sollol:instances', self.instance_id)

        # Heartbeat
        self.heartbeat_thread = threading.Thread(target=self._heartbeat)
        self.heartbeat_thread.start()

    def _heartbeat(self):
        """Update instance liveness every 5 seconds."""
        while True:
            self.redis.setex(
                f'sollol:instance:{self.instance_id}:alive',
                10,  # TTL: 10 seconds
                '1'
            )
            time.sleep(5)

    def select_optimal_node(self, context):
        # Get current global state
        all_instances = self.redis.smembers('sollol:instances')

        # Aggregate routing data from all instances
        for instance_id in all_instances:
            is_alive = self.redis.get(f'sollol:instance:{instance_id}:alive')
            if not is_alive:
                # Instance dead, remove it
                self.redis.srem('sollol:instances', instance_id)
                continue

            # Get this instance's view of cluster load
            node_loads = self.redis.hgetall(f'sollol:instance:{instance_id}:node_loads')
            # Aggregate with our view
            self._merge_node_loads(node_loads)

        # Now make routing decision with global state
        best_node = self._score_with_global_state(context)

        # Update our routing decision in shared state
        self.redis.hincrby(f'sollol:node:{best_node}:active_requests', 1)

        return best_node
```

**Advantages:**
- ✅ Relatively simple to implement
- ✅ Redis is battle-tested
- ✅ Fast (sub-millisecond operations)
- ✅ Eventually consistent state

**Disadvantages:**
- ⚠️ Redis becomes single point of failure
- ⚠️ Network overhead for every routing decision
- ⚠️ Added complexity in deployment

#### Option B: Gossip Protocol (Decentralized)

```python
class GossipCoordinator:
    """
    Decentralized coordination using gossip protocol.
    No central state store needed.
    """

    def __init__(self):
        self.peers = set()  # Other SOLLOL instances
        self.local_state = {}  # This instance's view

        # Periodically gossip with random peers
        self.gossip_thread = threading.Thread(target=self._gossip_loop)
        self.gossip_thread.start()

    def _gossip_loop(self):
        """Every 1 second, gossip with 3 random peers."""
        while True:
            random_peers = random.sample(self.peers, min(3, len(self.peers)))
            for peer in random_peers:
                # Send our state to peer
                self._send_state_to_peer(peer, self.local_state)

                # Receive peer's state
                peer_state = self._receive_state_from_peer(peer)

                # Merge states
                self._merge_states(peer_state)

            time.sleep(1)

    def _merge_states(self, peer_state):
        """Merge peer's view with ours using vector clocks."""
        for node, metrics in peer_state.items():
            if node not in self.local_state:
                self.local_state[node] = metrics
            else:
                # Keep most recent data (vector clock comparison)
                if metrics['version'] > self.local_state[node]['version']:
                    self.local_state[node] = metrics
```

**Advantages:**
- ✅ No central coordinator
- ✅ Fully decentralized
- ✅ Scales to many instances
- ✅ Resilient to failures

**Disadvantages:**
- ⚠️ Complex to implement correctly
- ⚠️ Eventually consistent (not immediate)
- ⚠️ Convergence time increases with cluster size

#### Option C: Hybrid (Recommended)

**Use etcd/Consul for coordination:**
- Service discovery built-in
- Distributed locks available
- Watch API for state changes
- Production-ready

```python
import etcd3

class EtcdCoordinator:
    def __init__(self):
        self.etcd = etcd3.client(host='localhost', port=2379)

        # Register this instance
        self.instance_id = str(uuid.uuid4())
        self.lease = self.etcd.lease(ttl=10)  # 10 second TTL
        self.etcd.put(
            f'/sollol/instances/{self.instance_id}',
            json.dumps({'started_at': time.time()}),
            lease=self.lease
        )

        # Watch for other instances
        self.etcd.add_watch_callback(
            '/sollol/instances/',
            self._on_instance_change,
            range_end='/sollol/instances0'  # Prefix watch
        )

    def select_optimal_node(self, context):
        # Atomic routing with distributed lock
        with self.etcd.lock('/sollol/routing-lock', ttl=1):
            # Read current global state
            cluster_state = self._get_cluster_state()

            # Make routing decision
            best_node = self._score_nodes(cluster_state, context)

            # Update global state atomically
            self.etcd.put(
                f'/sollol/nodes/{best_node}/active_requests',
                str(cluster_state[best_node]['active_requests'] + 1)
            )

            return best_node
```

**Advantages:**
- ✅ Production-ready (used by Kubernetes)
- ✅ Distributed locks
- ✅ Strong consistency available
- ✅ Built-in leader election

**Disadvantages:**
- ⚠️ Another service to deploy
- ⚠️ Added latency (~5-10ms per routing decision)

---

## Implementation Effort

### Phase 1: Basic Coordination (1-2 weeks)

**Goal:** Multiple instances detect each other and share basic state

**Implementation:**
1. Add Redis dependency
2. Implement instance registration/heartbeat
3. Share node load metrics
4. Aggregate state before routing decisions

**Deliverable:**
- Multiple SOLLOL instances can run concurrently
- They coordinate basic routing
- No resource conflicts

### Phase 2: Advanced Features (2-4 weeks)

**Goal:** Full distributed coordination

**Implementation:**
1. Distributed priority queue (across instances)
2. Request migration (move queued requests between instances)
3. Leader election for cluster-wide tasks
4. Metrics aggregation across instances

**Deliverable:**
- Global priority queue
- Load balancing between instances
- Cluster-wide observability

### Phase 3: Production Hardening (4+ weeks)

**Goal:** Enterprise-grade distributed system

**Implementation:**
1. Failure recovery mechanisms
2. Split-brain detection and resolution
3. Performance optimization (caching, batching)
4. Monitoring and alerting integration

---

## Current State vs. Ideal State

### Current State (v0.3.6)

**Architecture:**
```
App 1 → SOLLOL Instance 1 → Ollama Nodes
App 2 → SOLLOL Instance 2 → Ollama Nodes  ❌ NO COORDINATION
App 3 → SOLLOL Instance 3 → Ollama Nodes
```

**What works:**
- ✅ Single instance routing is intelligent
- ✅ Priority queue within an instance
- ✅ Failover within an instance

**What doesn't work:**
- ❌ No coordination between instances
- ❌ Duplicate routing decisions
- ❌ No global priority queue
- ❌ Stale cluster state

### Ideal State (Future)

**Architecture:**
```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ SOLLOL 1 │────▶│  Redis   │◀────│ SOLLOL 2 │
│          │     │  / etcd  │     │          │
└────┬─────┘     └──────────┘     └────┬─────┘
     │                                  │
     │        Shared Cluster State      │
     │                                  │
     └────────────┬─────────────────────┘
                  │
          ┌───────▼────────┐
          │  Ollama Nodes  │
          └────────────────┘
```

**What works:**
- ✅ Multiple instances coordinate
- ✅ Shared global state
- ✅ No routing conflicts
- ✅ Global priority queue
- ✅ Efficient load distribution

---

## Other Known Limitations

### 2. Routing Strategy Hardcoded to Performance-Based

**Problem:**

The intelligent router currently uses a **hardcoded performance-based routing strategy**. While this strategy is sophisticated (combining latency, success rate, VRAM availability, active load, and model warmth), users cannot select alternative routing policies.

**What's missing:**

```python
# What users CANNOT do today:
router = IntelligentRouter(strategy="round_robin")      # ❌ Not supported
router = IntelligentRouter(strategy="least_loaded")     # ❌ Not supported
router = IntelligentRouter(strategy="fairness")         # ❌ Not supported
router = IntelligentRouter(strategy="random")           # ❌ Not supported

# What users CAN do today:
router = IntelligentRouter()  # ✅ Always performance-based strategy
```

**Available routing strategies in other systems (like Hydra-Dev):**
- `round_robin` - Even distribution across nodes
- `least_loaded` - Route to least busy node
- `task_aware` - Route based on task type specialization
- `fastest_response` - Route to lowest latency node
- `random` - Random selection (baseline for testing)

**Why it's hardcoded:**

The current performance-based strategy was designed for **demonstration purposes** to showcase intelligent routing capabilities. It combines multiple signals into a single scoring function that works well for most use cases.

**Design decision:**

Rather than prematurely abstracting routing strategies, SOLLOL focused on implementing **one excellent strategy** with rich signals:
- VRAM-aware GPU routing
- Active request load balancing
- Model warmth tracking (avoids cold loads)
- Task type detection
- Priority-aware scoring
- Success rate + latency optimization

**Future design:**

Future versions will expose user-selectable routing policies through:

1. **Strategy parameter:**
   ```python
   router = IntelligentRouter(strategy="performance_based")  # Default
   router = IntelligentRouter(strategy="fairness")           # Even distribution
   router = IntelligentRouter(strategy="latency_optimized")  # Min latency
   ```

2. **NodeRegistry configuration:**
   ```yaml
   # registry_config.yaml
   routing:
     strategy: fairness
     fallback: performance_based
   ```

3. **Per-request override:**
   ```python
   pool.chat(model, messages, routing_hint="least_loaded")
   ```

**Workaround:**

For now, to change routing behavior, you can:

1. **Modify node priorities:**
   ```python
   registry.add_node("http://node1:11434", priority=0)  # High priority
   registry.add_node("http://node2:11434", priority=10) # Low priority
   ```

2. **Manually partition nodes:**
   ```python
   # Use specific subset of nodes
   pool = OllamaPool(nodes=[node1, node2])  # Ignores other nodes
   ```

3. **Disable intelligent routing:**
   ```python
   pool = OllamaPool(enable_intelligent_routing=False)  # Falls back to round-robin
   ```

**Impact:**

- ✅ Current strategy works well for most use cases
- ⚠️ Cannot optimize for specific workload patterns (fairness, latency-only, etc.)
- ⚠️ Cannot A/B test different routing strategies
- ⚠️ Cannot satisfy compliance requirements (e.g., "must distribute evenly for auditing")

**Status:** Documented as future work. Implementation effort: ~1-2 weeks.

---

### 3. Single-Machine Benchmarking Limitations

**Note:** This is NOT a limitation - single machine benchmarks **can** establish baseline performance and validate routing logic.

**What single-machine benchmarks CAN do:**
- ✅ Establish baseline latency/throughput for reference
- ✅ Test routing strategy logic and fairness
- ✅ Validate failover behavior
- ✅ Measure overhead of intelligent routing vs round-robin
- ✅ Test priority queue and load distribution

**What they CANNOT do:**
- ⚠️ Measure true network latency impact (containers share localhost)
- ⚠️ Validate cross-datacenter routing decisions
- ⚠️ Test real-world network failures (packet loss, jitter)
- ⚠️ Measure NUMA effects on multi-socket systems

**Bottom line:** Single-machine benchmarks are **valid and useful** for most testing scenarios. They become limiting only when testing network-specific features or datacenter-scale deployments.

### 4. No Streaming Stall Detection

**Problem:**

SOLLOL supports streaming responses but doesn't detect when a stream **stalls mid-generation** (no new tokens for extended period).

**What's missing:**

```python
# Hydra-Dev has this:
stall_timeout = 30  # seconds
last_response_time = time.time()

async for chunk in stream:
    if time.time() - last_response_time > stall_timeout:
        logger.warning(f"Stream stalled for {stall_timeout}s, aborting")
        break

    last_response_time = time.time()
    yield chunk

# SOLLOL only has this:
response = session.post(url, json=data, timeout=300)  # Global timeout
# ❌ No detection of mid-stream stalls
```

**Why it matters:**

Streaming can stall due to:
- VRAM exhaustion (model swapping)
- Network issues (TCP retransmission)
- Ollama backend hangs
- Model context overflow

Without stall detection, clients wait indefinitely for the next token, wasting resources.

**Current behavior:**

SOLLOL relies on global HTTP timeout (300s default). If a stream starts successfully but stalls after generating 10 tokens, the client waits the full 300s before timing out.

**Impact:**

- ⚠️ Long hangs on stalled streams
- ⚠️ Poor user experience (no early failure detection)
- ⚠️ Wasted connection resources

**Workaround:**

Set aggressive timeout:
```python
pool.chat(model, messages, stream=True, timeout=30)  # Global timeout
```

**Future implementation:**

```python
# In pool.py _make_streaming_request()
STALL_TIMEOUT = 30  # seconds
last_chunk_time = time.time()

for line in response.iter_lines():
    if time.time() - last_chunk_time > STALL_TIMEOUT:
        logger.warning(f"⚠️  Stream stalled for {STALL_TIMEOUT}s, aborting")
        raise TimeoutError(f"Stream stalled after {STALL_TIMEOUT}s")

    last_chunk_time = time.time()
    yield chunk
```

**Status:** Easy to implement. Implementation effort: ~1 day.

---

### 5. Priority Queue is Async-Only

**Problem:** Synchronous API can't use priority queue features
**Impact:** Sync wrapper bypasses queue
**Solution:** Need thread-safe sync queue implementation
**Status:** Works around with blocking HTTP calls

### 6. No Request Migration

**Problem:** Once routed, request can't move to different node
**Impact:** Sticky to initially-selected node even if better option appears
**Solution:** Implement request cancellation + re-routing
**Status:** Would require significant refactoring

### 7. Learning is Per-Instance, Not Cluster-Wide

**Problem:** Performance history not shared between instances
**Impact:** Each instance learns independently, redundant data
**Solution:** Shared performance metrics store
**Status:** Requires distributed coordination (see #1)

---

## Workarounds for Current Limitations

### For Multi-Instance Coordination

**Option 1:** Use shared gateway (recommended)
```bash
sollol up --port 8000
# All apps connect to http://localhost:8000
```

**Option 2:** Manual node partitioning
```python
# App 1: Only use nodes 1-2
pool1 = OllamaPool(nodes=['http://node1:11434', 'http://node2:11434'])

# App 2: Only use nodes 3-4
pool2 = OllamaPool(nodes=['http://node3:11434', 'http://node4:11434'])

# No overlap = no conflicts
```

**Option 3:** Time-based multiplexing
```python
# App 1: Runs during business hours
# App 2: Runs during off-hours
# No concurrent access = no conflicts
```

---

## Contributing

If you're interested in implementing distributed coordination:

1. **File an issue** describing your use case
2. **Discuss design** - Redis vs gossip vs etcd
3. **Prototype** a minimal working implementation
4. **Submit PR** with tests and documentation

**Priority:** This is a high-priority enhancement for production use cases.

---

## Summary

**The honest assessment:**

SOLLOL v0.3.6 is designed for **single-instance deployment**. Multiple independent instances will conflict on routing decisions because there's no distributed state coordination.

**This is not a feature - it's a limitation.**

For production multi-instance deployments, you currently need:
- Use shared gateway architecture (HTTP-based)
- OR manually partition nodes between instances
- OR accept suboptimal routing with conflicts

Implementing distributed coordination (Redis/etcd/gossip) would solve this, but adds significant complexity. This is documented as future work with clear implementation options outlined above.
