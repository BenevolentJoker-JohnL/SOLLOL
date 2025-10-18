# Multi-Application Architecture Guide

## The Question

**What happens when multiple applications use SOLLOL simultaneously?**

This is a critical design consideration that affects how you deploy SOLLOL in production.

---

## Three Deployment Patterns

### Pattern 1: Shared Gateway (Recommended ✓)

**Architecture:**
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   App 1     │     │   App 2     │     │   App 3     │
│   (Agent)   │     │  (Chatbot)  │     │  (Pipeline) │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                   ┌───────▼────────┐
                   │ SOLLOL Gateway │
                   │  (Port 8000)   │
                   └───────┬────────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
┌──────▼──────┐     ┌──────▼──────┐     ┌──────▼──────┐
│  Ollama 1   │     │  Ollama 2   │     │  Ollama 3   │
└─────────────┘     └─────────────┘     └─────────────┘
```

**How it works:**
- **One** SOLLOL gateway running (`sollol up --port 8000`)
- All applications send requests to `http://localhost:8000/api/chat`
- Gateway handles all routing, priority, and coordination

**Advantages:**
- ✅ **Coordinated routing** - Single view of cluster state
- ✅ **Global priority queue** - Fair scheduling across apps
- ✅ **Resource awareness** - No duplicate route decisions
- ✅ **Centralized monitoring** - One place for metrics

**Disadvantages:**
- ⚠️ Single point of failure (gateway crash = all apps fail)
- ⚠️ Gateway becomes bottleneck under extreme load

**When to use:**
- Multiple applications on same machine
- Need coordinated scheduling
- Want centralized observability
- **This is the recommended pattern for most use cases**

**Example:**
```python
# App 1 (multi-agent system)
import requests
response = requests.post(
    "http://localhost:8000/api/chat",
    json={"model": "llama3.2", "messages": [...], "priority": 8}
)

# App 2 (background batch job)
import requests
response = requests.post(
    "http://localhost:8000/api/chat",
    json={"model": "mistral", "messages": [...], "priority": 3}
)

# Gateway coordinates: App 1 gets priority, routes to best node
```

---

### Pattern 2: Independent Instances (⚠️ Not Recommended)

**Architecture:**
```
┌─────────────┐     ┌─────────────┐
│   App 1     │     │   App 2     │
│ + SOLLOL    │     │ + SOLLOL    │
│   Pool      │     │   Pool      │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └────────┬──────────┘
                │ (CONFLICT!)
       ┌────────┼────────┐
       │        │        │
┌──────▼──┐ ┌──▼────┐ ┌─▼───────┐
│Ollama 1 │ │Ollama2│ │Ollama 3 │
└─────────┘ └───────┘ └─────────┘
```

**How it works:**
- Each app creates its own `OllamaPool` instance
- Each pool independently routes to the same Ollama nodes
- **No coordination** between pools

**What happens:**
```python
# App 1
from sollol.sync_wrapper import OllamaPool
pool1 = OllamaPool.auto_configure()  # Discovers nodes 1,2,3

# App 2
from sollol.sync_wrapper import OllamaPool
pool2 = OllamaPool.auto_configure()  # Discovers same nodes 1,2,3

# Both apps route independently
# App 1: Routes to Node 1 (best score: 250)
# App 2: Routes to Node 1 (best score: 250)  <-- CONFLICT!
# Result: Node 1 gets 2x load, Nodes 2,3 idle
```

**Problems:**
- ❌ **Resource contention** - Both route to "best" node simultaneously
- ❌ **No global load awareness** - Each pool thinks nodes are idle
- ❌ **Priority conflicts** - High-priority in App 1 competes with low-priority in App 2
- ❌ **Inefficient** - Could have used different nodes for parallel execution
- ❌ **Unpredictable** - Race conditions in routing decisions

**Why this happens:**
- Each OllamaPool maintains its own state
- No inter-process communication
- No shared memory or coordination

**When this might be acceptable:**
- Apps run at different times (not concurrent)
- Very light load (conflicts rare)
- Apps use different models that require different nodes
- **Generally: Don't use this pattern unless you have a specific reason**

---

### Pattern 3: Multiple Gateways (⚠️ Limited Use Cases)

**Architecture:**
```
┌─────────────┐                    ┌─────────────┐
│   App 1     │                    │   App 2     │
└──────┬──────┘                    └──────┬──────┘
       │                                  │
┌──────▼──────────┐            ┌──────────▼──────┐
│ SOLLOL Gateway  │            │ SOLLOL Gateway  │
│  (Port 8000)    │            │  (Port 8001)    │
└──────┬──────────┘            └──────┬──────────┘
       │                              │
       └──────────────┬───────────────┘
                      │
       ┌──────────────┼──────────────┐
       │              │              │
┌──────▼──┐     ┌─────▼────┐  ┌─────▼────┐
│Ollama 1 │     │ Ollama 2 │  │ Ollama 3 │
└─────────┘     └──────────┘  └──────────┘
```

**How it works:**
- Each app runs its own SOLLOL gateway on different ports
- Gateways route to the same Ollama nodes
- **Partial coordination** - within each gateway, not across gateways

**Problems:**
- ❌ Same issues as Pattern 2 (no coordination between gateways)
- ❌ More resource overhead (2 gateway processes)
- ❌ More complex deployment

**Port conflict:**
```bash
# App 1 starts
sollol up --port 8000  # ✓ Works

# App 2 tries to start
sollol up --port 8000  # ✗ Error: Port already in use

# Solution: Different ports
sollol up --port 8001  # ✓ Works, but no coordination with App 1
```

**When to use:**
- Apps on different machines
- Network segmentation requirements
- Isolated failure domains
- **Better solution: Use Pattern 1 with HA gateway**

---

## Concurrency Within a Single Gateway

### Thread Safety

**The gateway IS thread-safe:**
```python
# Multiple concurrent requests to one gateway
import threading
import requests

def make_request(i):
    response = requests.post(
        "http://localhost:8000/api/chat",
        json={"model": "llama3.2", "messages": [{"role": "user", "content": f"Request {i}"}]}
    )
    return response

# 100 concurrent requests
threads = [threading.Thread(target=make_request, args=(i,)) for i in range(100)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# ✓ All requests handled correctly
# ✓ Priority queue coordinates execution
# ✓ Routing is atomic per request
```

**How the gateway handles concurrency:**
1. **FastAPI async endpoints** - Handles concurrent HTTP requests
2. **Priority queue** - Serializes routing decisions (thread-safe)
3. **Per-request routing** - Each request independently scored
4. **Async I/O** - Non-blocking Ollama node calls

---

## The Core Issue: State Coordination

### Why Multiple SOLLOL Instances Don't Coordinate

**Each SOLLOL instance maintains:**
```python
class IntelligentRouter:
    def __init__(self):
        # Local state - NOT shared across instances
        self.performance_history = {}  # Per-instance
        self.node_capabilities = {}    # Per-instance
        self.task_patterns = {}        # Per-instance
```

**No inter-process communication:**
- No shared memory
- No message queue
- No distributed state
- No cluster coordination

**Result:**
```
Time T0:
  Instance 1: "Node 1 is idle (cpu_load: 0.1) → score: 250"
  Instance 2: "Node 1 is idle (cpu_load: 0.1) → score: 250"

Time T1:
  Instance 1: Routes to Node 1
  Instance 2: Routes to Node 1  <-- Both chose same node

Time T2:
  Reality: Node 1 cpu_load = 0.8 (both requests)
  But neither instance knows this yet!
```

---

## Recommended Architecture Patterns

### Small Scale (1 machine, <10 apps)

**Use Pattern 1: Shared Gateway**
```bash
# Start one gateway
sollol up --port 8000

# All apps connect to it
# Python app:
import requests
requests.post("http://localhost:8000/api/chat", ...)

# Node.js app:
fetch("http://localhost:8000/api/chat", ...)

# Go app:
http.Post("http://localhost:8000/api/chat", ...)
```

### Medium Scale (Multiple machines, <100 apps)

**Option A: Gateway per machine + DNS/load balancer**
```
┌─────────┐     ┌─────────┐
│ Machine1│     │Machine 2│
│ +Gateway│     │+Gateway │
└────┬────┘     └────┬────┘
     └─────┬──────────┘
           │
    ┌──────▼──────┐
    │ Load        │
    │ Balancer    │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ Apps (100)  │
    └─────────────┘
```

**Option B: Single HA gateway with failover**
```
┌─────────────┐     ┌─────────────┐
│  Primary    │     │  Standby    │
│  Gateway    │────▶│  Gateway    │
└──────┬──────┘     └─────────────┘
       │              (Keepalive)
   ┌───▼───┐
   │ Apps  │
   └───────┘
```

### Large Scale (Distributed, >100 apps)

**Consider dedicated orchestration:**
- Kubernetes with SOLLOL operator
- Service mesh (Istio/Linkerd) for routing
- External scheduler (Nomad/Mesos)
- Message queue for request distribution

---

## What Needs to Be Built for True Multi-Instance Coordination

### Future Enhancement: Distributed State

**What would be needed:**
```python
class DistributedRouter:
    def __init__(self):
        # Shared state via Redis/etcd
        self.cluster_state = RedisBackend()

        # Distributed lock for atomic routing
        self.routing_lock = DistributedLock()

        # Pub/sub for node health updates
        self.health_updates = PubSub()

    def select_optimal_node(self, context):
        with self.routing_lock:
            # Get latest global state
            nodes = self.cluster_state.get_all_nodes()

            # Make routing decision
            selected = self._score_nodes(nodes)

            # Immediately update global state
            self.cluster_state.increment_node_load(selected)

            return selected
```

**Technologies that could enable this:**
- **Redis** - Shared state, pub/sub, distributed locks
- **etcd** - Distributed configuration
- **Consul** - Service mesh coordination
- **ZooKeeper** - Distributed coordination

**Complexity trade-off:**
- ✅ True coordination across instances
- ✅ Global load awareness
- ❌ Much more complex deployment
- ❌ New failure modes (Redis down = all routing fails)
- ❌ Latency overhead (network calls for every route decision)

---

## Current Recommendation

**For SOLLOL v0.3.6:**

### ✅ DO: Use Shared Gateway (Pattern 1)

```bash
# Terminal 1: Start gateway once
sollol up --port 8000

# Terminal 2: App 1
curl http://localhost:8000/api/chat -d '{"model": "llama3.2", ...}'

# Terminal 3: App 2
curl http://localhost:8000/api/chat -d '{"model": "mistral", ...}'

# Terminal 4: App 3
curl http://localhost:8000/api/chat -d '{"model": "qwen", ...}'
```

### ❌ DON'T: Create multiple independent SOLLOL instances

```python
# ❌ DON'T DO THIS (unless apps run at different times)
# app1.py
pool1 = OllamaPool.auto_configure()

# app2.py
pool2 = OllamaPool.auto_configure()

# They will conflict on routing decisions!
```

### 🔮 FUTURE: Distributed coordination

If you need true multi-instance coordination:
1. File a GitHub issue describing your use case
2. We can design distributed state coordination
3. Likely involves Redis/etcd for shared state

---

## FAQ

**Q: Can I run SOLLOL gateway in Docker and apps on host?**
A: Yes, expose gateway port: `docker run -p 8000:8000 sollol`

**Q: Can I run multiple gateways for high availability?**
A: Not currently - would need leader election. Use nginx/HAProxy failover for now.

**Q: What if I want app-level isolation?**
A: Add `app_id` to requests, gateway can prioritize per-app.

**Q: Can different apps use different priorities?**
A: Yes! High-priority apps can set `priority=10`, low-priority `priority=1`.

**Q: What happens if gateway crashes?**
A: All apps fail until gateway restarts. For HA, use keepalived + standby gateway.

**Q: Can apps on different machines share one gateway?**
A: Yes, just point them to gateway's IP: `http://gateway-host:8000/api/chat`

---

## Summary

| Pattern | Apps | Coordination | Recommended |
|---------|------|--------------|-------------|
| **Shared Gateway** | Many → One Gateway | ✅ Full | ✅ YES |
| **Independent Instances** | Each has own pool | ❌ None | ❌ NO |
| **Multiple Gateways** | Each has own gateway | ⚠️ Partial | ⚠️ LIMITED |

**The Rule:** One gateway per cluster, many apps per gateway.

**For distributed multi-gateway coordination:** This is a future enhancement requiring distributed state management. Current design assumes single gateway per cluster.
