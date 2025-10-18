# Redis Setup for Distributed Coordination

## What Problem Does This Solve?

**Without Redis:** Multiple SOLLOL instances don't coordinate. They route to the same "best" node simultaneously, causing resource conflicts.

**With Redis:** All SOLLOL instances share real-time cluster state through Redis, preventing routing conflicts.

---

## Quick Start

### 1. Install Redis

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

**macOS:**
```bash
brew install redis
brew services start redis
```

**Docker:**
```bash
docker run -d -p 6379:6379 --name redis redis:alpine
```

### 2. Install Python Redis Library

```bash
pip install redis>=5.0.0
```

### 3. Enable Distributed Mode

**Option A: Using CLI**
```bash
sollol up --redis-url redis://localhost:6379 --distributed
```

**Option B: Programmatic**
```python
from sollol.distributed_coordinator import create_coordinator
from sollol.intelligence import IntelligentRouter

# Create coordinator (automatically falls back to LocalCoordinator if Redis unavailable)
coordinator = create_coordinator(redis_url="redis://localhost:6379", enable_distributed=True)

# Use with router
router = IntelligentRouter(coordinator=coordinator)

# Cleanup when done
coordinator.close()
```

---

## How It Works

### Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ SOLLOL App 1 │     │ SOLLOL App 2 │     │ SOLLOL App 3 │
│ (instance A) │     │ (instance B) │     │ (instance C) │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       │      ┌─────────────┴─────────────┐      │
       │      │                           │      │
       └──────▼───────────────────────────▼──────┘
              │                           │
       ┌──────▼────────┐         ┌────────▼──────┐
       │  Redis Store  │         │  Heartbeats   │
       │  (Shared)     │         │  (10s TTL)    │
       └──────┬────────┘         └───────────────┘
              │
    Cluster State (Real-time)
              │
       ┌──────▼───────────────────────┐
       │    Ollama Nodes (1,2,3)      │
       └──────────────────────────────┘
```

### Service Discovery

Each SOLLOL instance:
1. Generates unique instance ID on startup
2. Registers in `sollol:instances` Redis set
3. Sends heartbeat every 5 seconds (TTL: 10s)
4. Dead instances auto-removed when heartbeat expires

```python
# Instance registration
self.instance_id = str(uuid.uuid4())
self.redis_client.sadd("sollol:instances", self.instance_id)
self.redis_client.setex(f"sollol:instance:{self.instance_id}:alive", 10, "1")
```

### State Sharing

Each instance updates its view of node metrics:
```python
coordinator.update_node_state("localhost:11434", {
    "active_requests": 3,
    "cpu_load": 0.4,
    "gpu_free_mem": 8000,
    "success_rate": 0.98,
    "avg_latency_ms": 250.0
})
```

When routing, get aggregated state from ALL instances:
```python
aggregated = coordinator.get_aggregated_node_state("localhost:11434")
# aggregated.active_requests = sum across all instances
# aggregated.cpu_load = average across all instances
# aggregated.gpu_free_mem = minimum across all instances
```

### Atomic Routing

Distributed lock ensures only one instance routes at a time:
```python
with coordinator.routing_lock():
    # Get current global state
    state = coordinator.get_aggregated_node_state(host)

    # Make routing decision
    selected_node = router.select_optimal_node(...)

    # Immediately update global state
    coordinator.increment_active_requests(selected_node)
```

---

## Redis Keys Used

### Instance Management
- `sollol:instances` - Set of active instance IDs
- `sollol:instance:{id}:info` - Instance metadata (started_at, hostname, version)
- `sollol:instance:{id}:alive` - Heartbeat key (TTL: 10s)
- `sollol:instance:{id}:node_state` - This instance's view of each node's state

### Cluster State
- `sollol:node:{host}:active_requests` - Atomic counter for active requests
- `sollol:routing_lock` - Distributed lock for routing coordination

### Example Redis Data
```bash
# View active instances
$ redis-cli
127.0.0.1:6379> SMEMBERS sollol:instances
1) "a3f2c9b1-4e5f-6a7b-8c9d-0e1f2a3b4c5d"
2) "f7e8d9c0-1a2b-3c4d-5e6f-7a8b9c0d1e2f"

# Check instance info
127.0.0.1:6379> GET sollol:instance:a3f2c9b1:info
"{\"instance_id\":\"a3f2c9b1-4e5f-6a7b-8c9d-0e1f2a3b4c5d\",\"started_at\":1759757000.0,\"hostname\":\"app-server-1\",\"version\":\"0.3.7\"}"

# Check active requests
127.0.0.1:6379> GET sollol:node:localhost:11434:active_requests
"12"
```

---

## Configuration Options

### RedisCoordinator Parameters

```python
from sollol.distributed_coordinator import RedisCoordinator

coordinator = RedisCoordinator(
    redis_url="redis://localhost:6379/0",  # Redis connection URL
    heartbeat_interval=5.0,                # Heartbeat frequency (seconds)
    state_ttl=30                          # State expiration (seconds)
)
```

### Advanced Redis URLs

**With authentication:**
```python
redis_url="redis://:password@localhost:6379/0"
```

**Remote Redis:**
```python
redis_url="redis://redis-server.example.com:6379/0"
```

**Redis Sentinel:**
```python
# Use redis-py-cluster or configure Sentinel separately
```

**Redis Cluster:**
```python
# Requires redis-py-cluster library
```

---

## Fallback Behavior

### Automatic Fallback to LocalCoordinator

If Redis is unavailable, SOLLOL automatically falls back to local mode:

```python
coordinator = create_coordinator(
    redis_url="redis://localhost:6379",
    enable_distributed=True
)
# If Redis connection fails, returns LocalCoordinator instead
# No errors, just logs warning and continues
```

**LocalCoordinator:**
- No distributed state (single instance only)
- Same API as RedisCoordinator
- Uses in-memory dictionaries
- No-op distributed lock (returns nullcontext)

---

## Production Deployment

### Redis High Availability

**Option 1: Redis Sentinel (Recommended)**
```bash
# 3 Redis instances + 3 Sentinel monitors
# Automatic failover on master failure
```

**Option 2: Redis Cluster**
```bash
# Sharded data across multiple nodes
# Higher throughput, but more complex
```

### Docker Compose Example

```yaml
services:
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

  sollol-gateway-1:
    image: sollol:latest
    environment:
      - REDIS_URL=redis://redis:6379/0
    command: sollol up --redis-url redis://redis:6379 --distributed
    depends_on:
      - redis

  sollol-gateway-2:
    image: sollol:latest
    environment:
      - REDIS_URL=redis://redis:6379/0
    command: sollol up --redis-url redis://redis:6379 --distributed --port 8001
    depends_on:
      - redis

volumes:
  redis-data:
```

### Kubernetes Example

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sollol-config
data:
  REDIS_URL: "redis://redis-service:6379/0"
  ENABLE_DISTRIBUTED: "true"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sollol-gateway
spec:
  replicas: 3  # Multiple instances coordinating via Redis
  template:
    spec:
      containers:
      - name: sollol
        image: sollol:latest
        envFrom:
        - configMapRef:
            name: sollol-config
        command: ["sollol", "up", "--redis-url", "$(REDIS_URL)", "--distributed"]
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
```

---

## Monitoring

### Check Instance Health

```bash
# How many instances are registered?
redis-cli SCARD sollol:instances

# List all instance IDs
redis-cli SMEMBERS sollol:instances

# Check specific instance is alive
redis-cli GET sollol:instance:{instance-id}:alive
```

### Monitor Cluster State

```bash
# View active requests per node
redis-cli GET sollol:node:localhost:11434:active_requests

# Check all node states from an instance
redis-cli HGETALL sollol:instance:{instance-id}:node_state
```

### Performance Metrics

```python
# In your SOLLOL instance
instances = coordinator.get_active_instances()
print(f"Active instances: {len(instances)}")

for instance in instances:
    print(f"Instance {instance.instance_id[:8]}")
    print(f"  Hostname: {instance.hostname}")
    print(f"  Started: {instance.started_at}")
    print(f"  Last heartbeat: {instance.last_heartbeat}")
```

---

## Troubleshooting

### Redis Connection Failed

**Error:** `ConnectionError: Redis connection failed: [Errno 111] Connection refused`

**Solutions:**
1. Check Redis is running: `redis-cli ping` (should return PONG)
2. Verify port: `sudo netstat -tlnp | grep 6379`
3. Check firewall: `sudo ufw allow 6379`
4. Test connection: `redis-cli -h localhost -p 6379 ping`

### Instance Not Showing Up

**Problem:** `get_active_instances()` returns empty list

**Debugging:**
```bash
# Check if instances registered
redis-cli SMEMBERS sollol:instances

# Check if heartbeat is updating
redis-cli GET sollol:instance:{instance-id}:alive
# Wait 5 seconds
redis-cli GET sollol:instance:{instance-id}:alive  # Should still exist
```

**Common causes:**
- Heartbeat thread not starting (check logs)
- Redis permissions issue
- Network connectivity between instance and Redis

### Routing Conflicts Still Happening

**Problem:** Multiple instances still routing to same node

**Check:**
1. Verify distributed mode enabled: `--distributed` flag or `enable_distributed=True`
2. Check routing lock is being used: Look for logs mentioning lock acquisition
3. Verify all instances connected to SAME Redis (not separate Redis instances)

**Test:**
```python
# In Instance 1
coordinator.increment_active_requests("localhost:11434")

# In Instance 2
state = coordinator.get_aggregated_node_state("localhost:11434")
print(state.active_requests)  # Should be 1
```

### High Latency with Distributed Mode

**Problem:** Routing decisions taking too long

**Causes:**
- Network latency to Redis (should be <1ms on localhost, <5ms on LAN)
- Lock contention (too many instances competing for routing lock)
- State aggregation overhead (checking many instances)

**Solutions:**
1. Use local/nearby Redis (same datacenter)
2. Reduce lock timeout: `coordinator.routing_lock(timeout=0.5)`
3. Batch route decisions (route multiple requests at once)
4. Consider eventual consistency instead of strict locking

---

## Performance Impact

### Overhead of Distributed Coordination

**Typical latency added per routing decision:**
- Redis on localhost: ~1-2ms
- Redis on LAN: ~3-5ms
- Redis on WAN: ~10-50ms (not recommended)

**Lock contention:**
- With 3 instances, 10 req/s each: Minimal contention (~0.1% lock wait time)
- With 10 instances, 100 req/s each: Moderate contention (~5% lock wait time)

**When distributed coordination is worth it:**
- ✅ Multiple SOLLOL instances on same machine
- ✅ 2-10 instances total
- ✅ Shared Ollama cluster
- ✅ Need coordinated priority queue

**When it's NOT worth it:**
- ❌ Single SOLLOL instance (use LocalCoordinator)
- ❌ >20 instances (lock contention too high)
- ❌ Separate Ollama clusters per instance (no sharing)

---

## Migration Guide

### From Single Instance to Distributed

**Before (local mode):**
```python
from sollol.sync_wrapper import OllamaPool

pool = OllamaPool.auto_configure()
response = pool.chat(messages=[...])
```

**After (shared gateway):**
```bash
# Terminal 1: Start gateway with Redis
sollol up --port 8000 --redis-url redis://localhost:6379 --distributed

# Terminal 2: App 1
curl http://localhost:8000/api/chat -d '{...}'

# Terminal 3: App 2
curl http://localhost:8000/api/chat -d '{...}'
```

**After (multiple gateways with Redis):**
```python
# App 1 - starts its own gateway
from sollol.distributed_coordinator import create_coordinator
from sollol.gateway import start_gateway

coordinator = create_coordinator(redis_url="redis://localhost:6379")
start_gateway(port=8000, coordinator=coordinator)

# App 2 - starts separate gateway
coordinator = create_coordinator(redis_url="redis://localhost:6379")
start_gateway(port=8001, coordinator=coordinator)

# Both gateways coordinate via Redis!
```

---

## Advanced Topics

### Custom State Aggregation

Override aggregation logic for specific metrics:

```python
class CustomCoordinator(RedisCoordinator):
    def get_aggregated_node_state(self, host):
        state = super().get_aggregated_node_state(host)

        # Custom logic: Use max latency instead of average
        # (more conservative routing)
        states = self._get_all_instance_states(host)
        state.avg_latency_ms = max(s['avg_latency_ms'] for s in states)

        return state
```

### Request Migration

Move queued requests between instances:

```python
# Instance A: Queue too long
if coordinator.get_queue_depth() > 100:
    # Publish request to shared queue
    coordinator.publish_request_to_global_queue(request)

# Instance B: Idle, pull from shared queue
if coordinator.get_queue_depth() == 0:
    request = coordinator.pull_request_from_global_queue()
```

### Leader Election

One instance handles cluster-wide tasks:

```python
# Try to become leader
if coordinator.try_acquire_leadership():
    # Only one instance gets here
    cleanup_old_metrics()
    aggregate_cluster_stats()
    coordinator.release_leadership()
```

---

## Testing

### Unit Tests with Mock Redis

```python
import pytest
from unittest.mock import MagicMock
from sollol.distributed_coordinator import RedisCoordinator

def test_coordinator():
    coordinator = RedisCoordinator(redis_url="redis://localhost:6379")
    coordinator.redis_client = MagicMock()  # Mock for testing

    coordinator.increment_active_requests("localhost:11434")
    coordinator.redis_client.incr.assert_called_with("sollol:node:localhost:11434:active_requests")
```

### Integration Tests with Real Redis

```python
import pytest
from sollol.distributed_coordinator import create_coordinator

@pytest.fixture
def redis_coordinator():
    coordinator = create_coordinator(redis_url="redis://localhost:6379")
    yield coordinator
    coordinator.close()

def test_multi_instance_coordination(redis_coordinator):
    # Create two coordinators
    coord1 = create_coordinator(redis_url="redis://localhost:6379")
    coord2 = create_coordinator(redis_url="redis://localhost:6379")

    # Coord1 increments
    coord1.increment_active_requests("localhost:11434")

    # Coord2 should see it
    state = coord2.get_aggregated_node_state("localhost:11434")
    assert state.active_requests == 1

    coord1.close()
    coord2.close()
```

---

## Summary

**Redis enables:**
- ✅ Multi-instance coordination
- ✅ Shared cluster state
- ✅ Atomic routing decisions
- ✅ Global priority queue
- ✅ Service discovery

**Setup steps:**
1. Install Redis: `brew install redis` / `apt install redis-server`
2. Install Python library: `pip install redis>=5.0.0`
3. Enable distributed mode: `sollol up --redis-url redis://localhost:6379 --distributed`

**Production checklist:**
- [ ] Redis high availability (Sentinel or Cluster)
- [ ] Monitor instance heartbeats
- [ ] Track routing lock contention
- [ ] Set up alerting for dead instances
- [ ] Benchmark latency overhead

For more details, see:
- [Multi-Application Architecture](MULTI_APP_ARCHITECTURE.md)
- [Known Limitations](KNOWN_LIMITATIONS.md)
- [Architecture Overview](ARCHITECTURE.md)
