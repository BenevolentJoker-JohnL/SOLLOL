# SOLLOL Lock Optimization - Planned Update

**Date**: 2025-10-21
**Priority**: High
**Impact**: 1.7-1.8x speedup for parallel embedding workloads
**Status**: Planned for future release

---

## Problem: Global Lock Serializes Parallel Requests

### Current Performance

**FlockParser batch processing (20 PDFs, 2 machines):**
- **Current**: 2179s (36.3 min) @ 109s per PDF
- **Expected**: ~1200s (20 min) @ 60s per PDF (**1.8x faster**)

### Root Cause

**File**: `/home/joker/SOLLOL/src/sollol/pool.py:919-920, 936-940`

SOLLOL's `_make_request()` uses a **global lock** to protect stats updates:

```python
def _make_request(self, endpoint, data, priority, timeout):
    # Lock #1: Stats increment
    with self._lock:
        self.stats["total_requests"] += 1  # ← GLOBAL LOCK!

    # Lock #2: Node selection (also takes self._lock internally!)
    node, decision = self._select_node(payload=data, priority=priority)

    # Lock #3: Active request tracking
    with self._lock:
        self.stats["node_performance"][node_key]["active_requests"] += 1

    # Network I/O (should be parallel but serialized by locks above!)
    response = self.session.post(url, json=data, timeout=timeout)
```

### Impact on Parallelism

**With 12 worker threads:**

```
Thread 1:  [Lock] → increment stats → [Unlock] → send request → wait for response
Thread 2:  [WAIT FOR LOCK...........] → increment → send → wait
Thread 3:  [WAIT FOR LOCK.......................] → increment → send
...
Thread 12: [WAIT FOR LOCK................................................]
```

**Result**: Requests are **serialized** despite using ThreadPoolExecutor!

### Comparison: Legacy FlockParser (FAST)

**FlockParser-legacy achieved 1.76x speedup (6 min → 3.4 min) by:**

```python
# Direct parallel requests - NO GLOBAL LOCK!
def embed_batch(texts):
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = []
        for i, text in enumerate(texts):
            node = nodes[i % len(nodes)]  # Simple round-robin
            future = executor.submit(
                requests.post,
                f"{node}/api/embed",
                json={"model": model, "input": text}
            )
            futures.append(future)
        return [f.result() for f in futures]
```

**No lock → True parallelism → 1.76x speedup!**

---

## Planned Solution: Atomic Counters + Fine-Grained Locking

### Phase 1: Replace Stats Dict with Atomic Counters

**Create atomic counter class:**

```python
from threading import Lock

class AtomicInt:
    """Thread-safe integer counter with minimal locking."""

    def __init__(self, value=0):
        self._value = value
        self._lock = Lock()

    def increment(self, amount=1):
        """Atomically increment and return new value."""
        with self._lock:
            self._value += amount
            return self._value

    def get(self):
        """Atomically read current value."""
        with self._lock:
            return self._value

    def set(self, value):
        """Atomically set value."""
        with self._lock:
            self._value = value
```

**Replace stats dictionary:**

```python
class OllamaPool:
    def __init__(self, ...):
        # BEFORE: Single dict with global lock
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            ...
        }
        self._lock = threading.Lock()  # Global lock for everything!

        # AFTER: Individual atomic counters
        self.total_requests = AtomicInt()
        self.successful_requests = AtomicInt()
        self.failed_requests = AtomicInt()

        # Keep lock only for complex operations
        self._node_lock = threading.Lock()  # For node list modifications only
        self._perf_lock = threading.Lock()  # For performance dict updates only
```

### Phase 2: Update _make_request() to Use Atomics

**BEFORE (slow):**

```python
def _make_request(self, endpoint, data, priority, timeout):
    with self._lock:  # ← Blocks all other threads!
        self.stats["total_requests"] += 1

    # ... rest of request ...
```

**AFTER (fast):**

```python
def _make_request(self, endpoint, data, priority, timeout):
    # Atomic increment - minimal lock contention!
    self.total_requests.increment()

    # ... rest of request runs in parallel ...
```

### Phase 3: Fine-Grained Locking for Node Performance

**BEFORE (slow):**

```python
with self._lock:  # ← Global lock for all nodes!
    self.stats["node_performance"][node_key]["active_requests"] += 1
```

**AFTER (fast):**

```python
# Per-node locks instead of global lock
with self._perf_lock:
    if node_key not in self.node_stats:
        self.node_stats[node_key] = {
            "active_requests": AtomicInt(),
            "total_requests": AtomicInt(),
            "latency_ms": AtomicInt(),
        }

# Atomic increment - no global lock needed!
self.node_stats[node_key]["active_requests"].increment()
```

---

## Expected Performance Improvement

### Benchmark: 20 PDFs, 2 Machines

**Current (with global lock):**
```
Total time: 2179s (36.3 min)
Per PDF: 109s
Speedup vs single machine: 1.12x (minimal)
```

**After fix (atomic counters):**
```
Total time: ~1200s (20 min)
Per PDF: ~60s
Speedup vs single machine: 1.8x (target!)
```

**Improvement**: **45% faster** (2179s → 1200s)

### Why This Matches Legacy Performance

**FlockParser-legacy benchmark:**
- Single CPU: 372s (6 min)
- 2 CPUs: 204s (3.4 min)
- **Speedup: 1.76x**

**SOLLOL after fix:**
- Removes serialization bottleneck
- Enables true parallel network I/O
- **Expected: 1.7-1.8x speedup** (matches legacy!)

---

## Implementation Checklist

### Phase 1: Core Atomic Infrastructure
- [ ] Create `AtomicInt` class in `pool.py`
- [ ] Create `AtomicFloat` class for latency tracking
- [ ] Add unit tests for atomic operations

### Phase 2: Stats Migration
- [ ] Replace `self.stats["total_requests"]` with `self.total_requests = AtomicInt()`
- [ ] Replace `self.stats["successful_requests"]` with atomic counter
- [ ] Replace `self.stats["failed_requests"]` with atomic counter
- [ ] Update `get_stats()` method to read from atomic counters

### Phase 3: Lock Optimization
- [ ] Replace `self._lock` with `self._node_lock` (for node list only)
- [ ] Create `self._perf_lock` for performance dict operations
- [ ] Remove lock from `_make_request()` hot path
- [ ] Keep lock only for:
  - Node list modifications (add/remove)
  - Complex performance dict updates

### Phase 4: Testing
- [ ] Benchmark: Single-threaded performance (baseline)
- [ ] Benchmark: 12 concurrent threads (should see ~1.8x speedup)
- [ ] Stress test: 100 concurrent threads
- [ ] Verify: No race conditions in stats
- [ ] Verify: Correct request counts after 10,000 requests

### Phase 5: Backward Compatibility
- [ ] Ensure `get_stats()` returns same dict format
- [ ] Ensure dashboard integration still works
- [ ] Update documentation

---

## Testing Strategy

### Test 1: Lock Contention Measurement

**Before fix:**
```python
import time
from concurrent.futures import ThreadPoolExecutor

def benchmark_embed():
    pool = OllamaPool(nodes=[...])
    texts = ["test" * 100] * 1000  # 1000 chunks

    start = time.time()
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(pool.embed, "mxbai-embed-large", text)
                   for text in texts]
        results = [f.result() for f in futures]

    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {len(texts)/elapsed:.2f} req/s")
```

**Expected results:**
- Before: ~5-10 req/s (serialized by lock)
- After: ~80-120 req/s (true parallelism)

### Test 2: Race Condition Verification

```python
# Hammer stats with concurrent updates
def stress_test():
    pool = OllamaPool(nodes=[...])

    def increment_stats():
        for _ in range(1000):
            pool.total_requests.increment()

    # 100 threads × 1000 increments = 100,000 total
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(increment_stats) for _ in range(100)]
        for f in futures:
            f.result()

    # Should be exactly 100,000
    assert pool.total_requests.get() == 100000
```

---

## Migration Path

### Option A: Gradual Migration (Safer)
1. Add atomic counters alongside existing stats dict
2. Update both in parallel (maintain compatibility)
3. Switch dashboards/monitoring to read from atomics
4. Remove old stats dict in next major version

### Option B: Direct Replacement (Faster)
1. Replace all stats dict access with atomic counters
2. Update `get_stats()` to build dict from atomics
3. Ship in next minor version (breaking change in internals only)

**Recommendation**: **Option B** - The `get_stats()` API stays the same, so external code doesn't break.

---

## Related Issues

### Issue 1: `_select_node()` Also Takes Lock

**File**: `pool.py:679-680`

```python
def _select_node(self, payload, priority):
    with self._lock:  # ← Another global lock!
        if not self.nodes:
            raise RuntimeError(...)
```

**Fix**: Use `self._node_lock` (separate from stats lock)

### Issue 2: Round-Robin Counter Race Condition

**File**: `pool.py:750`

```python
def _select_round_robin(self):
    with self._lock:
        self._round_robin_index = (self._round_robin_index + 1) % len(self.nodes)
```

**Fix**: Use `AtomicInt()` for round-robin index

---

## Performance Targets

### Target 1: Match Legacy FlockParser
- **Metric**: 1.7-1.8x speedup with 2 CPU nodes
- **Test**: 20 PDFs, uncached embeddings
- **Current**: 2179s
- **Target**: 1200-1300s

### Target 2: Scale to 12 Workers
- **Metric**: Linear scaling up to 12 concurrent requests
- **Current**: ~2x slowdown due to lock contention
- **Target**: 10-12x throughput improvement

### Target 3: Maintain Stats Accuracy
- **Metric**: 100% accurate request counts
- **Test**: 100,000 concurrent requests
- **Target**: Zero lost updates, zero race conditions

---

## Documentation Updates Needed

### User-Facing
- [ ] Update README.md performance benchmarks
- [ ] Add note about concurrent request improvements
- [ ] Update example code (if needed)

### Developer-Facing
- [ ] Document atomic counter design
- [ ] Add threading safety notes
- [ ] Update architecture diagrams

---

## Rollout Plan

### Week 1: Implementation
- Day 1-2: Implement `AtomicInt` class + tests
- Day 3-4: Migrate stats to atomic counters
- Day 5: Update `_make_request()` to remove global lock

### Week 2: Testing
- Day 1-2: Unit tests + integration tests
- Day 3-4: Performance benchmarks
- Day 5: Stress testing (race conditions)

### Week 3: Release
- Day 1-2: Code review
- Day 3: Documentation
- Day 4: Release candidate
- Day 5: Production release

---

## Success Criteria

**Must Have:**
- ✅ No race conditions (verified by stress tests)
- ✅ 1.7-1.8x speedup for 2-node parallel workloads
- ✅ Backward compatible `get_stats()` API

**Nice to Have:**
- ✅ Linear scaling up to 8-12 workers
- ✅ Reduced CPU usage (less lock contention)
- ✅ Better dashboard responsiveness

---

## References

- **Legacy benchmark**: `/home/joker/Documents/FlockParser-legacy/BENCHMARKS.md:36-38`
- **Current bottleneck**: `/home/joker/SOLLOL/src/sollol/pool.py:919-920`
- **sollol_compat fix**: `/home/joker/FlockParser/sollol_compat.py:187` (6 workers per node)

---

**Last Updated**: 2025-10-21
**Next Review**: Before next SOLLOL release
