# SOLLOL Routing Strategies

## Overview

SOLLOL OllamaPool now supports **5 routing strategies** for distributing requests across Ollama nodes. Each strategy is optimized for different use cases and goals.

## Available Strategies

### 1. **ROUND_ROBIN**
Simple rotation through nodes in order.

**Use when:**
- You want predictable, even distribution
- All nodes have similar capabilities
- You don't care about performance optimization

**Characteristics:**
- ✅ Zero overhead - no intelligence
- ✅ Predictable distribution
- ✅ Simple to understand
- ❌ Ignores node performance
- ❌ Doesn't adapt to load

**Example:**
```python
from sollol import OllamaPool
from sollol.routing_strategy import RoutingStrategy

pool = OllamaPool.auto_configure(
    routing_strategy=RoutingStrategy.ROUND_ROBIN
)
```

---

### 2. **LATENCY_FIRST**
Always routes to the node with lowest average latency.

**Use when:**
- Response time is critical
- You have heterogeneous hardware (mixed GPUs/CPUs)
- You want to maximize speed

**Characteristics:**
- ✅ Minimizes response time
- ✅ Automatically favors faster nodes
- ✅ Good for interactive applications
- ❌ Can overload fastest node
- ❌ Slower nodes may be underutilized

**Example:**
```python
pool = OllamaPool.auto_configure(
    routing_strategy=RoutingStrategy.LATENCY_FIRST
)
```

---

### 3. **LEAST_LOADED**
Routes to the node with fewest active requests.

**Use when:**
- You have high concurrent load
- You want to maximize parallelism
- Node capabilities are similar

**Characteristics:**
- ✅ Maximizes throughput
- ✅ Prevents bottlenecks
- ✅ Good for batch processing
- ✅ Distributes load evenly in real-time
- ❌ Doesn't consider node speed differences

**Example:**
```python
pool = OllamaPool.auto_configure(
    routing_strategy=RoutingStrategy.LEAST_LOADED
)
```

---

### 4. **FAIRNESS**
Distributes requests evenly based on total request count over time.

**Use when:**
- You want all nodes to get equal utilization
- You have heterogeneous hardware but want fair distribution
- You want to avoid starving slower nodes

**Characteristics:**
- ✅ Ensures all nodes get equal work over time
- ✅ Good for long-running systems
- ✅ Prevents node starvation
- ❌ May route to slower nodes unnecessarily
- ❌ Doesn't optimize for performance

**Example:**
```python
pool = OllamaPool.auto_configure(
    routing_strategy=RoutingStrategy.FAIRNESS
)
```

---

### 5. **INTELLIGENT** (Default)
Task-aware routing with performance learning.

**Use when:**
- You want automatic optimization
- You have diverse workloads (embeddings, chat, generation)
- You want the best overall performance

**Characteristics:**
- ✅ Analyzes request type automatically
- ✅ Learns from historical performance
- ✅ Adapts to changing conditions
- ✅ GPU-aware routing
- ✅ Task-specific optimization
- ❌ Slight overhead for analysis
- ❌ Requires warm-up period for learning

**Example:**
```python
pool = OllamaPool.auto_configure(
    routing_strategy=RoutingStrategy.INTELLIGENT  # This is the default
)
```

---

## Comparison Table

| Strategy | Latency | Throughput | Simplicity | Adaptability | Overhead |
|----------|---------|------------|------------|--------------|----------|
| **ROUND_ROBIN** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| **LATENCY_FIRST** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **LEAST_LOADED** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **FAIRNESS** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **INTELLIGENT** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## Usage Examples

### Switching Strategies at Runtime

You can change the routing strategy dynamically:

```python
from sollol import OllamaPool
from sollol.routing_strategy import RoutingStrategy

# Start with intelligent routing
pool = OllamaPool.auto_configure(
    routing_strategy=RoutingStrategy.INTELLIGENT
)

# Switch to latency-first for time-critical requests
pool.routing_strategy = RoutingStrategy.LATENCY_FIRST

# Switch to least-loaded for batch processing
pool.routing_strategy = RoutingStrategy.LEAST_LOADED
```

### Checking Current Strategy

```python
stats = pool.get_stats()
print(f"Current strategy: {stats['routing_strategy']}")
```

---

## Implementation Details

### Strategy Pattern

All strategies are implemented using clean extension points:

```python
# pool.py
def _select_node(self, payload, priority):
    if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
        return self._select_round_robin(), None
    elif self.routing_strategy == RoutingStrategy.LATENCY_FIRST:
        return self._select_latency_first(), None
    # ... etc
```

### Node Selection Methods

Each strategy has its own dedicated method:

- `_select_round_robin()` - Simple index rotation
- `_select_latency_first()` - Finds node with min average latency
- `_select_least_loaded()` - Finds node with min active requests
- `_select_fairness()` - Finds node with min total requests
- `_select_intelligent()` - Task analysis + performance learning

### Backwards Compatibility

The `enable_intelligent_routing` parameter is preserved for backwards compatibility:

```python
# Old way (still works)
pool = OllamaPool(enable_intelligent_routing=False)  # Uses ROUND_ROBIN

# New way (recommended)
pool = OllamaPool(routing_strategy=RoutingStrategy.ROUND_ROBIN)
```

---

## Recommendations

### For Interactive Applications
Use **LATENCY_FIRST** or **INTELLIGENT**

### For Batch Processing
Use **LEAST_LOADED** or **INTELLIGENT**

### For Long-Running Systems
Use **FAIRNESS** or **INTELLIGENT**

### For Testing/Development
Use **ROUND_ROBIN** for predictability

### For Production (General)
Use **INTELLIGENT** for best overall performance

---

## Testing

Run the routing strategy test suite:

```bash
python test_routing_strategies.py
```

This will verify all strategies work correctly and show their behavior patterns.

---

## Future Enhancements

Possible future strategies:

- **COST_AWARE** - Route based on node cost metrics
- **POWER_EFFICIENT** - Prioritize low-power nodes
- **LATENCY_WEIGHTED** - Weighted combination of latency + load
- **CUSTOM** - User-defined strategy callbacks

---

## Related Documentation

- **[OllamaPool API](src/sollol/pool.py)** - Main routing implementation
- **[RoutingStrategy Enum](src/sollol/routing_strategy.py)** - Strategy definitions
- **[Intelligent Router](src/sollol/intelligence.py)** - Task-aware routing logic
