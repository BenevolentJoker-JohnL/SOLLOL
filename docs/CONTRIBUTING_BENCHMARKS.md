# Contributing Benchmarks to SOLLOL

**Help us move features from EXPERIMENTAL to PROVEN with real-world performance data!**

---

## Why Benchmarks Matter

SOLLOL has many features marked as **EXPERIMENTAL** because we lack comprehensive benchmark data. Your production usage can help us:

1. **Validate performance claims** - Turn theory into proven results
2. **Identify bottlenecks** - Find real-world issues we haven't seen
3. **Optimize defaults** - Tune parameters based on actual workloads
4. **Build confidence** - Help others adopt features with proven data
5. **Move features to PROVEN** - Graduate experimental features to production-ready

**What we need most:**
- ⚠️ **Multi-agent orchestration** (SynapticLlamas pattern) - No speedup data
- ⚠️ **Multi-mode routing** (Hydra pattern) - No mode comparison benchmarks
- ⚠️ **Hedging strategy** - No latency reduction data
- ⚠️ **GPU controller** - Need long-term stability data (days/weeks)

---

## Quick Start: 3-Minute Benchmark

The fastest way to contribute:

```python
import time
from sollol import OllamaPool

pool = OllamaPool.auto_configure()

# 1. Record your setup
setup = {
    "nodes": len(pool.nodes),
    "node_details": [
        {"host": f"{n['host']}:{n['port']}", "type": "GPU/CPU"}
        for n in pool.nodes
    ],
    "use_case": "batch_embeddings",  # or multi_agent, code_synthesis, etc.
}

# 2. Run your workload and time it
start = time.time()

results = pool.embed_batch(
    model="mxbai-embed-large",
    inputs=["text sample" for _ in range(1000)],
    use_adaptive=True,
    priority=7
)

duration = time.time() - start

# 3. Collect stats
stats = pool.get_stats()

# 4. Share your results (see template below)
benchmark = {
    "setup": setup,
    "duration_seconds": duration,
    "items_processed": len([r for r in results if r is not None]),
    "success_rate": len([r for r in results if r is not None]) / len(results),
    "throughput": len(results) / duration,
    "stats": stats
}

print(f"Throughput: {benchmark['throughput']:.2f} items/sec")
print(f"Success rate: {benchmark['success_rate']:.2%}")

# Copy output and submit via GitHub issue or email
```

---

## Benchmark Templates

### Template 1: Batch Processing Benchmark

**What to measure**: Throughput (items/sec), success rate, speedup vs single node

```python
import time
import json
from sollol import OllamaPool

# Setup
pool = OllamaPool.auto_configure()
texts = ["Sample text for embedding"] * 1000  # Your actual workload

# Baseline: Single node
single_node_pool = OllamaPool()
single_node_pool.add_node(pool.nodes[0]['host'], pool.nodes[0]['port'])

print("Running single-node baseline...")
start = time.time()
baseline_results = single_node_pool.embed_batch(
    model="mxbai-embed-large",
    inputs=texts,
    use_adaptive=False,
    max_workers=1,
    priority=7
)
baseline_duration = time.time() - start

# Distributed: Multi-node with adaptive
print("Running distributed with adaptive parallelism...")
start = time.time()
distributed_results = pool.embed_batch(
    model="mxbai-embed-large",
    inputs=texts,
    use_adaptive=True,
    priority=7
)
distributed_duration = time.time() - start

# Calculate metrics
baseline_throughput = len(baseline_results) / baseline_duration
distributed_throughput = len(distributed_results) / distributed_duration
speedup = distributed_throughput / baseline_throughput

benchmark = {
    "pattern": "batch_processing",
    "date": "2025-11-02",
    "sollol_version": "0.9.61",  # Check your version

    "setup": {
        "num_nodes": len(pool.nodes),
        "nodes": [
            {
                "host": f"{n['host']}:{n['port']}",
                "hardware": "CPU/GPU",  # Fill in
                "model_loaded": "mxbai-embed-large"
            }
            for n in pool.nodes
        ],
        "workload": {
            "type": "embeddings",
            "model": "mxbai-embed-large",
            "items": len(texts),
            "avg_length": sum(len(t) for t in texts) / len(texts)
        }
    },

    "results": {
        "baseline": {
            "duration_seconds": baseline_duration,
            "throughput_items_per_sec": baseline_throughput,
            "items_processed": len([r for r in baseline_results if r is not None]),
            "success_rate": len([r for r in baseline_results if r is not None]) / len(baseline_results)
        },
        "distributed": {
            "duration_seconds": distributed_duration,
            "throughput_items_per_sec": distributed_throughput,
            "items_processed": len([r for r in distributed_results if r is not None]),
            "success_rate": len([r for r in distributed_results if r is not None]) / len(distributed_results),
            "speedup_vs_baseline": speedup
        }
    },

    "node_performance": pool.get_stats()["node_performance"]
}

# Save to file
with open('sollol_batch_benchmark.json', 'w') as f:
    json.dump(benchmark, f, indent=2)

print(f"\n✅ Benchmark saved to sollol_batch_benchmark.json")
print(f"Speedup: {speedup:.2f}x")
print(f"Baseline: {baseline_throughput:.2f} items/sec")
print(f"Distributed: {distributed_throughput:.2f} items/sec")
```

---

### Template 2: Multi-Agent Orchestration Benchmark

**What to measure**: Parallel speedup, GPU utilization, agent completion time

```python
import time
import json
from sollol import OllamaPool
from sollol.gpu_controller import SOLLOLGPUController, integrate_with_router
from sollol.intelligence import IntelligentRouter

# Setup
pool = OllamaPool.auto_configure()
router = IntelligentRouter()
gpu_controller = SOLLOLGPUController(pool=pool, priority_models=["llama3.2"])
integrate_with_router(router, gpu_controller)

# Define agents (use your actual agents)
agents = [
    {"name": "planner", "model": "llama3.2", "prompt": "Create a plan for..."},
    {"name": "researcher", "model": "llama3.2", "prompt": "Research topic..."},
    {"name": "analyst", "model": "llama3.2", "prompt": "Analyze data..."},
    {"name": "synthesizer", "model": "llama3.2", "prompt": "Synthesize findings..."},
    {"name": "reviewer", "model": "llama3.2", "prompt": "Review output..."},
]

# Serial execution
print("Running agents serially...")
serial_times = []
start = time.time()
for agent in agents:
    agent_start = time.time()
    response = pool.chat(
        model=agent["model"],
        messages=[{"role": "user", "content": agent["prompt"]}]
    )
    agent_duration = time.time() - agent_start
    serial_times.append(agent_duration)
serial_total = time.time() - start

# Parallel execution
print("Running agents in parallel...")
import concurrent.futures

parallel_times = []
start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=len(agents)) as executor:
    futures = []
    for agent in agents:
        agent_start = time.time()
        future = executor.submit(
            pool.chat,
            model=agent["model"],
            messages=[{"role": "user", "content": agent["prompt"]}]
        )
        futures.append((future, agent_start))

    for future, agent_start in futures:
        future.result()
        agent_duration = time.time() - agent_start
        parallel_times.append(agent_duration)

parallel_total = time.time() - start

benchmark = {
    "pattern": "multi_agent_orchestration",
    "date": "2025-11-02",
    "sollol_version": "0.9.61",

    "setup": {
        "num_nodes": len(pool.nodes),
        "num_agents": len(agents),
        "agents": agents,
        "gpu_controller_enabled": True
    },

    "results": {
        "serial": {
            "total_seconds": serial_total,
            "agent_times": serial_times,
            "avg_per_agent": sum(serial_times) / len(serial_times)
        },
        "parallel": {
            "total_seconds": parallel_total,
            "agent_times": parallel_times,
            "avg_per_agent": sum(parallel_times) / len(parallel_times),
            "speedup_vs_serial": serial_total / parallel_total
        }
    }
}

with open('sollol_multiagent_benchmark.json', 'w') as f:
    json.dump(benchmark, f, indent=2)

print(f"\n✅ Benchmark saved to sollol_multiagent_benchmark.json")
print(f"Serial: {serial_total:.1f}s ({len(agents)} agents × {sum(serial_times)/len(serial_times):.1f}s avg)")
print(f"Parallel: {parallel_total:.1f}s")
print(f"Speedup: {serial_total/parallel_total:.2f}x")
```

---

### Template 3: Routing Mode Comparison Benchmark

**What to measure**: Latency differences between FAST/RELIABLE/ASYNC modes

```python
import time
import json
from sollol import OllamaPool
from sollol.routing_modes import RoutingMode

pool = OllamaPool.auto_configure()

modes = [RoutingMode.FAST, RoutingMode.RELIABLE, RoutingMode.ASYNC]
test_prompts = [
    "Write a FastAPI endpoint",
    "Explain quantum computing",
    "Debug this Python code: ..."
] * 10  # 30 requests total

results = {}

for mode in modes:
    print(f"Testing {mode} mode...")
    pool.set_routing_mode(mode)

    latencies = []
    for prompt in test_prompts:
        start = time.time()
        response = pool.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )
        latency = time.time() - start
        latencies.append(latency)

    results[str(mode)] = {
        "latencies": latencies,
        "avg_latency": sum(latencies) / len(latencies),
        "p50_latency": sorted(latencies)[len(latencies)//2],
        "p95_latency": sorted(latencies)[int(len(latencies)*0.95)],
        "p99_latency": sorted(latencies)[int(len(latencies)*0.99)]
    }

benchmark = {
    "pattern": "routing_mode_comparison",
    "date": "2025-11-02",
    "sollol_version": "0.9.61",
    "setup": {
        "num_nodes": len(pool.nodes),
        "num_requests": len(test_prompts),
        "model": "llama3.2"
    },
    "results": results
}

with open('sollol_routing_modes_benchmark.json', 'w') as f:
    json.dump(benchmark, f, indent=2)

print(f"\n✅ Mode comparison saved to sollol_routing_modes_benchmark.json")
for mode, data in results.items():
    print(f"{mode}: {data['avg_latency']:.2f}s avg, {data['p95_latency']:.2f}s P95")
```

---

### Template 4: Long-Running Stability Test

**What to measure**: Success rate, memory usage, performance degradation over time

```python
import time
import json
import psutil
from sollol import OllamaPool

pool = OllamaPool.auto_configure()

# Run for 1 hour (or longer!)
duration_hours = 1
num_requests = 1000
delay_between_requests = (duration_hours * 3600) / num_requests

results = []
process = psutil.Process()

print(f"Running {num_requests} requests over {duration_hours} hour(s)...")

start_time = time.time()
for i in range(num_requests):
    request_start = time.time()

    # Memory usage
    mem_mb = process.memory_info().rss / 1024 / 1024

    # Make request
    try:
        response = pool.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": f"Test request {i}"}]
        )
        success = True
        error = None
    except Exception as e:
        success = False
        error = str(e)

    request_duration = time.time() - request_start

    results.append({
        "request_num": i,
        "timestamp": time.time() - start_time,
        "success": success,
        "error": error,
        "latency_seconds": request_duration,
        "memory_mb": mem_mb
    })

    # Progress
    if (i + 1) % 100 == 0:
        success_rate = sum(1 for r in results if r["success"]) / len(results)
        print(f"  {i+1}/{num_requests} requests - {success_rate:.2%} success")

    time.sleep(delay_between_requests)

total_duration = time.time() - start_time
success_rate = sum(1 for r in results if r["success"]) / len(results)

benchmark = {
    "pattern": "stability_test",
    "date": "2025-11-02",
    "sollol_version": "0.9.61",
    "setup": {
        "duration_hours": duration_hours,
        "num_requests": num_requests,
        "num_nodes": len(pool.nodes)
    },
    "results": {
        "total_duration_seconds": total_duration,
        "success_rate": success_rate,
        "requests": results,
        "final_memory_mb": results[-1]["memory_mb"],
        "initial_memory_mb": results[0]["memory_mb"],
        "memory_growth_mb": results[-1]["memory_mb"] - results[0]["memory_mb"]
    }
}

with open('sollol_stability_benchmark.json', 'w') as f:
    json.dump(benchmark, f, indent=2)

print(f"\n✅ Stability test saved to sollol_stability_benchmark.json")
print(f"Success rate: {success_rate:.2%}")
print(f"Memory growth: {benchmark['results']['memory_growth_mb']:.1f} MB")
```

---

## What Makes a Good Benchmark?

### ✅ Good Benchmarks Include:

1. **Hardware details**: CPU/GPU specs, RAM, network setup
2. **SOLLOL version**: Output of `pip show sollol`
3. **Baseline comparison**: Single node vs distributed
4. **Multiple runs**: At least 3 runs to show consistency
5. **Real workload**: Actual use case, not synthetic
6. **Context**: What you're trying to accomplish
7. **Issues encountered**: Bugs, edge cases, surprises

### ❌ Avoid:

1. **Synthetic benchmarks**: "Hello world" × 1000 doesn't help
2. **No baseline**: We need to see the speedup
3. **Missing details**: "It's faster" without numbers
4. **One-off runs**: Could be luck, run multiple times
5. **Perfect results**: Share failures too! They help us improve

---

## Example: Great Benchmark Submission

```json
{
  "pattern": "batch_processing",
  "date": "2025-11-02",
  "submitted_by": "user@example.com",
  "sollol_version": "0.9.61",

  "use_case": "RAG document ingestion for legal documents",

  "hardware": {
    "node_1": {
      "host": "10.9.66.15",
      "cpu": "AMD Ryzen 9 5950X (16 cores)",
      "ram": "64GB DDR4",
      "gpu": "None",
      "os": "Ubuntu 22.04"
    },
    "node_2": {
      "host": "10.9.66.154",
      "cpu": "Intel i5-7200U (2 cores)",
      "ram": "16GB DDR4",
      "gpu": "None",
      "os": "Ubuntu 22.04"
    }
  },

  "workload": {
    "description": "Embedding 2453 document chunks from 21 PDF legal documents",
    "model": "mxbai-embed-large",
    "avg_chunk_length": 512,
    "total_chars": 1256320
  },

  "results": {
    "baseline_single_node": {
      "node": "10.9.66.154",
      "duration_seconds": 1656,
      "throughput": 0.18,
      "chunks_processed": 298,
      "chunks_failed": 0
    },
    "distributed_sollol": {
      "nodes": 2,
      "duration_seconds": 301,
      "throughput": 1.0,
      "chunks_processed": 298,
      "chunks_failed": 0,
      "speedup": 5.5
    }
  },

  "configuration": {
    "use_adaptive": true,
    "priority": 7,
    "observer_sampling": false
  },

  "issues_encountered": [
    "Initially lost 11.7% of chunks due to high concurrency",
    "Fixed with SOLLOL v0.9.61 adaptive concurrency",
    "Observability sampling added 50% overhead when enabled"
  ],

  "notes": "Speedup is high because node_1 is 2.8x faster than node_2. With identical slow nodes, expected speedup ~2x."
}
```

**Why this is great:**
- ✅ Real production workload
- ✅ Detailed hardware specs
- ✅ Baseline for comparison
- ✅ Honest about issues
- ✅ Explains why speedup is high
- ✅ Version information
- ✅ Configuration details

---

## How to Submit Benchmarks

### Option 1: GitHub Issue (Preferred)

1. Run benchmark using template above
2. Save JSON file
3. Create GitHub issue: https://github.com/B-A-M-N/SOLLOL/issues/new
4. Title: `[Benchmark] Pattern Name - Your Use Case`
5. Attach JSON file
6. Add any additional context in description

### Option 2: Email

Send benchmark JSON to: benchmarks@sollol.dev (or appropriate email)

Subject: `[Benchmark] Pattern Name - SOLLOL vX.X.X`

### Option 3: Pull Request

1. Fork SOLLOL repository
2. Add benchmark to `benchmarks/community/`
3. Update `benchmarks/README.md` with summary
4. Submit PR

---

## Benchmark Rewards

We value your contributions! For high-quality benchmarks:

- 🏆 **Recognition**: Listed in CONTRIBUTING.md
- 📊 **Featured**: Best benchmarks featured in docs
- 🎯 **Influence**: Help shape SOLLOL's roadmap
- ✅ **Graduation**: Move features from EXPERIMENTAL → PROVEN
- 🐛 **Bug Bounty**: Critical issues found get priority fixes

---

## Current Benchmark Needs (Priority Order)

### 🔥 High Priority

1. **Multi-agent orchestration** (SynapticLlamas pattern)
   - Need: Serial vs parallel speedup data
   - Missing: GPU controller impact measurements
   - Goal: Move from EXPERIMENTAL to PROVEN

2. **Routing mode comparison** (Hydra pattern)
   - Need: FAST vs RELIABLE vs ASYNC latency data
   - Missing: Optimal mode selection guidance
   - Goal: Provide evidence-based mode recommendations

3. **Hedging strategy**
   - Need: Latency reduction vs resource cost
   - Missing: Any production data
   - Goal: Validate hedging effectiveness

### 📊 Medium Priority

4. **Long-term stability** (All patterns)
   - Need: Days/weeks of continuous operation
   - Missing: Memory leak detection
   - Goal: Production confidence

5. **Large-scale batch processing**
   - Need: 10K+ items benchmarks
   - Missing: Scaling characteristics
   - Goal: Optimize for large batches

### 📝 Low Priority

6. **Different model comparisons**
   - Different embedding models
   - Different LLM sizes (1B, 3B, 7B, 13B)
   - Different quantization levels

---

## Questions?

- 💬 **Discord**: Join #benchmarks channel
- 📧 **Email**: benchmarks@sollol.dev
- 📝 **Docs**: See [DISTRIBUTED_OLLAMA_GUIDE.md](DISTRIBUTED_OLLAMA_GUIDE.md)
- 🐛 **Issues**: https://github.com/B-A-M-N/SOLLOL/issues

---

**Thank you for helping make SOLLOL better!** Every benchmark helps us validate features, find bugs, and provide better recommendations to the community.
