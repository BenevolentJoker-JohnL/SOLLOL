# Experimental Features - Use at Your Own Risk

**Last Updated:** October 20, 2025
**Status:** Not production-ready, funding required for optimization

---

## ⚠️ Important Notice

The features documented here are **experimental** and **not recommended for production use**. They have been validated for basic functionality only and have significant limitations that require substantial engineering effort to resolve.

**Our Recommendation:** Use SOLLOL's stable features (task distribution, intelligent routing, observability dashboard) for production workloads.

---

## 🏗️ Production-Ready Applications Using SOLLOL

Before diving into experimental features, note that SOLLOL's **stable features** power two production-ready applications:

### [FlockParser](https://github.com/BenevolentJoker-JohnL/FlockParser)
**Distributed PDF Processing & RAG System**
- Uses SOLLOL for distributed embedding generation
- Load balances document processing across Ollama cluster
- Production-stable, actively maintained

### [SynapticLlamas](https://github.com/BenevolentJoker-JohnL/SynapticLlamas)
**Multi-Agent Collaboration Framework**
- Uses SOLLOL for parallel agent execution
- Distributes research, critique, and synthesis tasks
- Production-stable workflow orchestration

> **Both applications use ONLY the production-stable SOLLOL features** (task distribution, intelligent routing). They avoid the experimental features documented below.

### 🔧 In Development

The following projects are being integrated with SOLLOL and will enable future development:

- **Hydra**: Integration in progress - requires additional work
- **LlamaForge**: Integration in progress - requires additional work

These projects will leverage SOLLOL's stable features and contribute to future capabilities.

---

## Distributed Inference via llama.cpp RPC

### What It Is

Distribute model layer computation across multiple RPC backends using llama.cpp's RPC architecture.

**Note:** This is **NOT** model weight sharding. The coordinator still loads the full model into memory.

### Current Status: Experimental (Not Production-Ready)

**✅ What Works:**
- Basic layer distribution across 2-3 RPC backends
- Successfully validated with codellama:13b (13B parameters)
- Coordinator auto-start and lifecycle management
- RPC backend health monitoring

**❌ Known Issues & Limitations:**
1. **Performance:**
   - Startup: 2-5 minutes (vs 20 seconds local)
   - Inference: ~5 tok/s (vs ~20 tok/s local)
   - **5x slower than running locally** - defeats the purpose

2. **Reliability:**
   - Version sensitivity - coordinator and rpc-server must be exact same build
   - Frequent crashes with version mismatches (exit code -11)
   - Creates zombie processes that require manual cleanup
   - No automatic recovery from backend failures

3. **Operational Complexity:**
   - Requires manual configuration (rpc_backends.conf file)
   - Manual binary management across nodes
   - Complex troubleshooting when things break
   - No automated version management

4. **Memory Requirements:**
   - **Coordinator still needs full model** (e.g., 13GB for codellama:13b)
   - Does NOT solve the "model too big for single node" problem
   - RPC workers need less memory, but coordinator is the bottleneck

5. **Limited Testing:**
   - Only validated with 13B models
   - Only tested with 2-3 backends
   - No testing with GPU backends
   - No production validation

### Why It's Not Production-Ready

The distributed inference feature was implemented as a proof-of-concept to validate llama.cpp's RPC architecture. While it technically works for simple cases, it has fundamental issues:

1. **It's slower than not using it** - defeats the main benefit
2. **High maintenance burden** - version conflicts, manual setup, frequent troubleshooting
3. **No performance optimization** - startup and inference times not tuned
4. **Coordinator memory bottleneck** - still requires full model on one node

### What Would Be Needed for Production

To make distributed inference production-ready would require:

**Phase 1: Optimization (Estimated 2-3 months, funding required)**
- Performance tuning to reduce startup time from 2-5min to <30s
- Inference optimization to match or exceed local performance
- Automated version management and compatibility checking
- Robust error handling and automatic recovery
- Comprehensive testing across model sizes (7B-70B+)
- GPU backend validation
- Load testing and stress testing

**Phase 2: True Model Weight Sharding (Estimated 6-12 months, significant funding)**
- Eliminate coordinator memory bottleneck
- Enable true distributed model loading (no single node needs full model)
- See `distributed_pipeline.py` for research track

**Cost Estimate:** $50k-$100k for Phase 1, $200k+ for Phase 2

### How to Use (If You Insist)

**Prerequisites:**
- Identical llama.cpp build on ALL nodes (coordinator + all RPC backends)
- Sufficient RAM on coordinator node for full model
- Network connectivity between all nodes
- Patience for slow startup and inference

**Setup Steps:**

1. **Build llama.cpp (same version everywhere):**
   ```bash
   # On ALL nodes:
   cd /path/to/llama.cpp
   git checkout <exact-same-commit>
   cmake -B build -DGGML_CUDA=OFF -DLLAMA_BUILD_SERVER=ON
   cmake --build build
   ```

2. **Start RPC servers on worker nodes:**
   ```bash
   # On each worker node:
   /path/to/llama.cpp/build/bin/rpc-server --host 0.0.0.0 --port 50052
   ```

3. **Configure RPC backends on coordinator node:**
   ```bash
   # Create /path/to/SOLLOL/rpc_backends.conf:
   10.9.66.48:50052
   10.9.66.154:50052
   ```

4. **Start SOLLOL with distributed inference enabled:**
   ```json
   {
     "model_sharding_enabled": true,
     "rpc_backends": [
       {"host": "10.9.66.48", "port": 50052},
       {"host": "10.9.66.154", "port": 50052}
     ]
   }
   ```

**Troubleshooting Common Issues:**

**Coordinator crashes with exit -11:**
- Version mismatch between coordinator and rpc-server
- Solution: Rebuild all binaries from same commit

**Zombie llama-server processes:**
- Coordinator crashed during model loading
- Solution: `pkill -9 llama-server`, fix version mismatch, retry

**Slow performance:**
- Expected - not optimized
- No solution without funding for optimization work

**Connection refused:**
- RPC server not running on worker node
- Firewall blocking port 50052
- Check network connectivity

### Realistic Expectations

**What you'll experience:**
- ⏱️ 2-5 minute startup time (grab coffee)
- 🐌 ~5 tokens/second inference (could read email while waiting)
- 🔧 Frequent troubleshooting sessions
- 💾 Coordinator still needs 13GB+ RAM
- 😤 Frustration when versions mismatch

**What you won't get:**
- ❌ Production-grade performance
- ❌ Automatic version management
- ❌ Ability to run models bigger than your largest node
- ❌ Support without hiring dedicated engineers

### Alternative: Use Task Distribution Instead

For production workloads, use SOLLOL's **stable task distribution features**:

```python
from sollol import OllamaPool

# This is fast, reliable, and production-tested
pool = OllamaPool.auto_configure()

# Distribute 10 concurrent requests across nodes
responses = await asyncio.gather(*[
    pool.chat(model="llama3.2", messages=[...])
    for _ in range(10)
])
```

**Benefits:**
- ✅ 5-20ms routing overhead (vs 2-5min startup)
- ✅ Full local inference speed on each node
- ✅ Automatic failover and health monitoring
- ✅ No version management needed
- ✅ Production-tested and stable

---

## Funding & Partnership Opportunities

If you're interested in funding the optimization and production readiness of distributed inference, or exploring true model weight sharding, please open an issue or contact via GitHub.

**What funding would enable:**
- Phase 1: Production-ready distributed inference ($50k-$100k)
- Phase 2: True model weight sharding ($200k+)
- Ongoing maintenance and support

**Why this matters:**
- Enables running frontier models (70B-405B) on consumer hardware clusters
- Sovereign AI deployment without cloud dependencies
- Local infrastructure for teams and organizations

---

## Summary

**Use SOLLOL for:**
- ✅ Task distribution (proven, stable, fast)
- ✅ Intelligent routing (adaptive, optimized)
- ✅ Observability (dashboard, metrics, logs)

**Don't use SOLLOL for (yet):**
- ❌ Distributed inference (experimental, slow, brittle)
- ❌ Running models bigger than your largest node (coordinator bottleneck)
- ❌ Production workloads requiring distributed layer computation

**The honest truth:**
- Distributed inference works for demos and testing
- It's not ready for real-world use
- Optimization requires funding
- Task distribution is the stable, recommended approach

For questions or to report issues with experimental features, open a GitHub issue with the `experimental` label.
