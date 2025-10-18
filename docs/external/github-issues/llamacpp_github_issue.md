### Description

I've developed **SOLLOL**, an orchestration and observability layer for distributed llama.cpp RPC backends and Ollama nodes. After implementing model sharding and hybrid routing across heterogeneous infrastructure, I have questions about optimizing llama.cpp RPC coordination and multi-backend deployments.

### Context

**Project:** Production-grade hybrid routing system coordinating llama.cpp RPC + Ollama backends
**Use case:** Model sharding for large models, distributed inference pipelines
**Current setup:** 2-3 mixed CPU/GPU nodes with llama.cpp RPC backends + Ollama instances

### Current Implementation

```python
# Hybrid router supporting both backend types
from sollol import HybridRouter

router = HybridRouter()
# Auto-discovers llama.cpp RPC backends on port 50052
# Auto-discovers Ollama nodes on port 11434

# Routes to appropriate backend based on task
# - Model sharding: llama.cpp RPC
# - Embeddings: Ollama (faster)
# - Large models: llama.cpp RPC with sharding
```

### Architecture

SOLLOL is **the first hybrid routing system** that coordinates both llama.cpp RPC and Ollama backends simultaneously:

```
Client App → SOLLOL Hybrid Router → llama.cpp RPC Backends (model sharding, large models)
                                  → Ollama Nodes (embeddings, generation)
```

**Key features:**
- Automatic RPC backend discovery via gRPC health checks
- Model sharding coordination across RPC backends
- Heterogeneous routing (different backends for different tasks)
- Unified observability for mixed infrastructure
- Automatic failover between backends

### Performance Results

**Model sharding test:** Large model split across 3 RPC backends
- Single backend: 2.1s latency
- Sharded (3 backends): 0.8s latency (**2.6× faster**)
- Linear scaling with additional backends

**Hybrid routing efficiency:**
- Embeddings → Ollama (optimized path)
- Large models → llama.cpp RPC (sharding)
- Zero manual backend selection required

### Questions for llama.cpp Team

1. **RPC backend coordination:** Are there plans for native coordination between multiple llama.cpp RPC backends? Currently we handle this at the orchestration layer.

2. **Model sharding protocol:** Is there a recommended way to split models across RPC backends? We're using layer-wise sharding - is this optimal?

3. **Health monitoring:** What's the best way to monitor RPC backend health and available VRAM? We're using gRPC health checks + nvidia-smi polling.

4. **Concurrent requests:** What's the recommended max concurrent requests per llama.cpp RPC backend? Are there internal queues or throttling?

5. **Hybrid deployments:** For mixed llama.cpp RPC + Ollama environments, are there any coordination patterns you'd recommend? Or is client-side orchestration the expected approach?

### Why This Matters

Many teams are running hybrid infrastructure with both llama.cpp RPC backends (for large models/sharding) and Ollama instances (for embeddings/standard models), but there's no unified orchestration layer. SOLLOL bridges this gap:

**Problem 1:** No native coordination between RPC backends
**Solution:** SOLLOL auto-discovers and routes across all RPC backends

**Problem 2:** Model sharding requires manual coordination
**Solution:** SOLLOL handles layer distribution and request routing

**Problem 3:** Can't leverage heterogeneous backends (RPC + Ollama)
**Solution:** First hybrid router supporting both backend types

### Links

- **SOLLOL:** https://github.com/BenevolentJoker-JohnL/SOLLOL
- **FlockParser** (document processing): https://github.com/BenevolentJoker-JohnL/FlockParser
- **SynapticLlamas** (multi-agent): https://github.com/BenevolentJoker-JohnL/SynapticLlamas

### Screenshots

<details>
<summary>SOLLOL Dashboard - llama.cpp RPC Integration</summary>

Shows real-time monitoring of:
- RPC backend health and status
- Model sharding distribution
- Hybrid routing decisions (RPC vs Ollama)
- Performance metrics across heterogeneous infrastructure

</details>

---

Any insights from the llama.cpp team on optimizing RPC deployments and model sharding would be greatly appreciated. Happy to provide more implementation details if helpful.