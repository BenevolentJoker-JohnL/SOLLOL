### Description

I've developed **SOLLOL**, an orchestration and observability layer for distributed Ollama/llama.cpp deployments. After achieving 19-21 embeddings/sec throughput across multi-node clusters, I have questions about optimizing connection patterns and understanding Ollama's internal behavior for distributed workloads.

### Context

**Project:** Production-grade connection pooling and load balancing for multi-node Ollama setups
**Use case:** Document embedding pipelines (FlockParser) and multi-agent systems (SynapticLlamas)
**Current setup:** 2-3 mixed CPU/GPU nodes on local network

### Current Implementation

```python
# HTTP/2 with connection reuse
session = httpx.Client(
    transport=httpx.HTTPTransport(retries=3, http2=True),
    timeout=httpx.Timeout(300.0, connect=10.0),
    limits=httpx.Limits(
        max_keepalive_connections=40,
        max_connections=100,
        keepalive_expiry=30.0
    )
)

# Adaptive batching
- Small batches (<100 texts): ThreadPoolExecutor with 2 workers per node
- Large batches (>100 texts): Dask distributed processing
```

### Performance Results

**Test setup:** 2 Ollama nodes, `mxbai-embed-large` model

| Batch Size | Strategy | Throughput | Notes |
|------------|----------|------------|-------|
| 25 texts | ThreadPoolExecutor | ~19 emb/sec | baseline |
| 50 texts | ThreadPoolExecutor | ~21 emb/sec | baseline |
| 100 texts | ThreadPoolExecutor | ~21 emb/sec | baseline |
| 200 texts | Dask distributed | ~21 emb/sec | 1.46× faster than ThreadPool |
| 300 texts | Dask distributed | ~21 emb/sec | scales linearly |

**Key optimizations:**
- HTTP/2 multiplexing: ~30% latency reduction on concurrent requests
- Connection reuse: 10× speedup vs naive implementation
- Worker-local pool caching: eliminates Dask serialization overhead

### Questions

1. **Connection pooling:** Does `ollama serve` benefit from HTTP/2 multiplexing, or is HTTP/1.1 with keep-alive equally effective? Are there any connection-level optimizations we should be aware of?

2. **Concurrency limits:** What's the recommended maximum concurrent requests per Ollama instance? Are there internal queues or throttling mechanisms we should tune for?

3. **Request batching:** Does Ollama perform any internal batching of embedding requests? Understanding this would help optimize our client-side batching strategy.

4. **Connection lifecycle:** Would maintaining persistent/long-lived connections provide benefits beyond keep-alive headers? Do connections maintain any state between requests?

5. **Async API plans:** Are there plans for native async/streaming embedding APIs? This would allow more efficient non-blocking I/O patterns.

### Why This Matters

Many teams are running multi-node Ollama clusters (home labs, small businesses, research environments) but lack tooling for unified orchestration. SOLLOL aims to make distributed inference as simple as single-node deployments through:

- Zero-config node discovery
- Intelligent load balancing with VRAM awareness
- Real-time observability and metrics
- Adaptive routing strategies

Understanding Ollama's connection behavior and internal architecture would help optimize distributed client implementations.

### Links

- **SOLLOL:** https://github.com/BenevolentJoker-JohnL/SOLLOL
- **FlockParser** (document processing): https://github.com/BenevolentJoker-JohnL/FlockParser
- **SynapticLlamas** (multi-agent): https://github.com/BenevolentJoker-JohnL/SynapticLlamas

### Screenshots

<details>
<summary>SOLLOL Dashboard Screenshots</summary>

**Node health monitoring:**
- 2 active Ollama nodes with latency and status tracking
- 100% success rate across distributed requests

**Real-time routing decisions:**
- Activity logs showing request/response patterns
- Latency tracking per embedding request (163ms-2.3sec)
- Integrated Dask/Ray cluster visualization

**Distributed processing:**
- Task distribution across cluster workers
- Automatic adaptive routing (local threads vs distributed)

</details>

---

Any insights from the Ollama team on optimizing distributed deployments would be greatly appreciated. Happy to provide more details or testing data if helpful.
