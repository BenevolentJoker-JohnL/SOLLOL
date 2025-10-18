# Best Practices for Concurrent Embeddings at Scale

## Summary
We're building a distributed embedding system that processes large document batches across multiple Ollama nodes (2-5 nodes, mixed CPU/GPU). We've achieved 10-20x speedups with ThreadPoolExecutor + connection pooling, but wanted to check with the core team on best practices.

## Current Setup

**Architecture:**
- 2-5 Ollama nodes on local network
- Mixed CPU (slow, high capacity) + GPU (fast, limited VRAM) nodes
- Processing 100-1000+ document chunks per batch
- Using `mxbai-embed-large` primarily

**Optimizations Applied:**
```python
import httpx

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

# Parallel embedding with ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=node_count * 2) as executor:
    futures = [executor.submit(embed_single, text) for text in texts]
    results = [f.result() for f in as_completed(futures)]
```

**Performance:**
- Small batches (25 texts): ~19 emb/sec
- Medium batches (100 texts): ~21 emb/sec
- Large batches (200+ texts): ~20 emb/sec sustained

## Questions for Core Team

1. **Concurrent Connections:**
   - What's the recommended max concurrent requests per Ollama instance?
   - Any internal request queuing/throttling we should be aware of?

2. **Connection Pooling:**
   - Does Ollama benefit from HTTP/2 multiplexing, or is HTTP/1.1 with keep-alive sufficient?
   - Recommended keep-alive timeout settings?

3. **Model Loading:**
   - Best way to pre-warm models across nodes? (currently using minimal inference)
   - Any `/api/ps` or `/api/tags` optimizations for health checks?

4. **VRAM Management:**
   - Any built-in load shedding when VRAM is low?
   - Best way to detect VRAM saturation vs normal processing?

5. **Streaming vs Batch:**
   - For embeddings specifically, any performance difference between streaming and non-streaming mode?
   - Does batching multiple texts into single requests help vs parallel single requests?

## Context

We're building this for **FlockParser** (document RAG pipeline) and **SOLLOL** (connection pool library). Both are open source and aim to make distributed Ollama deployments production-ready. Any guidance on scaling best practices would be incredibly helpful!

**Repos:**
- FlockParser: https://github.com/BenevolentJoker-JohnL/FlockParser
- SOLLOL: https://github.com/BenevolentJoker-JohnL/SOLLOL

## What We've Learned

**What works:**
- HTTP/2 reduces latency by ~30% on concurrent requests
- ThreadPoolExecutor with 2 workers per node is optimal
- Connection reuse is critical (10x speedup)
- Response caching for repeated queries

**What doesn't:**
- Naive serial processing (painfully slow)
- Too many workers (>4 per node) causes contention
- Cold model loading adds 1-5s first-request latency

---

Would love to hear from the team or community on best practices for high-throughput distributed deployments! 🚀
