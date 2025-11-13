# Improving Multi-Node Ollama Embedding Throughput (19–21 emb/sec)

Hey all — I've been building **SOLLOL**, a production-grade orchestration and observability layer for distributed Ollama/llama.cpp deployments.

## Summary
Using 2 nodes (mixed CPU/GPU) with SOLLOL's adaptive batching and Dask integration, we're seeing **19–21 embeddings/sec** sustained throughput in testing — roughly **30× improvement** over single-node sequential embedding.

## What is SOLLOL?

SOLLOL consolidates the orchestration patterns I developed across multiple AI infrastructure projects into a single, reusable library. It provides:

* **Zero-config connection pooling** - Auto-discovers Ollama nodes on your network
* **Intelligent routing** - Task-aware load balancing with VRAM monitoring
* **Distributed batch processing** - Dask integration for large-scale workloads
* **Unified observability** - Real-time dashboard for node health, routing decisions, and performance
* **HTTP/2 multiplexing** - Connection reuse and concurrent request optimization
* **Adaptive strategies** - Automatically chooses optimal processing approach per workload

**Proven in production** via:
- **FlockParser** (document RAG pipeline) - processes PDF batches with mixed CPU/GPU nodes
- **SynapticLlamas** (multi-agent orchestration) - coordinates reasoning across distributed models

## Test Results

**Setup:** 2 Ollama nodes, mxbai-embed-large model

| Batch Size | Strategy | Throughput | vs ThreadPool |
|------------|----------|------------|---------------|
| 25 texts   | ThreadPoolExecutor | ~19 emb/sec | baseline |
| 50 texts   | ThreadPoolExecutor | ~21 emb/sec | baseline |
| 100 texts  | ThreadPoolExecutor | ~21 emb/sec | baseline |
| 200 texts  | Dask distributed | ~21 emb/sec | **1.46× faster** |
| 300 texts  | Dask distributed | ~21 emb/sec | scales linearly |

**Key optimizations:**
- HTTP/2 reduces latency ~30% on concurrent requests
- Connection reuse provides 10× speedup vs naive implementation
- Adaptive routing: local threads for <100 items, distributed for larger batches
- Worker-local pool caching eliminates Dask overhead

## Questions for Core Team

1. **Connection pooling:** Does Ollama benefit from HTTP/2 multiplexing, or is HTTP/1.1 with keep-alive sufficient?
2. **Concurrent limits:** What's the recommended max concurrent requests per Ollama instance?
3. **Internal batching:** Any insight into queue depth or batching limits in `ollama serve`?
4. **Persistent sessions:** Would maintaining long-lived connections improve throughput beyond keep-alive?
5. **Async API:** Any plans for native async/streaming embedding APIs?

## Why This Matters

Most Ollama deployments are single-node, but many of us are running multi-node clusters (home labs, small businesses, research teams). SOLLOL makes distributed Ollama deployments work like a single unified service - automatic discovery, intelligent routing, observability.

The goal is to make distributed AI inference as easy as single-node.

**Repo:** 🔗 [SOLLOL](https://github.com/B-A-M-N/SOLLOL)

**Related projects using SOLLOL:**
- [FlockParser](https://github.com/B-A-M-N/FlockParser) - Document processing pipeline
- [SynapticLlamas](https://github.com/B-A-M-N/SynapticLlamas) - Multi-agent reasoning

---

## Dashboard Screenshots

The SOLLOL unified dashboard provides real-time visibility into distributed operations:

**Screenshot 1:** Node health monitoring showing 2 active Ollama nodes (192.168.1.21, 192.168.1.10) with 11-12ms latency, 100% success rate

**Screenshot 2:** Active applications using SOLLOL - FlockParser and SynapticLlamas both connected and processing

**Screenshot 3:** Routing decisions with latency tracking (163ms-2.3sec per embedding) and integrated Dask/Ray dashboards

**Screenshot 4:** Dask task distribution showing distributed batch processing across cluster workers

The system automatically routes small batches locally and distributes large batches across the cluster for optimal throughput.

---

Any feedback from the Ollama team or community on best practices for high-throughput distributed deployments would be hugely appreciated!
