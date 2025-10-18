# Improving Multi-Node Ollama Embedding Throughput (19–21 emb/sec)

Hey all — I've been building **SOLLOL**, an orchestration and observability layer for Ollama / llama.cpp nodes. It's part of my larger AI infrastructure stack, alongside **FlockParser** (parallel embedding pipeline) and **SynapticLlamas** (distributed reasoning proof of concept).

## Summary
Using 3 nodes (1 GPU, 2 CPU) with Dask and adaptive batching, we're sustaining **19–21 embeddings/sec** — a ~30× throughput improvement over single-node sequential embedding.

## Current Setup

* Dask orchestrates distributed Ollama clients
* Each node runs an `ollama serve` endpoint
* Parallel batching with retry/failover logic
* Cached embeddings (MD5 batch caching)
* Monitoring via SOLLOL's dashboard

## Questions

1. Does Ollama currently reuse connections between embedding requests?
2. Are there recommended pool settings for concurrent requests?
3. Any insight into internal batching limits or queue depth in `ollama serve`?
4. Would persistent sessions improve throughput, or is every request stateless?
5. Any known plans for embedding stream APIs or async client support?

## Context
SOLLOL is intended as the *software layer* for AI infrastructure — handling orchestration, monitoring, and adaptive load-balancing for multi-node setups.

## What We've Learned So Far

* Legacy FlockParser with `embed_batch()` was 12× faster than SOLLOL's single-embed calls
* Adding Dask's distributed batching scaled that further (61× over baseline)
* Most bottlenecks trace back to Ollama connection overhead

**Repos:**
- 🔗 [SOLLOL](https://github.com/BenevolentJoker-JohnL/SOLLOL)
- 🔗 [FlockParser](https://github.com/BenevolentJoker-JohnL/FlockParser)
- 🔗 [SynapticLlamas](https://github.com/BenevolentJoker-JohnL/SynapticLlamas)

Any feedback or insights from the Ollama devs or others doing distributed orchestration would be massively appreciated.

---

## Follow-up Message (Post After ~8 Hours)

Here's the dashboard output — 10 PDFs processed in 2 minutes using mixed CPU/GPU nodes.

[Screenshot showing: SOLLOL dashboard with node metrics, throughput graph, and batch processing stats]

**Key metrics visible:**
- Node health/load distribution
- Real-time embedding throughput
- VRAM utilization across GPU nodes
- Batch processing progress

The adaptive routing automatically sends large batches (>100 texts) to Dask workers for distributed processing, while small batches use local ThreadPoolExecutor to avoid overhead.
