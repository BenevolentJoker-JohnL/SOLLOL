# Experimental Features - Quick Summary

**See [EXPERIMENTAL_FEATURES.md](../EXPERIMENTAL_FEATURES.md) for full details**

## TL;DR

**Distributed Inference Status:** Experimental, not production-ready

**What it is:** Layer distribution across RPC backends (NOT model weight sharding)

**Known Issues:**
- 5x slower than local inference
- 2-5 minute startup time
- Version-sensitive (exact build match required)
- Coordinator still needs full model in memory
- Frequent crashes with version mismatches

**Production-Ready Alternatives:**
- Use task distribution (stable, fast, reliable)
- Run models locally on your best node
- Wait for funding to optimize RPC features

**Recommendation:** Don't use distributed inference for production without dedicated engineering resources.

---

For realistic expectations and troubleshooting, see the full [EXPERIMENTAL_FEATURES.md](../EXPERIMENTAL_FEATURES.md) document.
