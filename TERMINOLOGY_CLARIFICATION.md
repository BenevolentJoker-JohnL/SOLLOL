# SOLLOL Distributed Inference - Terminology Clarification

**Date:** October 20, 2025
**Status:** Documentation update complete

## The Issue

The term "model sharding" has been used inconsistently throughout SOLLOL documentation and code, leading to confusion about the actual capability.

## What SOLLOL Actually Does: **Distributed Inference**

**Current Capability (v0.7.x):**
- **Distributed Inference (Layer Distribution)**
  - Model layers are distributed across RPC backends for parallel computation
  - llama.cpp coordinator loads the full model into memory
  - RPC workers handle distributed layer computation
  - ✅ Validated: 13B models with layers spread across 2-3 nodes

**What This Is NOT:**
- ❌ Model weight sharding (splitting model parameters across nodes)
- ❌ Distributed model storage (no single node needs full model)
- ❌ True parameter-level parallelism

## Updated Terminology

### ✅ Correct Terms:
- **Distributed Inference** - The overall capability
- **Layer Distribution** - How model layers are spread across RPC backends
- **Inference Parallelism** - Parallel computation of model layers
- **RPC Backend Coordination** - How llama.cpp coordinator manages workers

### ❌ Deprecated/Misleading Terms:
- ~~Model Sharding~~ (implies weight splitting, not layer distribution)
- ~~Parameter Sharding~~ (not implemented)
- ~~Distributed Model Loading~~ (coordinator still loads full model)

## Configuration Keys (Backward Compatible)

**Note:** Configuration keys like `model_sharding_enabled` are kept for backward compatibility but should be understood as enabling **distributed inference**, not true model weight sharding.

```json
{
  "model_sharding_enabled": true,  // Actually enables distributed inference
  "rpc_backends": [...]            // Backends for layer distribution
}
```

**Recommended new naming (for future major version):**
```json
{
  "distributed_inference_enabled": true,
  "rpc_backends": [...]
}
```

## Architecture Clarification

### Current: Distributed Inference
```
┌──────────────────┐
│   Coordinator    │ ← Loads FULL model (e.g., 7GB for codellama:13b)
│  (llama-server)  │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼───┐
│RPC #1│  │RPC #2│ ← Handle distributed layer computation
│Worker│  │Worker│   (layers spread across these backends)
└──────┘  └──────┘
```

- **Memory requirement:** Coordinator needs full model (13GB for codellama:13b)
- **Benefit:** Parallel layer computation across workers
- **Limitation:** Coordinator node must have sufficient RAM

### Future: True Model Weight Sharding
```
┌──────────────────┐
│   Coordinator    │ ← Only stores routing logic (minimal memory)
│  (orchestrator)  │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼───┐
│Node 1│  │Node 2│ ← Each stores PART of model weights
│Layers│  │Layers│   (e.g., layers 0-20 vs 21-40)
│ 0-20 │  │21-40 │
└──────┘  └──────┘
```

- **Memory requirement:** NO single node needs full model
- **Benefit:** Can run 70B+ models on nodes with <16GB RAM each
- **Status:** Experimental research track (`distributed_pipeline.py`)

## Documentation Updates Applied

All documentation has been updated to use correct terminology:

✅ **Updated files:**
- `README.md` - Clarified distributed inference vs future weight sharding
- `ARCHITECTURE.md` - Updated all "model sharding" → "distributed inference"
- `CONFIGURATION.md` - Clarified RPC backends purpose
- `DASHBOARD_RPC_FIX.md` - Updated panel descriptions
- `RPC_BACKEND_DISCOVERY_COMPLETE.md` - Terminology corrections
- `UNIFIED_OBSERVABILITY.md` - Terminology corrections

## Code Comments (Preserved for Compatibility)

Python code uses `model_sharding_enabled` in variable names for backward compatibility, but docstrings and comments now clarify this enables **distributed inference**, not weight sharding.

**Example:**
```python
# model_sharding_enabled: bool
#     Enable distributed inference (layer distribution across RPC backends).
#     Note: This is NOT model weight sharding - coordinator still needs full model.
```

## User Impact

**For existing users:**
- ✅ No breaking changes - all config keys work as before
- ✅ Clearer understanding of actual capabilities
- ✅ Better expectations for memory requirements

**For new users:**
- ✅ Accurate documentation of what SOLLOL can do
- ✅ Clear roadmap for future capabilities
- ✅ Proper terminology throughout docs

## Summary

**SOLLOL's "model sharding" feature should be understood as:**
- **Distributed Inference** using llama.cpp's RPC architecture
- Layers are distributed across backends for parallel computation
- Coordinator still loads the full model into memory
- Validated for 13B models across 2-3 nodes
- Future work will add true weight sharding (no full-model requirement)

---

**Related Issues:**
- Coordinator memory requirements for large models
- Future research and development needed for true weight sharding implementation
- See `distributed_pipeline.py` for research track

**Last Updated:** October 20, 2025
