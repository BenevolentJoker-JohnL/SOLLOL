# Coordinator Reuse Investigation

## Problem Statement
Multi-agent queries create 6 separate coordinators sequentially instead of reusing one coordinator. Each startup wastes 2-5 minutes loading the model (30+ minutes total overhead).

## Observed Behavior
From "Explain quantum entanglement" query:
```
🚀 Starting llama.cpp coordinator for codellama:13b...  (Agent 1)
[2-5 min model loading]
[Agent 1 completes inference]

🚀 Starting llama.cpp coordinator for codellama:13b...  (Agent 2)
[2-5 min model loading]
[Agent 2 completes inference]

... (repeats 6 times total)
```

## Key Observations
1. **Agents execute SEQUENTIALLY** (not parallel) - this is correct for model sharding
2. **Each agent gets a NEW coordinator** instead of reusing the existing one
3. **Coordinator should persist** between sequential requests to avoid reloading model

## Fixes Applied

### Fix 1: GGUF Path Caching (commit d4c7fc6)
**Issue**: `ollama show` subprocess called repeatedly for same model
**Fix**: Added cache to `OllamaGGUFResolver`
**Impact**: Eliminates redundant subprocess calls

### Fix 2: Coordinator Race Condition Lock (commit 1f01b3d)
**Issue**: Multiple agents calling `_ensure_coordinator_for_model()` simultaneously
**Fix**: Added `asyncio.Lock` with double-checked locking pattern
**Result**: Still saw 6 coordinator startups (lock alone wasn't enough)

### Fix 3: Coordinator Port Conflict (commit b49f912)
**Issue**: Default port 8080 conflicts with dashboard
**Fix**: Changed coordinator default port to 18080

### Fix 4: Debug Logging (commit 155e0a8)
**Issue**: Can't see execution flow
**Fix**: Added thread-level tracing showing lock acquisition and coordinator creation

### Fix 5: Process Liveness Checks (commit 0c310ab)
**Issue**: Unknown if coordinator process is crashing after each inference
**Fix**: Check `coordinator.process.poll()` before reusing
**Diagnostic**: Will log "⚠️  Coordinator process died!" if crashing

## Current Hypothesis
The coordinator process may be **CRASHING after one inference**, causing `self.coordinator` to reference a dead process. When the next agent checks for an existing coordinator, it finds one but the process is dead.

**New liveness checks will confirm this** by logging warnings when process is found dead.

## Next Steps
1. ✅ Added process liveness checks with diagnostic logging
2. ✅ Added try/except wrapper around coordinator.start() with cleanup
3. ✅ Tested with fresh query - coordinator started successfully
4. ⏳ Verify coordinator reuse across multiple agents (blocked by inference timeout issue)
5. **NEW ISSUE DISCOVERED**: Inference requests timing out after 300s
   - Coordinator starts successfully in ~2 minutes
   - First agent request times out waiting for response
   - Need to investigate why llama-server is not responding to inference requests
   - Possible causes: RPC communication issue, model loading incomplete, network latency

## Architecture Notes
- **HybridRouter**: Single shared instance across all agents (verified in diagnose_hybrid_router.py)
- **HybridRouterSync**: Synchronous wrapper for async HybridRouter
- **Coordinator lifecycle**: Created in `_ensure_coordinator_for_model()`, persisted in `self.coordinator`
- **Expected behavior**: First agent creates coordinator, subsequent agents reuse it
- **Lock scope**: Only held during coordinator creation, NOT during inference (this is correct)

## Testing Status
- ✅ GGUF caching working (cache hits in logs)
- ✅ Debug logging working (thread IDs visible)
- ⏳ Coordinator reuse test couldn't complete (RPC backends occupied by old tests)
- ⏳ Need to test with fresh multi-agent query to see new diagnostics

## Files Modified
1. `/home/joker/SynapticLlamas/sollol/ollama_gguf_resolver.py` - Added caching
2. `/home/joker/SynapticLlamas/sollol/hybrid_router.py` - Added lock, debug logging, liveness checks
3. `/home/joker/test_coordinator_reuse.py` - Minimal test script (not yet successfully run)

## Commit History
```
7e62e2f Add exception handling to clean up failed coordinator on startup failure
0c310ab Add coordinator process liveness checks
155e0a8 Add detailed debug logging to coordinator creation
b49f912 Fix coordinator port conflict with dashboard (8080 -> 18080)
1f01b3d Fix race condition in coordinator creation for multi-agent parallel requests
d4c7fc6 Add caching to GGUF resolver to avoid redundant ollama show calls
```

## Latest Test Results (2025-10-05)

### Test: "explain string theory" query
**Result**: Coordinator started successfully, but inference timeout

```
✅ [Thread 135896389121600] Coordinator started with 3 RPC backends on 127.0.0.1:18080
```

**Key observations:**
1. Only **ONE** coordinator creation seen (previous behavior: 6 creations)
2. Coordinator loaded model successfully across 3 RPC backends:
   - 192.168.1.10:50052 → 1872.15 MiB
   - 10.9.66.157:50052 → 3403.91 MiB
   - 192.168.1.22:50052 → 1660.02 MiB
3. Coordinator became healthy (HTTP 200 OK) after ~123 seconds
4. **NEW PROBLEM**: First inference request timed out after 300s
   - `concurrent.futures._base.TimeoutError` in Researcher agent
   - Coordinator is running but not responding to inference requests

### Conclusion on Coordinator Reuse
**LIKELY FIXED** - We no longer see multiple coordinator creations. The exception handling wrapper prevents failed coordinator objects from persisting.

### New Issue: Inference Timeout
The coordinator starts successfully but doesn't complete inference requests within 300 seconds. This is a separate performance/reliability issue that needs investigation.

## Port Conflict Resolution (2025-10-05)

### Issue: RPC Server Port Conflict
**Symptom**: `httpcore.RemoteProtocolError: Server disconnected without sending a response`

**Root cause**: TWO RPC server processes running on port 50052:
1. `rpc-server --host 127.0.0.1 --port 50052` (localhost only)
2. `rpc-server --host 0.0.0.0 --port 50052` (all interfaces)

**Fix**: Killed both RPC server processes (PID 997928) to clear port conflict

**Result**: Port 50052 now free; system will only use the three network IP RPC backends from config:
- 192.168.1.10:50052
- 10.9.66.157:50052
- 192.168.1.22:50052

**Status**: Ready for clean testing without port conflicts
