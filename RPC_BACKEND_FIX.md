# RPC Backend "undefined" Fix

## Problem

The unified dashboard was displaying "undefined" for llama.cpp RPC backend entries instead of showing proper metadata (host, port, latency, request count, etc.).

## Root Cause

The issue was in `src/sollol/unified_dashboard.py` where the code was incorrectly iterating over `registry.backends`:

```python
# BROKEN CODE (line 349)
for backend in registry.backends:
    host = backend["host"]  # ❌ FAILS - backend is a string, not a dict!
```

**Why this failed:**
- `registry.backends` is a `Dict[str, RPCBackend]` (dictionary mapping addresses to backend objects)
- When you iterate over a dict with `for x in dict:`, you get the **keys** (strings), not the values
- So `backend` was a string like `"192.168.1.21:50052"`, not an `RPCBackend` object
- Trying to access `backend["host"]` on a string returned `undefined` in the dashboard

## Solution

Fixed in 3 locations in `unified_dashboard.py`:

### 1. HTTP API `/api/network/backends` (line 345-365)

```python
# FIXED CODE
for backend_obj in registry.backends.values():
    backend_dict = backend_obj.to_dict()
    host = backend_dict["host"]
    port = backend_dict["port"]
    is_healthy = backend_dict["healthy"]
    metrics = backend_dict.get("metrics", {})
    backends.append({
        "url": f"{host}:{port}",
        "status": "healthy" if is_healthy else "offline",
        "latency_ms": metrics.get("avg_latency_ms", 0),
        "request_count": metrics.get("total_requests", 0),
        "failure_count": metrics.get("total_failures", 0),
    })
```

### 2. WebSocket `/ws/network/backends` (line 742-753)

```python
# FIXED CODE
for backend_obj in registry.backends.values():
    backend_addr = f"{backend_obj.host}:{backend_obj.port}"
    if backend_addr not in backends:
        backends.append(backend_addr)
```

### 3. WebSocket `/ws/network/rpc_activity` (line 951-960)

```python
# FIXED CODE
backends_to_monitor = [(f"{b.host}:{b.port}", f"{b.host}:{b.port}")
                     for b in registry.backends.values()]
```

## Changes Made

### Files Modified
- `src/sollol/unified_dashboard.py` (3 fixes)

### Files Added
- `tests/unit/test_rpc_backend_metadata.py` (5 unit tests)
- `tests/integration/test_dashboard_rpc_backends.py` (5 integration tests)

## Test Results

### Unit Tests (5/5 passed)
```bash
$ python -m pytest tests/unit/test_rpc_backend_metadata.py -v
tests/unit/test_rpc_backend_metadata.py::TestRPCBackendMetadata::test_backend_to_dict_structure PASSED
tests/unit/test_rpc_backend_metadata.py::TestRPCBackendMetadata::test_registry_backends_iteration PASSED
tests/unit/test_rpc_backend_metadata.py::TestRPCBackendMetadata::test_backend_metrics_tracking PASSED
tests/unit/test_rpc_backend_metadata.py::TestRPCBackendMetadata::test_registry_get_stats PASSED
tests/unit/test_rpc_backend_metadata.py::TestDiscoveryMetadata::test_discovery_return_structure PASSED
```

### Integration Tests (5/5 passed)
```bash
$ python -m pytest tests/integration/test_dashboard_rpc_backends.py -v
tests/integration/test_dashboard_rpc_backends.py::TestDashboardRPCBackends::test_api_backends_response_structure PASSED
tests/integration/test_dashboard_rpc_backends.py::TestDashboardRPCBackends::test_backend_metadata_fields PASSED
tests/integration/test_dashboard_rpc_backends.py::TestDashboardRPCBackends::test_no_undefined_values PASSED
tests/integration/test_dashboard_rpc_backends.py::TestDashboardRPCBackends::test_registry_iteration_fix PASSED
tests/integration/test_dashboard_rpc_backends.py::TestRouterBackendIntegration::test_router_backends_structure PASSED
```

## Expected Dashboard Output

After the fix, the dashboard will now correctly display:

### Before (Broken)
```
🔗 RPC Backends (llama.cpp)
- undefined:undefined (undefined ms, 0 requests)
- undefined:undefined (undefined ms, 0 requests)
```

### After (Fixed)
```
🔗 RPC Backends (llama.cpp)
- node1:50052 (7.5 ms, 125 requests, 0 failures) ✅
- node2:50052 (5.2 ms, 89 requests, 0 failures) ✅
```

## API Response Structure

The `/api/network/backends` endpoint now returns:

```json
{
  "backends": [
    {
      "url": "node1:50052",
      "status": "healthy",
      "latency_ms": 7.5,
      "request_count": 125,
      "failure_count": 0
    },
    {
      "url": "node2:50052",
      "status": "healthy",
      "latency_ms": 5.2,
      "request_count": 89,
      "failure_count": 0
    }
  ],
  "total": 2
}
```

## Verification

To verify the fix works:

1. **Start RPC backends:**
   ```bash
   # On machine 1
   rpc-server --host 0.0.0.0 --port 50052

   # On machine 2
   rpc-server --host 0.0.0.0 --port 50052
   ```

2. **Start SOLLOL with dashboard:**
   ```python
   from sollol import UnifiedDashboard, RayHybridRouter, OllamaPool
   from sollol.rpc_discovery import auto_discover_rpc_backends

   pool = OllamaPool(discover_all_nodes=True, exclude_localhost=True)
   rpc_backends = auto_discover_rpc_backends()

   router = RayHybridRouter(
       ollama_pool=pool,
       rpc_backends=rpc_backends,
       enable_distributed=True
   )

   dashboard = UnifiedDashboard(router=router)
   dashboard.run()
   ```

3. **Open dashboard:**
   ```bash
   http://localhost:8080
   ```

4. **Verify RPC backends section shows:**
   - Correct IP addresses and ports
   - Actual latency measurements
   - Request counts
   - Health status (green checkmarks for healthy backends)

## Future Enhancements

Additional improvements that could be made:

1. **Model name discovery**: Add model name/path information to RPC backend metadata
2. **Layer distribution**: Show which layers each backend is handling
3. **Memory usage**: Display memory consumption per backend
4. **Throughput metrics**: Track tokens/sec per backend

## Related Files

- `src/sollol/rpc_registry.py` - Registry implementation
- `src/sollol/rpc_discovery.py` - Backend discovery
- `src/sollol/llama_cpp_rpc.py` - RPC client implementation
- `src/sollol/unified_dashboard.py` - Dashboard implementation
