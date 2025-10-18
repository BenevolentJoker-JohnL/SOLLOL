# Known Issues

## ~~Dashboard Double Registration Bug~~ (SOLVED in v0.9.50)

**Issue**: Applications using `OllamaPool(register_with_dashboard=False)` and then manually creating a `DashboardClient` would appear twice in the dashboard applications list.

**Root Cause**: The `register_with_dashboard=False` parameter was ignored in `pool.py:310`. OllamaPool called `_auto_register_with_dashboard()` unconditionally during initialization, regardless of the flag value.

**Example of Affected Code**:
```python
# User creates pool with registration disabled
pool = OllamaPool(
    nodes=["http://192.168.1.21:11434"],
    register_with_dashboard=False  # Intending to register manually later
)

# Later, user manually registers with dashboard
from sollol.dashboard_client import DashboardClient
pool._dashboard_client = DashboardClient(
    app_name="MyApp",
    router_type="OllamaPool",
    auto_register=True
)

# Result: Pool auto-registered anyway (bug) + manual registration = 2 entries
```

**Solution** (Implemented in v0.9.50):
Added conditional check in `pool.py:310` to respect the `register_with_dashboard` flag:

```python
# Before (buggy):
self._auto_register_with_dashboard()

# After (fixed):
if self.register_with_dashboard:
    self._auto_register_with_dashboard()
```

**Workaround for v0.9.49 and earlier**:
If using older versions, manually unregister duplicate entries:
```bash
curl -X POST http://localhost:8080/api/applications/unregister \
  -H "Content-Type: application/json" \
  -d '{"app_id": "duplicate-app-id-here"}'
```

**Status**: ✅ RESOLVED - Fixed in v0.9.50 (source) and v0.9.50 (PyPI).

---

## ~~Dask Worker Logging Warnings~~ (SOLVED in v0.9.18)

**Issue**: When the UnifiedDashboard was initialized with `enable_dask=True`, Dask worker processes generated verbose "Task queue depth" warnings that spammed CLI output after clicking the dashboard link.

**Root Cause**: Dask worker **processes** run with completely isolated logging configurations that don't inherit from the main process. The warnings were triggered by HTTP requests to the Dask dashboard and logged at WARNING level from within worker processes.

**Solution** (Implemented in v0.9.18):
Use `processes=False` when creating LocalCluster to run workers as **threads** instead of separate processes:

```python
cluster = LocalCluster(
    n_workers=1,
    threads_per_worker=2,
    processes=False,  # Use threads, not processes
    dashboard_address=f":{dask_dashboard_port}",
    silence_logs=logging.CRITICAL,
)
```

**Why This Works**:
- Threaded workers run in the same process as the application
- They share the same logging configuration
- Application-level logging suppression (`logging.getLogger('distributed').setLevel(logging.ERROR)`) now works
- Dashboard functionality is unaffected

**Trade-offs**:
- Threaded workers share GIL (Global Interpreter Lock) with main process
- For SOLLOL's use case (lightweight dashboard observability), this is acceptable
- For compute-intensive tasks, process-based workers would be better

**Status**: ✅ RESOLVED - UnifiedDashboard now uses threaded workers by default.
