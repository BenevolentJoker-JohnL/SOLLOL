# Dask Environment Variable Propagation

## Problem Statement

When SOLLOL's `OllamaPool` is initialized with `enable_dask=True` and uses process workers (`processes=True`), the Dask worker processes do **not** inherit environment variables from the parent process. This causes critical issues with the `NetworkObserver` singleton, which requires specific environment variables to be set before initialization:

- `SOLLOL_OBSERVER_SAMPLING` - Controls whether to sample network events
- `SOLLOL_REDIS_URL` - Redis connection URL for pub/sub
- `SOLLOL_APP_NAME` - Application name for dashboard registration

### Why This Matters

The `NetworkObserver` class uses a singleton pattern and initializes **once** on first import. If the environment variables are not set before worker subprocess spawns, the observer will initialize without Redis configuration, causing:

1. No activity events published to Redis
2. Dashboard won't receive real-time updates
3. Live stats and logs won't appear in the SOLLOL dashboard

## Failed Approach: Using `env` Parameter

### What Didn't Work

Initially, we attempted to pass environment variables using the `env` parameter on `LocalCluster`:

```python
# ❌ THIS DOES NOT WORK RELIABLY
worker_env = {}
for key in ["SOLLOL_OBSERVER_SAMPLING", "SOLLOL_REDIS_URL", "SOLLOL_APP_NAME"]:
    if key in os.environ:
        worker_env[key] = os.environ[key]

cluster = LocalCluster(
    n_workers=2,
    threads_per_worker=2,
    processes=True,
    env=worker_env,  # ← Does not set vars before subprocess spawns
)
```

### Why It Failed

The `env` parameter on `LocalCluster` sets environment variables **after** the worker process has already started. By that time, SOLLOL's modules have already been imported and `NetworkObserver` has already initialized without the required configuration.

## Correct Solution: Dask Configuration System

### The Fix

Use Dask's configuration system with `distributed.nanny.pre-spawn-environ` to set environment variables **before** the worker subprocess spawns:

```python
# ✅ THIS WORKS CORRECTLY
import os
import dask
from dask.distributed import LocalCluster

# Extract environment variables from parent process
worker_env = {}
for key in ["SOLLOL_OBSERVER_SAMPLING", "SOLLOL_REDIS_URL", "SOLLOL_APP_NAME"]:
    if key in os.environ:
        worker_env[key] = os.environ[key]

# CRITICAL: Set Dask config BEFORE creating LocalCluster
if worker_env:
    dask.config.set({"distributed.nanny.pre-spawn-environ": worker_env})

# Now create the cluster - workers will have the environment variables
cluster = LocalCluster(
    n_workers=2,
    threads_per_worker=2,
    processes=True,
)
```

### Why This Works

The `distributed.nanny.pre-spawn-environ` configuration key is specifically designed for environment variables that must be set **before** the worker process starts. This ensures:

1. Variables are set in the subprocess environment before any imports
2. `NetworkObserver` initializes with correct Redis configuration
3. Dashboard integration works seamlessly

## Implementation in SOLLOL

The fix is implemented in `sollol/pool.py` (lines 730-755):

```python
# Propagate SOLLOL environment variables to Dask workers
# CRITICAL: Must use dask.config.set with 'distributed.nanny.pre-spawn-environ'
# to ensure variables are set BEFORE worker subprocess spawns
worker_env = {}
for key in ["SOLLOL_OBSERVER_SAMPLING", "SOLLOL_REDIS_URL", "SOLLOL_APP_NAME"]:
    if key in os.environ:
        worker_env[key] = os.environ[key]
        logger.info(f"🔧 Propagating {key}={os.environ[key]} to Dask workers")

if worker_env:
    logger.info(f"✅ Dask worker environment: {list(worker_env.keys())}")
    # Set Dask config to propagate env vars before worker spawn
    import dask
    dask.config.set({"distributed.nanny.pre-spawn-environ": worker_env})
    logger.info("✅ Dask configuration set: distributed.nanny.pre-spawn-environ")
else:
    logger.warning("⚠️  No SOLLOL environment variables found to propagate!")

# Start with conservative settings
cluster = LocalCluster(
    n_workers=len(self.nodes) if self.nodes else 2,
    threads_per_worker=2,
    processes=use_processes,
    memory_limit="auto",
    silence_logs=logging.WARNING,
)
```

## Usage in Applications

If you're building an application that uses SOLLOL with Dask, follow this pattern:

```python
#!/usr/bin/env python3
import os

# 1. Set environment variables BEFORE importing SOLLOL
os.environ["SOLLOL_OBSERVER_SAMPLING"] = "false"
os.environ["SOLLOL_REDIS_URL"] = "redis://localhost:6379"
os.environ["SOLLOL_APP_NAME"] = "MyApplication"

# 2. Now import and use SOLLOL
from sollol.pool import OllamaPool
from sollol.routing_strategy import RoutingStrategy

# 3. Create pool with Dask enabled - env vars will propagate correctly
pool = OllamaPool(
    nodes=None,
    routing_strategy=RoutingStrategy.ROUND_ROBIN,
    app_name="MyApplication",
    enable_dask=True,  # ✅ Workers will receive environment variables
    register_with_dashboard=True
)
```

## Testing Considerations

### Cannot Test from Stdin

When testing Dask with process workers, you **cannot** run code from stdin:

```bash
# ❌ THIS WILL FAIL
python3 -c "import os; os.environ['SOLLOL_APP_NAME']='test'; from sollol.pool import OllamaPool; ..."
```

**Error**: `FileNotFoundError: /path/to/<stdin>`

**Reason**: Dask worker processes need to re-execute the main module file, which doesn't exist for stdin scripts.

### Correct Testing Approach

Create a proper Python file:

```python
# test_dask.py
import os
os.environ["SOLLOL_REDIS_URL"] = "redis://localhost:6379"
os.environ["SOLLOL_APP_NAME"] = "TestApp"

from sollol.pool import OllamaPool

pool = OllamaPool(enable_dask=True)
```

Then run:
```bash
python3 test_dask.py
```

## Verification

To verify that environment variables are propagating correctly:

1. **Check Logs**: When creating the pool, you should see:
   ```
   🔧 Propagating SOLLOL_OBSERVER_SAMPLING=false to Dask workers
   🔧 Propagating SOLLOL_REDIS_URL=redis://localhost:6379 to Dask workers
   🔧 Propagating SOLLOL_APP_NAME=MyApp to Dask workers
   ✅ Dask worker environment: ['SOLLOL_OBSERVER_SAMPLING', 'SOLLOL_REDIS_URL', 'SOLLOL_APP_NAME']
   ✅ Dask configuration set: distributed.nanny.pre-spawn-environ
   ```

2. **Monitor Redis**: Subscribe to dashboard channels:
   ```bash
   redis-cli PSUBSCRIBE "sollol:dashboard:*"
   ```

   You should see activity events like:
   ```json
   {"timestamp": 1234567890.123, "backend": "10.9.66.154:11434",
    "event_type": "ollama_request", "details": {...}}
   ```

3. **Check Dashboard**: Navigate to `http://localhost:8080` and verify:
   - Application appears in the registered apps list
   - Real-time activity shows up in the live stats
   - Logs stream appears with application events

## Related Documentation

- [UNIFIED_OBSERVABILITY.md](./UNIFIED_OBSERVABILITY.md) - Overview of SOLLOL's observability architecture
- [DASHBOARD_RPC_FIX.md](./DASHBOARD_RPC_FIX.md) - Dashboard RPC registration details
- [EXPERIMENTAL_FEATURES.md](./EXPERIMENTAL_FEATURES.md) - Dask integration overview

## Technical References

- **Dask Documentation**: [Distributed Configuration](https://distributed.dask.org/en/latest/configuration.html)
- **Environment Variable Propagation**: `distributed.nanny.pre-spawn-environ` is the official mechanism for setting variables before worker subprocess initialization
- **Singleton Pattern**: NetworkObserver initializes once on first import and cannot be reconfigured later

## Version History

- **v0.9.64**: Fixed Dask environment variable propagation using `dask.config.set()` with `distributed.nanny.pre-spawn-environ`
- **v0.9.63**: Initial (failed) attempt using `env` parameter on LocalCluster
