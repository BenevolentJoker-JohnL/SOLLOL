# Distributed Node Configuration Fixes

**Date**: 2025-10-20
**Status**: ✅ Complete

## Issues Fixed

### 1. Parallel Ollama Nodes Not Loading (Duplicate Detection Bug)

**Problem**: SynapticLlamas showed "Using 1 Ollama nodes" instead of 2

**Root Cause**: `NodeRegistry._is_duplicate_node()` compared only IP addresses, treating `localhost:11434` and `localhost:11435` as duplicates (both resolve to 127.0.0.1)

**Fix**: Updated `/home/joker/SynapticLlamas/node_registry.py:51-74`

```python
# Before: Only checked IP
if new_ip == existing_ip:
    return existing_url

# After: Check both IP and port
new_port = url.split(':')[-1].rstrip('/')
existing_port = existing_url.split(':')[-1].rstrip('/')
if new_ip == existing_ip and new_port == existing_port:
    return existing_url
```

**Result**:
- ✅ Both local Ollama instances now register correctly
- ✅ Parallel mode enabled with 2 nodes
- ✅ Allows multiple Ollama instances on same machine with different ports

**Configuration**: `~/.synapticllamas_nodes.json`
```json
{
  "nodes": [
    {"url": "http://localhost:11434", "name": "ollama-localhost", "priority": 0},
    {"url": "http://localhost:11435", "name": "ollama-secondary", "priority": 0}
  ]
}
```

---

### 2. Stale RPC Backend IPs

**Problem**: References to unreachable IPs (10.9.66.48, 10.9.66.45) in configs

**Verification**:
```bash
nc -zv 10.9.66.45 50052  # ❌ Timeout
nc -zv 10.9.66.48 50052  # ❌ No route to host
nc -zv 10.9.66.154 50052 # ✅ Success (local machine)
```

**Fix**: Removed unreachable backends from:

1. **`/home/joker/SOLLOL/rpc_backends.conf`** (SOURCE OF TRUTH)
```bash
# Before:
10.9.66.48:50052
10.9.66.154:50052

# After:
10.9.66.154:50052
```

2. **`~/.synapticllamas.json`**
```json
{
  "rpc_backends": [
    {"host": "10.9.66.154", "port": 50052, ...}
  ],
  "task_distribution_enabled": true
}
```

**Result**: ✅ Only reachable RPC backend configured

---

### 3. RPC Auto-Discovery Not Finding Config File

**Problem**: `auto_discover_rpc_backends()` couldn't find `rpc_backends.conf`

**Root Cause**: Discovery used relative path `Path("rpc_backends.conf")`, which only worked when running from `/home/joker/SOLLOL`

**Fix**: Updated `/home/joker/SOLLOL/src/sollol/rpc_discovery.py:300-330`

```python
# Check multiple locations for rpc_backends.conf
possible_paths = [
    Path("rpc_backends.conf"),  # Current directory
    Path("/home/joker/SOLLOL/rpc_backends.conf"),  # SOLLOL directory
    Path(os.path.expanduser("~/SOLLOL/rpc_backends.conf")),  # Home/SOLLOL
    Path("/home/joker/SynapticLlamas/rpc_backends.conf"),  # SynapticLlamas
]

config_file = None
for path in possible_paths:
    if path.exists():
        config_file = path
        break
```

**Discovery Priority**:
1. Redis registry (`sollol:router:metadata`)
2. Config file (`rpc_backends.conf` in multiple locations)
3. Network scan (fallback)

**Result**: ✅ Auto-discovery now finds config file regardless of working directory

**Verification**:
```bash
$ python3 -c "from sollol.rpc_discovery import auto_discover_rpc_backends; \
  backends = auto_discover_rpc_backends(); \
  print(f'Discovered {len(backends)} RPC backend(s)')"

Discovered 1 RPC backend(s):
  - {'host': '10.9.66.154', 'port': 50052, 'has_gpu': False, ...}
```

---

## Testing

### Verify Node Registry
```bash
python3 -c "
import sys
sys.path.insert(0, '/home/joker/SynapticLlamas')
from node_registry import NodeRegistry
import os

registry = NodeRegistry()
registry.load_config(os.path.expanduser('~/.synapticllamas_nodes.json'))

print(f'✅ Loaded {len(registry.nodes)} nodes:')
for url, node in registry.nodes.items():
    status = '✅' if node.metrics.is_healthy else '❌'
    print(f'   {status} {node.name} ({url})')

healthy = len(registry.get_healthy_nodes())
print(f'\n✅ Healthy nodes: {healthy}')
print(f'✅ Parallel mode: {\"ENABLED\" if healthy >= 2 else \"DISABLED\"}')"
```

Expected output:
```
✅ Loaded 2 nodes:
   ✅ ollama-localhost (http://localhost:11434)
   ✅ ollama-secondary (http://localhost:11435)

✅ Healthy nodes: 2
✅ Parallel mode: ENABLED
```

### Verify RPC Discovery
```bash
python3 -c "from sollol.rpc_discovery import auto_discover_rpc_backends; \
  backends = auto_discover_rpc_backends(); \
  print(f'Discovered: {backends}')"
```

Expected: 1 backend at 10.9.66.154:50052

---

## Next Steps

1. **Restart SynapticLlamas** to see the changes:
   ```bash
   cd ~/SynapticLlamas && python main.py
   ```

2. **Run `distributed task` mode** - should show:
   - "Using 2 Ollama nodes for load balancing"
   - "Model sharding: DISABLED" (no remote RPC backends)

3. **Test long-form generation** - should show:
   - "PARALLEL MULTI-NODE MODE"
   - Chunks distributed across both local Ollama instances

4. **Add remote RPC backends** (when available):
   ```bash
   # Edit /home/joker/SOLLOL/rpc_backends.conf
   echo "NEW_IP:50052" >> /home/joker/SOLLOL/rpc_backends.conf
   ```

---

## Files Modified

### SynapticLlamas
- `/home/joker/SynapticLlamas/node_registry.py` - Fixed duplicate detection

### SOLLOL
- `/home/joker/SOLLOL/src/sollol/rpc_discovery.py` - Multi-path config search
- `/home/joker/SOLLOL/rpc_backends.conf` - Removed stale IPs

### Configs
- `~/.synapticllamas_nodes.json` - Updated to use both local Ollama instances
- `~/.synapticllamas.json` - Cleaned RPC backends, enabled task distribution

---

## Package Status

Both packages installed in editable mode - **changes are immediately active**:
- SynapticLlamas: `pip show synaptic-llamas` → `/home/joker/SynapticLlamas`
- SOLLOL: `pip show sollol` → `/home/joker/SOLLOL/src`

No reinstallation needed! 🎉
