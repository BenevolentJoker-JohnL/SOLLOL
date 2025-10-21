# SOLLOL Configuration - Quick Reference

**Essential commands for task distribution and performance optimization**

---

## 🚀 Quick Start

### Automatic (Recommended)

```bash
cd /home/joker/SynapticLlamas
python main.py
> mode distributed
```

**Result**: Auto-discovers all Ollama nodes on network, enables parallel if multi-machine

---

## 🎯 Key Decisions

### Do I have multiple physical machines?

**YES** → Parallel mode enabled automatically ✅
- Expected speedup: ~1.8x with 2 machines
- SOLLOL detects topology automatically

**NO** → Parallel mode disabled automatically ❌
- Sequential execution is 50-100% FASTER
- Prevents resource contention

### How do I verify?

```bash
# Check discovered nodes
cat ~/.synapticllamas_nodes.json

# Should show real IPs (not localhost):
{
  "nodes": [
    {"url": "http://10.9.66.154:11434"},  // ✅ Machine 1
    {"url": "http://10.9.66.194:11434"}   // ✅ Machine 2
  ]
}
```

---

## 📋 Configuration Matrix

| Scenario | Config Method | Parallel Mode | Performance |
|----------|---------------|---------------|-------------|
| **1 machine, 1 Ollama** | Auto-discovery | Disabled | Baseline |
| **1 machine, 2 Ollama (diff ports)** | Manual config | Disabled ⚠️ | **50% slower if enabled** |
| **2+ machines, 1 Ollama each** | Auto-discovery | Enabled ✅ | **~1.8x faster** |
| **2+ machines, custom ports** | Manual config | Enabled ✅ | ~1.8x faster |

---

## 🔧 Common Tasks

### Task 1: Enable Multi-Machine Parallelism

```bash
# Option A: Automatic (just start distributed mode)
python main.py
> mode distributed

# Option B: Verify auto-discovery found all machines
python3 -c "
from node_registry import NodeRegistry
registry = NodeRegistry(auto_discover=True)
print(f'Nodes: {len(registry.nodes)}')
"
```

### Task 2: Disable False Parallelism

**Nothing to do** - SOLLOL automatically detects same-machine nodes and disables parallel mode.

**Verify**:
```
ℹ️  All 2 nodes on same machine - parallel mode will be disabled
```

### Task 3: Add Remote Node Manually

```python
from node_registry import NodeRegistry
import os

registry = NodeRegistry()
registry.add_node("http://remote-machine:11434", name="remote-1")
registry.save_config(os.path.expanduser("~/.synapticllamas_nodes.json"))
```

### Task 4: Force Sequential Mode

```python
from sollol.pool import OllamaPool

pool = OllamaPool(nodes=discovered_nodes)

# Override: always use sequential
if pool.count_unique_physical_hosts() < 999:  # Always true
    execute_sequential(tasks)
```

### Task 5: Check Locality Status

```python
from sollol.pool import OllamaPool

pool = OllamaPool(nodes=[
    {'host': '10.9.66.154', 'port': '11434'},
    {'host': '10.9.66.194', 'port': '11434'}
])

print(f"Machines: {pool.count_unique_physical_hosts()}")
print(f"Parallel: {pool.should_use_parallel_execution(3)}")

# Expected output for 2 different machines:
# Machines: 2
# Parallel: True
```

---

## 🐛 Troubleshooting

### Problem: Shows 3 nodes but only have 2 machines

**Cause**: Duplicate localhost entry

**Fix**:
```bash
rm ~/.synapticllamas_nodes.json
# Restart in distributed mode - auto-discovery will fix it
```

### Problem: No nodes discovered

**Fix 1**: Check Ollama is running
```bash
curl http://localhost:11434/api/tags
```

**Fix 2**: Check firewall
```bash
sudo ufw allow 11434/tcp
```

**Fix 3**: Manual config as fallback
```bash
echo '{
  "nodes": [
    {"url": "http://10.9.66.154:11434", "name": "machine-1"}
  ]
}' > ~/.synapticllamas_nodes.json
```

### Problem: Parallel mode enabled but slower

**Cause 1**: Tasks too small (overhead > benefit)
- **Fix**: Use sequential for small tasks (< 30s each)

**Cause 2**: Network latency
- **Fix**: Use local nodes only

**Cause 3**: Actually same machine (detection failed)
- **Check**: Verify IPs are different machines

---

## 📊 Performance Expectations

### Multi-Machine Parallel (✅ Correct Usage)

```
Sequential: 100 seconds
Parallel (2 machines): 55 seconds (~1.8x faster)
Parallel (3 machines): 40 seconds (~2.5x faster)
```

### Same-Machine Parallel (❌ Incorrect Usage)

```
Sequential: 100 seconds
Parallel (same machine): 150-200 seconds (50-100% SLOWER)

⚠️  SOLLOL prevents this automatically
```

---

## 🎓 Advanced Configuration

### Custom Discovery Timeout

```python
# Fast local network
registry.discover_and_add_nodes(timeout=0.2)

# Slow network
registry.discover_and_add_nodes(timeout=1.0)
```

### Node Priority

```python
# High priority (GPU node)
registry.add_node("http://gpu:11434", priority=0)

# Low priority (fallback)
registry.add_node("http://cpu:11434", priority=2)
```

### Disable Auto-Discovery

```python
# Use config file only (faster startup)
registry = NodeRegistry(auto_discover=False)
registry.load_config("~/.synapticllamas_nodes.json")
```

---

## 📖 Full Documentation

**Comprehensive guide**: `/home/joker/SOLLOL/SOLLOL_CONFIGURATION_GUIDE.md`

**Topics covered**:
- Auto-discovery configuration
- Locality awareness details
- Manual configuration
- Performance tuning
- Integration examples
- Full troubleshooting guide

---

## ✅ Checklist: Is My Setup Optimal?

- [ ] Auto-discovery found all my Ollama nodes?
- [ ] Node IPs are real IPs (not localhost)?
- [ ] Locality detection shows correct machine count?
- [ ] Parallel mode matches my infrastructure (enabled for multi-machine)?
- [ ] Performance meets expectations (~1.8x with 2 machines)?

**If all ✅**: You're configured optimally!

**If any ❌**: See full documentation or troubleshooting section

---

**Last Updated**: 2025-10-21
