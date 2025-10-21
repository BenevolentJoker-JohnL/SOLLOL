# SOLLOL Discovery Priority Fix

**Date**: 2025-10-21
**Issue**: "3 nodes" reported when only 2 physical machines exist
**Root Cause**: Config file loaded BEFORE auto-discovery, creating localhost duplicate

## The Problem

User correctly identified: **"are you sure this isnt a failure of sollol"**

### What Happened

```
Config file: ~/.synapticllamas_nodes.json
- localhost:11434
- 10.9.66.154:11434
- 10.9.66.194:11434

Result: 3 nodes registered
Reality: Only 2 physical machines (localhost = 10.9.66.154)
```

### Load Order Issue

**Original flow** (`/home/joker/SynapticLlamas/main.py:383-431`):

```python
# Step 1: Load config file FIRST
if os.path.exists(NODES_CONFIG_PATH):
    global_registry.load_config(NODES_CONFIG_PATH)  # ← Loaded stale localhost

# Step 2: Auto-discover SECOND
if current_mode == "distributed":
    global_registry.discover_and_add_nodes()  # ← Found real IPs
```

**Problem**: Config file was PRIMARY, auto-discovery was SECONDARY.

**This defeats the purpose of SOLLOL's intelligent discovery!**

## The Fix

### 1. Reversed Priority (PRIMARY Fix)

**File**: `/home/joker/SynapticLlamas/main.py:383-441`

**New flow**:

```python
# DISTRIBUTED MODE: SOLLOL auto-discovery is PRIMARY
if current_mode == "distributed":
    # Step 1: Auto-discover FIRST (full network scan)
    discovered_count = global_registry.discover_and_add_nodes()

    if discovered_count > 0:
        # Save discovered nodes (becomes new config)
        global_registry.save_config(NODES_CONFIG_PATH)
    else:
        # Fallback: only load config if discovery fails
        if os.path.exists(NODES_CONFIG_PATH):
            global_registry.load_config(NODES_CONFIG_PATH)

# STANDARD MODE: Use config file
else:
    # Config file makes sense here (no auto-discovery)
    if os.path.exists(NODES_CONFIG_PATH):
        global_registry.load_config(NODES_CONFIG_PATH)
```

**Key change**: Auto-discovery runs FIRST and overwrites config file with fresh results.

### 2. Improved Deduplication (DEFENSE-IN-DEPTH)

**File**: `/home/joker/SynapticLlamas/node_registry.py:62-114`

Even if config file is loaded, NodeRegistry now deduplicates localhost vs real IP:

```python
def _is_duplicate_node(self, url: str) -> Optional[str]:
    """
    Check if this URL points to an already registered node.

    Handles localhost vs real IP deduplication:
    - localhost:11434 and 10.9.66.154:11434 are duplicates (same machine)
    - 127.0.0.1 and machine's real IP are duplicates
    """
    def normalize_ip(ip: str) -> str:
        """Convert localhost IPs to actual machine IP for comparison."""
        if ip.startswith("127.") or ip == "localhost":
            # Get actual machine IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("10.255.255.255", 1))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        return ip

    # Compare normalized IPs (localhost → real IP)
    new_ip_normalized = normalize_ip(new_ip)
    existing_ip_normalized = normalize_ip(existing_ip)

    if new_ip_normalized == existing_ip_normalized and new_port == existing_port:
        logger.info(f"🔍 Duplicate detected: {url} is same machine as {existing_url}")
        return existing_url
```

**Protection**: Even if config has localhost, adding the real IP won't create duplicates.

## Why This Is the Right Design

### SOLLOL's Responsibility

**What SOLLOL does** ✅:
- Scans network for ALL Ollama nodes
- Deduplicates within discovery results (localhost → real IP)
- Returns clean, deduplicated list

**What SOLLOL can't do** ❌:
- Know about external config files
- Prevent stale config entries
- Deduplicate across application-level sources

### Application's Responsibility

**What NodeRegistry should do** ✅:
- Make auto-discovery PRIMARY in distributed mode
- Use config file as fallback (discovery fails) or manual overrides only
- Deduplicate across ALL sources (config + discovery + manual)

**This is now implemented correctly.**

## Test Results

### Before Fix

```bash
$ cat ~/.synapticllamas_nodes.json
{
  "nodes": [
    {"url": "http://localhost:11434"},
    {"url": "http://10.9.66.154:11434"},
    {"url": "http://10.9.66.194:11434"}
  ]
}

$ python main.py
⚡ PARALLEL MULTI-TURN MODE: 3 nodes available  # ← WRONG (only 2 machines)
```

### After Fix

```bash
# Fresh auto-discovery (distributed mode)
$ python main.py
> mode distributed

🔍 Auto-discovering Ollama nodes on network...
✅ Discovered 2 Ollama node(s):
   • http://10.9.66.154:11434 (ollama-10-9-66-154)
   • http://10.9.66.194:11434 (ollama-10-9-66-194)
💾 Saved to ~/.synapticllamas_nodes.json

✅ Total Ollama nodes available: 2
✅ 2 physical machines detected - parallel mode will be enabled

$ cat ~/.synapticllamas_nodes.json
{
  "nodes": [
    {"url": "http://10.9.66.154:11434", "name": "ollama-10-9-66-154"},
    {"url": "http://10.9.66.194:11434", "name": "ollama-10-9-66-194"}
  ]
}
```

## Design Principles

### 1. Auto-Discovery First

In distributed mode, SOLLOL's auto-discovery should be the **PRIMARY and AUTHORITATIVE** source.

- ✅ Network scan finds current state
- ✅ Config file is GENERATED from discovery (not vice versa)
- ✅ Config acts as cache for faster startup
- ✅ Fresh discovery always overwrites stale config

### 2. Config File as Fallback

Config file should only be used when:
- Auto-discovery fails (no nodes found)
- Running in standard mode (no distributed features)
- User manually adds specific overrides

### 3. Defense in Depth

Even with correct load order, NodeRegistry must deduplicate:
- Multiple sources (config + discovery + manual)
- localhost vs real IP equivalence
- Same IP with same port = duplicate

## Related Components

### SOLLOL Discovery
**File**: `/home/joker/SOLLOL/src/sollol/discovery.py`
- Already had `_deduplicate_nodes()` function ✅
- Works correctly within its scope ✅
- Can't prevent application-level issues ✅

### NodeRegistry Deduplication
**File**: `/home/joker/SynapticLlamas/node_registry.py`
- Now handles localhost normalization ✅
- Prevents duplicates across sources ✅
- Defense in depth protection ✅

### Main Application Flow
**File**: `/home/joker/SynapticLlamas/main.py`
- Auto-discovery is now PRIMARY ✅
- Config file is fallback only ✅
- Correct separation of concerns ✅

## Conclusion

**The user was RIGHT to question this.**

The original design had config file as PRIMARY and auto-discovery as SECONDARY. This is backwards - SOLLOL's intelligent discovery should be the authoritative source in distributed mode.

**The fix**:
1. ✅ Reversed priority: auto-discovery PRIMARY, config FALLBACK
2. ✅ Improved deduplication: localhost → real IP normalization
3. ✅ Proper separation: config for standard mode, discovery for distributed mode

**Now SOLLOL truly "just works" - zero configuration needed for multi-machine setups.**
