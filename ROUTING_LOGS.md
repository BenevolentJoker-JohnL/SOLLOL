# SOLLOL Routing Logs

**Real-time visibility into all SOLLOL routing decisions across distributed instances**

## 🎯 What It Does

The SOLLOL routing logger captures every routing decision made by SOLLOL and displays them in real-time on the dashboard. This includes:

- **Ollama vs RPC routing decisions** - Why a model was sent to Ollama pool or RPC backends
- **Node selection decisions** - Which Ollama node was chosen and why (load, latency, VRAM)
- **Fallback events** - When SOLLOL automatically falls back from Ollama to RPC
- **Coordinator lifecycle** - llama.cpp coordinator startup/shutdown events
- **Cache hits** - When routing decisions are retrieved from cache

## 📊 Where to View

### 1. SOLLOL Dashboard (Automatic)

The routing logs appear automatically in the SOLLOL dashboard at `http://localhost:8080` in the **"🎯 SOLLOL Routing Decisions"** panel.

**Features:**
- ✅ Real-time streaming via WebSocket
- ✅ Color-coded by event type
- ✅ Automatic across all SOLLOL instances on the network
- ✅ Last 100 events kept in view
- ✅ Auto-scroll to latest events

**Color Coding:**
- **Cyan** - Route decisions
- **Yellow** - Fallback events
- **Green** - Coordinator starts, successful operations
- **Red** - Coordinator stops, errors
- **Blue** - Cache hits

### 2. Standalone CLI Viewer

Watch routing logs in a dedicated terminal:

```bash
# Watch all routing events (live)
python -m sollol.routing_viewer

# View last 100 events from history
python -m sollol.routing_viewer --history 100

# Filter by model
python -m sollol.routing_viewer --model llama3.2:3b

# Filter by backend
python -m sollol.routing_viewer --backend rpc

# Filter by event type
python -m sollol.routing_viewer --event-type ROUTE_DECISION

# Filter by instance
python -m sollol.routing_viewer --instance hostname_1234_abcd1234
```

## 🔧 Configuration

### Environment Variables

```bash
# Enable/disable routing logs (default: true)
export SOLLOL_ROUTING_LOG=true

# Enable console output for routing decisions (default: false)
export SOLLOL_ROUTING_LOG_CONSOLE=true

# Redis URL for log aggregation (default: redis://localhost:6379)
export SOLLOL_REDIS_URL=redis://localhost:6379
```

### Programmatic Control

```python
from sollol.routing_logger import get_routing_logger, enable_console_routing_log

# Enable console output for debugging
enable_console_routing_log()

# Get logger instance
logger = get_routing_logger(console_output=True)
```

## 📝 Event Types

### ROUTE_DECISION
Model routed to Ollama or RPC backend

```
🎯 llama3.2:3b → ollama | sufficient_resources (estimated 2.5GB)
```

### CACHE_HIT
Routing decision retrieved from cache

```
💾 llama3.2:3b → ollama (cached)
```

### FALLBACK_TRIGGERED
Automatic fallback from Ollama to RPC

```
⚠️  codellama:13b: ollama → rpc | ollama_error: Out of memory
```

### COORDINATOR_START
llama.cpp coordinator started for large model

```
🚀 Coordinator started: llama3.1:70b (3 RPC backends)
```

### COORDINATOR_STOP
llama.cpp coordinator shut down

```
⏹️  Coordinator stopped: llama3.1:70b
```

### OLLAMA_NODE_SELECTED
Specific Ollama node chosen for request

```
📡 llama3.2:3b → 192.168.1.21:11434 | lowest_latency (15ms) + high_vram (8192MB free)
```

## 🏗️ Architecture

```
┌──────────────────────┐
│  SOLLOL Instances    │
│  (HybridRouter,      │
│   OllamaPool)        │
└──────┬───────────────┘
       │ Publishes routing events
       ▼
┌──────────────────────┐
│  Redis Pub/Sub       │
│  Channel:            │
│  sollol:routing      │
│  _events             │
└──────┬───────────────┘
       │ Subscribes
       ▼
┌──────────────────────┐
│  Dashboard Service   │
│  WebSocket:          │
│  /ws/routing_events  │
└──────┬───────────────┘
       │ Streams to
       ▼
┌──────────────────────┐
│  Browser Dashboard   │
│  + CLI Viewer        │
└──────────────────────┘
```

**Key Points:**
- **Separate channel** - Routing logs use `sollol:routing_events`, not mixed with operational logs
- **Redis streams** - Last 10,000 events persisted for history viewing
- **Distributed** - All SOLLOL instances on the network publish to same Redis channel
- **Real-time** - WebSocket streaming with <100ms latency

## 🧪 Testing

Generate sample routing events:

```bash
cd /home/joker/SOLLOL
python test_routing_log.py
```

This will publish test events to Redis that you can view in the dashboard or CLI viewer.

## 🔍 Use Cases

### 1. Debugging Routing Decisions
**Problem:** Model unexpectedly routed to RPC instead of Ollama
**Solution:** Check routing logs to see the exact reason (insufficient resources, model size, etc.)

### 2. Performance Optimization
**Problem:** Want to understand which nodes are being selected
**Solution:** Watch node selection logs to see load balancing decisions

### 3. Troubleshooting Fallbacks
**Problem:** Frequent fallbacks from Ollama to RPC
**Solution:** Check fallback events to identify root cause (OOM errors, timeouts, etc.)

### 4. Multi-Instance Monitoring
**Problem:** Multiple SOLLOL instances running, need unified view
**Solution:** All instances publish to same Redis channel - single dashboard shows all

### 5. Audit Trail
**Problem:** Need to understand routing history for specific model
**Solution:** Use history mode: `python -m sollol.routing_viewer --history 1000 --model llama3.1:70b`

## 📚 Integration

The routing logger is automatically initialized when you use:
- `HybridRouter` - Logs high-level Ollama vs RPC decisions
- `OllamaPool` - Logs node selection within Ollama pool

**No additional configuration needed!** Just ensure Redis is running and the dashboard is started.

## 🚀 Quick Start

```bash
# 1. Ensure Redis is running
redis-cli ping  # Should return PONG

# 2. Start SOLLOL dashboard (if not already running)
# Dashboard starts automatically with HybridRouter

# 3. View routing logs
# Option A: Open browser to http://localhost:8080
# Option B: Use CLI viewer
python -m sollol.routing_viewer

# 4. Make some requests to trigger routing events
# (Use your SOLLOL-enabled application)
```

## 🎨 Example Output

```
[12:03:45.123] ROUTE_DECISION      | model=llama3.2:3b        | backend=ollama
  ├─ instance: workstation_42_a1b2c3d4
  ├─ reason: sufficient_resources (estimated 2.5GB)
  └─ parameters: 3B

[12:03:46.456] OLLAMA_NODE_SELECTED | model=llama3.2:3b        | backend=ollama
  ├─ instance: workstation_42_a1b2c3d4
  ├─ reason: lowest_latency (15ms) + high_vram (8192MB free)
  ├─ node: 192.168.1.21:11434
  └─ confidence: 0.92

[12:04:10.789] ROUTE_DECISION      | model=llama3.1:70b       | backend=rpc
  ├─ instance: workstation_42_a1b2c3d4
  ├─ reason: insufficient_ollama_resources (requires 40.0GB)
  └─ parameters: 70B

[12:04:11.234] COORDINATOR_START   | model=llama3.1:70b       | backend=llamacpp
  ├─ instance: workstation_42_a1b2c3d4
  ├─ coordinator: 127.0.0.1:18080
  └─ rpc_backends: 3
```

---

**Questions?** Check the [SOLLOL Documentation](./README.md) or open an issue on GitHub.
