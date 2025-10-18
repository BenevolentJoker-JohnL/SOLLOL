# SOLLOL Universal Network Observability

## Overview

SOLLOL now provides **universal network observability** - a centralized dashboard that monitors ALL applications using the SOLLOL infrastructure, regardless of which router they use or how they're deployed.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  SOLLOL Unified Dashboard (http://localhost:8080)          │
│                                                             │
│  Monitors:                                                  │
│  • All Ollama nodes (network-wide discovery)               │
│  • All RPC backends (llama.cpp distributed)                │
│  • All applications using SOLLOL                           │
│  • Model lifecycle events (load/unload/processing)         │
│  • Distributed tracing across all components               │
│  • Ray cluster metrics                                     │
│  • Dask worker metrics                                     │
└────────────────────────────────────────────────────────────┘
                      ▲ ▲ ▲
         ┌────────────┘ │ └────────────┐
         │              │              │
    ┌────▼───┐     ┌───▼────┐     ┌───▼────────┐
    │ App 1  │     │ App 2  │     │ App 3      │
    │RayAdv  │     │RayHybr │     │SynLlamas   │
    │Router  │     │Router  │     │            │
    └────────┘     └────────┘     └────────────┘
```

**Key Principle:** One dashboard monitors the entire infrastructure, not individual applications.

## Features

### 1. Network-Level Observability (Universal)

**Endpoints:**
- `GET /api/network/nodes` - All Ollama nodes on the network
- `GET /api/network/backends` - All RPC backends
- `GET /api/network/health` - Overall network health

**WebSocket Streams:**
- `ws://localhost:8080/ws/network/nodes` - Node state changes
- `ws://localhost:8080/ws/network/backends` - Backend connections
- `ws://localhost:8080/ws/ollama_activity` - Model lifecycle events

**Features:**
- Works WITHOUT a router (auto-discovers network)
- Works WITH a router (enhanced metrics)
- Real-time event-driven updates (not polling)

### 2. Application Tracking (New!)

**Purpose:** Track which applications are using SOLLOL

**Endpoints:**
- `GET /api/applications` - List all registered applications
- `POST /api/applications/register` - Register application
- `POST /api/applications/heartbeat` - Keep application active
- `POST /api/applications/<id>/unregister` - Explicitly unregister

**WebSocket Stream:**
- `ws://localhost:8080/ws/applications` - Application lifecycle events

**Visible in Dashboard:**
- Application name, router type, version
- Status (active/stale based on heartbeats)
- Uptime
- Custom metadata

**Auto-cleanup:**
- Applications that don't send heartbeats are marked stale
- Stale applications are removed after 2x timeout

### 3. Framework Metrics (Optional)

**Ray Dashboard:**
- Embedded at http://localhost:8265
- Task timeline, distributed tracing
- Actor states, resource utilization

**Dask Dashboard:**
- Embedded at http://localhost:8787
- Performance profiling, task graphs
- Worker utilization

## Usage

### For Dashboard Operators

**Start Centralized Dashboard:**
```python
from sollol import UnifiedDashboard

# Standalone - discovers network automatically
dashboard = UnifiedDashboard(
    dashboard_port=8080,
    ray_dashboard_port=8265,
    dask_dashboard_port=8787,
    enable_dask=True,  # Optional: enable Dask metrics
)

dashboard.run(host="0.0.0.0")
```

**Access:**
- Web UI: http://localhost:8080
- Ray: http://localhost:8265
- Dask: http://localhost:8787

### For Application Developers

**Register Your Application:**
```python
from sollol import DashboardClient, RayAdvancedRouter

# 1. Register with dashboard (automatic heartbeats)
client = DashboardClient(
    app_name="My Application",
    router_type="RayAdvancedRouter",
    version="1.0.0",
    dashboard_url="http://localhost:8080",
    metadata={
        "environment": "production",
        "node_count": 5,
    }
)

# 2. Use SOLLOL normally
router = RayAdvancedRouter(...)

# 3. Your app is now visible in the dashboard!
# Automatic heartbeats every 10s
# Auto-unregisters on shutdown
```

**Update Metadata During Runtime:**
```python
client.update_metadata({
    "requests_processed": 1000,
    "uptime_hours": 12.5,
})
```

### For SynapticLlamas Integration

**In your main.py or __init__.py:**
```python
from sollol import DashboardClient
import threading

def start_dashboard_client():
    """Register with SOLLOL dashboard."""
    client = DashboardClient(
        app_name="SynapticLlamas",
        router_type="HybridRouter",
        version="1.0.0",
        dashboard_url="http://localhost:8080",
    )

# Start in background
dashboard_thread = threading.Thread(
    target=start_dashboard_client,
    daemon=True,
    name="SOLLOLDashboardClient"
)
dashboard_thread.start()

# Your application continues normally...
```

Now SynapticLlamas appears in the SOLLOL dashboard alongside all other applications!

## WebSocket Event Streams

### Node Events (`/ws/network/nodes`)
```javascript
{
    "type": "node_discovered",
    "timestamp": 1234567890.123,
    "node": "http://192.168.1.10:11434",
    "message": "✅ New node discovered: http://192.168.1.10:11434"
}

{
    "type": "status_change",
    "timestamp": 1234567890.123,
    "node": "http://192.168.1.10:11434",
    "old_status": "healthy",
    "new_status": "unhealthy",
    "message": "Node http://192.168.1.10:11434: healthy → unhealthy"
}
```

### Application Events (`/ws/applications`)
```javascript
{
    "type": "app_registered",
    "timestamp": 1234567890.123,
    "app_id": "abc-123",
    "name": "MyApp",
    "router_type": "RayAdvancedRouter",
    "message": "📱 Application started: MyApp (RayAdvancedRouter)"
}

{
    "type": "app_stale",
    "timestamp": 1234567890.123,
    "app_id": "abc-123",
    "name": "MyApp",
    "message": "⚠️  Application not responding: MyApp (last seen 35s ago)"
}
```

### Model Activity Events (`/ws/ollama_activity`)
```javascript
{
    "type": "model_loaded",
    "timestamp": 1234567890.123,
    "node": "192.168.1.10:11434",
    "model": "llama3.2:3b",
    "message": "✅ Model loaded on 192.168.1.10:11434: llama3.2:3b"
}

{
    "type": "model_processing",
    "timestamp": 1234567890.123,
    "node": "192.168.1.10:11434",
    "model": "llama3.2:3b",
    "vram_gb": 2.4,
    "message": "🔄 Processing on 192.168.1.10:11434: llama3.2:3b (VRAM: 2.40GB)"
}
```

## Benefits

### For Operators
- **Single Pane of Glass**: Monitor entire infrastructure from one dashboard
- **Network-Wide Visibility**: See all nodes, backends, and applications
- **Event-Driven**: Real-time updates without constant polling
- **Application Tracking**: Know which apps are using your infrastructure

### For Developers
- **Easy Integration**: 3 lines of code to register
- **Automatic Monitoring**: Heartbeats and cleanup handled automatically
- **Router-Agnostic**: Works with any SOLLOL router
- **Centralized Logs**: All application logs in one place

### For SynapticLlamas
- **No Dashboard Code**: SOLLOL provides the dashboard
- **Automatic Discovery**: Nodes and backends auto-discovered
- **Application Visibility**: See SynapticLlamas alongside other apps
- **Unified Monitoring**: One dashboard for all SOLLOL-based applications

## Examples

See:
- `examples/unified_dashboard_demo.py` - Full dashboard demo
- `examples/dashboard_application_tracking.py` - Application registration example

## API Summary

### HTTP Endpoints

**Network Observability:**
- GET `/api/network/nodes` - All Ollama nodes
- GET `/api/network/backends` - All RPC backends
- GET `/api/network/health` - Network health summary

**Application Tracking:**
- GET `/api/applications` - All registered applications
- POST `/api/applications/register` - Register application
- POST `/api/applications/heartbeat` - Send heartbeat
- POST `/api/applications/<id>/unregister` - Unregister

**Traditional Metrics:**
- GET `/api/metrics` - SOLLOL router metrics
- GET `/api/traces` - Distributed traces
- GET `/api/ray/metrics` - Ray cluster metrics
- GET `/api/dask/metrics` - Dask cluster metrics
- GET `/api/prometheus` - Prometheus metrics export

**Logging:**
- WS `/ws/logs` - Centralized log streaming

### WebSocket Streams

**Real-Time Monitoring:**
- WS `/ws/network/nodes` - Node state changes
- WS `/ws/network/backends` - Backend connections
- WS `/ws/ollama_activity` - Model lifecycle
- WS `/ws/applications` - Application lifecycle
- WS `/ws/logs` - Log streaming

## Version

Added in: **SOLLOL v0.9.1**

## Summary

SOLLOL now provides **universal network observability** - any application using SOLLOL infrastructure can be monitored from a single centralized dashboard. The dashboard tracks:

1. **Infrastructure**: All Ollama nodes and RPC backends
2. **Applications**: All apps using SOLLOL (with automatic registration)
3. **Activity**: Model loading, processing, and network events
4. **Performance**: Ray/Dask metrics, distributed tracing, analytics

This makes SOLLOL a complete observability platform for distributed LLM inference.
