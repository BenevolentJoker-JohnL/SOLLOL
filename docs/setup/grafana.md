# Grafana + InfluxDB Metrics Visualization for SOLLOL

This guide shows you how to set up Grafana to visualize SOLLOL's time-series metrics stored in InfluxDB.

## Architecture

```
SOLLOL → InfluxDB (time-series storage) → Grafana (visualization)
```

- **InfluxDB**: Stores metrics (node health, latency, requests, etc.)
- **Grafana**: Renders beautiful dashboards and graphs
- **SOLLOL**: Automatically logs metrics if enabled

---

## 1. Install InfluxDB (Bare Metal)

### Ubuntu/Debian

```bash
# Add InfluxData repository
wget -q https://repos.influxdata.com/influxdata-archive_compat.key
echo '393e8779c89ac8d958f81f942f9ad7fb82a25e133faddaf92e15b16e6ac9ce4c influxdata-archive_compat.key' | sha256sum -c && cat influxdata-archive_compat.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg > /dev/null
echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg] https://repos.influxdata.com/debian stable main' | sudo tee /etc/apt/sources.list.d/influxdata.list

# Install InfluxDB 2.x
sudo apt-get update && sudo apt-get install influxdb2

# Start service
sudo systemctl start influxdb
sudo systemctl enable influxdb
```

### Initial Setup

```bash
# Open browser to http://localhost:8086
# Create initial user and organization
# Organization: sollol
# Bucket: sollol_metrics
# Username: admin
# Password: <your-password>

# Copy the generated token for later
```

---

## 2. Configure SOLLOL for InfluxDB

### Environment Variables

```bash
export SOLLOL_METRICS_BACKEND=influxdb
export INFLUX_URL=http://localhost:8086
export INFLUX_TOKEN=<your-influxdb-token>
export INFLUX_ORG=sollol
export INFLUX_BUCKET=sollol_metrics
```

### Programmatic Configuration

```python
from sollol import OllamaPool
from sollol.config import SOLLOLConfig
import os

# Set InfluxDB token from your setup
os.environ['INFLUX_TOKEN'] = 'your-token-here'

# Configure SOLLOL with InfluxDB enabled
config = SOLLOLConfig(
    influxdb_enabled=True,
    influxdb_url="http://localhost:8086",
    influxdb_org="sollol",
    influxdb_bucket="sollol_metrics"
)

# Create pool (metrics will be logged automatically)
pool = OllamaPool(
    discover_all_nodes=True,
    exclude_localhost=True
)
```

### Install Python Client

```bash
pip install sollol[metrics]
# or
pip install influxdb-client
```

---

## 3. Install Grafana (Bare Metal)

### Ubuntu/Debian

```bash
# Add Grafana repository
sudo apt-get install -y software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -

# Install Grafana
sudo apt-get update
sudo apt-get install grafana

# Start service
sudo systemctl start grafana-server
sudo systemctl enable grafana-server
```

Access Grafana at `http://localhost:3000` (default login: `admin/admin`)

---

## 4. Add InfluxDB Data Source in Grafana

1. **Open Grafana** → `http://localhost:3000`
2. **Login** (default: `admin/admin`, change on first login)
3. **Add Data Source**:
   - Go to **Configuration** → **Data Sources**
   - Click **Add data source**
   - Select **InfluxDB**

4. **Configure InfluxDB Connection**:
   ```
   Query Language: Flux
   URL: http://localhost:8086
   Organization: sollol
   Token: <your-influxdb-token>
   Default Bucket: sollol_metrics
   ```

5. **Test Connection** → Click "Save & Test"

---

## 5. Import SOLLOL Dashboard

### Option A: Manual Dashboard Creation

Create a new dashboard and add these panels:

#### **Panel 1: Node Health Over Time**
```flux
from(bucket: "sollol_metrics")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "node_health")
  |> filter(fn: (r) => r._field == "healthy")
  |> aggregateWindow(every: 30s, fn: mean)
```

#### **Panel 2: Average Latency by Node**
```flux
from(bucket: "sollol_metrics")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "node_health")
  |> filter(fn: (r) => r._field == "latency_ms")
  |> aggregateWindow(every: 1m, fn: mean)
  |> group(columns: ["node"])
```

#### **Panel 3: Request Success Rate**
```flux
from(bucket: "sollol_metrics")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "request")
  |> filter(fn: (r) => r._field == "success")
  |> aggregateWindow(every: 5m, fn: mean)
  |> map(fn: (r) => ({ r with _value: r._value * 100.0 }))
```

#### **Panel 4: VRAM Usage**
```flux
from(bucket: "sollol_metrics")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "node_health")
  |> filter(fn: (r) => r._field == "vram_usage_percent")
  |> aggregateWindow(every: 1m, fn: mean)
  |> group(columns: ["node"])
```

#### **Panel 5: RPC Backend Reachability**
```flux
from(bucket: "sollol_metrics")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "rpc_health")
  |> filter(fn: (r) => r._field == "reachable")
  |> aggregateWindow(every: 30s, fn: mean)
```

#### **Panel 6: Requests Per Second**
```flux
from(bucket: "sollol_metrics")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "request")
  |> filter(fn: (r) => r._field == "latency_ms")
  |> aggregateWindow(every: 1m, fn: count)
  |> map(fn: (r) => ({ r with _value: float(v: r._value) / 60.0 }))
```

### Option B: Use Dashboard JSON (Quick Setup)

Save the dashboard JSON below to `sollol_grafana_dashboard.json` and import it:

1. Go to **Dashboards** → **Import**
2. Upload `sollol_grafana_dashboard.json`
3. Select your InfluxDB data source
4. Click **Import**

---

## 6. Grafana Dashboard JSON Configuration

Create file: `sollol_grafana_dashboard.json`

```json
{
  "dashboard": {
    "title": "SOLLOL Metrics",
    "uid": "sollol-metrics",
    "timezone": "browser",
    "schemaVersion": 38,
    "version": 1,
    "refresh": "10s",
    "panels": [
      {
        "id": 1,
        "title": "Node Health Status",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 },
        "targets": [
          {
            "query": "from(bucket: \"sollol_metrics\")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n  |> filter(fn: (r) => r._measurement == \"node_health\")\n  |> filter(fn: (r) => r._field == \"healthy\")\n  |> aggregateWindow(every: 30s, fn: mean)\n  |> group(columns: [\"node\"])",
            "refId": "A"
          }
        ],
        "options": {
          "legend": { "displayMode": "table", "placement": "right" }
        }
      },
      {
        "id": 2,
        "title": "Average Latency (ms)",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 0 },
        "targets": [
          {
            "query": "from(bucket: \"sollol_metrics\")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n  |> filter(fn: (r) => r._measurement == \"node_health\")\n  |> filter(fn: (r) => r._field == \"latency_ms\")\n  |> aggregateWindow(every: 1m, fn: mean)\n  |> group(columns: [\"node\"])",
            "refId": "A"
          }
        ]
      },
      {
        "id": 3,
        "title": "Request Success Rate (%)",
        "type": "stat",
        "gridPos": { "h": 6, "w": 6, "x": 0, "y": 8 },
        "targets": [
          {
            "query": "from(bucket: \"sollol_metrics\")\n  |> range(start: -1h)\n  |> filter(fn: (r) => r._measurement == \"request\")\n  |> filter(fn: (r) => r._field == \"success\")\n  |> mean()\n  |> map(fn: (r) => ({ r with _value: r._value * 100.0 }))",
            "refId": "A"
          }
        ],
        "options": {
          "graphMode": "area",
          "colorMode": "value",
          "unit": "percent"
        }
      },
      {
        "id": 4,
        "title": "VRAM Usage (%)",
        "type": "timeseries",
        "gridPos": { "h": 6, "w": 12, "x": 6, "y": 8 },
        "targets": [
          {
            "query": "from(bucket: \"sollol_metrics\")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n  |> filter(fn: (r) => r._measurement == \"node_health\")\n  |> filter(fn: (r) => r._field == \"vram_usage_percent\")\n  |> aggregateWindow(every: 1m, fn: mean)\n  |> group(columns: [\"node\"])",
            "refId": "A"
          }
        ]
      },
      {
        "id": 5,
        "title": "RPC Backend Health",
        "type": "timeseries",
        "gridPos": { "h": 6, "w": 6, "x": 18, "y": 8 },
        "targets": [
          {
            "query": "from(bucket: \"sollol_metrics\")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n  |> filter(fn: (r) => r._measurement == \"rpc_health\")\n  |> filter(fn: (r) => r._field == \"reachable\")\n  |> aggregateWindow(every: 30s, fn: mean)\n  |> group(columns: [\"backend\"])",
            "refId": "A"
          }
        ]
      },
      {
        "id": 6,
        "title": "Requests Per Second",
        "type": "graph",
        "gridPos": { "h": 6, "w": 12, "x": 0, "y": 14 },
        "targets": [
          {
            "query": "from(bucket: \"sollol_metrics\")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n  |> filter(fn: (r) => r._measurement == \"request\")\n  |> filter(fn: (r) => r._field == \"latency_ms\")\n  |> aggregateWindow(every: 1m, fn: count)\n  |> map(fn: (r) => ({ r with _value: float(v: r._value) / 60.0 }))",
            "refId": "A"
          }
        ]
      }
    ]
  }
}
```

---

## 7. Verify Metrics are Flowing

### Check InfluxDB has data:

```bash
# Using influx CLI
influx query 'from(bucket:"sollol_metrics") |> range(start:-1h) |> limit(n:10)'
```

### Check SOLLOL is logging:

```python
from sollol.metrics_logger import is_enabled

if is_enabled():
    print("✅ Metrics logging is enabled")
else:
    print("❌ Metrics logging is disabled")
```

---

## 8. Metrics Reference

### Available Measurements

| Measurement      | Description                    | Tags                          | Fields                                                      |
| ---------------- | ------------------------------ | ----------------------------- | ----------------------------------------------------------- |
| `node_health`    | Ollama node health metrics     | `node`, `service`             | `healthy`, `latency_ms`, `models_loaded`, `vram_usage_%`    |
| `rpc_health`     | RPC backend health             | `backend`, `service`          | `reachable`, `latency_ms`                                   |
| `request`        | Request/response metrics       | `service`, `node`, `model`    | `latency_ms`, `success`, `tokens`, `error`                  |
| `routing_decision` | Router selection metrics     | `model`, `selected_node`      | `candidate_nodes`, `selection_latency_ms`                   |

---

## 9. Troubleshooting

### Metrics not appearing in InfluxDB?

```bash
# Check SOLLOL logs
tail -f ~/.sollol/logs/*.log | grep -i influx

# Verify token is set
echo $INFLUX_TOKEN

# Test connection manually
curl -H "Authorization: Token $INFLUX_TOKEN" http://localhost:8086/api/v2/buckets
```

### Grafana can't connect to InfluxDB?

- Verify InfluxDB is running: `sudo systemctl status influxdb`
- Check token permissions in InfluxDB UI
- Ensure bucket name matches exactly (`sollol_metrics`)

### No data in Grafana panels?

- Check time range (default: last 1 hour)
- Verify SOLLOL has been running and processing requests
- Run a test request to generate metrics

---

## 10. Example: Full Stack Startup

```bash
# 1. Start InfluxDB
sudo systemctl start influxdb

# 2. Start Grafana
sudo systemctl start grafana-server

# 3. Export InfluxDB token
export INFLUX_TOKEN=<your-token>

# 4. Start SOLLOL with metrics enabled
python3 -c "
from sollol import OllamaPool, RayHybridRouter
from sollol.rpc_discovery import auto_discover_rpc_backends

# Auto-discover network
pool = OllamaPool(discover_all_nodes=True, exclude_localhost=True)
rpc_backends = auto_discover_rpc_backends()

# Create router (metrics logged automatically)
router = RayHybridRouter(
    ollama_pool=pool,
    rpc_backends=rpc_backends,
    enable_distributed=True
)

print('✅ SOLLOL running with metrics enabled')
print('📊 View metrics: http://localhost:3000')
"
```

---

## Benefits of This Setup

✅ **Historical Metrics**: See performance trends over time
✅ **Real-time Monitoring**: 10-second refresh in Grafana
✅ **Multi-node Visibility**: Compare all nodes/backends at once
✅ **Alerting**: Set up Grafana alerts for failures
✅ **No Code Changes**: SOLLOL automatically logs metrics
✅ **Production-Ready**: InfluxDB + Grafana is industry-standard

---

## Next Steps

- **Alerts**: Configure Grafana alerts for node failures
- **Retention**: Set up InfluxDB retention policies
- **Backup**: Schedule InfluxDB backups
- **Scaling**: Add more SOLLOL instances (same InfluxDB)
