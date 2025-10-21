# SOLLOL Configuration Guide

SOLLOL can be configured in **three ways**, all equally valid. Choose based on your deployment scenario.

---

## 1. Environment Variables (Recommended for Docker/Kubernetes)

Perfect for containerized deployments where configuration is managed externally.

```bash
# Set environment variables
export SOLLOL_PORT=8000
export SOLLOL_RAY_WORKERS=16
export SOLLOL_DASK_WORKERS=8
export SOLLOL_BATCH_PROCESSING=true
export SOLLOL_AUTOBATCH_INTERVAL=30
export RPC_BACKENDS="192.168.1.10:50052,192.168.1.11:50052"
export OLLAMA_NODES="192.168.1.20:11434,192.168.1.21:11434"

# Start SOLLOL (reads env vars)
sollol up
# OR
python -m sollol.gateway
```

### Available Environment Variables

| Variable | Alternative | Default | Description |
|----------|-------------|---------|-------------|
| `SOLLOL_PORT` | `PORT` | `11434` | Gateway port |
| `SOLLOL_RAY_WORKERS` | `RAY_WORKERS` | `4` | Number of Ray actors for parallel execution |
| `SOLLOL_DASK_WORKERS` | `DASK_WORKERS` | `2` | Number of Dask workers for batch processing |
| `SOLLOL_BATCH_PROCESSING` | - | `true` | Enable Dask batch processing (true/false) |
| `SOLLOL_AUTOBATCH_INTERVAL` | `AUTOBATCH_INTERVAL` | `60` | Seconds between autobatch cycles |
| `SOLLOL_REDIS_URL` | - | `redis://localhost:6379` | Redis URL for GPU metadata and distributed state |
| `SOLLOL_DASHBOARD` | - | `true` | Enable observability dashboard |
| `SOLLOL_DASHBOARD_PORT` | - | `8080` | Dashboard port |
| `RPC_BACKENDS` | - | - | Comma-separated RPC backends for distributed inference (layer distribution) |
| `OLLAMA_NODES` | - | - | Comma-separated Ollama nodes for task distribution |

**Note:** `SOLLOL_*` prefixed variables take precedence over alternatives.

### Remote Coordinator Environment Variables

For intelligent coordinator placement across distributed clusters:

| Variable | Default | Description |
|----------|---------|-------------|
| `SOLLOL_REMOTE_COORDINATOR` | `true` | Enable remote coordinator execution |
| `SOLLOL_MODEL_VRAM_THRESHOLD_MB` | `16384` | VRAM threshold (16GB) for Ollama vs RPC routing |
| `RAY_ADDRESS` | `auto` | Ray cluster address (auto-detects or specify head node) |

**Example - Disable Remote Coordinator:**
```bash
export SOLLOL_REMOTE_COORDINATOR=false
sollol up
```

### Docker Example

> **Note**: Docker deployments are not fully battle-tested. Bare-metal deployment is recommended for production.

```dockerfile
FROM python:3.11-slim
RUN pip install sollol
ENV SOLLOL_RAY_WORKERS=16
ENV SOLLOL_PORT=11434
CMD ["sollol", "up"]
```

```bash
docker run -e SOLLOL_RAY_WORKERS=16 -p 11434:11434 sollol:latest
```

### Kubernetes Example

> **Note**: Kubernetes deployments are not fully battle-tested. Bare-metal deployment is recommended for production.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sollol
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: sollol
        image: sollol:latest
        env:
        - name: SOLLOL_RAY_WORKERS
          value: "16"
        - name: SOLLOL_DASK_WORKERS
          value: "8"
        - name: SOLLOL_PORT
          value: "11434"
        - name: RPC_BACKENDS
          value: "rpc-node-1:50052,rpc-node-2:50052"
        ports:
        - containerPort: 11434
```

---

## 2. Programmatic Configuration (Recommended for Python Applications)

Perfect for applications that embed SOLLOL and want full programmatic control.

### Basic Usage

```python
from sollol import SOLLOL

# Zero-config (uses defaults or env vars)
sollol = SOLLOL()
sollol.start(blocking=False)

# Your application continues here
# SOLLOL gateway runs in background thread on port 11434
```

### Full Configuration

```python
from sollol import SOLLOL

# Explicit configuration (overrides env vars)
sollol = SOLLOL(
    port=8000,
    ray_workers=16,
    dask_workers=8,
    enable_batch_processing=True,
    autobatch_interval=30,
    ollama_nodes=[
        {"host": "192.168.1.20", "port": 11434},
        {"host": "192.168.1.21", "port": 11434}
    ],
    rpc_backends=[
        {"host": "192.168.1.10", "port": 50052},
        {"host": "192.168.1.11", "port": 50052}
    ]
)

# Start in background (non-blocking)
sollol.start(blocking=False)

# Your application logic here
import requests
response = requests.post("http://localhost:8000/api/chat", json={
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "Hello!"}]
})

# Check status
status = sollol.get_status()
print(f"Ray workers: {status['ray_workers']}")
print(f"Dask workers: {status['dask_workers']}")

# Get health
health = sollol.get_health()
print(f"Intelligent routing: {health['intelligent_routing']['enabled']}")

# Stop when done
sollol.stop()
```

### Configuration Priority

1. **Explicit parameters** (highest priority)
2. **Environment variables** (SOLLOL_* prefix)
3. **Legacy environment variables** (fallback)
4. **Defaults** (lowest priority)

```python
# Example: Env var + explicit override
# ENV: SOLLOL_RAY_WORKERS=32
sollol = SOLLOL(ray_workers=16)  # Uses 16 (explicit override)
sollol = SOLLOL()  # Uses 32 (from env)
```

---

## 3. CLI Arguments (Recommended for Manual Operation)

Perfect for testing, development, and manual deployments.

### Basic Usage

```bash
# Zero-config (auto-discovers Ollama nodes and RPC backends)
sollol up

# Custom port
sollol up --port 8000

# Custom workers
sollol up --ray-workers 16 --dask-workers 8

# Disable batch processing
sollol up --no-batch-processing
```

### Full Configuration

```bash
sollol up \
  --port 8000 \
  --ray-workers 16 \
  --dask-workers 8 \
  --batch-processing \
  --autobatch-interval 30 \
  --rpc-backends "192.168.1.10:50052,192.168.1.11:50052" \
  --ollama-nodes "192.168.1.20:11434,192.168.1.21:11434"
```

### Available CLI Options

```bash
sollol up --help
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--port` | int | `11434` | Gateway port |
| `--ray-workers` | int | `4` | Ray actors for parallel execution |
| `--dask-workers` | int | `2` | Dask workers for batch processing |
| `--batch-processing` / `--no-batch-processing` | flag | `true` | Enable/disable batch processing |
| `--autobatch-interval` | int | `60` | Seconds between autobatch cycles |
| `--rpc-backends` | string | - | Comma-separated RPC backends |
| `--ollama-nodes` | string | - | Comma-separated Ollama nodes |

---

## Configuration Scenarios

### Scenario 1: Local Development (Zero-Config)

```bash
# Just start it - auto-discovers everything
sollol up
```

### Scenario 2: Production Docker Deployment

```bash
docker run \
  -e SOLLOL_RAY_WORKERS=32 \
  -e SOLLOL_DASK_WORKERS=16 \
  -e RPC_BACKENDS="rpc1:50052,rpc2:50052" \
  -p 11434:11434 \
  sollol:latest
```

### Scenario 3: Python Application Integration

```python
from sollol import SOLLOL
import os

# Read custom config from your app's config system
sollol = SOLLOL(
    port=int(os.getenv("MY_APP_SOLLOL_PORT", 11434)),
    ray_workers=16,
    enable_batch_processing=True
)
sollol.start(blocking=False)

# Your application continues
```

### Scenario 4: Multi-Environment Deployment

```bash
# staging.env
SOLLOL_RAY_WORKERS=4
SOLLOL_DASK_WORKERS=2

# production.env
SOLLOL_RAY_WORKERS=32
SOLLOL_DASK_WORKERS=16
SOLLOL_BATCH_PROCESSING=true
RPC_BACKENDS="prod-rpc-1:50052,prod-rpc-2:50052"

# Deploy
docker run --env-file production.env sollol:latest
```

---

## Verification

### Check Configuration

```bash
# Via health endpoint
curl http://localhost:11434/api/health | jq

# Returns:
{
  "status": "healthy",
  "service": "SOLLOL",
  "version": "0.4.0",
  "ray_parallel_execution": {
    "enabled": true,
    "actors": 16
  },
  "dask_batch_processing": {
    "enabled": true,
    "workers": 8
  },
  "intelligent_routing": {
    "enabled": true,
    "factors": "7-factor scoring"
  }
}
```

### Check Stats

```bash
curl http://localhost:11434/api/stats | jq
```

---

## Best Practices

1. **Use environment variables for deployment** (Docker, Kubernetes, cloud platforms)
2. **Use programmatic configuration for embedded applications**
3. **Use CLI arguments for development and testing**
4. **Always use `SOLLOL_*` prefixed env vars** for clarity (avoids conflicts)
5. **Start with zero-config** (auto-discovery) and only add explicit config when needed
6. **Monitor via `/api/health`** and `/api/stats` endpoints

---

## Troubleshooting

### Configuration Not Applied

1. Check precedence: Explicit > Env vars > Defaults
2. Verify env vars are exported: `echo $SOLLOL_RAY_WORKERS`
3. Check logs: SOLLOL prints configuration on startup
4. Query health endpoint: `curl localhost:11434/api/health`

### Auto-Discovery Not Finding Nodes

1. Ensure Ollama nodes are on same network
2. Check firewall rules (port 11434)
3. Manually specify nodes:
   ```bash
   export OLLAMA_NODES="node1:11434,node2:11434"
   sollol up
   ```

### Ray/Dask Not Starting

1. Check if processes already running: `pkill -f "ray::" && pkill -f "dask"`
2. Verify ports available: `netstat -tulpn | grep 8786`
3. Check logs for errors

---

## Examples

See:
- `examples/basic_usage.py` - Simple programmatic usage
- `examples/application_integration.py` - Full application integration
- `docker-compose.yml` - Docker deployment example
- `kubernetes/` - Kubernetes deployment manifests
