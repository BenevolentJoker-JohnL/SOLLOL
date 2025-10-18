# SOLLOL Integration Guide

This guide explains how to integrate SOLLOL into your Python application without any external configuration files or CLI commands.

## Why Application-Level Integration?

SOLLOL is designed to be **fully embedded within your application**, allowing you to:

- ✅ Control all configuration programmatically
- ✅ No external config files needed
- ✅ Dynamic runtime updates
- ✅ Full visibility into status and performance
- ✅ Seamless integration with existing app architecture

## Basic Integration

### Step 1: Install SOLLOL

```bash
pip install sollol
```

### Step 2: Configure in Your Application

```python
from sollol import SOLLOL, SOLLOLConfig

# In your application's initialization
class MyApplication:
    def __init__(self):
        # Define SOLLOL configuration specific to your app
        self.sollol_config = SOLLOLConfig(
            ray_workers=4,
            dask_workers=2,
            hosts=["127.0.0.1:11434"],  # Your Ollama instances
            autobatch_interval=60,
            routing_strategy="performance"
        )

        # Initialize SOLLOL instance
        self.sollol = SOLLOL(self.sollol_config)

    def start(self):
        """Start your application and SOLLOL together"""
        print("Starting application...")

        # Start SOLLOL in non-blocking mode
        self.sollol.start(blocking=False)

        print("Application ready!")
        print(f"SOLLOL gateway: http://localhost:8000")

    def stop(self):
        """Stop your application and SOLLOL together"""
        self.sollol.stop()
        print("Application stopped")
```

### Step 3: Use SOLLOL in Your Code

```python
import httpx

class MyApplication:
    # ... initialization code ...

    def query_llm(self, prompt: str):
        """Send a query through SOLLOL's gateway"""
        response = httpx.post(
            "http://localhost:8000/api/chat",
            json={
                "model": "llama3.2",
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30.0
        )
        return response.json()

    def embed_document(self, text: str):
        """Embed a document through SOLLOL"""
        response = httpx.post(
            "http://localhost:8000/api/embed",
            json={"text": text},
            timeout=10.0
        )
        return response.json()

    def queue_for_batch(self, documents: list):
        """Queue multiple documents for batch embedding"""
        response = httpx.post(
            "http://localhost:8000/api/embed/batch",
            json={"docs": documents},
            timeout=5.0
        )
        return response.json()
```

## Advanced Integration Patterns

### Pattern 1: Application-Specific Defaults

Different applications have different needs. Customize SOLLOL accordingly:

```python
# For a document-heavy RAG application
rag_config = SOLLOLConfig(
    ray_workers=2,          # Fewer live requests
    dask_workers=8,         # More batch processing
    autobatch_interval=30,  # Aggressive batching
    autobatch_max_batch_size=500
)

# For a low-latency chatbot
chatbot_config = SOLLOLConfig(
    ray_workers=8,          # Many concurrent users
    dask_workers=1,         # Minimal batching
    autobatch_enabled=False,
    adaptive_metrics_interval=15  # Fast routing updates
)

# For GPU-heavy workloads
gpu_config = SOLLOLConfig(
    hosts=[
        "gpu-server-1:11434",  # GPU nodes first (priority routing)
        "gpu-server-2:11434",
        "cpu-server-1:11434"   # CPU fallback
    ],
    routing_strategy="priority",
    dask_workers=6
)
```

### Pattern 2: Dynamic Resource Adjustment

Adjust SOLLOL resources based on your application's load:

```python
class AdaptiveApplication:
    def __init__(self):
        self.sollol = SOLLOL(SOLLOLConfig(ray_workers=2))
        self.sollol.start(blocking=False)

        # Monitor your application metrics
        self.request_count = 0

    def handle_request(self, request):
        self.request_count += 1

        # Scale up if needed
        if self.request_count > 1000 and self.request_count % 100 == 0:
            current_workers = self.sollol.config.ray_workers
            if current_workers < 8:
                self.sollol.update_config(ray_workers=current_workers + 2)
                print(f"Scaled up to {current_workers + 2} workers")

        # Process request using SOLLOL
        return self.query_llm(request.prompt)
```

### Pattern 3: Health Monitoring

Monitor SOLLOL's health from within your application:

```python
class MonitoredApplication:
    def __init__(self):
        self.sollol = SOLLOL(SOLLOLConfig())
        self.sollol.start(blocking=False)

    def health_check(self) -> dict:
        """Application health check that includes SOLLOL status"""
        sollol_health = self.sollol.get_health()
        sollol_stats = self.sollol.get_stats()

        return {
            "application": "healthy",
            "sollol": {
                "status": sollol_health.get("status"),
                "available_hosts": sum(
                    1 for h in sollol_stats.get("hosts", [])
                    if h["available"]
                ),
                "avg_latency": sum(
                    h["latency_ms"] for h in sollol_stats.get("hosts", [])
                ) / len(sollol_stats.get("hosts", [1]))
            }
        }
```

### Pattern 4: Multi-Environment Configuration

Different configs for dev, staging, and production:

```python
import os

def get_sollol_config():
    """Return environment-specific configuration"""
    env = os.getenv("APP_ENV", "dev")

    if env == "production":
        return SOLLOLConfig(
            ray_workers=8,
            dask_workers=6,
            hosts=[
                "ollama-prod-1:11434",
                "ollama-prod-2:11434",
                "ollama-prod-3:11434"
            ],
            adaptive_metrics_interval=20,
            gateway_port=8000
        )

    elif env == "staging":
        return SOLLOLConfig(
            ray_workers=4,
            dask_workers=2,
            hosts=["ollama-staging:11434"],
            gateway_port=8001
        )

    else:  # dev
        return SOLLOLConfig(
            ray_workers=1,
            dask_workers=1,
            hosts=["127.0.0.1:11434"],
            gateway_port=8000,
            metrics_enabled=False  # Less noise in dev
        )

# In your application
class MyApp:
    def __init__(self):
        config = get_sollol_config()
        self.sollol = SOLLOL(config)
```

## Integration Checklist

When integrating SOLLOL into your application:

- [ ] Create `SOLLOLConfig` with app-specific settings
- [ ] Initialize `SOLLOL` instance in your app's initialization
- [ ] Start SOLLOL with `blocking=False` to avoid blocking your app
- [ ] Update your HTTP clients to use `http://localhost:8000` (or your configured port)
- [ ] Add health checks using `sollol.get_health()` and `sollol.get_stats()`
- [ ] Implement graceful shutdown with `sollol.stop()`
- [ ] (Optional) Add dynamic scaling based on application metrics
- [ ] (Optional) Configure different settings for dev/staging/prod environments

## Common Integration Scenarios

### Scenario 1: Flask Application

```python
from flask import Flask
from sollol import SOLLOL, SOLLOLConfig

app = Flask(__name__)

# Initialize SOLLOL
sollol_config = SOLLOLConfig(hosts=["127.0.0.1:11434"])
sollol = SOLLOL(sollol_config)

@app.before_first_request
def start_sollol():
    sollol.start(blocking=False)

@app.route('/chat', methods=['POST'])
def chat():
    import httpx
    data = request.json
    response = httpx.post(
        "http://localhost:8000/api/chat",
        json=data
    )
    return response.json()

if __name__ == '__main__':
    app.run()
```

### Scenario 2: FastAPI Application

```python
from fastapi import FastAPI
from sollol import SOLLOL, SOLLOLConfig
import httpx

app = FastAPI()

# Initialize SOLLOL
sollol = SOLLOL(SOLLOLConfig())

@app.on_event("startup")
async def startup():
    sollol.start(blocking=False)

@app.on_event("shutdown")
async def shutdown():
    sollol.stop()

@app.post("/chat")
async def chat(prompt: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/chat",
            json={"model": "llama3.2", "messages": [
                {"role": "user", "content": prompt}
            ]}
        )
        return response.json()
```

### Scenario 3: Django Application

```python
# myapp/apps.py
from django.apps import AppConfig
from sollol import SOLLOL, SOLLOLConfig

sollol_instance = None

class MyAppConfig(AppConfig):
    name = 'myapp'

    def ready(self):
        global sollol_instance
        if sollol_instance is None:
            config = SOLLOLConfig(hosts=["127.0.0.1:11434"])
            sollol_instance = SOLLOL(config)
            sollol_instance.start(blocking=False)

# myapp/views.py
from .apps import sollol_instance
import httpx

def chat_view(request):
    prompt = request.POST.get('prompt')
    response = httpx.post(
        "http://localhost:8000/api/chat",
        json={"model": "llama3.2", "messages": [
            {"role": "user", "content": prompt}
        ]}
    )
    return JsonResponse(response.json())
```

## Troubleshooting

### Problem: "No available OLLOL hosts"

**Solution:** Verify your hosts configuration:

```python
status = sollol.get_health()
print(status)  # Check which hosts are available
```

### Problem: Port already in use

**Solution:** Use a different port:

```python
config = SOLLOLConfig(gateway_port=8001)
```

### Problem: SOLLOL not stopping cleanly

**Solution:** For now, kill Ray/Dask processes manually:

```bash
pkill -f "ray::"
pkill -f "dask"
```

## Best Practices

1. **Initialize once**: Create a single SOLLOL instance per application
2. **Use non-blocking mode**: Always use `blocking=False` for application integration
3. **Monitor health**: Regularly check `get_health()` and `get_stats()`
4. **Environment-specific configs**: Use different configurations for dev/staging/prod
5. **Graceful degradation**: Handle SOLLOL failures gracefully in your application
6. **Resource limits**: Set appropriate worker counts based on your infrastructure

---

For more examples, see the `examples/` directory in the SOLLOL repository.
