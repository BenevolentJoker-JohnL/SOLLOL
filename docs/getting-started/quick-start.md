# Quick Start

Get SOLLOL running in 5 minutes with Docker Compose.

## Prerequisites

- Docker and Docker Compose installed
- 8GB+ RAM recommended
- (Optional) GPU for optimal performance

## Step 1: Clone the Repository

```bash
git clone https://github.com/B-A-M-N/SOLLOL.git
cd SOLLOL
```

## Step 2: Start the Stack

```bash
# Start SOLLOL + 3 Ollama nodes + Prometheus + Grafana
docker-compose up -d

# Check status
docker-compose ps
```

## Step 3: Pull a Model

```bash
# Pull llama3.2 on all nodes
docker exec sollol-ollama-node-1-1 ollama pull llama3.2
docker exec sollol-ollama-node-2-1 ollama pull llama3.2
docker exec sollol-ollama-node-3-1 ollama pull llama3.2
```

## Step 4: Test the Setup

```bash
# Send a test request to SOLLOL (drop-in replacement on port 11434)
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Or use the standard Ollama API (SOLLOL is transparent!)
export OLLAMA_HOST=localhost:11434
ollama run llama3.2 "Hello!"
```

## Step 5: View the Dashboard

Open your browser:

- **SOLLOL Dashboard**: http://localhost:11434/dashboard.html
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9091

## Using the Python SDK

```python
from sollol import connect

# Connect to SOLLOL (drop-in replacement - same port as Ollama!)
sollol = connect("http://localhost:11434")

# Chat with intelligent routing
response = sollol.chat(
    "Explain quantum computing",
    priority=8  # High priority = faster nodes
)

print(response['message']['content'])
```

## Next Steps

- [**Configuration**](configuration.md) - Customize SOLLOL settings
- [**Architecture**](../architecture/overview.md) - Understand how it works
- [**Deployment**](../deployment/docker.md) - Deploy to production
- [**Benchmarks**](../benchmarks/overview.md) - Run performance tests

## Troubleshooting

### Ollama nodes not responding

```bash
# Check logs
docker-compose logs sollol
docker-compose logs ollama-node-1

# Restart a node
docker-compose restart ollama-node-1
```

### Port conflicts

If port 11434 is already in use (e.g., you have Ollama running), stop it first:

```bash
# Stop standalone Ollama (if running)
pkill ollama

# Or change SOLLOL's port in docker-compose.yml
ports:
  - "11435:11434"  # Change external port
```

### GPU not detected

Ensure NVIDIA Container Toolkit is installed:

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Then uncomment GPU sections in `docker-compose.yml`.

## Support

Need help? Open an issue on [GitHub](https://github.com/B-A-M-N/SOLLOL/issues).
