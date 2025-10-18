# Docker Setup and Testing Guide

## Quick Start

### 1. Start the Test Cluster

```bash
# Start 3 Ollama nodes on ports 11435, 11436, 11437
docker compose -f docker-compose.test.yml up -d

# Check status
docker compose -f docker-compose.test.yml ps
```

### 2. Run Functional Test

```bash
# Automated test script
./test_docker.sh
```

## Manual Testing

### Verify Nodes Are Running

```bash
# Check each node responds
curl http://localhost:11435/api/tags
curl http://localhost:11436/api/tags
curl http://localhost:11437/api/tags

# Should return: {"models":[]}
```

### Pull a Model to All Nodes

```bash
# Pull tinyllama (smallest model, ~637MB)
curl http://localhost:11435/api/pull -d '{"name": "tinyllama"}' &
curl http://localhost:11436/api/pull -d '{"name": "tinyllama"}' &
curl http://localhost:11437/api/pull -d '{"name": "tinyllama"}' &

# Wait for completion (check logs)
docker compose -f docker-compose.test.yml logs -f
```

### Test a Chat Request

```bash
# Send a request to one node
curl http://localhost:11435/api/chat -d '{
  "model": "tinyllama",
  "messages": [{"role": "user", "content": "Say hello in one word"}],
  "stream": false
}'

# Should return JSON with message.content
```

## Docker Compose Configuration

### docker-compose.test.yml

```yaml
services:
  ollama-1:
    image: ollama/ollama:latest
    ports:
      - "11435:11434"  # Map container port 11434 to host 11435
    volumes:
      - ollama-models-1:/root/.ollama
    restart: unless-stopped

  ollama-2:
    image: ollama/ollama:latest
    ports:
      - "11436:11434"
    volumes:
      - ollama-models-2:/root/.ollama
    restart: unless-stopped

  ollama-3:
    image: ollama/ollama:latest
    ports:
      - "11437:11434"
    volumes:
      - ollama-models-3:/root/.ollama
    restart: unless-stopped

volumes:
  ollama-models-1:
  ollama-models-2:
  ollama-models-3:
```

### Why 3 Nodes?

- **Minimum for testing:** Shows load distribution behavior
- **Different ports:** 11435-11437 (avoids conflict with host Ollama on 11434)
- **Separate volumes:** Each node has its own model storage

## Using with SOLLOL

### 1. Configure SOLLOL for Docker Nodes

Update `config/hosts.txt`:
```
http://localhost:11435
http://localhost:11436
http://localhost:11437
```

Or include your host Ollama:
```
http://localhost:11434  # Host Ollama
http://localhost:11435  # Docker node 1
http://localhost:11436  # Docker node 2
http://localhost:11437  # Docker node 3
```

### 2. Start SOLLOL Gateway

```bash
# Using CLI (if installed)
sollol up --port 8000

# Or with Python directly
PYTHONPATH=src python -m sollol.cli up --port 8000
```

### 3. Test SOLLOL Routing

```bash
# Send request through SOLLOL gateway
curl http://localhost:8000/api/chat -d '{
  "model": "tinyllama",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false
}'

# Check which node was selected
# Response includes: "_sollol_routing": {"host": "...", "score": ...}
```

## Common Issues

### Port Already in Use

**Error:** `failed to bind host port for 0.0.0.0:11435`

**Solution:** Something is using that port. Either:
```bash
# Stop the conflicting service
docker ps  # Find the container
docker stop <container-id>

# Or use different ports
# Edit docker-compose.test.yml to use 11438-11440
```

### Containers Won't Start

**Check logs:**
```bash
docker compose -f docker-compose.test.yml logs
```

**Common causes:**
- Docker daemon not running: `sudo systemctl start docker`
- Insufficient disk space: `docker system prune`
- Permission issues: `sudo usermod -aG docker $USER` then logout/login

### Models Not Syncing

**Note:** Each container has separate model storage. If you pull a model to one node, you must pull it to all nodes.

```bash
# Pull to all nodes in parallel
for port in 11435 11436 11437; do
  curl http://localhost:$port/api/pull -d '{"name": "tinyllama"}' &
done
wait
```

## Cleanup

### Stop Containers

```bash
# Stop but keep data
docker compose -f docker-compose.test.yml stop

# Stop and remove containers (keeps volumes)
docker compose -f docker-compose.test.yml down

# Stop, remove containers AND volumes (delete all models)
docker compose -f docker-compose.test.yml down -v
```

### Check Disk Usage

```bash
# See how much space Docker is using
docker system df

# Clean up unused resources
docker system prune -a
```

## Production Deployment

For production, use `docker-compose.yml` which includes:
- SOLLOL gateway container
- 3 Ollama nodes
- Prometheus for metrics
- Grafana for visualization

```bash
# Start full stack
docker compose up -d

# Access SOLLOL gateway
curl http://localhost:11434/api/chat  # Drop-in Ollama replacement

# View metrics
open http://localhost:3000  # Grafana dashboard
```

## Benchmarking with Docker

### Run Comparative Benchmark

```bash
# 1. Start cluster
docker compose -f docker-compose.test.yml up -d

# 2. Wait for nodes to be ready
sleep 10

# 3. Run benchmark
python benchmarks/run_benchmarks.py \
  --sollol-url http://localhost:8000 \
  --hosts localhost:11435,localhost:11436,localhost:11437 \
  --duration 60 \
  --concurrency 5

# Results saved to benchmarks/results/
```

### Important: Limitations

**Single-machine testing limitations:**
- All nodes share same CPU/memory/network
- No real heterogeneous hardware
- Can't prove intelligent routing beats round-robin on performance
- Can only prove routing logic works

**What you CAN test:**
- ✅ Request distribution across nodes
- ✅ Failover when node dies
- ✅ Priority queue behavior
- ✅ Routing decision logic

**What you CAN'T test:**
- ❌ Performance improvements from intelligent routing
- ❌ Resource-aware node selection benefits
- ❌ Network latency optimization

## Troubleshooting

### Check Container Health

```bash
# View all containers
docker compose -f docker-compose.test.yml ps

# Check specific container logs
docker compose -f docker-compose.test.yml logs ollama-1

# Execute command in container
docker compose -f docker-compose.test.yml exec ollama-1 /bin/bash

# Check if Ollama is responding inside container
docker compose -f docker-compose.test.yml exec ollama-1 curl http://localhost:11434/api/tags
```

### Reset Everything

```bash
# Nuclear option: delete everything
docker compose -f docker-compose.test.yml down -v
docker system prune -a -f
docker volume prune -f

# Start fresh
docker compose -f docker-compose.test.yml up -d
```

## Next Steps

1. **Verify basic functionality:** Run `./test_docker.sh`
2. **Pull test models:** At least `tinyllama` to all nodes
3. **Test SOLLOL integration:** Configure hosts.txt and start gateway
4. **Run benchmarks:** Use scripts in `benchmarks/`
5. **Monitor behavior:** Check logs and routing decisions

## Resources

- [Ollama Docker Documentation](https://hub.docker.com/r/ollama/ollama)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [SOLLOL Architecture](ARCHITECTURE.md)
- [Benchmarking Guide](BENCHMARKING.md)
