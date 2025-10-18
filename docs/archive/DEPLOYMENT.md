# SOLLOL Deployment Guide

Production deployment options for SOLLOL across different environments.

## Table of Contents

- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Deployments](#cloud-deployments)
- [Scaling Strategies](#scaling-strategies)
- [Monitoring & Observability](#monitoring--observability)

---

## Docker Deployment

### Single-Node Docker

**Dockerfile**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install -e .

# Copy application
COPY src/ ./src/
COPY config/ ./config/

# Expose ports
EXPOSE 11434 9090

# Run SOLLOL
CMD ["python", "-m", "sollol.cli", "up", "--workers", "4", "--port", "11434"]
```

**Build and Run**:
```bash
# Build image
docker build -t sollol:latest .

# Run container
docker run -d \
  -p 11434:11434 \
  -p 9090:9090 \
  -v $(pwd)/config:/app/config \
  --name sollol \
  sollol:latest
```

### Docker Compose (Recommended)

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  # SOLLOL Gateway (Drop-in replacement - listens on port 11434)
  sollol:
    build: .
    ports:
      - "11434:11434"  # Drop-in replacement for Ollama
      - "9090:9090"    # Prometheus metrics
    environment:
      - RAY_WORKERS=4
      - DASK_WORKERS=4
      - SOLLOL_AUTH_ENABLED=true
      - SOLLOL_PORT=11434
    volumes:
      - ./config/hosts.txt:/app/config/hosts.txt
      - sollol-data:/app/data
    networks:
      - sollol-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G

  # Ollama Node 1 (GPU) - Backend on different port
  ollama-gpu-1:
    image: ollama/ollama:latest
    ports:
      - "11435:11434"  # Backend node
    volumes:
      - ollama-models:/root/.ollama
    networks:
      - sollol-network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Ollama Node 2 (CPU) - Backend on different port
  ollama-cpu-1:
    image: ollama/ollama:latest
    ports:
      - "11436:11434"  # Backend node
    volumes:
      - ollama-models-cpu:/root/.ollama
    networks:
      - sollol-network

  # Prometheus (Metrics)
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    networks:
      - sollol-network
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  # Grafana (Visualization)
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    networks:
      - sollol-network
    depends_on:
      - prometheus

volumes:
  sollol-data:
  ollama-models:
  ollama-models-cpu:
  prometheus-data:
  grafana-data:

networks:
  sollol-network:
    driver: bridge
```

**prometheus.yml**:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'sollol'
    static_configs:
      - targets: ['sollol:9090']
```

**Start Stack**:
```bash
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f sollol

# Scale Ollama nodes
docker-compose up -d --scale ollama-cpu-1=3
```

---

## Kubernetes Deployment

### Basic Deployment

**sollol-deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sollol
  labels:
    app: sollol
spec:
  replicas: 2
  selector:
    matchLabels:
      app: sollol
  template:
    metadata:
      labels:
        app: sollol
    spec:
      containers:
      - name: sollol
        image: sollol:latest
        ports:
        - containerPort: 11434
          name: api
        - containerPort: 9090
          name: metrics
        env:
        - name: RAY_WORKERS
          value: "4"
        - name: DASK_WORKERS
          value: "4"
        - name: SOLLOL_AUTH_ENABLED
          value: "true"
        - name: SOLLOL_ADMIN_KEY
          valueFrom:
            secretKeyRef:
              name: sollol-secrets
              key: admin-key
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 11434
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/health
            port: 11434
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: sollol-service
spec:
  selector:
    app: sollol
  ports:
  - name: api
    port: 11434
    targetPort: 11434
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer
---
apiVersion: v1
kind: Secret
metadata:
  name: sollol-secrets
type: Opaque
stringData:
  admin-key: "your-admin-key-here"
```

**ollama-statefulset.yaml**:
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: ollama
spec:
  serviceName: ollama
  replicas: 3
  selector:
    matchLabels:
      app: ollama
  template:
    metadata:
      labels:
        app: ollama
    spec:
      containers:
      - name: ollama
        image: ollama/ollama:latest
        ports:
        - containerPort: 11434
        volumeMounts:
        - name: models
          mountPath: /root/.ollama
        resources:
          requests:
            nvidia.com/gpu: 1  # For GPU nodes
            memory: "16Gi"
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
  volumeClaimTemplates:
  - metadata:
      name: models
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 100Gi
---
apiVersion: v1
kind: Service
metadata:
  name: ollama
spec:
  clusterIP: None  # Headless service
  selector:
    app: ollama
  ports:
  - port: 11434
```

**Deploy to Kubernetes**:
```bash
# Create namespace
kubectl create namespace sollol

# Deploy SOLLOL
kubectl apply -f sollol-deployment.yaml -n sollol

# Deploy Ollama
kubectl apply -f ollama-statefulset.yaml -n sollol

# Check status
kubectl get pods -n sollol
kubectl get svc -n sollol

# Scale SOLLOL
kubectl scale deployment sollol --replicas=4 -n sollol

# Scale Ollama
kubectl scale statefulset ollama --replicas=5 -n sollol
```

### Horizontal Pod Autoscaler (HPA)

**sollol-hpa.yaml**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sollol-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sollol
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: sollol_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
```

---

## Cloud Deployments

### AWS EKS

**1. Create EKS Cluster**:
```bash
eksctl create cluster \
  --name sollol-cluster \
  --region us-west-2 \
  --nodegroup-name gpu-nodes \
  --node-type p3.2xlarge \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 5 \
  --managed
```

**2. Deploy SOLLOL**:
```bash
kubectl apply -f sollol-deployment.yaml
kubectl apply -f ollama-statefulset.yaml
kubectl apply -f sollol-hpa.yaml
```

**3. Expose with ALB**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: sollol-alb
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-ssl-cert: "arn:aws:acm:..."
spec:
  type: LoadBalancer
  selector:
    app: sollol
  ports:
  - port: 443
    targetPort: 11434
```

### Google Cloud GKE

**1. Create GKE Cluster**:
```bash
gcloud container clusters create sollol-cluster \
  --zone us-central1-a \
  --machine-type n1-standard-4 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 10 \
  --accelerator type=nvidia-tesla-t4,count=1
```

**2. Deploy**:
```bash
kubectl apply -f sollol-deployment.yaml
kubectl apply -f ollama-statefulset.yaml
```

### Azure AKS

**1. Create AKS Cluster**:
```bash
az aks create \
  --resource-group sollol-rg \
  --name sollol-cluster \
  --node-count 3 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-cluster-autoscaler \
  --min-count 2 \
  --max-count 10
```

---

## Scaling Strategies

### Vertical Scaling (More Resources)

```yaml
# Increase resources per pod
resources:
  requests:
    cpu: "8"
    memory: "16Gi"
  limits:
    cpu: "16"
    memory: "32Gi"
```

### Horizontal Scaling (More Pods)

```bash
# Manual scaling
kubectl scale deployment sollol --replicas=10

# Auto-scaling with HPA
kubectl autoscale deployment sollol \
  --cpu-percent=70 \
  --min=2 \
  --max=20
```

### Multi-Region Deployment

```yaml
# Deploy to multiple regions
regions:
  - us-west-2
  - us-east-1
  - eu-west-1

# Use geo-routing (Route53, CloudFlare)
# Each region has independent SOLLOL + Ollama cluster
```

---

## Monitoring & Observability

### Prometheus + Grafana

**Import SOLLOL Dashboard**:
```json
{
  "dashboard": {
    "title": "SOLLOL Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [{
          "expr": "rate(sollol_requests_total[5m])"
        }]
      },
      {
        "title": "Average Latency",
        "targets": [{
          "expr": "sollol_request_latency_seconds"
        }]
      },
      {
        "title": "Host Health",
        "targets": [{
          "expr": "sollol_host_success_rate"
        }]
      }
    ]
  }
}
```

### Logging (ELK Stack)

**filebeat.yml**:
```yaml
filebeat.inputs:
- type: container
  paths:
    - '/var/lib/docker/containers/*/*.log'
  processors:
    - add_kubernetes_metadata:
        in_cluster: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
```

### Tracing (Jaeger)

```python
# Add to gateway.py
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

tracer = trace.get_tracer(__name__)
FastAPIInstrumentor.instrument_app(app)
```

---

## Production Checklist

✅ **Security**:
- [ ] Enable authentication
- [ ] Use HTTPS/TLS
- [ ] Rotate API keys
- [ ] Configure firewall rules
- [ ] Enable audit logging

✅ **Reliability**:
- [ ] Set up health checks
- [ ] Configure auto-scaling
- [ ] Enable multi-AZ deployment
- [ ] Set up backup/restore
- [ ] Test failover scenarios

✅ **Performance**:
- [ ] Tune resource limits
- [ ] Enable caching
- [ ] Configure rate limiting
- [ ] Optimize batch sizes
- [ ] Monitor and profile

✅ **Observability**:
- [ ] Set up Prometheus metrics
- [ ] Configure Grafana dashboards
- [ ] Enable distributed tracing
- [ ] Set up log aggregation
- [ ] Create alerting rules

✅ **Disaster Recovery**:
- [ ] Regular backups
- [ ] Tested restore procedures
- [ ] Multi-region deployment
- [ ] Documented runbooks
- [ ] Incident response plan

---

For more help, see:
- [SECURITY.md](SECURITY.md) for security best practices
- [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- [BENCHMARKS.md](BENCHMARKS.md) for performance tuning

---

**SOLLOL** - Production-ready intelligent orchestration for AI workloads. 🚀
