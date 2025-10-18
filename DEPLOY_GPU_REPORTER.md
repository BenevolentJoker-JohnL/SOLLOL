# Deploying GPU Reporter to Ollama Nodes

## Quick Deploy (SSH Method)

For each Ollama node with a GPU, run these commands from your SOLLOL machine:

```bash
# Example for node 192.168.1.21
NODE_IP="192.168.1.21"
REDIS_IP="192.168.1.10"  # Your Redis server IP

# 1. Copy gpu_reporter.py to the node
scp /home/joker/SOLLOL/gpu_reporter.py $NODE_IP:/tmp/

# 2. SSH to the node and install dependencies
ssh $NODE_IP << 'EOF'
# Install dependencies
pip3 install --user gpustat redis requests

# Move reporter to a permanent location
sudo mkdir -p /opt/sollol
sudo mv /tmp/gpu_reporter.py /opt/sollol/
sudo chmod +x /opt/sollol/gpu_reporter.py

# Create systemd service
sudo tee /etc/systemd/system/sollol-gpu-reporter.service > /dev/null << 'SYSTEMD'
[Unit]
Description=SOLLOL GPU Reporter
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/sollol
ExecStart=/usr/bin/python3 /opt/sollol/gpu_reporter.py \
    --redis-host REDIS_IP_HERE \
    --redis-port 6379 \
    --node-id NODE_IP_HERE:11434 \
    --interval 5
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SYSTEMD

# Replace placeholders
sudo sed -i "s/REDIS_IP_HERE/$REDIS_IP/g" /etc/systemd/system/sollol-gpu-reporter.service
sudo sed -i "s/NODE_IP_HERE/$(hostname -I | awk '{print $1}')/g" /etc/systemd/system/sollol-gpu-reporter.service

# Start service
sudo systemctl daemon-reload
sudo systemctl enable sollol-gpu-reporter
sudo systemctl start sollol-gpu-reporter

# Check status
sudo systemctl status sollol-gpu-reporter
EOF
```

## Manual Deploy

If SSH doesn't work, manually on each node:

### 1. Install Dependencies
```bash
pip3 install --user gpustat redis requests
```

### 2. Copy gpu_reporter.py
Copy `/home/joker/SOLLOL/gpu_reporter.py` to the node at `/opt/sollol/gpu_reporter.py`

### 3. Run Reporter
```bash
python3 /opt/sollol/gpu_reporter.py \
    --redis-host 192.168.1.10 \
    --redis-port 6379 \
    --node-id $(hostname -I | awk '{print $1}'):11434 \
    --interval 5
```

## Verify It's Working

From your SOLLOL machine:

```bash
# Check Redis for GPU data
redis-cli keys "sollol:gpu:*"

# View specific node's GPU data
redis-cli get "sollol:gpu:192.168.1.21:11434"

# Check dashboard
curl -s http://localhost:8080/api/network/nodes | python3 -m json.tool
```

## Troubleshooting

### No GPU detected (gpus: [])
- **Check if Ollama is using GPU**: `curl http://NODE_IP:11434/api/ps`
  - If `size_vram: 0` for all models, Ollama is in CPU-only mode
  - Load a model to test: `ollama run llama3.2:1b "test"`

- **Check if nvidia-smi works**: `ssh NODE_IP nvidia-smi`
  - If it fails, GPU drivers may not be installed

### Reporter not publishing data
```bash
# Check reporter logs
ssh NODE_IP sudo journalctl -u sollol-gpu-reporter -f

# Check Redis connectivity from node
ssh NODE_IP redis-cli -h REDIS_IP ping
```

### Data expires too quickly
The TTL is now 120 seconds (2 minutes). If the reporter stops for any reason, data will expire. Check if the reporter service is running:

```bash
ssh NODE_IP sudo systemctl status sollol-gpu-reporter
```

## Expected Output

With reporter running on a node with NVIDIA RTX 4090:

```json
{
  "url": "http://192.168.1.21:11434",
  "status": "healthy",
  "free_vram_mb": 18432,
  "total_vram_mb": 24576,
  "gpu_vendor": "nvidia"
}
```
