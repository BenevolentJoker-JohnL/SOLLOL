#!/bin/bash
# SOLLOL Cluster Deployment Script
# Auto-discovers nodes and deploys SOLLOL updates across the entire cluster

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOLLOL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HEAD_NODE_IP=$(hostname -I | awk '{print $1}')

echo "=============================================="
echo "SOLLOL Cluster Deployment Script"
echo "=============================================="
echo "Head Node: $HEAD_NODE_IP"
echo ""

# Function to detect local network
detect_network() {
    local ip=$(hostname -I | awk '{print $1}')
    local network=$(echo $ip | cut -d. -f1-3)
    echo "${network}.0/24"
}

# Function to discover nodes running Ollama or RPC servers
discover_nodes() {
    echo "🔍 Discovering nodes on network..."

    local network=$(detect_network)
    echo "   Scanning network: $network"

    # Create temporary file for results
    local temp_file=$(mktemp)

    # Scan for Ollama nodes (port 11434)
    echo "   Scanning for Ollama nodes (port 11434)..."
    nmap -p 11434 --open -oG - "$network" 2>/dev/null | \
        grep "11434/open" | \
        awk '{print $2}' >> "$temp_file"

    # Scan for RPC backends (port 50052)
    echo "   Scanning for RPC backends (port 50052)..."
    nmap -p 50052 --open -oG - "$network" 2>/dev/null | \
        grep "50052/open" | \
        awk '{print $2}' >> "$temp_file"

    # Scan for SSH (port 22) - potential worker nodes
    echo "   Scanning for SSH nodes (port 22)..."
    nmap -p 22 --open -oG - "$network" 2>/dev/null | \
        grep "22/open" | \
        awk '{print $2}' >> "$temp_file"

    # Remove duplicates and localhost
    cat "$temp_file" | sort -u | grep -v "^127\." | grep -v "^$HEAD_NODE_IP$" || true
    rm -f "$temp_file"
}

# Check if nmap is installed
if ! command -v nmap &> /dev/null; then
    echo "⚠️  nmap not found. Installing..."
    sudo apt-get update && sudo apt-get install -y nmap
fi

# Discover nodes
DISCOVERED_NODES=$(discover_nodes)

if [ -z "$DISCOVERED_NODES" ]; then
    echo "❌ No nodes discovered on network"
    echo "   Make sure worker nodes are:"
    echo "   1. On the same network"
    echo "   2. Running Ollama or RPC backends"
    echo "   3. Have SSH enabled"
    exit 1
fi

echo ""
echo "✅ Discovered nodes:"
echo "$DISCOVERED_NODES" | while read node; do
    echo "   - $node"
done
echo ""

# Ask for confirmation
read -p "Deploy SOLLOL to these nodes? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled"
    exit 0
fi

# Check for SSH key
echo ""
echo "Step 1: Setting up SSH access"
echo "------------------------------"

if [ ! -f ~/.ssh/id_ed25519 ] && [ ! -f ~/.ssh/id_rsa ]; then
    echo "📝 Generating SSH key..."
    ssh-keygen -t ed25519 -C "sollol-cluster-admin" -N "" -f ~/.ssh/id_ed25519
    echo "✅ SSH key generated"
fi

# Determine which key to use
if [ -f ~/.ssh/id_ed25519 ]; then
    SSH_KEY=~/.ssh/id_ed25519
else
    SSH_KEY=~/.ssh/id_rsa
fi

echo "Using SSH key: $SSH_KEY"
echo ""

# Copy SSH keys to all nodes
echo "📤 Copying SSH key to worker nodes..."
echo "   (You may need to enter passwords)"
echo ""

echo "$DISCOVERED_NODES" | while read node; do
    echo "Setting up $node..."

    # Try to copy SSH key
    if ssh-copy-id -i "$SSH_KEY" "$USER@$node" 2>/dev/null; then
        echo "✅ SSH key copied to $node"
    else
        echo "⚠️  Failed to copy SSH key to $node (may already exist)"
    fi
done

echo ""
echo "Step 2: Testing SSH connectivity"
echo "---------------------------------"

ACCESSIBLE_NODES=""
FAILED_NODES=""

echo "$DISCOVERED_NODES" | while read node; do
    if ssh -o ConnectTimeout=5 -o BatchMode=yes "$USER@$node" "echo ok" &>/dev/null; then
        echo "✅ $node - SSH working"
        echo "$node" >> /tmp/sollol_accessible_nodes
    else
        echo "❌ $node - SSH failed"
        echo "$node" >> /tmp/sollol_failed_nodes
    fi
done

# Read accessible nodes
if [ -f /tmp/sollol_accessible_nodes ]; then
    ACCESSIBLE_NODES=$(cat /tmp/sollol_accessible_nodes)
    rm -f /tmp/sollol_accessible_nodes
fi

if [ -f /tmp/sollol_failed_nodes ]; then
    FAILED_NODES=$(cat /tmp/sollol_failed_nodes)
    rm -f /tmp/sollol_failed_nodes
fi

if [ -z "$ACCESSIBLE_NODES" ]; then
    echo ""
    echo "❌ No nodes are accessible via SSH"
    echo "   Please set up SSH keys manually or check firewall rules"
    exit 1
fi

echo ""
echo "Step 3: Deploying SOLLOL to accessible nodes"
echo "---------------------------------------------"

deploy_to_node() {
    local node=$1

    echo ""
    echo "📦 Deploying to $node..."

    # Check if SOLLOL directory exists
    if ssh "$USER@$node" "[ -d ~/SOLLOL ]"; then
        echo "   ✅ SOLLOL directory exists"

        # Pull latest changes
        echo "   📥 Pulling latest changes..."
        ssh "$USER@$node" "cd ~/SOLLOL && git pull origin main"

        # Update installation
        echo "   🔧 Updating installation..."
        ssh "$USER@$node" "cd ~/SOLLOL && pip install -e . --quiet"

    else
        echo "   📥 Cloning SOLLOL repository..."
        ssh "$USER@$node" "git clone https://github.com/B-A-M-N/SOLLOL.git ~/SOLLOL"

        # Install
        echo "   🔧 Installing SOLLOL..."
        ssh "$USER@$node" "cd ~/SOLLOL && pip install -e . --quiet"
    fi

    echo "   ✅ $node deployment complete"
}

# Deploy to each accessible node
echo "$ACCESSIBLE_NODES" | while read node; do
    deploy_to_node "$node" &
done

# Wait for all deployments to complete
wait

echo ""
echo "Step 4: Starting Ray workers"
echo "----------------------------"

echo "$ACCESSIBLE_NODES" | while read node; do
    echo "Starting Ray worker on $node..."

    # Stop existing Ray instance
    ssh "$USER@$node" "ray stop 2>/dev/null || true"

    # Start Ray worker
    if ssh "$USER@$node" "ray start --address='$HEAD_NODE_IP:6380'"; then
        echo "✅ Ray worker started on $node"
    else
        echo "⚠️  Failed to start Ray worker on $node"
    fi
done

echo ""
echo "Step 5: Registering GPU nodes"
echo "-----------------------------"

echo "$ACCESSIBLE_NODES" | while read node; do
    # Check if node has GPU
    if ssh "$USER@$node" "command -v nvidia-smi &>/dev/null || command -v rocm-smi &>/dev/null"; then
        echo "🎮 Registering GPU node: $node"

        ssh "$USER@$node" "cd ~/SOLLOL && python3 scripts/register_gpu_node.py --redis-url redis://$HEAD_NODE_IP:6379 --rpc-host $node --rpc-port 50052"

        echo "✅ GPU node $node registered"
    else
        echo "⏭️  $node has no GPU, skipping registration"
    fi
done

echo ""
echo "=============================================="
echo "✅ Cluster Deployment Complete!"
echo "=============================================="
echo ""
echo "Summary:"
echo "  Head Node: $HEAD_NODE_IP:6380"
echo "  Accessible Nodes: $(echo "$ACCESSIBLE_NODES" | wc -l)"
if [ ! -z "$FAILED_NODES" ]; then
    echo "  Failed Nodes: $(echo "$FAILED_NODES" | wc -l)"
fi
echo ""
echo "Verification:"
echo "  ray status"
echo "  redis-cli keys 'sollol:rpc:node:*'"
echo ""
echo "Next steps:"
echo "  1. Start SOLLOL: sollol up"
echo "  2. Test distributed inference"
echo "  3. Monitor Ray dashboard: http://$HEAD_NODE_IP:8265"
echo ""
