#!/bin/bash
# SOLLOL Distributed Inference Demo - Happy Path
# This script demonstrates a 3-node CPU distributed inference setup

set -e

echo "🚀 SOLLOL Distributed Inference Demo"
echo "======================================"
echo ""

# Configuration
COORDINATOR_PORT=18080
RPC_PORT_1=50052
RPC_PORT_2=50053
MODEL_PATH="/usr/share/ollama/.ollama/models/blobs/sha256-e73cc17c718156e5ad34b119eb363e2c10389a503673f9c36144c42dfde8334c"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found at $MODEL_PATH"
    echo "   Run: ollama pull codellama:13b"
    exit 1
fi

echo "📋 Setup Configuration:"
echo "   Model: codellama:13b"
echo "   Coordinator Port: $COORDINATOR_PORT"
echo "   RPC Backend 1: 10.9.66.45:$RPC_PORT_1"
echo "   RPC Backend 2: 10.9.66.48:$RPC_PORT_2"
echo ""

# Step 1: Verify RPC backends are running
echo "🔍 Step 1: Verifying RPC backends..."
echo ""

echo "   Checking 10.9.66.45:$RPC_PORT_1..."
if nc -z 10.9.66.45 $RPC_PORT_1 2>/dev/null; then
    echo "   ✅ RPC backend 1 is reachable"
else
    echo "   ⚠️  RPC backend 1 not reachable - ensure rpc-server is running on .45"
    echo "      SSH to node: ssh 10.9.66.45"
    echo "      Start server: nohup ~/llama.cpp/build-cpu/bin/rpc-server --host 0.0.0.0 --port $RPC_PORT_1 > /tmp/rpc-server.log 2>&1 &"
fi

echo "   Checking 10.9.66.48:$RPC_PORT_2..."
if nc -z 10.9.66.48 $RPC_PORT_2 2>/dev/null; then
    echo "   ✅ RPC backend 2 is reachable"
else
    echo "   ⚠️  RPC backend 2 not reachable - ensure rpc-server is running on .48"
    echo "      SSH to node: ssh 10.9.66.48"
    echo "      Start server: nohup ~/llama.cpp/build-cpu/bin/rpc-server --host 0.0.0.0 --port $RPC_PORT_2 > /tmp/rpc-server.log 2>&1 &"
fi
echo ""

# Step 2: Start coordinator (if not already running)
echo "🎯 Step 2: Starting llama.cpp coordinator..."
echo ""

if pgrep -f "llama-server.*--port $COORDINATOR_PORT" > /dev/null; then
    echo "   ℹ️  Coordinator already running on port $COORDINATOR_PORT"
    PID=$(pgrep -f "llama-server.*--port $COORDINATOR_PORT")
    echo "   PID: $PID"
else
    echo "   Starting coordinator with 2 RPC backends..."
    nohup llama-server \
        --model "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port $COORDINATOR_PORT \
        --rpc 10.9.66.45:$RPC_PORT_1,10.9.66.48:$RPC_PORT_2 \
        --ctx-size 2048 \
        --parallel 1 \
        > /tmp/coordinator-$COORDINATOR_PORT.log 2>&1 &

    echo "   ✅ Coordinator started (PID: $!)"
    echo "   Log: /tmp/coordinator-$COORDINATOR_PORT.log"
    echo "   Waiting for coordinator to be ready (40s for model loading)..."
    sleep 5
fi
echo ""

# Step 3: Wait for coordinator health check
echo "⏳ Step 3: Waiting for coordinator to be ready..."
echo ""

for i in {1..8}; do
    if curl -s http://127.0.0.1:$COORDINATOR_PORT/health > /dev/null 2>&1; then
        echo "   ✅ Coordinator is healthy!"
        break
    else
        echo "   ⏳ Attempt $i/8 - waiting 5s..."
        sleep 5
    fi
done
echo ""

# Step 4: Test distributed inference
echo "🧪 Step 4: Testing distributed inference..."
echo ""

RESPONSE=$(curl -s http://127.0.0.1:$COORDINATOR_PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "Write a haiku about distributed computing"}],
        "max_tokens": 50,
        "temperature": 0.7
    }')

if echo "$RESPONSE" | grep -q '"choices"'; then
    echo "   ✅ SUCCESS - Distributed inference working!"
    echo ""
    echo "   Response:"
    echo "$RESPONSE" | python3 -c "import sys, json; r=json.load(sys.stdin); print('   ', r['choices'][0]['message']['content'])" 2>/dev/null || echo "$RESPONSE" | head -c 200
    echo ""
    echo "   Tokens used: $(echo "$RESPONSE" | python3 -c "import sys, json; r=json.load(sys.stdin); print(r.get('usage', {}).get('total_tokens', 'N/A'))" 2>/dev/null)"
else
    echo "   ⚠️  Model still loading or error occurred"
    echo "   Response: $RESPONSE"
fi
echo ""

# Step 5: Show process memory distribution
echo "📊 Step 5: Memory distribution across nodes..."
echo ""

echo "   Local machine (.154) - Coordinator:"
COORDINATOR_PID=$(pgrep -f "llama-server.*--port $COORDINATOR_PORT" | head -1)
if [ -n "$COORDINATOR_PID" ]; then
    MEM=$(ps -o rss= -p $COORDINATOR_PID | awk '{printf "%.2f MB", $1/1024}')
    echo "      llama-server (PID $COORDINATOR_PID): $MEM"
else
    echo "      Not running"
fi
echo ""

echo "   Remote nodes - RPC backends:"
echo "      10.9.66.45:$RPC_PORT_1 - Check with: ssh 10.9.66.45 'ps aux | grep rpc-server'"
echo "      10.9.66.48:$RPC_PORT_2 - Check with: ssh 10.9.66.48 'ps aux | grep rpc-server'"
echo ""

# Final summary
echo "✨ Demo Complete!"
echo ""
echo "Next steps:"
echo "   1. Use SynapticLlamas: cd ~/SynapticLlamas && python main.py"
echo "   2. Run 'distributed model' command to enable model sharding"
echo "   3. Chat with codellama:13b distributed across 3 nodes"
echo ""
echo "Coordinator URL: http://127.0.0.1:$COORDINATOR_PORT"
echo "Health check:    curl http://127.0.0.1:$COORDINATOR_PORT/health"
echo "Logs:            tail -f /tmp/coordinator-$COORDINATOR_PORT.log"
echo ""
