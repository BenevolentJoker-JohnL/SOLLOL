#!/bin/bash
# SOLLOL - Configure Redis for network access
# This script must be run with sudo: sudo bash scripts/configure_redis_network.sh

set -e

COORDINATOR_IP="10.9.66.154"
REDIS_CONF="/etc/redis/redis.conf"

echo "=========================================="
echo "SOLLOL: Configure Redis Network Access"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Error: This script must be run as root"
    echo "   Usage: sudo bash scripts/configure_redis_network.sh"
    exit 1
fi

# Backup original config
if [ ! -f "$REDIS_CONF.backup" ]; then
    echo "📦 Backing up Redis config..."
    cp "$REDIS_CONF" "$REDIS_CONF.backup"
    echo "   Backup saved to: $REDIS_CONF.backup"
fi

# Check current bind configuration
echo ""
echo "🔍 Current Redis bind configuration:"
grep -n "^bind" "$REDIS_CONF" || grep -n "^# bind" "$REDIS_CONF" | head -3
echo ""

# Update bind configuration
echo "🔧 Updating Redis to listen on network interface..."
sed -i "s/^bind 127.0.0.1 ::1/bind 127.0.0.1 ::1 $COORDINATOR_IP/" "$REDIS_CONF"

# Verify change
echo ""
echo "✅ New bind configuration:"
grep -n "^bind" "$REDIS_CONF"
echo ""

# Restart Redis
echo "🔄 Restarting Redis service..."
systemctl restart redis-server

# Wait for Redis to start
sleep 2

# Verify Redis is running
if systemctl is-active --quiet redis-server; then
    echo "✅ Redis service is running"
else
    echo "❌ Error: Redis service failed to start"
    echo "   Restoring backup..."
    cp "$REDIS_CONF.backup" "$REDIS_CONF"
    systemctl restart redis-server
    exit 1
fi

# Check bind status
echo ""
echo "🔍 Verifying network binding..."
netstat -tuln | grep 6379 || ss -tuln | grep 6379

echo ""
echo "=========================================="
echo "✅ Redis Configuration Complete!"
echo "=========================================="
echo ""
echo "Redis is now listening on:"
echo "  - 127.0.0.1:6379 (localhost)"
echo "  - ::1:6379 (IPv6 localhost)"
echo "  - $COORDINATOR_IP:6379 (network interface)"
echo ""
echo "Next steps:"
echo "1. Test Redis connectivity from remote node:"
echo "   redis-cli -h $COORDINATOR_IP ping"
echo ""
echo "2. Register GPU nodes with coordinator:"
echo "   python3 scripts/register_gpu_node.py --redis-host $COORDINATOR_IP"
echo ""
echo "3. Verify GPU detection:"
echo "   PYTHONPATH=src python3 -c \"from sollol.rpc_discovery import auto_discover_rpc_backends; print(auto_discover_rpc_backends())\""
echo ""
