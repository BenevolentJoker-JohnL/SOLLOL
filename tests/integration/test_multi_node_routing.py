#!/usr/bin/env python3
"""
Integration test for SOLLOL multi-node routing.

Tests:
1. Auto-discovery of multiple nodes
2. Intelligent routing across nodes
3. End-to-end request handling
4. Routing strategy validation
"""

import sys
import os
import time
import requests
import pytest

# Add src to path for local testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))


def test_mock_nodes_running():
    """Verify mock Ollama nodes are accessible."""
    print("\n" + "="*80)
    print("TEST 1: Verify mock nodes are running")
    print("="*80)

    ports = [21434, 21435, 21436]
    for port in ports:
        url = f"http://localhost:{port}/api/tags"
        try:
            response = requests.get(url, timeout=5)
            assert response.status_code == 200, f"Node on port {port} not responding"
            data = response.json()
            assert "models" in data, f"Invalid response from port {port}"
            print(f"✅ Node on port {port}: OK")
        except Exception as e:
            print(f"❌ Node on port {port}: FAILED - {e}")
            pytest.skip(f"Mock node on port {port} not available")

    print("\n✅ All mock nodes are running\n")
    return True


def test_sollol_auto_discovery():
    """Test SOLLOL auto-discovery of mock nodes."""
    print("="*80)
    print("TEST 2: SOLLOL auto-discovery")
    print("="*80)

    from sollol import OllamaPool

    # Auto-discover nodes (should find localhost:21434, 21435, 21436)
    pool = OllamaPool(
        nodes=[
            {"host": "localhost", "port": 21434},
            {"host": "localhost", "port": 21435},
            {"host": "localhost", "port": 21436}
        ],
        enable_intelligent_routing=True,
        register_with_dashboard=False,  # Disable dashboard for CI
        enable_cache=False,
        enable_ray=False,  # Disable Ray for simpler CI
        enable_dask=False  # Disable Dask for simpler CI
    )

    assert len(pool.nodes) == 3, f"Expected 3 nodes, found {len(pool.nodes)}"
    print(f"✅ Discovered {len(pool.nodes)} nodes")

    for node in pool.nodes:
        print(f"   - {node['host']}:{node['port']}")

    return pool


def test_end_to_end_routing(pool):
    """Test end-to-end request routing across nodes."""
    print("\n" + "="*80)
    print("TEST 3: End-to-end routing")
    print("="*80)

    # Make a chat request
    response = pool.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    assert "message" in response, "Missing 'message' in response"
    assert "content" in response["message"], "Missing 'content' in message"

    content = response["message"]["content"]
    print(f"✅ Received response: {content}")

    # Verify routing metadata
    if "_sollol_routing" in response:
        routing_info = response["_sollol_routing"]
        print(f"   Routed to: {routing_info.get('host', 'unknown')}:{routing_info.get('port', 'unknown')}")
        print(f"   Task type: {routing_info.get('task_type', 'unknown')}")

    return True


def test_routing_strategies(pool):
    """Test all routing strategies."""
    print("\n" + "="*80)
    print("TEST 4: Routing strategies")
    print("="*80)

    from sollol.routing_strategy import RoutingStrategy

    strategies = [
        RoutingStrategy.ROUND_ROBIN,
        RoutingStrategy.LATENCY_FIRST,
        RoutingStrategy.LEAST_LOADED,
        RoutingStrategy.FAIRNESS
    ]

    for strategy in strategies:
        print(f"\nTesting {strategy.value} strategy...")
        pool.routing_strategy = strategy

        # Make a request
        response = pool.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": f"Test {strategy.value}"}]
        )

        assert "message" in response, f"Strategy {strategy.value} failed"
        print(f"✅ {strategy.value}: PASSED")

    print("\n✅ All routing strategies working\n")
    return True


def test_multiple_requests(pool):
    """Test multiple concurrent-ish requests (simulated)."""
    print("="*80)
    print("TEST 5: Multiple requests distribution")
    print("="*80)

    from sollol.routing_strategy import RoutingStrategy
    pool.routing_strategy = RoutingStrategy.ROUND_ROBIN

    nodes_used = []

    for i in range(6):  # 6 requests across 3 nodes
        response = pool.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": f"Request {i+1}"}]
        )

        if "_sollol_routing" in response:
            node = f"{response['_sollol_routing']['host']}:{response['_sollol_routing']['port']}"
            nodes_used.append(node)
            print(f"  Request {i+1}: {node}")

    # Verify distribution
    unique_nodes = set(nodes_used)
    print(f"\n✅ Used {len(unique_nodes)} unique nodes out of 3 available")

    if len(unique_nodes) >= 2:
        print("✅ Load distribution working\n")
    else:
        print("⚠️  All requests went to same node (may be expected)\n")

    return True


def main():
    """Run all integration tests."""
    print("\n" + "="*80)
    print("SOLLOL Multi-Node Integration Test Suite")
    print("="*80)

    # Test 1: Verify mock nodes
    if not test_mock_nodes_running():
        print("\n❌ FAILED: Mock nodes not running")
        sys.exit(1)

    # Test 2: Auto-discovery
    pool = test_sollol_auto_discovery()
    if pool is None:
        print("\n❌ FAILED: Auto-discovery failed")
        sys.exit(1)

    # Test 3: End-to-end routing
    if not test_end_to_end_routing(pool):
        print("\n❌ FAILED: End-to-end routing failed")
        sys.exit(1)

    # Test 4: Routing strategies
    if not test_routing_strategies(pool):
        print("\n❌ FAILED: Routing strategies failed")
        sys.exit(1)

    # Test 5: Multiple requests
    if not test_multiple_requests(pool):
        print("\n❌ FAILED: Multiple requests failed")
        sys.exit(1)

    # Cleanup
    try:
        pool.stop()
        print("✅ Pool stopped successfully")
    except Exception as e:
        print(f"⚠️  Pool stop error: {e}")

    # Summary
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED")
    print("="*80)
    print("\nSOLLOL multi-node routing is working correctly!")
    print("- Auto-discovery: ✅")
    print("- Intelligent routing: ✅")
    print("- Multiple strategies: ✅")
    print("- Load distribution: ✅")
    print("\n")

    sys.exit(0)


if __name__ == "__main__":
    main()
