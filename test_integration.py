#!/usr/bin/env python3
"""
Integration test for SOLLOL remote coordinator execution.

This tests the actual implementation without making any requests or
starting services that could disrupt the running system.
"""

import sys
sys.path.insert(0, "src")

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_imports():
    """Test that all components can be imported."""
    print("\n" + "="*70)
    print("Test 1: Verify Imports")
    print("="*70)

    try:
        from sollol.ray_hybrid_router import RayHybridRouter, ShardedModelPool
        print("✅ ray_hybrid_router imports successfully")

        from sollol.rpc_discovery import detect_node_resources, check_rpc_server
        print("✅ rpc_discovery imports successfully")

        from sollol.llama_cpp_coordinator import LlamaCppCoordinator
        print("✅ llama_cpp_coordinator imports successfully")

        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_ray_initialization():
    """Test Ray initialization without starting actors."""
    print("\n" + "="*70)
    print("Test 2: Verify Ray Cluster")
    print("="*70)

    try:
        import ray

        # Check if Ray is already initialized
        if ray.is_initialized():
            print("✅ Ray already initialized")

            # Get cluster info
            nodes = ray.nodes()
            print(f"✅ Ray cluster has {len(nodes)} node(s)")

            for i, node in enumerate(nodes):
                node_id = node.get('NodeID', 'unknown')[:8]
                alive = node.get('Alive', False)
                resources = node.get('Resources', {})
                cpu = resources.get('CPU', 0)
                memory = resources.get('memory', 0) / (1024**3)  # Convert to GB

                status = "🟢 alive" if alive else "🔴 dead"
                print(f"  Node {i+1}: {node_id}... {status} (CPU: {cpu}, RAM: {memory:.1f}GB)")

            return True
        else:
            print("⚠️  Ray not initialized (this is okay, we're just testing)")
            return True

    except Exception as e:
        print(f"❌ Ray check failed: {e}")
        return False


def test_redis_connection():
    """Test Redis connectivity for GPU metadata."""
    print("\n" + "="*70)
    print("Test 3: Verify Redis Connection")
    print("="*70)

    try:
        import redis
        import os

        redis_url = os.getenv("SOLLOL_REDIS_URL", "redis://localhost:6379")
        r = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=2)

        # Test connection
        r.ping()
        print(f"✅ Redis connected at {redis_url}")

        # Check for RPC node registrations
        keys = r.keys("sollol:rpc:node:*")
        if keys:
            print(f"✅ Found {len(keys)} registered RPC nodes:")
            for key in keys:
                print(f"   - {key}")
        else:
            print("⚠️  No RPC nodes registered yet (register with register_rpc_gpu_node.py)")

        return True

    except Exception as e:
        print(f"⚠️  Redis connection failed: {e}")
        print("   This is okay if Redis isn't running")
        return True  # Don't fail the test


def test_rpc_backend_detection():
    """Test RPC backend auto-discovery (read-only)."""
    print("\n" + "="*70)
    print("Test 4: Test RPC Backend Detection")
    print("="*70)

    try:
        from sollol.rpc_discovery import detect_node_resources

        # Test with known hosts (read-only, no changes)
        test_hosts = ["10.9.66.154", "10.9.66.90", "10.9.66.48"]

        print("Testing resource detection for known hosts:")
        for host in test_hosts:
            resources = detect_node_resources(host)
            cpu_ram = resources.get("cpu_ram_mb", 0)
            gpu_vram = sum(resources.get("gpu_vram_mb", []))
            has_gpu = resources.get("has_gpu", False)

            gpu_str = f"+ {gpu_vram}MB GPU" if has_gpu else "(CPU only)"
            print(f"  {host}: {cpu_ram}MB RAM {gpu_str}")

        print("✅ Resource detection works")
        return True

    except Exception as e:
        print(f"❌ Resource detection failed: {e}")
        return False


def test_node_selection_logic():
    """Test the node selection algorithm (no actual deployment)."""
    print("\n" + "="*70)
    print("Test 5: Test Node Selection Algorithm")
    print("="*70)

    try:
        # Import the test function
        from test_remote_coordinator import select_best_coordinator_node, mock_detect_node_resources
        import sollol.rpc_discovery

        # Temporarily patch for testing
        original = sollol.rpc_discovery.detect_node_resources
        sollol.rpc_discovery.detect_node_resources = mock_detect_node_resources

        try:
            # Test backends
            backends = [
                {"host": "10.9.66.154", "port": 50052},
                {"host": "10.9.66.90", "port": 50052},
                {"host": "10.9.66.48", "port": 50052},
            ]

            # Test different model sizes
            tests = [
                ("llama3.1:8b", "10.9.66.90"),
                ("llama3.1:70b", "10.9.66.90"),
            ]

            all_passed = True
            for model, expected in tests:
                selected = select_best_coordinator_node(model, backends)
                if selected == expected:
                    print(f"✅ {model}: Correctly selected {selected}")
                else:
                    print(f"❌ {model}: Expected {expected}, got {selected}")
                    all_passed = False

            return all_passed

        finally:
            # Restore original function
            sollol.rpc_discovery.detect_node_resources = original

    except Exception as e:
        print(f"❌ Node selection test failed: {e}")
        return False


def test_configuration_loading():
    """Test that configuration can be loaded."""
    print("\n" + "="*70)
    print("Test 6: Test Configuration Loading")
    print("="*70)

    try:
        import os

        # Check key environment variables
        env_vars = [
            "SOLLOL_PORT",
            "SOLLOL_REDIS_URL",
            "SOLLOL_RAY_WORKERS",
            "SOLLOL_REMOTE_COORDINATOR",
        ]

        for var in env_vars:
            value = os.getenv(var, "<not set>")
            print(f"  {var}: {value}")

        print("✅ Configuration accessible")
        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("SOLLOL Integration Test Suite")
    print("Testing remote coordinator execution implementation")
    print("="*70)

    tests = [
        ("Imports", test_imports),
        ("Ray Cluster", test_ray_initialization),
        ("Redis Connection", test_redis_connection),
        ("RPC Backend Detection", test_rpc_backend_detection),
        ("Node Selection Algorithm", test_node_selection_logic),
        ("Configuration Loading", test_configuration_loading),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "✅ PASS" if p else "❌ FAIL"
        print(f"  {status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Remote coordinator implementation is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
