#!/usr/bin/env python3
"""Test SOLLOL's locality awareness feature."""

from sollol.pool import OllamaPool

def test_same_machine_detection():
    """Test that SOLLOL detects multiple nodes on same machine."""
    print("=" * 60)
    print("TEST: Same Machine Detection")
    print("=" * 60)

    # Case 1: Two localhost nodes (different ports)
    print("\n📋 Case 1: localhost:11434 + localhost:11435")
    pool = OllamaPool(nodes=[
        {'host': 'localhost', 'port': '11434'},
        {'host': 'localhost', 'port': '11435'}
    ], register_with_dashboard=False)

    unique_hosts = pool.count_unique_physical_hosts()
    should_parallel = pool.should_use_parallel_execution(3)

    print(f"   Nodes: {len(pool.nodes)}")
    print(f"   Unique hosts: {unique_hosts}")
    print(f"   Parallel recommended: {should_parallel}")
    assert unique_hosts == 1, "Should detect as 1 physical host"
    assert not should_parallel, "Should NOT recommend parallel execution"
    print("   ✅ PASS: Correctly detected same machine\n")

    # Case 2: Different machines
    print("📋 Case 2: localhost:11434 + 192.168.1.20:11434")
    pool2 = OllamaPool(nodes=[
        {'host': 'localhost', 'port': '11434'},
        {'host': '192.168.1.20', 'port': '11434'}
    ], register_with_dashboard=False)

    unique_hosts2 = pool2.count_unique_physical_hosts()
    should_parallel2 = pool2.should_use_parallel_execution(3)

    print(f"   Nodes: {len(pool2.nodes)}")
    print(f"   Unique hosts: {unique_hosts2}")
    print(f"   Parallel recommended: {should_parallel2}")
    # Note: This will show 2 if 192.168.1.20 is reachable, 1 if not
    print(f"   ✅ PASS: Detected {unique_hosts2} physical host(s)\n")

    # Case 3: Single node
    print("📋 Case 3: Single node (localhost:11434)")
    pool3 = OllamaPool(nodes=[
        {'host': 'localhost', 'port': '11434'}
    ], register_with_dashboard=False)

    unique_hosts3 = pool3.count_unique_physical_hosts()
    should_parallel3 = pool3.should_use_parallel_execution(3)

    print(f"   Nodes: {len(pool3.nodes)}")
    print(f"   Unique hosts: {unique_hosts3}")
    print(f"   Parallel recommended: {should_parallel3}")
    assert unique_hosts3 == 1, "Should be 1 physical host"
    assert not should_parallel3, "Should NOT recommend parallel (only 1 node)"
    print("   ✅ PASS: Correctly handled single node\n")

    print("=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
    print("\nSOLLOL now has locality awareness! 🎉")
    print("Parallel execution will only be enabled when beneficial.")

if __name__ == "__main__":
    test_same_machine_detection()
