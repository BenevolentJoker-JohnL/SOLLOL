#!/usr/bin/env python3
"""
SOLLOL Routing Validation Script
Tests the RayHybridRouter routing logic to ensure correct behavior.
"""
import asyncio
import logging
import sys
sys.path.insert(0, '/home/joker/SOLLOL/src')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_routing_logic():
    """Test RayHybridRouter routing decisions"""
    from sollol.ray_hybrid_router import RayHybridRouter

    print("=" * 70)
    print("🧪 SOLLOL Routing Validation Test")
    print("=" * 70)
    print()

    # Test 1: RPC-only mode (no Ollama pool)
    print("📋 Test 1: RPC-only mode (large model → coordinator)")
    print("-" * 70)

    try:
        router = RayHybridRouter(
            ollama_pool=None,
            rpc_backends=[{"host": "10.9.66.45", "port": 50052}],
            coordinator_host="127.0.0.1",
            coordinator_base_port=18080,
            enable_distributed=True,
            auto_discover_rpc=False,
        )

        print(f"   ✅ Router created")
        print(f"      - Has RPC backends: {router.has_rpc_backends}")
        print(f"      - RPC backend count: {len(router.rpc_backends)}")
        print(f"      - Coordinator: {router.coordinator_host}:{router.coordinator_base_port}")
        print(f"      - Ollama pool: {router.ollama_pool is not None}")
        print()

        # Check routing decision
        should_use_rpc = router._should_use_rpc("codellama:13b")
        print(f"   📊 Routing decision for 'codellama:13b':")
        print(f"      - Should use RPC: {should_use_rpc}")
        print(f"      - Will route to: llama.cpp coordinator" if should_use_rpc else "      - Will route to: Ollama pool")
        print()

        # Test request
        print("   🚀 Sending test request...")
        messages = [{"role": "user", "content": "Say hello in exactly 3 words"}]

        response = await router.route_request(
            model="codellama:13b",
            messages=messages,
            max_tokens=10,
        )

        print(f"   ✅ Request successful!")
        content = response['choices'][0]['message']['content']
        print(f"      Response: {content[:100]}")
        print(f"      Tokens: {response['usage']['total_tokens']}")
        print()

    except Exception as e:
        print(f"   ❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        print()

    # Test 2: Model size threshold check
    print("📋 Test 2: Model size threshold validation")
    print("-" * 70)

    try:
        # Small models should not use RPC
        small_models = ["llama3:8b", "phi3:mini", "gemma:2b"]
        print("   Testing small models (should NOT use RPC):")
        for model in small_models:
            should_use = router._should_use_rpc(model)
            status = "❌ WRONG" if should_use else "✅ Correct"
            print(f"      {status} - {model}: use_rpc={should_use}")
        print()

        # Large models should use RPC
        large_models = ["codellama:13b", "llama3:70b", "mixtral:8x7b"]
        print("   Testing large models (SHOULD use RPC):")
        for model in large_models:
            should_use = router._should_use_rpc(model)
            status = "✅ Correct" if should_use else "❌ WRONG"
            print(f"      {status} - {model}: use_rpc={should_use}")
        print()

    except Exception as e:
        print(f"   ❌ Test 2 failed: {e}")
        print()

    # Test 3: Coordinator availability check
    print("📋 Test 3: Coordinator health check")
    print("-" * 70)

    try:
        import httpx
        coordinator_url = f"http://{router.coordinator_host}:{router.coordinator_base_port}/health"

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(coordinator_url)

            if response.status_code == 200:
                print(f"   ✅ Coordinator is healthy")
                print(f"      URL: {coordinator_url}")
                print(f"      Status: {response.status_code}")
                print(f"      Response: {response.json()}")
            else:
                print(f"   ⚠️  Coordinator returned non-200 status: {response.status_code}")
        print()

    except Exception as e:
        print(f"   ❌ Test 3 failed: {e}")
        print()

    print("=" * 70)
    print("✨ Validation Complete")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("   • Large models (>16GB) route to llama.cpp coordinator")
    print("   • Small models (<16GB) route to Ollama pool (if available)")
    print("   • RPC sharding uses direct HTTP, not Ray actors")
    print("   • Coordinator must be running on configured host:port")
    print()

if __name__ == "__main__":
    asyncio.run(test_routing_logic())
