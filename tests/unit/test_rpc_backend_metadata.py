"""
Unit tests for RPC backend discovery and metadata.

Tests:
1. RPC registry properly stores backend metadata
2. Discovery returns correct data structure
3. Dashboard API returns proper backend information
"""

import pytest

from sollol.rpc_registry import RPCBackend, RPCBackendRegistry, RPCMetrics


class TestRPCBackendMetadata:
    """Test RPC backend metadata handling."""

    def test_backend_to_dict_structure(self):
        """Test that RPCBackend.to_dict() returns expected structure."""
        backend = RPCBackend(host="10.9.66.48", port=50052)
        result = backend.to_dict()

        # Verify required fields
        assert "host" in result
        assert "port" in result
        assert "healthy" in result
        assert "metrics" in result

        # Verify types
        assert isinstance(result["host"], str)
        assert isinstance(result["port"], int)
        assert isinstance(result["healthy"], bool)
        assert isinstance(result["metrics"], dict)

        # Verify metrics structure
        metrics = result["metrics"]
        assert "total_requests" in metrics
        assert "total_failures" in metrics
        assert "avg_latency_ms" in metrics
        assert "success_rate" in metrics
        assert "last_check" in metrics

    def test_registry_backends_iteration(self):
        """Test that registry.backends can be properly iterated."""
        registry = RPCBackendRegistry()
        registry.add_backend("10.9.66.48", 50052)
        registry.add_backend("10.9.66.154", 50052)

        # registry.backends is Dict[str, RPCBackend]
        assert isinstance(registry.backends, dict)
        assert len(registry.backends) == 2

        # Test iteration over values()
        backend_dicts = [b.to_dict() for b in registry.backends.values()]
        assert len(backend_dicts) == 2

        # Verify each dict has proper structure
        for backend_dict in backend_dicts:
            assert "host" in backend_dict
            assert "port" in backend_dict
            assert backend_dict["host"] in ["10.9.66.48", "10.9.66.154"]
            assert backend_dict["port"] == 50052

    def test_backend_metrics_tracking(self):
        """Test that backend metrics are properly tracked."""
        backend = RPCBackend(host="10.9.66.48", port=50052)

        # Get initial metrics
        backend_dict = backend.to_dict()
        metrics = backend_dict["metrics"]

        # Verify initial values
        assert metrics["total_requests"] >= 0
        assert metrics["total_failures"] >= 0
        assert metrics["avg_latency_ms"] >= 0
        assert 0.0 <= metrics["success_rate"] <= 1.0

    def test_registry_get_stats(self):
        """Test registry.get_stats() returns proper structure."""
        registry = RPCBackendRegistry()
        registry.add_backend("10.9.66.48", 50052)
        registry.add_backend("10.9.66.154", 50052)

        stats = registry.get_stats()

        # Verify structure
        assert "total_backends" in stats
        assert "healthy_backends" in stats
        assert "backends" in stats

        # Verify values
        assert stats["total_backends"] == 2
        assert isinstance(stats["backends"], list)
        assert len(stats["backends"]) == 2

        # Each backend should be a dict with proper structure
        for backend in stats["backends"]:
            assert "host" in backend
            assert "port" in backend
            assert "metrics" in backend


class TestDiscoveryMetadata:
    """Test RPC discovery returns correct metadata."""

    def test_discovery_return_structure(self):
        """Test that discover_rpc_backends returns expected structure."""
        from sollol.rpc_discovery import discover_rpc_backends

        # Discovery returns list of dicts with host and port
        backends = discover_rpc_backends(cidr="10.9.66.0/24", timeout=0.1)

        # Should be a list
        assert isinstance(backends, list)

        # Each item should have host and port
        for backend in backends:
            assert isinstance(backend, dict)
            assert "host" in backend
            assert "port" in backend


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
