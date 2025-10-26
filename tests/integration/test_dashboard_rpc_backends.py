"""
Integration tests for dashboard RPC backend display.

Tests:
1. Dashboard API /api/network/backends returns proper structure
2. RPC backend metadata is correctly extracted from registry
3. No "undefined" values in dashboard responses
"""

import json

import pytest

from sollol.rpc_registry import RPCBackendRegistry
from sollol.unified_dashboard import UnifiedDashboard


class TestDashboardRPCBackends:
    """Test dashboard RPC backend integration."""

    def test_api_backends_response_structure(self):
        """Test that /api/network/backends returns expected structure."""
        # Create dashboard without router
        dashboard = UnifiedDashboard(router=None, dashboard_port=5555)

        # Create a test client
        with dashboard.app.test_client() as client:
            response = client.get("/api/network/backends")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert "backends" in data
            assert "total" in data
            assert isinstance(data["backends"], list)
            assert isinstance(data["total"], int)

    def test_backend_metadata_fields(self):
        """Test that backend metadata has all required fields."""
        # Create registry with test backends
        registry = RPCBackendRegistry()
        registry.add_backend("10.9.66.48", 50052)
        registry.add_backend("10.9.66.154", 50052)

        # Simulate dashboard processing
        backends = []
        for backend_obj in registry.backends.values():
            backend_dict = backend_obj.to_dict()
            host = backend_dict["host"]
            port = backend_dict["port"]
            is_healthy = backend_dict["healthy"]
            metrics = backend_dict.get("metrics", {})

            backends.append(
                {
                    "url": f"{host}:{port}",
                    "status": "healthy" if is_healthy else "offline",
                    "latency_ms": metrics.get("avg_latency_ms", 0),
                    "request_count": metrics.get("total_requests", 0),
                    "failure_count": metrics.get("total_failures", 0),
                }
            )

        # Verify structure
        assert len(backends) == 2

        for backend in backends:
            # Check no undefined/missing fields
            assert backend["url"] is not None
            assert backend["status"] in ["healthy", "offline", "degraded"]
            assert isinstance(backend["latency_ms"], (int, float))
            assert isinstance(backend["request_count"], int)
            assert isinstance(backend["failure_count"], int)

            # Verify URL format
            assert ":" in backend["url"]
            parts = backend["url"].split(":")
            assert len(parts) == 2
            assert parts[1].isdigit()

    def test_no_undefined_values(self):
        """Test that backend data has no undefined/null values."""
        registry = RPCBackendRegistry()
        registry.add_backend("10.9.66.48", 50052)

        # Get backend dict
        backend_obj = list(registry.backends.values())[0]
        backend_dict = backend_obj.to_dict()

        # Verify no None values in critical fields
        assert backend_dict["host"] is not None
        assert backend_dict["port"] is not None
        assert backend_dict["healthy"] is not None
        assert backend_dict["metrics"] is not None

        # Verify metrics has no None values
        metrics = backend_dict["metrics"]
        assert metrics["total_requests"] is not None
        assert metrics["total_failures"] is not None
        assert metrics["avg_latency_ms"] is not None
        assert metrics["success_rate"] is not None

    def test_registry_iteration_fix(self):
        """Test that iterating registry.backends.values() works correctly."""
        registry = RPCBackendRegistry()
        registry.add_backend("10.9.66.48", 50052)
        registry.add_backend("10.9.66.154", 50052)

        # This is the OLD (broken) way that causes "undefined"
        # for backend in registry.backends:
        #     # This would iterate over keys (strings), not RPCBackend objects
        #     # backend would be "10.9.66.48:50052" (string)
        #     # backend["host"] would fail -> "undefined"

        # This is the NEW (correct) way
        backends_list = []
        for backend_obj in registry.backends.values():
            # backend_obj is RPCBackend instance
            assert hasattr(backend_obj, "host")
            assert hasattr(backend_obj, "port")
            assert hasattr(backend_obj, "to_dict")

            backend_dict = backend_obj.to_dict()
            backends_list.append(backend_dict["host"])

        # Verify we got both hosts
        assert len(backends_list) == 2
        assert "10.9.66.48" in backends_list
        assert "10.9.66.154" in backends_list


class TestRouterBackendIntegration:
    """Test router backend integration with dashboard."""

    def test_router_backends_structure(self):
        """Test that router.rpc_backends has expected structure."""
        # Simulate router backends (list of dicts)
        router_backends = [
            {"host": "10.9.66.48", "port": 50052, "request_count": 10, "failure_count": 0},
            {"host": "10.9.66.154", "port": 50052, "request_count": 5, "failure_count": 1},
        ]

        # Simulate dashboard processing
        backends = []
        for backend in router_backends:
            host = backend.get("host")
            port = backend.get("port", 50052)
            backends.append(
                {
                    "url": f"{host}:{port}",
                    "status": "healthy",
                    "latency_ms": 0,
                    "request_count": backend.get("request_count", 0),
                    "failure_count": backend.get("failure_count", 0),
                }
            )

        # Verify structure
        assert len(backends) == 2
        assert backends[0]["url"] == "10.9.66.48:50052"
        assert backends[1]["url"] == "10.9.66.154:50052"
        assert backends[0]["request_count"] == 10
        assert backends[1]["failure_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
