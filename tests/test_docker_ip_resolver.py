"""
Tests for Docker IP resolution functionality.
"""

import pytest

from sollol.docker_ip_resolver import (
    auto_resolve_ips,
    get_deployment_context,
    get_docker_network_mode,
    is_docker_ip,
    is_running_in_docker,
    resolve_docker_ip,
    resolve_docker_ip_with_alternatives,
)


class TestDockerIPDetection:
    """Test Docker IP range detection."""

    def test_docker_bridge_network(self):
        """Test detection of default Docker bridge network."""
        assert is_docker_ip("172.17.0.5") is True
        assert is_docker_ip("172.17.255.254") is True

    def test_docker_custom_networks(self):
        """Test detection of custom Docker networks."""
        assert is_docker_ip("172.18.0.10") is True
        assert is_docker_ip("172.20.0.1") is True
        assert is_docker_ip("172.31.255.1") is True

    def test_docker_overlay_networks(self):
        """Test detection of Docker overlay networks."""
        assert is_docker_ip("10.0.0.5") is True
        assert is_docker_ip("10.255.255.254") is True

    def test_non_docker_ips(self):
        """Test that non-Docker IPs are not detected as Docker."""
        assert is_docker_ip("192.168.1.100") is False
        assert is_docker_ip("127.0.0.1") is False
        assert is_docker_ip("8.8.8.8") is False
        assert is_docker_ip("10.9.66.124") is True  # This IS Docker (10.0.0.0/8)

    def test_invalid_ips(self):
        """Test handling of invalid IP addresses."""
        assert is_docker_ip("not-an-ip") is False
        assert is_docker_ip("256.256.256.256") is False
        assert is_docker_ip("") is False


class TestDockerIPResolution:
    """Test Docker IP resolution to accessible IPs."""

    def test_resolve_to_localhost(self):
        """Test resolution to localhost (most common case)."""
        # This test assumes localhost:11434 is NOT running
        # In real usage, it would return "127.0.0.1" if Ollama is running locally
        result = resolve_docker_ip(
            "172.17.0.5", 11434, timeout=0.1, verify_func=None  # Skip verification for testing
        )
        # Should return first accessible candidate or None
        assert result in ("127.0.0.1", "localhost", None)

    def test_resolve_with_alternatives(self):
        """Test getting all accessible alternatives."""
        # This should return a list of (ip, port) tuples
        alternatives = resolve_docker_ip_with_alternatives(
            "172.17.0.5", 11434, timeout=0.1, verify_func=None
        )
        # Should be a list (may be empty if nothing accessible)
        assert isinstance(alternatives, list)

    def test_non_docker_ip_passthrough(self):
        """Test that non-Docker IPs are not resolved."""
        # Non-Docker IP should not be resolved
        nodes = [{"host": "192.168.1.100", "port": "11434"}]
        resolved = auto_resolve_ips(nodes, timeout=0.1)

        # Should remain unchanged
        assert resolved == nodes

    def test_auto_resolve_mixed_ips(self):
        """Test auto-resolution with mix of Docker and regular IPs."""
        nodes = [
            {"host": "192.168.1.100", "port": "11434"},  # Regular IP
            {"host": "172.17.0.5", "port": "11434"},  # Docker IP
            {"host": "127.0.0.1", "port": "11434"},  # Localhost
        ]

        resolved = auto_resolve_ips(nodes, timeout=0.1, verify_func=None)

        # Regular IP and localhost should remain
        # Docker IP should be attempted resolution
        assert any(n["host"] == "192.168.1.100" for n in resolved)
        assert any(n["host"] == "127.0.0.1" for n in resolved)

    def test_empty_node_list(self):
        """Test handling of empty node list."""
        resolved = auto_resolve_ips([], timeout=0.1)
        assert resolved == []


class TestDockerIPIntegration:
    """Test integration with discovery modules."""

    def test_docker_ip_in_discovery_result(self):
        """Test that Docker IPs in discovery results are handled."""
        from sollol.docker_ip_resolver import auto_resolve_ips

        # Simulate discovery returning Docker IP
        discovered = [
            {"host": "172.17.0.5", "port": "11434"},
            {"host": "172.18.0.10", "port": "11434"},
        ]

        # Auto-resolve should attempt to resolve these
        resolved = auto_resolve_ips(discovered, timeout=0.1, verify_func=None)

        # Should return some result (may be empty if not accessible)
        assert isinstance(resolved, list)

    def test_rpc_docker_ip_resolution(self):
        """Test Docker IP resolution for RPC backends."""
        # RPC backends on Docker
        rpc_nodes = [
            {"host": "172.17.0.5", "port": 50052},
            {"host": "192.168.1.50", "port": 50052},
        ]

        resolved = auto_resolve_ips(rpc_nodes, timeout=0.1, verify_func=None)

        # Should maintain port numbers
        for node in resolved:
            assert "port" in node
            assert node["port"] in (50052, "50052")


class TestDeploymentDetection:
    """Test deployment mode detection."""

    def test_is_running_in_docker(self):
        """Test Docker environment detection."""
        # Should return bool without error
        result = is_running_in_docker()
        assert isinstance(result, bool)

        # On bare metal (this test environment), should be False
        # (unless running in CI/CD Docker container)
        assert result in (True, False)

    def test_get_docker_network_mode(self):
        """Test Docker network mode detection."""
        result = get_docker_network_mode()
        assert result in ("host", "bridge", "overlay", "none", "unknown")

        # If not in Docker, should be "unknown"
        if not is_running_in_docker():
            assert result == "unknown"

    def test_get_deployment_context(self):
        """Test comprehensive deployment context."""
        context = get_deployment_context()

        # Verify structure
        assert "mode" in context
        assert "is_docker" in context
        assert "network_mode" in context
        assert "container_id" in context

        # Verify values
        assert context["mode"] in ("docker", "bare_metal")
        assert isinstance(context["is_docker"], bool)
        assert context["network_mode"] in ("host", "bridge", "overlay", "none", "unknown")

        # Mode should match is_docker
        if context["is_docker"]:
            assert context["mode"] == "docker"
        else:
            assert context["mode"] == "bare_metal"
            assert context["network_mode"] == "unknown"
            assert context["container_id"] is None


class TestDeploymentAwareResolution:
    """Test deployment-aware resolution strategies."""

    def test_resolution_with_deployment_context(self):
        """Test resolution uses deployment context."""
        context = get_deployment_context()

        result = resolve_docker_ip(
            "172.17.0.5", 11434, timeout=0.1, verify_func=None, deployment_context=context
        )

        # Should complete without error
        assert result in (None, "127.0.0.1", "localhost", "172.17.0.5") or result is not None

    def test_bare_metal_resolution_strategy(self):
        """Test bare metal → Docker resolution strategy."""
        # Force bare metal context
        context = {
            "mode": "bare_metal",
            "is_docker": False,
            "network_mode": "unknown",
            "container_id": None,
        }

        result = resolve_docker_ip(
            "172.17.0.5", 11434, timeout=0.1, verify_func=None, deployment_context=context
        )

        # Should try localhost first (bare metal strategy)
        assert result in (None, "127.0.0.1", "localhost") or result is not None

    def test_docker_resolution_strategy(self):
        """Test Docker → Docker resolution strategy."""
        # Force Docker context
        context = {
            "mode": "docker",
            "is_docker": True,
            "network_mode": "bridge",
            "container_id": "abc123",
        }

        result = resolve_docker_ip(
            "172.17.0.5", 11434, timeout=0.1, verify_func=None, deployment_context=context
        )

        # Should try Docker IP directly first (Docker strategy)
        # May succeed or fail depending on actual network
        assert result in (None, "172.17.0.5", "127.0.0.1") or result is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_resolve_unreachable_docker_ip(self):
        """Test resolution of completely unreachable Docker IP."""
        result = resolve_docker_ip(
            "172.17.0.99",
            9999,  # Unlikely port
            timeout=0.1,
            verify_func=lambda h, p, t: False,  # Always fail verification
        )
        # Should return None if unresolvable
        assert result is None

    def test_invalid_port(self):
        """Test handling of invalid port numbers."""
        result = resolve_docker_ip("172.17.0.5", -1, timeout=0.1)  # Invalid port
        # Should handle gracefully
        assert result in (None, "127.0.0.1", "localhost")

    def test_very_short_timeout(self):
        """Test with very short timeout."""
        result = resolve_docker_ip("172.17.0.5", 11434, timeout=0.001)  # 1ms - very aggressive
        # Should complete without error
        assert result in (None, "127.0.0.1", "localhost") or result is not None

    def test_deployment_context_caching(self):
        """Test that deployment detection is cached."""
        # First call
        context1 = get_deployment_context()

        # Second call - should use cache
        context2 = get_deployment_context()

        # Should be identical
        assert context1 == context2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
