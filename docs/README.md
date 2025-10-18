# SOLLOL Documentation

Complete documentation for SOLLOL - Super Ollama Load Balancer.

## Quick Links

- [Installation Guide](../INSTALLATION.md) - Complete setup for bare-metal deployment
- [Configuration](../CONFIGURATION.md) - All configuration options
- [Quick Start](../QUICK_START.md) - Get up and running fast
- [Architecture](../ARCHITECTURE.md) - System architecture overview

## Documentation Structure

### Setup Guides (`setup/`)

Step-by-step guides for setting up SOLLOL components:

- [Ray Cluster Setup](setup/ray-cluster.md) - Configure Ray for remote coordinator execution
- [Redis Setup](setup/redis.md) - Redis configuration for distributed state
- [GPU Monitoring](setup/gpu-monitoring-setup.md) - GPU metrics and monitoring
- [GPU Monitoring Guide](setup/gpu-monitoring-guide.md) - Detailed GPU monitoring guide
- [Grafana Setup](setup/grafana.md) - Observability dashboards
- [Remote Access](setup/remote-access.md) - Remote access configuration
- [Docker Setup](setup/docker.md) - Docker deployment (not fully battle-tested)
- [GPU Reporter Deployment](setup/deploy-gpu-reporter.md) - Deploy GPU reporting service

### Features (`features/`)

Detailed feature documentation:

- [Backends](features/backends.md) - Backend types (Ollama, RPC)
- [Batch Processing](features/batch-processing.md) - Batch API and processing
- [Routing Strategies](features/routing.md) - Intelligent routing algorithms
- [Deployment-Aware Resolution](features/deployment-aware-resolution.md) - Dynamic deployment detection
- [Docker Networking](features/docker-networking.md) - Docker IP resolution
- [Dashboard](features/dashboard.md) - Real-time observability dashboard

### Architecture (`architecture/`)

Deep dives into system architecture:

- [Remote Coordinator](architecture/remote-coordinator.md) - Intelligent coordinator placement
- [Multi-App Coordination](architecture/multi-app.md) - Running multiple SOLLOL instances

### Integration (`integration/`)

Integration guides and code examples:

- [Basic Integration](integration/basic.md) - Simple integration examples
- [Advanced Integration](integration/advanced.md) - Complex integration patterns
- [Code Walkthrough](integration/code-walkthrough.md) - Detailed code examples

### Benchmarks (`benchmarks/`)

Performance testing and results:

- [Benchmark Results](benchmarks/results.md) - Performance benchmarks
- [How to Benchmark](benchmarks/how-to-benchmark.md) - Run your own benchmarks
- [Test Results](benchmarks/test-results.md) - Test suite results
- [Distributed Testing](benchmarks/distributed-testing.md) - Distributed inference testing

### Troubleshooting (`troubleshooting/`)

Problem solving and known issues:

- [Known Issues](troubleshooting/known-issues.md) - Current known issues
- [Limitations](troubleshooting/limitations.md) - System limitations
- [RPC Fixes](troubleshooting/rpc-fixes.md) - Common RPC backend fixes
- [Coordinator Investigation](troubleshooting/coordinator-investigation.md) - Coordinator debugging

### External (`external/`)

External posts and community content:

- **Ollama Posts** (`external/ollama-posts/`) - Discord and community posts
- **GitHub Issues** (`external/github-issues/`) - Issue templates and examples

### Archive (`archive/`)

Historical and completed documentation:

- Phase 1/2 implementation docs
- Publishing announcements
- Legacy guides superseded by current docs

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.

## Need Help?

1. Check [Known Issues](troubleshooting/known-issues.md)
2. Review [Limitations](troubleshooting/limitations.md)
3. See [Installation Guide](../INSTALLATION.md) for setup help
4. Report issues: https://github.com/BenevolentJoker-JohnL/SOLLOL/issues
