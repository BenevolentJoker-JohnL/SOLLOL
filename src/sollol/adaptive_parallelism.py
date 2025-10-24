"""
Adaptive Parallelism Strategy for SOLLOL

Automatically chooses between sequential and parallel processing based on
cluster performance characteristics, preventing wasteful parallelization
when one GPU is dominant.

Key Decision Factors:
1. GPU Performance Gap - If one node is 10x faster, sequential is better
2. Node Count - More similar nodes = better parallelism
3. Network Latency - High latency favors fewer nodes
4. Batch Size - Small batches favor sequential (less overhead)
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AdaptiveParallelismStrategy:
    """
    Intelligently decides whether to use parallel or sequential processing
    based on real-time performance metrics.

    Works directly with OllamaPool's nodes and stats['node_performance'] structure.
    """

    def __init__(self, pool=None):
        """
        Initialize adaptive parallelism strategy.

        Args:
            pool: OllamaPool instance (optional, can be set later)
        """
        self.pool = pool
        self.performance_history = []  # Track decisions and results

    def set_pool(self, pool):
        """Set the OllamaPool after initialization."""
        self.pool = pool

    def should_parallelize(self, batch_size: int, model_name: Optional[str] = None) -> Tuple[bool, Dict]:
        """
        Decide whether to parallelize based on cluster state.

        Args:
            batch_size: Number of items to process
            model_name: Model being used (for GPU-specific logic)

        Returns:
            (should_parallelize: bool, reasoning: Dict)
        """
        if not self.pool:
            logger.warning("No OllamaPool set, defaulting to parallel")
            return True, {"reason": "no_pool", "detail": "No pool configured, using parallel by default"}

        nodes = self.pool.nodes
        node_performance = self.pool.stats.get("node_performance", {})

        # CASE 0: Single node - always parallel (will use multiple workers on same node)
        if len(nodes) <= 1:
            return True, {
                "reason": "single_node",
                "detail": "Single node - using parallel workers for better performance",
                "available_nodes": len(nodes),
            }

        # Calculate performance metrics
        node_speeds = []

        for i, node in enumerate(nodes, 1):
            node_key = f"{node['host']}:{node['port']}"
            stats = node_performance.get(node_key, {})

            # Calculate speed score based on stats
            total_requests = stats.get("total_requests", 0)
            failed_requests = stats.get("failed_requests", 0)
            successful_requests = total_requests - failed_requests
            avg_latency_ms = stats.get("latency_ms", 0.0)

            if successful_requests > 0 and avg_latency_ms > 0:
                # Use average latency (lower = faster)
                # Convert ms to seconds for speed score
                avg_latency = avg_latency_ms / 1000.0
                speed_score = 1.0 / max(avg_latency, 0.01)  # Higher = faster
            else:
                # Estimate based on availability
                is_available = stats.get("available", True)
                speed_score = 50 if is_available else 1  # Assume available = reasonably fast

            node_speeds.append({
                "node": node_key,
                "speed_score": speed_score,
                "total_requests": total_requests,
                "is_available": stats.get("available", True)
            })

        # Sort by speed (fastest first)
        node_speeds.sort(key=lambda x: x["speed_score"], reverse=True)

        fastest_node = node_speeds[0]
        slowest_node = node_speeds[-1]

        # Calculate speed ratio
        speed_ratio = fastest_node["speed_score"] / max(slowest_node["speed_score"], 1)

        # Decision logic
        reasoning = {
            "available_nodes": len(nodes),
            "batch_size": batch_size,
            "fastest_node": fastest_node["node"],
            "fastest_speed": fastest_node["speed_score"],
            "speed_ratio": speed_ratio,
        }

        # CASE 1: One GPU node is 5x+ faster than others
        if speed_ratio >= 5.0 and fastest_node["total_requests"] >= 5:
            # Sequential on fastest node is better (only if we have enough data)
            return False, {
                **reasoning,
                "reason": "dominant_node",
                "detail": f"Fastest node is {speed_ratio:.1f}x faster - sequential wins",
                "recommended_node": fastest_node["node"],
            }

        # CASE 2: Small batch (dynamic threshold based on node count)
        # Minimum chunks per node: 5 (so 2 nodes = 10 threshold, 3 nodes = 15, etc.)
        min_chunks_per_node = 5
        dynamic_threshold = len(nodes) * min_chunks_per_node

        if batch_size < dynamic_threshold:
            # Overhead of parallelism not worth it
            return False, {
                **reasoning,
                "reason": "small_batch",
                "detail": f"Batch size {batch_size} too small for parallel overhead (need {dynamic_threshold}+ for {len(nodes)} nodes)",
                "recommended_node": fastest_node["node"],
            }

        # CASE 3: Multiple similar-speed nodes
        if speed_ratio < 3.0:
            # Nodes are similar speed, parallelize!
            return True, {
                **reasoning,
                "reason": "balanced_cluster",
                "detail": f"Speed ratio {speed_ratio:.1f}x - parallel is efficient",
                "parallel_workers": len(nodes) * 2,
            }

        # CASE 4: Medium speed difference (3-5x)
        # Use fastest 2-3 nodes in parallel
        if speed_ratio < 5.0 and len(nodes) >= 3:
            return True, {
                **reasoning,
                "reason": "hybrid_parallel",
                "detail": f"Using top {min(3, len(nodes))} nodes in parallel",
                "parallel_workers": min(3, len(nodes)) * 2,
            }

        # Default: Parallel (safer default for SOLLOL)
        return True, {
            **reasoning,
            "reason": "default_parallel",
            "detail": "Defaulting to parallel for better performance",
        }

    def get_optimal_workers(self, batch_size: int) -> int:
        """
        Calculate optimal number of parallel workers.

        Args:
            batch_size: Number of items to process

        Returns:
            Number of workers (minimum 1)
        """
        if not self.pool:
            return min(4, batch_size)  # Default: 4 workers

        nodes = self.pool.nodes

        # IMPORTANT: Use conservative worker counts to avoid overwhelming Ollama servers
        # Ollama embedding endpoints can handle multiple concurrent requests efficiently
        # Multiple workers per node allows better saturation and throughput

        # For small node counts (1-3), use 2x workers to saturate endpoints
        if len(nodes) <= 3:
            base_workers = len(nodes) * 2  # 2 workers per node for good saturation
        else:
            base_workers = len(nodes) * 2  # Can use more for larger clusters

        # Adjust for batch size
        if batch_size < 50:
            workers = min(base_workers, batch_size)
        elif batch_size < 200:
            workers = base_workers
        else:
            # Large batch, can use slightly more workers
            workers = min(base_workers + 2, batch_size)  # Conservative increase

        # Minimum 1 worker, maximum 6 (even for large batches)
        return max(1, min(workers, 6))

    def get_recommended_node(self, model_name: Optional[str] = None) -> Optional[str]:
        """
        Get the recommended node for sequential execution.

        Prefers GPU nodes but falls back to CPU if needed.
        Important: GPU nodes might be set to CPU generation mode.

        Args:
            model_name: Model name (for GPU-specific routing)

        Returns:
            Node URL (host:port) or None
        """
        if not self.pool:
            return None

        nodes = self.pool.nodes
        node_performance = self.pool.stats.get("node_performance", {})

        if not nodes:
            return None

        # Calculate node performance scores
        node_info = []
        for node in nodes:
            node_key = f"{node['host']}:{node['port']}"
            stats = node_performance.get(node_key, {})

            # Get performance metrics
            total_requests = stats.get("total_requests", 0)
            failed_requests = stats.get("failed_requests", 0)
            successful_requests = total_requests - failed_requests
            avg_latency_ms = stats.get("latency_ms", 0.0)

            if successful_requests > 0 and avg_latency_ms > 0:
                # Use average latency in seconds
                avg_latency = avg_latency_ms / 1000.0
            else:
                avg_latency = 999.0  # High default for untested nodes

            # Check if node is available
            is_available = stats.get("available", True)

            # Try to detect GPU capability from stats or node info
            has_gpu = node.get("has_gpu", False)

            node_info.append({
                "key": node_key,
                "node": node,
                "avg_latency": avg_latency,
                "has_gpu": has_gpu,
                "is_available": is_available,
                "successful_requests": successful_requests
            })

        # Filter to available nodes only
        available_nodes = [n for n in node_info if n["is_available"]]

        if not available_nodes:
            # No available nodes, return fastest regardless of availability
            logger.warning("No available nodes, selecting fastest node")
            fastest = min(node_info, key=lambda n: n["avg_latency"])
            return fastest["key"]

        # Prefer GPU nodes for most models
        gpu_nodes = [n for n in available_nodes if n["has_gpu"]]

        if gpu_nodes:
            # Return fastest GPU node (lowest avg latency)
            fastest_gpu = min(gpu_nodes, key=lambda n: n["avg_latency"])
            logger.debug(f"Selected GPU node: {fastest_gpu['key']} (latency: {fastest_gpu['avg_latency']:.3f}s)")
            return fastest_gpu["key"]

        # Fallback to fastest CPU node
        fastest_cpu = min(available_nodes, key=lambda n: n["avg_latency"])
        logger.debug(f"Selected CPU node: {fastest_cpu['key']} (latency: {fastest_cpu['avg_latency']:.3f}s)")
        return fastest_cpu["key"]

    def print_parallelism_report(self, batch_size: int):
        """
        Print parallelism analysis report.

        Args:
            batch_size: Batch size to analyze
        """
        should_parallel, reasoning = self.should_parallelize(batch_size)

        print("\n" + "=" * 70)
        print("🔀 SOLLOL ADAPTIVE PARALLELISM REPORT")
        print("=" * 70)

        print(f"\nBatch Size: {batch_size}")
        print(f"Available Nodes: {reasoning.get('available_nodes', 0)}")

        if reasoning.get("fastest_node"):
            print(f"Fastest Node: {reasoning['fastest_node']}")
            print(f"Speed Ratio: {reasoning.get('speed_ratio', 0):.1f}x")

        print(f"\n{'✅ PARALLEL' if should_parallel else '⏭️  SEQUENTIAL'} Processing Recommended")
        print(f"Reason: {reasoning['reason']}")
        print(f"Detail: {reasoning['detail']}")

        if should_parallel:
            workers = reasoning.get("parallel_workers", 1)
            print(f"Recommended Workers: {workers}")
        else:
            recommended = reasoning.get("recommended_node", "unknown")
            print(f"Recommended Node: {recommended}")

        print("=" * 70 + "\n")

    def record_decision(self, decision: Dict, actual_time: float):
        """
        Record a parallelism decision and its outcome for learning.

        Args:
            decision: Decision dict from should_parallelize()
            actual_time: Actual execution time in seconds
        """
        self.performance_history.append(
            {"decision": decision, "actual_time": actual_time, "timestamp": time.time()}
        )

        # Keep only last 100 decisions
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]


# Example integration with SOLLOL's OllamaPool
def integrate_with_pool(pool, strategy: AdaptiveParallelismStrategy):
    """
    Integrate adaptive parallelism with OllamaPool.

    Args:
        pool: OllamaPool instance
        strategy: AdaptiveParallelismStrategy instance

    Example:
        strategy = AdaptiveParallelismStrategy(node_registry)
        integrate_with_pool(pool, strategy)
    """
    # Set registry from pool
    if hasattr(pool, "registry"):
        strategy.set_registry(pool.registry)

    logger.info("✅ Adaptive parallelism integrated with OllamaPool")


def print_parallelism_report(pool):
    """
    Print adaptive parallelism analysis report.

    Args:
        pool: OllamaPool instance with parallelism strategy
    """
    if not hasattr(pool, '_parallelism_strategy'):
        print("⚠️  Adaptive parallelism not enabled on this pool")
        return

    strategy = pool._parallelism_strategy
    print("\n" + "=" * 60)
    print("🔀 ADAPTIVE PARALLELISM ANALYSIS")
    print("=" * 60)

    # Get metrics from strategy
    if hasattr(strategy, 'metrics') and strategy.metrics:
        metrics = strategy.metrics
        print(f"\n📊 Decision Metrics:")
        print(f"   Sequential calls: {metrics.get('sequential_count', 0)}")
        print(f"   Parallel calls:   {metrics.get('parallel_count', 0)}")
        print(f"   Avg latency (seq): {metrics.get('avg_sequential_latency', 0):.2f}ms")
        print(f"   Avg latency (par): {metrics.get('avg_parallel_latency', 0):.2f}ms")

    # Show current thresholds
    print(f"\n⚙️  Current Thresholds:")
    print(f"   Latency threshold: {strategy.latency_threshold_ms}ms")
    print(f"   Queue threshold:   {strategy.queue_depth_threshold}")
    print(f"   Min parallel:      {strategy.min_parallel_requests}")

    # Show recent decisions
    if hasattr(strategy, 'recent_decisions'):
        print(f"\n📝 Recent Decisions:")
        for decision in strategy.recent_decisions[-5:]:
            print(f"   {decision['timestamp']}: {decision['mode']} - {decision['reason']}")

    print("=" * 60 + "\n")
