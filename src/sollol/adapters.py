"""
Adapter classes to wrap SOLLOL's function-based modules into class-based interfaces
for integration with SynapticLlamas.
"""

from datetime import datetime
from typing import Any, Dict, List


class PerformanceMemory:
    """
    Wrapper for SOLLOL's memory module to track performance history.

    Provides adaptive learning by recording actual execution times and
    improving duration predictions over time.
    """

    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.max_history = 1000

    def record_execution(
        self, node_url: str, task_type: str, model: str, duration_ms: float, success: bool
    ):
        """Record an execution for adaptive learning."""
        self.history.append(
            {
                "node_url": node_url,
                "task_type": task_type,
                "model": model,
                "duration_ms": duration_ms,
                "success": success,
                "timestamp": datetime.now(),
            }
        )

        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def get_average_duration(self, node_url: str, task_type: str, model: str = None) -> float:
        """Get average duration for a specific node/task/model combination."""
        relevant = [
            h
            for h in self.history
            if h["node_url"] == node_url
            and h["task_type"] == task_type
            and h["success"]
            and (model is None or h["model"] == model)
        ]

        if not relevant:
            return 0.0

        return sum(h["duration_ms"] for h in relevant) / len(relevant)

    def get_success_rate(self, node_url: str, task_type: str = None) -> float:
        """Get success rate for a node (optionally filtered by task type)."""
        relevant = [
            h
            for h in self.history
            if h["node_url"] == node_url and (task_type is None or h["task_type"] == task_type)
        ]

        if not relevant:
            return 1.0

        successful = sum(1 for h in relevant if h["success"])
        return successful / len(relevant)


class MetricsCollector:
    """
    Wrapper for SOLLOL's metrics module to collect routing and performance metrics.

    Tracks routing decisions, request completion, and provides summary statistics.
    """

    def __init__(self):
        self.routing_decisions: List[Dict[str, Any]] = []
        self.request_completions: List[Dict[str, Any]] = []
        self.task_type_counts: Dict[str, int] = {}
        self.agent_stats: Dict[str, Dict] = {}
        self.max_history = 1000

    def record_routing_decision(
        self,
        agent_name: str,
        task_type: str,
        priority: int,
        selected_node: str,
        score: float,
        routing_time_ms: float,
    ):
        """Record a routing decision."""
        self.routing_decisions.append(
            {
                "agent_name": agent_name,
                "task_type": task_type,
                "priority": priority,
                "selected_node": selected_node,
                "score": score,
                "routing_time_ms": routing_time_ms,
                "timestamp": datetime.now(),
            }
        )

        # Track task types
        self.task_type_counts[task_type] = self.task_type_counts.get(task_type, 0) + 1

        # Keep only recent history
        if len(self.routing_decisions) > self.max_history:
            self.routing_decisions = self.routing_decisions[-self.max_history :]

    def record_request_completion(
        self,
        agent_name: str,
        node_url: str,
        task_type: str,
        priority: int,
        duration_ms: float,
        success: bool,
    ):
        """Record request completion."""
        self.request_completions.append(
            {
                "agent_name": agent_name,
                "node_url": node_url,
                "task_type": task_type,
                "priority": priority,
                "duration_ms": duration_ms,
                "success": success,
                "timestamp": datetime.now(),
            }
        )

        # Update agent stats
        if agent_name not in self.agent_stats:
            self.agent_stats[agent_name] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_duration_ms": 0.0,
            }

        stats = self.agent_stats[agent_name]
        stats["total_requests"] += 1
        if success:
            stats["successful_requests"] += 1
        stats["total_duration_ms"] += duration_ms

        # Keep only recent history
        if len(self.request_completions) > self.max_history:
            self.request_completions = self.request_completions[-self.max_history :]

    def get_latency_percentiles(self) -> Dict[str, float]:
        """
        Calculate P50/P95/P99 latency percentiles from request history.

        Returns:
            Dict with p50_latency_ms, p95_latency_ms, p99_latency_ms
        """
        if not self.request_completions:
            return {"p50_latency_ms": 0.0, "p95_latency_ms": 0.0, "p99_latency_ms": 0.0}

        # Extract durations
        durations = [r["duration_ms"] for r in self.request_completions]

        # Calculate percentiles using numpy
        try:
            import numpy as np

            p50 = float(np.percentile(durations, 50))
            p95 = float(np.percentile(durations, 95))
            p99 = float(np.percentile(durations, 99))
        except ImportError:
            # Fallback to manual percentile calculation if numpy not available
            sorted_durations = sorted(durations)
            n = len(sorted_durations)

            def get_percentile(data, percentile):
                k = (n - 1) * (percentile / 100.0)
                f = int(k)
                c = f + 1
                if c >= n:
                    return data[-1]
                d0 = data[f]
                d1 = data[c]
                return d0 + (d1 - d0) * (k - f)

            p50 = get_percentile(sorted_durations, 50)
            p95 = get_percentile(sorted_durations, 95)
            p99 = get_percentile(sorted_durations, 99)

        return {"p50_latency_ms": p50, "p95_latency_ms": p95, "p99_latency_ms": p99}

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_requests = len(self.request_completions)
        successful_requests = sum(1 for r in self.request_completions if r["success"])

        avg_duration = 0.0
        if total_requests > 0:
            avg_duration = sum(r["duration_ms"] for r in self.request_completions) / total_requests

        # Get latency percentiles
        percentiles = self.get_latency_percentiles()

        return {
            "total_routing_decisions": len(self.routing_decisions),
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 1.0,
            "avg_duration_ms": avg_duration,
            "task_types": self.task_type_counts,
            "agents": self.agent_stats,
            "avg_routing_time_ms": (
                sum(r["routing_time_ms"] for r in self.routing_decisions)
                / len(self.routing_decisions)
                if self.routing_decisions
                else 0.0
            ),
            # Add percentiles to summary
            "p50_latency_ms": percentiles["p50_latency_ms"],
            "p95_latency_ms": percentiles["p95_latency_ms"],
            "p99_latency_ms": percentiles["p99_latency_ms"],
        }
