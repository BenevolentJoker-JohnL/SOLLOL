"""
Intelligent Routing Modes for SOLLOL

Provides three routing strategies optimized for different use cases:
- FAST: Performance-first (GPU, local, lowest latency)
- RELIABLE: Stability-first (proven nodes, high success rate)
- ASYNC: Resource-efficient (CPU OK, queue OK, non-blocking)
"""

import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from loguru import logger

# Bind app_name context to logger if SOLLOL_APP_NAME environment variable is set
_app_name = os.getenv("SOLLOL_APP_NAME", None)
if _app_name:
    logger = logger.bind(app=_app_name)


class RoutingMode(Enum):
    """
    Routing optimization strategy.

    FAST: Optimize for speed (GPU-first, local preference, lowest latency)
    RELIABLE: Optimize for success rate (proven stable nodes, even if slower)
    ASYNC: Optimize for resource efficiency (CPU OK, queue OK, non-blocking)
    """

    FAST = "fast"  # Performance-first (current default)
    RELIABLE = "reliable"  # Stability-first
    ASYNC = "async"  # Resource-efficient, can use CPU


class TaskPriority(Enum):
    """
    Task priority levels.

    Higher priority tasks get better resource allocation and preemption rights.
    """

    URGENT = 10  # User waiting, real-time response needed
    HIGH = 8  # Important but not blocking
    NORMAL = 5  # Default priority
    LOW = 2  # Background/batch processing
    DEFERRED = 0  # Run when resources available


@dataclass
class NodeStats:
    """Statistics for a single node"""

    node_id: str
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    timeouts: int = 0
    response_times: List[float] = field(default_factory=list)
    max_response_times: int = 100  # Keep last 100
    uptime_start: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)"""
        if self.total_requests == 0:
            return 1.0  # No data, assume good
        return self.successful / self.total_requests

    @property
    def avg_response_time(self) -> float:
        """Average response time in seconds"""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)

    @property
    def response_time_variance(self) -> float:
        """Response time variance (lower = more consistent)"""
        if len(self.response_times) < 2:
            return 0.0
        return statistics.variance(self.response_times)

    @property
    def uptime_hours(self) -> float:
        """Hours since tracking started"""
        return (datetime.now() - self.uptime_start).total_seconds() / 3600

    def record_request(self, success: bool, response_time: float, is_timeout: bool = False):
        """Record a request outcome"""
        self.total_requests += 1

        if success:
            self.successful += 1
        else:
            self.failed += 1

        if is_timeout:
            self.timeouts += 1

        # Record response time
        self.response_times.append(response_time)

        # Keep only recent response times
        if len(self.response_times) > self.max_response_times:
            self.response_times.pop(0)


class NodeReliabilityTracker:
    """
    Track node reliability metrics for RELIABLE routing mode.

    Monitors:
    - Success rates per node
    - Response time consistency
    - Timeout rates
    - Uptime
    """

    def __init__(self, app_name: Optional[str] = None):
        self.node_stats: Dict[str, NodeStats] = {}
        self.app_name = app_name

        # Bind app_name to logger if provided
        self._logger = logger
        if app_name:
            self._logger = logger.bind(app=app_name)

        self._logger.info("📊 NodeReliabilityTracker initialized")

    def get_or_create_stats(self, node_id: str) -> NodeStats:
        """Get stats for a node, creating if needed"""
        if node_id not in self.node_stats:
            self.node_stats[node_id] = NodeStats(node_id=node_id)
        return self.node_stats[node_id]

    def record_request(
        self, node_id: str, success: bool, response_time: float, is_timeout: bool = False
    ):
        """Record request outcome for a node"""
        stats = self.get_or_create_stats(node_id)
        stats.record_request(success, response_time, is_timeout)

        # Log warnings for poor performance (only if we have enough data)
        if stats.total_requests >= 10:  # Enough data
            # Only warn if success rate is actually low (not just initial state)
            if stats.success_rate < 0.9 and stats.failed > 0:
                self._logger.warning(
                    f"⚠️ Node {node_id} has low success rate: {stats.success_rate:.1%}"
                )
            if stats.response_time_variance > 10.0:
                self._logger.warning(
                    f"⚠️ Node {node_id} has high response time variance: {stats.response_time_variance:.2f}s"
                )

    def get_success_rate(self, node_id: str) -> float:
        """Get success rate for a node"""
        if node_id not in self.node_stats:
            return 1.0  # No data, assume good
        return self.node_stats[node_id].success_rate

    def get_most_reliable_nodes(
        self, node_ids: List[str], min_success_rate: float = 0.95
    ) -> List[str]:
        """
        Get nodes sorted by reliability.

        Returns nodes with success_rate >= min_success_rate,
        sorted by: success_rate DESC, variance ASC
        """
        candidates = []

        for node_id in node_ids:
            stats = self.get_or_create_stats(node_id)

            # Filter by minimum success rate
            if stats.success_rate < min_success_rate:
                self._logger.debug(
                    f"Node {node_id} filtered: {stats.success_rate:.1%} < {min_success_rate:.1%}"
                )
                continue

            candidates.append(
                {
                    "node_id": node_id,
                    "success_rate": stats.success_rate,
                    "variance": stats.response_time_variance,
                    "avg_time": stats.avg_response_time,
                }
            )

        # Sort by reliability metrics
        candidates.sort(
            key=lambda x: (
                x["success_rate"],  # Higher success rate first
                -x["variance"],  # Lower variance first (negative for DESC)
                x["avg_time"],  # Faster avg time first
            ),
            reverse=True,  # reverse=True makes success_rate DESC
        )

        return [c["node_id"] for c in candidates]

    def get_stats_summary(self) -> Dict:
        """Get summary statistics for all nodes"""
        summary = {}
        for node_id, stats in self.node_stats.items():
            summary[node_id] = {
                "success_rate": stats.success_rate,
                "avg_response_time": stats.avg_response_time,
                "variance": stats.response_time_variance,
                "total_requests": stats.total_requests,
                "uptime_hours": stats.uptime_hours,
            }
        return summary

    def reset_node_stats(self, node_id: str):
        """Reset statistics for a node (e.g., after maintenance)"""
        if node_id in self.node_stats:
            del self.node_stats[node_id]
            self._logger.info(f"🔄 Reset stats for node {node_id}")


@dataclass
class RoutingDecision:
    """
    Result of routing decision.

    Provides transparency into why a particular node was selected.
    """

    selected_node_id: str
    routing_mode: RoutingMode
    priority: int
    reason: str
    candidates_considered: int
    selection_time_ms: float

    # Optional metrics
    node_success_rate: Optional[float] = None
    node_vram_available: Optional[float] = None
    node_type: Optional[str] = None  # 'gpu' or 'cpu'


def get_routing_reason(mode: RoutingMode, node_id: str, node_type: str, **kwargs) -> str:
    """
    Generate human-readable routing decision reason.

    Args:
        mode: Routing mode used
        node_id: Selected node ID
        node_type: 'gpu' or 'cpu'
        **kwargs: Additional context (success_rate, vram, etc.)
    """
    if mode == RoutingMode.FAST:
        reasons = [f"{node_type.upper()}-first routing"]
        if kwargs.get("prefer_local"):
            reasons.append("local preference")
        if kwargs.get("min_vram"):
            reasons.append(f"min {kwargs['min_vram']}GB VRAM")
        return f"FAST mode: {', '.join(reasons)} → {node_id}"

    elif mode == RoutingMode.RELIABLE:
        success_rate = kwargs.get("success_rate", 0)
        reasons = [f"{success_rate:.1%} success rate"]
        variance = kwargs.get("variance")
        if variance is not None:
            reasons.append(f"variance {variance:.2f}s")
        return f"RELIABLE mode: {', '.join(reasons)} → {node_id}"

    elif mode == RoutingMode.ASYNC:
        reasons = []
        if kwargs.get("prefer_cpu"):
            reasons.append("CPU-preferred (GPU-saving)")
        reasons.append("background/queue OK")
        return f"ASYNC mode: {', '.join(reasons)} → {node_id} ({node_type})"

    return f"Auto-routing → {node_id}"


# Model size estimates for determining if model fits on node
MODEL_SIZE_ESTIMATES = {
    # Small models (< 2B params)
    "qwen3:0.5b": 0.5,
    "qwen3:1.7b": 2.0,
    "phi": 2.0,
    "tinyllama": 1.0,
    # Medium models (7-8B params)
    "llama3.1": 8.0,
    "llama3.2": 4.0,
    "mistral": 8.0,
    "qwen3:7b": 8.0,
    "qwen3:14b": 16.0,
    # Large models (13-20B)
    "gpt-oss:20b": 16.0,
    "wizard-math": 8.0,
    # Mixture of Experts (special case)
    "mixtral:8x7b": 28.0,  # ~47B total, ~13B active
    "qwen3:32b": 32.0,
    "qwq:32b": 32.0,
    # Default estimate (if not in list)
    "default": 8.0,
}


def estimate_model_vram_gb(model_name: str) -> float:
    """
    Estimate VRAM needed for a model in GB.

    Args:
        model_name: Name of the model (e.g., "qwen3:1.7b")

    Returns:
        Estimated VRAM in GB
    """
    model_lower = model_name.lower()

    # Check exact matches
    if model_lower in MODEL_SIZE_ESTIMATES:
        return MODEL_SIZE_ESTIMATES[model_lower]

    # Check partial matches
    for key, vram in MODEL_SIZE_ESTIMATES.items():
        if key in model_lower:
            return vram

    # Parse parameter count from name if possible
    # e.g., "some-model:7b" → 7B params → ~8GB
    if "b" in model_lower:
        parts = model_lower.split(":")
        if len(parts) > 1:
            param_str = parts[1].replace("b", "").strip()
            try:
                params = float(param_str)
                # Rough estimate: params * 1.2 (accounting for KV cache, etc.)
                return params * 1.2
            except ValueError:
                pass

    logger.debug(f"Using default VRAM estimate for {model_name}")
    return MODEL_SIZE_ESTIMATES["default"]


def can_model_fit_on_node(
    model_name: str,
    vram_available_mb: float,
    is_cpu: bool = False,
    ram_available_gb: Optional[float] = None,
) -> bool:
    """
    Check if model can fit on a node.

    Args:
        model_name: Name of the model
        vram_available_mb: VRAM available in MB (for GPU)
        is_cpu: Whether this is a CPU node
        ram_available_gb: RAM available in GB (for CPU)

    Returns:
        True if model can fit
    """
    model_vram_gb = estimate_model_vram_gb(model_name)

    if is_cpu:
        # CPU nodes: check RAM instead
        if ram_available_gb is None:
            logger.warning("RAM info not available for CPU node, assuming it fits")
            return True
        return ram_available_gb >= model_vram_gb
    else:
        # GPU nodes: check VRAM
        vram_available_gb = vram_available_mb / 1024
        return vram_available_gb >= model_vram_gb
