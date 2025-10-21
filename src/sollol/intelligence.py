"""
Intelligent routing engine - Context-aware request classification and optimization.

This module provides advanced routing decisions based on:
- Request content analysis
- Task type detection
- Historical performance patterns
- Resource requirements prediction
- VRAM-aware GPU placement (FlockParser enhancement)
"""

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Configurable VRAM threshold for GPU overload detection (in MB)
# Set via environment variable SOLLOL_MIN_VRAM_MB or defaults to 1000 MB
MIN_VRAM_THRESHOLD_MB = int(os.environ.get("SOLLOL_MIN_VRAM_MB", "1000"))

# Import VRAM-aware routing utilities
try:
    from sollol.intelligent_gpu_router import IntelligentGPURouter
    from sollol.model_sizes import estimate_model_size, can_fit_in_vram
    VRAM_ROUTING_AVAILABLE = True
except ImportError:
    VRAM_ROUTING_AVAILABLE = False
    logger.debug("VRAM-aware routing not available (intelligent_gpu_router not imported)")


@dataclass
class TaskContext:
    """Rich context about a request for intelligent routing decisions."""

    task_type: str  # 'generation', 'embedding', 'classification', 'extraction', etc.
    complexity: str  # 'simple', 'medium', 'complex'
    estimated_tokens: int
    model_preference: Optional[str]
    priority: int  # 1-10, higher = more important
    requires_gpu: bool
    estimated_duration_ms: float
    metadata: Dict


class IntelligentRouter:
    """
    Context-aware routing engine that makes smart decisions about which
    OLLOL node should handle each request.

    Unlike simple round-robin or random load balancing, this router:
    - Analyzes request content to determine task type
    - Predicts resource requirements
    - Routes based on node capabilities and current load
    - Learns from historical patterns
    """

    def __init__(self, coordinator=None, gpu_router=None, registry=None):
        self.task_patterns = {
            "embedding": [
                r"embed",
                r"vector",
                r"similarity",
                r"semantic.*search",
            ],
            "generation": [
                r"generat",
                r"creat",
                r"writ",
                r"complet",
                r"continue",
            ],
            "classification": [
                r"classif",
                r"categor",
                r"label",
                r"sentiment",
            ],
            "extraction": [
                r"extract",
                r"parse",
                r"identif",
                r"find.*entities",
            ],
            "summarization": [
                r"summar",
                r"condense",
                r"brief",
            ],
            "analysis": [
                r"analyz",
                r"evaluat",
                r"assess",
            ],
        }

        # Historical performance by task type and model
        self.performance_history: Dict[str, List[float]] = {}

        # Node capabilities (GPU, CPU-heavy models, etc.)
        self.node_capabilities: Dict[str, Dict] = {}

        # Distributed coordination (if enabled)
        self.coordinator = coordinator

        # VRAM-aware GPU routing (FlockParser enhancement)
        self.gpu_router = gpu_router
        if gpu_router is None and VRAM_ROUTING_AVAILABLE and registry:
            # Auto-create GPU router if registry available
            self.gpu_router = IntelligentGPURouter(registry=registry)
            logger.info("✅ VRAM-aware GPU routing enabled")

    def detect_task_type(self, payload: Dict) -> str:
        """
        Intelligently detect what kind of task this request represents.

        Args:
            payload: Request payload (messages, prompts, etc.)

        Returns:
            Task type string ('generation', 'embedding', 'classification', etc.)
        """
        # Check if it's explicitly an embedding request
        if "prompt" in payload and "model" in payload:
            model = payload.get("model", "").lower()
            if "embed" in model or "nomic" in model:
                return "embedding"

        # Analyze message content
        content = self._extract_content(payload)
        content_lower = content.lower()

        # Score each task type based on pattern matches
        scores = {}
        for task_type, patterns in self.task_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    score += 1
            if score > 0:
                scores[task_type] = score

        # Return highest scoring task type, or 'generation' as default
        if scores:
            return max(scores, key=scores.get)

        return "generation"

    def estimate_complexity(self, payload: Dict) -> Tuple[str, int]:
        """
        Estimate request complexity and token count.

        Returns:
            (complexity_level, estimated_tokens)
        """
        content = self._extract_content(payload)

        # Rough token estimation (4 chars ≈ 1 token)
        estimated_tokens = len(content) // 4

        # Classify complexity
        if estimated_tokens < 500:
            complexity = "simple"
        elif estimated_tokens < 2000:
            complexity = "medium"
        else:
            complexity = "complex"

        # Check for multi-turn conversation
        if isinstance(payload.get("messages"), list):
            if len(payload["messages"]) > 5:
                # Long conversations are more complex
                complexity = "complex" if complexity != "complex" else complexity
                estimated_tokens *= 1.5

        return complexity, int(estimated_tokens)

    def analyze_request(self, payload: Dict, priority: int = 5) -> TaskContext:
        """
        Perform full request analysis to build routing context.

        Args:
            payload: Request payload
            priority: Request priority (1-10)

        Returns:
            TaskContext with all routing information
        """
        task_type = self.detect_task_type(payload)
        complexity, tokens = self.estimate_complexity(payload)

        # Determine if GPU is beneficial
        # Embeddings ALWAYS benefit from GPU (vector operations)
        # Other tasks benefit from GPU for medium/complex workloads
        if task_type == "embedding":
            requires_gpu = True  # Always use GPU for embeddings
        else:
            requires_gpu = task_type in [
                "generation",
                "summarization",
                "analysis",
                "extraction",  # PDF extraction benefits from GPU
            ] and complexity in ["medium", "complex"]

        # Estimate duration based on historical data
        estimated_duration = self._estimate_duration(task_type, tokens)

        # Extract model preference if specified
        model_preference = payload.get("model")

        return TaskContext(
            task_type=task_type,
            complexity=complexity,
            estimated_tokens=tokens,
            model_preference=model_preference,
            priority=priority,
            requires_gpu=requires_gpu,
            estimated_duration_ms=estimated_duration,
            metadata={
                "analyzed_at": datetime.now().isoformat(),
                "confidence": "high" if task_type != "generation" else "medium",
            },
        )

    def select_optimal_node(
        self, context: TaskContext, available_hosts: List[Dict]
    ) -> Tuple[str, Dict]:
        """
        Select the optimal OLLOL node for this request based on context.

        With distributed coordination enabled, this method:
        1. Acquires distributed lock for atomic routing
        2. Gets aggregated cluster state from all instances
        3. Makes routing decision with global state
        4. Immediately updates global active request count

        GPU Overload Handling:
        - If all GPUs have low VRAM (< MIN_VRAM_THRESHOLD_MB), enables CPU fallback
        - Allows CPU-only nodes to compete when GPUs are overwhelmed

        Args:
            context: Task context from analyze_request()
            available_hosts: List of available host metadata

        Returns:
            (selected_host, routing_decision) tuple with reasoning
        """
        # Filter to only truly available hosts
        available = [h for h in available_hosts if h.get("available", True)]

        if not available:
            raise ValueError("No available hosts")

        # GPU Overload Detection: Check if all GPUs are overwhelmed
        if context.requires_gpu:
            gpu_hosts = [h for h in available if h.get("gpu_free_mem", 0) > 0]
            all_gpus_overwhelmed = all(
                h.get("gpu_free_mem", 0) < MIN_VRAM_THRESHOLD_MB for h in gpu_hosts
            ) if gpu_hosts else True

            if all_gpus_overwhelmed:
                # All GPUs overwhelmed - enable CPU fallback
                logger.warning(
                    f"All GPUs overwhelmed (VRAM < {MIN_VRAM_THRESHOLD_MB} MB) - enabling CPU fallback for {context.task_type}"
                )
                # Modify context to allow CPU fallback
                context = TaskContext(
                    task_type=context.task_type,
                    complexity=context.complexity,
                    estimated_tokens=context.estimated_tokens,
                    model_preference=context.model_preference,
                    priority=context.priority,
                    requires_gpu=False,  # ← Enable CPU fallback
                    estimated_duration_ms=context.estimated_duration_ms,
                    metadata={**context.metadata, "gpu_fallback_to_cpu": True},
                )

        # If distributed coordination enabled, use it
        if self.coordinator:
            # Use distributed lock for atomic routing decision
            with self.coordinator.routing_lock():
                # Update host metadata with aggregated state from all instances
                for host_meta in available:
                    host_addr = host_meta["host"]
                    aggregated_state = self.coordinator.get_aggregated_node_state(host_addr)

                    if aggregated_state:
                        # Override local state with global aggregated state
                        host_meta["active_requests"] = aggregated_state.active_requests
                        host_meta["cpu_load"] = aggregated_state.cpu_load
                        host_meta["gpu_free_mem"] = aggregated_state.gpu_free_mem
                        host_meta["success_rate"] = aggregated_state.success_rate
                        host_meta["latency_ms"] = aggregated_state.avg_latency_ms

                # Score each host with global state
                scored_hosts = []
                for host_meta in available:
                    score = self._score_host_for_context(host_meta, context)
                    scored_hosts.append((host_meta, score))

                # Sort by score (descending)
                scored_hosts.sort(key=lambda x: x[1], reverse=True)

                # Select best host
                best_host, best_score = scored_hosts[0]
                selected_host_addr = best_host["host"]

                # Immediately increment active requests in global state
                new_count = self.coordinator.increment_active_requests(selected_host_addr)

                # Build decision reasoning
                decision = {
                    "selected_host": selected_host_addr,
                    "score": best_score,
                    "task_type": context.task_type,
                    "complexity": context.complexity,
                    "reasoning": self._explain_decision(best_host, context, scored_hosts),
                    "alternatives": [
                        {"host": h["host"], "score": s} for h, s in scored_hosts[1:3]  # Top 2 alternatives
                    ],
                    "distributed": True,
                    "global_active_requests": new_count,
                }

                return selected_host_addr, decision

        # Local mode (no distributed coordination)
        # Score each host
        scored_hosts = []
        for host_meta in available:
            score = self._score_host_for_context(host_meta, context)
            scored_hosts.append((host_meta, score))

        # Sort by score (descending)
        scored_hosts.sort(key=lambda x: x[1], reverse=True)

        # Select best host
        best_host, best_score = scored_hosts[0]

        # Build decision reasoning
        decision = {
            "selected_host": best_host["host"],
            "score": best_score,
            "task_type": context.task_type,
            "complexity": context.complexity,
            "reasoning": self._explain_decision(best_host, context, scored_hosts),
            "alternatives": [
                {"host": h["host"], "score": s} for h, s in scored_hosts[1:3]  # Top 2 alternatives
            ],
            "distributed": False,
        }

        return best_host["host"], decision

    def _score_host_for_context(self, host_meta: Dict, context: TaskContext) -> float:
        """
        Score how well a host matches the request context AND resources.

        Scoring factors (in order of importance):
        1. Availability (binary: available or not)
        2. Resource adequacy (does it have what the task needs?)
        3. Current performance (latency, success rate)
        4. Current load (CPU, GPU utilization)
        5. Priority/preferences (host priority, task priority alignment)

        Higher score = better match for this request.
        """
        score = 100.0  # Start with baseline

        # Factor 1: Availability (CRITICAL - binary disqualification)
        if not host_meta.get("available", True):
            return 0.0

        # Factor 2: Resource adequacy (CRITICAL for resource-intensive tasks)
        # GPU requirements - balanced to use ALL hardware based on capacity
        if context.requires_gpu:
            gpu_mem = host_meta.get("gpu_free_mem", 0)
            active_requests = host_meta.get("active_requests", 0)

            if gpu_mem == 0:
                # No GPU but task needs it - heavy penalty
                score *= 0.2  # Still possible but very low priority
            elif gpu_mem < MIN_VRAM_THRESHOLD_MB:
                # Low VRAM GPU (1050Ti, etc.) - SHOULD BE USED for overflow when others loaded
                # Light penalty - will get used when high-VRAM GPUs saturate
                # VRAM-aware active_requests penalty (50% per request) prevents overload
                base_penalty = 0.90  # Only 10% penalty (lets it compete)

                # If this low-VRAM GPU is idle, it can take overflow
                # If it already has requests, penalize more to avoid overloading small GPU
                if active_requests == 0:
                    # Idle low-VRAM GPU - apply base penalty
                    score *= base_penalty
                else:
                    # Low-VRAM GPU already has requests - penalize more to avoid OOM
                    score *= base_penalty * 0.6  # Additional 40% penalty when loaded
            elif gpu_mem > 8000:
                # Excellent GPU availability - big bonus! (5090, 4090, etc.)
                score *= 2.0
            elif gpu_mem > 4000:
                # Good GPU availability - bonus! (3060, 3070, 4060, etc.)
                score *= 1.5
            elif gpu_mem >= MIN_VRAM_THRESHOLD_MB:
                # Mid-range VRAM (threshold to 4GB) - neutral
                score *= 1.0
        elif context.metadata.get("gpu_fallback_to_cpu"):
            # GPU fallback to CPU enabled - prefer hosts with available resources
            # This happens when all GPUs are overwhelmed
            gpu_mem = host_meta.get("gpu_free_mem", 0)
            if gpu_mem == 0:
                # CPU-only node - acceptable when GPU fallback enabled
                score *= 1.0  # No penalty (fair chance against overwhelmed GPUs)
            elif gpu_mem < MIN_VRAM_THRESHOLD_MB:
                # Overwhelmed GPU - slight penalty
                score *= 0.7
            else:
                # GPU with decent VRAM - still preferred
                score *= 1.2

        # CPU requirements based on complexity
        # Skip CPU penalties for low-VRAM overflow GPUs - any GPU is better than waiting
        gpu_mem = host_meta.get("gpu_free_mem", 4000)
        is_overflow_gpu = gpu_mem < MIN_VRAM_THRESHOLD_MB  # Low-VRAM GPU used for overflow

        cpu_load = host_meta.get("cpu_load", 0.5)  # Get cpu_load for all hosts

        if not is_overflow_gpu:  # Only apply CPU penalties to primary GPUs
            if context.complexity == "complex":
                # Complex tasks need low CPU load
                if cpu_load > 0.8:
                    score *= 0.3  # Very busy host, bad for complex tasks
                elif cpu_load < 0.3:
                    score *= 1.3  # Idle host, great for complex tasks
            elif context.complexity == "simple":
                # Simple tasks can tolerate higher load
                if cpu_load > 0.9:
                    score *= 0.7  # Still penalize very busy hosts
                # Don't bonus idle hosts for simple tasks

        # Factor 3: Current performance
        success_rate = host_meta.get("success_rate", 1.0)
        score *= success_rate  # Direct multiplier

        # Skip latency penalties for overflow GPUs - speed matters less than availability
        if not is_overflow_gpu:  # Only penalize latency on primary GPUs
            latency_ms = host_meta.get("latency_ms", 200.0)
            # Latency penalty scales with task priority and is more aggressive for extreme values
            latency_weight = 1.0 + (context.priority / 10.0)  # 1.0 to 2.0

            # Reduce penalty for fast local network latency (<200ms)
            # Anything under 200ms is considered "fast" - minimal penalty
            if latency_ms < 200:
                # Gentle penalty for local network latency
                latency_penalty = (latency_ms / 1000.0) * latency_weight  # Much smaller penalty
            elif latency_ms > 1000:
                # Exponential penalty for very high latency
                latency_penalty = (latency_ms / 100.0) * latency_weight
            else:
                # Standard penalty for medium latency (200-1000ms)
                latency_penalty = min(latency_ms / 100.0, 10.0) * latency_weight
            score /= 1 + latency_penalty

        # Factor 4: Additional load considerations
        # Active concurrent requests - BALANCED exponential penalty
        # Goal: Prioritize SPEED (fastest completion) over forced distribution
        # A powerful GPU with 1 request may still be faster than weak GPU with 0 requests
        active_requests = host_meta.get("active_requests", 0)
        if active_requests > 0:
            gpu_mem = host_meta.get("gpu_free_mem", 4000)  # Default to mid-range
            avg_latency = host_meta.get("latency_ms", MIN_VRAM_THRESHOLD_MB * 2)  # Historical avg latency

            # Calculate LIGHTER exponential penalty based on GPU capability
            # Key insight: Powerful GPUs can handle multiple requests efficiently
            # Small GPUs saturate quickly, large GPUs maintain performance
            if gpu_mem < MIN_VRAM_THRESHOLD_MB:
                # Small GPU (1050Ti) - saturates quickly
                # 0 req: 1.0x, 1 req: 0.50x, 2 req: 0.33x, 3 req: 0.25x
                # Still usable with multiple requests, just slower
                exponential_base = 1.5  # Reduced from 3.0
            elif gpu_mem < 4000:
                # Mid-small GPU (2-4GB)
                # 0 req: 1.0x, 1 req: 0.59x, 2 req: 0.42x, 3 req: 0.31x
                exponential_base = 1.3  # Reduced from 2.5
            elif gpu_mem < 8000:
                # Mid GPU (4-8GB, like 3060)
                # 0 req: 1.0x, 1 req: 0.71x, 2 req: 0.56x, 3 req: 0.45x
                exponential_base = 1.2  # Reduced from 2.0
            else:
                # Large GPU (8GB+, like 4090/5090) - minimal saturation
                # 0 req: 1.0x, 1 req: 0.83x, 2 req: 0.71x, 3 req: 0.62x
                # Powerful GPUs maintain good performance with multiple requests
                exponential_base = 1.15  # Reduced from 1.5

            # PERFORMANCE-AWARE: Adjust penalty based on actual latency
            # Fast nodes (low latency) get reduced penalty - they're efficient
            # Slow nodes (high latency) get increased penalty - already struggling
            if avg_latency < 1000:  # Fast node (<1s avg)
                # Reduce penalty for fast nodes - they can handle more
                exponential_base = max(1.05, exponential_base * 0.8)
            elif avg_latency > 5000:  # Slow node (>5s avg)
                # Increase penalty for slow nodes - already overloaded
                exponential_base = min(2.0, exponential_base * 1.3)

            # EXPONENTIAL penalty: score / (base ^ requests)
            # Less aggressive than before - allows powerful nodes to stay competitive
            score /= (exponential_base ** active_requests)

            # LINEAR penalty for fairness - combined with exponential
            # This provides base load balancing without being too aggressive
            linear_penalty = 0.15 * active_requests  # 15% per request
            score /= (1 + linear_penalty)

        # CPU load - historical average
        # Penalize heavily loaded nodes more for high-priority tasks
        if context.priority >= 7:  # High priority
            load_penalty = cpu_load * 3.0  # Aggressive penalty
        else:
            load_penalty = cpu_load * 1.5  # Standard penalty
        score /= 1 + load_penalty

        # Factor 5: Model warmth (avoid cold loads)
        # CRITICAL: Give huge bonus to nodes with model already loaded (avoid 18+ sec cold loads)
        if context.model_preference:
            loaded_models = host_meta.get("loaded_models", [])
            # Check if target model is loaded (handle both "model" and "model:tag" formats)
            model_base = context.model_preference.split(':')[0]
            is_loaded = any(
                context.model_preference == loaded or
                model_base in loaded or
                loaded.startswith(model_base)
                for loaded in loaded_models
            )
            if is_loaded:
                # Model already loaded - moderate bonus (1.5x) to avoid cold loads
                # Reduced from 3x to allow GPU capability to matter more
                score *= 1.5
                logger.debug(f"   ✅ Model {context.model_preference} already loaded on {host_meta.get('host', 'unknown')} - 1.5x bonus")
            elif loaded_models:  # Has other models loaded
                # Not loaded - slight penalty for nodes with full VRAM
                score *= 0.95

        # Factor 6: Priority alignment
        # Prefer priority 0 hosts for high-priority tasks
        host_priority = host_meta.get("priority", 999)
        if host_priority == 0 and context.priority >= 7:
            score *= 1.5  # Strong bonus for high-pri tasks on high-pri hosts
        elif host_priority == 0:
            score *= 1.2  # Standard bonus

        # Factor 6: Task-type specialization
        # If host has metadata about preferred task types, use it
        preferred_tasks = host_meta.get("preferred_task_types", [])
        if context.task_type in preferred_tasks:
            score *= 1.3

        # Factor 6.5: Resource capacity bonus
        # Prefer nodes with more CPU cores (especially for complex tasks)
        cpu_count = host_meta.get("cpu_count", 1)
        if cpu_count > 1:
            # Bonus scales with CPU count, capped at 1.5x
            # 4 cores = 1.1x, 8 cores = 1.2x, 16+ cores = 1.4x
            resource_bonus = min(1.0 + (cpu_count / 20.0), 1.5)
            score *= resource_bonus
            # Extra bonus for complex tasks on high-core-count machines
            if context.complexity == "complex" and cpu_count >= 8:
                score *= 1.2  # 20% extra for complex work on powerful nodes

        # Factor 7: Resource headroom for estimated duration
        # Penalize if estimated duration is long and host is already loaded
        if context.estimated_duration_ms > 5000:  # > 5 seconds
            if cpu_load > 0.6:
                score *= 0.7  # Don't want long tasks on busy hosts

        return score

    def _explain_decision(
        self, selected_host: Dict, context: TaskContext, all_scored: List[Tuple[Dict, float]]
    ) -> str:
        """
        Generate human-readable explanation of routing decision.
        """
        reasons = []

        host = selected_host["host"]
        latency = selected_host.get("latency_ms", 0)
        success_rate = selected_host.get("success_rate", 1.0)

        reasons.append(f"Task: {context.task_type} ({context.complexity})")
        reasons.append(f"Host {host}: latency={latency:.1f}ms, success={success_rate:.1%}")

        if context.requires_gpu:
            gpu_mem = selected_host.get("gpu_free_mem", 0)
            reasons.append(f"GPU preferred: {gpu_mem}MB available")

        if len(all_scored) > 1:
            second_best = all_scored[1]
            score_margin = all_scored[0][1] - second_best[1]
            if score_margin < 10:
                reasons.append(f"Close call (margin: {score_margin:.1f})")

        return "; ".join(reasons)

    def _extract_content(self, payload: Dict) -> str:
        """Extract text content from various payload formats."""
        # Chat format
        if "messages" in payload:
            messages = payload["messages"]
            if isinstance(messages, list):
                return " ".join(
                    [msg.get("content", "") for msg in messages if isinstance(msg, dict)]
                )

        # Embedding format
        if "prompt" in payload:
            return str(payload["prompt"])

        # Direct content
        if "content" in payload:
            return str(payload["content"])

        return ""

    def _estimate_duration(self, task_type: str, tokens: int) -> float:
        """
        Estimate request duration based on task type and token count.

        Returns:
            Estimated duration in milliseconds
        """
        # Base durations by task type (ms per token)
        base_rates = {
            "embedding": 0.5,  # Fast
            "classification": 1.0,  # Medium
            "extraction": 1.5,  # Medium-slow
            "generation": 3.0,  # Slow (autoregressive)
            "summarization": 2.5,  # Slow
            "analysis": 2.0,  # Medium-slow
        }

        rate = base_rates.get(task_type, 2.0)
        return tokens * rate

    def record_performance(self, task_type: str, model: str, actual_duration_ms: float):
        """
        Record actual performance for learning and optimization.

        This allows the router to improve over time.
        """
        key = f"{task_type}:{model}"
        if key not in self.performance_history:
            self.performance_history[key] = []

        self.performance_history[key].append(actual_duration_ms)

        # Keep only last 100 measurements
        if len(self.performance_history[key]) > 100:
            self.performance_history[key] = self.performance_history[key][-100:]


# Global router instance
_router = None


def get_router(coordinator=None) -> IntelligentRouter:
    """
    Get the global intelligent router instance.

    Args:
        coordinator: Optional distributed coordinator for multi-instance coordination

    Returns:
        IntelligentRouter instance (creates new one if coordinator provided, else returns global)
    """
    global _router

    # If coordinator provided, create new router with it
    if coordinator is not None:
        return IntelligentRouter(coordinator=coordinator)

    # Otherwise, return or create global instance
    if _router is None:
        _router = IntelligentRouter()

    return _router
