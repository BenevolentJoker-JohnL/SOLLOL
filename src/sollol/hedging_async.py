"""
Async Race-to-First (Hedging) Strategy for SOLLOL Gateway.

Sends the same request to multiple nodes simultaneously and uses whichever
responds first. Dramatically reduces tail latency for user-facing queries.

Trade-off:
- Better: p99 latency reduced by 50-90%
- Worse: 2-3x resource usage (multiple concurrent requests)
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AsyncHedgeResult:
    """Result from async hedged request."""

    winner_node: str
    response: Any
    latency_ms: float
    num_requests_sent: int
    num_responses_received: int
    cancelled_nodes: List[str]


class AsyncHedgingStrategy:
    """
    Async race-to-first hedging for SOLLOL gateway.

    Sends same request to multiple nodes, uses fastest response.
    Designed for async FastAPI integration.
    """

    def __init__(
        self,
        num_hedges: int = 2,
        enable_cancellation: bool = True,
        hedge_threshold_ms: float = 100.0,
    ):
        """
        Initialize async hedging strategy.

        Args:
            num_hedges: Number of parallel requests (2-3 recommended)
            enable_cancellation: Cancel slower requests after first response
            hedge_threshold_ms: Only hedge if expected latency > this threshold
        """
        self.num_hedges = num_hedges
        self.enable_cancellation = enable_cancellation
        self.hedge_threshold_ms = hedge_threshold_ms

        # Statistics
        self.total_requests = 0
        self.total_hedge_requests = 0
        self.total_wasted_requests = 0
        self.latency_reductions: List[float] = []

    def should_hedge(
        self,
        priority: int = 5,
        estimated_latency_ms: float = 0,
        cluster_load: float = 0.5,
    ) -> bool:
        """
        Decide whether to hedge this request.

        Args:
            priority: Request priority (1-10, 10 = highest)
            estimated_latency_ms: Estimated latency from routing
            cluster_load: Current cluster load (0-1)

        Returns:
            True if should hedge
        """
        # Don't hedge if cluster is overloaded (>80% load)
        if cluster_load > 0.8:
            logger.debug(f"⏭️  Skipping hedge: cluster load {cluster_load:.0%} > 80%")
            return False

        # Always hedge for high-priority tasks (8-10)
        if priority >= 8:
            logger.debug(f"🔥 Hedging: high priority ({priority}/10)")
            return True

        # Hedge if estimated latency is high
        if estimated_latency_ms > self.hedge_threshold_ms:
            logger.debug(
                f"🐌 Hedging: estimated latency {estimated_latency_ms:.0f}ms > {self.hedge_threshold_ms:.0f}ms"
            )
            return True

        logger.debug("⏭️  Skipping hedge: not critical enough")
        return False

    async def hedge_request(
        self,
        request_fn: Callable,
        nodes: List[Dict],
        *args,
        **kwargs,
    ) -> AsyncHedgeResult:
        """
        Send hedged request to multiple nodes, return fastest.

        Args:
            request_fn: Async function to execute request
            nodes: List of node dicts with 'host' and 'port'
            *args, **kwargs: Arguments to pass to request_fn

        Returns:
            AsyncHedgeResult with winner and stats
        """
        start_time = time.time()

        # Select top N nodes for hedging
        hedge_nodes = nodes[: self.num_hedges]
        node_keys = [f"{n['host']}:{n['port']}" for n in hedge_nodes]

        logger.info(f"🏁 Hedging request across {len(hedge_nodes)} nodes: {node_keys}")

        # Create tasks for all hedge requests
        tasks = []
        for node in hedge_nodes:
            task = asyncio.create_task(
                self._execute_request_with_timing(request_fn, node, *args, **kwargs)
            )
            tasks.append((task, node))

        # Wait for first successful response
        winner_node = None
        winner_response = None
        winner_latency = None
        responses_received = 0
        pending_tasks = [t[0] for t in tasks]

        try:
            while pending_tasks and winner_node is None:
                done, pending = await asyncio.wait(
                    pending_tasks, return_when=asyncio.FIRST_COMPLETED
                )

                for completed_task in done:
                    responses_received += 1

                    # Find which node this task belongs to
                    node = None
                    for task, n in tasks:
                        if task == completed_task:
                            node = n
                            break

                    node_key = f"{node['host']}:{node['port']}"

                    try:
                        response, latency = await completed_task

                        if winner_node is None:
                            # First successful response wins
                            winner_node = node_key
                            winner_response = response
                            winner_latency = latency

                            logger.info(
                                f"✅ Winner: {node_key} ({latency:.0f}ms) "
                                f"- cancelling {len(pending)} slower requests"
                            )

                            # Cancel remaining requests (if enabled)
                            if self.enable_cancellation:
                                for p in pending:
                                    p.cancel()

                            break  # Exit loop

                    except Exception as e:
                        logger.warning(f"⚠️  Hedge request failed on {node_key}: {str(e)[:100]}")
                        continue

                pending_tasks = list(pending)

        finally:
            # Ensure all tasks are cancelled
            for task, _ in tasks:
                if not task.done():
                    task.cancel()

        # Build result
        if winner_node is None:
            raise Exception("All hedge requests failed")

        total_latency = (time.time() - start_time) * 1000
        cancelled_nodes = [k for k in node_keys if k != winner_node]

        result = AsyncHedgeResult(
            winner_node=winner_node,
            response=winner_response,
            latency_ms=winner_latency,
            num_requests_sent=len(hedge_nodes),
            num_responses_received=responses_received,
            cancelled_nodes=cancelled_nodes,
        )

        # Update statistics
        self.total_requests += 1
        self.total_hedge_requests += len(hedge_nodes)
        self.total_wasted_requests += len(hedge_nodes) - 1

        logger.debug(
            f"📊 Hedge stats: sent={len(hedge_nodes)}, "
            f"received={responses_received}, wasted={len(hedge_nodes) - 1}"
        )

        return result

    async def _execute_request_with_timing(self, request_fn: Callable, node: Dict, *args, **kwargs):
        """
        Execute single async request with timing.

        Args:
            request_fn: Async request function
            node: Node dict with host/port
            *args, **kwargs: Arguments for request_fn

        Returns:
            (response, latency_ms)
        """
        start = time.time()

        try:
            # Execute async request
            response = await request_fn(node, *args, **kwargs)
            latency = (time.time() - start) * 1000

            return response, latency

        except Exception as e:
            node_key = f"{node['host']}:{node['port']}"
            logger.debug(f"Request failed on {node_key}: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        Get hedging statistics.

        Returns:
            Statistics dictionary
        """
        waste_percentage = (
            (self.total_wasted_requests / self.total_hedge_requests * 100)
            if self.total_hedge_requests > 0
            else 0
        )

        avg_latency_reduction = (
            sum(self.latency_reductions) / len(self.latency_reductions)
            if self.latency_reductions
            else 0
        )

        return {
            "enabled": True,
            "total_requests": self.total_requests,
            "total_hedge_requests": self.total_hedge_requests,
            "total_wasted_requests": self.total_wasted_requests,
            "waste_percentage": waste_percentage,
            "avg_latency_reduction_ms": avg_latency_reduction,
            "num_hedges_per_request": self.num_hedges,
            "hedge_threshold_ms": self.hedge_threshold_ms,
        }
