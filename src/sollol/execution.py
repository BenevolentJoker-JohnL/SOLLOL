"""
Distributed Execution Engine for SOLLOL

Provides parallel and asynchronous execution capabilities with intelligent
SOLLOL routing. Automatically distributes tasks across available nodes for
optimal performance.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, List, Optional

from .integration import SOLLOLLoadBalancer
from .tasks import DistributedTask, ExecutionResult, TaskResult

logger = logging.getLogger(__name__)


class DistributedExecutor:
    """
    Thread-based parallel executor with SOLLOL intelligent routing.

    Executes multiple tasks concurrently across distributed nodes using
    ThreadPoolExecutor and SOLLOL's intelligent routing engine.

    Example:
        >>> from sollol import DistributedExecutor, DistributedTask
        >>> executor = DistributedExecutor(load_balancer, max_workers=10)
        >>> tasks = [
        ...     DistributedTask("task1", {"prompt": "Hello"}, priority=5),
        ...     DistributedTask("task2", {"prompt": "World"}, priority=5)
        ... ]
        >>> result = executor.execute_parallel(tasks, executor_fn=my_execute_fn)
    """

    def __init__(self, load_balancer: SOLLOLLoadBalancer, max_workers: int = 10):
        """
        Initialize distributed executor.

        Args:
            load_balancer: SOLLOL load balancer instance
            max_workers: Maximum concurrent workers
        """
        self.load_balancer = load_balancer
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"✨ DistributedExecutor initialized with {max_workers} workers")

    def execute_parallel(
        self,
        tasks: List[DistributedTask],
        executor_fn: Callable[[DistributedTask, str], Any],
        merge_strategy: str = "collect",
    ) -> ExecutionResult:
        """
        Execute multiple tasks in parallel with SOLLOL routing.

        Args:
            tasks: List of DistributedTask objects to execute
            executor_fn: Function to execute each task (task, node_url) -> result
            merge_strategy: How to combine results ("collect", "vote", "merge", "best")

        Returns:
            ExecutionResult with merged results and statistics
        """
        start_time = time.time()
        logger.info(f"🚀 Starting parallel execution of {len(tasks)} tasks")

        # Submit all tasks concurrently
        futures = {}
        for task in tasks:
            future = self.executor.submit(self._execute_task, task, executor_fn)
            futures[future] = task

        # Collect results as they complete
        results = []
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                results.append(result)

                status_icon = "✅" if result.success else "❌"
                logger.info(
                    f"{status_icon} Task {result.task_id} completed in {result.duration_ms:.0f}ms "
                    f"on {result.node_url}"
                )
            except Exception as e:
                logger.error(f"❌ Task {task.task_id} failed: {e}")
                results.append(
                    TaskResult(
                        task_id=task.task_id,
                        node_url="unknown",
                        result=None,
                        duration_ms=0,
                        success=False,
                        error=str(e),
                    )
                )

        total_time = (time.time() - start_time) * 1000

        # Merge results based on strategy
        from .aggregation import ResultAggregator

        aggregator = ResultAggregator()
        merged_result = aggregator.merge(results, merge_strategy)

        # Calculate statistics
        successful = [r for r in results if r.success]
        avg_duration = sum(r.duration_ms for r in successful) / len(successful) if successful else 0
        sequential_time = sum(r.duration_ms for r in results)
        speedup = sequential_time / total_time if total_time > 0 else 1.0

        logger.info(
            f"✨ Parallel execution complete: {len(successful)}/{len(tasks)} successful "
            f"in {total_time:.0f}ms (speedup: {speedup:.2f}x)"
        )

        return ExecutionResult(
            merged_result=merged_result,
            individual_results=results,
            statistics={
                "total_tasks": len(tasks),
                "successful": len(successful),
                "failed": len(results) - len(successful),
                "total_duration_ms": total_time,
                "avg_task_duration_ms": avg_duration,
                "sequential_duration_ms": sequential_time,
                "speedup_factor": speedup,
                "parallel_efficiency": speedup / self.max_workers if self.max_workers > 0 else 0,
            },
            execution_mode="parallel",
        )

    def _execute_task(
        self, task: DistributedTask, executor_fn: Callable[[DistributedTask, str], Any]
    ) -> TaskResult:
        """Execute a single task with SOLLOL routing."""
        start_time = time.time()

        try:
            # Get routing decision from SOLLOL
            decision = self.load_balancer.route_request(
                payload=task.payload, agent_name=task.task_id, priority=task.priority
            )

            # Execute on selected node
            result = executor_fn(task, decision.node.url)
            duration_ms = (time.time() - start_time) * 1000

            # Record performance for adaptive learning
            self.load_balancer.record_performance(
                decision=decision, actual_duration_ms=duration_ms, success=True
            )

            return TaskResult(
                task_id=task.task_id,
                node_url=decision.node.url,
                result=result,
                duration_ms=duration_ms,
                success=True,
                routing_metadata=self.load_balancer.get_routing_metadata(decision),
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            import traceback

            logger.error(f"Task {task.task_id} failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

            return TaskResult(
                task_id=task.task_id,
                node_url="unknown",
                result=None,
                duration_ms=duration_ms,
                success=False,
                error=str(e),
            )

    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)
        logger.info("DistributedExecutor shutdown complete")


class AsyncDistributedExecutor:
    """
    AsyncIO-based parallel executor for non-blocking execution.

    Uses asyncio for highly efficient concurrent execution with SOLLOL routing.

    Example:
        >>> executor = AsyncDistributedExecutor(load_balancer)
        >>> result = await executor.execute_parallel_async(tasks, executor_fn)
    """

    def __init__(self, load_balancer: SOLLOLLoadBalancer):
        """
        Initialize async executor.

        Args:
            load_balancer: SOLLOL load balancer instance
        """
        self.load_balancer = load_balancer
        logger.info("✨ AsyncDistributedExecutor initialized")

    async def execute_parallel_async(
        self,
        tasks: List[DistributedTask],
        executor_fn: Callable[[DistributedTask, str], Any],
        merge_strategy: str = "collect",
    ) -> ExecutionResult:
        """
        Execute tasks in parallel using asyncio.

        Args:
            tasks: List of DistributedTask objects
            executor_fn: Function to execute each task
            merge_strategy: How to merge results

        Returns:
            ExecutionResult with merged results and statistics
        """
        start_time = time.time()
        logger.info(f"🚀 Starting async parallel execution of {len(tasks)} tasks")

        # Create coroutines for all tasks
        coroutines = [self._execute_task_async(task, executor_fn) for task in tasks]

        # Execute all tasks concurrently
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Process results
        task_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {tasks[i].task_id} failed: {result}")
                task_results.append(
                    TaskResult(
                        task_id=tasks[i].task_id,
                        node_url="unknown",
                        result=None,
                        duration_ms=0,
                        success=False,
                        error=str(result),
                    )
                )
            else:
                task_results.append(result)

        total_time = (time.time() - start_time) * 1000

        # Merge and return results
        from .aggregation import ResultAggregator

        aggregator = ResultAggregator()
        merged_result = aggregator.merge(task_results, merge_strategy)

        successful = [r for r in task_results if r.success]
        avg_duration = sum(r.duration_ms for r in successful) / len(successful) if successful else 0
        sequential_time = sum(r.duration_ms for r in task_results)
        speedup = sequential_time / total_time if total_time > 0 else 1.0

        logger.info(
            f"✨ Async execution complete: {len(successful)}/{len(tasks)} successful "
            f"in {total_time:.0f}ms (speedup: {speedup:.2f}x)"
        )

        return ExecutionResult(
            merged_result=merged_result,
            individual_results=task_results,
            statistics={
                "total_tasks": len(tasks),
                "successful": len(successful),
                "failed": len(task_results) - len(successful),
                "total_duration_ms": total_time,
                "avg_task_duration_ms": avg_duration,
                "sequential_duration_ms": sequential_time,
                "speedup_factor": speedup,
            },
            execution_mode="async_parallel",
        )

    async def _execute_task_async(
        self, task: DistributedTask, executor_fn: Callable[[DistributedTask, str], Any]
    ) -> TaskResult:
        """Execute a task asynchronously."""
        loop = asyncio.get_event_loop()

        # Run blocking execution in executor
        def blocking_call():
            start_time = time.time()

            # Get routing decision
            decision = self.load_balancer.route_request(
                payload=task.payload, agent_name=task.task_id, priority=task.priority
            )

            # Execute task
            result = executor_fn(task, decision.node.url)
            duration_ms = (time.time() - start_time) * 1000

            # Record performance
            self.load_balancer.record_performance(
                decision=decision, actual_duration_ms=duration_ms, success=True
            )

            return TaskResult(
                task_id=task.task_id,
                node_url=decision.node.url,
                result=result,
                duration_ms=duration_ms,
                success=True,
                routing_metadata=self.load_balancer.get_routing_metadata(decision),
            )

        return await loop.run_in_executor(None, blocking_call)
