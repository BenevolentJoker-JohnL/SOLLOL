"""
Distributed Tracing for SOLLOL

Tracks requests across Ollama nodes and Ray actors for full visibility.

Features:
- Automatic request ID generation
- Parent-child span relationships
- Per-hop latency tracking
- Backend attribution
- Dashboard integration
"""

import functools
import logging
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Context variables for distributed tracing
current_trace_id: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
current_span_id: ContextVar[Optional[str]] = ContextVar("span_id", default=None)


@dataclass
class Span:
    """A span represents one step in a distributed trace."""

    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation: str
    backend: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "in_progress"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dict for serialization."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "operation": self.operation,
            "backend": self.backend,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "metadata": self.metadata,
        }


class DistributedTracer:
    """Distributed tracer for SOLLOL requests."""

    def __init__(self, dashboard=None):
        """
        Initialize distributed tracer.

        Args:
            dashboard: UnifiedDashboard instance (optional)
        """
        self.dashboard = dashboard
        self.active_traces: Dict[str, List[Span]] = {}

    def start_trace(self, operation: str, backend: str = "unknown", **metadata) -> Span:
        """
        Start a new trace or add span to existing trace.

        Args:
            operation: Operation name (e.g., "chat", "generate")
            backend: Backend name (e.g., "ollama", "ray_pool_0")
            **metadata: Additional metadata

        Returns:
            Span object
        """
        # Get or create trace ID
        trace_id = current_trace_id.get()
        if not trace_id:
            trace_id = str(uuid.uuid4())
            current_trace_id.set(trace_id)

        # Get parent span ID
        parent_span_id = current_span_id.get()

        # Create new span
        span_id = str(uuid.uuid4())
        current_span_id.set(span_id)

        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation=operation,
            backend=backend,
            start_time=time.time(),
            metadata=metadata,
        )

        # Store span
        if trace_id not in self.active_traces:
            self.active_traces[trace_id] = []
        self.active_traces[trace_id].append(span)

        logger.debug(
            f"Started span {span_id[:8]} for trace {trace_id[:8]} "
            f"(operation: {operation}, backend: {backend})"
        )

        return span

    def end_span(self, span: Span, status: str = "success", **metadata):
        """
        End a span.

        Args:
            span: Span to end
            status: Final status ("success", "error", etc.)
            **metadata: Additional metadata to add
        """
        span.end_time = time.time()
        span.status = status
        span.metadata.update(metadata)

        logger.debug(f"Ended span {span.span_id[:8]} ({span.duration_ms:.2f}ms, status: {status})")

        # Send to dashboard if available
        if self.dashboard:
            try:
                self.dashboard.trace_history.append(span.to_dict())
            except Exception as e:
                logger.warning(f"Failed to send span to dashboard: {e}")

    def get_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get all spans for a trace."""
        spans = self.active_traces.get(trace_id, [])
        return [span.to_dict() for span in spans]

    def trace(self, operation: str = None, backend: str = "unknown"):
        """
        Decorator for automatic tracing.

        Usage:
            @tracer.trace(operation="chat", backend="ollama")
            async def chat(self, model, messages):
                ...
        """

        def decorator(func: Callable):
            op_name = operation or func.__name__

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                span = self.start_trace(op_name, backend)
                try:
                    result = await func(*args, **kwargs)
                    self.end_span(span, status="success")
                    return result
                except Exception as e:
                    self.end_span(span, status="error", error=str(e))
                    raise

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                span = self.start_trace(op_name, backend)
                try:
                    result = func(*args, **kwargs)
                    self.end_span(span, status="success")
                    return result
                except Exception as e:
                    self.end_span(span, status="error", error=str(e))
                    raise

            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator


# Global tracer instance
_global_tracer = None


def get_tracer(dashboard=None) -> DistributedTracer:
    """Get global tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = DistributedTracer(dashboard=dashboard)
    return _global_tracer


# Import asyncio at the end to avoid circular imports
import asyncio
