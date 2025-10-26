#!/usr/bin/env python3
"""
SOLLOL Routing Log Viewer

Real-time viewer for SOLLOL routing decisions across all instances on the network.

Usage:
    python -m sollol.routing_viewer                    # View all routing events
    python -m sollol.routing_viewer --model llama3.2   # Filter by model
    python -m sollol.routing_viewer --backend rpc      # Filter by backend
    python -m sollol.routing_viewer --event-type ROUTE_DECISION  # Filter by event
    python -m sollol.routing_viewer --instance hostname_1234  # Filter by instance
    python -m sollol.routing_viewer --history 100      # Show last 100 events from history

Environment variables:
    SOLLOL_REDIS_URL: Redis URL (default: redis://localhost:6379)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional

try:
    import redis
except ImportError:
    print("❌ Redis package not installed. Install with: pip install redis")
    sys.exit(1)


# ANSI color codes
class Colors:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


class RoutingViewer:
    """Real-time routing event viewer."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        filter_model: Optional[str] = None,
        filter_backend: Optional[str] = None,
        filter_event_type: Optional[str] = None,
        filter_instance: Optional[str] = None,
    ):
        """
        Initialize routing viewer.

        Args:
            redis_url: Redis connection URL
            filter_model: Only show events for this model
            filter_backend: Only show events for this backend (ollama/rpc)
            filter_event_type: Only show this event type
            filter_instance: Only show events from this instance
        """
        self.redis_url = redis_url
        self.filter_model = filter_model
        self.filter_backend = filter_backend
        self.filter_event_type = filter_event_type
        self.filter_instance = filter_instance

        # Connect to Redis
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            print(f"{Colors.GREEN}✅ Connected to Redis: {redis_url}{Colors.RESET}\n")
        except Exception as e:
            print(f"{Colors.RED}❌ Failed to connect to Redis: {e}{Colors.RESET}")
            sys.exit(1)

        # Channel and stream names
        self.channel = "sollol:routing_events"
        self.stream_key = "sollol:routing_stream"

        # Stats
        self.event_count = 0
        self.filtered_count = 0

    def _print_header(self):
        """Print viewer header."""
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 100}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}SOLLOL ROUTING LOG VIEWER{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 100}{Colors.RESET}\n")

        # Print filters
        filters = []
        if self.filter_model:
            filters.append(f"model={self.filter_model}")
        if self.filter_backend:
            filters.append(f"backend={self.filter_backend}")
        if self.filter_event_type:
            filters.append(f"event={self.filter_event_type}")
        if self.filter_instance:
            filters.append(f"instance={self.filter_instance}")

        if filters:
            print(f"{Colors.YELLOW}Filters active: {', '.join(filters)}{Colors.RESET}\n")

        print(f"{Colors.GRAY}Listening for routing events...{Colors.RESET}\n")

    def _should_display(self, event: dict) -> bool:
        """Check if event passes filters."""
        if self.filter_model and event.get("model") != self.filter_model:
            return False
        if self.filter_backend and event.get("backend") != self.filter_backend:
            return False
        if self.filter_event_type and event.get("event_type") != self.filter_event_type:
            return False
        if self.filter_instance and event.get("instance_id") != self.filter_instance:
            return False
        return True

    def _get_color_for_event(self, event_type: str) -> str:
        """Get color for event type."""
        color_map = {
            "ROUTE_DECISION": Colors.CYAN,
            "CACHE_HIT": Colors.BLUE,
            "TASK_QUEUED": Colors.BLUE,
            "TASK_START": Colors.GREEN,
            "TASK_COMPLETE": Colors.GREEN,
            "WORKER_LOAD": Colors.GRAY,
            "FALLBACK_TRIGGERED": Colors.YELLOW,
            "MODEL_SWITCH": Colors.MAGENTA,
            "COORDINATOR_START": Colors.GREEN,
            "COORDINATOR_STOP": Colors.RED,
            "RPC_BACKEND_SELECTED": Colors.CYAN,
            "OLLAMA_NODE_SELECTED": Colors.BLUE,
        }
        return color_map.get(event_type, Colors.WHITE)

    def _format_event(self, event: dict) -> str:
        """Format event for display."""
        event_type = event.get("event_type", "UNKNOWN")
        timestamp = event.get("timestamp", "")
        instance_id = event.get("instance_id", "unknown")
        hostname = event.get("hostname", "unknown")
        model = event.get("model", "N/A")
        backend = event.get("backend", "N/A")

        # Extract time from ISO timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            time_str = dt.strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
        except:
            time_str = timestamp

        # Color for event type
        color = self._get_color_for_event(event_type)

        # Build main line
        lines = []
        lines.append(
            f"{Colors.GRAY}[{time_str}]{Colors.RESET} "
            f"{color}{Colors.BOLD}{event_type:25s}{Colors.RESET} "
            f"{Colors.WHITE}│{Colors.RESET} "
            f"model={Colors.CYAN}{model:20s}{Colors.RESET} "
            f"{Colors.WHITE}│{Colors.RESET} "
            f"backend={Colors.YELLOW}{backend:10s}{Colors.RESET}"
        )

        # Add instance info
        lines.append(
            f"  {Colors.GRAY}├─ instance: {hostname} ({instance_id[:16]}...){Colors.RESET}"
        )

        # Add reason if available
        if "reason" in event:
            lines.append(f"  {Colors.GRAY}├─ reason: {Colors.WHITE}{event['reason']}{Colors.RESET}")

        # Add duration if available
        if "duration" in event:
            duration = event["duration"]
            lines.append(f"  {Colors.GRAY}├─ duration: {Colors.GREEN}{duration:.2f}s{Colors.RESET}")

        # Add coordinator info if available
        if "coordinator_host" in event:
            lines.append(
                f"  {Colors.GRAY}├─ coordinator: {event['coordinator_host']}:{event.get('coordinator_port', 'N/A')}{Colors.RESET}"
            )

        # Add RPC backend info if available
        if "rpc_backends" in event:
            lines.append(f"  {Colors.GRAY}├─ rpc_backends: {event['rpc_backends']}{Colors.RESET}")

        # Add node URL if available
        if "node_url" in event:
            lines.append(f"  {Colors.GRAY}├─ node: {event['node_url']}{Colors.RESET}")

        # Add confidence if available
        if "confidence" in event:
            confidence = event["confidence"]
            lines.append(f"  {Colors.GRAY}├─ confidence: {confidence:.2f}{Colors.RESET}")

        # Add parameter count if available
        if "parameter_count" in event:
            lines.append(f"  {Colors.GRAY}└─ parameters: {event['parameter_count']}B{Colors.RESET}")
        else:
            # Close the box
            lines[-1] = lines[-1].replace("├─", "└─")

        return "\n".join(lines)

    def _display_event(self, event: dict):
        """Display a single event."""
        self.event_count += 1

        if not self._should_display(event):
            self.filtered_count += 1
            return

        formatted = self._format_event(event)
        print(formatted)
        print()  # Blank line between events

    def show_history(self, count: int = 100):
        """Show recent events from Redis stream."""
        print(f"{Colors.CYAN}📜 Fetching last {count} events from history...{Colors.RESET}\n")

        try:
            # Read from stream (oldest to newest)
            messages = self.redis_client.xrevrange(self.stream_key, count=count)

            if not messages:
                print(f"{Colors.YELLOW}No historical events found{Colors.RESET}\n")
                return

            # Reverse to show oldest first
            messages.reverse()

            for msg_id, msg_data in messages:
                event_json = msg_data.get("event", "{}")
                try:
                    event = json.loads(event_json)
                    self._display_event(event)
                except json.JSONDecodeError:
                    continue

            print(f"{Colors.GREEN}✅ Displayed {len(messages)} historical events{Colors.RESET}\n")

        except Exception as e:
            print(f"{Colors.RED}❌ Error reading history: {e}{Colors.RESET}\n")

    def watch_live(self):
        """Watch live routing events."""
        self._print_header()

        try:
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe(self.channel)

            print(f"{Colors.GREEN}🔴 LIVE - Press Ctrl+C to stop{Colors.RESET}\n")

            for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        event = json.loads(message["data"])
                        self._display_event(event)
                    except json.JSONDecodeError:
                        continue

        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}⏸️  Stopped by user{Colors.RESET}")
            print(
                f"{Colors.GRAY}Total events: {self.event_count}, Filtered: {self.filtered_count}{Colors.RESET}\n"
            )
        except Exception as e:
            print(f"\n{Colors.RED}❌ Error: {e}{Colors.RESET}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SOLLOL Routing Log Viewer - Monitor routing decisions across all SOLLOL instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--redis-url",
        default=os.getenv("SOLLOL_REDIS_URL", "redis://localhost:6379"),
        help="Redis URL (default: redis://localhost:6379)",
    )
    parser.add_argument(
        "--model",
        help="Filter by model name",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "rpc", "llamacpp"],
        help="Filter by backend type",
    )
    parser.add_argument(
        "--event-type",
        help="Filter by event type (e.g., ROUTE_DECISION, FALLBACK_TRIGGERED)",
    )
    parser.add_argument(
        "--instance",
        help="Filter by instance ID",
    )
    parser.add_argument(
        "--history",
        type=int,
        metavar="N",
        help="Show last N events from history instead of live stream",
    )

    args = parser.parse_args()

    viewer = RoutingViewer(
        redis_url=args.redis_url,
        filter_model=args.model,
        filter_backend=args.backend,
        filter_event_type=args.event_type,
        filter_instance=args.instance,
    )

    if args.history:
        viewer.show_history(args.history)
    else:
        viewer.watch_live()


if __name__ == "__main__":
    main()
