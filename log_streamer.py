#!/usr/bin/env python3
"""
SOLLOL Log Streamer
Tails coordinator logs and publishes to Redis for dashboard visibility.
"""
import sys
import time
import redis
from pathlib import Path

def stream_logs(log_file: str, redis_url: str, channel: str):
    """
    Tail log file and publish each line to Redis channel.

    Args:
        log_file: Path to coordinator log file
        redis_url: Redis connection URL
        channel: Redis pub/sub channel name
    """
    # Connect to Redis
    r = redis.from_url(redis_url)

    # Open log file
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"❌ Log file not found: {log_file}", file=sys.stderr)
        sys.exit(1)

    print(f"📡 Streaming logs from {log_file} to Redis channel '{channel}'")

    # Seek to end of file (only stream new logs)
    with open(log_path, 'r') as f:
        # Go to end of file
        f.seek(0, 2)  # SEEK_END

        while True:
            line = f.readline()

            if line:
                # Strip newline but keep content
                line = line.rstrip('\n')

                # Only publish non-empty lines
                if line.strip():
                    try:
                        # Publish to Redis
                        r.publish(channel, line)
                        print(f"📤 {line[:80]}...", file=sys.stderr)
                    except Exception as e:
                        print(f"❌ Redis publish failed: {e}", file=sys.stderr)
            else:
                # No new line, wait a bit
                time.sleep(0.1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: log_streamer.py <log_file> [redis_url] [channel]")
        print("Example: log_streamer.py /tmp/coordinator-18080.log redis://localhost:6379 sollol:logs:llama_cpp")
        sys.exit(1)

    log_file = sys.argv[1]
    redis_url = sys.argv[2] if len(sys.argv) > 2 else "redis://localhost:6379"
    channel = sys.argv[3] if len(sys.argv) > 3 else "sollol:logs:llama_cpp"

    try:
        stream_logs(log_file, redis_url, channel)
    except KeyboardInterrupt:
        print("\n⏹️  Log streamer stopped")
    except Exception as e:
        print(f"❌ Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
