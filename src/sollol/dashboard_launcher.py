"""
Subprocess launcher for the standalone SOLLOL dashboard service.

Provides utilities to start/stop the dashboard as an independent process,
replacing the old daemon thread approach with proper process isolation.
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DashboardProcessLauncher:
    """
    Launches and manages the SOLLOL dashboard as a standalone subprocess.

    This provides proper process isolation and ensures the dashboard can
    independently aggregate logs and activity from distributed workers via Redis.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        port: int = 8080,
        ray_dashboard_port: int = 8265,
        dask_dashboard_port: int = 8787,
        enable_dask: bool = True,
        debug: bool = False,
    ):
        self.redis_url = redis_url
        self.port = port
        self.ray_dashboard_port = ray_dashboard_port
        self.dask_dashboard_port = dask_dashboard_port
        self.enable_dask = enable_dask
        self.debug = debug
        self.process: Optional[subprocess.Popen] = None
        self.log_file = None

    def start(self, background: bool = True) -> bool:
        """
        Start the dashboard service as a subprocess.

        Args:
            background: If True, run in background. If False, blocks until dashboard exits.

        Returns:
            True if started successfully, False otherwise
        """
        if self.process and self.process.poll() is None:
            logger.warning("Dashboard process already running")
            return False

        try:
            # Build command to run dashboard service
            cmd = [
                sys.executable,  # Use same Python interpreter
                "-m",
                "sollol.dashboard_service",
                "--port",
                str(self.port),
                "--redis-url",
                self.redis_url,
                "--ray-dashboard-port",
                str(self.ray_dashboard_port),
                "--dask-dashboard-port",
                str(self.dask_dashboard_port),
            ]

            if not self.enable_dask:
                cmd.append("--no-dask")

            if self.debug:
                cmd.append("--debug")

            # Setup logging
            log_dir = Path.home() / ".sollol" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = log_dir / f"dashboard_{self.port}.log"
            self.log_file = open(log_file_path, "a")

            msg = f"🚀 Launching SOLLOL Dashboard as subprocess on port {self.port}..."
            logger.info(msg)
            print(msg)

            # Start subprocess
            if background:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=self.log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,  # Detach from parent
                )

                # Wait a moment and check if it started successfully
                time.sleep(2)
                if self.process.poll() is not None:
                    # Process already exited
                    logger.error(
                        f"Dashboard process exited immediately (check log: {log_file_path})"
                    )
                    return False

                msg1 = f"✅ SOLLOL Dashboard started at http://0.0.0.0:{self.port}"
                msg2 = f"   📊 Features: Real-time logs, Activity monitoring, Ray/Dask dashboards"
                msg3 = f"   📡 Using Redis at {self.redis_url}"
                msg4 = f"   📝 Logs: {log_file_path}"
                msg5 = f"   🔧 PID: {self.process.pid}"

                logger.info(msg1)
                logger.info(msg2)
                logger.info(msg3)
                logger.info(msg4)
                logger.info(msg5)
                print(msg1)
                print(msg2)
                print(msg3)
                print(msg4)
                print(msg5)

                return True
            else:
                # Foreground mode (blocking)
                result = subprocess.run(cmd, stdout=self.log_file, stderr=subprocess.STDOUT)
                return result.returncode == 0

        except Exception as e:
            err_msg = f"⚠️  Failed to start dashboard subprocess: {e}"
            logger.error(err_msg)
            print(err_msg)
            return False

    def stop(self) -> bool:
        """
        Stop the dashboard subprocess.

        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.process:
            logger.warning("No dashboard process to stop")
            return False

        try:
            if self.process.poll() is None:
                # Process is still running
                logger.info(f"Stopping dashboard process (PID: {self.process.pid})")
                self.process.terminate()

                # Wait up to 5 seconds for graceful shutdown
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't stop
                    logger.warning("Dashboard didn't stop gracefully, forcing kill")
                    self.process.kill()
                    self.process.wait()

            if self.log_file:
                self.log_file.close()
                self.log_file = None

            logger.info("✅ Dashboard process stopped")
            return True

        except Exception as e:
            logger.error(f"Error stopping dashboard: {e}")
            return False

    def is_running(self) -> bool:
        """Check if dashboard process is running."""
        return self.process is not None and self.process.poll() is None

    def get_pid(self) -> Optional[int]:
        """Get the PID of the dashboard process."""
        return self.process.pid if self.process else None

    def __del__(self):
        """Cleanup on deletion."""
        if self.log_file:
            try:
                self.log_file.close()
            except Exception:
                pass


def launch_dashboard_subprocess(
    redis_url: str = "redis://localhost:6379",
    port: int = 8080,
    ray_dashboard_port: int = 8265,
    dask_dashboard_port: int = 8787,
    enable_dask: bool = True,
    debug: bool = False,
) -> DashboardProcessLauncher:
    """
    Convenience function to launch dashboard subprocess.

    Returns:
        DashboardProcessLauncher instance (allows stopping later)
    """
    launcher = DashboardProcessLauncher(
        redis_url=redis_url,
        port=port,
        ray_dashboard_port=ray_dashboard_port,
        dask_dashboard_port=dask_dashboard_port,
        enable_dask=enable_dask,
        debug=debug,
    )

    launcher.start(background=True)
    return launcher


if __name__ == "__main__":
    # Test launcher
    import argparse

    parser = argparse.ArgumentParser(description="Launch SOLLOL Dashboard Subprocess")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--redis-url", default="redis://localhost:6379")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    launcher = launch_dashboard_subprocess(
        redis_url=args.redis_url,
        port=args.port,
        debug=args.debug,
    )

    print(f"\nDashboard running at http://localhost:{args.port}")
    print("Press Ctrl+C to stop...\n")

    try:
        # Keep running until interrupted
        while launcher.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
        launcher.stop()
