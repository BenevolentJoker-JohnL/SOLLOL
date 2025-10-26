"""
GPU Reporter Auto-Setup - Automatically configure GPU monitoring

This module automatically:
1. Detects if GPU reporter service is already running
2. Installs gpustat and redis dependencies if needed
3. Installs and starts GPU reporter systemd service
4. Verifies Redis connectivity
"""

import logging
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class GPUAutoSetup:
    """Automatically setup and manage GPU reporter services."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        auto_install: bool = True,
        auto_start: bool = True,
    ):
        """
        Initialize GPU auto-setup.

        Args:
            redis_host: Redis server hostname
            redis_port: Redis server port
            auto_install: Automatically install dependencies if needed
            auto_start: Automatically start GPU reporter service if not running
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.auto_install = auto_install
        self.auto_start = auto_start

    def check_gpu_reporter_running(self) -> bool:
        """Check if GPU reporter service is already running."""
        try:
            # Check systemd user service
            result = subprocess.run(
                ["systemctl", "--user", "is-active", "sollol-gpu-reporter"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip() == "active":
                logger.debug("GPU reporter systemd service is active")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Check if process is running (fallback)
        try:
            result = subprocess.run(
                ["pgrep", "-f", "gpu_reporter.py"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                logger.debug("GPU reporter process is running")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return False

    def check_dependencies_installed(self) -> Dict[str, bool]:
        """Check if required dependencies are installed."""
        deps = {}

        # Check gpustat
        try:
            import gpustat

            deps["gpustat"] = True
            logger.debug("gpustat is installed")
        except ImportError:
            deps["gpustat"] = False
            logger.debug("gpustat not installed")

        # Check redis
        try:
            import redis

            deps["redis"] = True
            logger.debug("redis-py is installed")
        except ImportError:
            deps["redis"] = False
            logger.debug("redis-py not installed")

        return deps

    def install_dependencies(self) -> bool:
        """Install missing dependencies."""
        deps = self.check_dependencies_installed()

        missing = [name for name, installed in deps.items() if not installed]
        if not missing:
            logger.info("✅ All GPU monitoring dependencies already installed")
            return True

        if not self.auto_install:
            logger.warning(
                f"⚠️  Missing dependencies: {', '.join(missing)}. "
                f"Auto-install disabled. Please install manually."
            )
            return False

        logger.info(f"📦 Installing missing dependencies: {', '.join(missing)}")

        try:
            # Install via pip
            for dep in missing:
                logger.info(f"   Installing {dep}...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--user", dep],
                    check=True,
                    capture_output=True,
                    timeout=120,
                )

            logger.info("✅ Dependencies installed successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("❌ Dependency installation timed out")
            return False

    def check_redis_connectivity(self) -> bool:
        """Check if Redis server is accessible."""
        try:
            import redis

            client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            client.ping()
            logger.debug(f"✅ Redis server accessible at {self.redis_host}:{self.redis_port}")
            return True

        except Exception as e:
            logger.warning(
                f"⚠️  Redis server not accessible at {self.redis_host}:{self.redis_port}: {e}"
            )
            return False

    def auto_detect_node_id(self) -> str:
        """Auto-detect node ID (IP:port)."""
        try:
            # Get primary network interface IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            node_id = f"{local_ip}:11434"
            logger.debug(f"Auto-detected node ID: {node_id}")
            return node_id
        except Exception:
            logger.debug("Using fallback node ID: 127.0.0.1:11434")
            return "127.0.0.1:11434"

    def install_service(self, node_id: Optional[str] = None) -> bool:
        """
        Install GPU reporter systemd service.

        Args:
            node_id: Node identifier (auto-detected if not provided)

        Returns:
            True if successful
        """
        if node_id is None:
            node_id = self.auto_detect_node_id()

        logger.info(f"🔧 Installing GPU reporter service for node {node_id}")

        # Find installer script
        sollol_path = Path(__file__).parent.parent.parent
        installer_script = sollol_path / "scripts" / "install-gpu-reporter-service.sh"

        if not installer_script.exists():
            logger.error(f"❌ Installer script not found at: {installer_script}")
            return False

        try:
            # Make executable
            installer_script.chmod(0o755)

            # Run installer with non-interactive mode
            env = {
                **os.environ,
                "REDIS_HOST": self.redis_host,
                "REDIS_PORT": str(self.redis_port),
                "NODE_ID": node_id,
                "REPORT_INTERVAL": "5",
            }

            # Create user systemd directory
            systemd_dir = Path.home() / ".config" / "systemd" / "user"
            systemd_dir.mkdir(parents=True, exist_ok=True)

            # Copy and configure service file
            service_template = sollol_path / "systemd" / "sollol-gpu-reporter.service"
            service_dest = systemd_dir / "sollol-gpu-reporter.service"

            if not service_template.exists():
                logger.error(f"❌ Service template not found at: {service_template}")
                return False

            # Read template and replace placeholders
            with open(service_template, "r") as f:
                service_content = f.read()

            service_content = service_content.replace("%h", str(Path.home()))
            service_content = service_content.replace("%u", os.getenv("USER", "user"))
            service_content = service_content.replace("%REDIS_HOST%", self.redis_host)
            service_content = service_content.replace("%REDIS_PORT%", str(self.redis_port))
            service_content = service_content.replace("%NODE_ID%", node_id)
            service_content = service_content.replace("%REPORT_INTERVAL%", "5")

            # Write configured service file
            with open(service_dest, "w") as f:
                f.write(service_content)

            logger.info(f"   Service file: {service_dest}")

            # Reload systemd
            subprocess.run(
                ["systemctl", "--user", "daemon-reload"],
                check=True,
                timeout=10,
            )

            # Enable and start service
            subprocess.run(
                ["systemctl", "--user", "enable", "sollol-gpu-reporter.service"],
                check=True,
                timeout=10,
            )

            subprocess.run(
                ["systemctl", "--user", "start", "sollol-gpu-reporter.service"],
                check=True,
                timeout=10,
            )

            # Try to enable lingering (may require sudo, fail gracefully)
            try:
                subprocess.run(
                    ["loginctl", "enable-linger", os.getenv("USER")],
                    capture_output=True,
                    timeout=5,
                )
            except Exception:
                logger.debug("Could not enable lingering (may require sudo)")

            logger.info("✅ GPU reporter service installed and started")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install service: {e}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("❌ Service installation timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error during installation: {e}")
            return False

    def setup(self, node_id: Optional[str] = None) -> bool:
        """
        Perform complete GPU monitoring setup.

        Args:
            node_id: Node identifier (auto-detected if not provided)

        Returns:
            True if GPU monitoring is set up and running
        """
        logger.info("=" * 70)
        logger.info("🔧 GPU Monitoring Auto-Setup")
        logger.info("=" * 70)

        # Step 1: Check if already running
        if self.check_gpu_reporter_running():
            logger.info("✅ GPU reporter already running")
            return True

        # Step 2: Install dependencies
        logger.info("📦 Checking dependencies...")
        if not self.install_dependencies():
            logger.error("❌ Failed to install dependencies")
            return False

        # Step 3: Check Redis connectivity
        logger.info(f"🔌 Checking Redis connectivity ({self.redis_host}:{self.redis_port})...")
        redis_available = self.check_redis_connectivity()
        if not redis_available:
            logger.warning("⚠️  Redis not accessible. GPU monitoring will not work without Redis.")
            logger.warning(
                f"   Please ensure Redis is running at {self.redis_host}:{self.redis_port}"
            )
            # Continue anyway - service will retry connection

        # Step 4: Install and start service
        if not self.auto_start:
            logger.warning("Auto-start disabled. GPU reporter service not started.")
            return False

        if node_id is None:
            node_id = self.auto_detect_node_id()

        logger.info(f"🚀 Installing GPU reporter for node: {node_id}")

        if not self.install_service(node_id):
            logger.error("❌ Failed to install GPU reporter service")
            return False

        # Step 5: Verify service is running
        import time

        time.sleep(2)  # Give service time to start

        if self.check_gpu_reporter_running():
            logger.info("=" * 70)
            logger.info("✅ GPU Monitoring Setup Complete!")
            logger.info("=" * 70)
            logger.info("")
            logger.info("Useful commands:")
            logger.info("  systemctl --user status sollol-gpu-reporter")
            logger.info("  systemctl --user restart sollol-gpu-reporter")
            logger.info("  journalctl --user -u sollol-gpu-reporter -f")
            logger.info("")
            return True
        else:
            logger.warning("⚠️  Service installed but not running. Check logs:")
            logger.warning("  journalctl --user -u sollol-gpu-reporter -n 50")
            return False


def auto_setup_gpu_monitoring(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    node_id: Optional[str] = None,
    auto_install: bool = True,
    auto_start: bool = True,
) -> bool:
    """
    Convenience function for automatic GPU monitoring setup.

    Args:
        redis_host: Redis server hostname
        redis_port: Redis server port
        node_id: Node identifier (auto-detected if not provided)
        auto_install: Automatically install dependencies
        auto_start: Automatically start service

    Returns:
        True if GPU monitoring is set up and running

    Example:
        >>> from sollol.gpu_auto_setup import auto_setup_gpu_monitoring
        >>> auto_setup_gpu_monitoring(redis_host="10.9.66.154")
        ✅ GPU Monitoring Setup Complete!
    """
    setup = GPUAutoSetup(
        redis_host=redis_host,
        redis_port=redis_port,
        auto_install=auto_install,
        auto_start=auto_start,
    )
    return setup.setup(node_id=node_id)
