"""
Dashboard Client - Helper for applications to register with SOLLOL Unified Dashboard

This module provides easy integration for any application using SOLLOL to register
itself with the centralized dashboard for observability.

Usage:
    from sollol import DashboardClient

    # Register your application
    client = DashboardClient(
        app_name="My Application",
        router_type="RayAdvancedRouter",
        dashboard_url="http://localhost:8080"
    )

    # Client automatically sends heartbeats in background
    # When your app exits, client automatically unregisters
"""

import atexit
import logging
import threading
import time
import uuid
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class DashboardClient:
    """
    Client for registering applications with SOLLOL Unified Dashboard.

    Automatically handles:
    - Registration on startup
    - Periodic heartbeats to stay active
    - Unregistration on shutdown
    """

    def __init__(
        self,
        app_name: str,
        router_type: str = "unknown",
        version: str = "unknown",
        dashboard_url: str = "http://localhost:8080",
        heartbeat_interval: int = 10,
        metadata: Optional[Dict[str, Any]] = None,
        auto_register: bool = True,
    ):
        """
        Initialize dashboard client.

        Args:
            app_name: Human-readable application name
            router_type: Type of SOLLOL router being used
            version: Application version
            dashboard_url: URL of unified dashboard
            heartbeat_interval: Seconds between heartbeats
            metadata: Additional metadata to send
            auto_register: Automatically register on init
        """
        self.app_id = str(uuid.uuid4())
        self.app_name = app_name
        self.router_type = router_type
        self.version = version
        self.dashboard_url = dashboard_url.rstrip("/")
        self.heartbeat_interval = heartbeat_interval
        self.metadata = metadata or {}

        self._heartbeat_thread = None
        self._stop_heartbeat = threading.Event()
        self._registered = False

        # Auto-register
        if auto_register:
            self.register()

            # Start heartbeat thread
            self._start_heartbeat()

            # Register cleanup on exit
            atexit.register(self.unregister)

    def register(self) -> bool:
        """
        Register application with dashboard.

        Returns:
            True if registration successful, False otherwise
        """
        try:
            response = requests.post(
                f"{self.dashboard_url}/api/applications/register",
                json={
                    "app_id": self.app_id,
                    "name": self.app_name,
                    "router_type": self.router_type,
                    "version": self.version,
                    "metadata": self.metadata,
                },
                timeout=5,
            )

            if response.ok:
                self._registered = True
                logger.info(
                    f"📱 Registered with dashboard: {self.app_name} "
                    f"({self.router_type}) at {self.dashboard_url}"
                )
                return True
            else:
                logger.warning(f"Failed to register with dashboard: {response.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not connect to dashboard at {self.dashboard_url}: {e}")
            return False

    def heartbeat(self, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send heartbeat to dashboard.

        Args:
            metadata: Optional metadata updates

        Returns:
            True if heartbeat successful, False otherwise
        """
        if not self._registered:
            return False

        try:
            # Include full registration info to support auto-registration on dashboard restart
            payload = {
                "app_id": self.app_id,
                "name": self.app_name,
                "router_type": self.router_type,
                "version": self.version,
                "metadata": metadata if metadata else self.metadata,
            }

            response = requests.post(
                f"{self.dashboard_url}/api/applications/heartbeat",
                json=payload,
                timeout=5,
            )

            if response.ok:
                logger.debug(f"Heartbeat sent for {self.app_name}")
                return True
            else:
                logger.warning(f"Heartbeat failed: {response.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            logger.debug(f"Heartbeat error: {e}")
            return False

    def unregister(self):
        """Unregister application from dashboard."""
        if not self._registered:
            return

        # Stop heartbeat thread
        self._stop_heartbeat.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2)

        # Unregister from dashboard
        try:
            response = requests.post(
                f"{self.dashboard_url}/api/applications/{self.app_id}/unregister",
                timeout=5,
            )

            if response.ok:
                logger.info(f"📱 Unregistered from dashboard: {self.app_name}")
            else:
                logger.warning(f"Failed to unregister: {response.status_code}")

        except requests.exceptions.RequestException as e:
            logger.debug(f"Unregister error: {e}")

        self._registered = False

    def update_metadata(self, metadata: Dict[str, Any]):
        """
        Update application metadata.

        Args:
            metadata: New metadata to merge with existing
        """
        self.metadata.update(metadata)
        self.heartbeat(metadata=metadata)

    def _start_heartbeat(self):
        """Start background heartbeat thread."""
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name=f"DashboardHeartbeat-{self.app_name}",
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self):
        """Background loop that sends periodic heartbeats."""
        while not self._stop_heartbeat.is_set():
            # Sleep in small chunks so we can exit quickly
            for _ in range(self.heartbeat_interval * 2):  # Check every 0.5s
                if self._stop_heartbeat.is_set():
                    return
                time.sleep(0.5)

            # Send heartbeat
            self.heartbeat()
