"""
Log capture hooks for Ray and Dask workers.

This module provides functions that can be called from within Ray/Dask workers
to install the Redis log publisher, ensuring all distributed logs flow to the dashboard.

Usage in Ray:
    import ray
    from sollol.dashboard_log_hooks import install_log_hook_ray

    # Call this in Ray worker initialization
    ray.get(install_log_hook_ray.remote(redis_url="redis://localhost:6379"))

Usage in Dask:
    from dask.distributed import get_worker
    from sollol.dashboard_log_hooks import install_log_hook_dask

    # Call this in Dask worker
    install_log_hook_dask(redis_url="redis://localhost:6379")
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Track if hook already installed (per-process)
_hook_installed = False


def install_log_hook_dask(redis_url: str = "redis://localhost:6379") -> bool:
    """
    Install Redis log publisher in current Dask worker process.

    Args:
        redis_url: Redis connection URL

    Returns:
        True if installed successfully, False otherwise
    """
    global _hook_installed

    if _hook_installed:
        return True

    try:
        import redis

        from .dashboard_service import REDIS_LOG_CHANNEL, REDIS_LOG_STREAM, RedisLogPublisher

        # Create Redis client
        redis_client = redis.from_url(redis_url, decode_responses=True)

        # Create and configure handler
        handler = RedisLogPublisher(
            redis_client=redis_client,
            channel=REDIS_LOG_CHANNEL,
            stream_key=REDIS_LOG_STREAM,
            use_streams=True,
        )
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)

        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        _hook_installed = True
        logger.info(f"📡 [Dask Worker] Redis log publisher installed -> {redis_url}")
        return True

    except Exception as e:
        logger.warning(f"Failed to install Dask log hook: {e}")
        return False


def install_log_hook_main(redis_url: str = "redis://localhost:6379") -> bool:
    """
    Install Redis log publisher in main process.

    Args:
        redis_url: Redis connection URL

    Returns:
        True if installed successfully, False otherwise
    """
    global _hook_installed

    if _hook_installed:
        return True

    try:
        import redis

        from .dashboard_service import REDIS_LOG_CHANNEL, REDIS_LOG_STREAM, RedisLogPublisher

        # Create Redis client
        redis_client = redis.from_url(redis_url, decode_responses=True)

        # Create and configure handler
        handler = RedisLogPublisher(
            redis_client=redis_client,
            channel=REDIS_LOG_CHANNEL,
            stream_key=REDIS_LOG_STREAM,
            use_streams=True,
        )
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)

        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        _hook_installed = True
        logger.info(f"📡 [Main Process] Redis log publisher installed -> {redis_url}")
        return True

    except Exception as e:
        logger.warning(f"Failed to install main process log hook: {e}")
        return False


# Ray remote function for worker initialization
try:
    import ray

    @ray.remote
    def install_log_hook_ray(redis_url: str = "redis://localhost:6379") -> bool:
        """
        Ray remote function to install Redis log publisher in Ray worker.

        Args:
            redis_url: Redis connection URL

        Returns:
            True if installed successfully, False otherwise
        """
        global _hook_installed

        if _hook_installed:
            return True

        try:
            import redis

            from sollol.dashboard_service import (
                REDIS_LOG_CHANNEL,
                REDIS_LOG_STREAM,
                RedisLogPublisher,
            )

            # Create Redis client
            redis_client = redis.from_url(redis_url, decode_responses=True)

            # Create and configure handler
            handler = RedisLogPublisher(
                redis_client=redis_client,
                channel=REDIS_LOG_CHANNEL,
                stream_key=REDIS_LOG_STREAM,
                use_streams=True,
            )
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            handler.setLevel(logging.INFO)

            # Add to root logger
            root_logger = logging.getLogger()
            root_logger.addHandler(handler)

            _hook_installed = True
            logger.info(f"📡 [Ray Worker] Redis log publisher installed -> {redis_url}")
            return True

        except Exception as e:
            logger.warning(f"Failed to install Ray log hook: {e}")
            return False

except ImportError:
    # Ray not available, define placeholder
    install_log_hook_ray = None


def auto_install_hooks(redis_url: str = "redis://localhost:6379", ray_ref=None, dask_client=None):
    """
    Automatically install log hooks across all workers.

    Call this from the main orchestrator after Ray/Dask initialization.

    Args:
        redis_url: Redis connection URL
        ray_ref: Ray instance (if using Ray)
        dask_client: Dask client instance (if using Dask)
    """
    results = {
        "main": False,
        "ray_workers": 0,
        "dask_workers": 0,
    }

    # Install in main process
    results["main"] = install_log_hook_main(redis_url)

    # Install in Ray workers
    if ray_ref and install_log_hook_ray:
        try:
            # Get all Ray workers
            nodes = ray_ref.nodes()
            futures = []
            for node in nodes:
                # Submit hook installation task
                future = install_log_hook_ray.remote(redis_url)
                futures.append(future)

            # Wait for all installations
            ray_results = ray_ref.get(futures)
            results["ray_workers"] = sum(ray_results)
            logger.info(f"✅ Installed log hooks on {results['ray_workers']} Ray workers")
        except Exception as e:
            logger.warning(f"Failed to install Ray hooks: {e}")

    # Install in Dask workers
    if dask_client:
        try:
            # Run hook installation on all Dask workers
            def worker_setup(redis_url):
                from sollol.dashboard_log_hooks import install_log_hook_dask

                return install_log_hook_dask(redis_url)

            futures = dask_client.run(worker_setup, redis_url)
            dask_results = list(futures.values())
            results["dask_workers"] = sum(dask_results)
            logger.info(f"✅ Installed log hooks on {results['dask_workers']} Dask workers")
        except Exception as e:
            logger.warning(f"Failed to install Dask hooks: {e}")

    return results
