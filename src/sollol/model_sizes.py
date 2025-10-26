"""
Model Size Database for SOLLOL
Pre-calculated model sizes for instant VRAM-aware routing decisions.

Ported from FlockParser and expanded for SOLLOL's use cases.

Features:
- Static database for common models (fast lookup)
- Dynamic discovery from Ollama API (accurate, cached)
- Automatic cache updates when models are queried
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

# Cache file for discovered model sizes
# Format: {
#   "global": {"model_name": size_mb},  # Generic sizes
#   "nodes": {"node_url": {"model_name": size_mb}}  # Node-specific sizes
# }
CACHE_FILE = Path.home() / ".sollol" / "model_sizes_cache.json"


# Model size database in MB
# Based on actual VRAM measurements and Ollama model specifications
MODEL_SIZE_DB: Dict[str, int] = {
    # Embedding models (small)
    "mxbai-embed-large": 705,
    "nomic-embed-text": 274,
    "all-minilm": 45,
    "bge-large": 1340,
    "bge-small": 133,
    # Llama 3.x models
    "llama3.1:8b": 4700,
    "llama3.1:70b": 40000,
    "llama3.1:latest": 4700,
    "llama3.2:1b": 1300,
    "llama3.2:3b": 1900,
    "llama3.2:latest": 1900,
    "llama3:8b": 4700,
    "llama3:70b": 40000,
    # Qwen models
    "qwen2.5-coder:0.5b": 500,
    "qwen2.5-coder:1.5b": 900,
    "qwen2.5-coder:3b": 1800,
    "qwen2.5-coder:7b": 4400,
    "qwen2.5-coder:14b": 8800,
    "qwen2.5-coder:32b": 20000,
    "qwen2.5:0.5b": 500,
    "qwen2.5:1.5b": 900,
    "qwen2.5:3b": 1800,
    "qwen2.5:7b": 4400,
    "qwen2.5:14b": 8800,
    "qwen2.5:32b": 20000,
    "qwen2.5:72b": 45000,
    # CodeLlama models
    "codellama:7b": 3600,
    "codellama:13b": 6900,
    "codellama:34b": 19000,
    "codellama:70b": 40000,
    # Mistral models
    "mistral:7b": 4100,
    "mistral:latest": 4100,
    "mixtral:8x7b": 26000,
    "mixtral:8x22b": 80000,
    # Gemma models
    "gemma:2b": 1400,
    "gemma:7b": 4900,
    "gemma2:2b": 1600,
    "gemma2:9b": 5400,
    "gemma2:27b": 16000,
    # DeepSeek models
    "deepseek-coder:1.3b": 1000,
    "deepseek-coder:6.7b": 3900,
    "deepseek-coder:33b": 19000,
    # Phi models
    "phi3:mini": 2300,
    "phi3:medium": 7600,
    "phi3.5:mini": 2400,
    # Yi models
    "yi:6b": 3500,
    "yi:9b": 5200,
    "yi:34b": 19000,
    # Dolphin variants
    "dolphin-llama3:8b": 4700,
    "dolphin-mixtral:8x7b": 26000,
    # Neural Chat
    "neural-chat:7b": 4100,
    # Orca models
    "orca-mini:3b": 1900,
    "orca-mini:7b": 4100,
    "orca2:7b": 4100,
    "orca2:13b": 7400,
    # Vicuna
    "vicuna:7b": 4100,
    "vicuna:13b": 7400,
    # WizardLM
    "wizardlm2:7b": 4100,
    "wizardlm2:8x22b": 80000,
    # StarCoder
    "starcoder2:3b": 1800,
    "starcoder2:7b": 4100,
    "starcoder2:15b": 8500,
}


def _load_cache() -> Dict:
    """
    Load cached model sizes from disk.

    Returns dict with structure:
    {
        "global": {"model_name": size_mb},
        "nodes": {"node_url": {"model_name": size_mb}}
    }
    """
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)

                # Migrate old format if needed
                if "global" not in cache and "nodes" not in cache:
                    logger.debug("Migrating old cache format to new structure")
                    cache = {"global": cache, "nodes": {}}

                global_count = len(cache.get("global", {}))
                node_count = sum(len(models) for models in cache.get("nodes", {}).values())
                logger.debug(
                    f"Loaded cache: {global_count} global, "
                    f"{node_count} node-specific sizes from {CACHE_FILE}"
                )
                return cache
    except Exception as e:
        logger.debug(f"Could not load cache: {e}")

    return {"global": {}, "nodes": {}}


def _save_cache(cache: Dict):
    """Save cached model sizes to disk."""
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)

        global_count = len(cache.get("global", {}))
        node_count = sum(len(models) for models in cache.get("nodes", {}).values())
        logger.debug(f"Saved cache: {global_count} global, {node_count} node-specific sizes")
    except Exception as e:
        logger.warning(f"Could not save cache: {e}")


# In-memory cache for discovered sizes
_size_cache = _load_cache()


def query_ollama_model_size(
    model_name: str, ollama_url: str = "http://localhost:11434", cache_per_node: bool = True
) -> Optional[int]:
    """
    Query Ollama API to get actual model size and cache it.

    IMPORTANT: Caches per-node because the same model name can have
    different quantizations on different nodes (e.g., Q4 vs Q8).

    Args:
        model_name: Model name
        ollama_url: Ollama instance URL
        cache_per_node: Cache size per-node (recommended: True)

    Returns:
        Size in MB or None if query failed
    """
    try:
        response = requests.post(f"{ollama_url}/api/show", json={"name": model_name}, timeout=5)

        if response.status_code == 200:
            data = response.json()

            # Try to get size from model info
            size_bytes = data.get("size", 0)
            if size_bytes > 0:
                size_mb = int(size_bytes / (1024**2))
                logger.info(f"📊 Discovered model size: {model_name} @ {ollama_url} = {size_mb}MB")

                # Cache the discovered size (node-specific or global)
                if cache_per_node:
                    # Store per-node to handle different quantizations
                    if "nodes" not in _size_cache:
                        _size_cache["nodes"] = {}
                    if ollama_url not in _size_cache["nodes"]:
                        _size_cache["nodes"][ollama_url] = {}

                    _size_cache["nodes"][ollama_url][model_name.lower()] = size_mb
                else:
                    # Store globally (less accurate for mixed quantizations)
                    if "global" not in _size_cache:
                        _size_cache["global"] = {}
                    _size_cache["global"][model_name.lower()] = size_mb

                _save_cache(_size_cache)
                return size_mb

            # Try to get from details/parameters
            details = data.get("details", {})
            parameter_size = details.get("parameter_size", "")

            # Parse parameter size like "7B", "13B", etc.
            if parameter_size:
                import re

                param_match = re.search(r"(\d+\.?\d*)B", parameter_size.upper())
                if param_match:
                    param_count = float(param_match.group(1))
                    # Estimate: 1B params ≈ 600MB in FP16
                    estimated_mb = int(param_count * 600)
                    logger.info(
                        f"📊 Estimated from parameters: {model_name} @ {ollama_url} = {estimated_mb}MB "
                        f"({parameter_size})"
                    )

                    # Cache the estimate
                    if cache_per_node:
                        if "nodes" not in _size_cache:
                            _size_cache["nodes"] = {}
                        if ollama_url not in _size_cache["nodes"]:
                            _size_cache["nodes"][ollama_url] = {}
                        _size_cache["nodes"][ollama_url][model_name.lower()] = estimated_mb
                    else:
                        if "global" not in _size_cache:
                            _size_cache["global"] = {}
                        _size_cache["global"][model_name.lower()] = estimated_mb

                    _save_cache(_size_cache)
                    return estimated_mb

    except requests.exceptions.Timeout:
        logger.debug(f"Timeout querying Ollama at {ollama_url} for {model_name}")
    except Exception as e:
        logger.debug(f"Error querying Ollama at {ollama_url} for {model_name}: {e}")

    return None


def get_model_size(
    model_name: str, ollama_url: Optional[str] = None, use_ollama_discovery: bool = True
) -> Optional[int]:
    """
    Get model size in MB from database, cache, or Ollama API.

    Lookup order:
    1. Node-specific cache (if ollama_url provided) - most accurate
    2. Global cache (from previous queries)
    3. Static database (MODEL_SIZE_DB)
    4. Ollama API query (if enabled and URL provided)

    Args:
        model_name: Model name (e.g., "llama3.1:8b", "mxbai-embed-large")
        ollama_url: Optional Ollama URL for node-specific lookup and discovery
        use_ollama_discovery: Whether to query Ollama if not in cache/DB

    Returns:
        Size in MB or None if unknown
    """
    # Normalize model name
    normalized = model_name.lower().strip()
    normalized_no_latest = normalized.replace(":latest", "")

    # 1a. Check node-specific cache first (most accurate for that node)
    if ollama_url and "nodes" in _size_cache:
        node_cache = _size_cache["nodes"].get(ollama_url, {})
        if normalized in node_cache:
            logger.debug(f"Node-specific cache hit: {model_name} @ {ollama_url}")
            return node_cache[normalized]
        if normalized_no_latest in node_cache:
            return node_cache[normalized_no_latest]

    # 1b. Check global cache
    if "global" in _size_cache:
        if normalized in _size_cache["global"]:
            return _size_cache["global"][normalized]
        if normalized_no_latest in _size_cache["global"]:
            return _size_cache["global"][normalized_no_latest]

    # 2. Check static database
    if normalized in MODEL_SIZE_DB:
        return MODEL_SIZE_DB[normalized]
    if normalized_no_latest in MODEL_SIZE_DB:
        return MODEL_SIZE_DB[normalized_no_latest]

    # Fuzzy matching in static DB
    for known_model, size in MODEL_SIZE_DB.items():
        if known_model in normalized or normalized_no_latest in known_model:
            logger.debug(f"Fuzzy matched '{model_name}' to '{known_model}' ({size}MB)")
            return size

    # 3. Try Ollama API discovery (if enabled)
    if use_ollama_discovery and ollama_url:
        discovered_size = query_ollama_model_size(model_name, ollama_url, cache_per_node=True)
        if discovered_size:
            return discovered_size

    # Unknown model
    logger.debug(f"Model size unknown for '{model_name}'")
    return None


def estimate_model_size(
    model_name: str, default_mb: int = 5000, ollama_url: Optional[str] = None
) -> int:
    """
    Estimate model size with fallback to conservative default.

    Tries in order:
    1. Cached/database lookup
    2. Ollama API query (if URL provided)
    3. Parameter extraction from name
    4. Conservative default

    Args:
        model_name: Model name
        default_mb: Default size if unknown (default: 5GB)
        ollama_url: Optional Ollama URL for dynamic discovery

    Returns:
        Estimated size in MB
    """
    # Try cached/database lookup with optional Ollama discovery
    size = get_model_size(model_name, ollama_url=ollama_url, use_ollama_discovery=True)

    if size is not None:
        return size

    # Extract parameter count from name if possible
    # e.g., "custom-model:7b" -> assume 7B params ≈ 4GB
    import re

    param_match = re.search(r"(\d+\.?\d*)b", model_name.lower())

    if param_match:
        param_count = float(param_match.group(1))
        # Rough estimate: 1B params ≈ 600MB in FP16
        estimated_size = int(param_count * 600)
        logger.info(f"Estimated {model_name} size: {estimated_size}MB (from {param_count}B params)")

        # Cache the estimate
        _size_cache[model_name.lower()] = estimated_size
        _save_cache(_size_cache)

        return estimated_size

    # Conservative fallback
    logger.warning(f"Unknown model '{model_name}', using conservative estimate: {default_mb}MB")
    return default_mb


def can_fit_in_vram(model_name: str, available_vram_mb: int, safety_margin: float = 0.8) -> bool:
    """
    Check if model can fit in available VRAM with safety margin.

    Args:
        model_name: Model name
        available_vram_mb: Available VRAM in MB
        safety_margin: Use only this fraction of VRAM (default: 0.8 = 80%)

    Returns:
        True if model fits, False otherwise
    """
    model_size = estimate_model_size(model_name)
    usable_vram = int(available_vram_mb * safety_margin)

    fits = model_size <= usable_vram

    logger.debug(
        f"VRAM check: {model_name} ({model_size}MB) "
        f"{'fits in' if fits else 'too large for'} "
        f"{usable_vram}MB usable ({available_vram_mb}MB total)"
    )

    return fits


def add_model_size(model_name: str, size_mb: int, node_url: Optional[str] = None):
    """
    Add a new model size to the cache (runtime and persistent).

    Args:
        model_name: Model name
        size_mb: Size in MB
        node_url: Optional node URL for node-specific caching
    """
    if node_url:
        # Add to node-specific cache
        if "nodes" not in _size_cache:
            _size_cache["nodes"] = {}
        if node_url not in _size_cache["nodes"]:
            _size_cache["nodes"][node_url] = {}

        _size_cache["nodes"][node_url][model_name.lower()] = size_mb
        logger.info(f"Added model size to node cache: {model_name} @ {node_url} = {size_mb}MB")
    else:
        # Add to global cache
        if "global" not in _size_cache:
            _size_cache["global"] = {}

        _size_cache["global"][model_name.lower()] = size_mb
        logger.info(f"Added model size to global cache: {model_name} = {size_mb}MB")

    _save_cache(_size_cache)


def discover_model_sizes_from_nodes(registry, model_names: Optional[list] = None):
    """
    Discover model sizes from available nodes in the registry.

    This queries each node's Ollama instance to get actual model sizes
    and caches them for future use.

    Args:
        registry: NodeRegistry instance
        model_names: Optional list of model names to discover (discovers all if None)
    """
    if not registry:
        logger.warning("No registry provided for model size discovery")
        return

    discovered_count = 0
    nodes = list(registry.nodes.values())

    logger.info(f"🔍 Discovering model sizes from {len(nodes)} nodes...")

    for node in nodes:
        try:
            node_url = f"http://{node.host}:{node.port}"

            # Get list of models on this node
            response = requests.get(f"{node_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                available_models = data.get("models", [])

                for model_info in available_models:
                    model_name = model_info.get("name", "")

                    # Skip if we have a specific list and this isn't in it
                    if model_names and model_name not in model_names:
                        continue

                    # Skip if already cached
                    if model_name.lower() in _size_cache:
                        continue

                    # Query for this model's size
                    discovered_size = query_ollama_model_size(model_name, node_url)
                    if discovered_size:
                        discovered_count += 1

        except Exception as e:
            logger.debug(f"Could not discover models from {node.host}:{node.port}: {e}")

    logger.info(f"✅ Discovered {discovered_count} new model sizes")
    return discovered_count


def get_database_stats() -> Dict:
    """
    Get statistics about the model size database.

    Returns:
        Dict with database statistics
    """
    sizes = list(MODEL_SIZE_DB.values())

    return {
        "total_models": len(MODEL_SIZE_DB),
        "min_size_mb": min(sizes) if sizes else 0,
        "max_size_mb": max(sizes) if sizes else 0,
        "avg_size_mb": int(sum(sizes) / len(sizes)) if sizes else 0,
        "models_under_1gb": sum(1 for s in sizes if s < 1024),
        "models_under_5gb": sum(1 for s in sizes if s < 5120),
        "models_over_20gb": sum(1 for s in sizes if s > 20480),
    }


def print_model_database():
    """Print the entire model size database."""
    print("\n" + "=" * 80)
    print("📊 SOLLOL MODEL SIZE DATABASE")
    print("=" * 80)

    stats = get_database_stats()
    print(f"\nTotal Models: {stats['total_models']}")
    print(f"Size Range: {stats['min_size_mb']}MB - {stats['max_size_mb']}MB")
    print(f"Average Size: {stats['avg_size_mb']}MB")
    print(f"\nModels < 1GB: {stats['models_under_1gb']}")
    print(f"Models < 5GB: {stats['models_under_5gb']}")
    print(f"Models > 20GB: {stats['models_over_20gb']}")

    print("\n" + "-" * 80)
    print("Model Sizes (sorted by size):")
    print("-" * 80)

    sorted_models = sorted(MODEL_SIZE_DB.items(), key=lambda x: x[1])

    for model, size_mb in sorted_models:
        size_gb = size_mb / 1024
        print(f"  {model:40s} {size_mb:8d} MB ({size_gb:6.2f} GB)")

    print("=" * 80 + "\n")


__all__ = [
    "MODEL_SIZE_DB",
    "get_model_size",
    "estimate_model_size",
    "can_fit_in_vram",
    "add_model_size",
    "query_ollama_model_size",
    "discover_model_sizes_from_nodes",
    "get_database_stats",
    "print_model_database",
]
