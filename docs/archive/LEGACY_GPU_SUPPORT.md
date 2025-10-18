# Legacy GPU Support - Older NVIDIA GPUs

**SOLLOL now supports older NVIDIA GPUs with graceful fallback detection.**

---

## Supported GPU Generations

### ✅ Tier 1: Modern GPUs (Full Support)

**Generation**: Kepler and newer (2012+)

**Models**:
- GTX 600 series and newer (GTX 680, 690, etc.)
- GTX 700 series (GTX 750 Ti, 780, Titan, etc.)
- GTX 900 series (GTX 950, 960, 970, 980, etc.)
- GTX 10 series (GTX 1050 Ti, 1060, 1070, 1080, etc.)
- RTX 20 series (RTX 2060, 2070, 2080, etc.)
- RTX 30 series (RTX 3060, 3070, 3080, 3090, etc.)
- RTX 40 series (RTX 4060, 4070, 4080, 4090, etc.)
- RTX 50 series (RTX 5090 and above)

**Detection Method**: nvidia-smi --query-gpu (full support)

**VRAM Data**: ✅ Accurate real-time VRAM (total, used, free)

**Additional Data**: GPU utilization %, temperature

**Routing**: Full intelligent routing with VRAM-based prioritization

---

### ⚠️ Tier 2: Legacy GPUs (Limited Support)

**Generation**: Pre-Kepler (before 2012)

**Models**:
- GeForce 8 series (8800 GT, 8800 GTX, etc.)
- GeForce 9 series (9800 GTX, etc.)
- GTX 200 series (GTX 260, 280, etc.)
- GTX 400 series (GTX 460, 470, 480, etc.)
- GTX 500 series (GTX 550 Ti, 560, 570, 580, etc.)

**Detection Method**:
1. Basic nvidia-smi parsing (if available)
2. lspci detection (fallback)

**VRAM Data**: ❌ Unknown (returns 0)

**Routing**: Conservative penalty (80-90% score reduction)

**Behavior**:
- GPU is detected and available
- Routing prefers newer GPUs with known VRAM
- Legacy GPU used only as last resort
- Can still run models, but not prioritized

---

## Detection Flow

### Step 1: Try Modern nvidia-smi Query

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
```

**Success** (Kepler and newer):
```
0, NVIDIA GeForce RTX 4090, 24564, 4000, 20564, 45, 62
```

**Returns**:
```python
{
    "vendor": "NVIDIA",
    "gpus": [
        {
            "index": 0,
            "name": "NVIDIA GeForce RTX 4090",
            "total_mb": 24564,
            "used_mb": 4000,
            "free_mb": 20564,
            "utilization_percent": 45,
            "temperature_c": 62,
            "vendor": "NVIDIA"
        }
    ],
    "total_vram_mb": 24564,
    "used_vram_mb": 4000,
    "free_vram_mb": 20564
}
```

**Result**: ✅ Full support with accurate VRAM

---

### Step 2: Fallback to Basic nvidia-smi

**Used when**: --query-gpu not supported (older GPUs/drivers)

```bash
nvidia-smi
```

**Success** (older GPUs):
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 340.108                Driver Version: 340.108                   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 580     Off  | 0000:01:00.0     N/A |                  N/A |
| 40%   62C    P0    N/A /  N/A |    512MiB /  1535MiB |     N/A      Default |
+-------------------------------+----------------------+----------------------+
```

**Returns**:
```python
{
    "vendor": "NVIDIA",
    "gpus": [
        {
            "index": 0,
            "name": "GeForce GTX 580",
            "total_mb": 0,  # Unknown
            "used_mb": 0,   # Unknown
            "free_mb": 0,   # Unknown
            "utilization_percent": None,
            "temperature_c": None,
            "vendor": "NVIDIA",
            "legacy": True
        }
    ],
    "total_vram_mb": 0,
    "used_vram_mb": 0,
    "free_vram_mb": 0,
    "legacy_gpu": True
}
```

**Result**: ⚠️ GPU detected, but VRAM unknown → conservative routing

---

### Step 3: Fallback to lspci

**Used when**: nvidia-smi not available or failed

```bash
lspci | grep NVIDIA
```

**Success**:
```
01:00.0 VGA compatible controller: NVIDIA Corporation GF110 [GeForce GTX 580] (rev a1)
```

**Returns**:
```python
{
    "vendor": "NVIDIA",
    "gpus": [
        {
            "index": 0,
            "name": "GF110 [GeForce GTX 580]",
            "total_mb": 0,
            "used_mb": 0,
            "free_mb": 0,
            "utilization_percent": None,
            "temperature_c": None,
            "vendor": "NVIDIA",
            "legacy": True,
            "lspci_only": True
        }
    ],
    "total_vram_mb": 0,
    "used_vram_mb": 0,
    "free_vram_mb": 0,
    "legacy_gpu": True,
    "lspci_detection": True
}
```

**Result**: ⚠️ GPU presence detected, no driver access → conservative routing

---

## Routing Behavior

### Modern GPU vs Legacy GPU

**Scenario**: Complex generation task requiring GPU

| GPU | Free VRAM | Score | Priority |
|-----|-----------|-------|----------|
| RTX 4090 (modern) | 20000 MB | 128.31 | ✅ Top choice |
| GTX 580 (legacy) | 0 MB (unknown) | 12.83 | ⚠️ Last resort (90% penalty) |

**Result**: System heavily prioritizes modern GPU with known VRAM

---

### Multiple Legacy GPUs

**Scenario**: Only older GPUs available

| GPU | Free VRAM | Score | Priority |
|-----|-----------|-------|----------|
| GTX 580 | 0 MB | 12.83 | Equal |
| GTX 560 Ti | 0 MB | 12.83 | Equal |
| 8800 GTX | 0 MB | 12.83 | Equal |

**Result**: Round-robin or latency-based routing among legacy GPUs

---

### Mixed Legacy + Modern Cluster

**Hardware**:
- Node 1: GTX 580 (legacy, VRAM unknown)
- Node 2: GTX 1050 Ti (modern, 3GB)
- Node 3: RTX 3060 (modern, 12GB)

**Routing for GPU task**:

| Node | VRAM Known | Free VRAM | Score | Priority |
|------|------------|-----------|-------|----------|
| GTX 580 | ❌ No | 0 MB | 12.83 | ❌ Avoid |
| GTX 1050 Ti | ✅ Yes | 2500 MB | 64.15 | ⚠️ Acceptable |
| RTX 3060 | ✅ Yes | 10000 MB | 128.31 | ✅ Preferred |

**Result**: System prioritizes GPUs with known VRAM, avoiding legacy GPU

---

## Why VRAM=0 for Legacy GPUs?

### Design Decision

Setting `free_mb=0` for legacy GPUs (instead of guessing) is intentional:

**✅ Advantages**:
1. **No false confidence**: Don't pretend we know VRAM when we don't
2. **Conservative routing**: Prevents overloading unknown GPUs
3. **Clear signal**: 0 means "unknown", not "empty"
4. **Remote query fallback**: Forces Ollama /api/ps query if remote

**❌ Alternatives considered**:
1. **Guess VRAM by model**: Inaccurate, models have variants
2. **Set to max (e.g., 8GB)**: Risk of OOM if wrong
3. **Parse legacy nvidia-smi output**: Unreliable, format varies

**Conclusion**: VRAM=0 is the safest, most honest approach

---

## Remote Query Fallback

For legacy GPUs on **remote nodes**, SOLLOL queries Ollama directly:

```python
# Query remote Ollama node
GET http://remote-host:11434/api/ps

# Response includes loaded models
{
  "models": [
    {
      "name": "llama3.1:8b",
      "size_vram": 5000000000  # 5GB in VRAM
    }
  ]
}
```

**Calculation**:
```python
# Assume legacy GPU has ~4GB VRAM (conservative estimate)
estimated_total_vram = 4000  # MB
loaded_model_vram = 5000     # MB from /api/ps
free_vram = max(0, estimated_total_vram - loaded_model_vram)
```

**Result**: Even for legacy GPUs, we get **model load state** from Ollama API

---

## User Experience

### What Users See

#### Modern GPU (Full Support)

```python
from sollol import OllamaPool

pool = OllamaPool.auto_configure()
stats = pool.get_stats()

print(stats["vram_monitoring"])
# Output:
{
    "enabled": True,
    "gpu_type": "nvidia",
    "local_gpu": {
        "vendor": "NVIDIA",
        "total_vram_mb": 24564,
        "free_vram_mb": 20564,
        "used_vram_mb": 4000,
        "gpus": [
            {
                "index": 0,
                "name": "NVIDIA GeForce RTX 4090",
                "free_mb": 20564,
                "utilization_percent": 45,
                "temperature_c": 62
            }
        ]
    }
}
```

**Experience**: ✅ Full visibility into GPU state

---

#### Legacy GPU (Limited Support)

```python
from sollol import OllamaPool

pool = OllamaPool.auto_configure()
stats = pool.get_stats()

print(stats["vram_monitoring"])
# Output:
{
    "enabled": True,
    "gpu_type": "nvidia",
    "local_gpu": {
        "vendor": "NVIDIA",
        "total_vram_mb": 0,
        "free_vram_mb": 0,
        "used_vram_mb": 0,
        "legacy_gpu": True,  # ← Flag indicating legacy
        "gpus": [
            {
                "index": 0,
                "name": "GeForce GTX 580",
                "free_mb": 0,
                "legacy": True
            }
        ]
    }
}
```

**Experience**: ⚠️ GPU detected but limited info, conservative routing

---

## Recommendations

### For Legacy GPU Users

1. **Upgrade if possible**: Modern GPUs (GTX 600+) have full support
2. **Use remote Ollama nodes**: Remote query via /api/ps works better
3. **Monitor Ollama directly**: Check Ollama logs for actual VRAM usage
4. **Accept conservative routing**: System will prefer newer GPUs if available

### For Mixed Clusters

1. **Prioritize modern GPUs**: Place compute-heavy workloads on modern GPUs
2. **Use legacy GPUs for light tasks**: Small models, embeddings, etc.
3. **Monitor both**: Check stats for modern + legacy GPU status
4. **Plan upgrades**: Replace legacy GPUs for better performance visibility

---

## Compatibility Matrix

| GPU Series | Release Year | nvidia-smi --query-gpu | Basic nvidia-smi | lspci | VRAM Data |
|------------|--------------|------------------------|------------------|-------|-----------|
| RTX 50 | 2025+ | ✅ | ✅ | ✅ | ✅ Accurate |
| RTX 40 | 2022 | ✅ | ✅ | ✅ | ✅ Accurate |
| RTX 30 | 2020 | ✅ | ✅ | ✅ | ✅ Accurate |
| RTX 20 | 2018 | ✅ | ✅ | ✅ | ✅ Accurate |
| GTX 10 | 2016 | ✅ | ✅ | ✅ | ✅ Accurate |
| GTX 900 | 2014 | ✅ | ✅ | ✅ | ✅ Accurate |
| GTX 700 | 2013 | ✅ | ✅ | ✅ | ✅ Accurate |
| GTX 600 | 2012 | ✅ | ✅ | ✅ | ✅ Accurate |
| **GTX 500** | **2010** | **❌** | **⚠️** | **✅** | **❌ Unknown** |
| **GTX 400** | **2010** | **❌** | **⚠️** | **✅** | **❌ Unknown** |
| **GTX 200** | **2008** | **❌** | **⚠️** | **✅** | **❌ Unknown** |
| **9 series** | **2008** | **❌** | **⚠️** | **✅** | **❌ Unknown** |
| **8 series** | **2006** | **❌** | **⚠️** | **✅** | **❌ Unknown** |

**Cutoff**: Kepler architecture (2012) is the dividing line for full support

---

## Testing

### Test Legacy GPU Detection

```python
from sollol.vram_monitor import VRAMMonitor

monitor = VRAMMonitor()

# Check GPU type
print(f"GPU Type: {monitor.gpu_type}")  # nvidia, amd, intel, or none

# Get GPU info
info = monitor.get_local_vram_info()
if info:
    print(f"Vendor: {info['vendor']}")
    print(f"Legacy GPU: {info.get('legacy_gpu', False)}")
    print(f"Total VRAM: {info['total_vram_mb']} MB")
    print(f"Free VRAM: {info['free_vram_mb']} MB")

    for gpu in info["gpus"]:
        print(f"\nGPU {gpu['index']}: {gpu['name']}")
        if gpu.get("legacy"):
            print("  ⚠️  Legacy GPU - VRAM data unavailable")
        else:
            print(f"  ✅ Free VRAM: {gpu['free_mb']} MB")
```

---

## Summary

**SOLLOL supports older NVIDIA GPUs with graceful degradation**:

1. ✅ **Modern GPUs** (2012+): Full support with accurate VRAM data
2. ⚠️ **Legacy GPUs** (pre-2012): Detected but conservative routing
3. 🔄 **Remote query fallback**: Ollama /api/ps provides model load state
4. 🛡️ **No overload risk**: Conservative penalties prevent GPU overload

**Key Principle**: Never assume - detect what we can, be honest about what we can't, route conservatively.

---

**Status**: ✅ Production Ready
**Tested**: Modern + Legacy GPU scenarios
**Fallback**: 3-tier detection (nvidia-smi query → basic → lspci)
