# FlockParser Remote Access - Setup Guide

## Current Status

✅ **Local integration verified** - SynapticLlamas can access FlockParser on the same machine
✅ **Documentation complete** - 4 remote access methods documented
✅ **Test scripts created** - Verification tools available
⏳ **Remote access** - Ready to implement (awaiting machine details)

## Quick Test Results

```bash
$ python3 verify_flockparser_access.py
✅ FlockParser directory exists
✅ Knowledge base exists with 1004 chunk files
✅ Document index exists (2 documents, 11 chunks)
✅ Embedding dimension: 1024

$ python3 test_local_rag_integration.py
✅ Local integration test PASSED
✅ Adapter can initialize
✅ Document index is readable
✅ Chunk files are accessible
```

## Files in This Repository

### Documentation
- `FLOCKPARSER_REMOTE_ACCESS.md` - Complete guide (4 methods)
- `SYNAPTICLLAMAS_FLOCKPARSER_INTERFACE.md` - Architecture overview
- `SYNAPTICLLAMAS_RAG_INTEGRATION.md` - Technical data flow
- `SYNAPTICLLAMAS_RAG_COMMANDS.md` - User command reference
- `FLOCKPARSER_INTEGRATION_STATUS.md` - Implementation status

### Test Scripts
- `verify_flockparser_access.py` - Verify FlockParser file access
- `test_local_rag_integration.py` - Test SynapticLlamas adapter

## Recommended: NFS Setup

### Why NFS?
- ✅ **Transparent** - Zero code changes to SynapticLlamas
- ✅ **Fast** - Direct filesystem access
- ✅ **Simple** - Works with existing adapter
- ✅ **Read-only** - Safe for FlockParser
- ✅ **Automatic** - Mount once, use forever

### Setup Steps

#### On FlockParser Machine (Server)

1. **Install NFS server:**
   ```bash
   sudo apt update
   sudo apt install nfs-kernel-server
   ```

2. **Export FlockParser directory:**
   ```bash
   # Add export entry
   echo "/home/joker/FlockParser <client-ip>/24(ro,sync,no_subtree_check)" | sudo tee -a /etc/exports

   # Example for subnet 10.9.66.0/24:
   echo "/home/joker/FlockParser 10.9.66.0/24(ro,sync,no_subtree_check)" | sudo tee -a /etc/exports
   ```

3. **Apply and verify:**
   ```bash
   sudo exportfs -ra
   sudo exportfs -v
   ```

4. **Check NFS is running:**
   ```bash
   sudo systemctl status nfs-server
   ```

#### On SynapticLlamas Machine (Client)

1. **Install NFS client:**
   ```bash
   sudo apt update
   sudo apt install nfs-common
   ```

2. **Create mount point:**
   ```bash
   sudo mkdir -p /mnt/flockparser
   ```

3. **Mount FlockParser:**
   ```bash
   sudo mount <flockparser-ip>:/home/joker/FlockParser /mnt/flockparser

   # Example:
   sudo mount 192.168.1.21:/home/joker/FlockParser /mnt/flockparser
   ```

4. **Verify mount:**
   ```bash
   ls -la /mnt/flockparser/knowledge_base/
   cat /mnt/flockparser/document_index.json | jq '.documents | length'
   ```

5. **Make permanent (optional):**
   ```bash
   # Add to /etc/fstab
   echo "<flockparser-ip>:/home/joker/FlockParser /mnt/flockparser nfs ro,defaults 0 0" | sudo tee -a /etc/fstab

   # Example:
   echo "192.168.1.21:/home/joker/FlockParser /mnt/flockparser nfs ro,defaults 0 0" | sudo tee -a /etc/fstab
   ```

#### Update SynapticLlamas Configuration

**Zero code changes needed!** Just update the path:

```python
# In main.py or config
adapter = FlockParserAdapter(
    flockparser_path="/mnt/flockparser",  # Changed from /home/joker/FlockParser
    embedding_model="mxbai-embed-large",
    hybrid_router_sync=hybrid_router
)
```

#### Test Remote Access

```bash
# On SynapticLlamas machine:
cd /home/joker/SOLLOL
python3 verify_flockparser_access.py

# Should show:
# ✅ FlockParser directory exists: /mnt/flockparser
# ✅ Knowledge base exists with 1004 chunk files
# ✅ Document index exists (2 documents, 11 chunks)
```

## Alternative: HTTP API

If NFS is not available (firewall restrictions, etc.), use HTTP API:

### Create FlockParser API Server

**File:** `/home/joker/FlockParser/flockparser_remote_api.py`

```python
from fastapi import FastAPI, HTTPException
from pathlib import Path
import json

app = FastAPI(title="FlockParser Remote API")

FLOCKPARSER_ROOT = Path("/home/joker/FlockParser")
DOCUMENT_INDEX = FLOCKPARSER_ROOT / "document_index.json"
KNOWLEDGE_BASE = FLOCKPARSER_ROOT / "knowledge_base"

@app.get("/api/document_index")
async def get_document_index():
    """Get complete document index."""
    if not DOCUMENT_INDEX.exists():
        raise HTTPException(status_code=404, detail="Document index not found")
    with open(DOCUMENT_INDEX, 'r') as f:
        return json.load(f)

@app.get("/api/chunks/{chunk_filename}")
async def get_chunk(chunk_filename: str):
    """Get specific chunk by filename."""
    chunk_path = KNOWLEDGE_BASE / chunk_filename
    if not chunk_path.exists():
        raise HTTPException(status_code=404, detail="Chunk not found")
    with open(chunk_path, 'r') as f:
        return json.load(f)

@app.get("/api/stats")
async def get_stats():
    """Get knowledge base statistics."""
    with open(DOCUMENT_INDEX, 'r') as f:
        index = json.load(f)

    num_docs = len(index.get('documents', []))
    num_chunks = sum(len(doc.get('chunks', [])) for doc in index.get('documents', []))

    return {
        "num_documents": num_docs,
        "num_chunks": num_chunks,
        "document_names": [doc.get('original', 'unknown') for doc in index.get('documents', [])]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
```

**Start API:**
```bash
cd /home/joker/FlockParser
pip install fastapi uvicorn
python flockparser_remote_api.py
```

**Modify SynapticLlamas Adapter:**

Create a remote-aware version that fetches via HTTP instead of direct file access.

## Performance Comparison

| Method | Latency | Bandwidth | Setup | Code Changes |
|--------|---------|-----------|-------|--------------|
| **Local** | <1ms | N/A | ✅ Done | None |
| **NFS** | 1-5ms | High | Easy | Path only |
| **HTTP API** | 10-50ms | Medium | Moderate | Adapter mod |
| **SSHFS** | 5-20ms | Medium | Easy | Path only |
| **Rsync** | N/A | Low | Easy | Cron script |

## Next Steps

1. **Identify remote machine details:**
   - IP address of FlockParser machine
   - IP address of SynapticLlamas machine
   - Network connectivity (same subnet?)

2. **Choose method:**
   - Same datacenter/LAN → **NFS** (best performance)
   - Over internet → **HTTP API** (firewall-friendly)
   - Quick dev test → **SSHFS** (no sudo needed)

3. **Run setup commands** from this guide

4. **Test with verification scripts:**
   ```bash
   python3 verify_flockparser_access.py
   python3 test_local_rag_integration.py
   ```

5. **Enable RAG in SynapticLlamas:**
   ```bash
   cd /home/joker/SynapticLlamas
   python main.py --interactive
   SynapticLlamas> rag on
   SynapticLlamas> Explain quantum computing
   ```

## Troubleshooting

### NFS Mount Fails
```bash
# Check NFS server is running:
sudo systemctl status nfs-server

# Check firewall allows NFS (port 2049):
sudo ufw allow from <client-ip> to any port 2049

# Check exports:
sudo exportfs -v
```

### Permission Denied
```bash
# Verify export uses correct UID mapping:
# In /etc/exports, ensure no_root_squash if needed:
/home/joker/FlockParser 10.9.66.0/24(ro,sync,no_subtree_check,no_root_squash)

sudo exportfs -ra
```

### Slow Performance
- Use NFS instead of SSHFS
- Check network latency: `ping <flockparser-ip>`
- Consider local rsync cache

## Summary

✅ **Local access working** - SynapticLlamas can read FlockParser files
📋 **4 remote methods documented** - NFS, HTTP API, SSHFS, Rsync
🚀 **Ready to deploy** - Choose method based on network setup
🧪 **Verification scripts** - Test before and after remote setup

**Recommendation:** Start with NFS for best performance with zero code changes.
