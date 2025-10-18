# Backend Architecture

SOLLOL uses a generic pool interface that allows supporting multiple LLM backends.

## Current Implementation

**Ollama** (production-tested in SynapticLlamas and FlockParser)
- Port 11434 auto-discovery
- API endpoints: `/api/chat`, `/api/generate`, `/api/embed`
- Model lifecycle monitoring via `/api/ps`
- VRAM tracking and GPU-aware routing
- Health monitoring and automatic failover

## Architecture Design

SOLLOL's core abstractions are backend-agnostic:

### 1. Pool Interface
```python
class Pool:
    """Generic pool interface - any backend can implement this."""
    
    def __init__(self, nodes: List[Dict]):
        self.nodes = nodes  # [{"host": "...", "port": "..."}]
    
    def chat(self, model: str, messages: List[Dict], **kwargs) -> Dict:
        """Send chat request to optimal node."""
        pass
    
    def generate(self, model: str, prompt: str, **kwargs) -> Dict:
        """Generate text from prompt."""
        pass
```

### 2. Discovery Interface
```python
def discover_nodes(timeout=0.5) -> List[Dict[str, str]]:
    """
    Discover nodes on the network.
    
    Returns:
        List of node dicts with backend type:
        [{"host": "10.9.66.124", "port": "11434", "backend": "ollama"}]
    """
    pass
```

### 3. Activity Monitoring
```python
def get_node_activity(node_url: str, backend_type: str) -> Dict:
    """
    Get current activity on a node (loaded models, VRAM, etc.).
    
    Backend-specific implementation - dashboard calls appropriate method.
    """
    pass
```

## Extension Points

### Adding a New Backend

**Realistic effort estimate:** ~1000-2000 lines per backend
- Core implementation: ~250 lines
- Tests (unit + integration): ~300-500 lines
- Error handling and edge cases: ~200-300 lines
- Bug fixes and debugging: ~250-500 lines
- Documentation and examples: ~100-200 lines

**Example: vLLM Support**

1. **Discovery** (~50 lines)
   ```python
   # src/sollol/discovery.py
   def discover_vllm_nodes(timeout=0.5):
       """Discover vLLM nodes (default port 8000)."""
       return _scan_network(port=8000, health_check="/v1/models")
   ```

2. **Pool Implementation** (~100 lines)
   ```python
   # src/sollol/vllm_pool.py
   class VLLMPool:
       def chat(self, model, messages, **kwargs):
           # vLLM uses OpenAI-compatible API
           endpoint = "/v1/chat/completions"
           data = {"model": model, "messages": messages}
           return self._make_request(endpoint, data)
   ```

3. **Activity Monitoring** (~50 lines)
   ```python
   # src/sollol/unified_dashboard.py
   def _get_vllm_activity(self, node_url):
       """Get vLLM model status."""
       response = requests.get(f"{node_url}/v1/models")
       return response.json()["data"]  # List of loaded models
   ```

4. **Backend Detection** (~50 lines)
   ```python
   def detect_backend_type(node_url):
       """Auto-detect backend by trying known endpoints."""
       if _check_endpoint(node_url, "/api/tags"):
           return "ollama"
       elif _check_endpoint(node_url, "/v1/models"):
           return "vllm"
       elif _check_endpoint(node_url, "/info"):
           return "tgi"
       return "unknown"
   ```

## Future Backend Support

### vLLM
- **Use case:** High-throughput async inference
- **API:** OpenAI-compatible (`/v1/chat/completions`)
- **Default port:** 8000
- **Discovery:** Check `/v1/models` endpoint
- **Activity:** Parse `/v1/models` for loaded models

### Text Generation Inference (TGI)
- **Use case:** Hugging Face model serving
- **API:** `/generate`, `/generate_stream`
- **Default port:** 8080
- **Discovery:** Check `/info` endpoint
- **Activity:** Parse `/info` for model details

### llama.cpp Server
- **Use case:** Cross-platform CPU/GPU inference
- **API:** OpenAI-compatible (`/v1/chat/completions`)
- **Default port:** 8080
- **Discovery:** Check `/health` endpoint
- **Note:** SOLLOL already supports llama.cpp RPC for distributed sharding

### LocalAI
- **Use case:** Multi-backend wrapper (Llama, GPT4All, etc.)
- **API:** OpenAI-compatible
- **Default port:** 8080
- **Discovery:** Check `/readyz` endpoint

## Design Principles

1. **Duck Typing Over Inheritance**
   - Pools don't need to inherit from a base class
   - As long as they have `.nodes`, `.chat()`, `.generate()`, they work
   - Pythonic approach - if it quacks like a pool, it's a pool

2. **Backend Detection**
   - Auto-detect backend type by trying known endpoints
   - Fallback gracefully if detection fails
   - User can override with `backend` parameter

3. **Unified Observability**
   - UnifiedDashboard works with any backend
   - WebSocket events are backend-agnostic JSON
   - HTTP endpoints provide generic node/backend info

4. **Keep Core Generic**
   - Intelligent routing (task analysis, scoring) is backend-agnostic
   - Health monitoring (VRAM, latency) works for all backends
   - Ray/Dask distribution doesn't depend on LLM backend

## Why Ollama First?

1. **Production validation** - Used in SynapticLlamas and FlockParser
2. **Clustering gap** - Ollama has no built-in clustering, so SOLLOL adds real value
3. **Better to do one backend well** than many backends poorly
4. **Local AI focus** - Ollama is designed for on-premises deployment

## Adding Your Backend

Want to add support for a new backend? Here's the checklist:

- [ ] Implement discovery function (`discover_X_nodes()`)
- [ ] Create pool class (`XPool` with `.chat()`, `.generate()`)
- [ ] Add activity monitoring (`_get_X_activity()`)
- [ ] Add backend detection logic
- [ ] Write tests (unit + integration)
- [ ] Update this document

PRs welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Examples

### Using vLLM (Hypothetical)
```python
from sollol.vllm_pool import VLLMPool
from sollol.discovery import discover_vllm_nodes

# Auto-discover vLLM nodes
nodes = discover_vllm_nodes()
pool = VLLMPool(nodes)

# Same interface as OllamaPool!
response = pool.chat(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Mixed Backend Cluster (Future)
```python
from sollol.unified_pool import UnifiedPool

# Auto-detect and use ALL backends
pool = UnifiedPool.auto_discover()
# Finds: 3 Ollama nodes, 2 vLLM nodes, 1 TGI node

# Router intelligently selects backend based on:
# - Model availability
# - Node performance
# - Task requirements
response = pool.chat(model="llama3.2", messages=[...])
```

## Questions?

- **"Why not use OpenAI API everywhere?"** - Many backends have custom features not in OpenAI spec (Ollama's `/api/ps`, vLLM's async streaming, etc.)
- **"Should I use SOLLOL with vLLM?"** - Once implemented, yes! vLLM's async engine + SOLLOL's intelligent routing would be powerful
- **"Can I help add a backend?"** - Absolutely! File an issue or submit a PR

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system design
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute
- [examples/](examples/) - Usage examples
