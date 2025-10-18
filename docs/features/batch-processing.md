# Batch Processing API Documentation

**New in v0.7.0** - Complete RESTful API for asynchronous batch job management in SOLLOL.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Job Lifecycle](#job-lifecycle)
- [Examples](#examples)
- [Best Practices](#best-practices)
- [Error Handling](#error-handling)

---

## Overview

The Batch Processing API enables asynchronous execution of large-scale operations (embeddings, bulk inference) with:

- **UUID-based job tracking** - Track jobs across requests
- **Progress monitoring** - Real-time completion percentage
- **Automatic cleanup** - TTL-based job expiration (1 hour default)
- **Distributed execution** - Powered by Dask for parallel processing
- **Job management** - Submit, status, results, cancel, list

### When to Use Batch API

✅ **Use batch API when:**
- Processing hundreds or thousands of documents
- Embedding large document collections
- Bulk inference for batch predictions
- Long-running operations that shouldn't block
- Need to track progress of async operations

❌ **Don't use batch API when:**
- Processing < 10 items (use regular sync API)
- Need immediate results (batch is async)
- Processing one-off requests

---

## Quick Start

### 1. Submit a Batch Job

```python
import requests

response = requests.post("http://localhost:11434/api/batch/embed", json={
    "model": "nomic-embed-text",
    "documents": [
        "Document 1 content here",
        "Document 2 content here",
        # ... up to 10,000 documents
    ],
    "metadata": {"source": "knowledge_base"}  # Optional
})

job_id = response.json()["job_id"]
print(f"Job submitted: {job_id}")
```

### 2. Check Job Status

```python
status = requests.get(f"http://localhost:11434/api/batch/jobs/{job_id}").json()

print(f"Status: {status['status']}")
print(f"Progress: {status['progress']['percent']}%")
print(f"Completed: {status['progress']['completed_items']}/{status['progress']['total_items']}")
```

### 3. Get Results

```python
results = requests.get(f"http://localhost:11434/api/batch/results/{job_id}").json()

embeddings = results["results"]
errors = results["errors"]
print(f"Got {len(embeddings)} results")
```

---

## API Endpoints

### POST /api/batch/embed

Submit a batch embedding job.

**Request:**
```json
{
  "model": "nomic-embed-text",
  "documents": ["doc1", "doc2", ...],
  "metadata": {}  // Optional
}
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "running",
  "total_items": 1000,
  "message": "Batch embedding job submitted with 1000 documents"
}
```

**Limits:**
- Maximum 10,000 documents per batch
- Returns 400 if limit exceeded

---

### GET /api/batch/jobs/{job_id}

Get detailed job status.

**Response:**
```json
{
  "job_id": "uuid-string",
  "job_type": "embed",
  "status": "completed",
  "created_at": "2025-10-06T12:00:00",
  "started_at": "2025-10-06T12:00:01",
  "completed_at": "2025-10-06T12:00:45",
  "progress": {
    "total_items": 1000,
    "completed_items": 1000,
    "failed_items": 0,
    "percent": 100.0
  },
  "duration_seconds": 44.2,
  "metadata": {
    "model": "nomic-embed-text"
  }
}
```

**Status values:**
- `pending` - Job created, not started
- `running` - Job executing
- `completed` - Job finished successfully
- `failed` - Job failed with errors
- `cancelled` - Job was cancelled

---

### GET /api/batch/results/{job_id}

Retrieve job results and errors.

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "results": [
    {"embedding": [0.1, 0.2, ...]},
    {"embedding": [0.3, 0.4, ...]},
    ...
  ],
  "errors": [],
  "total_items": 1000,
  "completed_items": 1000,
  "failed_items": 0
}
```

**Notes:**
- Results available after job completes
- Results stored for 1 hour (TTL)
- Returns 404 if job not found

---

### DELETE /api/batch/jobs/{job_id}

Cancel a running job.

**Response:**
```json
{
  "job_id": "uuid-string",
  "cancelled": true,
  "message": "Job cancelled successfully"
}
```

**Notes:**
- Only works on `pending` or `running` jobs
- Returns 404 if already `completed` or `failed`

---

### GET /api/batch/jobs?limit=100

List recent batch jobs.

**Query Parameters:**
- `limit` - Maximum number of jobs to return (default: 100, max: 1000)

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "uuid-1",
      "job_type": "embed",
      "status": "completed",
      "progress": {"percent": 100.0, ...},
      ...
    },
    ...
  ],
  "total": 25
}
```

**Notes:**
- Jobs sorted by creation time (most recent first)
- Only returns jobs in memory (not expired)

---

## Job Lifecycle

```
┌─────────┐
│ PENDING │  Job created, waiting to start
└────┬────┘
     │
     ▼
┌─────────┐
│ RUNNING │  Job executing, can check progress
└────┬────┘
     │
     ├──────► ┌───────────┐
     │        │ COMPLETED │  Job finished successfully
     │        └───────────┘
     │
     ├──────► ┌────────┐
     │        │ FAILED │  Job failed with errors
     │        └────────┘
     │
     └──────► ┌───────────┐
              │ CANCELLED │  Job was cancelled by user
              └───────────┘
```

**TTL (Time-to-Live):**
- Completed/failed/cancelled jobs expire after 1 hour
- Automatic cleanup runs when new jobs are created
- Expired jobs removed from memory

---

## Examples

### Example 1: Embedding Large Document Collection

```python
import requests
import time

# Submit batch job
docs = [f"Document {i} content..." for i in range(5000)]

response = requests.post("http://localhost:11434/api/batch/embed", json={
    "model": "nomic-embed-text",
    "documents": docs,
    "metadata": {"collection": "research_papers"}
})

job_id = response.json()["job_id"]

# Poll for completion
while True:
    status = requests.get(f"http://localhost:11434/api/batch/jobs/{job_id}").json()

    if status["status"] in ["completed", "failed", "cancelled"]:
        break

    print(f"Progress: {status['progress']['percent']:.1f}%")
    time.sleep(2)

# Get results
if status["status"] == "completed":
    results = requests.get(f"http://localhost:11434/api/batch/results/{job_id}").json()
    print(f"Embedded {len(results['results'])} documents in {status['duration_seconds']:.2f}s")
else:
    print(f"Job {status['status']}: {status.get('errors')}")
```

### Example 2: Progress Callback with WebSocket (Future)

```python
# TODO: WebSocket support for real-time progress updates
# Currently: Use polling (shown above)
```

### Example 3: Managing Multiple Batch Jobs

```python
import requests

# Submit multiple jobs
job_ids = []
for batch in document_batches:
    response = requests.post("http://localhost:11434/api/batch/embed", json={
        "model": "nomic-embed-text",
        "documents": batch
    })
    job_ids.append(response.json()["job_id"])

# Monitor all jobs
while job_ids:
    for job_id in list(job_ids):
        status = requests.get(f"http://localhost:11434/api/batch/jobs/{job_id}").json()

        if status["status"] == "completed":
            print(f"Job {job_id} complete!")
            job_ids.remove(job_id)
        elif status["status"] in ["failed", "cancelled"]:
            print(f"Job {job_id} {status['status']}")
            job_ids.remove(job_id)

    time.sleep(5)

print("All jobs finished!")
```

---

## Best Practices

### 1. Batch Size

```python
# ✅ Good: Batch into reasonable chunks
docs = load_all_documents()  # 50,000 documents

# Split into batches of 5,000 (under 10K limit)
batch_size = 5000
for i in range(0, len(docs), batch_size):
    batch = docs[i:i+batch_size]
    submit_batch(batch)

# ❌ Bad: Submitting too many at once
submit_batch(docs[:50000])  # Exceeds 10K limit
```

### 2. Error Handling

```python
# ✅ Good: Check for errors
results = requests.get(f"/api/batch/results/{job_id}").json()

if results["failed_items"] > 0:
    print(f"Warning: {results['failed_items']} items failed")
    for error in results["errors"]:
        print(f"Error: {error}")

# ✅ Good: Handle 404 (job expired)
try:
    status = requests.get(f"/api/batch/jobs/{job_id}")
    status.raise_for_status()
except requests.HTTPError as e:
    if e.response.status_code == 404:
        print("Job expired or not found")
```

### 3. Polling Strategy

```python
# ✅ Good: Exponential backoff
delay = 1
while True:
    status = check_status(job_id)
    if status["status"] != "running":
        break

    time.sleep(delay)
    delay = min(delay * 1.5, 30)  # Cap at 30s

# ❌ Bad: Aggressive polling
while True:
    status = check_status(job_id)
    time.sleep(0.1)  # Too frequent
```

### 4. Metadata Usage

```python
# ✅ Good: Store context in metadata
requests.post("/api/batch/embed", json={
    "model": "nomic-embed-text",
    "documents": docs,
    "metadata": {
        "collection": "research_papers",
        "user_id": "user123",
        "batch_num": 5,
        "total_batches": 10
    }
})

# Later: Retrieve context
status = requests.get(f"/api/batch/jobs/{job_id}").json()
print(f"Batch {status['metadata']['batch_num']}/{status['metadata']['total_batches']}")
```

---

## Error Handling

### Common Error Codes

| Code | Meaning | Solution |
|------|---------|----------|
| 400 | Bad request (>10K docs) | Split into smaller batches |
| 404 | Job not found | Job expired or invalid ID |
| 503 | Batch processing disabled | Enable with `--batch-processing` |

### Example Error Response

```json
{
  "detail": "Maximum 10,000 documents per batch. Got 15,000."
}
```

### Handling Failures

```python
status = requests.get(f"/api/batch/jobs/{job_id}").json()

if status["status"] == "failed":
    # Job-level failure
    print(f"Job failed: {status['errors']}")

elif status["status"] == "completed" and status["progress"]["failed_items"] > 0:
    # Partial failure (some items failed)
    results = requests.get(f"/api/batch/results/{job_id}").json()
    print(f"Failed items: {results['failed_items']}")
    for error in results["errors"]:
        print(f"  - {error}")
```

---

## Statistics

Get batch processing statistics from the main stats endpoint:

```python
stats = requests.get("http://localhost:11434/api/stats").json()

batch_stats = stats["batch_jobs"]
print(f"Total jobs created: {batch_stats['total_jobs_created']}")
print(f"Total completed: {batch_stats['total_jobs_completed']}")
print(f"Total failed: {batch_stats['total_jobs_failed']}")
print(f"Active jobs: {batch_stats['active_jobs']}")
print(f"Pending jobs: {batch_stats['pending_jobs']}")
```

---

## Configuration

### Gateway Startup

```bash
# Enable batch processing (default: enabled)
sollol up --dask-workers 4

# Disable batch processing
sollol up --no-batch-processing

# Adjust autobatch interval
export SOLLOL_AUTOBATCH_INTERVAL=30  # seconds
sollol up
```

### Environment Variables

- `SOLLOL_BATCH_PROCESSING` - Enable/disable batch processing (default: true)
- `SOLLOL_DASK_WORKERS` - Number of Dask workers (default: 2)
- `SOLLOL_AUTOBATCH_INTERVAL` - Autobatch cycle interval in seconds (default: 60)

---

## Testing

Run the comprehensive batch API test suite:

```bash
# Start gateway on test port
sollol up --port 23000 --ray-workers 1 --dask-workers 2

# Run tests
python test_batch_api.py
```

Expected output:
```
✅ All batch API tests passed!

Batch API endpoints working:
  • POST /api/batch/embed - Submit batch embedding job
  • GET /api/batch/jobs/{job_id} - Get job status
  • GET /api/batch/results/{job_id} - Get job results
  • GET /api/batch/jobs - List jobs
  • DELETE /api/batch/jobs/{job_id} - Cancel job
  • GET /api/stats - Includes batch_jobs stats
```

---

## Implementation Details

### Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ POST /api/batch/embed
       ▼
┌─────────────────┐
│ FastAPI Gateway │──► BatchJobManager (job tracking)
└─────────┬───────┘
          │
          ▼
┌──────────────────┐
│  Dask Cluster    │──► Distributed execution
└──────────────────┘
          │
          ▼
┌──────────────────┐
│ Ollama Backend(s)│──► Actual embedding/inference
└──────────────────┘
```

### Job Manager Features

- **UUID-based IDs** - Unique job identification
- **5 states** - PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
- **TTL cleanup** - Automatic expiration (1 hour default)
- **Max jobs** - 1,000 jobs in memory (configurable)
- **Progress tracking** - Real-time completion percentage
- **Duration calculation** - created_at → completed_at

### Async Execution

Jobs execute asynchronously via:
1. Dask task graph creation (`embed_documents()`)
2. Async task submission (`_dask_client.compute()`)
3. Fire-and-forget tracking (`asyncio.create_task()`)
4. Bridge to async/await (`asyncio.gather` + `asyncio.to_thread`)

---

## Future Enhancements

Planned features for future releases:

- [ ] **WebSocket support** - Real-time progress updates
- [ ] **Batch chat/generate** - Support for chat and generation batches
- [ ] **Job priorities** - Priority levels for batch jobs
- [ ] **Job scheduling** - Delayed execution and recurring jobs
- [ ] **Result streaming** - Stream results as they complete
- [ ] **Job dependencies** - Chain jobs with dependencies
- [ ] **Batch templates** - Reusable batch job templates

---

## Support

For issues or questions:
- 📖 [Main Documentation](README.md)
- 🐛 [Report Issues](https://github.com/BenevolentJoker-JohnL/SOLLOL/issues)
- 💬 [Discussions](https://github.com/BenevolentJoker-JohnL/SOLLOL/discussions)

---

**Version:** v0.7.0
**Last Updated:** October 2025
