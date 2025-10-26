#!/usr/bin/env python3
"""
Lightweight mock Ollama server for CI testing.

Simulates basic Ollama API endpoints without requiring actual models.
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


class MockOllamaHandler(BaseHTTPRequestHandler):
    """Mock Ollama API request handler."""

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass  # Silent mode for CI

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/api/tags":
            # Return mock model list
            response = {
                "models": [
                    {
                        "name": "llama3.2",
                        "modified_at": "2024-01-01T00:00:00Z",
                        "size": 2000000000,
                        "digest": "sha256:mock",
                    }
                ]
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        elif self.path == "/":
            # Health check
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Ollama is running")

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        if self.path == "/api/chat":
            # Mock chat response
            try:
                request_data = json.loads(body)
                response = {
                    "model": request_data.get("model", "llama3.2"),
                    "created_at": "2024-01-01T00:00:00Z",
                    "message": {
                        "role": "assistant",
                        "content": f"Mock response from port {self.server.server_port}",
                    },
                    "done": True,
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                self.send_error(500, str(e))

        elif self.path == "/api/generate":
            # Mock generation response
            try:
                request_data = json.loads(body)
                response = {
                    "model": request_data.get("model", "llama3.2"),
                    "created_at": "2024-01-01T00:00:00Z",
                    "response": f"Mock generation from port {self.server.server_port}",
                    "done": True,
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                self.send_error(500, str(e))

        elif self.path == "/api/embeddings" or self.path == "/api/embed":
            # Mock embeddings response
            try:
                request_data = json.loads(body)
                # Return mock 384-dimensional embedding
                response = {
                    "model": request_data.get("model", "mxbai-embed-large"),
                    "embeddings": [[0.1] * 384],  # Mock embedding vector
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                self.send_error(500, str(e))

        else:
            self.send_response(404)
            self.end_headers()


def run_server(port=11434):
    """Run mock Ollama server."""
    server = HTTPServer(("0.0.0.0", port), MockOllamaHandler)
    print(f"Mock Ollama server running on port {port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nShutting down mock server on port {port}")
        server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock Ollama server for testing")
    parser.add_argument("--port", type=int, default=11434, help="Port to run on")
    args = parser.parse_args()

    run_server(args.port)
