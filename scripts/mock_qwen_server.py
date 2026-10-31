import http.server
import json
import socketserver
import sys

class MockQwenHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/v1/models", "/api/tags"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {"data": [{"id": "qwen2.5-coder:1.5b"}]}
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        _ = self.rfile.read(content_length)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        resp = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "chain_of_thought": "Analysis shows implementation matches ticket specifications.",
                            "where_it_went_wrong": "No issues detected.",
                            "root_cause": "N/A",
                            "suggested_fix": "N/A",
                            "confidence": "high",
                        })
                    }
                }
            ]
        }
        self.wfile.write(json.dumps(resp).encode("utf-8"))

    def log_message(self, format, *args):
        pass  # Quiet logging

if __name__ == "__main__":
    port = 11434
    try:
        with socketserver.TCPServer(("127.0.0.1", port), MockQwenHandler) as httpd:
            print(f"Mock Qwen server running on http://127.0.0.1:{port}", flush=True)
            httpd.serve_forever()
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
