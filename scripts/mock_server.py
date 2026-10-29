"""Tiny OpenAI-compatible mock: checks the request shape and answers with the first option letter."""
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

LOG = open("mock_requests.log", "w")


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(n))
        msgs = body["messages"]
        user = msgs[1]["content"]
        n_img = sum(1 for c in user if c["type"] == "image_url")
        n_vid = sum(1 for c in user if c["type"] == "video_url")
        text = user[-1]["text"]
        bad = [c for c in user if c["type"] == "image_url" and not c["image_url"]["url"].startswith("data:image/jpeg;base64,")]
        LOG.write(json.dumps(dict(n_img=n_img, n_vid=n_vid, first_text=user[0].get("text", "")[:60],
                                  labels=[c["text"] for c in user if c["type"] == "text"][:3], bad=len(bad)), ensure_ascii=False) + "\n")
        LOG.flush()
        ans = "{A}" if "A." in text else "{3}"
        out = json.dumps({"choices": [{"message": {"content": f"我看了 {n_img} 张图。{ans}"}}], "usage": {"prompt_tokens": 1}})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out.encode())

    def log_message(self, *a):
        pass


HTTPServer(("127.0.0.1", 8765), H).serve_forever()
