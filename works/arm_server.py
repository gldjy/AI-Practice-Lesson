#!/usr/bin/env python
"""
Serves the arm viewer and shares the current pose with arm_client.py.

Usage:
    python arm_server.py
    python arm_server.py --host 127.0.0.1 --port 8000 --no-browser

The page is available at http://127.0.0.1:8000/arm_model.html.
"""

import argparse
import json
import posixpath
import socket
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from arm_params import DEFAULT_POSE, JOINTS, format_pose, normalize_pose

WORKS_DIR = Path(__file__).resolve().parent
API_PATH = "/api/pose"


def normalized_path(raw):
    """Normalizes an HTTP path before applying the static allowlist."""
    return posixpath.normpath(unquote(urlparse(raw).path))


class PoseState:
    """Keeps the current pose safe to read from several HTTP threads."""

    def __init__(self, pose):
        self.pose = dict(pose)
        self.seq = 0
        self.updated = time.time()
        self.lock = threading.Lock()
        self.last_log = 0.0

    def snapshot(self):
        with self.lock:
            return dict(self.pose), self.seq, self.updated

    def update(self, pose):
        now = time.time()
        with self.lock:
            self.pose = dict(pose)
            self.seq += 1
            self.updated = now
            seq = self.seq
            should_log = now - self.last_log >= 1.0
            if should_log:
                self.last_log = now
        if should_log:
            print(f"[Pose {seq}] {format_pose(pose)}")

    def payload(self):
        pose, seq, updated = self.snapshot()
        return {"pose": pose, "seq": seq, "updated": updated}


STATE = PoseState(DEFAULT_POSE)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WORKS_DIR), **kwargs)

    def log_message(self, format, *args):
        # The client polls the pose endpoint, so the default access log
        # would flood the console.
        pass

    def list_directory(self, path):
        self.send_error(404)
        return None

    def do_GET(self):
        path = normalized_path(self.path)
        if path == API_PATH:
            self.send_json(STATE.payload())
            return
        if path == "/api/joints":
            self.send_json({"joints": JOINTS})
            return
        if path in ("/", "/arm_model.html"):
            self.path = "/arm_model.html"
        elif path.startswith("/res/"):
            self.path = path
        else:
            self.send_error(404)
            return
        try:
            super().do_GET()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        path = normalized_path(self.path)
        if path != API_PATH:
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 16 * 1024:
            self.send_json({"error": "请求体为空或过大"}, 400)
            return

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if isinstance(payload, dict) and "pose" in payload:
                raw = payload["pose"]
            else:
                raw = payload
            pose = normalize_pose(raw)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, 400)
            return

        STATE.update(pose)
        self.send_json(STATE.payload())

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


class ArmServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def get_lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return None


def main():
    parser = argparse.ArgumentParser(description="机械臂姿态网页与 HTTP 服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，局域网客户端用 0.0.0.0")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    if not (WORKS_DIR / "arm_model.html").exists():
        print("[Error] 找不到 arm_model.html")
        return 1

    try:
        server = ArmServer((args.host, args.port), Handler)
    except OSError as exc:
        print(f"[Error] 无法监听 {args.host}:{args.port}: {exc}")
        return 1

    browser_host = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
    local_url = f"http://{browser_host}:{args.port}/arm_model.html"
    print(f"[Info] 静态目录: {WORKS_DIR}")
    print(f"[Info] 初始姿态: {format_pose(DEFAULT_POSE)}")
    print(f"[Info] 网页: {local_url}")

    lan_ip = get_lan_ip()
    print(f"[Info] 客户端(本机): python arm_client.py --server {browser_host} --port {args.port}")
    if args.host in ("0.0.0.0", "") and lan_ip:
        print(f"[Info] 局域网: http://{lan_ip}:{args.port}/arm_model.html")
        print(f"[Info] 客户端(局域网): python arm_client.py --server {lan_ip} --port {args.port}")
    elif args.host in ("127.0.0.1", "localhost"):
        print("[Info] 仅本机可访问，需要局域网客户端时加 --host 0.0.0.0")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(local_url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Info] 服务已停止")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
