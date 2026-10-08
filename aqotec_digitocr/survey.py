import base64
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import cv2
import numpy as np
import requests

CORE_API = "http://supervisor/core/api"
TOKEN = os.environ.get("SUPERVISOR_TOKEN", "")
ROOT = Path("/data/survey_capture")
PORT = 8099
INTERVAL = 0.8
MAX_FILES = 140

_session = requests.Session()
_session.headers.update({"Authorization": f"Bearer {TOKEN}"})
_active = False
_camera_entity = None
_last_hash = None
_lock = threading.Lock()


def _fetch_camera():
    r = _session.get(
        f"{CORE_API}/camera_proxy/{quote(_camera_entity, safe='._')}",
        timeout=20,
    )
    r.raise_for_status()
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("camera_decode_failed")
    return img


def _screen_crop(img):
    h, w = img.shape[:2]
    return img[
        int(0.20 * h):int(0.92 * h),
        int(0.30 * w):int(0.97 * w),
    ].copy()


def _hash(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (3, 3), 0)
    return (small > np.median(small)).astype(np.uint8).reshape(-1)


def _distance(a, b):
    if a is None or b is None:
        return 999
    return int(np.count_nonzero(a != b))


def _files():
    ROOT.mkdir(parents=True, exist_ok=True)
    return sorted(ROOT.glob("*.jpg"), key=lambda p: p.stat().st_mtime)


def _capture_loop():
    global _last_hash
    ROOT.mkdir(parents=True, exist_ok=True)
    while True:
        if not _active:
            time.sleep(0.2)
            continue
        try:
            img = _screen_crop(_fetch_camera())
            hsh = _hash(img)
            with _lock:
                dist = _distance(_last_hash, hsh)
                if _last_hash is None or dist >= 18:
                    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                    ms = int((time.time() % 1) * 1000)
                    path = ROOT / f"{stamp}_{ms:03d}.jpg"
                    cv2.imwrite(
                        str(path),
                        img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 62],
                    )
                    _last_hash = hsh
                    files = _files()
                    for old in files[:-MAX_FILES]:
                        try:
                            old.unlink()
                        except OSError:
                            pass
        except Exception:
            pass
        time.sleep(INTERVAL)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        return

    def _json(self, status, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/status"):
            self._json(200, {
                "ok": True,
                "active": _active,
                "interval_s": INTERVAL,
                "files": len(_files()),
            })
            return
        if parsed.path == "/list":
            self._json(200, {
                "active": _active,
                "interval_s": INTERVAL,
                "files": [
                    {"name": p.name, "size": p.stat().st_size}
                    for p in _files()
                ],
            })
            return
        if parsed.path == "/image":
            name = (parse_qs(parsed.query).get("name") or [""])[0]
            if not name or Path(name).name != name:
                self._json(400, {"error": "invalid_name"})
                return
            path = ROOT / name
            if not path.exists():
                self._json(404, {"error": "not_found"})
                return
            self._json(200, {
                "name": name,
                "mime": "image/jpeg",
                "base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            })
            return
        self._json(404, {"error": "not_found"})

    def do_POST(self):
        global _active, _last_hash
        parsed = urlparse(self.path)
        if parsed.path == "/start":
            with _lock:
                _last_hash = None
            _active = True
            self._json(200, {"ok": True, "active": True, "interval_s": INTERVAL})
            return
        if parsed.path == "/stop":
            _active = False
            self._json(200, {"ok": True, "active": False, "files": len(_files())})
            return
        if parsed.path == "/clear":
            _active = False
            with _lock:
                _last_hash = None
                for p in _files():
                    try:
                        p.unlink()
                    except OSError:
                        pass
            self._json(200, {"ok": True, "active": False, "files": 0})
            return
        self._json(404, {"error": "not_found"})


def start(camera_entity):
    global _camera_entity
    _camera_entity = camera_entity
    threading.Thread(target=_capture_loop, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
