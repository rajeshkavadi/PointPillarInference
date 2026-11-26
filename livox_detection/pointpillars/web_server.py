"""Dependency-free HTTP + WebSocket server for the three.js viewer.

Only the Python standard library and NumPy are used, so the viewer works on a
fresh machine with nothing but the project's existing requirements installed.
There is no Flask, no rosbridge and no RViz involved.

Wire format (one binary WebSocket message per processed LiDAR frame), all
little endian and 4-byte aligned so the browser can read the cloud with a
zero-copy ``Float32Array`` view::

    offset  size      field
    0       4         magic b"LVP2" (single frame) or b"LVP3" (map mode)
    4       uint32    point count of this frame
    8       uint32    frame sequence number
    12      float64   header stamp (seconds)
    20      uint16    frame_id length
    22      uint16    JSON length
    [24     uint32    LVP3 only: number of new map points]
    24/28   n         frame_id, zero padded to a multiple of 4
    ...     m         JSON document, zero padded to a multiple of 4
    ...     n*16      float32 x, y, z, intensity per point (this frame)
    ...     k*16      LVP3 only: new map points in map/world coordinates

The JSON document holds the detections::

    {"frame_id": "...",
     "boxes": [{"c": "Car", "s": 0.95, "x": .., "y": .., "z": ..,
                "dx": .., "dy": .., "dz": .., "yaw": ..}, ...],
     "pose": {"x": 12.0, "y": 0.0},
     "meta": {"infer_ms": 4200.0, "points_in": 179472, "voxels": 17538,
              "map_points": 184259, "map_delta": 20220}}

In map mode both the frame points and the boxes are already expressed in the
map frame (world coordinates), so the browser can simply append the new map
points to what it already drew.
"""

import base64
import hashlib
import json
import mimetypes
import os
import socket
import struct
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import unquote, urlsplit

import numpy as np

MAGIC = b"LVP2"
MAGIC_MAP = b"LVP3"
HEADER = struct.Struct("<4sIIdHH")     # 24 bytes, no padding with "<"
HEADER_MAP = struct.Struct("<4sIIdHHI")  # 28 bytes: + new map point count
MAX_JSON = 65535
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

DEFAULT_WEBAPP_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "webapp"))


# ---------------------------------------------------------------------------
# Frame payload
# ---------------------------------------------------------------------------
def detections_payload(boxes, scores, labels, class_names, pose=None):
    """Convert arrays/lists of detections into the JSON-friendly box list.

    When ``pose`` (the sensor position in the map frame) is given, the box
    centres are offset into map coordinates to match the world-frame cloud.
    """
    out = []
    if boxes is None:
        return out
    ox, oy = (float(pose[0]), float(pose[1])) if pose else (0.0, 0.0)
    for i in range(len(boxes)):
        box = boxes[i]
        label = int(labels[i]) - 1
        name = class_names[label] if 0 <= label < len(class_names) else "object"
        out.append({
            "c": name,
            "s": round(float(scores[i]), 4),
            "x": round(float(box[0]) + ox, 3),
            "y": round(float(box[1]) + oy, 3),
            "z": round(float(box[2]), 3),
            "dx": round(float(box[3]), 3),
            "dy": round(float(box[4]), 3),
            "dz": round(float(box[5]), 3),
            "yaw": round(float(box[6]), 4),
        })
    return out


def encode_frame(seq, stamp, frame_id, points, boxes, meta=None,
                 map_points=None, pose=None):
    """Serialise one point cloud + detection frame for the browser.

    Passing ``map_points`` switches to the LVP3 message that also carries the
    points newly added to the persistent map (in map/world coordinates).
    """
    pts = np.ascontiguousarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 4:
        raise ValueError("points must be N x 4 float32, got %r" % (pts.shape,))
    frame_id = str(frame_id or "livox_frame")
    fid = frame_id.encode("utf-8")[:255]
    doc = {"frame_id": frame_id, "boxes": list(boxes or []), "meta": dict(meta or {})}
    if pose is not None:
        doc["pose"] = {"x": round(float(pose[0]), 3),
                       "y": round(float(pose[1]), 3)}
    if map_points is not None:
        map_pts = np.ascontiguousarray(map_points, dtype=np.float32).reshape(-1, 4)
        if map_pts.shape[0] > 0xFFFFFFFF:
            raise ValueError("too many map points")
        doc["meta"]["map_delta"] = int(map_pts.shape[0])
    else:
        map_pts = None
    js = json.dumps(doc, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(js) > MAX_JSON:
        doc["boxes"] = doc["boxes"][:200]
        doc["meta"]["truncated"] = True
        js = json.dumps(doc, separators=(",", ":")).encode("utf-8")
        if len(js) > MAX_JSON:
            raise ValueError("detection JSON too large (%d bytes)" % len(js))
    fid_pad = (len(fid) + 3) & ~3
    js_pad = (len(js) + 3) & ~3
    if map_pts is None:
        head = HEADER.pack(MAGIC, int(pts.shape[0]), int(seq), float(stamp),
                           len(fid), len(js))
        tail = [pts.tobytes()]
    else:
        head = HEADER_MAP.pack(MAGIC_MAP, int(pts.shape[0]), int(seq),
                               float(stamp), len(fid), len(js),
                               int(map_pts.shape[0]))
        tail = [pts.tobytes(), map_pts.tobytes()]
    return b"".join([
        head,
        fid, b"\x00" * (fid_pad - len(fid)),
        js, b"\x00" * (js_pad - len(js)),
    ] + tail)


# ---------------------------------------------------------------------------
# Latest-frame hub
# ---------------------------------------------------------------------------
class FrameHub(object):
    """Holds only the newest frame and fans it out to connected browsers."""

    def __init__(self, source="bag"):
        self._cond = threading.Condition()
        self._payload = None
        self._seq = 0
        self._points = 0
        self._boxes = 0
        self._stamp = 0.0
        self._frame_id = ""
        self._meta = {}
        self._published_at = 0.0
        self._fps = 0.0
        self._clients = 0
        self._total = 0
        self._bytes_sent = 0
        self._started_at = time.time()
        self.source = source

    def publish(self, payload, seq, point_count, box_count, stamp, frame_id, meta=None):
        now = time.time()
        with self._cond:
            if self._published_at:
                dt = max(now - self._published_at, 1e-3)
                inst = 1.0 / dt
                self._fps = inst if not self._fps else 0.7 * self._fps + 0.3 * inst
            self._payload = payload
            self._seq = int(seq)
            self._points = int(point_count)
            self._boxes = int(box_count)
            self._stamp = float(stamp)
            self._frame_id = frame_id
            self._meta = dict(meta or {})
            self._published_at = now
            self._total += 1
            self._cond.notify_all()

    def wait_for_frame(self, last_seq, timeout=1.0):
        with self._cond:
            if self._payload is None or self._seq <= last_seq:
                self._cond.wait(timeout)
            if self._payload is None or self._seq <= last_seq:
                return None, last_seq
            return self._payload, self._seq

    def client_connected(self):
        with self._cond:
            self._clients += 1

    def client_disconnected(self):
        with self._cond:
            self._clients = max(0, self._clients - 1)

    def add_bytes_sent(self, size):
        with self._cond:
            self._bytes_sent += int(size)

    def snapshot(self):
        now = time.time()
        with self._cond:
            return {
                "source": self.source,
                "server_time": now,
                "uptime_s": round(now - self._started_at, 1),
                "clients": self._clients,
                "frames": self._total,
                "seq": self._seq,
                "points": self._points,
                "boxes": self._boxes,
                "stamp": self._stamp,
                "frame_id": self._frame_id,
                "fps": round(self._fps, 3),
                "age_s": (round(now - self._published_at, 2)
                          if self._published_at else None),
                "meta": self._meta,
                "bytes_sent": self._bytes_sent,
            }


# ---------------------------------------------------------------------------
# WebSocket helpers
# ---------------------------------------------------------------------------
def _ws_write(sock, payload):
    header = bytearray([0x82])  # FIN + binary
    size = len(payload)
    if size < 126:
        header.append(size)
    elif size < 65536:
        header.append(126)
        header += struct.pack(">H", size)
    else:
        header.append(127)
        header += struct.pack(">Q", size)
    sock.sendall(bytes(header) + payload)


def _ws_ping(sock):
    sock.sendall(b"\x89\x00")


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
class _ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "livox_detection-web/1.0"

    # Quiet: one line per request is noise for a live viewer.
    def log_message(self, fmt, *args):  # noqa: A003
        pass

    # -- routing ---------------------------------------------------------
    def do_GET(self):  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/ws":
            return self._serve_websocket()
        if path in ("/", "/index.html"):
            return self._serve_file(os.path.join(self.server.webapp_dir, "index.html"))
        if path == "/api/status":
            body = json.dumps(self.server.hub.snapshot()).encode("utf-8")
            return self._send_bytes(body, "application/json", cache=False)
        if path.startswith("/static/"):
            target = self._static_target(path[len("/static/"):])
            if target is None:
                return self._send_error(404, "not found")
            return self._serve_file(target)
        return self._send_error(404, "not found")

    def _static_target(self, relative):
        rel = unquote(relative).replace("\\", "/")
        root = os.path.join(self.server.webapp_dir, "static")
        target = os.path.abspath(os.path.join(root, rel))
        if target != root and not target.startswith(root + os.sep):
            return None
        return target

    def _serve_file(self, path):
        if not os.path.isfile(path):
            return self._send_error(404, "not found")
        with open(path, "rb") as handle:
            body = handle.read()
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if path.endswith(".js"):
            ctype = "application/javascript"
        return self._send_bytes(body, ctype)

    def _send_bytes(self, body, content_type, cache=True):
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if not cache:
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, OSError):
            self.close_connection = True

    def _send_error(self, code, message):
        body = json.dumps({"error": message, "code": code}).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, OSError):
            self.close_connection = True

    # -- websocket -------------------------------------------------------
    def _serve_websocket(self):
        key = self.headers.get("Sec-WebSocket-Key")
        upgrade = (self.headers.get("Upgrade") or "").lower()
        if not key or upgrade != "websocket":
            return self._send_error(400, "expected a WebSocket upgrade")
        accept = base64.b64encode(
            hashlib.sha1((key + WS_GUID).encode("ascii")).digest()).decode("ascii")
        try:
            self.send_response(101, "Switching Protocols")
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept)
            self.end_headers()
            self.wfile.flush()
        except (BrokenPipeError, OSError):
            return

        hub = self.server.hub
        sock = self.connection
        hub.client_connected()
        last_seq = 0
        last_ping = time.time()
        try:
            sock.settimeout(30.0)
            while not self.server.stopping.is_set():
                payload, seq = hub.wait_for_frame(last_seq, timeout=1.0)
                if payload is None:
                    if time.time() - last_ping > 10.0:
                        _ws_ping(sock)
                        last_ping = time.time()
                    continue
                last_seq = seq
                _ws_write(sock, payload)
                hub.add_bytes_sent(len(payload))
                last_ping = time.time()
        except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
            pass
        finally:
            hub.client_disconnected()
            self.close_connection = True

    # Body-bearing requests are not used; keep the server from blocking.
    def do_POST(self):  # noqa: N802
        return self._send_error(405, "method not allowed")


class WebViewerServer(object):
    """Serves the webapp directory and streams frames over /ws."""

    def __init__(self, hub, host="127.0.0.1", port=8080, webapp_dir=None):
        self.hub = hub
        self.host = host
        self.port = int(port)
        self.webapp_dir = os.path.abspath(webapp_dir or DEFAULT_WEBAPP_DIR)
        self.stopping = threading.Event()
        self._httpd = None
        self._thread = None

    def start(self):
        self._httpd = _ThreadingHTTPServer((self.host, self.port), _Handler)
        self._httpd.hub = self.hub
        self._httpd.webapp_dir = self.webapp_dir
        self._httpd.stopping = self.stopping
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        kwargs={"poll_interval": 0.25},
                                        name="web-viewer")
        self._thread.daemon = True
        self._thread.start()
        return self

    @property
    def url(self):
        host = self.host if self.host not in ("0.0.0.0", "::") else "127.0.0.1"
        return "http://%s:%d" % (host, self.port)

    def stop(self):
        self.stopping.set()
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
