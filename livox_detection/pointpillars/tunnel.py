"""Expose the local web viewer through a Cloudflare quick tunnel.

``cloudflared tunnel --url http://127.0.0.1:8080`` creates a throw-away
``https://<random>.trycloudflare.com`` address that anyone can open from a
different network.  No Cloudflare account or token is needed; the tunnel lives
only as long as the viewer process.

The helper here starts cloudflared, watches its log for the public URL and
shuts it down with the viewer.  Only the standard library is used, so this adds
nothing to ``requirements.txt``.
"""

import os
import re
import shutil
import subprocess
import sys
import threading
import time

URL_PATTERN = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")

# Places cloudflared commonly lands when it is not on PATH (e.g. a pip-less
# user install into ~/.local/bin while running under sudo/cron).
SEARCH_DIRS = ("~/.local/bin", "/usr/local/bin", "/usr/bin", "/opt/homebrew/bin")

INSTALL_HINT = (
    "cloudflared is not installed; the viewer keeps running locally only.\n"
    "  install it (no root needed):\n"
    "    mkdir -p ~/.local/bin && curl -fL -o ~/.local/bin/cloudflared \\\n"
    "      https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \\\n"
    "      && chmod +x ~/.local/bin/cloudflared\n"
    "  then re-run the viewer (Cloudflare quick tunnels need outbound HTTPS)."
)


def find_cloudflared():
    """Return the cloudflared executable, or None when it is not available."""
    env = os.environ.get("CLOUDFLARED_BIN")
    if env and os.path.isfile(os.path.expanduser(env)):
        return os.path.abspath(os.path.expanduser(env))
    found = shutil.which("cloudflared")
    if found:
        return found
    for directory in SEARCH_DIRS:
        candidate = os.path.join(os.path.expanduser(directory), "cloudflared")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


class CloudflareTunnel(object):
    """A running Cloudflare quick tunnel pointing at one local port."""

    def __init__(self, port, host="127.0.0.1", binary=None, timeout=30.0):
        self.port = int(port)
        # A quick tunnel always connects over loopback; 0.0.0.0/:: still accept it.
        self.host = "127.0.0.1" if host in ("", "0.0.0.0", "::") else host
        self.binary = binary or find_cloudflared()
        self.timeout = float(timeout)
        self.url = None
        self.error = None
        self._proc = None
        self._reader = None
        self._ready = threading.Event()
        self._recent = []

    # -- lifecycle -------------------------------------------------------
    def start(self):
        if not self.binary:
            raise RuntimeError(INSTALL_HINT)
        command = [self.binary, "tunnel", "--no-autoupdate",
                   "--url", "http://%s:%d" % (self.host, self.port)]
        try:
            self._proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except OSError as exc:
            raise RuntimeError("cannot run %s (%s)" % (self.binary, exc))

        self._reader = threading.Thread(target=self._pump, name="cloudflared-log")
        self._reader.daemon = True
        self._reader.start()

        deadline = time.time() + self.timeout
        try:
            while time.time() < deadline:
                if self._ready.wait(0.2):
                    return self.url
                if self._proc.poll() is not None:
                    break
        except KeyboardInterrupt:
            self.stop()
            raise
        detail = self._recent[-1] if self._recent else "no log output"
        self.stop()
        raise RuntimeError(
            "no Cloudflare URL after %.0fs (is outbound HTTPS blocked?). %s"
            % (self.timeout, detail))

    def stop(self):
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except OSError:
            pass
        self._proc = None
        self.url = None

    # -- internals -------------------------------------------------------
    def _pump(self):
        """Scan cloudflared's log (URL is printed to stderr, merged above)."""
        stdout = self._proc.stdout
        if stdout is None:
            return
        for raw in iter(stdout.readline, b""):
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            self._recent.append(line)
            if len(self._recent) > 40:
                del self._recent[0]
            match = URL_PATTERN.search(line)
            if match and not self.url:
                self.url = match.group(0)
                self._ready.set()

    def wait(self):
        """Block until the tunnel exits (used only when it is already running)."""
        if self._proc is not None:
            self._proc.wait()


def print_share_help(stream=None):
    stream = stream or sys.stdout
    stream.write(INSTALL_HINT + "\n")
    stream.flush()
