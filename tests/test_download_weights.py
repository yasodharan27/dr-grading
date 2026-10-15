"""The weights download: resume after a mid-file timeout (HTTP Range), full re-download when the server ignores Range,
retries with the listed waits, and a size-checked rename. A local HTTP server simulates the network; no internet is used."""
import hashlib
import importlib.util
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("dw", os.path.join(ROOT, "scripts", "download_weights.py"))
dw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dw)

PAYLOAD = bytes(range(256)) * 4096                       # 1 MiB of distinct bytes


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class Server:
    """Serves PAYLOAD. `plan` is a list with one entry per request: 'ok' (whole file), ('cut', n) (send n bytes then drop
    the connection) or ('cut-after-range', n). `honour_range` decides whether a Range request gets 206 or a full 200."""

    def __init__(self, plan, honour_range=True):
        self.plan, self.honour_range, self.requests = list(plan), honour_range, []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                start = 0
                rng = self.headers.get("Range")
                if rng and outer.honour_range:
                    start = int(rng.split("=")[1].split("-")[0])
                outer.requests.append(rng)
                action = outer.plan.pop(0) if outer.plan else "ok"
                body = PAYLOAD[start:]
                if rng and outer.honour_range:
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if action == "ok":
                    self.wfile.write(body)
                else:
                    self.wfile.write(body[:action[1]])
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.close_connection = True

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/f.bin"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def server_factory():
    servers = []

    def make(plan, honour_range=True):
        s = Server(plan, honour_range)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()


def test_timeout_mid_file_then_range_resume_gives_identical_bytes(server_factory, tmp_path):
    server = server_factory([("cut", 300_000), "ok"], honour_range=True)
    target = str(tmp_path / "f.bin")
    waits, lines = [], []
    dw.fetch(server.url, target, len(PAYLOAD), sleep=waits.append, log=lines.append)
    assert open(target, "rb").read() == PAYLOAD and _sha(open(target, "rb").read()) == _sha(PAYLOAD)
    assert server.requests == [None, f"bytes=300000-"]           # the second attempt asked only for the missing bytes
    assert waits == [5]
    assert len(lines) == 1 and "attempt 2 of 6" in lines[0] and "300000 of" in lines[0]
    assert not os.path.exists(target + ".part")


def test_server_that_ignores_range_gets_a_full_download(server_factory, tmp_path):
    server = server_factory([("cut", 200_000), "ok"], honour_range=False)
    target = str(tmp_path / "f.bin")
    dw.fetch(server.url, target, len(PAYLOAD), sleep=lambda _: None, log=lambda _: None)
    assert open(target, "rb").read() == PAYLOAD
    assert server.requests[1] == "bytes=200000-"                 # asked for the rest, got 200 and the whole file again


def test_several_timeouts_use_the_listed_waits(server_factory, tmp_path):
    # each cut sends a small piece, so the file needs four requests: three cuts, then the rest
    server = server_factory([("cut", 100_000), ("cut", 100_000), ("cut", 100_000), "ok"])
    target = str(tmp_path / "f.bin")
    waits = []
    dw.fetch(server.url, target, len(PAYLOAD), sleep=waits.append, log=lambda _: None)
    assert waits == [5, 10, 20]
    assert server.requests == [None, "bytes=100000-", "bytes=200000-", "bytes=300000-"]
    assert open(target, "rb").read() == PAYLOAD


def test_gives_up_after_six_attempts_and_keeps_the_partial_file(server_factory, tmp_path):
    server = server_factory([("cut", 50_000)] * 10)
    target = str(tmp_path / "f.bin")
    waits = []
    with pytest.raises(dw.DownloadError) as error:
        dw.fetch(server.url, target, len(PAYLOAD), sleep=waits.append, log=lambda _: None)
    assert "giving up after 6 attempts" in str(error.value)
    assert waits == [5, 10, 20, 40, 60]
    assert not os.path.exists(target)                            # no final file
    assert os.path.getsize(target + ".part") > 0                 # the partial is kept for a later run


def test_size_mismatch_is_never_renamed(server_factory, tmp_path, monkeypatch):
    server = server_factory(["ok"])
    target = str(tmp_path / "f.bin")
    with pytest.raises(dw.DownloadError):
        dw.fetch(server.url, target, len(PAYLOAD) + 1, attempts=1, sleep=lambda _: None, log=lambda _: None)
    assert not os.path.exists(target)


def test_a_file_already_complete_in_the_part_file_is_renamed_without_a_request(server_factory, tmp_path):
    server = server_factory([])
    target = str(tmp_path / "f.bin")
    open(target + ".part", "wb").write(PAYLOAD)
    dw.fetch(server.url, target, len(PAYLOAD), sleep=lambda _: None, log=lambda _: None)
    assert open(target, "rb").read() == PAYLOAD and server.requests == []
