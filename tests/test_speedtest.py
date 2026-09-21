import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

import speedtest
from speedtest import RequestResult, add_cache_buster, main, run, summarize, validate_url

PAYLOAD_SIZE = 256 * 1024


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen_paths: list[str] = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        Handler.seen_paths.append(self.path)
        path = urlsplit(self.path).path
        body = b"x" * PAYLOAD_SIZE
        if path == "/file":
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/chunked":
            # Без Content-Length: тело передаётся chunked-кодированием.
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i in range(0, len(body), 32 * 1024):
                part = body[i:i + 32 * 1024]
                self.wfile.write(f"{len(part):x}\r\n".encode() + part + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
        elif path == "/truncated":
            # Обещаем больше, чем отдаём, и закрываем соединение.
            self.send_response(200)
            self.send_header("Content-Length", str(len(body) * 2))
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True
        elif path == "/empty":
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def quiet_run(url, **kw):
    return run(url, out=io.StringIO(), **kw)


# --- чистые функции ---------------------------------------------------------

def test_summary_uses_total_bytes_over_total_time_not_mean_of_speeds():
    # 10 MB за 1 s и 10 MB за 9 s: среднее скоростей = 5.56 MB/s, правильно = 2 MB/s.
    results = [
        RequestResult(1, ok=True, bytes=10_000_000, ttfb=0.1, total=1.0),
        RequestResult(2, ok=True, bytes=10_000_000, ttfb=0.1, total=9.0),
    ]
    s = summarize(results)
    assert s.throughput_bps == pytest.approx(2_000_000)
    assert s.mean_time == pytest.approx(5.0)
    assert s.transfer_throughput_bps == pytest.approx(20_000_000 / 9.8)


def test_summary_ignores_failed_requests():
    results = [
        RequestResult(1, ok=True, bytes=1000, ttfb=0.1, total=1.0),
        RequestResult(2, ok=False, error="boom"),
    ]
    s = summarize(results)
    assert (s.succeeded, s.attempted, s.total_bytes) == (1, 2, 1000)
    assert s.stdev_time == 0.0


def test_summary_none_when_all_failed():
    assert summarize([RequestResult(1, ok=False)]) is None
    assert summarize([]) is None


@pytest.mark.parametrize("url", ["ftp://host/file", "example.com/img.jpg", "http://", ""])
def test_validate_url_rejects_bad(url):
    with pytest.raises(ValueError):
        validate_url(url)


def test_cache_buster_keeps_existing_query():
    busted = add_cache_buster("https://host/img.jpg?a=1")
    q = parse_qs(urlsplit(busted).query)
    assert q["a"] == ["1"] and "_speedtest" in q
    assert add_cache_buster("http://h/x") != add_cache_buster("http://h/x")


def test_speed_units():
    assert speedtest.fmt_mbit(1_000_000) == "8.00 Mbit/s"
    assert speedtest.fmt_mbyte(1_000_000) == "1.00 MB/s"


# --- сеть (локальный сервер) -------------------------------------------------

@pytest.mark.parametrize("keepalive", [True, False])
def test_successful_run_counts_bytes(server, keepalive):
    results = quiet_run(f"{server}/file", count=3, keepalive=keepalive)
    assert len(results) == 3
    assert all(r.ok and r.bytes == PAYLOAD_SIZE for r in results)
    assert all(0 < r.ttfb <= r.total for r in results)


def test_chunked_without_content_length(server):
    [r] = quiet_run(f"{server}/chunked", count=1)
    assert r.ok and r.bytes == PAYLOAD_SIZE


def test_http_error_is_reported(server):
    [r] = quiet_run(f"{server}/missing", count=1)
    assert not r.ok and r.status == 404


def test_truncated_body_is_failure(server):
    [r] = quiet_run(f"{server}/truncated", count=1)
    assert not r.ok and r.bytes <= PAYLOAD_SIZE


def test_empty_body_is_failure(server):
    [r] = quiet_run(f"{server}/empty", count=1)
    assert not r.ok


def test_connection_refused():
    [r] = quiet_run("http://127.0.0.1:1/file", count=1, timeout=2)
    assert not r.ok and "ConnectionError" in r.error


def test_warmup_not_counted_and_cache_bust_applied(server):
    Handler.seen_paths.clear()
    results = quiet_run(f"{server}/file", count=2, warmup=True)
    assert len(results) == 2
    assert len(Handler.seen_paths) == 3
    assert len(set(Handler.seen_paths)) == 3  # у каждого запроса уникальный URL


def test_no_cache_bust_uses_plain_url(server):
    Handler.seen_paths.clear()
    quiet_run(f"{server}/file", count=2, cache_bust=False)
    assert Handler.seen_paths == ["/file", "/file"]


# --- CLI ------------------------------------------------------------------------

def test_main_success(server, capsys):
    assert main([f"{server}/file", "-n", "2"]) == 0
    out = capsys.readouterr().out
    assert "Успешных запросов:     2/2" in out and "Mbit/s" in out and "MB/s" in out


def test_main_all_failed(server):
    assert main([f"{server}/missing", "-n", "2"]) == 1


@pytest.mark.parametrize("argv", [["ftp://x/y"], ["http://x/y", "-n", "0"], ["http://x/y", "-t", "0"]])
def test_main_bad_args(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
