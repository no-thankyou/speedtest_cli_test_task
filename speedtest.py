#!/usr/bin/env python3
"""Замер скорости загрузки: N последовательных HTTP GET к одному URL.

Для каждого запроса отдельно меряются TTFB (задержка до получения заголовков)
и полное время (TTFB + передача тела). Итоговая скорость считается как
суммарные байты / суммарное время, а не как среднее скоростей отдельных запросов.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
import urllib3

__version__ = "1.0.0"

CHUNK_SIZE = 64 * 1024
USER_AGENT = f"speedtest-cli/{__version__} (+python-requests)"

EXIT_OK = 0
EXIT_ALL_FAILED = 1
EXIT_INTERRUPTED = 130


@dataclass
class RequestResult:
    index: int
    ok: bool
    bytes: int = 0
    ttfb: float = 0.0       # секунды от отправки запроса до получения заголовков
    total: float = 0.0      # секунды от отправки запроса до последнего байта тела
    status: int | None = None
    error: str | None = None

    @property
    def transfer(self) -> float:
        """Время передачи тела (без установки соединения и ожидания ответа)."""
        return max(self.total - self.ttfb, 0.0)


@dataclass
class Summary:
    attempted: int
    succeeded: int
    total_bytes: int
    total_time: float
    mean_time: float
    median_time: float
    min_time: float
    max_time: float
    stdev_time: float
    mean_ttfb: float
    throughput_bps: float           # байт/с = Σbytes / Σtotal
    transfer_throughput_bps: float  # байт/с = Σbytes / Σtransfer (без задержки)


def validate_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"ожидается http(s) URL, получено: {url!r}")
    return url


def add_cache_buster(url: str) -> str:
    """Добавляет уникальный query-параметр, чтобы обойти промежуточные кэши."""
    parts = urlsplit(url)
    query = parse_qsl(parts.query, keep_blank_values=True)
    query.append(("_speedtest", uuid.uuid4().hex))
    return urlunsplit(parts._replace(query=urlencode(query)))


def build_headers(no_cache: bool) -> dict[str, str]:
    headers = {
        "User-Agent": USER_AGENT,
        # Просим несжатый ответ: считаем байты, реально прошедшие по сети.
        "Accept-Encoding": "identity",
    }
    if no_cache:
        headers["Cache-Control"] = "no-cache"
        headers["Pragma"] = "no-cache"
    return headers


def measure_once(
    session: requests.Session,
    url: str,
    index: int,
    timeout: float,
    cache_bust: bool,
) -> RequestResult:
    target = add_cache_buster(url) if cache_bust else url
    headers = build_headers(no_cache=cache_bust)
    received = 0
    start = time.perf_counter()
    ttfb = 0.0
    try:
        # stream=True: get() возвращается сразу после получения заголовков.
        with session.get(target, headers=headers, timeout=timeout, stream=True) as resp:
            ttfb = time.perf_counter() - start
            if not 200 <= resp.status_code < 300:
                return RequestResult(index, ok=False, status=resp.status_code,
                                     ttfb=ttfb, error=f"HTTP {resp.status_code} {resp.reason}")
            # decode_content=False: считаем сырые байты с провода, даже если сервер
            # проигнорировал Accept-Encoding и прислал сжатое тело.
            for chunk in resp.raw.stream(CHUNK_SIZE, decode_content=False):
                received += len(chunk)
            total = time.perf_counter() - start

            expected = resp.headers.get("Content-Length")
            if expected is not None and expected.isdigit() and int(expected) != received:
                return RequestResult(index, ok=False, status=resp.status_code, bytes=received,
                                     ttfb=ttfb, total=total,
                                     error=f"получено {received} из {expected} байт")
            if received == 0:
                return RequestResult(index, ok=False, status=resp.status_code, ttfb=ttfb,
                                     total=total, error="пустое тело ответа")
            return RequestResult(index, ok=True, status=resp.status_code, bytes=received,
                                 ttfb=ttfb, total=total)
    except (requests.RequestException, urllib3.exceptions.HTTPError) as exc:
        return RequestResult(index, ok=False, bytes=received, ttfb=ttfb,
                             total=time.perf_counter() - start,
                             error=f"{type(exc).__name__}: {exc}")


def summarize(results: list[RequestResult]) -> Summary | None:
    ok = [r for r in results if r.ok]
    if not ok:
        return None
    times = [r.total for r in ok]
    total_bytes = sum(r.bytes for r in ok)
    total_time = sum(times)
    total_transfer = sum(r.transfer for r in ok)
    return Summary(
        attempted=len(results),
        succeeded=len(ok),
        total_bytes=total_bytes,
        total_time=total_time,
        mean_time=statistics.fmean(times),
        median_time=statistics.median(times),
        min_time=min(times),
        max_time=max(times),
        stdev_time=statistics.stdev(times) if len(times) > 1 else 0.0,
        mean_ttfb=statistics.fmean(r.ttfb for r in ok),
        throughput_bps=total_bytes / total_time if total_time > 0 else 0.0,
        transfer_throughput_bps=total_bytes / total_transfer if total_transfer > 0 else 0.0,
    )


def fmt_mbit(bps: float) -> str:
    return f"{bps * 8 / 1e6:.2f} Mbit/s"


def fmt_mbyte(bps: float) -> str:
    return f"{bps / 1e6:.2f} MB/s"


def fmt_size(n: int) -> str:
    return f"{n / 1e6:.2f} MB"


def fmt_result(r: RequestResult, width: int) -> str:
    prefix = f"#{r.index:<{width}}"
    if not r.ok:
        error = r.error or ""
        if len(error) > 100:
            error = error[:97] + "..."
        return f"{prefix}  FAILED  {error}"
    speed = r.bytes / r.total if r.total > 0 else 0.0
    return (f"{prefix}  {fmt_size(r.bytes):>10}  {r.total:7.3f} s  "
            f"TTFB {r.ttfb * 1000:6.0f} ms  {fmt_mbit(speed):>15}")


def fmt_summary(s: Summary) -> str:
    lines = [
        "",
        f"Успешных запросов:     {s.succeeded}/{s.attempted}",
        f"Скачано всего:         {fmt_size(s.total_bytes)} ({s.total_bytes} байт)",
        f"Среднее время запроса: {s.mean_time:.3f} s "
        f"(медиана {s.median_time:.3f}, min {s.min_time:.3f}, max {s.max_time:.3f}, "
        f"σ {s.stdev_time:.3f})",
        f"Средний TTFB:          {s.mean_ttfb * 1000:.0f} ms",
        f"Скорость передачи тела (без TTFB): {fmt_mbit(s.transfer_throughput_bps)}",
        "",
        f"СКОРОСТЬ: {fmt_mbit(s.throughput_bps)}  ({fmt_mbyte(s.throughput_bps)})",
    ]
    return "\n".join(lines)


def run(
    url: str,
    count: int = 10,
    timeout: float = 30.0,
    keepalive: bool = True,
    warmup: bool = False,
    cache_bust: bool = True,
    out=sys.stdout,
    results: list[RequestResult] | None = None,
) -> list[RequestResult]:
    """Выполняет замер и печатает построчный прогресс.

    Результаты дописываются в `results` по мере выполнения, чтобы при Ctrl+C
    вызывающий код мог посчитать статистику по уже завершённым запросам.
    """
    if results is None:
        results = []
    width = len(str(count))
    shared = requests.Session() if keepalive else None
    try:
        if warmup:
            session = shared or requests.Session()
            try:
                w = measure_once(session, url, 0, timeout, cache_bust)
            finally:
                if shared is None:
                    session.close()
            status = "ok" if w.ok else f"FAILED ({w.error})"
            print(f"Прогревочный запрос (не входит в статистику): {status}", file=out)

        for i in range(1, count + 1):
            # Без keep-alive — новая сессия (и новое TCP/TLS-соединение) на каждый запрос.
            session = shared or requests.Session()
            try:
                result = measure_once(session, url, i, timeout, cache_bust)
            finally:
                if shared is None:
                    session.close()
            results.append(result)
            print(fmt_result(result, width), file=out, flush=True)
    finally:
        if shared is not None:
            shared.close()
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Замер скорости загрузки: N последовательных запросов к одному URL.",
    )
    p.add_argument("url", help="адрес тяжёлого файла (http/https)")
    p.add_argument("-n", "--count", type=int, default=10,
                   help="число запросов (по умолчанию 10)")
    p.add_argument("-t", "--timeout", type=float, default=30.0,
                   help="таймаут на подключение и паузу между чанками, с (по умолчанию 30)")
    p.add_argument("--no-keepalive", action="store_true",
                   help="новое соединение на каждый запрос (DNS+TCP+TLS каждый раз)")
    p.add_argument("--warmup", action="store_true",
                   help="сделать прогревочный запрос, не входящий в статистику")
    p.add_argument("--no-cache-bust", action="store_true",
                   help="не добавлять уникальный query-параметр и no-cache заголовки")
    p.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    args = p.parse_args(argv)
    if args.count < 1:
        p.error("--count должен быть >= 1")
    if args.timeout <= 0:
        p.error("--timeout должен быть > 0")
    try:
        validate_url(args.url)
    except ValueError as exc:
        p.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mode = "новое соединение на запрос" if args.no_keepalive else "keep-alive"
    print(f"URL: {args.url}\nЗапросов: {args.count}, режим: {mode}\n")

    results: list[RequestResult] = []
    interrupted = False
    try:
        run(args.url, args.count, args.timeout,
            keepalive=not args.no_keepalive, warmup=args.warmup,
            cache_bust=not args.no_cache_bust, results=results)
    except KeyboardInterrupt:
        interrupted = True
        print("\nПрервано пользователем.", file=sys.stderr)

    summary = summarize(results)
    if summary is None:
        if not interrupted:
            print("\nНи один запрос не завершился успешно.", file=sys.stderr)
        return EXIT_INTERRUPTED if interrupted else EXIT_ALL_FAILED
    print(fmt_summary(summary))
    return EXIT_INTERRUPTED if interrupted else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
