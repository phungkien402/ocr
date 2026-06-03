#!/usr/bin/env python3
"""Load test for OCR /v1/extract endpoint.

Đo throughput + latency khi N user gọi đồng thời. Output bảng stats: p50/p95/p99,
errors, RPS. Dùng asyncio + httpx — không spawn process, lightweight.

Usage:
    # 10 user đồng thời, mỗi user 1 request
    python3 loadtest.py --image test.jpg --concurrency 10

    # 50 request tổng, max 5 đồng thời (queue dần)
    python3 loadtest.py --image test.jpg --total 50 --concurrency 5

    # Endpoint khác, key khác
    python3 loadtest.py --url https://ocr.company.vn --key xxxxxx \\
        --image test.jpg --concurrency 20

    # Ramp up: tăng dần concurrency 1→20 over 30s
    python3 loadtest.py --image test.jpg --ramp 30 --concurrency 20
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Missing httpx. Install: pip install httpx")
    raise SystemExit(1)


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    return s[f] if f == k else s[f] + (s[f + 1] - s[f]) * (k - f)


async def one_request(client, args, sem, image_bytes, idx):
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.post(
                f"{args.url.rstrip('/')}/v1/extract",
                headers={
                    "X-API-Key": args.key,
                    "ngrok-skip-browser-warning": "1",
                },
                files={"file": (Path(args.image).name, image_bytes, "image/jpeg")},
                timeout=args.timeout,
            )
            elapsed = time.perf_counter() - t0
            ok = (r.status_code == 200)
            return {
                "idx": idx,
                "status": r.status_code,
                "ok": ok,
                "elapsed": elapsed,
                "error": None if ok else (r.text[:200] if not ok else None),
            }
        except Exception as e:
            return {
                "idx": idx,
                "status": 0,
                "ok": False,
                "elapsed": time.perf_counter() - t0,
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            }


async def ramp_one_request(client, args, image_bytes, idx, delay):
    """For ramp mode: wait `delay` then fire request without semaphore."""
    await asyncio.sleep(delay)
    sem = asyncio.Semaphore(99999)  # no limit during ramp
    return await one_request(client, args, sem, image_bytes, idx)


async def run(args):
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {args.image}")
        raise SystemExit(1)
    image_bytes = image_path.read_bytes()

    total = args.total if args.total else args.concurrency
    print(f"━━━ OCR Load Test ━━━")
    print(f"URL:         {args.url}")
    print(f"Image:       {image_path.name} ({len(image_bytes)} bytes)")
    print(f"Total reqs:  {total}")
    print(f"Concurrency: {args.concurrency}")
    if args.ramp:
        print(f"Ramp:        {args.ramp}s linear")
    print(f"Timeout:     {args.timeout}s per request")
    print()

    sem = asyncio.Semaphore(args.concurrency)
    start = time.perf_counter()

    async with httpx.AsyncClient(verify=not args.insecure) as client:
        # Quick health check first
        try:
            r = await client.get(f"{args.url.rstrip('/')}/health",
                                 headers={"ngrok-skip-browser-warning": "1"},
                                 timeout=10)
            if r.status_code != 200:
                print(f"[WARN] /health returned {r.status_code} — server may be down")
        except Exception as e:
            print(f"[ERROR] Cannot reach /health: {e}")
            print("Aborting load test.")
            return

        # Fire requests
        if args.ramp:
            # Linear ramp: spread requests evenly over `ramp` seconds
            delays = [args.ramp * i / total for i in range(total)]
            tasks = [ramp_one_request(client, args, image_bytes, i, d)
                     for i, d in enumerate(delays)]
        else:
            tasks = [one_request(client, args, sem, image_bytes, i)
                     for i in range(total)]

        # Progress indicator
        results = []
        done = 0
        for fut in asyncio.as_completed(tasks):
            r = await fut
            results.append(r)
            done += 1
            if done % max(1, total // 20) == 0 or done == total:
                ok_count = sum(1 for x in results if x["ok"])
                print(f"  [{done}/{total}] ok={ok_count} fail={done-ok_count}",
                      end="\r", flush=True)
        print()

    total_time = time.perf_counter() - start

    # ── Stats ────────────────────────────────────────────────────────────
    ok_results = [r for r in results if r["ok"]]
    fail_results = [r for r in results if not r["ok"]]
    latencies = [r["elapsed"] for r in ok_results]

    print()
    print(f"━━━ Results ━━━")
    print(f"Total time:    {total_time:.2f}s")
    print(f"Success:       {len(ok_results)}/{total}  ({100*len(ok_results)/total:.1f}%)")
    print(f"Failed:        {len(fail_results)}")
    if total_time > 0:
        print(f"Throughput:    {len(ok_results)/total_time:.2f} req/s")
    print()

    if latencies:
        print(f"Latency (successful requests only):")
        print(f"  min:    {min(latencies):.2f}s")
        print(f"  p50:    {pct(latencies, 0.50):.2f}s")
        print(f"  p90:    {pct(latencies, 0.90):.2f}s")
        print(f"  p95:    {pct(latencies, 0.95):.2f}s")
        print(f"  p99:    {pct(latencies, 0.99):.2f}s")
        print(f"  max:    {max(latencies):.2f}s")
        print(f"  mean:   {statistics.mean(latencies):.2f}s")
        if len(latencies) > 1:
            print(f"  stdev:  {statistics.stdev(latencies):.2f}s")

    if fail_results:
        print()
        print(f"━━━ Errors (first 5) ━━━")
        err_summary = {}
        for r in fail_results:
            key = f"HTTP {r['status']}" if r['status'] else (r['error'] or 'unknown').split(':')[0]
            err_summary[key] = err_summary.get(key, 0) + 1
        for kind, count in sorted(err_summary.items(), key=lambda x: -x[1]):
            print(f"  {count:4}x  {kind}")
        print()
        for r in fail_results[:5]:
            print(f"  req#{r['idx']:3} status={r['status']} elapsed={r['elapsed']:.2f}s")
            if r["error"]:
                print(f"          {r['error'][:150]}")

    # Save JSON for further analysis
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "config": vars(args),
                "total_time": total_time,
                "summary": {
                    "total": total,
                    "ok": len(ok_results),
                    "failed": len(fail_results),
                    "throughput_rps": len(ok_results) / total_time if total_time > 0 else 0,
                    "p50": pct(latencies, 0.50),
                    "p95": pct(latencies, 0.95),
                    "p99": pct(latencies, 0.99),
                },
                "results": results,
            }, f, ensure_ascii=False, indent=2)
        print()
        print(f"Detailed results saved to {args.output}")


def main():
    p = argparse.ArgumentParser(
        description="Load test OCR /v1/extract",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--url", default=os.environ.get("OCR_URL", "http://localhost:8502"),
                   help="Base URL (default: env OCR_URL or http://localhost:8502)")
    p.add_argument("--key", default=os.environ.get("OCR_API_KEY", ""),
                   help="API key (default: env OCR_API_KEY)")
    p.add_argument("--image", "-i", required=True, help="Image file to upload")
    p.add_argument("--concurrency", "-c", type=int, default=10,
                   help="Max concurrent requests (default: 10)")
    p.add_argument("--total", "-n", type=int, default=None,
                   help="Total requests (default: same as concurrency)")
    p.add_argument("--ramp", type=int, default=0,
                   help="Ramp seconds: spread requests linearly (overrides concurrency limit)")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="Per-request timeout in seconds (default: 120)")
    p.add_argument("--insecure", action="store_true",
                   help="Skip TLS verify (self-signed cert)")
    p.add_argument("--output", "-o", help="Save full results JSON")
    args = p.parse_args()

    if not args.key:
        print("ERROR: API key required. Pass --key or set env OCR_API_KEY")
        raise SystemExit(1)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
