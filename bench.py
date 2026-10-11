from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request

PAYLOAD = json.dumps({"text": "That's all I needed, thank you.", "pause_ms": 620,
                      "stability_ms": 280, "language": "en"}).encode()


def call(url: str) -> float:
    start = time.perf_counter()
    req = urllib.request.Request(url, data=PAYLOAD, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as response:
        response.read()
    return (time.perf_counter() - start) * 1000


def summary(samples: list[float]) -> dict:
    ordered = sorted(samples)
    pct = lambda q: ordered[min(int(q * (len(ordered) - 1)), len(ordered) - 1)]
    elapsed_s = sum(samples) / 1000
    return {"requests": len(samples), "p50_ms": round(statistics.median(samples), 3),
            "p95_ms": round(pct(0.95), 3), "p99_ms": round(pct(0.99), 3),
            "mean_ms": round(statistics.mean(samples), 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--requests", type=int, default=1200)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()
    for _ in range(30):
        call(args.url)
    sequential = [call(args.url) for _ in range(args.requests)]
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        concurrent_samples = list(pool.map(lambda _: call(args.url), range(args.requests)))
    wall_s = time.perf_counter() - started
    result = {"target": args.url, "sequential": summary(sequential),
              "concurrent": summary(concurrent_samples), "concurrent_wall_seconds": round(wall_s, 3),
              "concurrent_throughput_rps": round(args.requests / wall_s, 2),
              "notes": "Includes local HTTP/JSON overhead; sandbox measurements, not deployment SLOs."}
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
