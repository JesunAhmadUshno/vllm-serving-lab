"""Concurrent load test client for the serving API, Level 2 Build.

Fires a fixed number of chat completion requests at a running server
with a thread pool, and aggregates latency and throughput from the
per-request results. Aggregation is a pure function so it is unit
tested without a server.
"""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def percentile(values, pct):
    """Nearest-rank percentile: the smallest value at or above pct."""
    if not values:
        return None
    ordered = sorted(values)
    rank = -(-pct * len(ordered) // 100)  # ceil(pct * n / 100)
    rank = max(1, min(len(ordered), int(rank)))
    return ordered[rank - 1]


def aggregate(results, wall_seconds):
    """Aggregate per-request dicts into load test summary numbers."""
    ok = [r for r in results if r.get("ok")]
    latencies = [r["latency_seconds"] for r in ok]
    tokens = sum(r.get("completion_tokens", 0) for r in ok)
    return {
        "requests": len(results),
        "succeeded": len(ok),
        "failed": len(results) - len(ok),
        "wall_seconds": round(wall_seconds, 3),
        "requests_per_second": round(len(ok) / wall_seconds, 3) if wall_seconds else None,
        "completion_tokens_per_second": (round(tokens / wall_seconds, 2)
                                         if wall_seconds else None),
        "p50_latency_seconds": (round(percentile(latencies, 50), 4)
                                if latencies else None),
        "p95_latency_seconds": (round(percentile(latencies, 95), 4)
                                if latencies else None),
    }


def one_request(base_url, prompt, max_tokens):
    body = json.dumps({
        "model": "served-model",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(base_url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            payload = json.loads(response.read())
        return {"ok": True, "latency_seconds": time.time() - started,
                "completion_tokens": payload["usage"]["completion_tokens"]}
    except Exception as exc:  # recorded, not hidden: the summary counts failures
        return {"ok": False, "latency_seconds": time.time() - started,
                "error": type(exc).__name__}


def run_load_test(base_url, prompts, concurrency=4, max_tokens=96):
    started = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(
            lambda p: one_request(base_url, p, max_tokens), prompts))
    return aggregate(results, time.time() - started), results
