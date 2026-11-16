"""Measure local HTTP search latency at a fixed offered request rate."""
import argparse
import asyncio
import hashlib
from time import perf_counter

import httpx

from search.education import synthetic_lessons
from search.experiment import latency, runtime, write_json


async def trial(client, queries, mode, depth, rate, count):
    payloads = [
        {"query": queries[i % len(queries)], "mode": mode, "rerank_depth": depth}
        for i in range(count)
    ]
    for payload in payloads[:10]:
        response = await client.post("/search", json=payload)
        response.raise_for_status()

    start = perf_counter()
    slots = asyncio.Semaphore(16)

    async def request(i, payload):
        scheduled = start + i / rate
        await asyncio.sleep(max(0, scheduled - perf_counter()))
        async with slots:
            sent = perf_counter()
            try:
                response = await client.post("/search", json=payload)
                response.raise_for_status()
                stages = response.json()["stage_ms"]
                finished = perf_counter()
                return ((finished - sent) * 1000, stages, (sent - scheduled) * 1000)
            except (httpx.HTTPError, ValueError, KeyError):
                return (None, {}, (sent - scheduled) * 1000)

    outcomes = await asyncio.gather(*(request(i, payload) for i, payload in enumerate(payloads)))
    duration = perf_counter() - start
    completed = [row for row in outcomes if row[0] is not None]
    names = sorted({name for _, stages, _ in completed for name in stages})
    return {
        "offered_qps": rate, "achieved_success_qps": len(completed) / duration,
        "requests": count, "successes": len(completed), "errors": count - len(completed),
        "http_latency_ms": latency([row[0] for row in completed]) if completed else {},
        "dispatch_delay_ms": latency([row[2] for row in outcomes]),
        "stage_ms": {name: latency([stages[name] for _, stages, _ in completed if name in stages])
                     for name in names},
    }


async def benchmark(url, rate, count):
    ds = synthetic_lessons()
    queries = [ds.queries[qid] for qid in ds.splits["test"]]
    async with httpx.AsyncClient(base_url=url, timeout=10, trust_env=False, limits=httpx.Limits(max_connections=16)) as client:
        (await client.get("/health")).raise_for_status()
        settings = [("bm25", 100), ("rrf", 100), ("lambdamart", 30), ("lambdamart", 100)]
        measurements = {}
        for mode, depth in settings:
            key = f"{mode}_{depth}"
            measurements[key] = await trial(client, queries, mode, depth, rate, count)
            print(key, measurements[key], flush=True)
        return {
            "setup": runtime(2), "client": "async HTTP/1.1 with up to 16 in flight", "url": url,
            "query_set": "synthetic lessons / topic-held-out test",
            "query_sha256": hashlib.sha256("\n".join(queries).encode()).hexdigest(),
            "results": measurements,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--qps", type=int, default=25)
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--output", default="runs/load.json")
    args = parser.parse_args()
    if args.qps < 1 or args.requests < 1:
        parser.error("--qps and --requests must be positive")
    write_json(args.output, asyncio.run(benchmark(args.url, args.qps, args.requests)))


if __name__ == "__main__":
    main()
