from __future__ import annotations

import json
import statistics
import time
import requests
import numpy as np

HOST = "http://127.0.0.1:8000"
NUM_CHANNELS = 32
WINDOW_SIZE = 64
N_REQUESTS = 200


def random_signal(c: int, w: int) -> list[list[float]]:
    rng = np.random.default_rng(0)
    sig = rng.normal(0.0, 1.0, size=(c, w)).astype(np.float32)
    return sig.tolist()


def main():
    latencies = []
    payload = {"signal": random_signal(NUM_CHANNELS, WINDOW_SIZE)}

    # warmup
    for _ in range(5):
        requests.post(f"{HOST}/infer", json=payload, timeout=5)

    for _ in range(N_REQUESTS):
        t0 = time.perf_counter()
        r = requests.post(f"{HOST}/infer", json=payload, timeout=5)
        r.raise_for_status()
        lat = r.json()["latency_ms"]
        latencies.append(lat)

    print(f"n={len(latencies)} avg={statistics.mean(latencies):.2f} ms median={statistics.median(latencies):.2f} ms")
    print(f"p95={np.percentile(latencies, 95):.2f} ms p99={np.percentile(latencies, 99):.2f} ms")


if __name__ == "__main__":
    main()