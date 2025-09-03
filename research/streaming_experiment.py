import sys
import pathlib
import time
import json
from pathlib import Path

import numpy as np
import torch

here = pathlib.Path(__file__).resolve()
sys.path.append(str(here.parent))
sys.path.append(str(here.parents[1] / "demos"))
sys.path.append(str(here.parents[1] / "data"))

from mobile_architectures import MobileEEGNet, TinyEEGNet
from realistic_demo import EEGNet


def sliding_windows(signal: np.ndarray, window: int, step: int):
    # signal: [C, T]
    t = signal.shape[1]
    idx = 0
    while idx + window <= t:
        yield signal[:, idx:idx + window]
        idx += step


def measure_stream_latency(model: torch.nn.Module, signal: np.ndarray, window: int, step: int) -> dict:
    device = torch.device("cpu")
    model = model.to(device).eval()

    times = []
    with torch.no_grad():
        # Warmup
        w0 = torch.from_numpy(signal[:, :window]).float().unsqueeze(0)
        _ = model(w0)
        # Measure
        for w in sliding_windows(signal, window, step):
            x = torch.from_numpy(w).float().unsqueeze(0)
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    if not times:
        return {"mean_ms": None, "p95_ms": None, "throughput_hz": 0}

    mean_ms = float(np.mean(times))
    p95_ms = float(np.percentile(times, 95))
    # Effective decision rate determined by step size (samples per decision)
    # For fs=250 Hz: throughput ≈ 250/step decisions per second
    return {"mean_ms": mean_ms, "p95_ms": p95_ms}


def run():
    print("Streaming vs Batch Experiment (CPU)")
    Path("research/results").mkdir(parents=True, exist_ok=True)

    # Simulate a continuous signal (22 channels, 60 seconds @ 250 Hz)
    fs = 250
    duration_s = 60
    num_channels = 22
    total_samples = fs * duration_s
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((num_channels, total_samples)).astype(np.float32)

    # Window sizes (samples) and overlaps
    window_sizes = [250, 500, 1000]  # 1.0s, 2.0s, 4.0s
    overlaps = [0.0, 0.5]  # 0% and 50%

    results = {}
    for w in window_sizes:
        # Re-instantiate models per window so EEGNet classifier matches sample length
        models = {
            "EEGNet": EEGNet(n_channels=num_channels, n_classes=4, samples=w),
            "MobileEEGNet": MobileEEGNet(n_channels=num_channels, n_classes=4, samples=w),
            "TinyEEGNet": TinyEEGNet(n_channels=num_channels, n_classes=4, samples=w),
        }

        for name, model in models.items():
            results.setdefault(name, {})
            for ov in overlaps:
                step = int(w * (1.0 - ov)) or 1
                m = measure_stream_latency(model, signal, window=w, step=step)
                throughput_hz = fs / step
                key = f"w{w}_ov{int(ov*100)}"
                results[name][key] = {
                    "mean_ms": m["mean_ms"],
                    "p95_ms": m["p95_ms"],
                    "throughput_hz": float(throughput_hz),
                }
                print(f"{name:12} | w={w:4d} ov={ov:.1f} | mean={m['mean_ms']:.2f} ms, p95={m['p95_ms']:.2f} ms, rate~{throughput_hz:.1f} Hz")

    out = Path("research/results/streaming_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved streaming results to {out}")


if __name__ == "__main__":
    run()


