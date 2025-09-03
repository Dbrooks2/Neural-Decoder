import torch
import time
import numpy as np


class BCIBenchmark:
    def __init__(self) -> None:
        self.models = {}

    def register_model(self, name: str, model: torch.nn.Module, sample_input: torch.Tensor) -> None:
        params = sum(p.numel() for p in model.parameters())
        size_mb = params * 4 / 1024**2
        self.models[name] = {
            "model": model,
            "input": sample_input,
            "params": params,
            "size_mb": size_mb,
        }
        print(f"Registered {name}: {params:,} parameters, {size_mb:.2f} MB")

    def benchmark_speed(self, model_name: str, num_runs: int = 100) -> dict:
        model = self.models[model_name]["model"]
        sample_input = self.models[model_name]["input"]

        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)

        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(sample_input)
                end = time.perf_counter()
                times.append((end - start) * 1000.0)

        return {
            "mean_ms": float(np.mean(times)),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
        }

    def compare_all(self) -> dict:
        results = {}
        for name in self.models.keys():
            print(f"Benchmarking {name}…")
            speed = self.benchmark_speed(name)
            results[name] = {
                "speed": speed,
                "params": self.models[name]["params"],
                "size_mb": self.models[name]["size_mb"],
            }
            print(f"  Avg: {speed['mean_ms']:.2f} ms | P95: {speed['p95_ms']:.2f} ms")
        return results


