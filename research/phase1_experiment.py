import sys
import pathlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Local imports
here = pathlib.Path(__file__).resolve()
sys.path.append(str(here.parent))
sys.path.append(str(here.parents[1] / "demos"))
sys.path.append(str(here.parents[1] / "data"))

from mobile_architectures import MobileEEGNet, TinyEEGNet
from benchmark_framework import BCIBenchmark
from realistic_demo import EEGNet  # EEG baseline
from realistic_eeg_data import generate_realistic_motor_imagery_eeg


def train_and_evaluate(model: torch.nn.Module,
                       train_x: torch.Tensor,
                       train_y: torch.Tensor,
                       test_x: torch.Tensor,
                       test_y: torch.Tensor,
                       epochs: int = 5,
                       batch_size: int = 32) -> float:
    """Minimal training loop for accuracy benchmarking (CPU)."""
    device = torch.device("cpu")  # Force CPU to avoid GPU compatibility issues
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_ds = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Eval
    model.eval()
    with torch.no_grad():
        preds = model(test_x.to(device)).argmax(dim=1).cpu()
    acc = (preds == test_y).float().mean().item() * 100.0
    return acc


def run_experiment() -> dict:
    print("Phase 1: Architecture Comparison (Speed + Accuracy)")
    print("=" * 40)

    Path("research/results").mkdir(parents=True, exist_ok=True)

    # ------------------------
    # Speed Benchmark (latency)
    # ------------------------
    benchmark = BCIBenchmark()
    sample_input = torch.randn(1, 22, 1000)  # [B, C, T]

    models = {
        "EEGNet": EEGNet(n_channels=22, n_classes=4, samples=1000),
        "MobileEEGNet": MobileEEGNet(n_channels=22, n_classes=4, samples=1000),
        "TinyEEGNet": TinyEEGNet(n_channels=22, n_classes=4, samples=1000),
    }

    for name, model in models.items():
        benchmark.register_model(name, model, sample_input)

    speed_results = benchmark.compare_all()

    # ------------------------
    # Accuracy Benchmark
    # ------------------------
    print("\nGenerating small realistic dataset for accuracy…")
    data, labels = generate_realistic_motor_imagery_eeg(
        n_trials=200, n_channels=22, fs=250, trial_length=2.0
    )

    # Torch tensors
    data_t = torch.FloatTensor(data)
    labels_t = torch.LongTensor(labels)

    # Train/Test split
    n_train = int(0.8 * len(data_t))
    train_x, test_x = data_t[:n_train], data_t[n_train:]
    train_y, test_y = labels_t[:n_train], labels_t[n_train:]

    print("\nTraining quick accuracy benchmarks (few epochs per model)…")
    # Re-instantiate models for accuracy to ensure correct input shapes (e.g., EEGNet depends on samples)
    acc_models = {
        "EEGNet": EEGNet(n_channels=22, n_classes=4, samples=int(data.shape[2])),
        "MobileEEGNet": MobileEEGNet(n_channels=22, n_classes=4, samples=int(data.shape[2])),
        "TinyEEGNet": TinyEEGNet(n_channels=22, n_classes=4, samples=int(data.shape[2])),
    }

    accuracy_results = {}
    for name, model in acc_models.items():
        print(f"  Training {name}…")
        acc = train_and_evaluate(model, train_x, train_y, test_x, test_y, epochs=5, batch_size=32)
        accuracy_results[name] = acc
        print(f"    Accuracy: {acc:.2f}%")

    # ------------------------
    # Combine & Save Results
    # ------------------------
    combined = {}
    for name in models.keys():
        combined[name] = {
            "params": int(speed_results[name]["params"]),
            "size_mb": float(speed_results[name]["size_mb"]),
            "latency_ms_mean": float(speed_results[name]["speed"]["mean_ms"]),
            "latency_ms_p95": float(speed_results[name]["speed"]["p95_ms"]),
            "accuracy_percent": float(accuracy_results[name]),
        }

    print("\nResults Summary:")
    print("-" * 40)
    for name, result in combined.items():
        print(f"{name}:")
        print(f"  Parameters: {result['params']:,}")
        print(f"  Size: {result['size_mb']:.2f} MB")
        print(f"  Latency: {result['latency_ms_mean']:.2f} ms (P95: {result['latency_ms_p95']:.2f} ms)")
        print(f"  Accuracy: {result['accuracy_percent']:.2f}%")
        print()

    out_path = Path("research/results/phase1_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved results to {out_path}")

    return combined


if __name__ == "__main__":
    run_experiment()
    print("Phase 1 experiment complete!")


