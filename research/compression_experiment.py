import sys
import pathlib
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

here = pathlib.Path(__file__).resolve()
sys.path.append(str(here.parent))
sys.path.append(str(here.parents[1] / "demos"))
sys.path.append(str(here.parents[1] / "data"))

from mobile_architectures import MobileEEGNet, TinyEEGNet
from benchmark_framework import BCIBenchmark
from realistic_demo import EEGNet
from realistic_eeg_data import generate_realistic_motor_imagery_eeg


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def dynamic_quantize_linear_only(model: nn.Module) -> nn.Module:
    # Dynamic quantization supports nn.Linear on CPU
    # Keep a copy to avoid in-place issues
    m = copy.deepcopy(model).cpu().eval()
    try:
        quantized = torch.ao.quantization.quantize_dynamic(
            m, {nn.Linear}, dtype=torch.qint8
        )
        return quantized
    except Exception:
        return m


def magnitude_prune(model: nn.Module, amount: float) -> nn.Module:
    m = copy.deepcopy(model)
    for module in m.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            try:
                prune.l1_unstructured(module, name="weight", amount=amount)
                prune.remove(module, "weight")
            except Exception:
                pass
    return m


def quick_accuracy(model: nn.Module,
                   train_x: torch.Tensor,
                   train_y: torch.Tensor,
                   test_x: torch.Tensor,
                   test_y: torch.Tensor,
                   epochs: int = 2,
                   batch_size: int = 32) -> float:
    device = torch.device("cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = torch.utils.data.TensorDataset(train_x, train_y)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(test_x.to(device)).argmax(dim=1).cpu()
    return (preds == test_y).float().mean().item() * 100.0


def run() -> None:
    print("Compression Experiment: pruning + dynamic quantization (CPU)")
    Path("research/results").mkdir(parents=True, exist_ok=True)

    # Small dataset for quick accuracy check (define first to get correct sample length)
    print("\nGenerating small dataset for accuracy checks…")
    data, labels = generate_realistic_motor_imagery_eeg(n_trials=160, n_channels=22, fs=250, trial_length=2.0)
    data_t = torch.FloatTensor(data)
    labels_t = torch.LongTensor(labels)
    n_train = int(0.8 * len(data_t))
    train_x, test_x = data_t[:n_train], data_t[n_train:]
    train_y, test_y = labels_t[:n_train], labels_t[n_train:]

    # Models (fresh instances with dataset sample length)
    samples = int(data_t.shape[2])
    base_models = {
        "EEGNet": EEGNet(n_channels=22, n_classes=4, samples=samples),
        "MobileEEGNet": MobileEEGNet(n_channels=22, n_classes=4, samples=samples),
        "TinyEEGNet": TinyEEGNet(n_channels=22, n_classes=4, samples=samples),
    }

    # Latency benchmark setup (consistent with model sample length)
    bench = BCIBenchmark()
    sample_input = torch.randn(1, 22, samples)
    for name, model in base_models.items():
        bench.register_model(name, model, sample_input)
    baseline = {}
    print("\nBaseline latency…")
    for name in base_models.keys():
        baseline[name] = bench.benchmark_speed(name)

    results = {}

    for name, model in base_models.items():
        print(f"\nModel: {name}")
        results[name] = {
            "baseline": {
                "params": count_params(model),
                "latency_ms": baseline[name]["mean_ms"],
            },
            "pruned": {},
            "quantized": {},
        }

        # Accuracy baseline (very quick)
        base_acc = quick_accuracy(copy.deepcopy(model), train_x, train_y, test_x, test_y, epochs=2)
        results[name]["baseline"]["accuracy"] = base_acc
        print(f"  Baseline acc: {base_acc:.2f}% | latency: {baseline[name]['mean_ms']:.2f} ms")

        # Pruning at two levels
        for amt in [0.5, 0.9]:
            pruned = magnitude_prune(model, amount=amt)
            bench.register_model(f"{name}_pruned_{int(amt*100)}", pruned, sample_input)
            sp = bench.benchmark_speed(f"{name}_pruned_{int(amt*100)}")
            acc = quick_accuracy(copy.deepcopy(pruned), train_x, train_y, test_x, test_y, epochs=2)
            results[name]["pruned"][f"{int(amt*100)}"] = {
                "params": count_params(pruned),
                "latency_ms": sp["mean_ms"],
                "accuracy": acc,
            }
            print(f"  Pruned {int(amt*100)}% | acc: {acc:.2f}% | lat: {sp['mean_ms']:.2f} ms")

        # Dynamic quantization (Linear layers)
        quant = dynamic_quantize_linear_only(model)
        bench.register_model(f"{name}_quant", quant, sample_input)
        sq = bench.benchmark_speed(f"{name}_quant")
        qacc = quick_accuracy(copy.deepcopy(quant), train_x, train_y, test_x, test_y, epochs=2)
        results[name]["quantized"] = {
            "params": count_params(quant),
            "latency_ms": sq["mean_ms"],
            "accuracy": qacc,
        }
        print(f"  Quantized | acc: {qacc:.2f}% | lat: {sq['mean_ms']:.2f} ms")

    out = Path("research/results/compression_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved compression results to {out}")


if __name__ == "__main__":
    run()


