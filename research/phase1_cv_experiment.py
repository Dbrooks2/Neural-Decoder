import json
import math
import os
import pathlib
from pathlib import Path
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Local imports (add demos/ and data/ to path)
HERE = pathlib.Path(__file__).resolve()
import sys
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parents[1] / "demos"))
sys.path.append(str(HERE.parents[1] / "data"))

from realistic_demo import EEGNet
from mobile_architectures import MobileEEGNet, TinyEEGNet
from realistic_eeg_data import generate_realistic_motor_imagery_eeg
from benchmark_framework import BCIBenchmark


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_models(samples: int) -> Dict[str, nn.Module]:
    return {
        "EEGNet": EEGNet(n_channels=22, n_classes=4, samples=samples),
        "MobileEEGNet": MobileEEGNet(n_channels=22, n_classes=4, samples=samples),
        "TinyEEGNet": TinyEEGNet(n_channels=22, n_classes=4, samples=samples),
    }


def train_one(model: nn.Module,
              train_x: torch.Tensor,
              train_y: torch.Tensor,
              val_x: torch.Tensor,
              val_y: torch.Tensor,
              lr: float = 1e-3,
              max_epochs: int = 30,
              patience: int = 5,
              batch_size: int = 64) -> Tuple[nn.Module, Dict[str, Any]]:
    device = torch.device("cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(max_epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * xb.size(0)
        train_loss = total / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x.to(device))
            val_loss = criterion(val_logits, val_y.to(device)).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"history": history}


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
    device = torch.device("cpu")
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
    y_true = y.numpy()
    acc = accuracy_score(y_true, preds) * 100.0
    macro_f1 = f1_score(y_true, preds, average="macro") * 100.0
    cm = confusion_matrix(y_true, preds).tolist()
    return {"accuracy": acc, "macro_f1": macro_f1, "confusion_matrix": cm}


def benchmark_latency(models: Dict[str, nn.Module], samples: int) -> Dict[str, Any]:
    bench = BCIBenchmark()
    sample_input = torch.randn(1, 22, samples)
    for name, m in models.items():
        bench.register_model(name, m, sample_input)
    out = {}
    for name in models.keys():
        out[name] = bench.benchmark_speed(name)
    return out


def plot_summary_bar(out_dir: Path, summary: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    models = list(summary.keys())
    acc = np.array([summary[m]["accuracy_mean"] for m in models])
    acc_std = np.array([summary[m]["accuracy_std"] for m in models])
    f1 = np.array([summary[m]["macro_f1_mean"] for m in models])
    f1_std = np.array([summary[m]["macro_f1_std"] for m in models])

    x = np.arange(len(models))
    width = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(x - width/2, acc, width, yerr=acc_std, capsize=4)
    ax1.set_xticks(x); ax1.set_xticklabels(models, rotation=20)
    ax1.set_ylabel("Accuracy (%)"); ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_title("Mean±Std (k-fold × seeds)")
    ax2.bar(x + width/2, f1, width, yerr=f1_std, capsize=4, color="#F58518")
    ax2.set_xticks(x); ax2.set_xticklabels(models, rotation=20)
    ax2.set_ylabel("Macro-F1 (%)"); ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "phase1_cv_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_history(out_dir: Path, model_name: str, seed: int, fold: int, history: Dict[str, List[float]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(history.get("train_loss", []), label="train_loss")
    ax.plot(history.get("val_loss", []), label="val_loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.grid(True, alpha=0.3)
    ax.set_title(f"{model_name} seed={seed} fold={fold} Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{model_name}_seed{seed}_fold{fold}_history.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_aggregate(out_dir: Path, all_metrics: Dict[str, List[Dict[str, Any]]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = ["Rest", "Left", "Right", "Both"]
    for name, runs in all_metrics.items():
        if not runs:
            continue
        agg = np.zeros((4, 4), dtype=float)
        for r in runs:
            cm = np.array(r.get("confusion_matrix", np.zeros((4, 4))))
            if cm.shape == (4, 4):
                agg += cm
        # Normalize rows to percentages
        row_sums = agg.sum(axis=1, keepdims=True) + 1e-8
        norm = (agg / row_sums) * 100.0
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{name} Confusion Matrix (Avg %)")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{norm[i, j]:.1f}", ha="center", va="center",
                        color=("white" if norm[i, j] > 50 else "black"))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_confusion_avg.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust Phase 1 CV Benchmark")
    parser.add_argument("--trials", type=int, default=1000, help="Number of synthetic trials")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4", help="Comma-separated random seeds")
    parser.add_argument("--epochs", type=int, default=30, help="Max epochs per training run")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--lrs", type=str, default="0.001,0.0005", help="Comma-separated learning rates")
    args = parser.parse_args()

    results_dir = HERE.parent / "results" / "phase1_cv"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Config
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    k_folds = int(args.folds)
    lrs = [float(s) for s in args.lrs.split(",") if s.strip() != ""]

    # Data
    print(f"Generating synthetic data: trials={args.trials}")
    data, labels = generate_realistic_motor_imagery_eeg(
        n_trials=int(args.trials), n_channels=22, fs=250, trial_length=2.0
    )
    x = torch.FloatTensor(data)
    y = torch.LongTensor(labels)
    samples = int(x.shape[2])

    # Latency baseline (untrained)
    print("Measuring latency baseline…")
    lat = benchmark_latency(build_models(samples), samples)

    # CV + seeds
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    all_metrics: Dict[str, List[Dict[str, Any]]] = {"EEGNet": [], "MobileEEGNet": [], "TinyEEGNet": []}

    total_runs = len(seeds) * k_folds * 3 * len(lrs)
    run_idx = 0
    for seed in seeds:
        set_seed(seed)
        for fold, (train_idx, test_idx) in enumerate(skf.split(x, y)):
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            # small val split for early stopping
            x_tr, x_val, y_tr, y_val = train_test_split(
                x_train, y_train, test_size=0.2, stratify=y_train, random_state=seed
            )

            models = build_models(samples)
            for name, model in models.items():
                best = {"score": -1.0, "lr": None, "model": None}
                best_history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
                for lr in lrs:
                    run_idx += 1
                    print(f"[seed={seed} fold={fold+1}/{k_folds} model={name} lr={lr}] ({run_idx}/{total_runs})")
                    m_copy = build_models(samples)[name]
                    trained, info = train_one(m_copy, x_tr, y_tr, x_val, y_val,
                                           lr=lr, max_epochs=int(args.epochs), patience=int(args.patience))
                    val_res = evaluate(trained, x_val, y_val)
                    if val_res["accuracy"] > best["score"]:
                        best = {"score": val_res["accuracy"], "lr": lr, "model": trained}
                        best_history = info.get("history", {"train_loss": [], "val_loss": []})
                # Evaluate on test
                test_res = evaluate(best["model"], x_test, y_test)
                test_res.update({"seed": seed, "fold": fold, "best_lr": best["lr"]})
                all_metrics[name].append(test_res)
                # Save training curve for this run
                plot_history(results_dir / "histories", name, seed, fold, best_history)

    # Aggregate
    summary: Dict[str, Any] = {}
    for name, runs in all_metrics.items():
        accs = np.array([r["accuracy"] for r in runs])
        f1s = np.array([r["macro_f1"] for r in runs])
        summary[name] = {
            "accuracy_mean": float(accs.mean()), "accuracy_std": float(accs.std()),
            "macro_f1_mean": float(f1s.mean()), "macro_f1_std": float(f1s.std()),
            "latency_mean_ms": float(lat[name]["mean_ms"]), "latency_p95_ms": float(lat[name]["p95_ms"]),
            "runs": runs,
        }

    # Save
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plot
    plot_summary_bar(results_dir, summary)
    plot_confusion_aggregate(results_dir / "confusions", all_metrics)
    print(f"Saved CV summary to {results_dir}")


if __name__ == "__main__":
    main()


