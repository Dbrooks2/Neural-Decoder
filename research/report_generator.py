import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


RESULTS_DIR = Path(__file__).resolve().parent / "results"
PHASE1_JSON = RESULTS_DIR / "phase1_summary.json"
COMPRESSION_JSON = RESULTS_DIR / "compression_summary.json"
STREAMING_JSON = RESULTS_DIR / "streaming_summary.json"
REPORT_PDF = RESULTS_DIR / "bci_benchmark_report.pdf"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fig_title(ax, title: str):
    ax.set_title(title, fontsize=12, pad=10)


def plot_phase1_summary(pp: PdfPages, data: Dict[str, Any]):
    if not data:
        return

    models = list(data.keys())
    acc = np.array([data[m]["accuracy_percent"] for m in models], dtype=float)
    lat = np.array([data[m]["latency_ms_mean"] for m in models], dtype=float)
    size = np.array([data[m]["size_mb"] for m in models], dtype=float)
    params = np.array([data[m]["params"] for m in models], dtype=float)

    # 1) Speed vs Accuracy (bubble=size)
    fig, ax = plt.subplots(figsize=(7, 5))
    bubble = (size / (size.max() if size.max() > 0 else 1.0) + 0.1) * 800
    ax.scatter(lat, acc, s=bubble, alpha=0.6)
    for i, m in enumerate(models):
        ax.annotate(m, (lat[i], acc[i]), fontsize=9, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, alpha=0.3)
    fig_title(ax, "Phase 1: Speed vs Accuracy (bubble ∝ size MB)")
    pp.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # 2) Params and Size bars
    x = np.arange(len(models))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(x, params / 1e3)
    ax1.set_xticks(x); ax1.set_xticklabels(models, rotation=20)
    ax1.set_ylabel("Params (K)"); ax1.grid(True, axis="y", alpha=0.3)
    fig_title(ax1, "Model Parameters")

    ax2.bar(x, size)
    ax2.set_xticks(x); ax2.set_xticklabels(models, rotation=20)
    ax2.set_ylabel("Size (MB)"); ax2.grid(True, axis="y", alpha=0.3)
    fig_title(ax2, "Model Size")
    pp.savefig(fig, bbox_inches="tight"); plt.close(fig)


def plot_compression_summary(pp: PdfPages, data: Dict[str, Any]):
    if not data:
        return

    for model_name, sections in data.items():
        # Build arrays for baseline/pruned/quant
        labels: List[str] = ["baseline"]
        acc: List[float] = [sections.get("baseline", {}).get("accuracy", np.nan)]
        lat: List[float] = [sections.get("baseline", {}).get("latency_ms", np.nan)]

        pruned = sections.get("pruned", {})
        for key in sorted(pruned.keys(), key=lambda k: int(k)):
            labels.append(f"pruned_{key}%")
            acc.append(pruned[key].get("accuracy", np.nan))
            lat.append(pruned[key].get("latency_ms", np.nan))

        q = sections.get("quantized", {})
        if q:
            labels.append("quantized")
            acc.append(q.get("accuracy", np.nan))
            lat.append(q.get("latency_ms", np.nan))

        x = np.arange(len(labels))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(x, acc, color="#4C78A8")
        ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20)
        ax1.set_ylabel("Accuracy (%)")
        ax1.grid(True, axis="y", alpha=0.3)
        fig_title(ax1, f"{model_name} – Accuracy vs Compression")

        ax2.bar(x, lat, color="#F58518")
        ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=20)
        ax2.set_ylabel("Latency (ms)")
        ax2.grid(True, axis="y", alpha=0.3)
        fig_title(ax2, f"{model_name} – Latency vs Compression")

        pp.savefig(fig, bbox_inches="tight"); plt.close(fig)


def parse_streaming_grid(d: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Extract distinct windows and overlaps
    keys = list(d.keys())
    windows = sorted({int(k.split("_")[0][1:]) for k in keys})
    ovpcts = sorted({int(k.split("_")[1][2:]) for k in keys})

    mean_grid = np.zeros((len(windows), len(ovpcts)))
    p95_grid = np.zeros_like(mean_grid)
    for i, w in enumerate(windows):
        for j, ov in enumerate(ovpcts):
            entry = d.get(f"w{w}_ov{ov}", {})
            mean_grid[i, j] = entry.get("mean_ms", np.nan)
            p95_grid[i, j] = entry.get("p95_ms", np.nan)
    return np.array(windows), np.array(ovpcts), mean_grid, p95_grid


def plot_streaming_summary(pp: PdfPages, data: Dict[str, Any]):
    if not data:
        return

    for model_name, grid in data.items():
        windows, ovpcts, mean_grid, p95_grid = parse_streaming_grid(grid)

        # Heatmaps: mean and p95
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        im1 = ax1.imshow(mean_grid, aspect="auto", cmap="viridis")
        ax1.set_xticks(range(len(ovpcts))); ax1.set_xticklabels([f"{o}%" for o in ovpcts])
        ax1.set_yticks(range(len(windows))); ax1.set_yticklabels([str(w) for w in windows])
        ax1.set_xlabel("Overlap (%)"); ax1.set_ylabel("Window (samples)")
        fig_title(ax1, f"{model_name} – Streaming Mean Latency (ms)")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        im2 = ax2.imshow(p95_grid, aspect="auto", cmap="magma")
        ax2.set_xticks(range(len(ovpcts))); ax2.set_xticklabels([f"{o}%" for o in ovpcts])
        ax2.set_yticks(range(len(windows))); ax2.set_yticklabels([str(w) for w in windows])
        ax2.set_xlabel("Overlap (%)"); ax2.set_ylabel("Window (samples)")
        fig_title(ax2, f"{model_name} – Streaming P95 Latency (ms)")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        pp.savefig(fig, bbox_inches="tight"); plt.close(fig)


def main():
    phase1 = load_json(PHASE1_JSON)
    compression = load_json(COMPRESSION_JSON)
    streaming = load_json(STREAMING_JSON)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with PdfPages(REPORT_PDF) as pp:
        # Cover page
        fig, ax = plt.subplots(figsize=(7.5, 10))
        ax.axis("off")
        text = [
            "Neural Decoder Benchmark Report",
            "",
            "Contents:",
            "  • Phase 1: Speed vs Accuracy vs Size",
            "  • Compression: Pruning & Quantization Trade-offs",
            "  • Streaming: Window/Overlap Latency Analysis",
        ]
        ax.text(0.05, 0.95, text[0], fontsize=18, va="top")
        ax.text(0.05, 0.80, "\n".join(text[2:]), fontsize=12, va="top")
        pp.savefig(fig, bbox_inches="tight"); plt.close(fig)

        plot_phase1_summary(pp, phase1)
        plot_compression_summary(pp, compression)
        plot_streaming_summary(pp, streaming)

    print(f"Saved report to {REPORT_PDF}")


if __name__ == "__main__":
    main()


