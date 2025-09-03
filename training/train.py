from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from .models import build_model
from .data import DataConfig, make_dataloaders


def train_epoch(model, dl, optimizer, device):
    model.train()
    total = 0.0
    count = 0
    loss_fn = nn.SmoothL1Loss()
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * x.size(0)
        count += x.size(0)
    return total / max(count, 1)


def eval_epoch(model, dl, device):
    model.eval()
    total = 0.0
    count = 0
    loss_fn = nn.SmoothL1Loss()
    with torch.inference_mode():
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            total += loss.item() * x.size(0)
            count += x.size(0)
    return total / max(count, 1)


def save_artifacts(model, norm_stats, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Plain state dict
    torch.save({"state_dict": model.state_dict()}, out_dir / "model.pt")

    # Scripted model for faster CPU inference
    model.eval()
    example = torch.zeros(1, model.num_channels, model.window_size)
    scripted = torch.jit.trace(model, example)
    scripted.save(str(out_dir / "model_scripted.pt"))

    # Normalization stats
    np.savez(out_dir / "normalizer.npz", mean=norm_stats["mean"], std=norm_stats["std"]) 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_channels", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise_std", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="artifacts")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = DataConfig(
        num_channels=args.num_channels,
        window_size=args.window_size,
        num_samples=args.num_samples,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    train_dl, val_dl, norm_stats = make_dataloaders(cfg, batch_size=args.batch_size, num_workers=0)

    model = build_model(cfg.num_channels, cfg.window_size).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_dl, optimizer, device)
        va_loss = eval_epoch(model, val_dl, device)
        if va_loss < best_val:
            best_val = va_loss
            save_artifacts(model, norm_stats, Path(args.out_dir))
        print(f"epoch={epoch} train={tr_loss:.4f} val={va_loss:.4f}")
    print(f"done in {time.time() - start:.1f}s; best_val={best_val:.4f}")


if __name__ == "__main__":
    main()