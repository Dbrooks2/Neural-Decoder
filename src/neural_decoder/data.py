from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class DataConfig:
    num_channels: int = 32
    window_size: int = 64  # timesteps per window
    num_samples: int = 10000
    noise_std: float = 0.5
    seed: int = 42


class SyntheticNeuralDataset(Dataset):
    """
    Generates synthetic neural signals from latent 2D intention (cursor velocity).

    - Latent velocity is smooth (low-frequency random walk)
    - Neural channels are linear mixtures of the latent + Gaussian noise
    - Each sample returns a window of shape [num_channels, window_size] and target [2]
    """

    def __init__(self, cfg: DataConfig) -> None:
        super().__init__()
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)

        # Random linear mixing from latent (2 dims) to channels
        self.mixing = rng.normal(0.5, 0.2, size=(cfg.num_channels, 2)).astype(np.float32)

        # Pre-generate latent velocities and signals
        T = cfg.num_samples + cfg.window_size
        latent = self._generate_latent_velocity(T, rng)  # [T, 2]
        signals = self._mix_to_channels(latent, rng)     # [T, C]

        # Build sliding windows
        X = []
        Y = []
        for t in range(cfg.num_samples):
            window = signals[t : t + cfg.window_size]           # [W, C]
            target = latent[t + cfg.window_size - 1]            # [2]
            X.append(window.T)  # [C, W]
            Y.append(target)

        self.X = torch.tensor(np.stack(X, axis=0), dtype=torch.float32)
        self.Y = torch.tensor(np.stack(Y, axis=0), dtype=torch.float32)

        # Standardization stats per-channel
        self.channel_mean = self.X.mean(dim=(0, 2), keepdim=True)
        self.channel_std = self.X.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)

        # Normalize in-place (save stats for inference)
        self.X = (self.X - self.channel_mean) / self.channel_std

    @staticmethod
    def _generate_latent_velocity(T: int, rng: np.random.Generator) -> np.ndarray:
        # Low-frequency 2D random walk, then smoothed
        steps = rng.normal(0.0, 0.5, size=(T, 2)).astype(np.float32)
        latent = np.cumsum(steps, axis=0)
        # Exponential smoothing
        alpha = 0.05
        for t in range(1, T):
            latent[t] = alpha * latent[t] + (1 - alpha) * latent[t - 1]
        # Normalize range
        latent /= (np.std(latent, axis=0, keepdims=True) + 1e-6)
        return latent.astype(np.float32)

    def _mix_to_channels(self, latent: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # signals[t, c] = mixing[c]  latent[t] + noise
        base = latent @ self.mixing.T  # [T, C]
        noise = rng.normal(0.0, self.cfg.noise_std, size=base.shape).astype(np.float32)
        signals = base + noise
        return signals.astype(np.float32)

    def __len__(self) -> int:
        return self.cfg.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


def make_dataloaders(cfg: DataConfig, batch_size: int = 64, num_workers: int = 0):
    ds = SyntheticNeuralDataset(cfg)
    n_train = int(len(ds) * 0.9)
    n_val = len(ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    norm_stats = {
        "mean": ds.channel_mean.squeeze(0).squeeze(-1).numpy(),
        "std": ds.channel_std.squeeze(0).squeeze(-1).numpy(),
    }
    return train_dl, val_dl, norm_stats