from __future__ import annotations

import torch
from torch import nn


class CNNLSTMDecoder(nn.Module):
    """
    A lightweight CNN+LSTM decoder for spatiotemporal neural signals.

    Input shape: [batch, num_channels, window_size]
    Output: [batch, 2] -> (dx, dy) cursor velocity
    """

    def __init__(
        self,
        num_channels: int,
        window_size: int,
        conv_channels: int = 32,
        lstm_hidden_size: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.window_size = window_size

        # Time-domain convolution that mixes across channels and extracts temporal features
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=num_channels,
                out_channels=conv_channels,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
        )

        # LSTM over time (sequence length = window_size)
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

        # Small initialization for stability
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.inference_mode()
    def warmup(self, device: torch.device | str = "cpu") -> None:
        """Run a dummy forward pass to initialize kernels/caches for low-latency inference."""
        self.eval()
        x = torch.zeros(1, self.num_channels, self.window_size, device=device)
        _ = self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, num_channels, window_size] -> (dx, dy)"""
        feats = self.temporal_conv(x)  # [B, C', T]
        feats = feats.transpose(1, 2)  # [B, T, C'] -> sequence first for LSTM
        out, _ = self.lstm(feats)      # [B, T, H]
        last = out[:, -1, :]           # [B, H]
        pred = self.head(last)         # [B, 2]
        return pred


def build_model(num_channels: int, window_size: int) -> CNNLSTMDecoder:
    return CNNLSTMDecoder(
        num_channels=num_channels,
        window_size=window_size,
        conv_channels=32,
        lstm_hidden_size=64,
        lstm_layers=1,
        dropout=0.1,
    )