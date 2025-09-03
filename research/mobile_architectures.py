import torch
import torch.nn as nn
import torch.nn.functional as F


class MobileEEGNet(nn.Module):
    """Mobile-optimized EEGNet variant using depthwise separable convolutions."""

    def __init__(self, n_channels: int = 22, n_classes: int = 4, samples: int = 1000):
        super().__init__()
        # Temporal conv on the time axis (shared across channels via 2D conv)
        self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 32), padding=(0, 16), bias=False)
        # Depthwise spatial conv across channels
        self.spatial_conv = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(n_channels, 1), groups=8, bias=False)

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)

        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout = nn.Dropout(p=0.25)

        # Lightweight separable temporal conv block
        self.sep_temporal = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 8), padding=(0, 4), groups=16, bias=False),
            nn.Conv2d(16, 32, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = x.unsqueeze(1)  # [B, 1, C, T]
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = F.elu(x)

        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.sep_temporal(x)
        x = self.pool2(x)
        x = self.dropout(x)
        return self.head(x)


class TinyEEGNet(nn.Module):
    """Ultra-lightweight 1D-CNN for very low-latency environments (e.g., edge)."""

    def __init__(self, n_channels: int = 22, n_classes: int = 4, samples: int = 1000):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=8, kernel_size=16, stride=4, padding=8, bias=False)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=8, stride=2, padding=4, bias=False)
        self.bn2 = nn.BatchNorm1d(16)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(16, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


class MobileEEGNet(nn.Module):
    def __init__(self, n_channels=22, n_classes=4, samples=1000):
        super().__init__()
        self.temporal_conv = nn.Conv2d(1, 8, (1, 32), padding=(0, 16))
        self.spatial_conv = nn.Conv2d(8, 16, (n_channels, 1), groups=8)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout = nn.Dropout(0.25)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, n_classes)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.elu(self.temporal_conv(x))
        x = self.pool1(F.elu(self.spatial_conv(x)))
        x = self.dropout(x)
        x = self.pool2(x)
        return self.classifier(x)

class TinyEEGNet(nn.Module):
    def __init__(self, n_channels=22, n_classes=4, samples=1000):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 8, kernel_size=16, stride=4)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=8, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, n_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
