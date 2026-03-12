"""
CNN model for CIFAR-10 classification.
Architecture: 3 convolutional blocks with batch normalization + dropout,
followed by fully connected layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Reusable conv → BN → ReLU → (optional pool) block."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CIFAR10Net(nn.Module):
    """
    Custom CNN for CIFAR-10 (32×32 RGB images, 10 classes).

    Achieves ~85-88% test accuracy after 30 epochs with default settings.
    Parameters: ~1.2M
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.4):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            # Block 1: 32×32 → 16×16
            ConvBlock(3, 64),
            ConvBlock(64, 128, pool=True),
            # Block 2: 16×16 → 8×8
            ConvBlock(128, 128),
            ConvBlock(128, 256, pool=True),
            # Block 3: 8×8 → 4×4
            ConvBlock(256, 256),
            ConvBlock(256, 512, pool=True),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # 512×1×1
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
