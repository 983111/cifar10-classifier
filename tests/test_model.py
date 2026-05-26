from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from model import CIFAR10Net


def test_cifar10net_output_shape() -> None:
    model = CIFAR10Net()
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    assert y.shape == (1, 10)
