"""
CIFAR-10 data loading with augmentation for training.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).
    Splits the official 50k training set into train/val.
    """
    train_data = datasets.CIFAR10(data_dir, train=True,  download=True, transform=get_transforms(True))
    test_data  = datasets.CIFAR10(data_dir, train=False, download=True, transform=get_transforms(False))

    # Deterministic train/val split
    n_val   = int(len(train_data) * val_split)
    n_train = len(train_data) - n_val
    train_set, val_set = random_split(
        train_data, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    # Val set should use test-time transforms
    val_set.dataset.transform = get_transforms(False)

    loader_kwargs = dict(
        batch_size=batch_size, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = DataLoader(train_set, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_set,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_data, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
