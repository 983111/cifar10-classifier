"""Training script for CIFAR-10 classifier."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
from torch.utils.data import DataLoader

from config import apply_cli_overrides, load_config
from data import get_loaders
from model import CIFAR10Net, count_parameters

logger = logging.getLogger(__name__)


def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)

    return running_loss / total, correct / total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CIFAR-10 classifier")
    p.add_argument("--config", type=str, default="./config.yaml")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--dropout", type=float)
    p.add_argument("--data-dir", type=str)
    p.add_argument("--out-dir", type=str)
    p.add_argument("--workers", type=int)
    p.add_argument("--patience", type=int)
    p.add_argument("--use-wandb", action="store_true")
    return p.parse_args()


def maybe_init_wandb(cfg: dict[str, Any]) -> Any:
    if not cfg["use_wandb"]:
        return None
    try:
        import wandb

        run = wandb.init(project=cfg["wandb_project"], config=cfg)
        logger.info("W&B initialized: %s", run.name)
        return run
    except ImportError:
        logger.warning("wandb requested but not installed; continuing without it.")
        return None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_cli_overrides(
        cfg,
        {
            "train.epochs": args.epochs,
            "train.batch_size": args.batch_size,
            "train.lr": args.lr,
            "train.dropout": args.dropout,
            "train.data_dir": args.data_dir,
            "train.out_dir": args.out_dir,
            "train.workers": args.workers,
            "train.patience": args.patience,
            "train.use_wandb": args.use_wandb or None,
        },
    )
    train_cfg = cfg.train
    out_dir = Path(train_cfg.out_dir)
    setup_logging(out_dir)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    logger.info("Using device: %s", device)

    train_loader, val_loader, _ = get_loaders(train_cfg.data_dir, train_cfg.batch_size, num_workers=train_cfg.workers)
    logger.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    model = CIFAR10Net(dropout=float(train_cfg.dropout)).to(device)
    logger.info("Parameters: %s", f"{count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=train_cfg.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = OneCycleLR(optimizer, max_lr=train_cfg.lr, steps_per_epoch=len(train_loader), epochs=train_cfg.epochs, pct_start=0.3, anneal_strategy="cos")
    wandb_run = maybe_init_wandb(dict(train_cfg))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, int(train_cfg.epochs) + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        logger.info(
            "Epoch %3d/%d [%.1fs] %s train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f lr=%.6f (best=%.4f)",
            epoch,
            train_cfg.epochs,
            elapsed,
            "✓" if improved else " ",
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            current_lr,
            best_val_acc,
        )
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "lr": current_lr})

        if patience_counter >= int(train_cfg.patience):
            logger.info("Early stopping triggered after %d epochs.", epoch)
            break

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Training complete. Best val accuracy: %.4f", best_val_acc)


if __name__ == "__main__":
    main()
