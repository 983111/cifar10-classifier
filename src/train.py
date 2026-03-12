"""
Training script for CIFAR-10 classifier.

Usage:
    python src/train.py
    python src/train.py --epochs 50 --batch-size 256 --lr 0.01
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

from data import get_loaders
from model import CIFAR10Net, count_parameters


# ──────────────────────────────────────────────
# Training / validation helpers
# ──────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scheduler, device):
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
        total   += imgs.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total   += imgs.size(0)

    return running_loss / total, correct / total


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train CIFAR-10 classifier")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch-size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=0.01)
    p.add_argument("--dropout",    type=float, default=0.4)
    p.add_argument("--data-dir",   type=str,   default="./data")
    p.add_argument("--out-dir",    type=str,   default="./outputs")
    p.add_argument("--workers",    type=int,   default=4)
    p.add_argument("--patience",   type=int,   default=10,
                   help="Early stopping patience (epochs without val improvement)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ──────────────────────────────────
    device = (
        torch.device("cuda")  if torch.cuda.is_available() else
        torch.device("mps")   if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Using device: {device}")

    # ── Data ────────────────────────────────────
    train_loader, val_loader, _ = get_loaders(
        args.data_dir, args.batch_size, num_workers=args.workers
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ───────────────────────────────────
    model = CIFAR10Net(dropout=args.dropout).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # ── Optimizer & Scheduler ───────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=0.9, weight_decay=5e-4, nesterov=True,
    )
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader), epochs=args.epochs,
        pct_start=0.3, anneal_strategy="cos",
    )

    # ── Training loop ───────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss,   val_acc   = evaluate(  model, val_loader,   criterion, device)

        elapsed = time.time() - t0
        history["train_loss"].append(train_loss)
        history["train_acc" ].append(train_acc)
        history["val_loss"  ].append(val_loss)
        history["val_acc"   ].append(val_acc)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        marker = "✓" if improved else " "
        print(
            f"Epoch {epoch:3d}/{args.epochs} [{elapsed:.1f}s] {marker} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"(best={best_val_acc:.4f})"
        )

        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    # ── Save artefacts ───────────────────────────
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Training complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"  Model saved to: {out_dir / 'best_model.pth'}")
    print(f"  History saved to: {out_dir / 'history.json'}")


if __name__ == "__main__":
    main()
