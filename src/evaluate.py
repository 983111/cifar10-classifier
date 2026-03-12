"""
Evaluate a trained CIFAR-10 model on the test set.

Usage:
    python src/evaluate.py
    python src/evaluate.py --checkpoint outputs/best_model.pth
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from data import get_loaders, CLASSES
from model import CIFAR10Net


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)

    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CIFAR-10 classifier")
    p.add_argument("--checkpoint", type=str,   default="./outputs/best_model.pth")
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--data-dir",   type=str,   default="./data")
    p.add_argument("--out-dir",    type=str,   default="./outputs")
    p.add_argument("--workers",    type=int,   default=4)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device("cuda")  if torch.cuda.is_available() else
        torch.device("mps")   if torch.backends.mps.is_available() else
        torch.device("cpu")
    )

    # ── Load model ────────────────────────────
    model = CIFAR10Net().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Test loader ───────────────────────────
    _, _, test_loader = get_loaders(args.data_dir, args.batch_size, num_workers=args.workers)

    # ── Inference ─────────────────────────────
    preds, labels = run_inference(model, test_loader, device)

    # ── Metrics ───────────────────────────────
    overall_acc = (preds == labels).mean()
    print(f"\nOverall test accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)\n")
    print(classification_report(labels, preds, target_names=CLASSES, digits=4))

    # Per-class breakdown
    cm = confusion_matrix(labels, preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class accuracy:")
    for cls, acc in zip(CLASSES, per_class_acc):
        bar = "█" * int(acc * 30)
        print(f"  {cls:12s} {acc:.4f}  {bar}")

    # ── Save results ──────────────────────────
    results = {
        "overall_accuracy": float(overall_acc),
        "per_class_accuracy": {cls: float(a) for cls, a in zip(CLASSES, per_class_acc)},
        "confusion_matrix": cm.tolist(),
    }
    results_path = out_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
