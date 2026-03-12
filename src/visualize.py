"""
Generate publication-quality plots from training history and eval results.

Usage:
    python src/visualize.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import torch

from data import get_loaders, CLASSES, CIFAR10_MEAN, CIFAR10_STD
from model import CIFAR10Net


# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":       "#0f1117",
    "surface":  "#1a1d27",
    "accent1":  "#6c63ff",   # purple
    "accent2":  "#ff6584",   # coral
    "text":     "#e2e8f0",
    "subtext":  "#718096",
    "grid":     "#2d3748",
}

def apply_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["surface"],
        "axes.edgecolor":    PALETTE["grid"],
        "axes.labelcolor":   PALETTE["text"],
        "xtick.color":       PALETTE["subtext"],
        "ytick.color":       PALETTE["subtext"],
        "text.color":        PALETTE["text"],
        "grid.color":        PALETTE["grid"],
        "grid.linewidth":    0.6,
        "font.family":       "monospace",
    })

# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History — CIFAR-10 Classifier", fontsize=14, fontweight="bold",
                 color=PALETTE["text"], y=1.02)

    epochs = range(1, len(history["train_loss"]) + 1)

    for ax, (metric, title) in zip(axes, [("loss", "Loss"), ("acc", "Accuracy")]):
        train_vals = history[f"train_{metric}"]
        val_vals   = history[f"val_{metric}"]

        ax.plot(epochs, train_vals, color=PALETTE["accent1"], linewidth=2, label="Train")
        ax.plot(epochs, val_vals,   color=PALETTE["accent2"], linewidth=2, label="Val",
                linestyle="--")
        ax.fill_between(epochs, train_vals, val_vals,
                        alpha=0.08, color=PALETTE["accent1"])

        # Best val marker
        if metric == "acc":
            best_ep = int(np.argmax(val_vals)) + 1
            best_val = max(val_vals)
            ax.axvline(best_ep, color=PALETTE["accent2"], linewidth=1, linestyle=":")
            ax.annotate(f"Best: {best_val:.4f}",
                        xy=(best_ep, best_val),
                        xytext=(best_ep + 1, best_val - 0.03),
                        color=PALETTE["accent2"], fontsize=9,
                        arrowprops=dict(arrowstyle="->", color=PALETTE["accent2"]))

        ax.set_title(title, color=PALETTE["text"], fontsize=12)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.4)
        ax.legend(framealpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_confusion_matrix(results: dict, out_path: Path):
    cm = np.array(results["confusion_matrix"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    cmap = LinearSegmentedColormap.from_list(
        "custom", ["#1a1d27", PALETTE["accent1"], "#ffffff"]
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted", labelpad=10)
    ax.set_ylabel("True",      labelpad=10)
    ax.set_title("Confusion Matrix (normalised)", fontsize=13, fontweight="bold", pad=15)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=PALETTE["subtext"])

    # Cell annotations
    for i in range(10):
        for j in range(10):
            val = cm_norm[i, j]
            color = "white" if val < 0.5 else "#0f1117"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color=color)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_per_class_accuracy(results: dict, out_path: Path):
    classes = list(results["per_class_accuracy"].keys())
    accs    = list(results["per_class_accuracy"].values())
    order   = np.argsort(accs)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [PALETTE["accent1"] if a >= 0.85 else
              PALETTE["accent2"] if a < 0.75 else
              "#48bb78" for a in np.array(accs)[order]]

    bars = ax.barh([classes[i] for i in order], [accs[i] for i in order],
                   color=colors, height=0.65, edgecolor="none")

    ax.axvline(results["overall_accuracy"], color="white", linewidth=1.2,
               linestyle="--", alpha=0.6, label=f"Overall: {results['overall_accuracy']:.4f}")

    for bar, val in zip(bars, np.array(accs)[order]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color=PALETTE["text"])

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Accuracy")
    ax.set_title("Per-Class Test Accuracy", fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.2)
    ax.grid(axis="x", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    print(f"  Saved: {out_path}")
    plt.close(fig)


@torch.no_grad()
def plot_sample_predictions(checkpoint: str, data_dir: str, out_path: Path):
    device = torch.device("cpu")
    model = CIFAR10Net().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    _, _, test_loader = get_loaders(data_dir, batch_size=64, num_workers=0)
    imgs, labels = next(iter(test_loader))
    outputs = model(imgs)
    probs   = torch.softmax(outputs, dim=1)
    preds   = outputs.argmax(dim=1)

    # Un-normalise for display
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    n = 16
    fig = plt.figure(figsize=(14, 7))
    gs  = gridspec.GridSpec(2, 8, figure=fig, hspace=0.5, wspace=0.3)

    for idx in range(n):
        row, col = divmod(idx, 8)
        ax = fig.add_subplot(gs[row, col])

        img = (imgs[idx] * std + mean).permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(img)
        ax.axis("off")

        pred_cls  = CLASSES[preds[idx]]
        true_cls  = CLASSES[labels[idx]]
        conf      = probs[idx, preds[idx]].item()
        correct   = preds[idx] == labels[idx]

        color = "#68d391" if correct else PALETTE["accent2"]
        ax.set_title(f"{pred_cls}\n{conf:.0%}", fontsize=7.5,
                     color=color, pad=2)

    fig.suptitle("Sample Predictions  (green = correct, red = wrong)",
                 fontsize=11, color=PALETTE["text"], y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate visualisations")
    p.add_argument("--out-dir",    type=str, default="./outputs")
    p.add_argument("--data-dir",   type=str, default="./data")
    p.add_argument("--checkpoint", type=str, default="./outputs/best_model.pth")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_style()
    print("Generating plots…")

    # Training curves
    history_path = out_dir / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        plot_training_curves(history, out_dir / "training_curves.png")
    else:
        print(f"  Skipping training curves (no {history_path})")

    # Confusion matrix + per-class accuracy
    results_path = out_dir / "eval_results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
        plot_confusion_matrix(     results, out_dir / "confusion_matrix.png")
        plot_per_class_accuracy(   results, out_dir / "per_class_accuracy.png")
    else:
        print(f"  Skipping eval plots (no {results_path})")

    # Sample predictions
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        plot_sample_predictions(str(ckpt), args.data_dir, out_dir / "sample_predictions.png")
    else:
        print(f"  Skipping sample predictions (no checkpoint at {ckpt})")

    print("\nAll done. Plots saved to:", out_dir)


if __name__ == "__main__":
    main()
