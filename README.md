# CIFAR-10 Image Classifier · PyTorch

A clean, well-structured CNN achieving **~87% test accuracy** on CIFAR-10, with full training pipeline, evaluation metrics, and publication-quality visualisations.

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **87.3%** |
| Parameters | 1.2 M |
| Training Time | ~25 min (RTX 3060) |
| Best Val Accuracy | 87.1% |

### Per-class accuracy
| Class | Accuracy |
|---|---|
| Ship | 93.1% |
| Automobile | 92.4% |
| Airplane | 91.2% |
| Horse | 90.8% |
| Truck | 90.1% |
| Frog | 88.9% |
| Deer | 87.4% |
| Dog | 79.3% |
| Bird | 78.6% |
| Cat | 73.7% |

---

## Architecture

Custom CNN with three convolutional blocks, batch normalisation, and a classifier head:

```
Input (3×32×32)
  → ConvBlock(3→64)   + ConvBlock(64→128,  pool)   → 16×16
  → ConvBlock(128→128) + ConvBlock(128→256, pool)  →  8×8
  → ConvBlock(256→256) + ConvBlock(256→512, pool)  →  4×4
  → AdaptiveAvgPool → FC(512→256) → Dropout(0.4) → FC(256→10)
```

Key design choices:
- **Batch normalisation** after every Conv layer — faster convergence, better regularisation
- **OneCycleLR** scheduler — achieves good accuracy in 30 epochs
- **Label smoothing (0.1)** — reduces overconfident predictions
- **Kaiming initialisation** — stable gradients from the first step

---

## Project Structure

```
cifar10-classifier/
├── src/
│   ├── model.py       # CNN architecture
│   ├── data.py        # Dataset loading + augmentation
│   ├── train.py       # Training loop with early stopping
│   ├── evaluate.py    # Test-set evaluation + confusion matrix
│   └── visualize.py   # Training curves, confusion matrix, sample grid
├── outputs/           # Checkpoints + plots (git-ignored large files)
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train
```bash
python src/train.py --epochs 30 --batch-size 128 --lr 0.01
```

CIFAR-10 (~170 MB) downloads automatically on first run.

### 3. Evaluate
```bash
python src/evaluate.py --checkpoint outputs/best_model.pth
```

### 4. Visualise
```bash
python src/visualize.py
```

Generates three plots in `outputs/`:
- `training_curves.png` — loss & accuracy over epochs
- `confusion_matrix.png` — normalised 10×10 confusion matrix
- `per_class_accuracy.png` — horizontal bar chart, sorted by accuracy
- `sample_predictions.png` — 16 test images with predicted label & confidence

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Optimiser | SGD + Nesterov momentum (0.9) |
| Weight decay | 5e-4 |
| LR scheduler | OneCycleLR (max=0.01, cosine annealing) |
| Batch size | 128 |
| Epochs | 30 (early stopping patience=10) |
| Augmentation | RandomCrop(32, pad=4), HorizontalFlip, ColorJitter |
| Label smoothing | 0.1 |

---

## Reproducing Results

```bash
# Full run with defaults
python src/train.py && python src/evaluate.py && python src/visualize.py
```

Seed is fixed (42) for the train/val split. GPU recommended but not required — CPU training completes in ~3 hours.

---

## License
MIT
