# src/main.py
# Run from project root: python src/main.py

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from train import train


def main():
    parser = argparse.ArgumentParser(description="Train CRNN on preprocessed FMA mel spectrograms")
    parser.add_argument("--data_dir", default="data", help="Directory containing X.npy and y.npy")
    parser.add_argument("--save_path", default="data/best_model.pt", help="Path to save best checkpoint")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rnn_hidden", type=int, default=128)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--n_classes", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8, help="DataLoader worker processes")
    args = parser.parse_args()

    # Memory-map X so the full ~16 GB tensor is never loaded into RAM at once
    X = np.load(os.path.join(args.data_dir, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(args.data_dir, "y.npy"))

    print(f"X: {X.shape}  y: {y.shape}  device: cuda/cpu auto  classes: {args.n_classes}")

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    train(
        X, y,
        n_classes=args.n_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        rnn_hidden=args.rnn_hidden,
        val_split=args.val_split,
        save_path=args.save_path,
        num_workers=args.workers,
    )

    print(f"\nBest checkpoint saved to {args.save_path}")


if __name__ == "__main__":
    main()
