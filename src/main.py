# src/main.py
# Run from project root: python src/main.py

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from train import train
from utils.logger import RunLogger


def main():
    parser = argparse.ArgumentParser(description="Train CRNN on preprocessed FMA mel spectrograms")
    parser.add_argument("--data_dir", default="data", help="Directory containing X.npy and y.npy")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rnn_hidden", type=int, default=128)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--n_classes", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8, help="DataLoader worker processes")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization strength")
    parser.add_argument("--results_dir", default="results", help="Root directory for run artifacts")
    parser.add_argument("--run_label", default="final_run", help="Human-readable label for this run")
    parser.add_argument("--confusion_matrix", action="store_true",
                        help="Compute and save confusion matrix on val set after training")
    parser.add_argument("--class_weights", action="store_true",
                        help="Use inverse-frequency class weights in CrossEntropyLoss")
    args = parser.parse_args()

    X = np.load(os.path.join(args.data_dir, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(args.data_dir, "y.npy"))
    print(f"X: {X.shape}  y: {y.shape}  classes: {args.n_classes}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.results_dir, f"{args.run_label}_{timestamp}")
    save_path = os.path.join(run_dir, "best_model.pt")
    os.makedirs(run_dir, exist_ok=True)

    config = {
        "run_label": args.run_label,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "rnn_hidden": args.rnn_hidden,
        "val_split": args.val_split,
        "n_classes": args.n_classes,
        "weight_decay": args.weight_decay,
        "workers": args.workers,
        "data_dir": args.data_dir,
        "dataset_size": int(X.shape[0]),
    }
    logger = RunLogger(run_dir=run_dir, config=config)

    train(
        X, y,
        n_classes=args.n_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        rnn_hidden=args.rnn_hidden,
        val_split=args.val_split,
        save_path=save_path,
        num_workers=args.workers,
        weight_decay=args.weight_decay,
        logger=logger,
        save_confusion_matrix=args.confusion_matrix,
        use_class_weights=args.class_weights,
    )


if __name__ == "__main__":
    main()
