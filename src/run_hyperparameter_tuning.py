# src/run_hyperparameter_tuning.py

"""
Example script to run the hyperparameter tuning strategy.

Usage:
    python run_hyperparameter_tuning.py

This will:
    1. Load your data (X, y)
    2. Run Phase 1: Test learning rates
    3. Run Phase 2: Test batch sizes
    4. Run Phase 3: Test RNN hidden units
    5. Generate results and comparison report
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from hyperparameter_tuning import HyperparameterTuner

def load_data():
    """
    Load your training data here.
    Replace this with your actual data loading logic.
    
    Returns:
        X: Training features (N, 1, n_mels, time)
        y: Training labels (N,)
        n_classes: Number of classes
    """
    # Load from npy files — run from project root
    try:
        # Memory-map X so the full ~16 GB tensor is never loaded into RAM at once
        X = np.load("data/X.npy", mmap_mode="r")
        y = np.load("data/y.npy")
        n_classes = len(np.unique(y))

        print(f"Loaded data:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Number of classes: {n_classes}")

        return X, y, n_classes

    except FileNotFoundError:
        print("Error: Could not find data/X.npy or data/y.npy")
        print("\nRun preprocessing first: python src/data/preprocess.py")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run 3-phase hyperparameter tuning")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Epochs per tuning run (default 15 — ~4-5 hrs for all 9 runs)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="L2 weight decay applied to all runs")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING RUNNER")
    print("="*70)
    print(f"Epochs per run: {args.epochs}  |  Weight decay: {args.weight_decay}")

    # Load data
    print("\n[1/3] Loading data...")
    X, y, n_classes = load_data()

    # Initialize tuner
    print("\n[2/3] Initializing tuner...")
    tuner = HyperparameterTuner(X, y, n_classes, epochs=args.epochs, weight_decay=args.weight_decay)
    print("[ok] Tuner initialized")

    # Run all phases
    print("\n[3/3] Running tuning phases...")
    print("This may take a while depending on your data size and system.\n")
    tuner.run_all_phases()


if __name__ == "__main__":
    main()
