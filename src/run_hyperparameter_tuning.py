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
    # Example: Load from npy files
    try:
        X = np.load("X.npy")
        y = np.load("y.npy")
        n_classes = len(np.unique(y))
        
        print(f"Loaded data:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Number of classes: {n_classes}")
        
        return X, y, n_classes
    
    except FileNotFoundError:
        print("Error: Could not find X.npy or y.npy")
        print("\nPlease ensure your data files are in the current directory:")
        print("  - X.npy (training features)")
        print("  - y.npy (training labels)")
        sys.exit(1)


def main():
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING RUNNER")
    print("="*70)
    
    # Load data
    print("\n[1/3] Loading data...")
    X, y, n_classes = load_data()
    
    # Initialize tuner
    print("\n[2/3] Initializing tuner...")
    tuner = HyperparameterTuner(X, y, n_classes, results_dir="results")
    print("✓ Tuner initialized")
    
    # Run all phases
    print("\n[3/3] Running tuning phases...")
    print("This may take a while depending on your data size and system.\n")
    tuner.run_all_phases()


if __name__ == "__main__":
    main()
