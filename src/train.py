# src/train.py

import os
import time

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from models.crnn import CRNN
from data.dataset import MelSpectrogramDataset


def train(X, y, n_classes=16, epochs=20, batch_size=32, lr=1e-3,
          rnn_hidden=128, val_split=0.2, save_path=None, num_workers=4, return_metrics=False):
    """
    Train CRNN model with optional metrics tracking.

    Args:
        X: Input features (numpy array or memmap, shape N x 1 x n_mels x time)
        y: Labels (numpy array, shape N)
        n_classes: Number of output classes
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        rnn_hidden: Number of RNN hidden units
        val_split: Fraction of data held out for validation
        save_path: If set, best checkpoint (by val accuracy) is saved here
        num_workers: DataLoader worker processes for prefetching
        return_metrics: If True, return detailed metrics instead of just model

    Returns:
        If return_metrics is False:
            model: Trained model
        If return_metrics is True:
            dict with keys: model, metrics (containing loss, accuracy, etc.)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Track metrics
    metrics = {
        "device": str(device),
        "epochs_trained": 0,
        "total_training_time": 0.0,
        "peak_memory_mb": 0.0,
        "epoch_history": [],
    }

    n = len(y)
    idx = np.random.default_rng(42).permutation(n)
    split = int(n * (1 - val_split))
    train_idx, val_idx = idx[:split], idx[split:]

    full_dataset = MelSpectrogramDataset(X, y)
    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=device.type == "cuda",
        persistent_workers=True,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
        persistent_workers=True,
    )

    model = CRNN(n_classes=n_classes, rnn_hidden=rnn_hidden).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Get process for memory tracking
    process = psutil.Process(os.getpid())

    start_time = time.time()
    best_val_acc = 0.0

    for epoch in range(epochs):
        # --- train ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        # Memory tracking per epoch
        process.memory_info()  # warm up

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(preds, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # --- validate ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                _, predicted = torch.max(model(X_batch), 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        val_acc = val_correct / val_total

        # Get memory usage
        mem_mb = process.memory_info().rss / 1024 / 1024
        metrics["peak_memory_mb"] = max(metrics["peak_memory_mb"], mem_mb)

        saved = False
        if save_path and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            saved = True

        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "memory_mb": mem_mb,
        }
        metrics["epoch_history"].append(epoch_data)
        ckpt = " *" if saved else ""
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Mem: {mem_mb:.1f}MB{ckpt}")

    metrics["total_training_time"] = time.time() - start_time
    metrics["epochs_trained"] = epochs

    if return_metrics:
        return {"model": model, "metrics": metrics}
    return model
