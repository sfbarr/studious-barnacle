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

try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_available = True
except Exception:
    _nvml_available = False


def _gpu_stats(device):
    """Return (gpu_util_pct, gpu_mem_mb) for the active CUDA device, or (None, None)."""
    if device.type != "cuda":
        return None, None
    gpu_mem_mb = torch.cuda.memory_allocated(device) / 1024 ** 2
    gpu_util_pct = None
    if _nvml_available:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)
            gpu_util_pct = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        except Exception:
            pass
    return gpu_util_pct, gpu_mem_mb


def train(X, y, n_classes=16, epochs=20, batch_size=32, lr=1e-3,
          rnn_hidden=128, val_split=0.2, save_path=None, num_workers=4,
          weight_decay=1e-4, return_metrics=False, logger=None,
          save_confusion_matrix=False, use_class_weights=False):
    """
    Train CRNN model.

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
        weight_decay: L2 regularization strength for Adam optimizer
        return_metrics: If True, return detailed metrics dict alongside model
        logger: Optional RunLogger; if provided, all output goes through it
        save_confusion_matrix: If True and logger is set, save val confusion matrix
        use_class_weights: If True, weight loss by inverse class frequency
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    if use_class_weights:
        counts = np.bincount(y[train_idx], minlength=n_classes).astype(float)
        w = torch.tensor(len(train_idx) / (n_classes * counts), dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    process = psutil.Process(os.getpid())

    run_start = time.time()
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()

        # --- train ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        batch_times = []
        process.memory_info()  # warm up

        for X_batch, y_batch in train_loader:
            batch_start = time.perf_counter()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            batch_times.append(time.perf_counter() - batch_start)

            total_loss += loss.item()
            _, predicted = torch.max(preds, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        batch_time_mean = float(np.mean(batch_times))
        batch_time_std = float(np.std(batch_times))

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

        mem_mb = process.memory_info().rss / 1024 / 1024
        metrics["peak_memory_mb"] = max(metrics["peak_memory_mb"], mem_mb)
        epoch_time = time.time() - epoch_start
        gpu_util_pct, gpu_mem_mb = _gpu_stats(device)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            if save_path:
                torch.save(model.state_dict(), save_path)

        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "memory_mb": mem_mb,
        }
        metrics["epoch_history"].append(epoch_data)

        if logger:
            logger.log_epoch(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                epoch_time_sec=epoch_time,
                memory_mb=mem_mb,
                is_best=is_best,
                batch_time_mean_sec=batch_time_mean,
                batch_time_std_sec=batch_time_std,
                gpu_util_pct=gpu_util_pct,
                gpu_mem_mb=gpu_mem_mb,
            )
        else:
            marker = " *" if is_best else ""
            print(
                f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
                f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
                f"Mem: {mem_mb:.1f} MB{marker}"
            )

    total_time = time.time() - run_start
    metrics["total_training_time"] = total_time
    metrics["epochs_trained"] = epochs

    # --- collect val predictions once for metrics + optional confusion matrix ---
    if logger:
        from sklearn.metrics import (balanced_accuracy_score, classification_report,
                                     f1_score)
        model.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                batch_preds = torch.argmax(model(X_batch), dim=1).cpu().numpy()
                all_pred.extend(batch_preds)
                all_true.extend(y_batch.numpy())

        if save_confusion_matrix:
            cm = np.zeros((n_classes, n_classes), dtype=int)
            for true, pred in zip(all_true, all_pred):
                cm[true][pred] += 1
            logger.log_confusion_matrix(cm)

        val_metrics = {
            "macro_f1": round(f1_score(all_true, all_pred, average="macro", zero_division=0), 6),
            "weighted_f1": round(f1_score(all_true, all_pred, average="weighted", zero_division=0), 6),
            "balanced_accuracy": round(balanced_accuracy_score(all_true, all_pred), 6),
            "per_class": classification_report(all_true, all_pred, output_dict=True, zero_division=0),
        }
        metrics["val_metrics"] = val_metrics
        logger.log_val_metrics(val_metrics)

    if logger:
        summary = logger.finalize(
            best_epoch=best_epoch,
            best_val_acc=best_val_acc,
            total_time_sec=total_time,
            peak_memory_mb=metrics["peak_memory_mb"],
        )
        metrics["logger_summary"] = summary

    if return_metrics:
        return {"model": model, "metrics": metrics}
    return model
