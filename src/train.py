# src/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import psutil
import os

from models.crnn import CRNN
from data.dataset import MelSpectrogramDataset

def train(X, y, n_classes, epochs=20, batch_size=32, lr=1e-3, rnn_hidden=128, return_metrics=False):
    """
    Train CRNN model with optional metrics tracking.
    
    Args:
        X: Input features
        y: Labels
        n_classes: Number of classes
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        rnn_hidden: Number of RNN hidden units (default: 128)
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
        "epoch_history": []
    }
    
    dataset = MelSpectrogramDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = CRNN(n_classes=n_classes, rnn_hidden=rnn_hidden).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Get process for memory tracking
    process = psutil.Process(os.getpid())
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Memory tracking per epoch
        process.memory_info()  # warm up
        
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(preds.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        
        # Get memory usage
        mem_info = process.memory_info()
        memory_mb = mem_info.rss / 1024 / 1024
        metrics["peak_memory_mb"] = max(metrics["peak_memory_mb"], memory_mb)
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        
        epoch_data = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "accuracy": accuracy,
            "memory_mb": memory_mb
        }
        metrics["epoch_history"].append(epoch_data)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | Mem: {memory_mb:.1f}MB")
    
    total_time = time.time() - start_time
    metrics["total_training_time"] = total_time
    metrics["epochs_trained"] = epochs
    
    if return_metrics:
        return {"model": model, "metrics": metrics}
    else:
        return model