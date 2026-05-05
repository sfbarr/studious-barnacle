# src/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.crnn import CRNN
from data.dataset import MelSpectrogramDataset
from utils.logger import TrainingLogger

def train(X, y, n_classes=16, epochs=20, batch_size=32, lr=1e-3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    # ---- metadata for logging ----
    metadata = {
        "model": "CRNN",
        "n_classes": n_classes,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "device": str(device),
        "dataset_size": len(X)
    }

    logger = TrainingLogger(metadata=metadata)

    dataset = MelSpectrogramDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CRNN(n_classes=n_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # ---- accuracy calc ----
            predicted = torch.argmax(preds, dim=1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
       
        avg_loss = total_loss / len(loader)
        accuracy = correct / total

        logger.log_epoch(
            epoch=epoch + 1,
            loss=round(avg_loss, 4),
            accuracy=round(accuracy, 4)
        )

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Accuracy: {accuracy:.4f}")

    logger.save()

    return model