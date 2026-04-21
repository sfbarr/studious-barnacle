# src/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.crnn import CRNN
from data.dataset import MelSpectrogramDataset

def train(X, y, n_classes, epochs=20, batch_size=32, lr=1e-3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MelSpectrogramDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CRNN(n_classes=n_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

    return model