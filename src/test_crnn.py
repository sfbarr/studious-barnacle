# src/test_crnn.py

import numpy as np
import torch
from models.crnn import CRNN
from data.dataset import MelSpectrogramDataset

# Parameters
N = 8  # batch size
n_mels = 128
n_time = 256
n_classes = 10

# Generate random log-mel spectrogram data and labels
X = np.random.randn(N, 1, n_mels, n_time).astype(np.float32)
y = np.random.randint(0, n_classes, size=(N,))

dataset = MelSpectrogramDataset(X, y)
X_batch, y_batch = dataset[0]
print(f"Sample X shape: {X_batch.shape}, dtype: {X_batch.dtype}")
print(f"Sample y: {y_batch}")

# DataLoader test
loader = torch.utils.data.DataLoader(dataset, batch_size=4)
for Xb, yb in loader:
    print(f"Batch X: {Xb.shape}, Batch y: {yb.shape}")
    break

# Model test
model = CRNN(n_classes=n_classes, n_mels=n_mels)
output = model(torch.tensor(X))
print(f"Model output shape: {output.shape}")

# Training step test
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
preds = model(torch.tensor(X))
loss = criterion(preds, torch.tensor(y))
loss.backward()
optimizer.step()
print(f"Single training step loss: {loss.item():.4f}")
