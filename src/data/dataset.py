# src/data/dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset


class MelSpectrogramDataset(Dataset):
    def __init__(self, X, y):
        """
        X: numpy array or memmap (N, 1, n_mels, time) — not loaded into RAM here
        y: numpy array (N,)
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # np.array() materializes one sample from the memmap into a contiguous buffer
        x = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y
