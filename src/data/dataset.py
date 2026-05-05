# src/data/dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset


class MelSpectrogramDataset(Dataset):
    def __init__(self, X, y):
        """
        X: numpy array or memmap (N, 1, n_mels, time) — not loaded into RAM here
        y: numpy array (N,)

        On Windows, DataLoader workers are spawned (not forked), so the Dataset
        is pickled into each worker. Memmap file handles can't survive pickling,
        so we store the path and reopen lazily inside each worker process.
        """
        if isinstance(X, np.memmap):
            self._X_path = X.filename
            self._X = None
        else:
            self._X_path = None
            self._X = X
        self.y = y

    def _get_X(self):
        if self._X is None:
            self._X = np.load(self._X_path, mmap_mode="r")
        return self._X

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # np.array() materializes one sample from the memmap into a contiguous buffer
        x = torch.from_numpy(np.array(self._get_X()[idx], dtype=np.float32))
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y
