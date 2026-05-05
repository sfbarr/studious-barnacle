# src/models/crnn.py

import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, n_classes, n_mels=128, rnn_hidden=128):
        super().__init__()

        # ---- CNN feature extractor ----
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        # ---- RNN for temporal modeling ----
        self.rnn_hidden = rnn_hidden
        self.rnn = nn.GRU(
            input_size=128 * (n_mels // 8),  # after pooling
            hidden_size=self.rnn_hidden,
            batch_first=True,
            bidirectional=True
        )

        # ---- classifier ----
        self.fc = nn.Sequential(
            nn.Linear(self.rnn_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # x: (B, 1, n_mels, time)

        x = self.cnn(x)  # (B, C, F, T)

        b, c, f, t = x.shape

        # reshape for RNN: treat time as sequence
        x = x.permute(0, 3, 1, 2)  # (B, T, C, F)
        x = x.reshape(b, t, c * f)  # (B, T, features)

        rnn_out, _ = self.rnn(x)  # (B, T, 2H)

        # temporal pooling (simple + strong baseline)
        x = rnn_out.mean(dim=1)

        return self.fc(x)