# run.py or main.py

from train import train

# once data is ready, we can run this to train the model

import numpy as np

X = np.load("data/X.npy")
y = np.load("data/y.npy")

model = train(X, y, n_classes=16)