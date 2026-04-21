# run.py or main.py

from train import train

# once data is ready, we can run this to train the model

import numpy as np

X = np.load("X.npy")
y = np.load("y.npy")

model = train(X, y, n_classes=10)