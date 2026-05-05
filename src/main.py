# run.py or main.py

from train import train

# once data is ready, we can run this to train the model

import numpy as np

X = np.load("X.npy")
y = np.load("y.npy")

model = train(X, y, n_classes=10)

# # Test if logger worked
# import numpy as np
# from train import train

# # fake data for testing
# X = np.random.randn(100, 1, 128, 256).astype(np.float32)
# y = np.random.randint(0, 10, size=(100,))

# train(X, y, n_classes=10, epochs=2)