import json
import os
import time
from datetime import datetime


class TrainingLogger:
    def __init__(self, log_dir="logs", metadata=None):
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_path = os.path.join(log_dir, f"training_run_{timestamp}.json")

        self.start_time = time.time()

        self.data = {
            "metadata": metadata or {},
            "epochs": [],
            "training_time_seconds": None
        }

    def log_epoch(self, epoch, loss, accuracy):
        self.data["epochs"].append({
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy
        })

    def save(self):
        self.data["training_time_seconds"] = round(time.time() - self.start_time, 2)

        with open(self.log_path, "w") as f:
            json.dump(self.data, f, indent=4)

        print(f"Training log saved to {self.log_path}")