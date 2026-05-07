# src/hyperparameter_tuning.py

"""
3-phase hyperparameter tuning strategy:
  Phase 1: Learning rates  [1e-4, 1e-3, 1e-2]
  Phase 2: Batch sizes     [16, 32, 64]
  Phase 3: RNN hidden units [64, 128, 256]

Each run is fully logged: per-epoch CSV, config JSON, run summary JSON.
A cross-run summary CSV is written at the end for R/LaTeX analysis.
"""

from train import train
from utils.logger import TuningLogger


class HyperparameterTuner:
    def __init__(self, X, y, n_classes, results_dir="results", epochs=15,
                 weight_decay=1e-4, use_class_weights=False):
        self.X = X
        self.y = y
        self.n_classes = n_classes
        self.weight_decay = weight_decay
        self.use_class_weights = use_class_weights
        self.default_params = {
            "epochs": epochs,
            "batch_size": 32,
            "lr": 1e-3,
            "rnn_hidden": 128,
        }
        self.tuning_logger = TuningLogger(results_dir=results_dir)

    def _run(self, phase, label, lr, batch_size, rnn_hidden):
        config = {
            "lr": lr,
            "batch_size": batch_size,
            "rnn_hidden": rnn_hidden,
            "epochs": self.default_params["epochs"],
            "weight_decay": self.weight_decay,
            "n_classes": self.n_classes,
        }
        run_logger = self.tuning_logger.new_run_logger(phase, label, config)
        result = train(
            self.X, self.y,
            n_classes=self.n_classes,
            epochs=config["epochs"],
            batch_size=batch_size,
            lr=lr,
            rnn_hidden=rnn_hidden,
            weight_decay=self.weight_decay,
            return_metrics=True,
            logger=run_logger,
            use_class_weights=self.use_class_weights,
        )
        # finalize is called inside train(); summary is stored in metrics
        summary = result["metrics"]["logger_summary"]
        val_metrics = result["metrics"].get("val_metrics")
        self.tuning_logger.record_run(phase, label, config, summary, val_metrics=val_metrics)
        return summary

    def run_phase_1_learning_rates(self):
        print("\n" + "=" * 60)
        print("PHASE 1: Learning Rates [1e-4, 1e-3, 1e-2]")
        print("=" * 60)

        candidates = [1e-4, 1e-3, 1e-2]
        results = []
        for lr in candidates:
            label = f"lr_{lr:.0e}"
            summary = self._run(
                phase=1, label=label,
                lr=lr,
                batch_size=self.default_params["batch_size"],
                rnn_hidden=self.default_params["rnn_hidden"],
            )
            results.append((lr, summary))

        best_lr, best_summary = max(results, key=lambda x: x[1]["best_val_acc"])
        self.tuning_logger.log_phase_winner(1, f"lr={best_lr}", best_summary["best_val_acc"])
        return best_lr

    def run_phase_2_batch_sizes(self, best_lr):
        print("\n" + "=" * 60)
        print(f"PHASE 2: Batch Sizes [16, 32, 64]  (lr={best_lr})")
        print("=" * 60)

        candidates = [16, 32, 64]
        results = []
        for bs in candidates:
            label = f"bs_{bs}"
            summary = self._run(
                phase=2, label=label,
                lr=best_lr,
                batch_size=bs,
                rnn_hidden=self.default_params["rnn_hidden"],
            )
            results.append((bs, summary))

        best_bs, best_summary = max(results, key=lambda x: x[1]["best_val_acc"])
        self.tuning_logger.log_phase_winner(2, f"batch_size={best_bs}", best_summary["best_val_acc"])
        return best_bs

    def run_phase_3_rnn_hidden_units(self, best_lr, best_bs):
        print("\n" + "=" * 60)
        print(f"PHASE 3: RNN Hidden Units [64, 128, 256]  (lr={best_lr}, bs={best_bs})")
        print("=" * 60)

        candidates = [64, 128, 256]
        results = []
        for hidden in candidates:
            label = f"rnn_{hidden}"
            summary = self._run(
                phase=3, label=label,
                lr=best_lr,
                batch_size=best_bs,
                rnn_hidden=hidden,
            )
            results.append((hidden, summary))

        best_hidden, best_summary = max(results, key=lambda x: x[1]["best_val_acc"])
        self.tuning_logger.log_phase_winner(3, f"rnn_hidden={best_hidden}", best_summary["best_val_acc"])
        return best_hidden

    def run_all_phases(self):
        best_lr = self.run_phase_1_learning_rates()
        best_bs = self.run_phase_2_batch_sizes(best_lr)
        best_hidden = self.run_phase_3_rnn_hidden_units(best_lr, best_bs)

        recommendations = {
            "lr": best_lr,
            "batch_size": best_bs,
            "rnn_hidden": best_hidden,
            "weight_decay": self.weight_decay,
        }
        tuning_dir = self.tuning_logger.finalize(recommendations)

        print("\n" + "=" * 60)
        print("FINAL RECOMMENDATIONS")
        print("=" * 60)
        print(f"  lr:          {best_lr}")
        print(f"  batch_size:  {best_bs}")
        print(f"  rnn_hidden:  {best_hidden}")
        print(f"  weight_decay:{self.weight_decay}")
        print(f"\nAll artifacts in: {tuning_dir}")

        return recommendations
