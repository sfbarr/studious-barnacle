import csv
import json
import os
import time
from datetime import datetime


class RunLogger:
    """File-based logger for a single training run."""

    def __init__(self, run_dir, config):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir

        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        self._log_path = os.path.join(run_dir, "training.log")
        self._log_file = open(self._log_path, "w", buffering=1)

        self._csv_path = os.path.join(run_dir, "epoch_metrics.csv")
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=["epoch", "train_loss", "train_acc", "val_acc",
                        "epoch_time_sec", "batch_time_mean_sec", "batch_time_std_sec",
                        "memory_mb", "gpu_mem_mb", "gpu_util_pct", "is_best"],
        )
        self._csv_writer.writeheader()
        self._csv_file.flush()

        self._epoch_rows = []
        self._write(f"Run started: {datetime.now().isoformat()}")
        self._write(f"Config: {json.dumps(config)}")
        self._write("-" * 80)

    def _write(self, msg):
        print(msg)
        self._log_file.write(msg + "\n")

    def log_epoch(self, epoch, train_loss, train_acc, val_acc,
                  epoch_time_sec, memory_mb, is_best=False,
                  batch_time_mean_sec=None, batch_time_std_sec=None,
                  gpu_util_pct=None, gpu_mem_mb=None):
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_acc": round(val_acc, 6),
            "epoch_time_sec": round(epoch_time_sec, 2),
            "batch_time_mean_sec": round(batch_time_mean_sec, 4) if batch_time_mean_sec is not None else None,
            "batch_time_std_sec": round(batch_time_std_sec, 4) if batch_time_std_sec is not None else None,
            "memory_mb": round(memory_mb, 1),
            "gpu_mem_mb": round(gpu_mem_mb, 1) if gpu_mem_mb is not None else None,
            "gpu_util_pct": gpu_util_pct,
            "is_best": is_best,
        }
        self._epoch_rows.append(row)
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        gpu_str = ""
        if gpu_mem_mb is not None:
            gpu_str = f" | GPU: {gpu_mem_mb:.0f} MB"
            if gpu_util_pct is not None:
                gpu_str += f" {gpu_util_pct}%"
        marker = " *" if is_best else ""
        self._write(
            f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
            f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
            f"Time: {epoch_time_sec:.1f}s | Batch: {batch_time_mean_sec:.3f}s | "
            f"RAM: {memory_mb:.0f} MB{gpu_str}{marker}"
        )

    def log_val_metrics(self, val_metrics):
        path = os.path.join(self.run_dir, "val_metrics.json")
        with open(path, "w") as f:
            json.dump(val_metrics, f, indent=2)
        self._write(
            f"Val metrics | macro_f1={val_metrics['macro_f1']:.4f} | "
            f"weighted_f1={val_metrics['weighted_f1']:.4f} | "
            f"balanced_acc={val_metrics['balanced_accuracy']:.4f}"
        )

    def log_confusion_matrix(self, cm, class_names=None):
        cm_path = os.path.join(self.run_dir, "confusion_matrix.csv")
        with open(cm_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["true\\pred"] + (
                list(class_names) if class_names else [str(i) for i in range(len(cm))]
            )
            writer.writerow(header)
            for i, row in enumerate(cm):
                label = class_names[i] if class_names else str(i)
                writer.writerow([label] + list(row))
        self._write(f"Confusion matrix saved to {cm_path}")

    def finalize(self, best_epoch, best_val_acc, total_time_sec, peak_memory_mb):
        summary = {
            "best_epoch": best_epoch,
            "best_val_acc": round(best_val_acc, 6),
            "total_time_sec": round(total_time_sec, 2),
            "peak_memory_mb": round(peak_memory_mb, 1),
            "epochs_trained": len(self._epoch_rows),
        }
        with open(os.path.join(self.run_dir, "run_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        self._write("-" * 80)
        self._write(
            f"Run complete | best_val_acc={best_val_acc:.4f} at epoch {best_epoch} | "
            f"total={total_time_sec:.1f}s | peak_mem={peak_memory_mb:.0f} MB"
        )
        self._write(f"Artifacts: {self.run_dir}")
        self._log_file.close()
        self._csv_file.close()
        return summary


class TuningLogger:
    """Orchestrates per-run RunLoggers across all hyperparameter tuning phases."""

    def __init__(self, results_dir="results"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tuning_dir = os.path.join(results_dir, f"tuning_{timestamp}")
        os.makedirs(self.tuning_dir, exist_ok=True)

        self._log_file = open(os.path.join(self.tuning_dir, "tuning.log"), "w", buffering=1)
        self._all_runs = []

        self._write(f"Hyperparameter tuning started: {datetime.now().isoformat()}")
        self._write("=" * 80)

    def _write(self, msg):
        print(msg)
        self._log_file.write(msg + "\n")

    def new_run_logger(self, phase, label, config):
        run_dir = os.path.join(self.tuning_dir, f"phase{phase}", label)
        self._write(f"\n[Phase {phase}] Starting: {label}")
        return RunLogger(run_dir, config)

    def record_run(self, phase, label, config, summary, val_metrics=None):
        row = {
            "phase": phase,
            "label": label,
            "lr": config.get("lr"),
            "batch_size": config.get("batch_size"),
            "rnn_hidden": config.get("rnn_hidden"),
            "epochs_per_run": config.get("epochs"),
            "weight_decay": config.get("weight_decay"),
            "best_val_acc": summary["best_val_acc"],
            "best_epoch": summary["best_epoch"],
            "total_time_sec": summary["total_time_sec"],
            "peak_memory_mb": summary["peak_memory_mb"],
            "macro_f1": val_metrics.get("macro_f1") if val_metrics else None,
            "weighted_f1": val_metrics.get("weighted_f1") if val_metrics else None,
            "balanced_accuracy": val_metrics.get("balanced_accuracy") if val_metrics else None,
        }
        self._all_runs.append(row)
        self._write(
            f"[Phase {phase}] {label}: best_val_acc={summary['best_val_acc']:.4f} "
            f"at epoch {summary['best_epoch']} ({summary['total_time_sec']:.1f}s)"
        )

    def log_phase_winner(self, phase, label, val_acc):
        self._write(f"\n>>> Phase {phase} winner: {label}  val_acc={val_acc:.4f}")

    def finalize(self, recommendations):
        if self._all_runs:
            csv_path = os.path.join(self.tuning_dir, "all_runs_summary.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self._all_runs[0].keys()))
                writer.writeheader()
                writer.writerows(self._all_runs)
            self._write(f"\nAll-runs summary: {csv_path}")

        with open(os.path.join(self.tuning_dir, "tuning_summary.json"), "w") as f:
            json.dump({"recommendations": recommendations, "all_runs": self._all_runs}, f, indent=2)

        self._write("=" * 80)
        self._write("Tuning complete. Recommendations:")
        for k, v in recommendations.items():
            self._write(f"  {k}: {v}")
        self._write(f"Tuning artifacts: {self.tuning_dir}")
        self._log_file.close()
        return self.tuning_dir
