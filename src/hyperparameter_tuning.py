# src/hyperparameter_tuning.py

"""
Orchestrates the recommended 3-phase hyperparameter tuning strategy:
Phase 1: Test learning rates [1e-4, 1e-3, 1e-2]
Phase 2: Test batch sizes [16, 32, 64]
Phase 3: Test RNN hidden units [64, 128, 256]
"""

from train import train

class HyperparameterTuner:
    def __init__(self, X, y, n_classes, results_dir="results"):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            X: Training data
            y: Training labels
            n_classes: Number of classes
            results_dir: Directory to save results
        """
        self.X = X
        self.y = y
        self.n_classes = n_classes
        
        # Default hyperparameters
        self.default_params = {
            "epochs": 20,
            "batch_size": 32,
            "lr": 1e-3,
            "rnn_hidden": 128  # Will be passed to model, not directly to train()
        }
    
    def run_phase_1_learning_rates(self):
        """
        Phase 1: Test different learning rates.
        Learning rates: [1e-4, 1e-3, 1e-2]
        """
        print("\n" + "="*60)
        print("PHASE 1: Testing Learning Rates")
        print("="*60)
        
        learning_rates = [1e-4, 1e-3, 1e-2]
        phase_results = []
        
        for i, lr in enumerate(learning_rates, 1):
            run_id = f"lr_{lr}"
            print(f"\nRun {i}/3: Testing lr={lr}")
            print("-" * 40)
            
            hyperparams = {
                "lr": lr,
                "batch_size": self.default_params["batch_size"],
                "epochs": self.default_params["epochs"]
            }
            
            result = train(
                self.X, self.y,
                n_classes=self.n_classes,
                epochs=hyperparams["epochs"],
                batch_size=hyperparams["batch_size"],
                lr=hyperparams["lr"],
                rnn_hidden=self.default_params["rnn_hidden"],
                return_metrics=True
            )
            
            metrics = result["metrics"]
            
            # Track for phase summary
            best_acc = max([e["accuracy"] for e in metrics["epoch_history"]])
            final_loss = metrics["epoch_history"][-1]["loss"]
            
            phase_results.append({
                "run_id": run_id,
                "hyperparameters": hyperparams,
                "best_accuracy": best_acc,
                "final_loss": final_loss,
                "training_time_seconds": metrics["total_training_time"],
                "peak_memory_mb": metrics["peak_memory_mb"]
            })
        
        # Find best learning rate
        best_run = max(phase_results, key=lambda x: x["best_accuracy"])
        best_lr = best_run["hyperparameters"]["lr"]
        
        print(f"\n✓ Phase 1 Complete!")
        print(f"  Best learning rate: {best_lr}")
        print(f"  Best accuracy: {best_run['best_accuracy']:.4f}")
        
        return best_lr
    
    def run_phase_2_batch_sizes(self, best_lr):
        """
        Phase 2: Test different batch sizes.
        Batch sizes: [16, 32, 64]
        Uses the best learning rate from Phase 1.
        """
        print("\n" + "="*60)
        print("PHASE 2: Testing Batch Sizes")
        print(f"  (Using best LR from Phase 1: {best_lr})")
        print("="*60)
        
        batch_sizes = [16, 32, 64]
        phase_results = []
        
        for i, batch_size in enumerate(batch_sizes, 1):
            run_id = f"bs_{batch_size}"
            print(f"\nRun {i}/3: Testing batch_size={batch_size}")
            print("-" * 40)
            
            hyperparams = {
                "lr": best_lr,
                "batch_size": batch_size,
                "epochs": self.default_params["epochs"]
            }
            
            result = train(
                self.X, self.y,
                n_classes=self.n_classes,
                epochs=hyperparams["epochs"],
                batch_size=hyperparams["batch_size"],
                lr=hyperparams["lr"],
                rnn_hidden=self.default_params["rnn_hidden"],
                return_metrics=True
            )
            
            metrics = result["metrics"]
            
            
            # Track for phase summary
            best_acc = max([e["accuracy"] for e in metrics["epoch_history"]])
            final_loss = metrics["epoch_history"][-1]["loss"]
            
            phase_results.append({
                "run_id": run_id,
                "hyperparameters": hyperparams,
                "best_accuracy": best_acc,
                "final_loss": final_loss,
                "training_time_seconds": metrics["total_training_time"],
                "peak_memory_mb": metrics["peak_memory_mb"]
            })
        
        # Find best batch size
        best_run = max(phase_results, key=lambda x: x["best_accuracy"])
        best_bs = best_run["hyperparameters"]["batch_size"]
        
        print(f"\n✓ Phase 2 Complete!")
        print(f"  Best batch size: {best_bs}")
        print(f"  Best accuracy: {best_run['best_accuracy']:.4f}")
        
        return best_bs
    
    def run_phase_3_rnn_hidden_units(self, best_lr, best_bs):
        """
        Phase 3: Test different RNN hidden units.
        Hidden units: [64, 128, 256]
        Uses best parameters from Phases 1 and 2.
        """
        print("\n" + "="*60)
        print("PHASE 3: Testing RNN Hidden Units")
        print(f"  (Using best LR={best_lr}, batch_size={best_bs})")
        print("="*60)
        
        hidden_units = [64, 128, 256]
        phase_results = []
        
        for i, hidden in enumerate(hidden_units, 1):
            run_id = f"rnn_{hidden}"
            print(f"\nRun {i}/3: Testing rnn_hidden={hidden}")
            print("-" * 40)
            
            hyperparams = {
                "lr": best_lr,
                "batch_size": best_bs,
                "epochs": self.default_params["epochs"],
                "rnn_hidden": hidden
            }
            
            # Now we actually vary the rnn_hidden parameter
            result = train(
                self.X, self.y,
                n_classes=self.n_classes,
                epochs=hyperparams["epochs"],
                batch_size=hyperparams["batch_size"],
                lr=hyperparams["lr"],
                rnn_hidden=hyperparams["rnn_hidden"],
                return_metrics=True
            )
            
            metrics = result["metrics"]
            
            # Track for phase summary
            best_acc = max([e["accuracy"] for e in metrics["epoch_history"]])
            final_loss = metrics["epoch_history"][-1]["loss"]
            
            phase_results.append({
                "run_id": run_id,
                "hyperparameters": hyperparams,
                "best_accuracy": best_acc,
                "final_loss": final_loss,
                "training_time_seconds": metrics["total_training_time"],
                "peak_memory_mb": metrics["peak_memory_mb"]
            })
        
        # Find best hidden units
        best_run = max(phase_results, key=lambda x: x["best_accuracy"])
        best_hidden = best_run["hyperparameters"]["rnn_hidden"]
        
        print(f"\n✓ Phase 3 Complete!")
        print(f"  Best RNN hidden units: {best_hidden}")
        print(f"  Best accuracy: {best_run['best_accuracy']:.4f}")
        
        return best_hidden
    
    def run_all_phases(self):
        """Run all three phases sequentially."""
        print("\n" + "="*60)
        print("STARTING HYPERPARAMETER TUNING - 3 PHASE STRATEGY")
        print("="*60)
        
        # Phase 1: Learning rates
        best_lr = self.run_phase_1_learning_rates()
        
        # Phase 2: Batch sizes
        best_bs = self.run_phase_2_batch_sizes(best_lr)
        
        # Phase 3: RNN hidden units
        best_hidden = self.run_phase_3_rnn_hidden_units(best_lr, best_bs)
        
        # Generate final report
        print("\n" + "="*60)
        print("GENERATING FINAL REPORT")
        print("="*60)
        
        # Print final recommendations
        print("\n" + "="*60)
        print("FINAL RECOMMENDATIONS")
        print("="*60)
        print(f"Learning Rate:    {best_lr}")
        print(f"Batch Size:       {best_bs}")
        print(f"RNN Hidden Units: {best_hidden}")
        print("\nCheck results/ directory for detailed logs and comparison report.")


def main():
    """
    Example usage. Replace X, y, n_classes with your actual data.
    """
    print("Hyperparameter Tuning Setup Complete!")
    print("\nTo run tuning, use in your main script:")
    print("""
    from hyperparameter_tuning import HyperparameterTuner
    
    # Assuming X, y are loaded
    tuner = HyperparameterTuner(X, y, n_classes=10)
    tuner.run_all_phases()
    """)


if __name__ == "__main__":
    main()
