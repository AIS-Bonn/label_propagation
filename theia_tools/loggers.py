from torch.utils.tensorboard import SummaryWriter
import time
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

class MetricLogger:
    def __init__(self, log_dir, run_name=None, save_dir=None):
        """
        Args:
            log_dir (str): base directory like "runs/"
            run_name (str): specific experiment name like "lambda_0.01_lr_1e-4"
            save_dir (str): optional save directory for best model
        """
        full_log_dir = os.path.join(log_dir, run_name) if run_name else log_dir
        self.writer = SummaryWriter(log_dir=full_log_dir)
        self.start_time = time.time()
        self.best_val_metric = -float('inf')
        self.save_dir = save_dir

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    def log_scalar(self, name, value, step):
        self.writer.add_scalar(name, value, step)

    def log_histogram(self, name, values, step):
        self.writer.add_histogram(name, values, step)

    def log_image(self, name, img_tensor, step):
        self.writer.add_image(name, img_tensor, step)

    def log_learning_rate(self, optimizer, step):
        lr = optimizer.param_groups[0]['lr']
        self.writer.add_scalar('LearningRate', lr, step)

    def epoch_timing(self, epoch):
        elapsed = time.time() - self.start_time
        self.log_scalar('Timing/EpochDuration', elapsed, epoch)
        self.start_time = time.time()

    def update_best(self, model, val_metric, epoch, model_name="best_model.pth"):
        if val_metric > self.best_val_metric:
            self.best_val_metric = val_metric
            if self.save_dir:
                model_path = os.path.join(self.save_dir, model_name)
                torch.save(model.state_dict(), model_path)
                print(f"New best model saved at epoch {epoch} with val_metric={val_metric:.4f}")

    def close(self):
        self.writer.close()
    
    def log_hparams(self, hparams: dict, metrics: dict = None):
        """
        Logs hyperparameters and optional final metrics.
    
        Args:
            hparams (dict): dictionary of hyperparameters
            metrics (dict): optional dictionary of final evaluation metrics (e.g., final loss, mAP)
        """
        self.writer.add_hparams(hparams, metrics or {})
    
    def log_confusion_matrix(self, name, eval_results, epoch):
        cm = eval_results.confusion_matrix()
        labels = eval_results.classes

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        # Show values inside the matrix
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black")

        fig.tight_layout()

        self.writer.add_figure(name, fig, global_step=epoch)

    def log_pr_matrix(self, name, eval_results, epoch):
        report = eval_results.report()

        for class_name, metrics in report.items():
            precision = metrics["precision"]
            recall = metrics["recall"]
            f1 = metrics["f1-score"]

            self.writer.add_scalar(f"{name}/{class_name}/precision", precision, epoch)
            self.writer.add_scalar(f"{name}/{class_name}/recall", recall, epoch)
            self.writer.add_scalar(f"{name}/{class_name}/f1", f1, epoch)
        
        # Log average metrics
        precisions = [metrics["precision"] for metrics in report.values()]
        recalls = [metrics["recall"] for metrics in report.values()]
        f1s = [metrics["f1-score"] for metrics in report.values()]

        self.writer.add_scalar("{name}/avg_precision", sum(precisions) / len(precisions), epoch)
        self.writer.add_scalar("{name}/avg_recall", sum(recalls) / len(recalls), epoch)
        self.writer.add_scalar("{name}/avg_f1", sum(f1s) / len(f1s), epoch)
