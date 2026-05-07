from __future__ import annotations
import os, csv
import torch
import pytorch_lightning as pl
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5


class LoggingModel(pl.LightningModule):
    def on_test_epoch_end(self):
        if len(self.test_preds) == 0:
            return
        preds = torch.cat(self.test_preds, dim=0).numpy()
        targets = torch.cat(self.test_targets, dim=0).numpy()
        paths = self.test_paths

        num_classes = self.hparams.num_classes
        cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        for t, p in zip(targets, preds):
            cm[t, p] += 1
        fig = self._plot_confusion_matrix(cm.numpy())

        log_dir = self._get_logger_dir()
        os.makedirs(log_dir, exist_ok=True)

        fig_path = os.path.join(log_dir, "test_confusion_matrix.png")
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)

        out_csv1 = os.path.join(log_dir, "test_predictions.csv")
        with open(out_csv1, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["target", "pred"])
            for t, p in zip(targets.tolist(), preds.tolist()):
                writer.writerow([t, p])
        out_2 = os.path.join(log_dir, "summary.txt")
        num_classes = self.hparams.num_classes

        cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
        for t, p in zip(targets.tolist(), preds.tolist()):
            cm[t, p] += 1

        total = int(cm.sum().item())
        tp_total = int(torch.trace(cm).item())
        fp_total = int((cm.sum(dim=0) - torch.diag(cm)).sum().item())
        fn_total = int((cm.sum(dim=1) - torch.diag(cm)).sum().item())
        tn_total = int(num_classes * total - (tp_total + fp_total + fn_total))

        total_correct = tp_total
        total_incorrect = total - total_correct

        per_class_lines = []
        for k in range(num_classes):
            support_k = int(cm[k, :].sum().item())          
            correct_k = int(cm[k, k].item())
            incorrect_k = support_k - correct_k
            per_class_lines.append(
                f"class {k}: total={support_k}, correct={correct_k}, incorrect={incorrect_k}"
            )

        with open(out_2, "w") as f:
            f.write("=== Test Summary ===\n")
            f.write(f"num_test_samples: {total}\n")
            f.write(f"TP_total: {tp_total}\n")
            f.write(f"TN_total (one-vs-all aggregated): {tn_total}\n")
            f.write(f"FP_total: {fp_total}\n")
            f.write(f"FN_total: {fn_total}\n")
            f.write(f"total_correct: {total_correct}\n")
            f.write(f"total_incorrect: {total_incorrect}\n")
            f.write("\n=== Per-class ===\n")
            for line in per_class_lines:
                f.write(line + "\n")

        self.test_preds.clear()
        self.test_targets.clear()
        self.test_paths.clear()

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.optim_cfg.lr,
            weight_decay=self.optim_cfg.weight_decay
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=self.optim_cfg.scheduler_patience, factor=self.optim_cfg.scheduler_factor
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1
            }
        }

    @staticmethod
    def _plot_confusion_matrix(cm):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
        ax.imshow(cm, interpolation="nearest")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        num_classes = cm.shape[0]
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        fig.tight_layout()
        return fig

    def _get_logger_dir(self) -> str:
        if self.logger is None:
            raise RuntimeError()
        required_attrs = ("save_dir", "name", "version")
        for attr in required_attrs:
            if not hasattr(self.logger, attr):
                raise RuntimeError()
            val = getattr(self.logger, attr)
            if val is None or (isinstance(val, str) and val.strip() == ""):
                raise RuntimeError()
        return os.path.join(self.logger.save_dir, str(self.logger.name), f'version_{str(self.logger.version)}')
