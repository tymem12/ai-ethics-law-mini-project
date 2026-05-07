import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from dataset import GaussianDataModule
from models.pointnet.pointnet import PointNetGSSystem, OptimConfig
from models.pointnet2.pointnet2 import PointNet2GSSystem
from models.pointneXt.pointneXt import PointNeXtSystem
from models.pointMLP.pointMLP import PointMLPSystem


def get_model(model_name: str):
    if model_name == "pointnet":
        return PointNetGSSystem
    elif model_name == "pointnet2":
        return PointNet2GSSystem
    elif model_name == 'pointneXt':
        return PointNeXtSystem
    elif model_name == 'pointMLP':
        return PointMLPSystem
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
def main_run_script(
    *,
    # Data
    model_name: str, 
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    val_split: float = 0.1,
    num_points: int = 75000,
    sampling: str = "random",              
    seed: int = 42,

    # Model
    in_channels_total: int = 11,
    dropout_p: float = 0.3,
    feature_transform: bool = True,
    ft_reg_weight: float = 0.001,

    # Optim
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 90,
    precision: str = "32",        # choices: ["32", "16-mixed", "bf16-mixed"]
    accumulate_grad_batches: int = 1,

    # Scheduling / callbacks
    lr_patience: int = 10,
    lr_factor: float = 0.5,
    early_stop_patience: int = 80,
    save_top_k: int = 1,

    # Logging
    log_dir: str = "/kaggle/working/lightning_logs",
    run_name: str = "pointnet_gs_60000_v1",
):
    if precision not in {"32", "16-mixed", "bf16-mixed"}:
        raise ValueError(f'Invalid precision="{precision}". Choose from "32", "16-mixed", "bf16-mixed".')

    pl.seed_everything(seed, workers=True)
    dm = GaussianDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        num_points=num_points,
        sampling=sampling,
        seed=seed,
    )
    
    dm.setup()
    num_classes = dm.num_classes
    optim_cfg = OptimConfig(
        lr=lr,
        weight_decay=weight_decay,
        scheduler_patience=lr_patience,
        scheduler_factor=lr_factor,
    )
    model = get_model(model_name)(
        num_classes=num_classes,
        in_channels_total=in_channels_total,
        feature_transform=feature_transform,
        dropout_p=dropout_p,
        optim_cfg=optim_cfg,
        ft_reg_weight=ft_reg_weight,
    )

    logger = TensorBoardLogger(save_dir=log_dir, name=run_name)

    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename="pointnet-{epoch:02d}",
        save_top_k=save_top_k,
        monitor="val/loss",
        mode="min",
        save_last=True,
    )
    es_cb = EarlyStopping(
        monitor="val/loss",
        patience=early_stop_patience,
        mode="min",
    )
    lr_mon = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[ckpt_cb, es_cb, lr_mon],
        max_epochs=max_epochs,
        deterministic=False,
        accelerator="auto",
        devices='auto',
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=5,

    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path=ckpt_cb.best_model_path or "best")
    best_ckpt = ckpt_cb.best_model_path
    tb_dir = logger.log_dir
    test_csv = os.path.join(logger.log_dir, "test_predictions.csv")


    return {
        "best_checkpoint_path": best_ckpt,
        "logger_dir": tb_dir,
        "test_predictions_csv": test_csv,
        "trainer": trainer,
        "model": model,
    }






def parse_args():
    p = argparse.ArgumentParser(description="Train PointNet (11-channel) with Lightning")

    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True,
                   help="Dataset root with train/ and test/ subfolders")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--num_points", type=int, default=8192)

    p.add_argument("--sampling", type=str, default="random", choices=["fps", "random"])
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--in_channels_total", type=int, default=11)
    p.add_argument("--dropout_p", type=float, default=0.3)
    p.add_argument("--feature_transform", action="store_true", default=True)
    p.add_argument("--ft_reg_weight", type=float, default=0.001)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=90)
    p.add_argument("--precision", type=str, default="16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--accumulate_grad_batches", type=int, default=1)

    p.add_argument("--lr_patience", type=int, default=10)
    p.add_argument("--lr_factor", type=float, default=0.5)
    p.add_argument("--early_stop_patience", type=int, default=90)
    p.add_argument("--save_top_k", type=int, default=1)

    p.add_argument("--log_dir", type=str, default="lightning_logs")
    p.add_argument("--run_name", type=str, default="pointnet_gs")

    return p.parse_args()


def main():
    args = parse_args()
    results = main_run_script(**vars(args))

    best_ckpt = results.get("best_checkpoint_path")
    tb_dir = results.get("logger_dir")
    test_csv = results.get("test_predictions_csv")
    print("Running with args:", args)

    print(f"\nBest checkpoint: {best_ckpt}")
    print(f"TensorBoard logs: {tb_dir}")
    print(f"Test predictions CSV: {test_csv}")


if __name__ == "__main__":
    main()
