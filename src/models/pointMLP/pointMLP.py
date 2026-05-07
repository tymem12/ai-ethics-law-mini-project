from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
from dataclasses import dataclass
import os, csv
import matplotlib.pyplot as plt
import numpy as np
from models.baseModel import LoggingModel

@dataclass
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

def masked_max_pool(x, mask, keepdim: bool = False):
    if mask is None:
        return x.max(dim=2, keepdim=keepdim).values
    x = x.masked_fill(~mask[:, None, :], float('-inf'))
    return x.max(dim=2, keepdim=keepdim).values

class ResidualMLP1d(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, norm='bn'):
        super().__init__()
        self.proj = nn.Identity() if c_in == c_out else nn.Conv1d(c_in, c_out, 1)
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_hidden, 1),
            nn.BatchNorm1d(c_hidden) if norm == 'bn' else nn.Identity(),
            nn.GELU(),
            nn.Conv1d(c_hidden, c_out, 1),
            nn.BatchNorm1d(c_out) if norm == 'bn' else nn.Identity(),
        )

    def forward(self, x):  
        return F.gelu(self.net(x) + self.proj(x))

class ResidualMLP2d(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, norm='bn'):
        super().__init__()
        self.proj = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1)
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, 1),
            nn.BatchNorm2d(c_hidden) if norm == 'bn' else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(c_hidden, c_out, 1),
            nn.BatchNorm2d(c_out) if norm == 'bn' else nn.Identity(),
        )

    def forward(self, x):  # x: (B, C_in, M, K)
        return F.gelu(self.net(x) + self.proj(x))

class GeometricAffine(nn.Module):
    def __init__(self, c_feat, hidden=64):
        super().__init__()
        self.cond = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * c_feat)  
        )

    def forward(self, feats_g, offsets):
        B, C, M, K = feats_g.shape
        with torch.no_grad():
            mu = offsets.mean(dim=3)                     
            rad = (offsets.pow(2).sum(dim=1) + 1e-8).sqrt() 
            mean_r = rad.mean(dim=2)                     
            var_r = rad.var(dim=2, unbiased=False)       
            cen_norm = mu.pow(2).sum(dim=1).sqrt()       

            geo = torch.stack([cen_norm, mean_r, var_r], dim=-1)  # (B, M, 3)

        theta = self.cond(geo)                           
        gamma, beta = theta.split(C, dim=-1)             
        gamma = gamma.permute(0, 2, 1).unsqueeze(-1)     
        beta = beta.permute(0, 2, 1).unsqueeze(-1)      
        return gamma * feats_g + beta

def farthest_point_sample(xyz, m, mask=None):
    B, N, _ = xyz.shape
    device = xyz.device
    if mask is None:
        mask = torch.ones(B, N, dtype=torch.bool, device=device)

    inds = torch.zeros(B, m, dtype=torch.long, device=device)
    first = torch.argmax(mask.to(torch.int64), dim=1)
    dists = torch.full((B, N), float('inf'), device=device)
    chosen = first

    for i in range(m):
        inds[:, i] = chosen
        cur = xyz[torch.arange(B, device=device), chosen]
        dist = torch.sum((xyz - cur[:, None, :]) ** 2, dim=-1)  # (B, N)
        dists = torch.minimum(dists, torch.where(mask, dist, torch.full_like(dist, float('inf'))))
        chosen = torch.argmax(torch.where(mask, dists, torch.full_like(dists, -1.0)), dim=1)
    return inds

def knn_group(xyz_query, xyz_all, feats_all, k, mask_all=None):
    B, N, _ = xyz_all.shape
    M = xyz_query.shape[1]
    device = xyz_all.device

    d = torch.cdist(xyz_query, xyz_all, p=2)  
    if mask_all is not None:
        big = torch.full_like(d, 1e9)
        d = torch.where(mask_all[:, None, :], d, big)

    idx = d.topk(k=k, dim=2, largest=False).indices

    batch_idx = torch.arange(B, device=device)[:, None, None]
    grouped_xyz = xyz_all[batch_idx, idx, :]            
    centroid_xyz = xyz_query[:, :, None, :]             
    offsets = (grouped_xyz - centroid_xyz).permute(0, 3, 1, 2).contiguous()

    feats_all_T = feats_all.permute(0, 2, 1).contiguous()
    grouped_feats = feats_all_T[batch_idx, idx, :].permute(0, 3, 1, 2).contiguous()

    return grouped_feats, offsets
class PointMLPStage(nn.Module):
    def __init__(self, c_in, c_mid, c_out, k=24):
        super().__init__()
        self.k = k
        self.ga = GeometricAffine(c_in, hidden=min(64, c_in))
        self.pre  = ResidualMLP2d(c_in,  c_mid, c_out)
        self.post = ResidualMLP1d(c_out, c_mid, c_out)

    def forward(self, xyz_in, feats_in, m, mask_in=None):
        B, N, _ = xyz_in.shape
        device = xyz_in.device
        idx = farthest_point_sample(xyz_in, m, mask=mask_in)  
        batch_idx = torch.arange(B, device=device)[:, None]
        xyz_q = xyz_in[batch_idx, idx, :]                     
        mask_out = torch.ones(B, m, dtype=torch.bool, device=device)

        g_feats, offsets = knn_group(xyz_q, xyz_in, feats_in, self.k, mask_all=mask_in) 

        g_feats = self.ga(g_feats, offsets)                   
        g_feats = self.pre(g_feats)                           
        feats_pooled = g_feats.max(dim=3).values              
        feats_out = self.post(feats_pooled)                   
        return xyz_q, feats_out, mask_out

class PointMLPBackbone(nn.Module):
    def __init__(self,
                 in_channels_total: int = 11,
                 widths=(64, 128, 256, 512),
                 mids=(64, 128, 256, 256),
                 ks=(24, 24, 24, 24),
                 ratios=(0.5, 0.25, 0.125, 0.0625)):
        super().__init__()
        assert in_channels_total >= 3
        self.ratios = ratios
        self.embed = nn.Sequential(
            nn.Conv1d(in_channels_total, widths[0], 1),
            nn.BatchNorm1d(widths[0]),
            nn.GELU(),
        )

        stages = []
        for i in range(len(widths)-1):
            stages.append(PointMLPStage(widths[i], mids[i], widths[i+1], k=ks[i]))
        self.stages = nn.ModuleList(stages)

        self.tail = nn.Sequential(
            nn.Conv1d(widths[-1], widths[-1], 1),
            nn.BatchNorm1d(widths[-1]),
            nn.GELU(),
        )

    def forward(self, xyz_norm_bn3, extras_b8n=None, mask=None):
        B, N, _ = xyz_norm_bn3.shape
        xyz_t = xyz_norm_bn3.transpose(1, 2).contiguous()  # (B, 3, N)
        if extras_b8n is not None:
            feats = torch.cat([xyz_t, extras_b8n], dim=1)
        else:
            feats = xyz_t

        feats = self.embed(feats)  # (B, C0, N)
        xyz = xyz_norm_bn3
        cur_mask = mask

        for i, stage in enumerate(self.stages):
            m = max(1, int(N * self.ratios[i]))
            xyz, feats, cur_mask = stage(xyz, feats, m, mask_in=cur_mask)
            N = m  

        feats = self.tail(feats)   # (B, C_last, M_last)
        global_feat = feats.max(dim=2).values
        return global_feat   # (B, C_last)

class PointMLPCls(nn.Module):
    def __init__(self, num_classes: int, in_channels_total: int = 11, dropout_p: float = 0.3):
        super().__init__()
        self.backbone = PointMLPBackbone(in_channels_total=in_channels_total)
        c_last = 512
        self.head = nn.Sequential(
            nn.Linear(c_last, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes),
        )

    def forward(self, xyz_norm_bn3, extras_b8n, mask):
        feat = self.backbone(xyz_norm_bn3, extras_b8n, mask)
        logits = self.head(feat)
        return logits, None, None


class PointMLPSystem(LoggingModel):
    def __init__(
        self,
        num_classes: int,
        in_channels_total: int = 11,
        dropout_p: float = 0.3,
        optim_cfg: OptimConfig = OptimConfig(),
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["optim_cfg"])
        self.model = PointMLPCls(
            num_classes=num_classes,
            in_channels_total=in_channels_total,
            dropout_p=dropout_p
        )
        self.optim_cfg = optim_cfg

        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

        self.test_preds = []
        self.test_targets = []
        self.test_paths = []


    def _shared_step(self, batch, stage: str):
        gauss = batch["gauss"]              
        xyz = batch["xyz_normalized"]       
        y = batch["label"]                  
        mask = batch['mask']                

        extras = gauss[:, 3:, :] if gauss.shape[1] >= 11 else gauss[:, 0:, :]

        logits, _, _ = self.model(xyz, extras, mask)     
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        if stage == "train":
            acc = self.train_acc(preds, y)
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        elif stage == "val":
            acc = self.val_acc(preds, y)
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        else:  # test
            self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.test_acc.update(preds, y)
            self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, "train")
        return loss

    def on_sanity_check_start(self):
        print("→ PointMLPSystem device:", self.device)

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch, "test")
        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(y.detach().cpu())
        return loss
