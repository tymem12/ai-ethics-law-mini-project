from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torchmetrics.classification import MulticlassAccuracy
import matplotlib.pyplot as plt
from models.baseModel import LoggingModel

def masked_max_pool(x, mask, keepdim: bool = False):
    if mask is None:
        return x.max(dim=2, keepdim=keepdim).values
    x = x.masked_fill(~mask[:, None, :], float('-inf'))
    return x.max(dim=2, keepdim=keepdim).values
@dataclass
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

def fps_indices(xyz_bn3: torch.Tensor, npoint: int | None, mask_bn: torch.Tensor | None):

    B, N, _ = xyz_bn3.shape
    device = xyz_bn3.device
    if npoint is None:
        npoint = N
    npoint = min(npoint, N)

    sel = []
    for b in range(B):
        valid = mask_bn[b] if mask_bn is not None else torch.ones(N, dtype=torch.bool, device=device)
        idx_valid = torch.nonzero(valid, as_tuple=False).squeeze(1)
        if idx_valid.numel() == 0:
            sel_idx = torch.zeros(npoint, dtype=torch.long, device=device)
        else:
            pts = xyz_bn3[b:b+1, idx_valid]                    
            m = min(npoint, pts.shape[1])
            picked = _torch_fps(pts, m)[0]
            sel_idx = idx_valid[picked]
            if m < npoint:  
                pad = sel_idx[-1].repeat(npoint - m)
                sel_idx = torch.cat([sel_idx, pad], dim=0)
        sel.append(sel_idx.unsqueeze(0))
    return torch.cat(sel, dim=0) 


def _torch_fps(x: torch.Tensor, n_sample: int):
    b, n, _ = x.shape
    assert n_sample <= n
    device = x.device
    idx = torch.zeros((b, n_sample), dtype=torch.long, device=device)
    idx[:, 0] = torch.randint(0, n, (b,), device=device)
    dist = torch.full((b, n), float("inf"), device=device)
    last = x[torch.arange(b), idx[:, 0]] 
    for i in range(1, n_sample):
        d = torch.sum((x - last[:, None, :]) ** 2, dim=-1)
        dist = torch.minimum(dist, d)
        idx[:, i] = torch.max(dist, dim=1).indices
        last = x[torch.arange(b), idx[:, i]]
    return idx

def query_ball_point(radius: float, nsample: int,
                     new_xyz_bm3: torch.Tensor, xyz_bn3: torch.Tensor,
                     mask_bn: torch.Tensor | None):
    B, M, _ = new_xyz_bm3.shape
    N = xyz_bn3.shape[1]

    diff = new_xyz_bm3[:, :, None, :] - xyz_bn3[:, None, :, :] 
    d2 = (diff * diff).sum(-1)                                 

    if mask_bn is not None:
        d2 = d2.masked_fill(~mask_bn[:, None, :], float('inf'))

    within = d2 <= (radius * radius)                           
    d2_mask = d2.masked_fill(~within, float('inf'))
    k = min(nsample, N)
    idx = torch.topk(d2_mask, k=k, dim=-1, largest=False).indices 

    first_valid = within.float().argmax(dim=-1) 
    if k < nsample:
        pad = first_valid.unsqueeze(-1).expand(B, M, nsample - k)
        idx = torch.cat([idx, pad], dim=-1)     

    gather_d2 = d2.gather(-1, idx)
    bad = torch.isinf(gather_d2)
    if bad.any():
        idx = torch.where(bad, first_valid.unsqueeze(-1).expand_as(idx), idx)

    none_valid = ~within.any(dim=-1)
    if none_valid.any():
        idx[none_valid] = 0
    return idx

class SharedMLP1x1(nn.Module):
    def __init__(self, channels: list[int]):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for ci, co in zip(channels[:-1], channels[1:]):
            self.convs.append(nn.Conv2d(ci, co, kernel_size=1, bias=False))
            self.bns.append(nn.BatchNorm2d(co))
    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)), inplace=True)
        return x


def pairwise_distance(a, b, a_mask=None, b_mask=None):
    diff = a[:, :, None, :] - b[:, None, :, :]
    dist2 = (diff * diff).sum(-1)

    if b_mask is not None:
        dist2 = dist2.masked_fill(~b_mask[:, None, :], float('inf'))
    if a_mask is not None:
        dist2 = dist2.masked_fill(~a_mask[:, :, None], float('inf'))
    return dist2

def gather_index_2d(x, idx):
    B, N = x.shape[:2]
    idx_flat = idx.reshape(B, -1)  
    batch_idx = torch.arange(B, device=x.device)[:, None]
    gathered = x[batch_idx, idx_flat]  
    return gathered.view(B, idx.shape[1], idx.shape[2], -1)

def uniform_centroid_indices(mask, npoint):
    B, N = mask.shape
    device = mask.device
    idx = torch.linspace(0, N - 1, steps=npoint, device=device).round().long().clamp(0, N-1)
    idx = idx.unsqueeze(0).expand(B, -1) 
    valid = mask.gather(1, idx)
    if valid.all():
        return idx
    fixed = idx.clone()
    for b in range(B):
        m = mask[b]
        valid_positions = torch.nonzero(m, as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            fixed[b] = 0
        else:
            pos = idx[b].float()
            vp = valid_positions.float()
            nearest = (pos[:, None] - vp[None, :]).abs().argmin(dim=1)
            fixed[b] = valid_positions[nearest]
    return fixed

class SetAbstractionMSG(nn.Module):
    def __init__(self, npoint: int,
                 radius_list: list[float],
                 nsample_list: list[int],
                 in_channels: int,
                 mlp_list: list[list[int]],
                 use_xyz: bool = True):
        super().__init__()
        assert len(radius_list) == len(nsample_list) == len(mlp_list)
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.use_xyz = use_xyz

        in0 = in_channels + (3 if use_xyz else 0)
        self.mlps = nn.ModuleList([SharedMLP1x1([in0] + mlp) for mlp in mlp_list])

    def forward(self, xyz_bn3, feat_bcn, mask_bn=None):
        B, N, _ = xyz_bn3.shape
        device = xyz_bn3.device

        cent_idx = fps_indices(xyz_bn3, self.npoint, mask_bn)
        new_xyz = xyz_bn3.gather(1, cent_idx[..., None].expand(-1, -1, 3))
        new_mask = mask_bn.gather(1, cent_idx) if mask_bn is not None else torch.ones(B, self.npoint, dtype=torch.bool, device=device)

        branch_feats = []
        for radius, nsample, mlp in zip(self.radius_list, self.nsample_list, self.mlps):
            idx = query_ball_point(radius, nsample, new_xyz, xyz_bn3, mask_bn) 

            group_xyz = gather_index_2d(xyz_bn3, idx) 
            group_xyz = group_xyz - new_xyz[:, :, None, :]

            if feat_bcn is None:
                local = group_xyz
            else:
                feat_bnc = feat_bcn.transpose(1, 2)
                group_feat = gather_index_2d(feat_bnc, idx)
                local = torch.cat([group_xyz, group_feat], dim=-1) if self.use_xyz else group_feat

            local = local.permute(0, 3, 1, 2).contiguous()
            local = mlp(local)
            local = local.max(dim=3).values
            branch_feats.append(local)

        new_feat = torch.cat(branch_feats, dim=1) 
        return new_xyz, new_feat, new_mask


class SetAbstractionGlobal(nn.Module):
    def __init__(self, in_channels: int, mlp_channels: list[int], use_xyz: bool = True):
        super().__init__()
        in0 = in_channels + (3 if use_xyz else 0)
        self.mlp = SharedMLP1x1([in0] + mlp_channels)
        self.use_xyz = use_xyz

    def forward(self, xyz_bn3, feat_bcn, mask_bn=None):
        if feat_bcn is None:
            local = xyz_bn3.transpose(1, 2)[:, :, None, :]        # (B,3,1,N)
        else:
            if self.use_xyz:
                local = torch.cat([xyz_bn3.transpose(1, 2), feat_bcn], dim=1)[:, :, None, :]  # (B,3+C,1,N)
            else:
                local = feat_bcn[:, :, None, :]

        if mask_bn is not None:
            neg_inf = torch.tensor(float('-inf'), device=local.device, dtype=local.dtype)
            local = local.masked_fill(~mask_bn[:, None, None, :], neg_inf)

        x = self.mlp(local)     
        x = x.max(dim=3).values 
        x = x.squeeze(2)        
        return xyz_bn3.mean(dim=1, keepdim=True), x, torch.ones(x.shape[0], 1, dtype=torch.bool, device=xyz_bn3.device)

class PointNet2Feat_MSG(nn.Module):
    def __init__(self, in_channels_total: int = 11):
        super().__init__()
        assert in_channels_total >= 3
        in_feat = in_channels_total - 3 

        self.sa1 = SetAbstractionMSG(
            npoint=512,
            radius_list=[0.1, 0.2, 0.4],
            nsample_list=[16, 32, 128],
            in_channels=in_feat,
            mlp_list=[[32, 32, 64],
                      [64, 64, 128],
                      [64, 96, 128]],
            use_xyz=True
        )
        self.sa2 = SetAbstractionMSG(
            npoint=128,
            radius_list=[0.2, 0.4, 0.8],
            nsample_list=[32, 64, 128],
            in_channels=320,
            mlp_list=[[64, 64, 128],
                      [128, 128, 256],
                      [128, 128, 256]],
            use_xyz=True
        )
        self.sa3 = SetAbstractionGlobal(
            in_channels=640,
            mlp_channels=[256, 512, 1024],
            use_xyz=True
        )

    def forward(self, xyz_bn3, extras_b8n, mask_bn=None):
        B, N, _ = xyz_bn3.shape
        if extras_b8n is None:
            extras_b8n = torch.zeros(B, 8, N, device=xyz_bn3.device, dtype=xyz_bn3.dtype)

        xyz, feat1, mask1 = self.sa1(xyz_bn3, extras_b8n, mask_bn)
        xyz, feat2, mask2 = self.sa2(xyz,        feat1,       mask1)
        _,    g,     _   = self.sa3(xyz,         feat2,       mask2)
        return g


class PointNet2Cls_MSG(nn.Module):
    def __init__(self, num_classes: int, in_channels_total: int = 11, dropout_p: float = 0.4):
        super().__init__()
        self.feat = PointNet2Feat_MSG(in_channels_total=in_channels_total)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(p=dropout_p)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz_bn3, extras_b8n, mask_bn):
        x = self.feat(xyz_bn3, extras_b8n, mask_bn)          # (B,1024)
        x = F.relu(self.bn1(self.fc1(x)), inplace=True)
        x = self.drop(x)
        x = F.relu(self.bn2(self.fc2(x)), inplace=True)
        x = self.drop(x)
        return self.fc3(x)


class PointNet2GSSystem(LoggingModel):
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
        self.model = PointNet2Cls_MSG(
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

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch_gpu = {
            "xyz_normalized": batch["xyz_normalized"].to(device, non_blocking=True),
            "gauss": batch["gauss"].to(device, non_blocking=True),   
            "mask": batch["mask"].to(device, non_blocking=True),
            "label": batch["label"].to(device, non_blocking=True),
        }
        return batch_gpu

    def _shared_step(self, batch, stage: str):
        gauss = batch["gauss"]              
        xyz = batch["xyz_normalized"]       
        y = batch["label"]                  
        mask = batch['mask']                

        extras = gauss[:, 0:, :]

        logits = self.model(xyz, extras, mask)
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
        print("→ LM device:", self.device)

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch, "test")
        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(y.detach().cpu())
        return loss
