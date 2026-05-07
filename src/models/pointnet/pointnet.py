from __future__ import print_function, annotations
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from dataclasses import dataclass

import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
import matplotlib.pyplot as plt
import os
import csv
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
    
class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x, mask=None):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = masked_max_pool(x, mask, keepdim=True)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x, mask = None):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = masked_max_pool(x, mask, keepdim=True)   # <— use mask
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetFeatGSSeparate(nn.Module):
    def __init__(self, global_feat: bool = True, feature_transform: bool = False, in_channels_total: int = 11):
        super().__init__()
        assert in_channels_total >= 3
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.in_channels_total = in_channels_total

        self.stn = STN3d()                         
        self.conv1 = nn.Conv1d(in_channels_total, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        if feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, xyz_norm_bn3, gs_extra_b8n, mask = None):
        B, N, _ = xyz_norm_bn3.shape
        xyz = xyz_norm_bn3.transpose(2, 1).contiguous()

        trans = self.stn(xyz, mask)                                  
        xyz_t = torch.bmm(xyz.transpose(2,1), trans).transpose(2,1) 

        if gs_extra_b8n is not None:
            x = torch.cat([xyz_t, gs_extra_b8n], dim=1)
        else:
            x = xyz_t

        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x, mask)
            x = torch.bmm(x.transpose(2,1), trans_feat).transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x                                            
        x = F.relu(self.bn2(self.conv2(x)))                      
        x = self.bn3(self.conv3(x))                              
        x = masked_max_pool(x, mask, keepdim=False)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x_exp = x.view(B, 1024, 1).repeat(1, 1, N) 
            return torch.cat([x_exp, pointfeat], 1), trans, trans_feat 

class PointNetClsGSSeparate(nn.Module):
    def __init__(self, num_classes: int, feature_transform: bool = True, in_channels_total: int = 11, dropout_p: float = 0.3):
        super().__init__()
        self.feat = PointNetFeatGSSeparate(global_feat=True, feature_transform=feature_transform, in_channels_total=in_channels_total)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, xyz_norm_bn3, gs_extra_b8n, mask):
        x, trans, trans_feat = self.feat(xyz_norm_bn3, gs_extra_b8n, mask)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x, trans, trans_feat

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

class PointNetGSSystem(LoggingModel):
    def __init__(
        self,
        num_classes: int,
        in_channels_total: int = 11,
        feature_transform: bool = True,
        ft_reg_weight: float = 0.001,
        dropout_p: float = 0.3,
        optim_cfg: OptimConfig = OptimConfig(),
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["optim_cfg"])
        self.model = PointNetClsGSSeparate(
            num_classes=num_classes,
            feature_transform=feature_transform,
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

        extras = gauss[:, 0:, :]
        logits, _, trans_feat = self.model(xyz, extras, mask)  
        loss = F.cross_entropy(logits, y)
        if self.hparams.feature_transform and trans_feat is not None:
            loss = loss + self.hparams.ft_reg_weight * feature_transform_regularizer(trans_feat)

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

    