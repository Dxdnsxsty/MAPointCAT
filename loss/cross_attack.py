# -*- coding: UTF-8 -*-
"""
@Project MAPointCAT
@File    cross_attack.py.py
@IDE     PyCharm
@Author  小帅天一(tianyi-yan@qq.com)
@Date    2026/4/22 16:12
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttackConsistencyLoss(nn.Module):
    def __init__(self, metric='cosine'):
        super().__init__()
        self.metric = metric

    def pair_loss(self, x, y):
        if self.metric == 'cosine':
            x = F.normalize(x, dim=1)
            y = F.normalize(y, dim=1)
            return (1.0 - (x * y).sum(dim=1)).mean()
        else:
            return ((x - y) ** 2).sum(dim=1).mean()

    def forward(self, feats):
        # feats: list of [B, C]
        if len(feats) < 2:
            return torch.tensor(0.0, device=feats[0].device if len(feats) > 0 else 'cuda')
        loss = 0.0
        cnt = 0
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                loss += self.pair_loss(feats[i], feats[j])
                cnt += 1
        return loss / cnt
