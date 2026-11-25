# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


def binary_brier_loss(logits, labels):
    labels = labels.float().view(-1)
    prob = F.softmax(logits, dim=-1)[:, 1]
    return torch.mean((prob - labels) ** 2)
