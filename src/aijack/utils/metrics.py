import torch
from torch.nn import functional as F


def total_variance(x):
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


def crossentropyloss_between_logits(y1, y2):
    return -torch.sum(F.log_softmax(y1, dim=1) * F.log_softmax(y2, dim=1), dim=1)
