import torch
from torch.nn import functional as F


def total_variance(x):
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


def crossentropyloss_between_logits(y_pred_logit, y_true_logit):
    return torch.mean(
        -torch.sum(
            F.log_softmax(y_pred_logit, dim=1) * F.softmax(y_true_logit, dim=1), dim=1
        )
    )
