import torch


def total_variation(x):
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


def label_matching(pred, label):
    onehot_label = torch.eye(pred.shape[-1])[label]
    return torch.sqrt(torch.sum((pred - onehot_label) ** 2))
