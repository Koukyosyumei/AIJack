import numpy as np
import torch
from sklearn.metrics import accuracy_score
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


def accuracy_torch_dataloader(model, dataloader, device="cpu", xpos=1, ypos=2):
    in_preds = []
    in_label = []
    with torch.no_grad():
        for data in dataloader:
            inputs = data[xpos]
            labels = data[ypos]
            inputs = inputs.to(device)
            labels = labels.to(device).to(torch.int64)
            outputs = model(inputs)
            in_preds.append(outputs)
            in_label.append(labels)
        in_preds = torch.cat(in_preds)
        in_label = torch.cat(in_label)

    return accuracy_score(
        np.array(torch.argmax(in_preds, axis=1).cpu()), np.array(in_label.cpu())
    )
