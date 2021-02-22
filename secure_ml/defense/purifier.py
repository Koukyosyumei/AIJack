import torch
import torch.nn as nn
import torch.nn.functional as F


class Purifier_Cifar10(nn.Module):
    """ autoencoder for purification on Cifar10
        reference https://arxiv.org/abs/2005.03915
    """

    def __init__(self):
        super(Purifier_Cifar10, self).__init__()
        self.L1 = nn.Linear(10, 7)
        self.bn1 = nn.BatchNorm1d(7)
        self.L2 = nn.Linear(7, 4)
        self.bn2 = nn.BatchNorm1d(4)
        self.L3 = nn.Linear(4, 7)
        self.bn3 = nn.BatchNorm1d(7)
        self.L4 = nn.Linear(7, 10)

    def forward(self, x):
        # 10 -> 7
        x = self.L1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # 7 -> 4
        x = self.L2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # 4 -> 7
        x = self.L3(x)
        x = self.bn3(x)
        x = F.relu(x)
        # 7 -> 10
        x = self.L4(x)

        return x


def PurifierLoss(prediction,
                 pred_purified,
                 lam=0.2,
                 purifier_criterion=nn.MSELoss(),
                 accuracy_criterion=nn.CrossEntropyLoss()):
    """basic loss function for purification
       reference https://arxiv.org/abs/2005.03915

    Args:
        prediction:
        pred_purified:
        lam:
        purifier_criterion:
        accuracy_criterion:

    Return:
        loss_purifier: weighted average of two loss function
    """

    loss_1 = purifier_criterion(pred_purified, prediction)
    loss_2 = accuracy_criterion(pred_purified,
                                torch.argmax(prediction, axis=1))
    loss_purifier = loss_1 + lam * loss_2

    return loss_purifier
