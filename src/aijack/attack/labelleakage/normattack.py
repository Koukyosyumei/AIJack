import torch
from sklearn.metrics import roc_auc_score

from ..base_attack import BaseAttacker


class NormAttack(BaseAttacker):
    def __init__(self, target_model):
        """Class that implement normattack
        Args:
            target_model: target splotnn model
        """
        super().__init__(target_model)

    def extract_intermidiate_gradient(self, outputs, target_client_index=0):
        pass

    def attack(self, dataloader, criterion, device="cpu", target_client_index=0):
        """Culculate leak_auc on the given SplitNN model
           reference: https://arxiv.org/abs/2102.08504
        Args:
            dataloader (torch dataloader): dataloader for evaluation
            criterion: loss function for training
            device: cpu or GPU
        Returns:
            score: culculated leak auc
        """
        epoch_labels = []
        epoch_g_norm = []
        for i, data in enumerate(dataloader, 0):

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = self.target_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            grad_from_server = self.extract_intermidiate_gradient(outputs)
            g_norm = grad_from_server.pow(2).sum(dim=1).sqrt()
            epoch_labels.append(labels)
            epoch_g_norm.append(g_norm)

        epoch_labels = torch.cat(epoch_labels)
        epoch_g_norm = torch.cat(epoch_g_norm)
        score = roc_auc_score(epoch_labels, epoch_g_norm.view(-1, 1))
        return score


class SplitNNNormAttack(NormAttack):
    def __init__(self, target_model, target_client_index=0):
        super().__init__(target_model)
        self.target_client_index = target_client_index

    def extract_intermidiate_gradient(self, outputs):
        self.target_model.backward(outputs.grad)
        return self.target_model.clients[self.target_client_index].grad_from_next_client
