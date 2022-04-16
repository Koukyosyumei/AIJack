import torch
from sklearn.metrics import roc_auc_score

from ...manager import BaseManager


def attach_normattack_to_splitnn(
    cls, attack_criterion, target_client_index=0, device="cpu"
):
    class NormAttackSplitNNWrapper(cls):
        def __init__(self, *args, **kwargs):
            super(NormAttackSplitNNWrapper, self).__init__(*args, **kwargs)
            self.attack_criterion = attack_criterion
            self.target_client_index = target_client_index
            self.device = device

        def extract_intermidiate_gradient(self, outputs):
            self.backward_gradient(outputs.grad)
            return self.clients[self.target_client_index].grad_from_next_client

        def attack(self, dataloader):
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
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self(inputs)
                loss = self.attack_criterion(outputs, labels)
                loss.backward()

                grad_from_server = self.extract_intermidiate_gradient(outputs)
                g_norm = grad_from_server.pow(2).sum(dim=1).sqrt()
                epoch_labels.append(labels)
                epoch_g_norm.append(g_norm)

            epoch_labels = torch.cat(epoch_labels)
            epoch_g_norm = torch.cat(epoch_g_norm)
            score = roc_auc_score(epoch_labels, epoch_g_norm.view(-1, 1))
            return score

    return NormAttackSplitNNWrapper


class NormAttackManager(BaseManager):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def attach(self, cls):
        return attach_normattack_to_splitnn(cls, *self.args, **self.kwargs)
