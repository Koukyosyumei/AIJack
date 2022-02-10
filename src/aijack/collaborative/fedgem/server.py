import torch
from torch import nn

from ..core import BaseServer


class FedGEMServer(BaseServer):
    def __init__(
        self,
        clients,
        global_model,
        len_public_dataloader,
        y_dim=1,
        self_evaluation_func=None,
        base_loss_func=nn.CrossEntropyLoss(),
        kldiv_loss_func=nn.KLDivLoss(),
        server_id=0,
        lr=0.1,
        epsilon=0.1,
    ):
        super(FedGEMServer, self).__init__(clients, global_model, server_id=server_id)
        self.len_public_dataloader = len_public_dataloader
        self.lr = lr
        self.epsilon = epsilon
        self.self_evaluation_func = self_evaluation_func
        self.base_loss_func = base_loss_func
        self.kldiv_loss_func = kldiv_loss_func

        self.global_pool_of_logits = torch.ones((len_public_dataloader, y_dim)) * float(
            "inf"
        )
        self.predicted_values = torch.ones((len_public_dataloader, y_dim)) * float(
            "inf"
        )

    def action(self):
        self.distribtue()

    def update(self, idxs, x):
        """Register the predicted logits to self.predicted_values"""
        self.predicted_values[idxs] = self.server_model(x)

    def distribtue(self):
        """Distribute the logits of public dataset to each client."""
        for client in self.clients:
            client.download(self.predicted_values)

    def self_evaluation_on_the_public_dataset(self, idxs, x, y):
        """Execute self evaluation on the public dataset

        Args:
            idxs (torch.Tensor): indexs of x
            x (torch.Tensor): input data
            y (torch.Tensr): labels of x

        Returns:
            the loss
        """
        y_pred = self.server_model(x)
        correct_idx, incorrect_idx = self.self_evaluation_func(y_pred, y)

        loss_s1 = 0
        loss_s2 = 0
        loss_s3 = 0

        # for each sample that the server predicts correctly
        loss_s1 += self.base_loss_func(y_pred[correct_idx], y[correct_idx])
        self.global_pool_of_logits[idxs[correct_idx]] = y_pred[correct_idx]

        # for each sample that the server predicts wrongly
        s_incorrect_not_star_idx = [
            ici
            for ici in incorrect_idx
            if float("int") != self.global_pool_of_logits[idxs[ici]].item()
        ]
        loss_s2 += self.epsilon * self.base_loss_func(
            y_pred[s_incorrect_not_star_idx], y[s_incorrect_not_star_idx]
        ) + (1 - self.epsilon) * self.kldiv_loss_func(
            self.global_pool_of_logits[idxs[s_incorrect_not_star_idx]].log(),
            y_pred[s_incorrect_not_star_idx],
        )

        s_incorrect_star_idx = list(set(incorrect_idx) - set(s_incorrect_not_star_idx))
        loss_s3 += self.epsilon * self.base_loss_func(
            y_pred[s_incorrect_star_idx], y[s_incorrect_star_idx]
        ) + (1 - self.epsilon) * self.kldiv_loss_func(
            self._get_knowledge_from_clients(
                x[s_incorrect_star_idx], y[s_incorrect_star_idx]
            ).log(),
            y_pred[s_incorrect_star_idx],
        )

        loss = loss_s1 + loss_s2 + loss_s3
        return loss

    def _get_knowledge_from_clients(self, x, y):
        knowledge = 0
        for client in self.clients:
            y_pred = client.upload(x)
            correct_idx, _ = self.self_evaluation_func(y_pred, y)
            ep = torch.zeros((y_pred.shape[0]))
            if len(correct_idx) != 0:
                ep[correct_idx] += -1 * torch.sum(
                    y_pred[correct_idx] * torch.log(y_pred[correct_idx]), dim=1
                )
            knowledge += (1 / ep).softmax(dim=1) * y_pred
        return knowledge
