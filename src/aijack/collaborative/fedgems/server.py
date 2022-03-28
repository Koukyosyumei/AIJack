import torch
from torch import nn

from ..core import BaseServer


class FedGEMSServer(BaseServer):
    def __init__(
        self,
        clients,
        global_model,
        len_public_dataloader,
        output_dim=1,
        self_evaluation_func=None,
        base_loss_func=nn.CrossEntropyLoss(),
        kldiv_loss_func=nn.KLDivLoss(),
        server_id=0,
        lr=0.1,
        epsilon=0.75,
        device="cpu",
    ):
        super(FedGEMSServer, self).__init__(clients, global_model, server_id=server_id)
        self.len_public_dataloader = len_public_dataloader
        self.lr = lr
        self.epsilon = epsilon
        self.self_evaluation_func = self_evaluation_func
        self.base_loss_func = base_loss_func
        self.kldiv_loss_func = kldiv_loss_func
        self.output_dim = output_dim
        self.device = device

        self.global_pool_of_logits = torch.ones((len_public_dataloader, output_dim)).to(
            self.device
        ) * float("inf")
        self.predicted_values = torch.ones((len_public_dataloader, output_dim)).to(
            self.device
        ) * float("inf")

    def action(self):
        self.distribtue()

    def update(self, idxs, x):
        """Register the predicted logits to self.predicted_values"""
        self.predicted_values[idxs] = self.server_model(x).detach().to(self.device)

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
        if len(correct_idx) != 0:
            loss_s1 += self.base_loss_func(
                y_pred[correct_idx], y[correct_idx].to(torch.int64)
            )
            self.global_pool_of_logits[idxs[correct_idx]] = y_pred[correct_idx].detach()

        # for each sample that the server predicts wrongly
        s_incorrect_not_star_idx = [
            iid.item()
            for iid in incorrect_idx
            if self.global_pool_of_logits[idxs[iid]][0].item() != float("inf")
        ]
        if len(s_incorrect_not_star_idx) != 0:
            loss_s2 += self.epsilon * self.base_loss_func(
                y_pred[s_incorrect_not_star_idx],
                y[s_incorrect_not_star_idx].to(torch.int64),
            ) + (1 - self.epsilon) * self.kldiv_loss_func(
                self.global_pool_of_logits[idxs[s_incorrect_not_star_idx]]
                .softmax(dim=-1)
                .log(),
                y_pred[s_incorrect_not_star_idx].softmax(dim=-1),
            )

        s_incorrect_star_idx = list(
            set(incorrect_idx.cpu().tolist()) - set(s_incorrect_not_star_idx)
        )
        if len(s_incorrect_star_idx) != 0:
            loss_s3 += self.epsilon * self.base_loss_func(
                y_pred[s_incorrect_star_idx], y[s_incorrect_star_idx]
            ) + (1 - self.epsilon) * self.kldiv_loss_func(
                self._get_knowledge_from_clients(
                    x[s_incorrect_star_idx], y[s_incorrect_star_idx]
                )
                .softmax(dim=-1)
                .log(),
                y_pred[s_incorrect_star_idx].softmax(dim=-1),
            )
        loss = loss_s1 + loss_s2 + loss_s3
        return loss

    def _get_knowledge_from_clients(self, x, y):
        client_weight = torch.zeros(self.num_clients, y.shape[0]).to(self.device)
        client_knowledge = torch.zeros(
            self.num_clients, y.shape[0], self.output_dim
        ).to(self.device)
        for cid, client in enumerate(self.clients):
            y_pred = client.upload(x).to(self.device)
            client_knowledge[cid] = y_pred
            correct_idx, _ = self.self_evaluation_func(y_pred, y)
            if len(correct_idx) != 0:
                ep = torch.zeros((y_pred.shape[0])).to(self.device)
                ep[correct_idx] += -1 * torch.sum(
                    y_pred[correct_idx].softmax(dim=-1)
                    * torch.log(y_pred[correct_idx].softmax(dim=-1)),
                    dim=1,
                )
                client_weight[cid, correct_idx] = 1 / ep[correct_idx]

        client_weight = (
            client_weight.softmax(dim=0)
            .reshape(self.num_clients, y.shape[0], 1)
            .expand(self.num_clients, y.shape[0], self.output_dim)
        )

        ensembled_knowledge = torch.sum(
            client_weight * client_knowledge,
            dim=0,
        )

        return ensembled_knowledge.detach()
