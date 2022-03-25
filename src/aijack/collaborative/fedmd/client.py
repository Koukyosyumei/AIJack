import torch
from torch import nn

from ...utils.utils import torch_round_x_decimal
from ..core import BaseClient


class FedMDClient(BaseClient):
    def __init__(
        self,
        model,
        public_dataloader,
        output_dim=1,
        batch_size=8,
        user_id=0,
        base_loss_func=nn.CrossEntropyLoss(),
        consensus_loss_func=nn.L1Loss(),
        round_decimal=None,
        device="cpu",
    ):
        super(FedMDClient, self).__init__(model, user_id=user_id)
        self.public_dataloader = public_dataloader
        self.batch_size = batch_size
        self.base_loss_func = base_loss_func
        self.consensus_loss_func = consensus_loss_func
        self.round_decimal = round_decimal
        self.device = device

        self.predicted_values_of_server = None

        len_public_dataloader = len(self.public_dataloader.dataset)
        self.logit2server = torch.ones((len_public_dataloader, output_dim)).to(
            self.device
        ) * float("inf")

    def upload(self):
        for data in self.public_dataloader:
            idx = data[0]
            x = data[1]
            x = x.to(self.device)
            self.logit2server[idx, :] = self(x).detach()

        if self.round_decimal is None:
            return self.logit2server
        else:
            return torch_round_x_decimal(self.logit2server, self.round_decimal)

    def download(self, predicted_values_of_server):
        self.predicted_values_of_server = predicted_values_of_server

    def approach_consensus(self, consensus_optimizer):
        running_loss = 0

        for data in self.public_dataloader:
            idx = data[0]
            x = data[1].to(self.device)
            y_consensus = self.predicted_values_of_server[idx, :].to(self.device)
            consensus_optimizer.zero_grad()
            y_pred = self(x)
            loss = self.consensus_loss_func(y_pred, y_consensus)
            loss.backward()
            consensus_optimizer.step()
            running_loss += loss.item()

        return running_loss
