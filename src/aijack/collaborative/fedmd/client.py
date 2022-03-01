import torch
from torch import nn

from ..core import BaseClient


class FedMDClient(BaseClient):
    def __init__(
        self,
        model,
        public_dataloader,
        batch_size=8,
        user_id=0,
        base_loss_func=nn.CrossEntropyLoss(),
        consensus_loss_func=nn.L1Loss(),
    ):
        super(FedMDClient, self).__init__(model, user_id=user_id)
        self.public_dataloader = public_dataloader
        self.batch_size = batch_size
        self.base_loss_func = base_loss_func
        self.consensus_loss_func = consensus_loss_func
        self.predicted_values_of_server = None

    def upload(self):
        y_pred = []
        for (x, _) in self.public_dataloader:
            y_pred.append(self(x))
        return torch.cat(y_pred)

    def download(self, predicted_values_of_server):
        self.predicted_values_of_server = predicted_values_of_server

    def approach_consensus(self, consensus_optimizer):
        running_loss = 0
        for (x, _), y_consensus in zip(
            self.public_dataloader,
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(self.predicted_values_of_server),
                batch_size=self.public_dataloader.batch_size,
            ),
        ):
            x = self.transform(x)
            consensus_optimizer.zero_grad()
            y_pred = self(x)
            loss = self.consensus_loss_func(y_pred, y_consensus)
            loss.backward()
            consensus_optimizer.step()
            running_loss += loss.item()

        return running_loss
