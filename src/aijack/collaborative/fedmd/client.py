import torch
from torch import nn

from ...utils.utils import torch_round_x_decimal, worker_init_fn
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

    def upload(self):
        y_pred = []
        for data in self.public_dataloader:
            x = data[1].to(self.device)
            y_pred.append(self(x).detach())

        result = torch.cat(y_pred)
        if self.round_decimal is None:
            return result
        else:
            return torch_round_x_decimal(result, self.round_decimal)

    def download(self, predicted_values_of_server):
        self.predicted_values_of_server = predicted_values_of_server

    def approach_consensus(self, consensus_optimizer):
        running_loss = 0

        g = torch.Generator()
        g.manual_seed(0)

        for data_x, data_y in zip(
            self.public_dataloader,
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(self.predicted_values_of_server),
                batch_size=self.public_dataloader.batch_size,
                worker_init_fn=worker_init_fn,
                generator=g,
                shuffle=True,
            ),
        ):
            x = data_x[1].to(self.device)
            y_consensus = data_y[0].to(self.device)
            consensus_optimizer.zero_grad()
            y_pred = self(x)
            loss = self.consensus_loss_func(y_pred, y_consensus)
            loss.backward()
            consensus_optimizer.step()
            running_loss += loss.item()

        return running_loss
