import numpy as np
import torch
from sklearn.metrics import accuracy_score
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
        for data in self.public_dataloader:
            x = data[0].to(self.device)
            y_pred.append(self(x).detach())
        return torch.cat(y_pred)

    def download(self, predicted_values_of_server):
        self.predicted_values_of_server = predicted_values_of_server

    def approach_consensus(self, consensus_optimizer):
        running_loss = 0
        for data_x, data_y in zip(
            self.public_dataloader,
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(self.predicted_values_of_server),
                batch_size=self.public_dataloader.batch_size,
            ),
        ):
            x = data_x[0].to(self.device)
            y_consensus = data_y[0].to(self.device)
            consensus_optimizer.zero_grad()
            y_pred = self(x)
            loss = self.consensus_loss_func(y_pred, y_consensus)
            loss.backward()
            consensus_optimizer.step()
            running_loss += loss.item()

        return running_loss

    def score(self, dataloader):
        in_preds = []
        in_label = []
        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self(inputs)
                in_preds.append(outputs)
                in_label.append(labels)
            in_preds = torch.cat(in_preds)
            in_label = torch.cat(in_label)

        return accuracy_score(
            np.array(torch.argmax(in_preds, axis=1).cpu()), np.array(in_label.cpu())
        )
