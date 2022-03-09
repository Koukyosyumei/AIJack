import torch

from ...utils.metrics import crossentropyloss_between_logits
from ..core import BaseClient


class DSFLClient(BaseClient):
    def __init__(self, model, public_dataloader, device="cpu", user_id=0):
        super().__init__(model, user_id)
        self.public_dataloader = public_dataloader
        self.device = device
        self.global_logit = None

    def upload(self):
        y_pred = []
        for data in self.public_dataloader:
            x = data[1]
            x = x.to(self.device)
            y_pred.append(self(x).detach())
        return torch.cat(y_pred)

    def download(self, global_logit):
        self.global_logit = global_logit

    def approach_consensus(self, consensus_optimizer):
        running_loss = 0
        for global_data, global_logit_data in zip(
            self.public_dataloader,
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(self.global_logit),
                batch_size=self.public_dataloader.batch_size,
            ),
        ):
            x = global_data[1].to(self.device)
            y_global = global_logit_data[0].to(self.device).detach()
            consensus_optimizer.zero_grad()
            y_local = self(x)
            loss_consensus = crossentropyloss_between_logits(y_local, y_global)
            loss_consensus.backward()
            consensus_optimizer.step()
            running_loss += loss_consensus.item()
        return running_loss
