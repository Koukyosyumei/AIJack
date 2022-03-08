import torch

from ...utils.metrics import crossentropyloss_between_logits
from ..core import BaseServer


class DSFLServer(BaseServer):
    def __init__(
        self,
        clients,
        global_model,
        public_dataloader,
        aggregation="ERA",
        server_id=0,
        device="cpu",
    ):
        super(DSFLServer, self).__init__(clients, global_model, server_id=server_id)
        self.public_dataloader = public_dataloader
        self.aggregation = aggregation
        self.consensus = None
        self.device = device

    def action(self):
        self.update()
        self.distribute()

    def update(self):
        if self.aggregation == "ERA":
            self._entropy_reduction_aggregation()
        elif self.aggregation == "SA":
            self._simple_aggregation()
        else:
            raise NotImplementedError(f"{self.aggregation} is not supported")

    def update_globalmodel(self, global_optimizer):
        running_loss = 0
        for global_data, global_logit_data in enumerate(
            self.public_dataloader,
            torch.utils.dataset.dataloader(
                torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(self.consensus),
                    batch_size=self.public_dataloader.batch_size,
                )
            ),
        ):
            x = global_data[1].to(self.device)
            y_global = global_logit_data[0].to(self.device)
            global_optimizer.zero_grad()
            y_local = self(x)
            loss_consensus = crossentropyloss_between_logits(y_global, y_local)
            loss_consensus.backward()
            global_optimizer.step()
            running_loss += loss_consensus.item()
        return running_loss

    def distribute(self):
        """Distribute the logits of public dataset to each client."""
        for client in self.clients:
            client.download(self.consensus)

    def _entropy_reduction_aggregation(self):
        self._simple_aggregation()
        self.consensus = torch.softmax(self.consensus, dim=1)

    def _simple_aggregation(self):
        self.consensus = self.clients[0].upload() / len(self.clients)
        for client in self.clients[1:]:
            self.consensus += client.upload() / len(self.clients)
