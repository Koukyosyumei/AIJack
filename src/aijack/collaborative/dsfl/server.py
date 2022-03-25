import torch

from ...utils.metrics import crossentropyloss_between_logits
from ...utils.utils import worker_init_fn
from ..core import BaseServer


class DSFLServer(BaseServer):
    def __init__(
        self,
        clients,
        global_model,
        public_dataloader,
        aggregation="ERA",
        distillation_loss_name="crossentropy",
        era_temperature=0.1,
        server_id=0,
        device="cpu",
    ):
        super(DSFLServer, self).__init__(clients, global_model, server_id=server_id)
        self.public_dataloader = public_dataloader
        self.aggregation = aggregation
        self.era_temperature = era_temperature
        self.consensus = None
        self.device = device

        self._set_distillation_loss(distillation_loss_name)

    def _set_distillation_loss(self, name):
        if name == "crossentropy":
            self.distillation_loss = crossentropyloss_between_logits
        elif name == "L2":
            self.distillation_loss = torch.nn.MSELoss()
        elif name == "L1":
            self.distillation_loss = torch.nn.L1Loss()
        else:
            raise NotImplementedError(f"{name} is not supported")

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
        for global_data, global_logit_data in zip(
            self.public_dataloader,
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(self.consensus),
                batch_size=self.public_dataloader.batch_size,
                worker_init_fn=worker_init_fn,
                shuffle=False,
            ),
        ):
            x = global_data[1].to(self.device)
            y_global = global_logit_data[0].to(self.device)
            y_global_test = torch.mean(
                torch.stack([client(x) for client in self.clients]), dim=0
            )
            print(y_global[:, :3])
            print(y_global_test[:, :3])
            assert torch.all(y_global == y_global_test)
            global_optimizer.zero_grad()
            y_pred = self(x)
            loss_consensus = self.distillation_loss(y_pred, y_global)
            loss_consensus.backward()
            global_optimizer.step()
            running_loss += loss_consensus.item()
        running_loss /= len(self.public_dataloader)
        return running_loss

    def distribute(self):
        """Distribute the logits of public dataset to each client."""
        for client in self.clients:
            client.download(self.consensus)

    def _entropy_reduction_aggregation(self):
        self._simple_aggregation()
        self.consensus = torch.softmax(self.consensus / self.era_temperature, dim=1)

    def _simple_aggregation(self):
        self.consensus = self.clients[0].upload() / len(self.clients)
        for client in self.clients[1:]:
            self.consensus += client.upload() / len(self.clients)
